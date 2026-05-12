"""Step 4 of Phase 1 — interference analysis with three clean deltas.

The original Stage 5 design only had a single ``delta_PAI = post − pre``
where ``pre`` came from Gemini 3.1 (Experiment 1) and ``post`` from
Qwen3-32B + LoRA. That delta confounds two effects:

1. The base-model switch (Gemini 3.1 → Qwen3-32B).
2. The LoRA fine-tune itself.

Once a Qwen-base baseline run exists (``baseline_qwen_train.json`` and
``baseline_qwen_heldout.json``) and a held-out adapter run exists
(``post_intervention_heldout_r{rank}.json``), this module decomposes the
single delta into three meaningful ones, per case and per DP, per rank:

* **Δ_memorization** = Qwen+LoRA(train) − Qwen-base(train)
  — Pure LoRA effect on cases the adapter was trained on. Largely
  reflects memorization of the corrective targets.
* **Δ_generalization** = Qwen+LoRA(heldout) − Qwen-base(heldout)
  — LoRA effect on cases the adapter has *never seen*. This is the
  generalization measurement.
* **Δ_vs_Gemini** = Qwen+LoRA(heldout) − Gemini(heldout from Exp1)
  — Direct comparison against the Exp1 baseline, but on a clean
  out-of-train set (so it is a fair "did Qwen+LoRA beat Gemini?").

The legacy DP × rank heatmap and observational interference table are
preserved so existing reports keep rendering, but the canonical output
now lives under ``deltas_clean``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.stage5_scoring.matrices import DECISION_POINTS, DP_LABELS

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _load_post_results(stage5_dir: Path, *, target: str) -> Dict[int, Dict[str, Any]]:
    """Load the adapter-mode result files for a given target.

    ``target="train"``   → post_intervention_r{rank}.json (legacy filenames)
    ``target="heldout"`` → post_intervention_heldout_r{rank}.json
    """
    out: Dict[int, Dict[str, Any]] = {}
    pattern = (
        "post_intervention_r*.json" if target == "train"
        else "post_intervention_heldout_r*.json"
    )
    for path in sorted(stage5_dir.glob(pattern)):
        # filename ends with _r{N}.json
        rank = int(path.stem.split("_r")[-1])
        payload = read_json(path)
        # On train, accept any file whose payload doesn't claim target=heldout
        # (legacy files predate the target field). On heldout, require explicit.
        payload_target = payload.get("target")
        if target == "train" and payload_target == "heldout":
            continue
        if target == "heldout" and payload_target != "heldout":
            continue
        # And skip baseline files if any sneak in (they have is_baseline=True).
        if payload.get("is_baseline"):
            continue
        out[rank] = payload
    return out


def _load_baseline(stage5_dir: Path, target: str) -> Optional[Dict[str, Any]]:
    path = stage5_dir / f"baseline_qwen_{target}.json"
    if not path.exists():
        return None
    return read_json(path)


# ---------------------------------------------------------------------------
# Legacy: DP × rank heatmap on train (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _heatmap_dp_by_rank(post: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for rank, payload in post.items():
        ok = [r for r in payload["results"] if r["status"] == "ok"]
        if not ok:
            continue
        row: Dict[str, float] = {}
        for dp in DECISION_POINTS:
            row[DP_LABELS[dp]] = round(
                mean(r["delta_dp_means"][dp] for r in ok), 4
            )
        row["overall"] = round(mean(r["delta_PAI"] for r in ok), 4)
        table[str(rank)] = row
    return table


def _interference_table(post: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rank, payload in post.items():
        ok = [r for r in payload["results"] if r["status"] == "ok"]
        if not ok:
            continue

        per_rank: Dict[str, Dict[str, float]] = {}
        for dp_i in DECISION_POINTS:
            sorted_ok = sorted(
                ok, key=lambda r: r["pre_dp_means"].get(dp_i, 0.0)
            )
            idx = int(len(sorted_ok) * 2 / 3)
            cutoff = (
                sorted_ok[idx]["pre_dp_means"].get(dp_i, 0.0) if sorted_ok else 0.0
            )
            high_cases = [
                r
                for r in sorted_ok
                if r["pre_dp_means"].get(dp_i, 0.0) >= cutoff
            ]
            if not high_cases:
                continue

            row: Dict[str, float] = {}
            for dp_j in DECISION_POINTS:
                deltas = [r["delta_dp_means"][dp_j] for r in high_cases]
                row[DP_LABELS[dp_j]] = round(mean(deltas), 4)
            per_rank[DP_LABELS[dp_i]] = row

        out[str(rank)] = per_rank
    return out


def _effective_rank(heatmap: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    if not heatmap:
        return {}

    sorted_ranks = sorted(int(k) for k in heatmap.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for dp in DECISION_POINTS:
        label = DP_LABELS[dp]
        first_positive = None
        deltas = []
        for r in sorted_ranks:
            d = heatmap[str(r)].get(label)
            deltas.append({"rank": r, "delta": d})
            if d is not None and d > 0 and first_positive is None:
                first_positive = r
        out[label] = {
            "first_positive_rank": first_positive,
            "deltas_by_rank": deltas,
        }
    return out


# ---------------------------------------------------------------------------
# New: three clean deltas — memorization, generalization, vs Gemini
# ---------------------------------------------------------------------------

def _results_by_id(payload: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not payload:
        return {}
    return {
        r["prompt_id"]: r
        for r in payload["results"]
        if r["status"] == "ok"
    }


def _per_dp_diff(
    minuend: Dict[str, Any],
    subtrahend_dp: Dict[str, float],
) -> Dict[str, float]:
    """Compute ``minuend.post_dp_means[dp] − subtrahend_dp[dp]`` per DP."""
    out = {}
    for dp in DECISION_POINTS:
        a = minuend["post_dp_means"].get(dp)
        b = subtrahend_dp.get(dp, 0.0)
        if a is None:
            out[dp] = None
        else:
            out[dp] = round(a - b, 4)
    return out


def _build_clean_deltas(
    *,
    post_train: Dict[int, Dict[str, Any]],
    post_heldout: Dict[int, Dict[str, Any]],
    baseline_train: Optional[Dict[str, Any]],
    baseline_heldout: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """For each rank present, compute the three clean deltas.

    Returns a structure shaped like::

        {
          "memorization":   {rank: {"per_case": [...], "summary": {...}}},
          "generalization": {rank: {...}},
          "vs_gemini":      {rank: {...}},
          "notes": {missing pieces},
        }
    """
    base_train = _results_by_id(baseline_train)
    base_held = _results_by_id(baseline_heldout)

    output: Dict[str, Any] = {
        "memorization": {},
        "generalization": {},
        "vs_gemini": {},
        "notes": {
            "baseline_train_available": bool(base_train),
            "baseline_heldout_available": bool(base_held),
            "heldout_ranks_available": sorted(post_heldout.keys()),
            "train_ranks_available": sorted(post_train.keys()),
        },
    }

    # Δ_memorization: needs post_train + base_train.
    if base_train:
        for rank, payload in post_train.items():
            per_case: List[Dict[str, Any]] = []
            dp_deltas: Dict[str, List[float]] = {dp: [] for dp in DECISION_POINTS}
            pai_deltas: List[float] = []
            for r in payload["results"]:
                if r["status"] != "ok":
                    continue
                pid = r["prompt_id"]
                base = base_train.get(pid)
                if not base:
                    continue
                base_pai = base["post_PAI"]
                base_dp = base["post_dp_means"]
                d_pai = round(r["post_PAI"] - base_pai, 4)
                d_dp = _per_dp_diff(r, base_dp)
                per_case.append({
                    "prompt_id": pid,
                    "qwen_lora_PAI": r["post_PAI"],
                    "qwen_base_PAI": base_pai,
                    "delta_memorization_PAI": d_pai,
                    "delta_memorization_dp_means": d_dp,
                })
                pai_deltas.append(d_pai)
                for dp in DECISION_POINTS:
                    if d_dp[dp] is not None:
                        dp_deltas[dp].append(d_dp[dp])
            summary = {
                "n": len(per_case),
                "delta_PAI_mean": round(mean(pai_deltas), 4) if pai_deltas else None,
                "delta_dp_means": {
                    dp: (round(mean(vs), 4) if vs else None)
                    for dp, vs in dp_deltas.items()
                },
            }
            output["memorization"][str(rank)] = {
                "per_case": per_case,
                "summary": summary,
            }

    # Δ_generalization: needs post_heldout + base_held.
    if base_held:
        for rank, payload in post_heldout.items():
            per_case = []
            dp_deltas = {dp: [] for dp in DECISION_POINTS}
            pai_deltas = []
            for r in payload["results"]:
                if r["status"] != "ok":
                    continue
                pid = r["prompt_id"]
                base = base_held.get(pid)
                if not base:
                    continue
                base_pai = base["post_PAI"]
                base_dp = base["post_dp_means"]
                d_pai = round(r["post_PAI"] - base_pai, 4)
                d_dp = _per_dp_diff(r, base_dp)
                per_case.append({
                    "prompt_id": pid,
                    "qwen_lora_PAI": r["post_PAI"],
                    "qwen_base_PAI": base_pai,
                    "delta_generalization_PAI": d_pai,
                    "delta_generalization_dp_means": d_dp,
                })
                pai_deltas.append(d_pai)
                for dp in DECISION_POINTS:
                    if d_dp[dp] is not None:
                        dp_deltas[dp].append(d_dp[dp])
            summary = {
                "n": len(per_case),
                "delta_PAI_mean": round(mean(pai_deltas), 4) if pai_deltas else None,
                "delta_dp_means": {
                    dp: (round(mean(vs), 4) if vs else None)
                    for dp, vs in dp_deltas.items()
                },
            }
            output["generalization"][str(rank)] = {
                "per_case": per_case,
                "summary": summary,
            }

    # Δ_vs_Gemini on heldout — already computed inside post_heldout as
    # `delta_PAI` (post_PAI − pre_PAI where pre is Gemini from Exp1) so we
    # just summarise it here.
    for rank, payload in post_heldout.items():
        per_case = []
        dp_deltas = {dp: [] for dp in DECISION_POINTS}
        pai_deltas = []
        for r in payload["results"]:
            if r["status"] != "ok":
                continue
            per_case.append({
                "prompt_id": r["prompt_id"],
                "qwen_lora_PAI": r["post_PAI"],
                "gemini_PAI": r["pre_PAI"],
                "delta_vs_gemini_PAI": r["delta_PAI"],
                "delta_vs_gemini_dp_means": r["delta_dp_means"],
            })
            pai_deltas.append(r["delta_PAI"])
            for dp in DECISION_POINTS:
                dp_deltas[dp].append(r["delta_dp_means"][dp])
        summary = {
            "n": len(per_case),
            "delta_PAI_mean": round(mean(pai_deltas), 4) if pai_deltas else None,
            "delta_dp_means": {
                dp: (round(mean(vs), 4) if vs else None)
                for dp, vs in dp_deltas.items()
            },
        }
        output["vs_gemini"][str(rank)] = {
            "per_case": per_case,
            "summary": summary,
        }

    return output


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_interference_analysis(cfg: PipelineConfig) -> Dict[str, Any]:
    """Aggregate every available post/baseline run into a single report.

    The legacy heatmap + observational interference table are still
    produced for the train-target adapter runs (so older readers of
    ``interference_analysis.json`` see the same shape). The three clean
    deltas live under ``deltas_clean``.
    """
    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")

    post_train = _load_post_results(stage5_dir, target="train")
    post_heldout = _load_post_results(stage5_dir, target="heldout")
    baseline_train = _load_baseline(stage5_dir, "train")
    baseline_heldout = _load_baseline(stage5_dir, "heldout")

    if not post_train and not post_heldout:
        raise FileNotFoundError(
            "No post_intervention_r*.json files found in data/stage5/. "
            "Run `run-stage5-query` first."
        )

    heatmap = _heatmap_dp_by_rank(post_train) if post_train else {}
    interference = _interference_table(post_train) if post_train else {}
    effective = _effective_rank(heatmap)

    deltas_clean = _build_clean_deltas(
        post_train=post_train,
        post_heldout=post_heldout,
        baseline_train=baseline_train,
        baseline_heldout=baseline_heldout,
    )

    payload: Dict[str, Any] = {
        "ranks_present": {
            "train": sorted(post_train.keys()),
            "heldout": sorted(post_heldout.keys()),
        },
        "baselines_present": {
            "train": baseline_train is not None,
            "heldout": baseline_heldout is not None,
        },
        # Legacy (train-target, vs-Gemini delta)
        "dp_by_rank_heatmap": heatmap,
        "effective_rank_by_dp": effective,
        "interference_top_tercile": interference,
        # New (three clean deltas)
        "deltas_clean": deltas_clean,
    }
    write_json(payload, stage5_dir / "interference_analysis.json")

    logger.info(
        "Stage 5 interference: train ranks=%s, heldout ranks=%s, baselines train=%s heldout=%s",
        payload["ranks_present"]["train"],
        payload["ranks_present"]["heldout"],
        payload["baselines_present"]["train"],
        payload["baselines_present"]["heldout"],
    )

    if not deltas_clean["memorization"] and not deltas_clean["generalization"]:
        logger.warning(
            "Stage 5 interference: no clean deltas computed. "
            "Run `run-stage5-baseline --target both` and "
            "`run-stage5-query --target heldout` to populate them."
        )
    return payload
