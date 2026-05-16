"""Causal cross-dimensional interference matrix from per-DP isolated LoRAs.

Per the proposal:
    "Test cross-dimensional interference (correcting one DP and measuring
     effect on others)."

Given the five per-DP LoRA adapters produced by
:func:`src.stage5_finetune.per_dp_train.run_per_dp_finetune`, this module:

1. Re-queries each adapter on the *evaluation* case set (the held-out
   set if available, otherwise the corrective train set).
2. Scores the model output with the PAI matrices.
3. Computes the per-case, per-DP delta vs the Qwen3-32B baseline (if
   that baseline exists) or vs the Gemini-3.1 reference.
4. Aggregates the deltas into a 5×5 matrix ``M[i, j]`` =
   *mean change in DP_j when only DP_i was the training target*.

The diagonal is the within-DP correction effect. The off-diagonal is
the cross-DP interference (positive = collateral benefit, negative =
collateral damage).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest
from src.stage5_finetune.tinker_query import (
    Target,
    _is_complete_run,
    _load_cases_for_target,
    query_one,
)
from src.stage5_finetune.tinker_train import DEFAULT_BASE_MODEL
from src.stage5_scoring.matrices import DECISION_POINTS, DP_LABELS

logger = logging.getLogger("pipeline")


def _load_per_dp_adapters(stage5_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return ``{dp: adapter_metadata}`` for every per-DP adapter on disk."""
    per_dp_dir = stage5_dir / "adapters" / "per_dp"
    if not per_dp_dir.exists():
        raise FileNotFoundError(
            "data/stage5/adapters/per_dp/ not found. Run "
            "`run-stage5-per-dp-finetune` first."
        )
    out: Dict[str, Dict[str, Any]] = {}
    for dp in DECISION_POINTS:
        path = per_dp_dir / f"{dp}.json"
        if path.exists():
            out[dp] = read_json(path)
    if not out:
        raise FileNotFoundError(
            f"No per-DP adapters found in {per_dp_dir}."
        )
    return out


def _per_dp_results_filename(dp: str, target: Target) -> str:
    return f"per_dp_query_{target}_{dp}.json"


def _baseline_dp_means(
    stage5_dir: Path,
    target: Target,
    cases: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Return per-case Qwen baseline ``{prompt_id: {dp: score}}`` if the
    baseline file exists, else fall back to per-case Gemini-from-Exp1
    ``dp_means`` (the only baseline we'd otherwise have)."""
    baseline_path = stage5_dir / f"baseline_qwen_{target}.json"
    if baseline_path.exists():
        payload = read_json(baseline_path)
        return {
            r["prompt_id"]: r["post_dp_means"]
            for r in payload["results"]
            if r["status"] == "ok"
        }
    logger.warning(
        "Causal interference: no Qwen baseline for target=%s — "
        "falling back to Gemini reference. Run `run-stage5-baseline` "
        "to get a clean baseline.",
        target,
    )
    return {c["prompt_id"]: c["dp_means"] for c in cases}


def run_causal_interference(
    cfg: PipelineConfig,
    *,
    target: Target = "heldout",
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    skip_query: bool = False,
    corrective_file: str = "corrective_training_data.json",
    force: bool = False,
) -> Dict[str, Any]:
    """For each per-DP adapter, re-query the chosen case set, score, and
    build the 5×5 interference matrix.

    Resume:
    - Per-DP query files (``per_dp_query_{target}_{dp}.json``) that already
      exist and cover the full case set with matching target are reused.
    - ``skip_query=True`` forces reuse of any existing file regardless of
      completeness/freshness (legacy escape hatch).
    - ``force=True`` ignores existing files and re-queries everything.
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")

    adapters = _load_per_dp_adapters(stage5_dir)
    cases = _load_cases_for_target(stage5_dir, target, corrective_file=corrective_file)
    expected_ids = [c["prompt_id"] for c in cases]

    from src.stage5_finetune.corrective_data import _load_narratives
    narratives = _load_narratives(cfg.stage4_dir)

    per_dp_payloads: Dict[str, Dict[str, Any]] = {}

    for trained_dp, meta in adapters.items():
        out_path = stage5_dir / _per_dp_results_filename(trained_dp, target)

        if not force:
            if skip_query and out_path.exists():
                per_dp_payloads[trained_dp] = read_json(out_path)
                logger.info("Causal interference: reusing %s (skip_query)", out_path)
                continue
            if _is_complete_run(out_path, target=target, expected_case_ids=expected_ids):
                per_dp_payloads[trained_dp] = read_json(out_path)
                logger.info(
                    "Causal interference: trained_dp=%s already complete in %s — skipping",
                    trained_dp, out_path.name,
                )
                continue

        payload = query_one(
            cfg,
            cases=cases,
            narratives=narratives,
            target=target,
            rank=meta.get("rank"),
            adapter_uri=meta["adapter_uri"],
            base_model=base_model,
            max_tokens=max_tokens,
            temperature=temperature,
            output_dir=None,
        )
        payload["trained_dp"] = trained_dp
        write_json(payload, out_path)
        per_dp_payloads[trained_dp] = payload
        logger.info(
            "Causal interference: trained_dp=%s queried %d cases on target=%s",
            trained_dp, payload["summary"]["ok"], target,
        )

    baseline_dp = _baseline_dp_means(stage5_dir, target, cases)
    baseline_is_gemini = not (stage5_dir / f"baseline_qwen_{target}.json").exists()

    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for trained_dp, payload in per_dp_payloads.items():
        row: Dict[str, List[float]] = {dp: [] for dp in DECISION_POINTS}
        for r in payload["results"]:
            if r["status"] != "ok":
                continue
            base = baseline_dp.get(r["prompt_id"])
            if not base:
                continue
            for measured_dp in DECISION_POINTS:
                model_score = r["post_dp_means"].get(measured_dp)
                base_score = base.get(measured_dp)
                if model_score is None or base_score is None:
                    continue
                row[measured_dp].append(model_score - base_score)
        matrix[trained_dp] = {
            measured_dp: (round(mean(vs), 4) if vs else None)
            for measured_dp, vs in row.items()
        }

    matrix_labeled = {
        DP_LABELS[trained_dp]: {
            DP_LABELS[measured_dp]: matrix[trained_dp][measured_dp]
            for measured_dp in DECISION_POINTS
        }
        for trained_dp in matrix
    }

    diagonal = {
        DP_LABELS[dp]: matrix[dp][dp]
        for dp in matrix
        if dp in matrix and matrix[dp].get(dp) is not None
    }
    off_diagonal: List[Dict[str, Any]] = []
    for trained_dp in matrix:
        for measured_dp in DECISION_POINTS:
            if trained_dp == measured_dp:
                continue
            v = matrix[trained_dp].get(measured_dp)
            if v is None:
                continue
            off_diagonal.append({
                "trained_dp": DP_LABELS[trained_dp],
                "measured_dp": DP_LABELS[measured_dp],
                "delta": v,
            })

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    payload = {
        "target": target,
        "baseline_source": "qwen_base" if not baseline_is_gemini else "gemini",
        "n_cases": len(cases),
        "matrix": matrix_labeled,
        "diagonal_within_dp_effect": diagonal,
        "off_diagonal_top": sorted(
            off_diagonal, key=lambda r: abs(r["delta"]), reverse=True
        )[:10],
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
    }
    out_path = stage5_dir / "causal_interference_matrix.json"
    write_json(payload, out_path)

    manifest = build_manifest(
        stage="stage5_causal_interference",
        params={
            "target": target,
            "base_model": base_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_cases": len(cases),
            "n_adapters": len(adapters),
        },
        row_count=len(adapters) * len(DECISION_POINTS),
        extra={
            "baseline_source": payload["baseline_source"],
            "matrix": matrix_labeled,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage5_dir / "manifest_causal_interference.json")

    logger.info(
        "Causal interference matrix written to %s (target=%s, baseline=%s)",
        out_path, target, payload["baseline_source"],
    )
    for trained_dp, row in matrix_labeled.items():
        logger.info("  trained=%s row=%s", trained_dp, row)

    return payload
