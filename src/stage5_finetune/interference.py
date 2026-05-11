"""Step 4 of Phase 1 — cross-dimensional interference analysis.

The corrective training set targets *all five* DPs simultaneously, so the
interference test we can run here is observational rather than causal:

For each DP, we look at the cases whose pre-intervention sub-score on
that DP was already high (top tercile). If the post-intervention sub-score
on that DP drops in those cases, then training a unified corrective LoRA
hurt a dimension that was already correct — i.e. cross-dimensional
interference.

We also produce a |DP × rank| heatmap of mean per-DP delta so you can see
which dimensions improve at which ranks (the "effective rank" diagnostic).

The truly causal interference test (correct only DP_i, measure spillover
on DP_j ≠ i) requires per-DP isolated LoRAs and is left as future work —
spec'd in the comments at the bottom of this file so it's easy to wire in
later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.stage5_scoring.matrices import DECISION_POINTS, DP_LABELS

logger = logging.getLogger("pipeline")


def _load_post_results(stage5_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for path in sorted(stage5_dir.glob("post_intervention_r*.json")):
        rank = int(path.stem.split("_r")[-1])
        out[rank] = read_json(path)
    return out


def _heatmap_dp_by_rank(post: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean per-DP delta, indexed by [rank][dp_label]."""
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
    """For each rank, compute the post-intervention delta on DP_j conditioned
    on cases that started in the top tercile on DP_i.

    Output: ``{rank: {dp_i: {dp_j: mean_delta_on_dp_j}}}``.
    """
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
    """For each DP, find the lowest rank at which the per-DP delta first
    becomes positive (a coarse proxy for effective LoRA rank)."""
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


def run_interference_analysis(cfg: PipelineConfig) -> Dict[str, Any]:
    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    post = _load_post_results(stage5_dir)
    if not post:
        raise FileNotFoundError(
            "No post_intervention_r*.json files found in data/stage5/. "
            "Run `python -m src.pipeline.cli run-stage5-query` first."
        )

    heatmap = _heatmap_dp_by_rank(post)
    interference = _interference_table(post)
    effective = _effective_rank(heatmap)

    payload = {
        "ranks_present": sorted(int(k) for k in heatmap.keys()),
        "dp_by_rank_heatmap": heatmap,
        "effective_rank_by_dp": effective,
        "interference_top_tercile": interference,
    }
    write_json(payload, stage5_dir / "interference_analysis.json")

    logger.info("Stage 5 interference: ranks=%s", payload["ranks_present"])
    for rank, row in heatmap.items():
        logger.info("  rank=%s heatmap=%s", rank, row)
    return payload


# ---------------------------------------------------------------------------
# Future work — per-DP isolated LoRAs
#
# The current corrective dataset targets all 5 DPs at once because we have
# 30 cases total. Once you have ≥50 cases per (DP × cell) you can:
#
#   1. Group corrective examples by the DP they intend to correct.
#   2. Train one LoRA per DP (5 LoRAs) at the rank chosen above.
#   3. For each (i, j) pair, run query_one_adapter on the LoRA trained
#      to correct DP_i and aggregate the delta on DP_j.
#   4. The (i, j) matrix is the causal interference heatmap.
#
# That experiment is conceptually identical to what's already wired here;
# it just needs the per-DP corrective dataset and an outer loop over the
# 5 LoRAs. Slot it under src/stage5_finetune/per_dp_train.py when ready.
# ---------------------------------------------------------------------------
