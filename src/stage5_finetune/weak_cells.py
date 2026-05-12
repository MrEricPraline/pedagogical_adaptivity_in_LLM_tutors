"""Identify weak (decision_point × condition) cells from Experiment 1 results.

Per the proposal:
    "Using the Experiment 1 model results, identify the decision point ×
     condition cells with the lowest PAI scores."

For every DP, the PAI scoring is conditioned on a tuple of learner
variables (e.g., ``content_level`` is conditioned on
``(bloom_band, knowledge_state)``). The Cartesian product of those
condition values defines the cells for that DP:

    content_level       — bloom_band × knowledge_state                  → 3×3   = 9
    student_task        — learning_stage × bloom_band                   → 4×3   = 12
    tutor_role          — learning_context × knowledge_state            → 3×3   = 9
    student_engagement  — learning_context × bloom_band × learning_stage → 3×3×4 = 36
    disciplinary_method — subject_family × learning_stage               → 4×4   = 16
                                                                        Total  = 82

For each cell we take the mean of the per-case DP sub-score
(``case["dp_means"][dp]``) across all cases whose conditions match.
"Weak cells" are the bottom-K cells by that mean.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest
from src.stage5_scoring.matrices import DECISION_POINTS

logger = logging.getLogger("pipeline")


def _load_scored(stage5_dir: Path) -> List[Dict[str, Any]]:
    path = stage5_dir / "scored_dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            "scored_dataset.json not found. Run `run-stage5-score` first."
        )
    return read_json(path)


def _cell_key(dp: str, condition_values: Tuple[str, ...]) -> str:
    """Stable string id for a (DP, condition_tuple) pair, e.g.
    ``content_level|bloom_band=RU|knowledge_state=novice``.
    """
    _, keys = DECISION_POINTS[dp]
    parts = [f"{k}={v}" for k, v in zip(keys, condition_values)]
    return dp + "|" + "|".join(parts)


def compute_cells(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For every DP × condition cell, compute the mean PAI sub-score across
    all cases falling in that cell.

    Returns one row per cell with::

        {
            "cell_id":         "content_level|bloom_band=RU|knowledge_state=novice",
            "decision_point":  "content_level",
            "conditions":      {"bloom_band": "RU", "knowledge_state": "novice"},
            "mean_pai":        float,
            "n_cases":         int,
            "case_ids":        ["P0001", ...],
        }
    """
    rows: List[Dict[str, Any]] = []

    for dp, (_matrices, condition_keys) in DECISION_POINTS.items():
        buckets: Dict[Tuple[str, ...], List[Tuple[str, float]]] = defaultdict(list)
        for case in scored:
            cond = tuple(case[k] for k in condition_keys)
            buckets[cond].append((case["prompt_id"], case["dp_means"][dp]))

        for cond, items in buckets.items():
            ids = [pid for pid, _ in items]
            scores = [s for _, s in items]
            rows.append({
                "cell_id": _cell_key(dp, cond),
                "decision_point": dp,
                "conditions": dict(zip(condition_keys, cond)),
                "mean_pai": round(mean(scores), 4),
                "n_cases": len(ids),
                "case_ids": ids,
            })

    rows.sort(key=lambda r: r["mean_pai"])
    return rows


def run_identify_weak_cells(
    cfg: PipelineConfig,
    *,
    k: int = 10,
) -> Dict[str, Any]:
    """Identify the bottom-K cells by mean PAI and write them to disk.

    Output: ``data/stage5/weak_cells.json``::

        {
            "k":              <int>,
            "n_cells_total":  <int>,
            "all_cells":      [...],    # every cell, sorted ascending by mean_pai
            "weak_cells":     [...],    # the bottom-K of all_cells
        }
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    stage5_dir.mkdir(parents=True, exist_ok=True)

    scored = _load_scored(stage5_dir)
    rows = compute_cells(scored)
    weak = rows[:k]

    payload = {
        "k": k,
        "n_cells_total": len(rows),
        "weak_cells": weak,
        "all_cells": rows,
    }
    out_path = stage5_dir / "weak_cells.json"
    write_json(payload, out_path)

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    manifest = build_manifest(
        stage="stage5_weak_cells",
        params={
            "k": k,
            "input_scored": "data/stage5/scored_dataset.json",
            "output": "data/stage5/weak_cells.json",
        },
        row_count=len(rows),
        extra={
            "weakest_cell_id": weak[0]["cell_id"] if weak else None,
            "weakest_mean_pai": weak[0]["mean_pai"] if weak else None,
            "weak_dp_breakdown": _dp_breakdown(weak),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage5_dir / "manifest_weak_cells.json")

    logger.info(
        "Stage 5 weak cells: %d cells total, bottom-%d range [%.4f, %.4f] → %s",
        len(rows), k,
        weak[0]["mean_pai"] if weak else 0.0,
        weak[-1]["mean_pai"] if weak else 0.0,
        out_path,
    )
    for row in weak:
        logger.info(
            "  %s  mean_pai=%+.4f  n=%d",
            row["cell_id"], row["mean_pai"], row["n_cases"],
        )

    return payload


def _dp_breakdown(cells: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cells:
        out[c["decision_point"]] = out.get(c["decision_point"], 0) + 1
    return out
