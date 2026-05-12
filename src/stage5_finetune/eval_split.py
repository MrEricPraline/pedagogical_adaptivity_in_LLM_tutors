"""Build a held-out evaluation set that is disjoint from the corrective train set.

The original Stage 5 design trains and evaluates on the same 30 lowest-PAI
cases, so post-intervention deltas reflect a mixture of memorization and
generalization. To separate the two, we add a held-out split: the next-K
cases by ascending ``prompt_PAI``, mutually exclusive with whichever cases
are already in ``corrective_training_data.json``.

The held-out set is what the corrective LoRA has *never seen*, so deltas
measured on it isolate the generalization effect of the fine-tune.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest

logger = logging.getLogger("pipeline")


def _load_scored(stage5_dir: Path) -> List[Dict[str, Any]]:
    json_path = stage5_dir / "scored_dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(
            "scored_dataset.json not found. Run `run-stage5-score` first."
        )
    return read_json(json_path)


def _load_train_ids(stage5_dir: Path) -> List[str]:
    corrective_path = stage5_dir / "corrective_training_data.json"
    if not corrective_path.exists():
        raise FileNotFoundError(
            "corrective_training_data.json not found. Run "
            "`run-stage5-corrective` first."
        )
    examples = read_json(corrective_path)
    return [ex["prompt_id"] for ex in examples]


def run_build_heldout(
    cfg: PipelineConfig,
    *,
    k: int = 30,
) -> Dict[str, Any]:
    """Pick the next-K lowest-PAI cases not already in the train set."""
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    stage5_dir.mkdir(parents=True, exist_ok=True)

    scored = _load_scored(stage5_dir)
    train_ids = set(_load_train_ids(stage5_dir))

    heldout: List[Dict[str, Any]] = []
    for case in scored:  # sorted ascending by prompt_PAI
        pid = case["prompt_id"]
        if pid in train_ids:
            continue
        heldout.append(case)
        if len(heldout) >= k:
            break

    if len(heldout) < k:
        logger.warning(
            "Stage 5 heldout: requested k=%d but only %d disjoint cases available",
            k, len(heldout),
        )

    heldout_ids = [c["prompt_id"] for c in heldout]
    assert set(heldout_ids).isdisjoint(train_ids), (
        "Held-out set overlaps with train set — bug in selection logic."
    )

    out_path = stage5_dir / "eval_heldout_cases.json"
    write_json(heldout, out_path)

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    pai_range = (
        {"min": heldout[0]["prompt_PAI"], "max": heldout[-1]["prompt_PAI"]}
        if heldout else {"min": None, "max": None}
    )
    manifest = build_manifest(
        stage="stage5_heldout",
        params={
            "k": k,
            "input_scored": "data/stage5/scored_dataset.json",
            "input_train": "data/stage5/corrective_training_data.json",
            "output": "data/stage5/eval_heldout_cases.json",
        },
        row_count=len(heldout),
        extra={
            "requested_k": k,
            "produced": len(heldout),
            "train_size": len(train_ids),
            "heldout_pai_range": pai_range,
            "first_train_pai": next(
                (c["prompt_PAI"] for c in scored if c["prompt_id"] in train_ids), None
            ),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage5_dir / "manifest_heldout.json")

    logger.info(
        "Stage 5 heldout: built %d cases disjoint from %d train ids (PAI [%s, %s]) → %s",
        len(heldout), len(train_ids),
        pai_range["min"], pai_range["max"], out_path,
    )

    return {"heldout": heldout, "output": out_path}
