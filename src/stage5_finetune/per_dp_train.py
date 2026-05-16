"""Per-DP isolated LoRAs — five adapters, each correcting only one DP.

Per the proposal:
    "Test cross-dimensional interference (correcting one DP and measuring
     effect on others)."

Causal interference requires per-DP isolated LoRAs: one adapter trained
to correct only DP_i, evaluated on its effect on every DP_j (including
j = i). The 5×5 matrix of mean deltas is the causal interference
heatmap.

This module:

1. Partitions the stratified corrective dataset
   (``corrective_training_data_per_dp.json``) by ``target_dp`` — each
   example is included only in the partition matching its weak cell's DP.
2. For each DP, trains one LoRA adapter on its partition using
   :func:`src.stage5_finetune.tinker_train.train_one_rank` (one rank,
   chosen by the user — typically the effective rank discovered in the
   unified sweep).
3. Writes each adapter metadata to ``data/stage5/adapters/per_dp/<dp>.json``.
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
from src.stage5_finetune.tinker_train import (
    DEFAULT_BASE_MODEL,
    train_one_rank,
)
from src.stage5_scoring.matrices import DECISION_POINTS

logger = logging.getLogger("pipeline")


def _load_per_dp_corrective(stage5_dir: Path) -> List[Dict[str, Any]]:
    path = stage5_dir / "corrective_training_data_per_dp.json"
    if not path.exists():
        raise FileNotFoundError(
            "corrective_training_data_per_dp.json not found. Run "
            "`run-stage5-corrective-stratified --target-dp-only` first."
        )
    return read_json(path)


def _partition_by_dp(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket examples by their target_dps. An example may serve multiple
    weak cells (different DPs), in which case it lands in every relevant
    bucket — but the assistant target was rewritten to correct only the
    single DP of the weak cell that drove its inclusion, so the same
    case in two buckets is two distinct training examples with two
    different targets.

    Implementation: read ``example["target_dps"]`` and replicate the
    example into each DP bucket. The retargeting step in
    :func:`corrective_data._retarget_to_single_dp` was applied during
    stratified construction *for the first cell that served the case*
    only, so reading ``target_dps[0]`` per-bucket would only match the
    first served cell. To keep this simple and faithful, the per-DP
    construction below uses *only* the example as written (single DP
    target equal to ``target_dps[0]``).
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {dp: [] for dp in DECISION_POINTS}
    for ex in examples:
        # target_dps is preserved by corrective_data.run_build_corrective_data_stratified.
        primary = (ex.get("target_dps") or [None])[0]
        if primary in buckets:
            buckets[primary].append(ex)
    return buckets


def run_per_dp_finetune(
    cfg: PipelineConfig,
    *,
    rank: int,
    epochs: int = 3,
    lr: float = 1e-4,
    base_model: str = DEFAULT_BASE_MODEL,
    dps: List[str] | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Train one LoRA adapter per DP in ``dps`` (defaults to all 5).

    Resume: if ``data/stage5/adapters/per_dp/{dp}.json`` already exists
    and records a finished run at the same rank/base_model, that DP is
    **skipped** and the existing adapter metadata is reused. Pass
    ``force=True`` to re-train ignoring existing files.
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    per_dp_dir = stage5_dir / "adapters" / "per_dp"
    per_dp_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_per_dp_corrective(stage5_dir)
    buckets = _partition_by_dp(examples)

    target_dps = dps or list(DECISION_POINTS.keys())
    results: Dict[str, Dict[str, Any]] = {}

    for dp in target_dps:
        bucket = buckets.get(dp, [])
        if not bucket:
            logger.warning(
                "Per-DP fine-tune: no training examples for %s, skipping",
                dp,
            )
            continue

        out_path = per_dp_dir / f"{dp}.json"
        if not force and out_path.exists():
            try:
                existing = read_json(out_path)
                if (
                    existing.get("rank") == rank
                    and existing.get("base_model") == base_model
                    and existing.get("adapter_uri")
                ):
                    logger.info(
                        "Per-DP fine-tune: dp=%s already trained at rank=%d in %s — "
                        "skipping (use --force to redo)",
                        dp, rank, out_path.name,
                    )
                    results[dp] = existing
                    continue
            except Exception:  # noqa: BLE001
                pass  # fall through and retrain

        logger.info(
            "Per-DP fine-tune: dp=%s rank=%d n=%d",
            dp, rank, len(bucket),
        )
        metadata = train_one_rank(
            examples=bucket,
            rank=rank,
            base_model=base_model,
            epochs=epochs,
            lr=lr,
            output_dir=None,
        )
        metadata["target_dp"] = dp
        write_json(metadata, out_path)
        results[dp] = metadata

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    manifest = build_manifest(
        stage="stage5_per_dp_finetune",
        params={
            "rank": rank,
            "epochs": epochs,
            "learning_rate": lr,
            "base_model": base_model,
            "input_corrective": "data/stage5/corrective_training_data_per_dp.json",
            "output_dir": "data/stage5/adapters/per_dp",
        },
        row_count=len(results),
        extra={
            "adapters": {
                dp: {
                    "adapter_uri": meta["adapter_uri"],
                    "final_loss": meta["history"][-1]["mean_loss"],
                    "duration_seconds": meta["duration_seconds"],
                    "n_examples": meta["n_examples"],
                }
                for dp, meta in results.items()
            },
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage5_dir / "manifest_per_dp_finetune.json")
    return {"results": results, "manifest": manifest}
