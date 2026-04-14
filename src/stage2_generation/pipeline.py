"""Stage 2 orchestrator: read factorial CSV -> generate narratives -> validate -> save."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.common.io_utils import append_jsonl, read_csv, read_jsonl, write_json
from src.common.schemas import FACTORIAL_COLUMNS, NarrativeResult
from src.common.text_utils import count_words
from src.pipeline.config import PipelineConfig
from src.pipeline.manifests import build_manifest, save_manifest
from src.stage2_generation.checkpoint import Checkpoint
from src.stage2_generation.prompt_builder import build_messages
from src.stage2_generation.provider_xai import XAIProvider
from src.stage2_generation.validator import is_clean, should_retry, validate_narrative

logger = logging.getLogger("pipeline")


def _build_result(
    row: Dict[str, str],
    narrative: str,
    checks: Dict[str, bool],
    cfg: PipelineConfig,
    status: str,
    error: str = "",
) -> Dict:
    wc = count_words(narrative) if narrative else 0
    return NarrativeResult(
        prompt_id=row["prompt_id"],
        bloom=row["bloom"],
        bloom_band=row["bloom_band"],
        subject=row["subject"],
        subject_family=row["subject_family"],
        knowledge_state=row["knowledge_state"],
        learning_stage=row["learning_stage"],
        learning_context=row["learning_context"],
        narrative=narrative,
        word_count=wc,
        validation=checks,
        validation_clean=is_clean(checks) if checks else False,
        generator_provider="xai",
        generator_model=cfg.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
        generation_status=status,
        error_message=error,
    ).to_dict()


def run_stage2(cfg: PipelineConfig) -> None:
    input_path = cfg.input_path or cfg.stage1_dir / "factorial_sample.csv"
    output_dir = cfg.output_dir or cfg.stage2_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "case_narratives.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"

    # --- Load input ---
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Stage 1 output not found at {input_path}. Run Stage 1 first."
        )

    all_rows: List[Dict[str, str]] = read_csv(Path(input_path))
    logger.info("Stage 2: loaded %d rows from %s", len(all_rows), input_path)

    # --- Slice ---
    start = cfg.start or 0
    end = cfg.end or len(all_rows)
    rows = all_rows[start:end]
    logger.info("Stage 2: processing rows %d–%d (%d cases)", start, end - 1, len(rows))

    # --- Resume ---
    ckpt = Checkpoint(checkpoint_path)
    if cfg.resume:
        already = len(ckpt.completed)
        logger.info("Stage 2: resume enabled — %d cases already completed", already)

    # --- Provider ---
    provider = XAIProvider(cfg)

    # --- Generation loop ---
    processed = 0
    succeeded = 0
    failed = 0

    for i, row in enumerate(rows):
        pid = row["prompt_id"]

        if cfg.resume and ckpt.is_done(pid):
            continue

        messages = build_messages(
            subject=row["subject"],
            bloom=row["bloom"],
            knowledge_state=row["knowledge_state"],
            learning_stage=row["learning_stage"],
            learning_context=row["learning_context"],
        )

        narrative = ""
        checks: Dict[str, bool] = {}
        status = "ok"
        error = ""

        try:
            narrative = provider.generate(messages)
            checks = validate_narrative(
                narrative, cfg.word_count_min, cfg.word_count_max
            )

            # One retry for recoverable validation failures
            if not is_clean(checks) and should_retry(checks):
                logger.debug("Retrying %s (validation: %s)", pid, checks)
                narrative = provider.generate(messages)
                checks = validate_narrative(
                    narrative, cfg.word_count_min, cfg.word_count_max
                )

            if not is_clean(checks):
                status = "validation_failed"
                failed += 1
            else:
                succeeded += 1

        except Exception as exc:
            status = "error"
            error = str(exc)
            failed += 1
            logger.error("Error generating %s: %s", pid, exc)

        result = _build_result(row, narrative, checks, cfg, status, error)
        append_jsonl(result, jsonl_path)
        ckpt.mark_done(pid)
        processed += 1

        if processed % cfg.checkpoint_every == 0:
            ckpt.save()
            logger.info(
                "Stage 2: checkpoint at %d processed (%d ok, %d failed)",
                processed, succeeded, failed,
            )

        if processed % 100 == 0:
            logger.info("Stage 2: progress %d / %d", processed, len(rows))

    # --- Final checkpoint ---
    ckpt.save()

    # --- Write consolidated JSON ---
    if jsonl_path.exists():
        all_results = read_jsonl(jsonl_path)
        write_json(all_results, output_dir / "case_narratives.json")

    # --- Manifest ---
    manifest = build_manifest(
        stage="stage2_generation",
        params={
            "model": cfg.model,
            "provider": "xai",
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "input_path": str(input_path),
            "start": start,
            "end": end,
            "resume": cfg.resume,
        },
        row_count=processed,
        extra={
            "succeeded": succeeded,
            "failed": failed,
            "total_in_checkpoint": len(ckpt.completed),
        },
    )
    save_manifest(manifest, output_dir)

    logger.info(
        "Stage 2: complete — %d processed, %d succeeded, %d failed",
        processed, succeeded, failed,
    )
