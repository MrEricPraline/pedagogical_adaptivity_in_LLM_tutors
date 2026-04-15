"""Regenerate cases that were flagged by dedup, with enforced diversity."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

from src.common.io_utils import (
    append_jsonl,
    read_csv,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from src.common.schemas import FORBIDDEN_TERMS
from src.common.text_utils import count_words
from src.pipeline.config import PipelineConfig
from src.stage2_generation.checkpoint import Checkpoint
from src.stage2_generation.diversity import FORBIDDEN_OPENING_RE, build_diverse_messages
from src.stage2_generation.provider_xai import XAIProvider

logger = logging.getLogger("pipeline")


def _validate_diverse(text: str, existing_openings: Set[str], cfg: PipelineConfig) -> Dict[str, bool]:
    """Extended validation including opening uniqueness and pattern checks."""
    text = text.strip()
    wc = count_words(text)
    prefix = text[:100]

    forbidden_found = []
    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
            forbidden_found.append(term)

    return {
        "nonempty_ok": len(text) > 0,
        "single_paragraph_ok": "\n" not in text,
        "word_count_ok": cfg.word_count_min <= wc <= cfg.word_count_max,
        "forbidden_terms_ok": len(forbidden_found) == 0,
        "opening_unique_ok": prefix not in existing_openings,
        "opening_pattern_ok": not bool(FORBIDDEN_OPENING_RE.match(text)),
    }


def _is_clean(checks: Dict[str, bool]) -> bool:
    return all(checks.values())


def run_regenerate(cfg: PipelineConfig) -> None:
    """Regenerate cases listed in cases_to_regenerate.csv with diverse prompts."""
    stage2_dir = cfg.output_dir or cfg.stage2_dir
    regen_csv = stage2_dir / "cases_to_regenerate.csv"
    cleaned_jsonl = stage2_dir / "cases_cleaned.jsonl"
    regen_jsonl = stage2_dir / "cases_regenerated.jsonl"
    ckpt_path = stage2_dir / "regen_checkpoint.json"
    final_jsonl = stage2_dir / "cases_final.jsonl"

    if not regen_csv.exists():
        raise FileNotFoundError(
            f"No regeneration list at {regen_csv}. Run 'dedup' first."
        )

    all_cases = read_csv(regen_csv)
    start = cfg.start or 0
    end = cfg.end or len(all_cases)
    cases = all_cases[start:end]
    logger.info("Regenerate: %d cases to process (of %d total)", len(cases), len(all_cases))

    # Collect existing openings to avoid collisions
    existing_openings: Set[str] = set()
    if cleaned_jsonl.exists():
        for r in read_jsonl(cleaned_jsonl):
            existing_openings.add(r.get("narrative", "")[:100])
    if regen_jsonl.exists():
        for r in read_jsonl(regen_jsonl):
            existing_openings.add(r.get("narrative", "")[:100])
    logger.info("Regenerate: %d existing openings to avoid", len(existing_openings))

    # Resume
    ckpt = Checkpoint(ckpt_path)
    if cfg.resume:
        logger.info("Regenerate: resume — %d already done", len(ckpt.completed))

    provider = XAIProvider(cfg)
    processed = 0
    succeeded = 0
    failed = 0

    for row in cases:
        pid = row["prompt_id"]
        if cfg.resume and ckpt.is_done(pid):
            continue

        best_narrative = ""
        best_checks: Dict[str, bool] = {}
        best_status = "validation_failed"
        error = ""

        try:
            for _attempt in range(3):
                messages = build_diverse_messages(row)
                narrative = provider.generate(messages)
                checks = _validate_diverse(narrative, existing_openings, cfg)

                if _is_clean(checks):
                    best_narrative = narrative
                    best_checks = checks
                    best_status = "ok"
                    break
                best_narrative = narrative
                best_checks = checks

        except Exception as exc:
            best_status = "error"
            error = str(exc)
            logger.error("Regenerate error %s: %s", pid, exc)

        wc = count_words(best_narrative) if best_narrative else 0
        result = {
            "prompt_id": pid,
            "bloom": row["bloom"],
            "bloom_band": row["bloom_band"],
            "subject": row["subject"],
            "subject_family": row["subject_family"],
            "knowledge_state": row["knowledge_state"],
            "learning_stage": row["learning_stage"],
            "learning_context": row["learning_context"],
            "narrative": best_narrative,
            "word_count": wc,
            "validation": best_checks,
            "validation_clean": _is_clean(best_checks) if best_checks else False,
            "generator_provider": "xai",
            "generator_model": cfg.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_status": best_status,
            "error_message": error,
        }

        append_jsonl(result, regen_jsonl)
        existing_openings.add(best_narrative[:100])
        ckpt.mark_done(pid)
        processed += 1

        if best_status == "ok":
            succeeded += 1
        else:
            failed += 1

        if processed % cfg.checkpoint_every == 0:
            ckpt.save()
            logger.info(
                "Regenerate: checkpoint at %d (%d ok, %d failed)",
                processed, succeeded, failed,
            )

        if processed % 100 == 0:
            logger.info("Regenerate: progress %d / %d", processed, len(cases))

    ckpt.save()

    # --- Merge into final dataset ---
    logger.info("Regenerate: merging into final dataset …")
    by_pid: Dict[str, dict] = {}

    if cleaned_jsonl.exists():
        for r in read_jsonl(cleaned_jsonl):
            by_pid[r["prompt_id"]] = r
    if regen_jsonl.exists():
        for r in read_jsonl(regen_jsonl):
            by_pid[r["prompt_id"]] = r

    final = sorted(by_pid.values(), key=lambda x: x["prompt_id"])
    write_jsonl(final, final_jsonl)
    write_json(final, stage2_dir / "cases_final.json")

    ok_count = sum(1 for r in final if r.get("generation_status") == "ok")
    logger.info(
        "Regenerate: complete — %d processed, %d ok, %d failed | final dataset: %d cases (%d ok)",
        processed, succeeded, failed, len(final), ok_count,
    )
