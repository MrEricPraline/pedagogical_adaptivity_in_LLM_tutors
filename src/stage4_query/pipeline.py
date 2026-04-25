"""Stage 4 orchestrator: query Gemini 3.1 Pro Preview for activity plans.

Reads cases_final.jsonl (the validated narratives produced by Stages 2/3),
sends each one to Gemini with a strict JSON schema demanding exactly 5
learning activities annotated along five pedagogical dimensions, and writes
the structured responses incrementally to JSONL.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.common.io_utils import append_jsonl, read_jsonl, write_json
from src.pipeline.config import PROJECT_ROOT, PipelineConfig
from src.pipeline.manifests import build_manifest, save_manifest
from src.stage4_query.checkpoint import completed_prompt_ids
from src.stage4_query.prompt_builder import (
    SYSTEM_INSTRUCTION,
    build_response_schema,
    build_user_prompt,
)
from src.stage4_query.provider_gemini import GeminiProvider
from src.stage4_query.validator import normalize_response, validate_response

logger = logging.getLogger("pipeline")


def _display_path(path: Path | str) -> str:
    """Return `path` relative to the project root when possible.

    Used in console logs and the manifest so the username and absolute
    filesystem layout never leak (e.g. `/Users/<name>/...`).
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        # Path is outside the project tree — fall back to a basic relative form
        try:
            return os.path.relpath(str(p), str(PROJECT_ROOT))
        except ValueError:
            return p.name

# Fields that should NOT be propagated from the input record into the output
# (they describe the upstream generator, not the Stage 4 query).
_DROP_FIELDS = {
    "narrative",  # captured separately
    "validation",
    "validation_clean",
    "generator_provider",
    "generator_model",
    "generated_at",
    "generation_status",
    "error_message",
    "word_count",
}


def _resolve_input_path(cfg: PipelineConfig) -> Path:
    """Pick the cases_final.jsonl input file, preferring stage3 if present."""
    if cfg.input_path is not None:
        return Path(cfg.input_path)

    stage3 = cfg.stage3_dir / "cases_final.jsonl"
    if stage3.exists():
        return stage3

    return cfg.stage2_dir / "cases_final.jsonl"


def _carry_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the input record with Stage-2 generator fields removed."""
    return {k: v for k, v in record.items() if k not in _DROP_FIELDS}


def _build_output_record(
    *,
    case: Dict[str, Any],
    cfg: PipelineConfig,
    raw_text: str,
    parsed: Optional[Dict[str, Any]],
    status: str,
    error: str,
    elapsed: float,
) -> Dict[str, Any]:
    record: Dict[str, Any] = _carry_metadata(case)
    record["narrative"] = case.get("narrative", "")
    record["response"] = parsed
    record["raw_response_text"] = raw_text
    record["query_status"] = status
    record["error_message"] = error
    record["elapsed_seconds"] = round(elapsed, 3)
    record["target_model"] = cfg.model
    record["thinking_level"] = cfg.thinking_level
    record["temperature"] = cfg.temperature
    record["max_output_tokens"] = cfg.max_output_tokens
    record["queried_at"] = datetime.now(timezone.utc).isoformat()
    return record


def _query_one_case(
    provider: GeminiProvider,
    case: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Query Gemini for a single case and return (status, error, raw, parsed, elapsed)."""
    narrative = case.get("narrative", "")
    user_prompt = build_user_prompt(narrative)

    start = time.time()
    raw_text = ""
    try:
        raw_text = provider.generate_json(
            system_instruction=SYSTEM_INSTRUCTION,
            user_prompt=user_prompt,
            response_schema=schema,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        return {
            "status": "failed",
            "error": str(exc),
            "raw": raw_text,
            "parsed": None,
            "elapsed": elapsed,
        }

    elapsed = time.time() - start

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid",
            "error": f"JSON decode error: {exc}",
            "raw": raw_text,
            "parsed": None,
            "elapsed": elapsed,
        }

    is_valid, reason = validate_response(parsed)
    if not is_valid:
        return {
            "status": "invalid",
            "error": f"Schema validation failed: {reason}",
            "raw": raw_text,
            "parsed": parsed,
            "elapsed": elapsed,
        }

    return {
        "status": "ok",
        "error": "",
        "raw": raw_text,
        "parsed": normalize_response(parsed),
        "elapsed": elapsed,
    }


def run_stage4(cfg: PipelineConfig) -> None:
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    input_path = _resolve_input_path(cfg)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else cfg.stage4_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "gemini_responses.jsonl"
    json_path = output_dir / "gemini_responses.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Stage 4 input not found at {_display_path(input_path)}. "
            "Run earlier stages first or pass --input <path>."
        )

    all_cases: List[Dict[str, Any]] = read_jsonl(input_path)
    logger.info(
        "Stage 4: loaded %d cases from %s", len(all_cases), _display_path(input_path)
    )

    start = cfg.start or 0
    end = cfg.end if cfg.end is not None else len(all_cases)
    cases = all_cases[start:end]
    logger.info(
        "Stage 4: processing rows %d–%d (%d cases) with model=%s thinking_level=%s",
        start, end - 1, len(cases), cfg.model, cfg.thinking_level,
    )

    already_done: Set[str] = set()
    if cfg.resume:
        already_done = completed_prompt_ids(jsonl_path)
        logger.info(
            "Stage 4: resume enabled — %d cases already completed in %s",
            len(already_done), jsonl_path.name,
        )

    provider = GeminiProvider(
        api_key=cfg.gemini_api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        thinking_level=cfg.thinking_level,
        retries=cfg.retries,
        requests_per_minute=cfg.requests_per_minute,
    )

    schema = build_response_schema()

    processed = 0
    skipped = 0
    succeeded = 0
    failed = 0
    interrupted = False

    # Total cases that will actually be queried in this run (excluding resume-skips)
    to_query = sum(
        1 for c in cases
        if c.get("prompt_id") and not (cfg.resume and c.get("prompt_id") in already_done)
    )
    logger.info(
        "Stage 4: %d cases to query in this run (%d will be skipped via --resume)",
        to_query, len(cases) - to_query,
    )

    try:
        for case in cases:
            pid = case.get("prompt_id")
            if not pid:
                logger.warning("Stage 4: skipping record without prompt_id: %r", case)
                continue

            if cfg.resume and pid in already_done:
                skipped += 1
                continue

            outcome = _query_one_case(provider, case, schema)
            record = _build_output_record(
                case=case,
                cfg=cfg,
                raw_text=outcome["raw"],
                parsed=outcome["parsed"],
                status=outcome["status"],
                error=outcome["error"],
                elapsed=outcome["elapsed"],
            )

            append_jsonl(record, jsonl_path)
            processed += 1
            if outcome["status"] == "ok":
                succeeded += 1
            else:
                failed += 1

            # Per-case progress line: counter + id + status + elapsed + running tally
            status = outcome["status"]
            marker = "OK  " if status == "ok" else "FAIL"
            meta_bits = []
            if case.get("subject"):
                meta_bits.append(str(case["subject"]))
            if case.get("bloom"):
                meta_bits.append(str(case["bloom"]))
            meta = f" [{' / '.join(meta_bits)}]" if meta_bits else ""
            err_tail = ""
            if status != "ok" and outcome["error"]:
                err_tail = f" :: {outcome['error'][:160]}"
            logger.info(
                "[%4d/%-4d] %s  %s  %5.2fs  ok=%d fail=%d%s%s",
                processed, to_query, marker, pid, outcome["elapsed"],
                succeeded, failed, meta, err_tail,
            )

            if processed % max(cfg.checkpoint_every, 1) == 0:
                logger.info(
                    "Stage 4: checkpoint — processed=%d ok=%d failed=%d skipped=%d",
                    processed, succeeded, failed, skipped,
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Stage 4: interrupted by user (Ctrl+C) after %d processed "
            "(%d ok, %d failed, %d skipped). Re-run with --resume to continue.",
            processed, succeeded, failed, skipped,
        )

    # Consolidated JSON snapshot. The JSONL is append-only and may contain
    # multiple records for the same prompt_id (e.g. an earlier "failed" entry
    # followed by an "ok" entry from a --resume run). We dedupe here so the
    # consolidated view shows one record per case, preferring "ok" over any
    # other status, then falling back to the most recent entry by `queried_at`.
    # Wrapped so that even on Ctrl+C / unexpected error the snapshot + manifest
    # reflect everything written to the JSONL up to that point.
    if jsonl_path.exists():
        all_results = read_jsonl(jsonl_path)
        deduped: Dict[str, Dict[str, Any]] = {}
        for record in all_results:
            pid = record.get("prompt_id")
            if not pid:
                continue
            current = deduped.get(pid)
            if current is None:
                deduped[pid] = record
                continue
            current_ok = current.get("query_status") == "ok"
            new_ok = record.get("query_status") == "ok"
            if new_ok and not current_ok:
                deduped[pid] = record
            elif new_ok == current_ok:
                # Same status quality; keep the latest by queried_at
                if record.get("queried_at", "") >= current.get("queried_at", ""):
                    deduped[pid] = record
        write_json(list(deduped.values()), json_path)

    finished_at = datetime.now(timezone.utc)
    duration = round(time.time() - started_perf, 3)

    manifest = build_manifest(
        stage="stage4_query_gemini",
        params={
            "model": cfg.model,
            "thinking_level": cfg.thinking_level,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
            "requests_per_minute": cfg.requests_per_minute,
            "retries": cfg.retries,
            "input_file": _display_path(input_path),
            "output_jsonl": _display_path(jsonl_path),
            "output_json": _display_path(json_path),
            "start": start,
            "end": end,
            "resume": cfg.resume,
        },
        row_count=processed,
        extra={
            "total_input_cases": len(cases),
            "processed_cases": processed,
            "skipped_cases": skipped,
            "successful_cases": succeeded,
            "failed_cases": failed,
            "interrupted": interrupted,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    save_manifest(manifest, output_dir)

    logger.info(
        "Stage 4: complete — processed=%d ok=%d failed=%d skipped=%d duration=%ss",
        processed, succeeded, failed, skipped, duration,
    )
