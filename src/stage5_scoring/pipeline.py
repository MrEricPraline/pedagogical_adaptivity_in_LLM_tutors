"""Stage 5 — apply the PAI scoring matrices to Stage 4 outputs.

Reads the consolidated Gemini snapshot (``data/stage4/gemini_responses.json``)
or, if missing, the incremental JSONL (``gemini_responses.jsonl``) and
produces:

* ``data/stage5/scored_dataset.json`` — one record per case with per-DP
  means, per-activity PAIs, and the aggregate ``prompt_PAI``. Records are
  sorted ascending by ``prompt_PAI`` so the lowest-performing cases come
  first (this is what Phase 1 corrective fine-tuning consumes).
* ``data/stage5/scored_dataset.jsonl`` — same data, JSONL form.
* ``data/stage5/manifest.json`` — provenance metadata.

The module also exposes :func:`run_stage5_scoring` for use by the CLI.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional

from src.common.io_utils import read_json, read_jsonl, write_json, write_jsonl
from src.pipeline.config import DATA_DIR, PROJECT_ROOT, PipelineConfig
from src.pipeline.manifests import build_manifest, save_manifest
from src.stage5_scoring.scorer import (
    CASE_METADATA_KEYS,
    extract_meta,
    score_response,
)

logger = logging.getLogger("pipeline")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return path.name


def _resolve_input(cfg: PipelineConfig) -> Path:
    """Pick the Stage 4 source file. Prefer the consolidated JSON snapshot."""
    if cfg.input_path is not None:
        return Path(cfg.input_path)

    candidate = cfg.stage4_dir / "gemini_responses.json"
    if candidate.exists():
        return candidate

    fallback = cfg.stage4_dir / "gemini_responses.jsonl"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Stage 5 input not found: expected "
        "data/stage4/gemini_responses.json or .jsonl. "
        "Run Stage 4 first or pass --input <path>."
    )


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return read_json(path)


def _is_scoreable(record: Mapping[str, Any]) -> bool:
    if record.get("query_status") != "ok":
        return False
    response = record.get("response")
    if not isinstance(response, dict):
        return False
    activities = response.get("activities")
    return isinstance(activities, list) and len(activities) == 5


def _score_one(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = extract_meta(record)
    activities = record["response"]["activities"]
    scores = score_response(activities, meta)

    out: Dict[str, Any] = {
        "prompt_id": record["prompt_id"],
        # Carry every Stage 1/2 conditioning variable so downstream code
        # (corrective data, interference analysis) does not need to re-join.
        **{k: record[k] for k in CASE_METADATA_KEYS if k in record},
        "bloom": record.get("bloom"),
        "subject": record.get("subject"),
        # Per-activity selections kept for traceability + later regeneration.
        "selections": [
            {dp: act.get(dp, {}).get("selection") for dp in scores["activity_scores"][i]}
            for i, act in enumerate(activities)
        ],
        "activity_scores": [
            {**{k: round(v, 4) for k, v in act_scores.items()}}
            for act_scores in scores["activity_scores"]
        ],
        "dp_means": {dp: round(v, 4) for dp, v in scores["dp_means"].items()},
        "prompt_PAI": round(scores["prompt_PAI"], 4),
    }
    return out


def _summarise(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scored:
        return {"count": 0}

    pais = [r["prompt_PAI"] for r in scored]
    dp_keys = list(scored[0]["dp_means"].keys())
    dp_global_means = {
        dp: round(mean(r["dp_means"][dp] for r in scored), 4) for dp in dp_keys
    }
    return {
        "count": len(scored),
        "prompt_PAI": {
            "mean": round(mean(pais), 4),
            "min": round(min(pais), 4),
            "max": round(max(pais), 4),
        },
        "dp_global_means": dp_global_means,
        "low_30_threshold": round(sorted(pais)[min(29, len(pais) - 1)], 4),
    }


def run_stage5_scoring(cfg: PipelineConfig) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    input_path = _resolve_input(cfg)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / "scored_dataset.json"
    jsonl_out = output_dir / "scored_dataset.jsonl"

    logger.info("Stage 5 scoring: loading %s", _display_path(input_path))
    records = _load_records(input_path)
    logger.info("Stage 5 scoring: %d input records", len(records))

    scored: List[Dict[str, Any]] = []
    skipped_status: List[str] = []
    failed_score: List[Dict[str, Any]] = []

    for rec in records:
        pid = rec.get("prompt_id", "<unknown>")
        if not _is_scoreable(rec):
            skipped_status.append(pid)
            continue
        try:
            scored.append(_score_one(rec))
        except Exception as exc:  # noqa: BLE001
            failed_score.append({"prompt_id": pid, "error": str(exc)})
            logger.warning("Stage 5 scoring: %s failed (%s)", pid, exc)

    # Deduplicate by prompt_id keeping the highest PAI (defensive — Stage 4
    # already dedupes its consolidated JSON, but JSONL inputs may carry
    # multiple entries from resumed runs).
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in scored:
        prev = by_id.get(r["prompt_id"])
        if prev is None or r["prompt_PAI"] > prev["prompt_PAI"]:
            by_id[r["prompt_id"]] = r
    scored = list(by_id.values())

    scored.sort(key=lambda r: r["prompt_PAI"])

    write_json(scored, json_out)
    write_jsonl(scored, jsonl_out)

    summary = _summarise(scored)

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    manifest = build_manifest(
        stage="stage5_pai_scoring",
        params={
            "input_file": _display_path(input_path),
            "output_json": _display_path(json_out),
            "output_jsonl": _display_path(jsonl_out),
        },
        row_count=len(scored),
        extra={
            "input_records": len(records),
            "scored": len(scored),
            "skipped_non_ok_or_invalid": len(skipped_status),
            "failed_to_score": len(failed_score),
            "summary": summary,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    save_manifest(manifest, output_dir)

    logger.info(
        "Stage 5 scoring: complete — scored=%d skipped=%d failed=%d "
        "(mean PAI=%.4f, range=[%.4f, %.4f]) in %.2fs",
        len(scored), len(skipped_status), len(failed_score),
        summary.get("prompt_PAI", {}).get("mean", 0.0),
        summary.get("prompt_PAI", {}).get("min", 0.0),
        summary.get("prompt_PAI", {}).get("max", 0.0),
        duration,
    )

    return {
        "scored": scored,
        "summary": summary,
        "skipped": skipped_status,
        "failed": failed_score,
        "output_json": json_out,
    }
