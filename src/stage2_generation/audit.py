"""Dataset audit: validate the final JSONL for quality and consistency."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List

from src.common.io_utils import read_jsonl
from src.common.schemas import FORBIDDEN_TERMS
from src.stage2_generation.diversity import FORBIDDEN_OPENING_RE

logger = logging.getLogger("pipeline")


def run_audit(path: Path) -> bool:
    """Run all quality checks and print a report. Returns True if clean."""
    if not path.exists():
        logger.error("Audit: file not found — %s", path)
        return False

    rows = read_jsonl(path)
    total = len(rows)
    logger.info("Audit: %s — %d cases", path.name, total)

    issues = 0

    # 1. Duplicate prompt_ids
    pid_counts = Counter(r["prompt_id"] for r in rows)
    dups = {pid: cnt for pid, cnt in pid_counts.items() if cnt > 1}
    _log_section("Duplicate prompt_ids", len(dups))
    for pid, cnt in sorted(dups.items()):
        logger.info("  %s: %dx", pid, cnt)
    issues += len(dups)

    # 2. Generation status
    status_counts = Counter(r.get("generation_status", "unknown") for r in rows)
    logger.info("[2] Generation status:")
    for s, c in sorted(status_counts.items()):
        logger.info("     %s: %d (%.1f%%)", s, c, 100 * c / total)

    # 3. Similar openings
    openings: dict = {}
    for r in rows:
        prefix = r.get("narrative", "")[:100]
        openings.setdefault(prefix, []).append(r["prompt_id"])
    dup_groups = {k: v for k, v in openings.items() if len(v) > 1}
    dup_total = sum(len(v) for v in dup_groups.values())
    _log_section("Groups sharing first 100 chars", len(dup_groups))
    logger.info("     Total cases in groups: %d", dup_total)
    for prefix, pids in sorted(dup_groups.items(), key=lambda x: -len(x[1]))[:5]:
        logger.info('     [%dx] "%s…"', len(pids), prefix[:60])
    issues += len(dup_groups)

    # 4. Forbidden opening patterns
    bad_openings = [
        r["prompt_id"] for r in rows
        if FORBIDDEN_OPENING_RE.match(r.get("narrative", ""))
    ]
    _log_section("Forbidden opening patterns", len(bad_openings))
    issues += len(bad_openings)

    # 5. Forbidden terms
    cases_with_terms: List = []
    for r in rows:
        text_lower = r.get("narrative", "").lower()
        found = [
            t for t in FORBIDDEN_TERMS
            if re.search(r"\b" + re.escape(t) + r"\b", text_lower)
        ]
        if found:
            cases_with_terms.append((r["prompt_id"], found))
    _log_section("Cases with forbidden terms", len(cases_with_terms))
    for pid, terms in cases_with_terms[:10]:
        logger.info("     %s: %s", pid, terms)
    issues += len(cases_with_terms)

    # 6. Word count
    wcs = [r.get("word_count", len(r.get("narrative", "").split())) for r in rows]
    in_range = sum(1 for w in wcs if 80 <= w <= 120)
    out_of_range = total - in_range
    logger.info("[6] Word count:")
    logger.info("     80–120: %d/%d (%.1f%%)", in_range, total, 100 * in_range / total)
    logger.info("     Min: %d  Max: %d  Avg: %.1f", min(wcs), max(wcs), sum(wcs) / len(wcs))
    issues += out_of_range

    # 7. Variable balance
    logger.info("[7] Variable balance:")
    for col in ["bloom", "knowledge_state", "learning_stage", "learning_context", "subject"]:
        dist = Counter(r.get(col, "") for r in rows)
        counts = [c for _, c in sorted(dist.items())]
        spread = (max(counts) - min(counts)) / (sum(counts) / len(counts)) if counts else 0
        logger.info(
            "     %s (%d levels): min=%d  max=%d  spread=%.2f",
            col, len(dist), min(counts), max(counts), spread,
        )

    # 8. Unique coverage
    unique_pids = len(set(r["prompt_id"] for r in rows))
    logger.info("[8] Unique prompt_ids: %d (expected 2000)", unique_pids)

    # Summary
    clean = issues == 0
    logger.info("=" * 60)
    if clean:
        logger.info("PASS: dataset is clean and ready.")
    else:
        logger.info("ISSUES FOUND: %d", issues)
        if dups:
            logger.info("  - %d duplicate prompt_ids", len(dups))
        if dup_groups:
            logger.info("  - %d groups with similar openings", len(dup_groups))
        if bad_openings:
            logger.info("  - %d forbidden opening patterns", len(bad_openings))
        if cases_with_terms:
            logger.info("  - %d cases with forbidden terms", len(cases_with_terms))
        if out_of_range:
            logger.info("  - %d cases outside 80–120 word range", out_of_range)

    return clean


def _log_section(label: str, count: int) -> None:
    logger.info("[*] %s: %d", label, count)
