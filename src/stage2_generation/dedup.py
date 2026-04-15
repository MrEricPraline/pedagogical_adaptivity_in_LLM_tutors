"""Deduplicate prompt_ids and identify narratives with similar openings."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.common.io_utils import read_jsonl, write_csv, write_jsonl
from src.common.schemas import FACTORIAL_COLUMNS

logger = logging.getLogger("pipeline")

PREFIX_LEN = 100


def run_dedup(stage2_dir: Path) -> Tuple[int, int]:
    """Deduplicate and split the JSONL into clean + to-regenerate sets.

    Returns (kept_count, regen_count).
    """
    jsonl_path = stage2_dir / "case_narratives.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No JSONL at {jsonl_path}. Run Stage 2 first.")

    rows = read_jsonl(jsonl_path)
    logger.info("Dedup: loaded %d rows from %s", len(rows), jsonl_path)

    # 1. Remove duplicate prompt_ids (keep first)
    seen: OrderedDict = OrderedDict()
    dup_count = 0
    for r in rows:
        pid = r["prompt_id"]
        if pid not in seen:
            seen[pid] = r
        else:
            dup_count += 1
    unique = list(seen.values())
    logger.info("Dedup: removed %d duplicate prompt_ids -> %d unique", dup_count, len(unique))

    # 2. Group by first N chars of narrative
    groups: Dict[str, List[dict]] = {}
    for r in unique:
        prefix = r.get("narrative", "")[:PREFIX_LEN]
        groups.setdefault(prefix, []).append(r)

    to_keep: List[dict] = []
    to_regenerate: List[dict] = []
    for members in groups.values():
        to_keep.append(members[0])
        to_regenerate.extend(members[1:])

    shared_groups = sum(1 for g in groups.values() if len(g) > 1)
    logger.info("Dedup: %d groups with shared openings", shared_groups)
    logger.info("Dedup: keeping %d, marking %d for regeneration", len(to_keep), len(to_regenerate))

    # 3. Write outputs
    write_jsonl([r for r in to_keep], stage2_dir / "cases_cleaned.jsonl")

    regen_rows = sorted(to_regenerate, key=lambda x: x["prompt_id"])
    write_csv(
        [{k: r[k] for k in FACTORIAL_COLUMNS} for r in regen_rows],
        stage2_dir / "cases_to_regenerate.csv",
        FACTORIAL_COLUMNS,
    )

    logger.info("Dedup: wrote cases_cleaned.jsonl and cases_to_regenerate.csv")
    return len(to_keep), len(to_regenerate)
