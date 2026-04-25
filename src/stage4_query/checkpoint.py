"""Resume support for Stage 4: read existing JSONL and skip completed cases."""

from __future__ import annotations

from pathlib import Path
from typing import Set

from src.common.io_utils import read_jsonl


def completed_prompt_ids(jsonl_path: Path) -> Set[str]:
    """Return the set of prompt_ids in `jsonl_path` whose query_status == 'ok'.

    Records with any other status (failed, invalid, error) are NOT considered
    complete, so a `--resume` run will retry them.
    """
    if not jsonl_path.exists():
        return set()

    done: Set[str] = set()
    for record in read_jsonl(jsonl_path):
        pid = record.get("prompt_id")
        status = record.get("query_status")
        if pid and status == "ok":
            done.add(pid)
    return done
