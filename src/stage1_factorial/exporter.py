"""Export factorial sample to CSV and JSONL."""

from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import write_csv, write_jsonl
from src.common.schemas import FACTORIAL_COLUMNS, FactorialRow


def export_csv(rows: List[FactorialRow], output_dir: Path) -> Path:
    path = output_dir / "factorial_sample.csv"
    write_csv([r.to_dict() for r in rows], path, FACTORIAL_COLUMNS)
    return path


def export_jsonl_from_dicts(rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Write JSONL from plain dicts (e.g. rows read from an existing CSV)."""
    path = output_dir / "factorial_sample.jsonl"
    write_jsonl(rows, path)
    return path
