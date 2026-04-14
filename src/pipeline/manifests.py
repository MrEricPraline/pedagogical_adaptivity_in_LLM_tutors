"""Helpers to create and write stage manifests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.io_utils import write_json


def build_manifest(
    stage: str,
    params: Dict[str, Any],
    row_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "params": params,
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: Dict[str, Any], directory: Path) -> Path:
    path = directory / "manifest.json"
    write_json(manifest, path)
    return path
