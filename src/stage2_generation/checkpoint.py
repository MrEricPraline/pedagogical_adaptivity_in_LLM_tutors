"""Checkpoint management for Stage 2 resume support."""

from __future__ import annotations

from pathlib import Path
from typing import Set

from src.common.io_utils import read_json, write_json


class Checkpoint:
    """Track which prompt_ids have been completed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.completed: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = read_json(self.path)
            self.completed = set(data.get("completed_ids", []))

    def save(self) -> None:
        write_json(
            {"completed_ids": sorted(self.completed), "count": len(self.completed)},
            self.path,
        )

    def mark_done(self, prompt_id: str) -> None:
        self.completed.add(prompt_id)

    def is_done(self, prompt_id: str) -> bool:
        return prompt_id in self.completed
