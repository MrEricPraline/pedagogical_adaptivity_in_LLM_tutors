"""Tests for IO utilities: CSV, JSONL, JSON roundtrip."""

import json
from pathlib import Path

from src.common.io_utils import (
    read_csv,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


def test_csv_roundtrip(tmp_path: Path):
    rows = [
        {"prompt_id": "P-0001", "bloom": "Remember", "subject": "Math"},
        {"prompt_id": "P-0002", "bloom": "Apply", "subject": "Physics"},
    ]
    path = tmp_path / "test.csv"
    write_csv(rows, path, columns=["prompt_id", "bloom", "subject"])
    loaded = read_csv(path)
    assert len(loaded) == 2
    assert loaded[0]["prompt_id"] == "P-0001"
    assert loaded[1]["subject"] == "Physics"


def test_jsonl_roundtrip(tmp_path: Path):
    rows = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
    ]
    path = tmp_path / "test.jsonl"
    write_jsonl(rows, path)
    loaded = read_jsonl(path)
    assert loaded == rows


def test_json_roundtrip(tmp_path: Path):
    data = {"key": "value", "nested": {"a": 1}}
    path = tmp_path / "test.json"
    write_json(data, path)
    loaded = read_json(path)
    assert loaded == data


def test_jsonl_append(tmp_path: Path):
    from src.common.io_utils import append_jsonl

    path = tmp_path / "append.jsonl"
    append_jsonl({"id": 1}, path)
    append_jsonl({"id": 2}, path)
    loaded = read_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == 1
    assert loaded[1]["id"] == 2


def test_csv_unicode(tmp_path: Path):
    rows = [{"name": "José", "city": "São Paulo"}]
    path = tmp_path / "unicode.csv"
    write_csv(rows, path, columns=["name", "city"])
    loaded = read_csv(path)
    assert loaded[0]["name"] == "José"
