"""Tests for Stage 1 factorial generator."""

from src.common.schemas import (
    BLOOM_LEVELS,
    FACTORIAL_COLUMNS,
    KNOWLEDGE_STATES,
    LEARNING_CONTEXTS,
    LEARNING_STAGES,
    SUBJECTS,
)
from src.stage1_factorial.generator import generate_factorial


def test_row_count():
    rows = generate_factorial()
    expected = (
        len(BLOOM_LEVELS)
        * len(KNOWLEDGE_STATES)
        * len(LEARNING_STAGES)
        * len(LEARNING_CONTEXTS)
        * len(SUBJECTS)
    )
    assert len(rows) == expected == 2160


def test_prompt_ids_unique():
    rows = generate_factorial()
    ids = [r.prompt_id for r in rows]
    assert len(ids) == len(set(ids))


def test_prompt_id_format():
    rows = generate_factorial()
    for r in rows:
        assert r.prompt_id.startswith("P-")
        assert len(r.prompt_id) == 6  # P-XXXX


def test_all_columns_present():
    rows = generate_factorial()
    d = rows[0].to_dict()
    assert set(d.keys()) == set(FACTORIAL_COLUMNS)


def test_bloom_bands_assigned():
    rows = generate_factorial()
    for r in rows:
        assert r.bloom_band in ("Lower-order", "Middle-order", "Higher-order")


def test_subject_families_assigned():
    rows = generate_factorial()
    families = {r.subject_family for r in rows}
    assert families == {"STEM", "Humanities", "Social Sciences"}
