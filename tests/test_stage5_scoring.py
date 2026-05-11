"""Tests for the Stage 5 PAI scoring layer."""

from __future__ import annotations

import pytest

from src.stage5_scoring.matrices import DECISION_POINTS, M1A, M1B
from src.stage5_scoring.scorer import (
    extract_meta,
    find_optimal_selections,
    score_dp_selection,
    score_response,
)


CASE_META = {
    "bloom_band": "RU",
    "knowledge_state": "novice",
    "learning_stage": "conceptual_orientation",
    "learning_context": "guided",
    "subject_family": "natural",
}


def test_decision_points_registry_complete():
    assert set(DECISION_POINTS) == {
        "content_level",
        "student_task",
        "tutor_role",
        "student_engagement",
        "disciplinary_method",
    }


def test_score_dp_selection_matches_matrix_mean():
    # content_level: M1A on bloom_band, M1B on knowledge_state.
    expected = (M1A["a"]["RU"] + M1B["a"]["novice"]) / 2
    assert score_dp_selection("content_level", "a", CASE_META) == pytest.approx(expected)


def test_score_dp_invalid_selection():
    with pytest.raises(ValueError):
        score_dp_selection("content_level", "z", CASE_META)


def test_score_dp_unknown_dp():
    with pytest.raises(ValueError):
        score_dp_selection("not_a_dp", "a", CASE_META)


def test_extract_meta_missing_keys():
    with pytest.raises(KeyError):
        extract_meta({"bloom_band": "RU"})


def test_find_optimal_selections_picks_highest():
    """For an RU/novice case the optimal content_level should be 'a' (1.0 + 1.0)/2 = 1.0."""
    optimal = find_optimal_selections(CASE_META)
    assert optimal["content_level"]["selection"] == "a"
    assert optimal["content_level"]["score"] == pytest.approx(1.0)


def test_find_optimal_returns_all_dps():
    optimal = find_optimal_selections(CASE_META)
    assert set(optimal.keys()) == set(DECISION_POINTS.keys())
    for info in optimal.values():
        assert info["selection"] in {"a", "b", "c", "d"}
        assert -1.0 <= info["score"] <= 1.0


def _build_activity(selections):
    return {
        dp: {"selection": sel, "description": "x"}
        for dp, sel in selections.items()
    }


def test_score_response_aggregates():
    selections = {dp: "a" for dp in DECISION_POINTS}
    activities = [_build_activity(selections) for _ in range(5)]

    out = score_response(activities, CASE_META)
    assert "activity_scores" in out
    assert "dp_means" in out
    assert "prompt_PAI" in out
    assert len(out["activity_scores"]) == 5
    # All activities identical → activity PAI is constant and equal to prompt PAI.
    assert out["activity_scores"][0]["activity_PAI"] == pytest.approx(out["prompt_PAI"])


def test_optimal_response_is_at_or_above_any_other():
    """Manual sanity: response built from optimal selections should outscore
    a response built from random selections."""
    optimal = find_optimal_selections(CASE_META)
    optimal_sel = {dp: info["selection"] for dp, info in optimal.items()}
    optimal_act = [_build_activity(optimal_sel) for _ in range(5)]
    optimal_score = score_response(optimal_act, CASE_META)["prompt_PAI"]

    other_sel = {dp: ("d" if optimal_sel[dp] == "a" else "a") for dp in DECISION_POINTS}
    other_act = [_build_activity(other_sel) for _ in range(5)]
    other_score = score_response(other_act, CASE_META)["prompt_PAI"]

    assert optimal_score >= other_score
