"""Pure scoring functions for the Pedagogical Adaptivity Index."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Mapping, Tuple

from src.stage5_scoring.matrices import DECISION_POINTS

CASE_METADATA_KEYS: Tuple[str, ...] = (
    "bloom_band",
    "knowledge_state",
    "learning_stage",
    "learning_context",
    "subject_family",
)

VALID_SELECTIONS = ("a", "b", "c", "d")


def extract_meta(record: Mapping[str, Any]) -> Dict[str, str]:
    """Return the five conditioning variables from a Stage 4 record."""
    missing = [k for k in CASE_METADATA_KEYS if k not in record]
    if missing:
        raise KeyError(f"Record is missing case metadata: {missing}")
    return {k: record[k] for k in CASE_METADATA_KEYS}


def score_dp_selection(dp: str, selection: str, meta: Mapping[str, str]) -> float:
    """Score one DP selection given the case's learner conditions.

    Returns the mean of the relevant sub-matrix lookups.
    """
    if selection not in VALID_SELECTIONS:
        raise ValueError(f"Invalid selection {selection!r} (expected one of {VALID_SELECTIONS})")
    if dp not in DECISION_POINTS:
        raise ValueError(f"Unknown decision point {dp!r}")

    matrices, keys = DECISION_POINTS[dp]
    return mean(matrix[selection][meta[key]] for matrix, key in zip(matrices, keys))


def score_activity(activity: Mapping[str, Any], meta: Mapping[str, str]) -> Dict[str, float]:
    """Return per-DP scores for one activity.

    `activity` is one element of the Stage 4 ``response.activities`` list and
    must contain ``{dp_name: {"selection": "a"|...}}`` for every DP.
    """
    out: Dict[str, float] = {}
    for dp in DECISION_POINTS:
        block = activity.get(dp)
        if not isinstance(block, dict):
            raise ValueError(f"Activity is missing DP block {dp!r}")
        sel = block.get("selection")
        out[dp] = score_dp_selection(dp, sel, meta)
    return out


def score_response(
    activities: List[Mapping[str, Any]],
    meta: Mapping[str, str],
) -> Dict[str, Any]:
    """Score a full 5-activity response and return per-activity + aggregate scores.

    Output structure:
        {
            "activity_scores": [   # length == len(activities)
                {"content_level": float, ..., "activity_PAI": float},
                ...
            ],
            "dp_means": {"content_level": float, ..., "disciplinary_method": float},
            "prompt_PAI": float,
        }
    """
    if not activities:
        raise ValueError("Cannot score an empty list of activities")

    activity_scores: List[Dict[str, float]] = []
    dp_accumulator: Dict[str, List[float]] = {dp: [] for dp in DECISION_POINTS}

    for act in activities:
        per_dp = score_activity(act, meta)
        per_dp["activity_PAI"] = mean(per_dp.values())
        activity_scores.append(per_dp)
        for dp, value in per_dp.items():
            if dp in dp_accumulator:
                dp_accumulator[dp].append(value)

    dp_means = {dp: mean(vals) for dp, vals in dp_accumulator.items()}
    prompt_pai = mean(dp_means.values())

    return {
        "activity_scores": activity_scores,
        "dp_means": dp_means,
        "prompt_PAI": prompt_pai,
    }


def find_optimal_selections(meta: Mapping[str, str]) -> Dict[str, Dict[str, float]]:
    """For a learner profile, find the selection per DP that maximises the
    sub-matrix mean. Ties broken by lexicographic order on the selection
    letter (deterministic for reproducibility).

    Returns: ``{dp: {"selection": "a"|..., "score": float}}``.
    """
    optimal: Dict[str, Dict[str, float]] = {}
    for dp in DECISION_POINTS:
        best_sel = "a"
        best_score = float("-inf")
        for sel in VALID_SELECTIONS:
            score = score_dp_selection(dp, sel, meta)
            if score > best_score:
                best_score = score
                best_sel = sel
        optimal[dp] = {"selection": best_sel, "score": best_score}
    return optimal
