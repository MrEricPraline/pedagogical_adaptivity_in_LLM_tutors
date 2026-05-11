"""Pedagogical Adaptivity Index — theory-grounded scoring sub-matrices.

Each decision point (DP1–DP5) is conditioned on one or more learner
variables. For every (selection ∈ {a,b,c,d}, condition value) pair the
matrix returns a score in [-1.0, +1.0] that quantifies the pedagogical
appropriateness of that selection for that condition.

The PAI of an activity is the mean of the five DP sub-scores (each DP
sub-score is itself the mean across its conditioning variables). The PAI
of a case is the mean of the activity-level PAIs across the 5 activities
the model designed for that case.

These matrices must remain in sync with the corrective-data and
fine-tuning code in ``src.stage5_finetune``. Edit here only.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

# ---------------------------------------------------------------------------
# Decision Point 1 — Content level
# Conditioned on: bloom_band, knowledge_state
# ---------------------------------------------------------------------------

M1A: Dict[str, Dict[str, float]] = {
    "a": {"RU":  1.0, "AA": -0.5, "EC": -1.0},
    "b": {"RU":  0.5, "AA":  0.5, "EC": -0.5},
    "c": {"RU": -0.5, "AA":  1.0, "EC":  0.5},
    "d": {"RU": -1.0, "AA":  0.5, "EC":  1.0},
}

M1B: Dict[str, Dict[str, float]] = {
    "a": {"novice":  1.0, "informed":  0.0, "misinformed": -0.5},
    "b": {"novice":  0.5, "informed":  1.0, "misinformed":  0.0},
    "c": {"novice": -0.5, "informed":  0.5, "misinformed":  0.5},
    "d": {"novice": -1.0, "informed": -0.5, "misinformed":  1.0},
}

# ---------------------------------------------------------------------------
# Decision Point 2 — Student task
# Conditioned on: learning_stage, bloom_band
# ---------------------------------------------------------------------------

M2A: Dict[str, Dict[str, float]] = {
    "a": {"conceptual_orientation":  1.0, "skill_building": -0.5, "competency_development": -0.5, "comprehensive_mastery": -0.5},
    "b": {"conceptual_orientation": -0.5, "skill_building":  1.0, "competency_development":  0.0, "comprehensive_mastery": -0.5},
    "c": {"conceptual_orientation":  0.0, "skill_building": -0.5, "competency_development":  1.0, "comprehensive_mastery":  0.5},
    "d": {"conceptual_orientation": -0.5, "skill_building": -0.5, "competency_development":  0.5, "comprehensive_mastery":  1.0},
}

M2B: Dict[str, Dict[str, float]] = {
    "a": {"RU":  1.0, "AA":  0.0, "EC": -0.5},
    "b": {"RU":  0.5, "AA":  0.5, "EC": -0.5},
    "c": {"RU": -0.5, "AA":  1.0, "EC":  0.5},
    "d": {"RU": -1.0, "AA":  0.5, "EC":  1.0},
}

# ---------------------------------------------------------------------------
# Decision Point 3 — Tutor role
# Conditioned on: learning_context, knowledge_state
# ---------------------------------------------------------------------------

M3A: Dict[str, Dict[str, float]] = {
    "a": {"guided": -0.5, "collaborative": -1.0, "autonomous":  1.0},
    "b": {"guided":  1.0, "collaborative":  0.0, "autonomous":  0.0},
    "c": {"guided":  0.5, "collaborative":  1.0, "autonomous": -0.5},
    "d": {"guided": -0.5, "collaborative":  0.5, "autonomous":  0.5},
}

M3B: Dict[str, Dict[str, float]] = {
    "a": {"novice":  1.0, "informed":  0.0, "misinformed":  0.5},
    "b": {"novice":  0.5, "informed":  0.5, "misinformed":  0.0},
    "c": {"novice": -0.5, "informed":  0.5, "misinformed":  1.0},
    "d": {"novice": -1.0, "informed":  1.0, "misinformed": -0.5},
}

# ---------------------------------------------------------------------------
# Decision Point 4 — Student engagement
# Conditioned on: learning_context, bloom_band, learning_stage
# ---------------------------------------------------------------------------

M4A: Dict[str, Dict[str, float]] = {
    "a": {"guided":  0.0, "collaborative": -1.0, "autonomous": -0.5},
    "b": {"guided":  0.5, "collaborative": -0.5, "autonomous":  0.5},
    "c": {"guided":  0.5, "collaborative":  0.0, "autonomous":  1.0},
    "d": {"guided":  0.5, "collaborative":  1.0, "autonomous": -0.5},
}

M4B: Dict[str, Dict[str, float]] = {
    "a": {"RU":  1.0, "AA": -0.5, "EC": -1.0},
    "b": {"RU":  0.5, "AA":  0.5, "EC":  0.0},
    "c": {"RU":  0.0, "AA":  1.0, "EC":  0.5},
    "d": {"RU": -0.5, "AA":  0.5, "EC":  1.0},
}

M4C: Dict[str, Dict[str, float]] = {
    "a": {"conceptual_orientation":  0.5, "skill_building": -0.5, "competency_development": -0.5, "comprehensive_mastery": -1.0},
    "b": {"conceptual_orientation": -0.5, "skill_building":  1.0, "competency_development":  0.0, "comprehensive_mastery": -0.5},
    "c": {"conceptual_orientation":  0.5, "skill_building":  0.5, "competency_development":  1.0, "comprehensive_mastery":  0.5},
    "d": {"conceptual_orientation":  0.0, "skill_building": -0.5, "competency_development":  0.5, "comprehensive_mastery":  1.0},
}

# ---------------------------------------------------------------------------
# Decision Point 5 — Disciplinary method
# Conditioned on: subject_family, learning_stage
# ---------------------------------------------------------------------------

M5A: Dict[str, Dict[str, float]] = {
    "a": {"formal":  1.0, "natural": -0.5, "humanistic": -1.0, "applied":  0.0},
    "b": {"formal":  0.0, "natural":  1.0, "humanistic":  0.0, "applied":  0.5},
    "c": {"formal": -0.5, "natural":  0.0, "humanistic":  1.0, "applied":  0.5},
    "d": {"formal":  0.5, "natural":  0.5, "humanistic":  0.5, "applied":  0.5},
}

M5B: Dict[str, Dict[str, float]] = {
    "a": {"conceptual_orientation": -0.5, "skill_building":  1.0, "competency_development":  0.5, "comprehensive_mastery":  0.0},
    "b": {"conceptual_orientation":  1.0, "skill_building":  0.5, "competency_development":  0.0, "comprehensive_mastery":  0.0},
    "c": {"conceptual_orientation":  0.5, "skill_building": -0.5, "competency_development":  0.5, "comprehensive_mastery":  0.5},
    "d": {"conceptual_orientation":  0.0, "skill_building": -0.5, "competency_development":  0.5, "comprehensive_mastery":  1.0},
}

# ---------------------------------------------------------------------------
# Decision-point registry
#
# Each entry maps DP field name -> (sub-matrices, condition keys to extract
# from the case metadata). The order inside each tuple must match.
# ---------------------------------------------------------------------------

DecisionPointSpec = Tuple[List[Dict[str, Dict[str, float]]], List[str]]

DECISION_POINTS: Dict[str, DecisionPointSpec] = {
    "content_level":       ([M1A, M1B],         ["bloom_band", "knowledge_state"]),
    "student_task":        ([M2A, M2B],         ["learning_stage", "bloom_band"]),
    "tutor_role":          ([M3A, M3B],         ["learning_context", "knowledge_state"]),
    "student_engagement":  ([M4A, M4B, M4C],    ["learning_context", "bloom_band", "learning_stage"]),
    "disciplinary_method": ([M5A, M5B],         ["subject_family", "learning_stage"]),
}

# Short DP labels used in result tables / heatmaps
DP_LABELS: Dict[str, str] = {
    "content_level":       "DP1",
    "student_task":        "DP2",
    "tutor_role":          "DP3",
    "student_engagement":  "DP4",
    "disciplinary_method": "DP5",
}


def conditions_from_meta(meta: Mapping[str, str]) -> Dict[str, List[str]]:
    """Return the ordered condition values per decision point for `meta`.

    Raises KeyError if a required condition is missing — callers should
    enforce upstream that every Stage 4 record carries all five learner
    variables.
    """
    return {
        dp: [meta[k] for k in keys]
        for dp, (_matrices, keys) in DECISION_POINTS.items()
    }
