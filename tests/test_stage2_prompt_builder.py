"""Tests for Stage 2 prompt builder."""

from src.common.schemas import (
    BLOOM_LEVELS,
    FORBIDDEN_TERMS,
    KNOWLEDGE_STATES,
    LEARNING_CONTEXTS,
    LEARNING_STAGES,
    SUBJECTS,
)
from src.stage2_generation.prompt_builder import (
    BLOOM_INTENT,
    CONTEXT_DESC,
    KNOWLEDGE_DESC,
    STAGE_DESC,
    SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
)


def test_all_bloom_levels_mapped():
    assert set(BLOOM_INTENT.keys()) == set(BLOOM_LEVELS)


def test_all_knowledge_states_mapped():
    assert set(KNOWLEDGE_DESC.keys()) == set(KNOWLEDGE_STATES)


def test_all_learning_stages_mapped():
    assert set(STAGE_DESC.keys()) == set(LEARNING_STAGES)


def test_all_learning_contexts_mapped():
    assert set(CONTEXT_DESC.keys()) == set(LEARNING_CONTEXTS)


def test_user_prompt_contains_subject():
    prompt = build_user_prompt("Physics", "Apply", "novice", "skill_building", "guided")
    assert "Physics" in prompt


def test_system_prompt_no_forbidden_terms():
    lower = SYSTEM_PROMPT.lower()
    for term in FORBIDDEN_TERMS:
        assert term.lower() not in lower, f"System prompt contains forbidden term: {term}"


def test_user_prompt_no_forbidden_terms():
    for bloom in BLOOM_LEVELS:
        for ks in KNOWLEDGE_STATES:
            for ls in LEARNING_STAGES:
                for lc in LEARNING_CONTEXTS:
                    prompt = build_user_prompt("Mathematics", bloom, ks, ls, lc)
                    lower = prompt.lower()
                    for term in FORBIDDEN_TERMS:
                        assert term.lower() not in lower, (
                            f"Forbidden term '{term}' in prompt for "
                            f"bloom={bloom}, ks={ks}, ls={ls}, lc={lc}"
                        )


def test_build_messages_structure():
    msgs = build_messages("History", "Remember", "informed", "conceptual_orientation", "collaborative")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert len(msgs[0]["content"]) > 0
    assert len(msgs[1]["content"]) > 0
