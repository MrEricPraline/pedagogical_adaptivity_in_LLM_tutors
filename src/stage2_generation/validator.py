"""Narrative validation checks."""

from __future__ import annotations

from typing import Dict

from src.common.schemas import FORBIDDEN_TERMS
from src.common.text_utils import contains_forbidden_terms, count_words, is_single_paragraph


def validate_narrative(
    text: str,
    word_min: int = 80,
    word_max: int = 120,
) -> Dict[str, bool]:
    """Run all checks and return a dict of {check_name: passed}."""
    wc = count_words(text)
    forbidden_found = contains_forbidden_terms(text, FORBIDDEN_TERMS)

    return {
        "nonempty_ok": len(text.strip()) > 0,
        "single_paragraph_ok": is_single_paragraph(text),
        "word_count_ok": word_min <= wc <= word_max,
        "forbidden_terms_ok": len(forbidden_found) == 0,
    }


def is_clean(checks: Dict[str, bool]) -> bool:
    return all(checks.values())


def should_retry(checks: Dict[str, bool]) -> bool:
    """Retry when the failure is recoverable (length or forbidden terms)."""
    if checks.get("nonempty_ok") is False:
        return False
    return not checks.get("word_count_ok", True) or not checks.get("forbidden_terms_ok", True)
