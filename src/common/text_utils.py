"""Text analysis helpers for narrative validation."""

import re
from typing import List


def count_words(text: str) -> int:
    """Count words using simple whitespace splitting."""
    return len(text.split())


def contains_forbidden_terms(text: str, terms: List[str]) -> List[str]:
    """Return list of forbidden terms found in *text* (case-insensitive)."""
    text_lower = text.lower()
    found: List[str] = []
    for term in terms:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lower):
            found.append(term)
    return found


def is_single_paragraph(text: str) -> bool:
    """True if text has no internal line breaks (ignoring leading/trailing whitespace)."""
    stripped = text.strip()
    return "\n" not in stripped
