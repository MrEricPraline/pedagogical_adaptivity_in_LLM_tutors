"""Tests for Stage 2 narrative validator."""

from src.stage2_generation.validator import is_clean, should_retry, validate_narrative


def test_valid_narrative():
    text = " ".join(["word"] * 100)
    checks = validate_narrative(text)
    assert checks["nonempty_ok"] is True
    assert checks["single_paragraph_ok"] is True
    assert checks["word_count_ok"] is True
    assert checks["forbidden_terms_ok"] is True
    assert is_clean(checks)


def test_empty_narrative():
    checks = validate_narrative("")
    assert checks["nonempty_ok"] is False
    assert not is_clean(checks)
    assert not should_retry(checks)


def test_too_short():
    text = " ".join(["word"] * 30)
    checks = validate_narrative(text)
    assert checks["word_count_ok"] is False
    assert should_retry(checks)


def test_too_long():
    text = " ".join(["word"] * 200)
    checks = validate_narrative(text)
    assert checks["word_count_ok"] is False
    assert should_retry(checks)


def test_multiple_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here."
    # Still within word bounds if padded
    padded = text + " " + " ".join(["word"] * 80)
    checks = validate_narrative(padded)
    assert checks["single_paragraph_ok"] is False


def test_forbidden_term_bloom():
    text = " ".join(["word"] * 90) + " Bloom level detected."
    checks = validate_narrative(text)
    assert checks["forbidden_terms_ok"] is False
    assert should_retry(checks)


def test_forbidden_term_scaffolding():
    text = " ".join(["word"] * 90) + " using scaffolding theory."
    checks = validate_narrative(text)
    assert checks["forbidden_terms_ok"] is False


def test_custom_word_bounds():
    text = " ".join(["word"] * 50)
    checks = validate_narrative(text, word_min=40, word_max=60)
    assert checks["word_count_ok"] is True


def test_boundary_word_count():
    text_80 = " ".join(["word"] * 80)
    assert validate_narrative(text_80)["word_count_ok"] is True

    text_120 = " ".join(["word"] * 120)
    assert validate_narrative(text_120)["word_count_ok"] is True

    text_121 = " ".join(["word"] * 121)
    assert validate_narrative(text_121)["word_count_ok"] is False
