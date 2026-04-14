"""Generate the full-factorial sample from experimental variables."""

import itertools
from typing import List

from src.common.schemas import (
    BLOOM_BAND_MAP,
    BLOOM_LEVELS,
    KNOWLEDGE_STATES,
    LEARNING_CONTEXTS,
    LEARNING_STAGES,
    SUBJECT_FAMILY_MAP,
    SUBJECTS,
    FactorialRow,
)


def generate_factorial() -> List[FactorialRow]:
    """Return every combination of the five experimental variables."""
    rows: List[FactorialRow] = []
    counter = 0

    for bloom, knowledge, stage, context, subject in itertools.product(
        BLOOM_LEVELS,
        KNOWLEDGE_STATES,
        LEARNING_STAGES,
        LEARNING_CONTEXTS,
        SUBJECTS,
    ):
        counter += 1
        rows.append(
            FactorialRow(
                prompt_id=f"P-{counter:04d}",
                bloom=bloom,
                bloom_band=BLOOM_BAND_MAP[bloom],
                subject=subject,
                subject_family=SUBJECT_FAMILY_MAP[subject],
                knowledge_state=knowledge,
                learning_stage=stage,
                learning_context=context,
            )
        )

    return rows
