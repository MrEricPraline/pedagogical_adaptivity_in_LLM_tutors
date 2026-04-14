"""Experiment variable definitions and data schemas."""

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Bloom's taxonomy
# ---------------------------------------------------------------------------

BLOOM_LEVELS: List[str] = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

BLOOM_BAND_MAP: Dict[str, str] = {
    "Remember": "Lower-order",
    "Understand": "Lower-order",
    "Apply": "Middle-order",
    "Analyze": "Middle-order",
    "Evaluate": "Higher-order",
    "Create": "Higher-order",
}

# ---------------------------------------------------------------------------
# Subjects and families
# ---------------------------------------------------------------------------

SUBJECT_FAMILY_MAP: Dict[str, str] = {
    "Mathematics": "STEM",
    "Physics": "STEM",
    "Biology": "STEM",
    "Chemistry": "STEM",
    "Computer Science": "STEM",
    "History": "Humanities",
    "Literature": "Humanities",
    "Philosophy": "Humanities",
    "Economics": "Social Sciences",
    "Psychology": "Social Sciences",
}

SUBJECTS: List[str] = list(SUBJECT_FAMILY_MAP.keys())

# ---------------------------------------------------------------------------
# Other experimental variables
# ---------------------------------------------------------------------------

KNOWLEDGE_STATES: List[str] = ["novice", "informed", "misinformed"]

LEARNING_STAGES: List[str] = [
    "conceptual_orientation",
    "skill_building",
    "competency_development",
    "comprehensive_mastery",
]

LEARNING_CONTEXTS: List[str] = ["guided", "collaborative", "autonomous"]

# ---------------------------------------------------------------------------
# Column ordering for Stage 1 output
# ---------------------------------------------------------------------------

FACTORIAL_COLUMNS: List[str] = [
    "prompt_id",
    "bloom",
    "bloom_band",
    "subject",
    "subject_family",
    "knowledge_state",
    "learning_stage",
    "learning_context",
]

# ---------------------------------------------------------------------------
# Forbidden terms for narrative validation
# ---------------------------------------------------------------------------

FORBIDDEN_TERMS: List[str] = [
    "bloom",
    "bloom's level",
    "knowledge state",
    "learning stage",
    "learning context",
    "cognitive level",
    "schema",
    "assimilation",
    "accommodation",
    "competency",
    "scaffolding theory",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FactorialRow:
    prompt_id: str
    bloom: str
    bloom_band: str
    subject: str
    subject_family: str
    knowledge_state: str
    learning_stage: str
    learning_context: str

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "bloom": self.bloom,
            "bloom_band": self.bloom_band,
            "subject": self.subject,
            "subject_family": self.subject_family,
            "knowledge_state": self.knowledge_state,
            "learning_stage": self.learning_stage,
            "learning_context": self.learning_context,
        }


@dataclass
class NarrativeResult:
    prompt_id: str
    bloom: str
    bloom_band: str
    subject: str
    subject_family: str
    knowledge_state: str
    learning_stage: str
    learning_context: str
    narrative: str = ""
    word_count: int = 0
    validation: Dict[str, bool] = field(default_factory=dict)
    validation_clean: bool = False
    generator_provider: str = ""
    generator_model: str = ""
    generated_at: str = ""
    generation_status: str = ""
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "bloom": self.bloom,
            "bloom_band": self.bloom_band,
            "subject": self.subject,
            "subject_family": self.subject_family,
            "knowledge_state": self.knowledge_state,
            "learning_stage": self.learning_stage,
            "learning_context": self.learning_context,
            "narrative": self.narrative,
            "word_count": self.word_count,
            "validation": self.validation,
            "validation_clean": self.validation_clean,
            "generator_provider": self.generator_provider,
            "generator_model": self.generator_model,
            "generated_at": self.generated_at,
            "generation_status": self.generation_status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_factorial_row(cls, row: FactorialRow) -> "NarrativeResult":
        return cls(
            prompt_id=row.prompt_id,
            bloom=row.bloom,
            bloom_band=row.bloom_band,
            subject=row.subject,
            subject_family=row.subject_family,
            knowledge_state=row.knowledge_state,
            learning_stage=row.learning_stage,
            learning_context=row.learning_context,
        )
