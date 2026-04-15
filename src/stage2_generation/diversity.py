"""Diversity pools and enhanced prompt builder for narrative regeneration."""

from __future__ import annotations

import random
import re
from typing import Dict, List

from src.stage2_generation.prompt_builder import (
    BLOOM_INTENT,
    CONTEXT_DESC,
    KNOWLEDGE_DESC,
    STAGE_DESC,
)

# ---------------------------------------------------------------------------
# Diversity pools
# ---------------------------------------------------------------------------

NAMES: List[str] = [
    "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley",
    "Quinn", "Avery", "Jamie", "Dakota", "Skyler", "Reese", "Finley",
    "Rowan", "Emery", "Kai", "Noor", "Sasha", "Priya", "Leo",
    "Tomás", "Wei", "Amara", "Yuki", "Dani", "Nico", "Lena", "Ravi", "Zara",
]

AGE_PROFILES: List[str] = [
    "a 14-year-old middle school student",
    "a 15-year-old high school freshman",
    "a 16-year-old high school sophomore",
    "a 17-year-old high school junior",
    "an 18-year-old high school senior",
    "a 19-year-old college freshman",
    "a 20-year-old college sophomore",
    "a 21-year-old college junior",
    "a 22-year-old college senior",
    "a 23-year-old graduate student",
    "a 25-year-old master's student",
    "a 28-year-old returning adult learner",
    "a 30-year-old professional taking evening classes",
    "a 35-year-old career changer enrolled in a bootcamp",
    "a 40-year-old parent studying part-time online",
]

SETTINGS: List[str] = [
    "at a kitchen table late at night",
    "on a commuter train with a tablet",
    "at a desk in a shared dorm room",
    "in a noisy campus cafeteria",
    "at a picnic table in a park between classes",
    "in a community center computer lab",
    "at a standing desk in a co-working space",
    "in a basement study room with flickering lights",
    "on a couch with a laptop balanced on their knees",
    "in a public library reading room at midday",
    "at a coffee shop counter near the window",
    "in the back row of a nearly empty lecture hall",
    "in a home office with the door closed",
    "at a lab bench between experiments",
    "on a bedroom floor surrounded by open textbooks",
    "in a tutoring center waiting area",
    "at a dining table while siblings do homework nearby",
    "on a bus using a phone to review notes",
    "in an after-school program room",
    "at a workstation in a makerspace",
]

# ---------------------------------------------------------------------------
# Forbidden opening patterns
# ---------------------------------------------------------------------------

FORBIDDEN_OPENING_RE = re.compile(
    r"^(In a (quiet|cozy|small|dimly.lit)\s+(corner|room|nook|space))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Enhanced system prompt
# ---------------------------------------------------------------------------

DIVERSE_SYSTEM_PROMPT = (
    "You are a writer of realistic educational scenarios. "
    "Write a single paragraph of 80 to 120 words describing a concrete, "
    "realistic learning or tutoring situation. "
    "IMPORTANT STYLE RULES: "
    "- Do NOT start with 'In a quiet corner', 'In a cozy corner', 'In a small corner', "
    "or any prepositional phrase about location as the opening words. "
    "- Do NOT start with a setting description. Start with the student doing something. "
    "- Use the specific student name, age, and setting provided. "
    "- The scenario must feel natural and unique. "
    "- Do NOT use any pedagogical or theoretical terminology. "
    "- Do NOT mention any framework, taxonomy, theory name, or technical education label. "
    "- Include a concrete learning goal. "
    "- Write only the paragraph — no title, no bullet points, no labels."
)


def build_diverse_messages(row: Dict[str, str]) -> list:
    """Build chat messages with randomised name, age, and setting."""
    name = random.choice(NAMES)
    age_profile = random.choice(AGE_PROFILES)
    setting = random.choice(SETTINGS)

    intent = BLOOM_INTENT[row["bloom"]].format(subject=row["subject"])
    knowledge = KNOWLEDGE_DESC[row["knowledge_state"]]
    stage = STAGE_DESC[row["learning_stage"]]
    context = CONTEXT_DESC[row["learning_context"]]

    user_msg = (
        f"Write a realistic scenario about {name}, {age_profile}, who {intent}. "
        f"The student {knowledge} and {stage}. "
        f"The situation takes place {context}, {setting}. "
        f"Start the paragraph with {name} doing an action — "
        f"do NOT open with a location or setting description. "
        f"Describe what {name} is doing, what challenge they face, "
        f"and what specific learning goal they are working toward."
    )

    return [
        {"role": "system", "content": DIVERSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
