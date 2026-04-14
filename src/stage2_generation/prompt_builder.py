"""Translate experimental variables into natural-language prompts for Grok."""

from typing import Dict

# ---------------------------------------------------------------------------
# Natural-language mappings (no pedagogical jargon)
# ---------------------------------------------------------------------------

BLOOM_INTENT: Dict[str, str] = {
    "Remember": "needs to recall specific facts or definitions about {subject}",
    "Understand": "is trying to explain a concept in their own words in {subject}",
    "Apply": "needs to use a concept to solve a new problem in {subject}",
    "Analyze": "is working on breaking down a complex topic and examining relationships in {subject}",
    "Evaluate": "is trying to critically assess different approaches to a problem in {subject}",
    "Create": "wants to design or produce something original in {subject}",
}

KNOWLEDGE_DESC: Dict[str, str] = {
    "novice": "has no prior knowledge of the topic",
    "informed": "has a solid foundational understanding of the topic",
    "misinformed": "holds an incorrect belief about a key concept that needs to be addressed",
}

STAGE_DESC: Dict[str, str] = {
    "conceptual_orientation": "is just beginning to explore the topic",
    "skill_building": "is practicing and developing specific skills in the area",
    "competency_development": "is working toward reliable proficiency",
    "comprehensive_mastery": "is refining deep, integrated expertise",
}

CONTEXT_DESC: Dict[str, str] = {
    "guided": "in a one-on-one tutoring session",
    "collaborative": "in a small group learning setting",
    "autonomous": "studying independently",
}

# ---------------------------------------------------------------------------
# System prompt (constant across all cases)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a writer of realistic educational scenarios. "
    "Write a single paragraph of 80 to 120 words describing a concrete, "
    "realistic learning or tutoring situation. The scenario must feel natural "
    "and specific — like a real moment between a student and their learning "
    "environment. Do NOT use any pedagogical or theoretical terminology. "
    "Do NOT mention any framework, taxonomy, theory name, or technical "
    "education label. Just describe what is happening, what the student is "
    "doing, and what they are trying to achieve. Include a concrete learning "
    "goal. Write only the paragraph — no title, no bullet points, no labels."
)


def build_user_prompt(
    subject: str,
    bloom: str,
    knowledge_state: str,
    learning_stage: str,
    learning_context: str,
) -> str:
    """Compose the user-message prompt from translated variable descriptions."""
    intent = BLOOM_INTENT[bloom].format(subject=subject)
    knowledge = KNOWLEDGE_DESC[knowledge_state]
    stage = STAGE_DESC[learning_stage]
    context = CONTEXT_DESC[learning_context]

    return (
        f"Write a realistic scenario about a student who {intent}. "
        f"The student {knowledge} and {stage}. "
        f"The situation takes place {context}. "
        f"Describe what the student is doing, what challenge they face, "
        f"and what specific learning goal they are working toward."
    )


def build_messages(
    subject: str,
    bloom: str,
    knowledge_state: str,
    learning_stage: str,
    learning_context: str,
) -> list:
    """Return the full message list ready for the chat completions API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                subject, bloom, knowledge_state, learning_stage, learning_context
            ),
        },
    ]
