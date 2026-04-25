"""Prompt and structured-output schema for Stage 4 (Gemini 3.1 Pro Preview).

Given a learner case narrative, the model must return exactly 5 learning
activities, each annotated along five pedagogical dimensions.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# System instruction (constant across all Stage 4 cases)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are an experienced tutor designing personalized instruction for "
    "individual learners. You have deep knowledge of pedagogy and "
    "instructional design. Given a specific learner and their learning "
    "situation, you design a sequence of learning activities that best serve "
    "that learner's needs and goals."
)

# ---------------------------------------------------------------------------
# Fixed instructional block appended after the case narrative
# ---------------------------------------------------------------------------

INSTRUCTIONS_BLOCK = """Design a sequence of 5 learning activities to help this student achieve the learning goal. For each activity, make the following selections and provide a brief description (1-2 sentences) explaining your choice.

1. Content level — Select the approach that best fits this activity:
   (a) Foundational exposition: introduce basic concepts with explicit definitions and simple examples.
   (b) Structured explanation: build understanding through organized presentation with moderate complexity.
   (c) Analytical treatment: explore relationships, patterns, and underlying mechanisms.
   (d) Critical-advanced engagement: confront competing perspectives, edge cases, and require sophisticated reasoning.

2. Student task — Select the cognitive work the student will perform:
   (a) Representation building: construct initial understanding through analogies, examples, or explanatory narratives.
   (b) Iterative practice: repeatedly apply a procedure or skill with corrective feedback.
   (c) Contextual bridging: apply knowledge to situated problems in context.
   (d) Connective synthesis: integrate multiple elements into expert-level frameworks.

3. Tutor role — Select how you will interact with the student:
   (a) Directive guidance: provide step-by-step instruction, close monitoring, and explicit corrective feedback.
   (b) Complementary support: offer targeted assistance on specific points, deferring to a primary instructor.
   (c) Facilitative coaching: pose questions, prompt reflection, and guide without directing.
   (d) Autonomy-supportive: provide resources and challenge with minimal intervention, letting the learner lead.

4. Student engagement — Select how the student will engage:
   (a) Receptive engagement: attend to presented information through reading, listening, or observing.
   (b) Manipulative engagement: work with provided material through sorting, labeling, or drilling.
   (c) Generative engagement: produce original outputs such as explanations, designs, or syntheses.
   (d) Collaborative engagement: co-construct understanding through dialogue and peer exchange.

5. Disciplinary method — Select the approach to the subject matter:
   (a) Procedural-algorithmic: use step-by-step procedures, formal methods, or rule application.
   (b) Empirical-investigative: use observation, hypothesis testing, or evidence-based reasoning.
   (c) Interpretive-argumentative: use textual analysis, perspective-taking, or rhetorical reasoning.
   (d) Design-creative: use open-ended production, prototyping, or iterative creation.

Return your response in the specified JSON format with exactly 5 activities. For each activity, provide the selection letter a, b, c, or d and a brief description of your choice."""

DIMENSIONS = (
    "content_level",
    "student_task",
    "tutor_role",
    "student_engagement",
    "disciplinary_method",
)


def build_user_prompt(narrative: str) -> str:
    """Compose the full user prompt: case narrative + instructions block."""
    narrative = (narrative or "").strip()
    return f"Learner case:\n\n{narrative}\n\n{INSTRUCTIONS_BLOCK}"


# ---------------------------------------------------------------------------
# JSON schema for Gemini structured output
# ---------------------------------------------------------------------------

def _dimension_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selection": {"type": "string", "enum": ["a", "b", "c", "d"]},
            "description": {"type": "string"},
        },
        "required": ["selection", "description"],
        "propertyOrdering": ["selection", "description"],
    }


def build_response_schema() -> Dict[str, Any]:
    """Return an OpenAPI-style JSON schema enforcing exactly 5 activities."""
    activity = {
        "type": "object",
        "properties": {
            "activity_number": {"type": "integer", "minimum": 1, "maximum": 5},
            "content_level": _dimension_schema(),
            "student_task": _dimension_schema(),
            "tutor_role": _dimension_schema(),
            "student_engagement": _dimension_schema(),
            "disciplinary_method": _dimension_schema(),
        },
        "required": [
            "activity_number",
            "content_level",
            "student_task",
            "tutor_role",
            "student_engagement",
            "disciplinary_method",
        ],
        "propertyOrdering": [
            "activity_number",
            "content_level",
            "student_task",
            "tutor_role",
            "student_engagement",
            "disciplinary_method",
        ],
    }

    return {
        "type": "object",
        "properties": {
            "activities": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": activity,
            }
        },
        "required": ["activities"],
        "propertyOrdering": ["activities"],
    }
