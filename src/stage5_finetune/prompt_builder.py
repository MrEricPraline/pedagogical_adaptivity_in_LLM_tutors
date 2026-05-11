"""Prompt + response templates shared by the corrective dataset, the
Tinker training loop, and the post-fine-tune sampling pass.

Keeping these strings in one place guarantees that pre-/post-intervention
queries use identical formatting (otherwise the comparison of pre/post
PAI deltas would be confounded by prompt drift).
"""

from __future__ import annotations

from typing import Dict, List, Mapping

from src.stage4_query.prompt_builder import (
    INSTRUCTIONS_BLOCK,
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)

__all__ = [
    "SYSTEM_INSTRUCTION",
    "INSTRUCTIONS_BLOCK",
    "build_user_prompt",
    "OPTION_DESCRIPTIONS",
    "describe_selection",
]

# ---------------------------------------------------------------------------
# Per-(decision-point, selection) description variants
#
# Five variants per cell so the 5 activities of one case can use different
# phrasings while still committing to the pedagogically optimal selection.
# This avoids training the model to memorise five identical strings (which
# would push the LoRA toward repeating the same sentence verbatim) while
# preserving the correct selection signal.
# ---------------------------------------------------------------------------

OPTION_DESCRIPTIONS: Dict[str, Dict[str, List[str]]] = {
    "content_level": {
        "a": [
            "Introduce the foundational concept with explicit definitions and a simple, accessible example.",
            "Lay out the basic idea using clear definitions and an everyday example to anchor understanding.",
            "Walk the learner through the concept from scratch with plain language and one or two clear examples.",
            "Frame the topic at an introductory level with definitions first, then a concrete illustration.",
            "Open with the core definition and a single, vivid example before introducing any nuance.",
        ],
        "b": [
            "Present the material at moderate complexity in an organized sequence that builds structured understanding.",
            "Offer a structured explanation that connects the parts of the topic in a logical, accessible order.",
            "Move beyond definitions into an organized account that links the main components of the concept.",
            "Build understanding step by step, scaffolding each piece onto the previous one with moderate detail.",
            "Use a structured walkthrough that surfaces relationships between sub-concepts at a manageable pace.",
        ],
        "c": [
            "Explore the underlying relationships, mechanisms, and patterns to deepen analytical understanding.",
            "Push beyond surface description into the structural relationships that make the topic work.",
            "Examine the mechanisms and patterns that connect the concept to other ideas in the field.",
            "Surface the analytical core: why the phenomenon behaves this way and what governs it.",
            "Analyze the topic in terms of its constitutive relationships and the patterns they produce.",
        ],
        "d": [
            "Confront competing perspectives, edge cases, and demands for sophisticated reasoning.",
            "Challenge the learner with contested interpretations, anomalies, and high-level reasoning tasks.",
            "Engage the topic critically: weigh competing accounts, surface tensions, and require justified positions.",
            "Bring forward edge cases and rival frameworks that demand careful, advanced reasoning.",
            "Treat the topic at expert depth, explicitly comparing perspectives and probing limit cases.",
        ],
    },
    "student_task": {
        "a": [
            "Build initial understanding through analogies, worked examples, or short explanatory narratives.",
            "Construct a first mental model using accessible analogies and a worked example.",
            "Form a representation of the idea by exploring concrete examples and intuitive analogies.",
            "Use narratives, metaphors, and worked examples to make the new concept feel familiar.",
            "Anchor the concept in the learner's experience through a vivid analogy or example-driven story.",
        ],
        "b": [
            "Practice the procedure repeatedly with structured corrective feedback to build fluency.",
            "Drill the skill through iterative cycles, each closed by targeted feedback.",
            "Engage in deliberate practice on the procedure with prompt, specific corrections.",
            "Repeat the operation across varied items, receiving feedback after each attempt.",
            "Build fluency by applying the procedure many times under close formative feedback.",
        ],
        "c": [
            "Apply the knowledge to a situated problem that requires adapting it to a new context.",
            "Bridge from textbook procedure to a realistic problem embedded in a meaningful context.",
            "Transfer the concept to an authentic scenario that demands judgment about how to apply it.",
            "Use the knowledge in a contextualized task where the right move is not pre-specified.",
            "Connect the abstract concept to a concrete situated case that requires adaptation.",
        ],
        "d": [
            "Integrate multiple concepts into a coherent expert-level framework.",
            "Synthesize several elements of the domain into one unified explanation or model.",
            "Connect distinct ideas into a single integrative argument or design.",
            "Pull together the strands of the topic into a coherent, expert-style synthesis.",
            "Produce a unifying framework that ties multiple concepts into one explanatory whole.",
        ],
    },
    "tutor_role": {
        "a": [
            "Provide explicit step-by-step instruction, monitor closely, and give direct corrective feedback.",
            "Lead the learner with clear instructions and immediate, specific corrections.",
            "Stay in directive mode: model the move, then have the learner mirror it under close watch.",
            "Walk the learner through the procedure with explicit guidance and frequent corrections.",
            "Take a hands-on instructional role with constant supervision and direct feedback.",
        ],
        "b": [
            "Offer targeted support on specific points while deferring to the primary instructor.",
            "Provide complementary scaffolding on the points where the learner stumbles, leaving the main thread to the teacher.",
            "Step in only on focused difficulties; otherwise let the primary instruction proceed.",
            "Act as a secondary support, intervening on narrow gaps without taking over.",
            "Supplement the primary instructor with focused micro-explanations on demand.",
        ],
        "c": [
            "Pose guiding questions, prompt reflection, and facilitate discovery without directing.",
            "Coach the learner with open questions that surface their reasoning rather than supplying answers.",
            "Use Socratic prompts to nudge thinking and let the learner formulate the next move.",
            "Facilitate by asking; resist telling. Push the learner to articulate and justify.",
            "Guide through carefully placed questions that prompt reflection and self-correction.",
        ],
        "d": [
            "Provide resources and challenges while giving the learner space to lead their own learning.",
            "Stay autonomy-supportive: hand over the steering wheel and respond when invited.",
            "Curate materials and high-quality challenges, then step back and let the learner drive.",
            "Be available with resources but let the learner set pace and direction.",
            "Support without intervening: provide tools and trust the learner to lead.",
        ],
    },
    "student_engagement": {
        "a": [
            "Attend to presented information through focused reading, listening, or observation.",
            "Engage receptively: read carefully, listen attentively, or watch a worked demonstration.",
            "Take in the material through focused observation before producing anything.",
            "Absorb the explanation through close reading or active listening.",
            "Engage by attending to a clear presentation of the material.",
        ],
        "b": [
            "Work with provided materials through sorting, labeling, annotation, or structured exercises.",
            "Manipulate the given items: classify, order, annotate, or fill in a structured worksheet.",
            "Engage hands-on with structured material — labelling, matching, or organizing pieces.",
            "Work the material physically or visually: sort, label, drill, or annotate.",
            "Interact with the content through structured manipulation rather than original production.",
        ],
        "c": [
            "Produce original outputs — explanations, designs, solutions — that demonstrate understanding.",
            "Generate something new: an explanation, an artifact, or a worked solution.",
            "Author original work that makes the learner's understanding visible.",
            "Produce an explanation, artifact, or design that goes beyond reproduction.",
            "Create an original output that the tutor can inspect for understanding.",
        ],
        "d": [
            "Co-construct understanding through dialogue, discussion, and collaborative knowledge-building.",
            "Engage collaboratively, building meaning with peers through active discussion.",
            "Work in dialogue with others: explain, challenge, and refine ideas together.",
            "Participate in collaborative knowledge-building with peers as the primary medium.",
            "Construct understanding jointly through structured peer exchange and dialogue.",
        ],
    },
    "disciplinary_method": {
        "a": [
            "Apply step-by-step procedures, formal rules, or algorithmic methods native to the discipline.",
            "Use the discipline's procedural-algorithmic toolkit: rules, formulas, and ordered steps.",
            "Lean on formal procedures and rule application as the disciplinary method.",
            "Engage the topic procedurally: identify the rule, apply it cleanly, verify the result.",
            "Treat the task as a procedural exercise grounded in the discipline's formal methods.",
        ],
        "b": [
            "Use observation, hypothesis testing, and evidence-based reasoning to investigate the topic.",
            "Adopt an empirical posture: observe, hypothesize, gather evidence, revise.",
            "Investigate the question through observation and disciplined evidence-weighing.",
            "Frame the activity as empirical inquiry — propose, test, and revise based on data.",
            "Engage the topic by gathering evidence and reasoning from it carefully.",
        ],
        "c": [
            "Engage in textual analysis, perspective-taking, and interpretive reasoning about the subject.",
            "Approach the topic interpretively: read closely, weigh perspectives, argue from evidence.",
            "Use hermeneutic methods — close reading, perspective-taking, interpretive argument.",
            "Frame the work as interpretive: analyze the text or case and argue a defensible reading.",
            "Apply interpretive-argumentative reasoning to surface meaning and defend a position.",
        ],
        "d": [
            "Produce original work through open-ended design, prototyping, or creative iteration.",
            "Use a design-creative method: prototype, test, and iterate toward an original outcome.",
            "Treat the task as design work — open-ended production with iterative refinement.",
            "Engage creatively: produce something new, then refine it through iterative cycles.",
            "Use open-ended design and prototyping as the disciplinary method.",
        ],
    },
}


def describe_selection(dp: str, selection: str, activity_index: int) -> str:
    """Return one of the five variant descriptions for (dp, selection),
    rotating across activities so the 5 activities of one case do not
    share an identical string for the same DP."""
    variants = OPTION_DESCRIPTIONS[dp][selection]
    return variants[activity_index % len(variants)]


def build_optimal_response(
    optimal_selections: Mapping[str, str],
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:
    """Build the 5-activity assistant response that uses the optimal
    selection for every DP and rotates through description variants."""
    activities: List[Dict[str, Dict[str, str]]] = []
    for i in range(5):
        activity: Dict[str, Dict[str, str]] = {"activity_number": i + 1}  # type: ignore[assignment]
        for dp, sel in optimal_selections.items():
            activity[dp] = {
                "selection": sel,
                "description": describe_selection(dp, sel, i),
            }
        activities.append(activity)
    return {"activities": activities}
