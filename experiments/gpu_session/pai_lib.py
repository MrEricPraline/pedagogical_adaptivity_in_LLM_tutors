"""
pai_lib.py — shared prompt construction, PAI scoring, and sequence-level measures.

Used by both run_oos_gpu.py (Step 1) and run_base_control.py (Step 2), so that
every arm is prompted and scored identically. Do not edit the prompt: it is
reproduced verbatim from the original pipeline (corrective_training_data.json),
and any change would break comparability with the existing arms.
"""
import json
from collections import Counter
from math import log2
from statistics import mean

# ────────────────────────────── PROMPT ──────────────────────────────

SYSTEM_PROMPT = (
    "You are an experienced tutor designing personalized instruction for individual "
    "learners. You have deep knowledge of pedagogy and instructional design. Given a "
    "specific learner and their learning situation, you design a sequence of learning "
    "activities that best serve that learner's needs and goals."
)

TASK_BLOCK = """Design a sequence of 5 learning activities to help this student achieve the learning goal. For each activity, make the following selections and provide a brief description (1-2 sentences) explaining your choice.

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

Return your response in the specified JSON format with exactly 5 activities. For each activity, provide the selection letter a, b, c, or d and a brief description of your choice.
"""


def build_messages(narrative: str):
    """Exactly the message list used to train and query every arm."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Learner case:\n\n{narrative}\n\n{TASK_BLOCK}"},
    ]


# ───────────────────────── PAI SCORING MATRICES ─────────────────────────
# 11 sub-matrices. DP -> list of (matrix, conditioning-variable key).

M1A = {'a': {'RU': 1.0, 'AA': -0.5, 'EC': -1.0}, 'b': {'RU': 0.5, 'AA': 0.5, 'EC': -0.5},
       'c': {'RU': -0.5, 'AA': 1.0, 'EC': 0.5}, 'd': {'RU': -1.0, 'AA': 0.5, 'EC': 1.0}}
M1B = {'a': {'novice': 1.0, 'informed': 0.0, 'misinformed': -0.5}, 'b': {'novice': 0.5, 'informed': 1.0, 'misinformed': 0.0},
       'c': {'novice': -0.5, 'informed': 0.5, 'misinformed': 0.5}, 'd': {'novice': -1.0, 'informed': -0.5, 'misinformed': 1.0}}
M2A = {'a': {'conceptual_orientation': 1.0, 'skill_building': -0.5, 'competency_development': -0.5, 'comprehensive_mastery': -0.5},
       'b': {'conceptual_orientation': -0.5, 'skill_building': 1.0, 'competency_development': 0.0, 'comprehensive_mastery': -0.5},
       'c': {'conceptual_orientation': 0.0, 'skill_building': -0.5, 'competency_development': 1.0, 'comprehensive_mastery': 0.5},
       'd': {'conceptual_orientation': -0.5, 'skill_building': -0.5, 'competency_development': 0.5, 'comprehensive_mastery': 1.0}}
M2B = {'a': {'RU': 1.0, 'AA': 0.0, 'EC': -0.5}, 'b': {'RU': 0.5, 'AA': 0.5, 'EC': -0.5},
       'c': {'RU': -0.5, 'AA': 1.0, 'EC': 0.5}, 'd': {'RU': -1.0, 'AA': 0.5, 'EC': 1.0}}
M3A = {'a': {'guided': -0.5, 'collaborative': -1.0, 'autonomous': 1.0}, 'b': {'guided': 1.0, 'collaborative': 0.0, 'autonomous': 0.0},
       'c': {'guided': 0.5, 'collaborative': 1.0, 'autonomous': -0.5}, 'd': {'guided': -0.5, 'collaborative': 0.5, 'autonomous': 0.5}}
M3B = {'a': {'novice': 1.0, 'informed': 0.0, 'misinformed': 0.5}, 'b': {'novice': 0.5, 'informed': 0.5, 'misinformed': 0.0},
       'c': {'novice': -0.5, 'informed': 0.5, 'misinformed': 1.0}, 'd': {'novice': -1.0, 'informed': 1.0, 'misinformed': -0.5}}
M4A = {'a': {'guided': 0.0, 'collaborative': -1.0, 'autonomous': -0.5}, 'b': {'guided': 0.5, 'collaborative': -0.5, 'autonomous': 0.5},
       'c': {'guided': 0.5, 'collaborative': 0.0, 'autonomous': 1.0}, 'd': {'guided': 0.5, 'collaborative': 1.0, 'autonomous': -0.5}}
M4B = {'a': {'RU': 1.0, 'AA': -0.5, 'EC': -1.0}, 'b': {'RU': 0.5, 'AA': 0.5, 'EC': 0.0},
       'c': {'RU': 0.0, 'AA': 1.0, 'EC': 0.5}, 'd': {'RU': -0.5, 'AA': 0.5, 'EC': 1.0}}
M4C = {'a': {'conceptual_orientation': 0.5, 'skill_building': -0.5, 'competency_development': -0.5, 'comprehensive_mastery': -1.0},
       'b': {'conceptual_orientation': -0.5, 'skill_building': 1.0, 'competency_development': 0.0, 'comprehensive_mastery': -0.5},
       'c': {'conceptual_orientation': 0.5, 'skill_building': 0.5, 'competency_development': 1.0, 'comprehensive_mastery': 0.5},
       'd': {'conceptual_orientation': 0.0, 'skill_building': -0.5, 'competency_development': 0.5, 'comprehensive_mastery': 1.0}}
M5A = {'a': {'formal': 1.0, 'natural': -0.5, 'humanistic': -1.0, 'applied': 0.0}, 'b': {'formal': 0.0, 'natural': 1.0, 'humanistic': 0.0, 'applied': 0.5},
       'c': {'formal': -0.5, 'natural': 0.0, 'humanistic': 1.0, 'applied': 0.5}, 'd': {'formal': 0.5, 'natural': 0.5, 'humanistic': 0.5, 'applied': 0.5}}
M5B = {'a': {'conceptual_orientation': -0.5, 'skill_building': 1.0, 'competency_development': 0.5, 'comprehensive_mastery': 0.0},
       'b': {'conceptual_orientation': 1.0, 'skill_building': 0.5, 'competency_development': 0.0, 'comprehensive_mastery': 0.0},
       'c': {'conceptual_orientation': 0.5, 'skill_building': -0.5, 'competency_development': 0.5, 'comprehensive_mastery': 0.5},
       'd': {'conceptual_orientation': 0.0, 'skill_building': -0.5, 'competency_development': 0.5, 'comprehensive_mastery': 1.0}}

DP_SPEC = {
    'content_level':        [(M1A, 'bloom_band'), (M1B, 'knowledge_state')],
    'student_task':         [(M2A, 'learning_stage'), (M2B, 'bloom_band')],
    'tutor_role':           [(M3A, 'learning_context'), (M3B, 'knowledge_state')],
    'student_engagement':   [(M4A, 'learning_context'), (M4B, 'bloom_band'), (M4C, 'learning_stage')],
    'disciplinary_method':  [(M5A, 'subject_family'), (M5B, 'learning_stage')],
}
DPS = list(DP_SPEC)


def score_response(response: dict, case: dict):
    """response: {'activities':[{dp:{'selection':...}}...]}  case: has the 5 condition keys.
    Returns (prompt_PAI, dp_means, activity_scores)."""
    acts = response.get('activities', [])
    per_dp = {dp: [] for dp in DPS}
    act_scores = []
    for a in acts:
        dp_vals = {}
        for dp, mats in DP_SPEC.items():
            sel = a[dp]['selection'].strip().lower()
            vals = [m[sel][case[cond]] for m, cond in mats]
            dp_vals[dp] = mean(vals)
            per_dp[dp].append(dp_vals[dp])
        act_scores.append(mean(dp_vals.values()))
    dp_means = {dp: round(mean(v), 4) for dp, v in per_dp.items() if v}
    pai = round(mean(act_scores), 4) if act_scores else None
    return pai, dp_means, [round(s, 4) for s in act_scores]


def diversity(response: dict):
    """Sequence-level measures. Returns (repetition_rate, selection_entropy, fully_monotone)."""
    acts = response.get('activities', [])
    if len(acts) < 2:
        return None, None, None
    reps, ents = [], []
    for dp in DPS:
        sels = [a[dp]['selection'] for a in acts if dp in a]
        if not sels:
            continue
        c = Counter(sels)
        n = len(sels)
        reps.append(max(c.values()) / n)
        ents.append(-sum((v / n) * log2(v / n) for v in c.values()))
    rr, se = mean(reps), mean(ents)
    return round(rr, 4), round(se, 4), rr == 1.0


def parse_json_response(text: str):
    """Extract the activities JSON from a model completion; tolerant of code fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(t[start:end + 1])


def summarize(results):
    """Aggregate PAI + diversity across a list of scored results."""
    ok = [r for r in results if r.get('PAI') is not None]
    if not ok:
        return {}
    mono = [r for r in ok if r.get('fully_monotone')]
    return {
        'n': len(ok),
        'PAI_mean': round(mean(r['PAI'] for r in ok), 4),
        'repetition_mean': round(mean(r['repetition_rate'] for r in ok), 4),
        'entropy_mean': round(mean(r['selection_entropy'] for r in ok), 4),
        'pct_fully_monotone': round(100 * len(mono) / len(ok), 1),
        'dp_means': {dp: round(mean(r['dp_means'][dp] for r in ok if dp in r['dp_means']), 4) for dp in DPS},
    }
