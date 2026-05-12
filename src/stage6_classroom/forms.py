"""Stage 6, Phase 2 — generate the per-student evaluation forms.

Per the proposal:
    Round 1 — Pre-intervention outputs. Students receive the model's
    original outputs for a selected set of cases. They evaluate each
    output using a 6-item rubric: five items corresponding to the five
    decision points plus one holistic item. All items use a 4-point
    scale (clearly inappropriate, somewhat inappropriate, somewhat
    appropriate, clearly appropriate). Students also provide a brief
    written justification for their lowest-rated item.

    Round 2 — Post-intervention outputs. Same rubric, same cases,
    randomized presentation. Students don't know which round corresponds
    to which model version.

    Assignment: Each student evaluates approximately 10 cases per round
    (5 cases in Round 1, same 5 in Round 2)... With 30 students × 10
    evaluations per round, each case receives 5 evaluations per round.

For ``S`` students and ``C`` selected cases, this module emits, for each
student, a JSON document containing all 2C evaluation items
(C pre-intervention + C post-intervention) in a shuffled, unlabelled
order. Each item carries the case narrative + the model output, and
the same 6-item rubric. The internal mapping (which item is pre/post)
is written to a separate ``assignment_key.json`` not given to students.

Inputs:
    data/stage6/phase2_cases.json        — output of case_selection
    data/stage4/gemini_responses.json    — pre-intervention outputs
    data/stage5/post_intervention_{...}.json — post-intervention outputs

Output:
    data/stage6/forms/student_{NN}.json  — per-student form
    data/stage6/forms/assignment_key.json — pre/post key per item
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest

logger = logging.getLogger("pipeline")


RUBRIC_ITEMS: List[Dict[str, str]] = [
    {"id": "content_level",       "prompt": "Given this learner's situation, how appropriate is the model's choice of content level?"},
    {"id": "student_task",        "prompt": "Given this learner's situation, how appropriate is the model's choice of student task?"},
    {"id": "tutor_role",          "prompt": "Given this learner's situation, how appropriate is the model's choice of tutor role?"},
    {"id": "student_engagement",  "prompt": "Given this learner's situation, how appropriate is the model's choice of engagement mode?"},
    {"id": "disciplinary_method", "prompt": "Given this learner's situation, how appropriate is the model's choice of disciplinary method?"},
    {"id": "holistic",            "prompt": "Overall, how well does this instructional plan serve this learner's needs?"},
]

LIKERT_SCALE: List[Dict[str, Any]] = [
    {"value": 1, "label": "clearly inappropriate"},
    {"value": 2, "label": "somewhat inappropriate"},
    {"value": 3, "label": "somewhat appropriate"},
    {"value": 4, "label": "clearly appropriate"},
]

JUSTIFICATION_PROMPT = (
    "Briefly explain (1–2 sentences) why you gave your lowest rating its score."
)


def _load_pre_outputs(stage4_dir: Path) -> Dict[str, Dict[str, Any]]:
    json_path = stage4_dir / "gemini_responses.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found.")
    records = read_json(json_path)
    return {r["prompt_id"]: r for r in records}


def _load_post_outputs(stage5_dir: Path, target: str, rank: int) -> Dict[str, Dict[str, Any]]:
    fname = (
        f"post_intervention_r{rank}.json"
        if target == "train"
        else f"post_intervention_{target}_r{rank}.json"
    )
    payload = read_json(stage5_dir / fname)
    return {r["prompt_id"]: r for r in payload["results"]}


def _load_narratives(stage4_dir: Path) -> Dict[str, str]:
    json_path = stage4_dir / "gemini_responses.json"
    records = read_json(json_path)
    return {r["prompt_id"]: r.get("narrative", "") for r in records}


def _assign_cases_to_students(
    case_ids: List[str],
    *,
    n_students: int,
    cases_per_student: int,
    raters_per_case: int,
    seed: int,
) -> List[List[str]]:
    """Round-robin assign cases to students so every case is rated by
    ``raters_per_case`` students and every student rates
    ``cases_per_student`` cases.

    Requires ``n_students * cases_per_student == len(case_ids) * raters_per_case``.
    """
    total_seats = n_students * cases_per_student
    expected = len(case_ids) * raters_per_case
    if total_seats != expected:
        raise ValueError(
            f"Assignment shape mismatch: {n_students}*{cases_per_student}={total_seats} "
            f"≠ {len(case_ids)}*{raters_per_case}={expected}"
        )

    rng = random.Random(seed)
    pool = case_ids * raters_per_case
    rng.shuffle(pool)
    assignments: List[List[str]] = []
    for i in range(n_students):
        chunk = pool[i * cases_per_student : (i + 1) * cases_per_student]
        # Guarantee no duplicate cases inside a single student's set.
        # If a duplicate slipped in, swap it with someone else's surplus.
        seen = set()
        for j, pid in enumerate(chunk):
            attempt = 0
            while pid in seen and attempt < 10 * n_students:
                # Try swapping with another student's slot.
                other = rng.randrange(n_students)
                other_chunk = assignments[other] if other < len(assignments) else None
                if other_chunk is not None:
                    for k, other_pid in enumerate(other_chunk):
                        if other_pid not in seen and pid not in other_chunk:
                            chunk[j], other_chunk[k] = other_chunk[k], chunk[j]
                            pid = chunk[j]
                            break
                attempt += 1
            seen.add(pid)
        assignments.append(chunk)
    return assignments


def run_build_phase2_forms(
    cfg: PipelineConfig,
    *,
    n_students: int = 30,
    cases_per_student: int = 5,
    raters_per_case: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate one form per student plus the assignment key.

    Each form contains the student's ``cases_per_student`` cases × 2
    rounds (pre + post) = ``2 × cases_per_student`` shuffled items.
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = DATA_DIR / "stage5"
    stage6_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage6")
    cases_path = stage6_dir / "phase2_cases.json"
    if not cases_path.exists():
        raise FileNotFoundError(
            f"{cases_path} not found. Run `run-stage6-select-cases` first."
        )
    cases_meta = read_json(cases_path)
    selected = cases_meta["cases"]
    rank = cases_meta["rank"]
    target = cases_meta["target"]

    case_ids = [c["prompt_id"] for c in selected]

    pre_records = _load_pre_outputs(cfg.stage4_dir)
    post_records = _load_post_outputs(stage5_dir, target=target, rank=rank)
    narratives = _load_narratives(cfg.stage4_dir)

    # Verify every selected case has both pre and post outputs.
    missing = [pid for pid in case_ids if pid not in pre_records or pid not in post_records]
    if missing:
        raise FileNotFoundError(
            f"Missing pre/post outputs for {len(missing)} cases: {missing[:5]}..."
        )

    assignments = _assign_cases_to_students(
        case_ids,
        n_students=n_students,
        cases_per_student=cases_per_student,
        raters_per_case=raters_per_case,
        seed=seed,
    )

    forms_dir = stage6_dir / "forms"
    forms_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    assignment_key: List[Dict[str, Any]] = []

    for student_idx, assigned_cases in enumerate(assignments, start=1):
        items: List[Dict[str, Any]] = []
        for pid in assigned_cases:
            for variant in ("pre", "post"):
                model_record = pre_records[pid] if variant == "pre" else post_records[pid]
                model_response = (
                    model_record.get("response")
                    or model_record.get("post_response")
                    or model_record.get("response_data")
                    or {}
                )
                item_uid = f"S{student_idx:02d}_{pid}_{variant}_{rng.randrange(10**6):06d}"
                items.append({
                    "item_uid": item_uid,
                    "prompt_id": pid,
                    "narrative": narratives[pid],
                    "model_output": model_response,
                    "rubric": RUBRIC_ITEMS,
                    "scale": LIKERT_SCALE,
                    "justification_prompt": JUSTIFICATION_PROMPT,
                    # Blinded view — students never see ``variant``.
                })
                assignment_key.append({
                    "item_uid": item_uid,
                    "student_id": f"S{student_idx:02d}",
                    "prompt_id": pid,
                    "variant": variant,
                    "rank": rank,
                    "target": target,
                })

        rng.shuffle(items)

        form = {
            "student_id": f"S{student_idx:02d}",
            "n_items": len(items),
            "instructions": (
                "You will see model-generated instructional plans for several learning "
                "scenarios. For each plan, rate the appropriateness of each of the five "
                "pedagogical decisions and the plan as a whole. For your lowest-rated "
                "item per plan, write a 1–2 sentence justification."
            ),
            "items": items,
        }
        write_json(form, forms_dir / f"student_{student_idx:02d}.json")

    write_json(assignment_key, stage6_dir / "assignment_key.json")

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    manifest = build_manifest(
        stage="stage6_phase2_forms",
        params={
            "n_students": n_students,
            "cases_per_student": cases_per_student,
            "raters_per_case": raters_per_case,
            "seed": seed,
            "rank": rank,
            "target": target,
            "input_cases": str(cases_path),
            "output_dir": str(forms_dir),
        },
        row_count=n_students,
        extra={
            "n_items_total": len(assignment_key),
            "n_unique_cases": len(case_ids),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage6_dir / "manifest_phase2_forms.json")

    logger.info(
        "Phase 2 forms: %d students × %d cases × 2 rounds = %d items → %s",
        n_students, cases_per_student, len(assignment_key), forms_dir,
    )
    return {
        "n_students": n_students,
        "n_items_total": len(assignment_key),
        "forms_dir": forms_dir,
        "assignment_key_path": stage6_dir / "assignment_key.json",
    }
