"""Build the corrective LoRA training set from the lowest-PAI cases.

For each of the bottom-N cases (default N=30) by ``prompt_PAI``:

* find the pedagogically optimal selection per DP given that case's
  learner conditions (the selection that maximises the PAI sub-matrix
  mean — see :func:`src.stage5_scoring.scorer.find_optimal_selections`),
* compose a 3-message chat (system + user prompt + assistant response),
  where the assistant's response designs 5 activities using the optimal
  selections and rotated description variants.

The resulting dataset is written to ``data/stage5/corrective_training_data.json``
and is the input to :func:`src.stage5_finetune.tinker_train.train`.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import read_json, read_jsonl, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest
from src.stage5_finetune.prompt_builder import (
    SYSTEM_INSTRUCTION,
    build_optimal_response,
    build_user_prompt,
)
from src.stage5_scoring.matrices import DECISION_POINTS
from src.stage5_scoring.scorer import (
    CASE_METADATA_KEYS,
    extract_meta,
    find_optimal_selections,
)

logger = logging.getLogger("pipeline")


def _load_scored(stage5_dir: Path) -> List[Dict[str, Any]]:
    json_path = stage5_dir / "scored_dataset.json"
    if json_path.exists():
        return read_json(json_path)
    raise FileNotFoundError(
        "scored_dataset.json not found. Run `python -m src.pipeline.cli "
        "run-stage5-score` first."
    )


def _load_narratives(stage4_dir: Path) -> Dict[str, str]:
    json_path = stage4_dir / "gemini_responses.json"
    jsonl_path = stage4_dir / "gemini_responses.jsonl"
    if json_path.exists():
        records = read_json(json_path)
    elif jsonl_path.exists():
        records = read_jsonl(jsonl_path)
    else:
        raise FileNotFoundError(
            f"Stage 4 output not found in {stage4_dir} (need gemini_responses.json or .jsonl)."
        )
    return {r["prompt_id"]: r.get("narrative", "") for r in records if r.get("prompt_id")}


def build_corrective_example(
    case: Dict[str, Any],
    narrative: str,
) -> Dict[str, Any]:
    """Compose one corrective training example as a chat-format triple."""
    meta = extract_meta(case)
    optimal = find_optimal_selections(meta)
    optimal_selections = {dp: info["selection"] for dp, info in optimal.items()}

    response_payload = build_optimal_response(optimal_selections)
    assistant_content = json.dumps(response_payload, indent=2, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_prompt(narrative)},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "prompt_id": case["prompt_id"],
        "messages": messages,
        "optimal_selections": optimal_selections,
        "optimal_dp_scores": {dp: round(info["score"], 4) for dp, info in optimal.items()},
        "pre_intervention_PAI": case["prompt_PAI"],
        "pre_intervention_dp_means": case.get("dp_means", {}),
        "case_metadata": {k: case[k] for k in CASE_METADATA_KEYS if k in case},
    }


def run_build_corrective_data(
    cfg: PipelineConfig,
    *,
    n: int = 30,
) -> Dict[str, Any]:
    """Materialise the corrective dataset for the bottom-`n` cases by PAI.

    This is the *global* (un-stratified) mode. For the proposal-aligned
    stratified mode (50-100 examples per weak cell), use
    :func:`run_build_corrective_data_stratified`.
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    stage5_dir.mkdir(parents=True, exist_ok=True)

    scored = _load_scored(stage5_dir)
    if len(scored) < n:
        logger.warning(
            "Stage 5 corrective: only %d scored cases available (n=%d requested)",
            len(scored), n,
        )
        n = len(scored)

    narratives = _load_narratives(cfg.stage4_dir)

    low_n = scored[:n]  # already sorted ascending by prompt_PAI
    examples: List[Dict[str, Any]] = []
    missing_narratives: List[str] = []

    for case in low_n:
        pid = case["prompt_id"]
        narrative = narratives.get(pid)
        if not narrative:
            missing_narratives.append(pid)
            continue
        examples.append(build_corrective_example(case, narrative))

    if missing_narratives:
        logger.warning(
            "Stage 5 corrective: %d cases skipped (no narrative): %s",
            len(missing_narratives), missing_narratives,
        )

    out_path = stage5_dir / "corrective_training_data.json"
    write_json(examples, out_path)

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    manifest = build_manifest(
        stage="stage5_corrective_data",
        params={
            "n": n,
            "input_scored": "data/stage5/scored_dataset.json",
            "output": "data/stage5/corrective_training_data.json",
        },
        row_count=len(examples),
        extra={
            "requested_n": n,
            "produced": len(examples),
            "missing_narratives": missing_narratives,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
            "pre_PAI_range": {
                "min": min((e["pre_intervention_PAI"] for e in examples), default=None),
                "max": max((e["pre_intervention_PAI"] for e in examples), default=None),
            },
        },
    )
    # Write manifest to a dedicated filename so we don't overwrite the
    # scoring manifest already living in stage5/.
    write_json(manifest, stage5_dir / "manifest_corrective.json")

    logger.info(
        "Stage 5 corrective: built %d examples (PAI range [%.4f, %.4f]) → %s",
        len(examples),
        min((e["pre_intervention_PAI"] for e in examples), default=0.0),
        max((e["pre_intervention_PAI"] for e in examples), default=0.0),
        out_path,
    )

    return {"examples": examples, "output": out_path}


# ---------------------------------------------------------------------------
# Stratified mode — proposal-aligned: 50-100 examples per weak cell
# ---------------------------------------------------------------------------

def _load_weak_cells(stage5_dir: Path) -> List[Dict[str, Any]]:
    path = stage5_dir / "weak_cells.json"
    if not path.exists():
        raise FileNotFoundError(
            "weak_cells.json not found. Run `run-stage5-weak-cells` first."
        )
    return read_json(path)["weak_cells"]


def run_build_corrective_data_stratified(
    cfg: PipelineConfig,
    *,
    per_cell: int = 75,
    target_dp_only: bool = False,
) -> Dict[str, Any]:
    """Materialise the corrective dataset stratified by weak cell.

    For every weak cell, sample up to ``per_cell`` cases that fall in the
    cell's condition combination — preferring those with the lowest
    per-case score on that DP (the most-failed cases inside the cell).

    A single case can serve multiple weak cells (its conditions overlap
    several DPs), but each case is emitted only once in the output; the
    "served cells" list is preserved as metadata.

    ``target_dp_only=False`` (default) — the assistant target uses the
    full optimal selection across all 5 DPs (the same shape used in the
    global mode). This is the right target for the unified rank sweep.

    ``target_dp_only=True`` — the assistant target keeps the optimal
    selection for the weak cell's DP and writes the model's *current*
    (possibly suboptimal) selection for the other 4 DPs. Used by the
    per-DP isolated LoRAs so each adapter only learns to correct its
    own DP without forcing the others.
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    stage5_dir.mkdir(parents=True, exist_ok=True)

    scored = _load_scored(stage5_dir)
    scored_by_id = {c["prompt_id"]: c for c in scored}
    narratives = _load_narratives(cfg.stage4_dir)
    weak_cells = _load_weak_cells(stage5_dir)

    examples: List[Dict[str, Any]] = []
    # In dedupe mode (target_dp_only=False), we key by prompt_id so the
    # same case never appears twice with the same full-optimal target.
    # In per-DP mode, we key by (prompt_id, target_dp) so a case that
    # serves two weak cells with different DPs produces two distinct
    # training examples — one per DP it is being trained to correct.
    selected: Dict[Any, Dict[str, Any]] = {}
    cell_assignments: Dict[str, List[str]] = {}
    missing_narratives: List[str] = []

    for cell in weak_cells:
        target_dp = cell["decision_point"]
        cell_id = cell["cell_id"]

        # Order candidates ascending by their per-case DP score so we
        # take the worst offenders first inside this cell.
        candidates = sorted(
            (scored_by_id[pid] for pid in cell["case_ids"] if pid in scored_by_id),
            key=lambda c: c["dp_means"][target_dp],
        )

        kept_for_cell: List[str] = []
        for case in candidates:
            if len(kept_for_cell) >= per_cell:
                break
            pid = case["prompt_id"]
            narrative = narratives.get(pid)
            if not narrative:
                missing_narratives.append(pid)
                continue

            dedupe_key = (pid, target_dp) if target_dp_only else pid
            if dedupe_key in selected:
                # Already emitted for this (case, target) — record that
                # this cell also served the same example.
                if cell_id not in selected[dedupe_key]["served_cells"]:
                    selected[dedupe_key]["served_cells"].append(cell_id)
                if target_dp not in selected[dedupe_key]["target_dps"]:
                    selected[dedupe_key]["target_dps"].append(target_dp)
                kept_for_cell.append(pid)
                continue

            example = build_corrective_example(case, narrative)
            example["served_cells"] = [cell_id]
            example["target_dps"] = [target_dp]

            if target_dp_only:
                example = _retarget_to_single_dp(
                    example=example,
                    case=case,
                    narrative=narrative,
                    target_dp=target_dp,
                )

            examples.append(example)
            selected[dedupe_key] = example
            kept_for_cell.append(pid)

        cell_assignments[cell_id] = kept_for_cell
        logger.info(
            "Stage 5 corrective (stratified): cell=%s served %d cases (target=%s)",
            cell_id, len(kept_for_cell), target_dp,
        )

    # Drop duplicate prompt_ids from missing_narratives.
    missing_narratives = sorted(set(missing_narratives))
    if missing_narratives:
        logger.warning(
            "Stage 5 corrective (stratified): %d cases skipped (no narrative)",
            len(missing_narratives),
        )

    out_name = (
        "corrective_training_data_per_dp.json"
        if target_dp_only
        else "corrective_training_data_stratified.json"
    )
    out_path = stage5_dir / out_name
    write_json(examples, out_path)

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    manifest = build_manifest(
        stage="stage5_corrective_data_stratified",
        params={
            "per_cell": per_cell,
            "target_dp_only": target_dp_only,
            "input_scored": "data/stage5/scored_dataset.json",
            "input_weak_cells": "data/stage5/weak_cells.json",
            "output": f"data/stage5/{out_name}",
        },
        row_count=len(examples),
        extra={
            "per_cell_target": per_cell,
            "n_weak_cells": len(weak_cells),
            "n_unique_cases": len(examples),
            "missing_narratives": missing_narratives,
            "cell_assignments_size": {cid: len(ids) for cid, ids in cell_assignments.items()},
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    manifest_name = (
        "manifest_corrective_per_dp.json"
        if target_dp_only
        else "manifest_corrective_stratified.json"
    )
    write_json(manifest, stage5_dir / manifest_name)

    logger.info(
        "Stage 5 corrective (stratified): %d unique cases across %d weak cells "
        "(per_cell=%d, target_dp_only=%s) → %s",
        len(examples), len(weak_cells), per_cell, target_dp_only, out_path,
    )
    return {"examples": examples, "output": out_path, "cell_assignments": cell_assignments}


def _retarget_to_single_dp(
    *,
    example: Dict[str, Any],
    case: Dict[str, Any],
    narrative: str,
    target_dp: str,
) -> Dict[str, Any]:
    """Rewrite the assistant target so only ``target_dp`` is corrected.

    The other 4 DPs reuse the model's actual Stage-4 selections (from
    ``case["selections"]``) so the LoRA isn't forced to memorize them.
    The user + system messages stay identical.
    """
    from src.stage5_finetune.prompt_builder import (
        OPTION_DESCRIPTIONS,
        describe_selection,
    )

    meta = extract_meta(case)
    optimal = find_optimal_selections(meta)

    # ``case["selections"]`` shape: list of 5 activities, each with
    # ``{dp: "a"|"b"|"c"|"d"}``. Mirror it back, replacing only target_dp
    # with the optimal selection.
    activities_out: List[Dict[str, Any]] = []
    for idx, current_sel in enumerate(case["selections"], start=1):
        activity_block: Dict[str, Any] = {"activity_number": idx}
        for dp in DECISION_POINTS:
            if dp == target_dp:
                sel_letter = optimal[dp]["selection"]
            else:
                sel_letter = current_sel.get(dp, optimal[dp]["selection"])
            activity_block[dp] = {
                "selection": sel_letter,
                "description": describe_selection(dp, sel_letter, activity_index=idx - 1),
            }
        activities_out.append(activity_block)

    import json as _json
    new_assistant = _json.dumps({"activities": activities_out}, indent=2, ensure_ascii=False)
    example = dict(example)
    example["messages"] = [
        example["messages"][0],          # system
        example["messages"][1],          # user
        {"role": "assistant", "content": new_assistant},
    ]
    return example
