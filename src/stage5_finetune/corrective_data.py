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
    """Materialise the corrective dataset for the bottom-`n` cases by PAI."""
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
