"""Stage 6, Phase 2 — select ~30 cases for the classroom evaluation.

Per the proposal:
    "Approximately 30 cases, selected to maximize the range of PAI deltas:
     10 cases where the model improved substantially (large positive PAI
     delta), 10 where it improved modestly or not at all (small or zero
     delta), and 10 where it scored high at baseline and remained high
     (control — confirming that students rate consistently when quality
     doesn't change)."

Input: a post-intervention file (held-out or train) for the
chosen LoRA rank, plus the Experiment-1 scored dataset.

Output: ``data/stage6/phase2_cases.json`` with the 30 selected
prompt_ids labelled by stratum (large-improvement / modest / control).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest

logger = logging.getLogger("pipeline")


def _select_strata(
    rows: List[Dict[str, Any]],
    *,
    n_per_stratum: int,
    control_threshold: float,
    modest_threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Partition rows into the three strata required by the proposal.

    Rows are filtered to status=ok before being passed in.

    * **large** — top ``n_per_stratum`` by ``delta_PAI``.
    * **modest** — ``n_per_stratum`` rows with smallest ``|delta_PAI|``.
    * **control** — ``n_per_stratum`` rows where Gemini ``pre_PAI`` was
      already high (≥ ``control_threshold``) and the LoRA delta stayed
      small (``|delta_PAI| ≤ modest_threshold``).
    """
    by_delta_desc = sorted(rows, key=lambda r: r["delta_PAI"], reverse=True)
    large = by_delta_desc[:n_per_stratum]
    large_ids = {r["prompt_id"] for r in large}

    remaining = [r for r in rows if r["prompt_id"] not in large_ids]
    by_abs_delta = sorted(remaining, key=lambda r: abs(r["delta_PAI"]))
    modest = by_abs_delta[:n_per_stratum]
    modest_ids = {r["prompt_id"] for r in modest}

    remaining = [
        r for r in rows
        if r["prompt_id"] not in large_ids and r["prompt_id"] not in modest_ids
    ]
    candidates_control = [
        r for r in remaining
        if r["pre_PAI"] >= control_threshold and abs(r["delta_PAI"]) <= modest_threshold
    ]
    candidates_control.sort(key=lambda r: r["pre_PAI"], reverse=True)
    control = candidates_control[:n_per_stratum]

    return {"large_improvement": large, "modest_or_zero": modest, "control": control}


def run_select_phase2_cases(
    cfg: PipelineConfig,
    *,
    rank: int,
    target: str = "heldout",
    n_per_stratum: int = 10,
    control_threshold: float = 0.4,
    modest_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Pick 30 cases (3 strata × n_per_stratum) for the classroom rounds."""
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = DATA_DIR / "stage5"
    stage6_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage6")
    stage6_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"post_intervention_r{rank}.json"
        if target == "train"
        else f"post_intervention_{target}_r{rank}.json"
    )
    post_path = stage5_dir / fname
    if not post_path.exists():
        raise FileNotFoundError(
            f"{post_path} not found. Run `run-stage5-query --target {target} --rank {rank}` first."
        )

    payload = read_json(post_path)
    rows = [r for r in payload["results"] if r["status"] == "ok"]
    if len(rows) < 3 * n_per_stratum:
        logger.warning(
            "Phase 2 selection: only %d ok rows available; need %d for full strata.",
            len(rows), 3 * n_per_stratum,
        )

    strata = _select_strata(
        rows,
        n_per_stratum=n_per_stratum,
        control_threshold=control_threshold,
        modest_threshold=modest_threshold,
    )

    cases_out: List[Dict[str, Any]] = []
    for stratum_name, stratum_rows in strata.items():
        for r in stratum_rows:
            cases_out.append({
                "prompt_id": r["prompt_id"],
                "stratum": stratum_name,
                "rank": rank,
                "target": target,
                "pre_PAI": r["pre_PAI"],
                "post_PAI": r["post_PAI"],
                "delta_PAI": r["delta_PAI"],
            })

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    out = {
        "source_post_intervention": str(post_path),
        "rank": rank,
        "target": target,
        "n_per_stratum": n_per_stratum,
        "control_threshold": control_threshold,
        "modest_threshold": modest_threshold,
        "stratum_sizes": {k: len(v) for k, v in strata.items()},
        "cases": cases_out,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
    }
    out_path = stage6_dir / "phase2_cases.json"
    write_json(out, out_path)

    manifest = build_manifest(
        stage="stage6_phase2_case_selection",
        params={
            "rank": rank,
            "target": target,
            "n_per_stratum": n_per_stratum,
            "control_threshold": control_threshold,
            "modest_threshold": modest_threshold,
            "input": str(post_path),
            "output": str(out_path),
        },
        row_count=len(cases_out),
        extra={
            "stratum_sizes": out["stratum_sizes"],
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage6_dir / "manifest_phase2_cases.json")

    logger.info(
        "Phase 2 selection: %d cases (large=%d, modest=%d, control=%d) → %s",
        len(cases_out),
        out["stratum_sizes"]["large_improvement"],
        out["stratum_sizes"]["modest_or_zero"],
        out["stratum_sizes"]["control"],
        out_path,
    )
    return out
