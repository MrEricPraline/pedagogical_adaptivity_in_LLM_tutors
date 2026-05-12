"""Stage 6, Phase 2 — statistical analysis of student ratings.

Per the proposal's Statistical Analysis Plan:
    - Independent t-tests / Mann-Whitney U: model PAI vs student PAI per
      decision point and per condition (RQ1).
    - Paired t-test / Wilcoxon signed-rank: pre/post model PAI per
      corrected dimension (RQ2a).
    - Pearson or Spearman correlation: PAI delta vs student rating delta
      per decision point and per case (RQ2b-c).
    - Krippendorff's alpha or ICC: inter-rater reliability across 5
      students per case.
    - Effect sizes (Cohen's d or η²) for all significant comparisons.
    - Thematic analysis (Braun & Clarke, 2006) on written justifications.

This module covers the *quantitative* checks. Thematic analysis on the
written justifications is left for the analyst to do qualitatively.

Inputs:
    data/stage6/assignment_key.json     — item_uid → (student, pid, variant)
    data/stage6/ratings_raw.json        — student responses; expected schema:
        [{"item_uid": str, "ratings": {<rubric_id>: int1..4}, "justification": str},
         ...]
    data/stage6/phase2_cases.json       — selected cases + the automated PAI deltas

Output:
    data/stage6/phase2_analysis.json
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest

logger = logging.getLogger("pipeline")

RUBRIC_IDS = (
    "content_level",
    "student_task",
    "tutor_role",
    "student_engagement",
    "disciplinary_method",
    "holistic",
)


def _try_scipy():
    """Return scipy.stats if available, else None (degrade gracefully)."""
    try:
        from scipy import stats  # type: ignore
        return stats
    except ImportError:
        return None


def _wilcoxon(stats_mod, pre: List[float], post: List[float]) -> Dict[str, Any]:
    if stats_mod is None or len(pre) < 5:
        return {"statistic": None, "p_value": None, "n": len(pre), "note": "scipy_unavailable_or_n<5"}
    res = stats_mod.wilcoxon(pre, post)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": len(pre)}


def _correlation(stats_mod, xs: List[float], ys: List[float]) -> Dict[str, Any]:
    if stats_mod is None or len(xs) < 5:
        return {
            "pearson":  {"r": None, "p_value": None, "n": len(xs)},
            "spearman": {"rho": None, "p_value": None, "n": len(xs)},
            "note": "scipy_unavailable_or_n<5",
        }
    pr = stats_mod.pearsonr(xs, ys)
    sr = stats_mod.spearmanr(xs, ys)
    return {
        "pearson":  {"r": float(pr.statistic), "p_value": float(pr.pvalue), "n": len(xs)},
        "spearman": {"rho": float(sr.statistic), "p_value": float(sr.pvalue), "n": len(xs)},
    }


def _cohens_d(pre: List[float], post: List[float]) -> Optional[float]:
    """Paired Cohen's d on the delta = post − pre."""
    if len(pre) < 2:
        return None
    deltas = [b - a for a, b in zip(pre, post)]
    sd = pstdev(deltas) if len(deltas) > 1 else 0.0
    if sd == 0:
        return None
    return mean(deltas) / sd


def _icc_one_way(scores_by_case: Dict[str, List[float]]) -> Optional[float]:
    """Compute a one-way random ICC(1,1) as a quick inter-rater reliability
    proxy (Krippendorff's α would be cleaner but requires more deps).

    scores_by_case maps case_id → list of rater scores (length k, equal
    across cases).
    """
    cases = [v for v in scores_by_case.values() if len(v) >= 2]
    if not cases:
        return None
    k = len(cases[0])
    if any(len(v) != k for v in cases):
        # uneven panel — fall back to mean within-case agreement
        all_means = [mean(v) for v in cases]
        if pstdev(all_means) == 0:
            return None
        # Approximate ICC by 1 − var_within / var_total
        var_within = mean(pstdev(v) ** 2 for v in cases)
        var_total = pstdev([s for v in cases for s in v]) ** 2
        return None if var_total == 0 else 1.0 - var_within / var_total
    grand_mean = mean(s for v in cases for s in v)
    msr = k * sum((mean(v) - grand_mean) ** 2 for v in cases) / (len(cases) - 1)
    mse = sum((s - mean(v)) ** 2 for v in cases for s in v) / (len(cases) * (k - 1))
    if msr + (k - 1) * mse == 0:
        return None
    return (msr - mse) / (msr + (k - 1) * mse)


def run_phase2_analysis(cfg: PipelineConfig) -> Dict[str, Any]:
    """Aggregate ratings and run the full statistical pipeline."""
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage6_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage6")

    cases_path = stage6_dir / "phase2_cases.json"
    key_path = stage6_dir / "assignment_key.json"
    ratings_path = stage6_dir / "ratings_raw.json"
    for p in (cases_path, key_path, ratings_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")

    cases_meta = read_json(cases_path)
    key = read_json(key_path)
    ratings = read_json(ratings_path)

    key_by_uid = {row["item_uid"]: row for row in key}
    case_by_pid = {c["prompt_id"]: c for c in cases_meta["cases"]}

    # Index ratings: (case, variant) → list of {student, rubric→int, justification}
    bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in ratings:
        item_uid = entry["item_uid"]
        meta = key_by_uid.get(item_uid)
        if not meta:
            logger.warning("Unknown item_uid %r — skipping", item_uid)
            continue
        bucket[(meta["prompt_id"], meta["variant"])].append({
            "student_id": meta["student_id"],
            "rubric": entry.get("ratings", {}),
            "justification": entry.get("justification", ""),
        })

    stats_mod = _try_scipy()

    # Per-rubric: paired pre vs post (case-level means averaged across raters).
    paired_per_rubric: Dict[str, Dict[str, Any]] = {}
    correlation_per_rubric: Dict[str, Dict[str, Any]] = {}
    correlation_per_stratum: Dict[str, Dict[str, Dict[str, Any]]] = {
        s: {} for s in ("large_improvement", "modest_or_zero", "control")
    }

    # Collect across-case pre/post for each rubric ID
    for rubric_id in RUBRIC_IDS:
        pre_means: List[float] = []
        post_means: List[float] = []
        case_pai_deltas: List[float] = []
        case_strata: List[str] = []

        for pid, case_meta in case_by_pid.items():
            pre_raters = bucket.get((pid, "pre"), [])
            post_raters = bucket.get((pid, "post"), [])
            pre_vals = [r["rubric"].get(rubric_id) for r in pre_raters if rubric_id in r["rubric"]]
            post_vals = [r["rubric"].get(rubric_id) for r in post_raters if rubric_id in r["rubric"]]
            if not pre_vals or not post_vals:
                continue
            pre_means.append(mean(pre_vals))
            post_means.append(mean(post_vals))
            case_pai_deltas.append(case_meta["delta_PAI"])
            case_strata.append(case_meta["stratum"])

        paired_per_rubric[rubric_id] = {
            "n_cases": len(pre_means),
            "pre_mean": round(mean(pre_means), 4) if pre_means else None,
            "post_mean": round(mean(post_means), 4) if post_means else None,
            "rating_delta_mean": (
                round(mean(b - a for a, b in zip(pre_means, post_means)), 4)
                if pre_means else None
            ),
            "wilcoxon": _wilcoxon(stats_mod, pre_means, post_means),
            "cohens_d": _cohens_d(pre_means, post_means),
        }
        rating_deltas = [b - a for a, b in zip(pre_means, post_means)]
        correlation_per_rubric[rubric_id] = _correlation(stats_mod, case_pai_deltas, rating_deltas)

        # Per-stratum correlation.
        for stratum in correlation_per_stratum:
            xs = [d for d, s in zip(case_pai_deltas, case_strata) if s == stratum]
            ys = [d for d, s in zip(rating_deltas, case_strata) if s == stratum]
            correlation_per_stratum[stratum][rubric_id] = _correlation(stats_mod, xs, ys)

    # Inter-rater reliability per rubric per variant
    icc_per_rubric: Dict[str, Dict[str, Optional[float]]] = {}
    for rubric_id in RUBRIC_IDS:
        per_variant: Dict[str, Optional[float]] = {}
        for variant in ("pre", "post"):
            scores_by_case: Dict[str, List[float]] = {}
            for (pid, v), raters in bucket.items():
                if v != variant:
                    continue
                vals = [r["rubric"].get(rubric_id) for r in raters if rubric_id in r["rubric"]]
                if vals:
                    scores_by_case[pid] = [float(x) for x in vals]
            per_variant[variant] = _icc_one_way(scores_by_case)
        icc_per_rubric[rubric_id] = per_variant

    # Justifications: one item per case+variant.
    justifications: List[Dict[str, Any]] = []
    for (pid, variant), raters in bucket.items():
        for r in raters:
            j = (r.get("justification") or "").strip()
            if j:
                justifications.append({
                    "prompt_id": pid,
                    "variant": variant,
                    "student_id": r["student_id"],
                    "text": j,
                })

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    payload = {
        "scipy_available": stats_mod is not None,
        "n_cases": len(case_by_pid),
        "n_ratings": len(ratings),
        "paired_pre_vs_post_per_rubric": paired_per_rubric,
        "correlation_pai_delta_vs_rating_delta_per_rubric": correlation_per_rubric,
        "correlation_per_stratum": correlation_per_stratum,
        "icc_per_rubric": icc_per_rubric,
        "n_justifications": len(justifications),
        "justifications": justifications,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
    }
    out_path = stage6_dir / "phase2_analysis.json"
    write_json(payload, out_path)

    manifest = build_manifest(
        stage="stage6_phase2_analysis",
        params={
            "input_ratings": str(ratings_path),
            "input_key": str(key_path),
            "input_cases": str(cases_path),
            "output": str(out_path),
        },
        row_count=len(ratings),
        extra={
            "scipy_available": stats_mod is not None,
            "n_cases": len(case_by_pid),
            "n_justifications": len(justifications),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage6_dir / "manifest_phase2_analysis.json")

    logger.info(
        "Phase 2 analysis: %d ratings on %d cases, scipy=%s → %s",
        len(ratings), len(case_by_pid), stats_mod is not None, out_path,
    )
    return payload
