"""Step 3 of Phase 1 — query the model (base or LoRA-fine-tuned) and re-score with PAI.

Two query modes are supported:

* **Adapter mode** — pass ``adapter_uri`` to query the LoRA-fine-tuned
  Qwen3-32B for a given rank.
* **Baseline mode** — pass ``adapter_uri=None`` to query the bare base
  model (Qwen3-32B, no LoRA). Used by ``run-stage5-baseline`` so we can
  separate the LoRA effect from the base-model change (Gemini → Qwen).

Two case-set targets are supported:

* **train** — the cases used to build the corrective training set
  (default, legacy behavior). Adapter output → ``post_intervention_r{rank}.json``.
* **heldout** — the disjoint cases from ``eval_heldout_cases.json``.
  Adapter output → ``post_intervention_heldout_r{rank}.json``.

A baseline run writes to ``baseline_qwen_{target}.json``.

Field semantics inside each result row:

* ``pre_PAI`` — the reference score from Experiment 1 (Gemini 3.1).
* ``post_PAI`` — the PAI scored on this run's output (Qwen-base if
  baseline, Qwen+LoRA if adapter).
* ``delta_PAI`` = ``post_PAI − pre_PAI``. Note this is "vs Gemini" — to
  isolate the *LoRA effect* you must subtract the corresponding baseline
  delta (see ``interference.py``).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Literal, Optional

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest
from src.stage4_query.validator import validate_response
from src.stage5_finetune.prompt_builder import (
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)
from src.stage5_finetune.tinker_train import DEFAULT_BASE_MODEL, _require_tinker
from src.stage5_scoring.matrices import DECISION_POINTS
from src.stage5_scoring.scorer import score_response

logger = logging.getLogger("pipeline")

Target = Literal["train", "heldout"]


def _make_sampling_client(base_model: str, adapter_uri: Optional[str]):
    """Materialise a Tinker sampling client.

    If ``adapter_uri`` is None, the bare base model is used (no LoRA).
    """
    _require_tinker()
    import tinker

    service_client = tinker.ServiceClient()
    if adapter_uri is None:
        return service_client.create_sampling_client(base_model=base_model)
    return service_client.create_sampling_client(
        base_model=base_model,
        model_path=adapter_uri,
    )


def _generate(sampling_client, messages, *, max_tokens: int, temperature: float) -> str:
    """Run one chat completion through the sampling client and return text."""
    import tinker

    tokenizer = sampling_client.get_tokenizer()

    try:
        from tinker_cookbook.renderers.qwen3 import Qwen3InstructRenderer  # type: ignore
        renderer = Qwen3InstructRenderer(tokenizer=tokenizer)
        model_input = renderer.build_generation_prompt(messages)
        stop_seqs = renderer.get_stop_sequences()
    except ImportError:
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        model_input = tinker.ModelInput.from_ints(tokens=tokens)
        stop_seqs = None

    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_seqs,
    )
    sample_future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    response = sample_future.result()

    if not getattr(response, "sequences", None):
        raise RuntimeError(f"Empty sampling response: {response!r}")
    seq = response.sequences[0]
    tokens_np = getattr(seq, "tokens_np", None)
    if tokens_np is None:
        raise RuntimeError(f"Sampling sequence missing tokens_np: {seq!r}")

    return tokenizer.decode(tokens_np.tolist(), skip_special_tokens=True)


def _parse_response_text(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    return json.loads(text)


def _output_filename(*, target: Target, rank: Optional[int], is_baseline: bool) -> str:
    """Compute the per-run output filename.

    Legacy behavior preserved: adapter+train still writes
    ``post_intervention_r{rank}.json`` (no target prefix) so existing files
    stay valid.
    """
    if is_baseline:
        return f"baseline_qwen_{target}.json"
    if target == "train":
        return f"post_intervention_r{rank}.json"
    return f"post_intervention_{target}_r{rank}.json"


def query_one(
    cfg: PipelineConfig,
    *,
    cases: List[Dict[str, Any]],
    narratives: Dict[str, str],
    target: Target,
    rank: Optional[int] = None,
    adapter_uri: Optional[str] = None,
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Query the model on every case and produce a results file.

    ``adapter_uri=None`` → baseline (Qwen3-32B without LoRA).
    ``adapter_uri=<uri>`` + ``rank`` → LoRA-corrected model at the given rank.

    Output schema keeps the legacy field names ``pre_PAI`` (Gemini reference),
    ``post_PAI`` (this run), ``delta_PAI`` (post − pre = vs-Gemini), so the
    existing interference analysis keeps working.
    """
    is_baseline = adapter_uri is None
    label = "baseline" if is_baseline else f"rank={rank}"
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    sampling_client = _make_sampling_client(base_model, adapter_uri)
    logger.info(
        "Stage 5 query: target=%s %s — querying %d cases",
        target, label, len(cases),
    )

    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        pid = case["prompt_id"]
        narrative = narratives.get(pid, "")
        pre_pai = case.get("prompt_PAI")
        pre_dp = case.get("dp_means", {})
        if not narrative:
            results.append({
                "prompt_id": pid,
                "rank": rank,
                "is_baseline": is_baseline,
                "status": "skipped_no_narrative",
                "pre_PAI": pre_pai,
                "post_PAI": None,
                "delta_PAI": None,
            })
            continue

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": build_user_prompt(narrative)},
        ]

        try:
            raw_text = _generate(
                sampling_client, messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 5 query: %s sampling failed (%s)", pid, exc)
            results.append({
                "prompt_id": pid,
                "rank": rank,
                "is_baseline": is_baseline,
                "status": f"sampling_error: {exc}",
                "pre_PAI": pre_pai,
                "post_PAI": None,
                "delta_PAI": None,
            })
            continue

        try:
            parsed = _parse_response_text(raw_text)
            ok, reason = validate_response(parsed)
            if not ok:
                raise ValueError(f"Schema validation failed: {reason}")

            meta = {
                "bloom_band": case["bloom_band"],
                "knowledge_state": case["knowledge_state"],
                "learning_stage": case["learning_stage"],
                "learning_context": case["learning_context"],
                "subject_family": case["subject_family"],
            }
            scores = score_response(parsed["activities"], meta)

            post_pai = scores["prompt_PAI"]
            post_dp = scores["dp_means"]

            results.append({
                "prompt_id": pid,
                "rank": rank,
                "is_baseline": is_baseline,
                "status": "ok",
                "pre_PAI": pre_pai,
                "post_PAI": round(post_pai, 4),
                "delta_PAI": round(post_pai - (pre_pai or 0.0), 4),
                "pre_dp_means": pre_dp,
                "post_dp_means": {dp: round(v, 4) for dp, v in post_dp.items()},
                "delta_dp_means": {
                    dp: round(post_dp[dp] - pre_dp.get(dp, 0.0), 4)
                    for dp in post_dp
                },
                "post_response": parsed,
                "post_response_text": raw_text,
            })
            logger.info(
                "  [%2d/%d] %s %s pre=%+.4f post=%+.4f Δ=%+.4f",
                idx, len(cases), pid, label,
                pre_pai if pre_pai is not None else float("nan"),
                post_pai, post_pai - (pre_pai or 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 5 query: %s parse/score failed (%s)", pid, exc)
            results.append({
                "prompt_id": pid,
                "rank": rank,
                "is_baseline": is_baseline,
                "status": f"parse_error: {exc}",
                "pre_PAI": pre_pai,
                "post_PAI": None,
                "delta_PAI": None,
                "post_response_text": raw_text,
            })

    ok_results = [r for r in results if r["status"] == "ok"]
    deltas = [r["delta_PAI"] for r in ok_results]
    summary: Dict[str, Any] = {
        "target": target,
        "rank": rank,
        "is_baseline": is_baseline,
        "total": len(results),
        "ok": len(ok_results),
        "improved": sum(1 for d in deltas if d > 0),
        "regressed": sum(1 for d in deltas if d < 0),
        "unchanged": sum(1 for d in deltas if d == 0),
        "delta_PAI_mean": round(mean(deltas), 4) if deltas else None,
        "delta_PAI_min": round(min(deltas), 4) if deltas else None,
        "delta_PAI_max": round(max(deltas), 4) if deltas else None,
        "post_PAI_mean": round(mean(r["post_PAI"] for r in ok_results), 4) if ok_results else None,
    }
    if ok_results:
        per_dp_delta = {}
        for dp in DECISION_POINTS:
            vals = [r["delta_dp_means"][dp] for r in ok_results]
            per_dp_delta[dp] = round(mean(vals), 4)
        summary["delta_dp_means"] = per_dp_delta

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    payload = {
        "target": target,
        "rank": rank,
        "is_baseline": is_baseline,
        "adapter_uri": adapter_uri,
        "base_model": base_model,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
        "summary": summary,
        "results": results,
    }

    if output_dir is not None:
        fname = _output_filename(target=target, rank=rank, is_baseline=is_baseline)
        write_json(payload, output_dir / fname)

    logger.info(
        "Stage 5 query: target=%s %s done — ok=%d/%d, mean Δ=%s, improved=%d in %.1fs",
        target, label, len(ok_results), len(results),
        summary["delta_PAI_mean"], summary["improved"], duration,
    )

    return payload


# Legacy wrapper for older callers.
def query_one_adapter(
    cfg: PipelineConfig,
    *,
    rank: int,
    adapter_uri: str,
    cases: List[Dict[str, Any]],
    narratives: Dict[str, str],
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Legacy wrapper. Calls query_one with target=train + adapter."""
    return query_one(
        cfg,
        cases=cases,
        narratives=narratives,
        target="train",
        rank=rank,
        adapter_uri=adapter_uri,
        base_model=base_model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_dir=output_dir,
    )


def _load_cases_for_target(
    stage5_dir: Path,
    target: Target,
    *,
    corrective_file: str = "corrective_training_data.json",
) -> List[Dict[str, Any]]:
    """Resolve which case list to query depending on the target.

    For ``target="train"`` the case list is the set of prompt_ids in
    ``corrective_file`` — i.e. exactly the cases the adapter was trained
    on. Override ``corrective_file`` to point at the stratified set
    (``corrective_training_data_stratified.json``) when querying an
    adapter that was fine-tuned on it.
    """
    if target == "train":
        corrective_path = stage5_dir / corrective_file
        scored_path = stage5_dir / "scored_dataset.json"
        if not corrective_path.exists():
            raise FileNotFoundError(
                f"{corrective_path} not found. Build the matching corrective set first."
            )
        if not scored_path.exists():
            raise FileNotFoundError("scored_dataset.json not found.")
        corrective = read_json(corrective_path)
        # Dedupe case_ids while preserving order (per-DP files repeat ids).
        seen: set = set()
        case_ids: List[str] = []
        for ex in corrective:
            pid = ex["prompt_id"]
            if pid not in seen:
                seen.add(pid)
                case_ids.append(pid)
        by_id = {r["prompt_id"]: r for r in read_json(scored_path)}
        return [by_id[pid] for pid in case_ids if pid in by_id]

    if target == "heldout":
        heldout_path = stage5_dir / "eval_heldout_cases.json"
        if not heldout_path.exists():
            raise FileNotFoundError(
                "eval_heldout_cases.json not found. Run `run-stage5-heldout` first."
            )
        return read_json(heldout_path)

    raise ValueError(f"Unknown target: {target!r}")


def _is_complete_run(
    payload_path: Path,
    *,
    target: Target,
    expected_case_ids: List[str],
) -> bool:
    """Return True iff ``payload_path`` already contains a finished run that
    matches the current ``target`` and the exact ``expected_case_ids``.

    Used by ``run_query_post_intervention`` and ``run_query_baseline`` to
    skip ranks whose output file is already present and consistent. Stale
    files (different target, different case count, or different case ids)
    are reported as incomplete so they get redone.
    """
    if not payload_path.exists():
        return False
    try:
        payload = read_json(payload_path)
    except Exception:  # noqa: BLE001
        return False
    if payload.get("target") != target:
        return False
    results = payload.get("results") or []
    ok_ids = [r["prompt_id"] for r in results if r.get("status") == "ok"]
    if set(ok_ids) != set(expected_case_ids):
        return False
    return True


def run_query_post_intervention(
    cfg: PipelineConfig,
    *,
    ranks: List[int] | None = None,
    target: Target = "train",
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    corrective_file: str = "corrective_training_data.json",
    force: bool = False,
) -> Dict[str, Any]:
    """For every trained rank, re-query the chosen case set and score it.

    Resume behavior (default): if ``post_intervention_{target}_r{rank}.json``
    (or the legacy ``post_intervention_r{rank}.json`` when target=train)
    already exists and contains a finished run with the same target and
    the same set of prompt_ids, that rank is **skipped** and its existing
    file is reused. Set ``force=True`` to ignore existing files.
    """
    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    adapters_dir = stage5_dir / "adapters"
    if not adapters_dir.exists():
        raise FileNotFoundError(
            "data/stage5/adapters/ not found. Run `run-stage5-finetune` first."
        )

    cases = _load_cases_for_target(stage5_dir, target, corrective_file=corrective_file)
    expected_ids = [c["prompt_id"] for c in cases]

    from src.stage5_finetune.corrective_data import _load_narratives
    narratives = _load_narratives(cfg.stage4_dir)

    if ranks is None:
        ranks = sorted(
            int(p.stem.split("_r")[-1])
            for p in adapters_dir.glob("adapter_r*.json")
        )

    started_at = datetime.now(timezone.utc)
    started_perf = time.time()
    payloads: Dict[int, Dict[str, Any]] = {}

    for r in ranks:
        meta_path = adapters_dir / f"adapter_r{r}.json"
        if not meta_path.exists():
            logger.warning("Stage 5 query: no adapter metadata for rank=%d, skipping", r)
            continue

        out_name = _output_filename(target=target, rank=r, is_baseline=False)
        out_path = stage5_dir / out_name
        if not force and _is_complete_run(out_path, target=target, expected_case_ids=expected_ids):
            logger.info(
                "Stage 5 query: rank=%d target=%s already complete in %s — skipping (use --force to redo)",
                r, target, out_name,
            )
            payloads[r] = read_json(out_path)
            continue

        adapter_meta = read_json(meta_path)
        payloads[r] = query_one(
            cfg,
            cases=cases,
            narratives=narratives,
            target=target,
            rank=r,
            adapter_uri=adapter_meta["adapter_uri"],
            base_model=base_model,
            max_tokens=max_tokens,
            temperature=temperature,
            output_dir=stage5_dir,
        )

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    manifest = build_manifest(
        stage=f"stage5_query_{target}",
        params={
            "target": target,
            "base_model": base_model,
            "ranks": list(payloads.keys()),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_cases": len(cases),
        },
        row_count=sum(p["summary"]["ok"] for p in payloads.values()),
        extra={
            "summaries": {str(r): p["summary"] for r, p in payloads.items()},
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    manifest_name = "manifest_query.json" if target == "train" else f"manifest_query_{target}.json"
    write_json(manifest, stage5_dir / manifest_name)
    return {"payloads": payloads, "manifest": manifest}


def run_query_baseline(
    cfg: PipelineConfig,
    *,
    target: Target = "heldout",
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    corrective_file: str = "corrective_training_data.json",
    force: bool = False,
) -> Dict[str, Any]:
    """Query the bare base model (no LoRA) on a target case set.

    Resume: if ``baseline_qwen_{target}.json`` already exists with a
    matching target and full case coverage, the run is skipped and the
    existing file is loaded instead. Pass ``force=True`` to redo it.
    """
    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    stage5_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases_for_target(stage5_dir, target, corrective_file=corrective_file)
    expected_ids = [c["prompt_id"] for c in cases]

    out_name = _output_filename(target=target, rank=None, is_baseline=True)
    out_path = stage5_dir / out_name
    if not force and _is_complete_run(out_path, target=target, expected_case_ids=expected_ids):
        logger.info(
            "Stage 5 baseline: target=%s already complete in %s — skipping",
            target, out_name,
        )
        return {"payload": read_json(out_path), "manifest": None}

    from src.stage5_finetune.corrective_data import _load_narratives
    narratives = _load_narratives(cfg.stage4_dir)

    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    payload = query_one(
        cfg,
        cases=cases,
        narratives=narratives,
        target=target,
        rank=None,
        adapter_uri=None,
        base_model=base_model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_dir=stage5_dir,
    )

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    manifest = build_manifest(
        stage=f"stage5_baseline_{target}",
        params={
            "target": target,
            "base_model": base_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_cases": len(cases),
        },
        row_count=payload["summary"]["ok"],
        extra={
            "summary": payload["summary"],
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage5_dir / f"manifest_baseline_{target}.json")
    return {"payload": payload, "manifest": manifest}
