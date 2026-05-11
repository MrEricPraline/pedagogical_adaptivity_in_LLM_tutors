"""Step 3 of Phase 1 — query the LoRA-fine-tuned model and re-score with PAI.

For every adapter produced by :func:`src.stage5_finetune.tinker_train.run_finetune`
we replay the same 30 low-PAI cases through the corrected model, parse the
JSON response, and apply the scoring matrices to compute the
post-intervention PAI per case and per DP. The pre/post deltas are the
core diagnostic this stage produces.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from src.common.io_utils import read_json, write_json
from src.pipeline.config import DATA_DIR, PipelineConfig
from src.pipeline.manifests import build_manifest
from src.stage4_query.validator import VALID_SELECTIONS, validate_response
from src.stage5_finetune.prompt_builder import (
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)
from src.stage5_finetune.tinker_train import DEFAULT_BASE_MODEL, _require_tinker
from src.stage5_scoring.matrices import DECISION_POINTS
from src.stage5_scoring.scorer import score_response

logger = logging.getLogger("pipeline")


def _make_sampling_client(base_model: str, adapter_uri: str):
    """Materialise a Tinker sampling client from an adapter URI."""
    _require_tinker()
    import tinker

    service_client = tinker.ServiceClient()
    return service_client.create_sampling_client(
        base_model=base_model,
        model_path=adapter_uri,
    )


def _generate(sampling_client, messages, *, max_tokens: int, temperature: float) -> str:
    """Run one chat completion through the sampling client and return text.

    Uses the cookbook's Qwen3 renderer when available so the prompt
    encoding matches what was used during training. Falls back to the
    tokenizer's ``apply_chat_template`` otherwise.
    """
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
    """Best-effort JSON extraction. Tolerates models that wrap output in
    ```json fences or emit leading prose."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip ``` or ```json fence
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    # If the model emitted trailing text after the JSON, find the closing brace
    # of the outermost object and slice there.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    return json.loads(text)


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
    """Query the corrected model on every case and produce a results file."""
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    sampling_client = _make_sampling_client(base_model, adapter_uri)
    logger.info(
        "Stage 5 query: rank=%d adapter=%s — querying %d cases",
        rank, adapter_uri, len(cases),
    )

    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        pid = case["prompt_id"]
        narrative = narratives.get(pid, "")
        if not narrative:
            results.append({
                "prompt_id": pid,
                "rank": rank,
                "status": "skipped_no_narrative",
                "pre_PAI": case.get("prompt_PAI"),
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
                "status": f"sampling_error: {exc}",
                "pre_PAI": case.get("prompt_PAI"),
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

            pre_pai = case["prompt_PAI"]
            post_pai = scores["prompt_PAI"]
            pre_dp = case.get("dp_means", {})
            post_dp = scores["dp_means"]

            results.append({
                "prompt_id": pid,
                "rank": rank,
                "status": "ok",
                "pre_PAI": pre_pai,
                "post_PAI": round(post_pai, 4),
                "delta_PAI": round(post_pai - pre_pai, 4),
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
                "  [%2d/%d] %s rank=%d pre=%+.4f post=%+.4f Δ=%+.4f",
                idx, len(cases), pid, rank, pre_pai, post_pai, post_pai - pre_pai,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 5 query: %s parse/score failed (%s)", pid, exc)
            results.append({
                "prompt_id": pid,
                "rank": rank,
                "status": f"parse_error: {exc}",
                "pre_PAI": case.get("prompt_PAI"),
                "post_PAI": None,
                "delta_PAI": None,
                "post_response_text": raw_text,
            })

    ok = [r for r in results if r["status"] == "ok"]
    deltas = [r["delta_PAI"] for r in ok]
    summary = {
        "rank": rank,
        "total": len(results),
        "ok": len(ok),
        "improved": sum(1 for d in deltas if d > 0),
        "regressed": sum(1 for d in deltas if d < 0),
        "unchanged": sum(1 for d in deltas if d == 0),
        "delta_PAI_mean": round(mean(deltas), 4) if deltas else None,
        "delta_PAI_min": round(min(deltas), 4) if deltas else None,
        "delta_PAI_max": round(max(deltas), 4) if deltas else None,
        "post_PAI_mean": round(mean(r["post_PAI"] for r in ok), 4) if ok else None,
    }
    if ok:
        # Per-DP deltas, averaged across the cases scored for this rank.
        per_dp_delta = {}
        for dp in DECISION_POINTS:
            vals = [r["delta_dp_means"][dp] for r in ok]
            per_dp_delta[dp] = round(mean(vals), 4)
        summary["delta_dp_means"] = per_dp_delta

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)
    payload = {
        "rank": rank,
        "adapter_uri": adapter_uri,
        "base_model": base_model,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
        "summary": summary,
        "results": results,
    }

    if output_dir is not None:
        write_json(payload, output_dir / f"post_intervention_r{rank}.json")

    logger.info(
        "Stage 5 query: rank=%d done — ok=%d/%d, mean Δ=%s, improved=%d in %.1fs",
        rank, len(ok), len(results), summary["delta_PAI_mean"],
        summary["improved"], duration,
    )

    return payload


def run_query_post_intervention(
    cfg: PipelineConfig,
    *,
    ranks: List[int] | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """For every trained rank, re-query the low-PAI cases and score them."""
    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    adapters_dir = stage5_dir / "adapters"
    if not adapters_dir.exists():
        raise FileNotFoundError(
            "data/stage5/adapters/ not found. Run "
            "`python -m src.pipeline.cli run-stage5-finetune` first."
        )

    scored_path = stage5_dir / "scored_dataset.json"
    if not scored_path.exists():
        raise FileNotFoundError("data/stage5/scored_dataset.json not found.")

    corrective_path = stage5_dir / "corrective_training_data.json"
    if not corrective_path.exists():
        raise FileNotFoundError("data/stage5/corrective_training_data.json not found.")

    scored_all = read_json(scored_path)
    corrective = read_json(corrective_path)
    case_ids = [ex["prompt_id"] for ex in corrective]
    by_id = {r["prompt_id"]: r for r in scored_all}
    cases = [by_id[pid] for pid in case_ids if pid in by_id]

    # Narratives come from Stage 4 — we need them for the user prompt.
    from src.stage5_finetune.corrective_data import _load_narratives
    narratives = _load_narratives(cfg.stage4_dir)

    if ranks is None:
        # Discover trained adapters from disk.
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
        adapter_meta = read_json(meta_path)
        payloads[r] = query_one_adapter(
            cfg,
            rank=r,
            adapter_uri=adapter_meta["adapter_uri"],
            cases=cases,
            narratives=narratives,
            base_model=base_model,
            max_tokens=max_tokens,
            temperature=temperature,
            output_dir=stage5_dir,
        )

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    manifest = build_manifest(
        stage="stage5_query_post_intervention",
        params={
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
    write_json(manifest, stage5_dir / "manifest_query.json")
    return {"payloads": payloads, "manifest": manifest}
