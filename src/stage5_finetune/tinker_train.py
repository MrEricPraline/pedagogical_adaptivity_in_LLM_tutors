"""LoRA fine-tune Qwen3-32B on Tinker using the corrective training set.

Implements Step 2 of Phase 1 (corrective fine-tuning). The same code path
is used for every LoRA rank in the diagnostic sweep r ∈ {1, 4, 8, 16}.

This module imports ``tinker`` lazily so that the rest of the Stage 5
pipeline (scoring, corrective-data generation, post-hoc analysis) remains
runnable without the Tinker SDK installed. The Tinker SDK is gated behind
a waitlist; we want everything around it to work today and only the
training/sampling steps to require it.

Tinker API references (verified against tinker-cookbook tutorials 02 and
303):

* ``ServiceClient()`` — entry point, picks up ``TINKER_API_KEY``.
* ``service_client.create_lora_training_client(base_model, rank)`` →
  ``TrainingClient``.
* ``training_client.forward_backward(data, "cross_entropy")`` returns a
  ``Future``; resolve with ``.result()``. Accepts a list of ``Datum``.
* ``training_client.optim_step(types.AdamParams(learning_rate=lr))`` —
  also returns a ``Future``.
* ``training_client.save_state(path)`` — async; returns a Future whose
  ``.result()`` carries the URI used at sampling time.
* ``training_client.save_weights_and_get_sampling_client(name)`` —
  convenience wrapper that materialises a sampling client.
"""

from __future__ import annotations

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

logger = logging.getLogger("pipeline")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-32B"
DEFAULT_RANKS = (1, 4, 8, 16)


def _require_tinker():
    """Import tinker lazily; raise a helpful error if it's missing."""
    try:
        import tinker  # noqa: F401
        from tinker import types  # noqa: F401
    except ImportError as exc:  # pragma: no cover — environment-dependent
        raise RuntimeError(
            "The `tinker` SDK is required for Stage 5 fine-tuning. "
            "Install it with `pip install tinker` once your waitlist access "
            "is granted, then export TINKER_API_KEY=<your_key>."
        ) from exc

    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is not set. Export it (or add it to .env) before "
            "running Stage 5 fine-tuning."
        )


def _get_qwen3_renderer(training_client):
    """Return a Qwen3InstructRenderer bound to the training client's tokenizer.

    The cookbook's registry-based ``get_renderer`` helper requires each
    renderer module to be imported before lookup; importing the class
    directly is simpler and avoids order bugs.
    """
    try:
        from tinker_cookbook.renderers.qwen3 import Qwen3InstructRenderer  # type: ignore
    except ImportError:
        return None
    return Qwen3InstructRenderer(tokenizer=training_client.get_tokenizer())


def _build_datums(examples: List[Dict[str, Any]], training_client, base_model: str):
    """Convert chat-format examples → list of tinker.Datum objects.

    Delegates to the cookbook's canonical ``conversation_to_datum`` helper,
    which:

    * runs the Qwen3 renderer to produce aligned (tokens, per-token weights),
    * right-shifts inputs / left-shifts targets for next-token prediction,
    * builds the ``TensorData`` records for ``weights`` and ``target_tokens``
      that the ``cross_entropy`` loss function on the Tinker server expects.

    Falls back to a manual implementation only if ``tinker-cookbook`` is not
    installed (so the server-side ``loss_fn`` still receives both keys).
    """
    import tinker  # local import — only reached after _require_tinker()

    try:
        from tinker_cookbook.supervised.common import datum_from_model_input_weights
        from tinker_cookbook.renderers import TrainOnWhat
    except ImportError:
        datum_from_model_input_weights = None
        TrainOnWhat = None  # type: ignore

    renderer = _get_qwen3_renderer(training_client)

    datums: List[Any] = []
    for ex in examples:
        messages = ex["messages"]

        if renderer is not None and datum_from_model_input_weights is not None:
            model_input, weights = renderer.build_supervised_example(
                messages,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            datums.append(
                datum_from_model_input_weights(
                    model_input=model_input,
                    weights=weights,
                    max_length=None,
                    reduction="mean",
                )
            )
            continue

        # Manual fallback: build the right-shifted Datum without the cookbook.
        # Reproduces the structure expected by the cross_entropy loss_fn:
        # loss_fn_inputs = {"weights": TensorData, "target_tokens": TensorData}
        tokenizer = training_client.get_tokenizer()
        full = tokenizer.apply_chat_template(messages, tokenize=True)
        prompt_only = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        if len(full) < len(prompt_only) + 1:
            raise ValueError(
                "Chat template returned fewer assistant tokens than expected "
                f"(full={len(full)}, prompt={len(prompt_only)})."
            )
        per_token_weights = [0.0] * len(prompt_only) + [1.0] * (len(full) - len(prompt_only))
        input_tokens = full[:-1]
        target_tokens = full[1:]
        per_token_weights = per_token_weights[1:]
        total_w = sum(per_token_weights) or 1.0
        per_token_weights = [w / total_w for w in per_token_weights]

        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "weights": tinker.TensorData(
                        data=per_token_weights,
                        dtype="float32",
                        shape=[len(per_token_weights)],
                    ),
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[len(target_tokens)],
                    ),
                },
            )
        )
    return datums


def train_one_rank(
    *,
    examples: List[Dict[str, Any]],
    rank: int,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 3,
    lr: float = 1e-4,
    log_every: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """LoRA fine-tune at a single rank. Returns metadata + adapter handle."""
    _require_tinker()
    import tinker
    from tinker import types

    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    logger.info(
        "Stage 5 fine-tune: rank=%d base=%s epochs=%d lr=%.2e (n=%d examples)",
        rank, base_model, epochs, lr, len(examples),
    )

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=rank,
    )

    datums = _build_datums(examples, training_client, base_model)
    token_lens = [d.model_input.length for d in datums]
    logger.info(
        "Stage 5 fine-tune: tokenized %d examples (min/max/mean tokens = %d/%d/%.1f)",
        len(datums), min(token_lens), max(token_lens), mean(token_lens),
    )

    history: List[Dict[str, Any]] = []
    total_steps = 0

    def _extract_loss(metrics: Dict[str, float]) -> float:
        """Find a sensible scalar to log as "loss" from the API metrics dict.

        Tinker reports per-step metrics under a few possible keys depending
        on the loss function. We prefer the canonical ``loss`` key but fall
        back to anything that looks like a loss.
        """
        if not metrics:
            return float("nan")
        for key in ("loss", "train_loss", "loss/mean", "cross_entropy_loss"):
            if key in metrics:
                return float(metrics[key])
        for k, v in metrics.items():
            if "loss" in k.lower():
                return float(v)
        return float("nan")

    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []

        for i, datum in enumerate(datums, start=1):
            fwd_future = training_client.forward_backward([datum], "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=lr))

            fwd_result = fwd_future.result()
            optim_future.result()

            loss_val = _extract_loss(getattr(fwd_result, "metrics", {}) or {})
            epoch_losses.append(loss_val)
            total_steps += 1

            if i % log_every == 0 or i == len(datums):
                window = [v for v in epoch_losses[-log_every:] if v == v]  # filter NaN
                avg = mean(window) if window else float("nan")
                logger.info(
                    "  rank=%d epoch=%d/%d step=%d/%d loss(window)=%.4f",
                    rank, epoch, epochs, i, len(datums), avg,
                )

        clean = [v for v in epoch_losses if v == v]
        history.append({
            "epoch": epoch,
            "mean_loss": round(mean(clean), 4) if clean else None,
            "min_loss": round(min(clean), 4) if clean else None,
            "max_loss": round(max(clean), 4) if clean else None,
        })
        logger.info(
            "  rank=%d epoch=%d complete — mean loss=%s",
            rank, epoch, history[-1]["mean_loss"],
        )

    # Persist the adapter to Tinker's sampler weight store and capture the
    # resulting URI ("tinker://.../weights/...") so we can recreate a
    # SamplingClient across CLI invocations.
    adapter_name = f"corrective_r{rank}_e{epochs}"
    save_future = training_client.save_weights_for_sampler(adapter_name)
    save_result = save_future.result()
    adapter_uri = getattr(save_result, "path", None) or adapter_name

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    metadata = {
        "rank": rank,
        "base_model": base_model,
        "epochs": epochs,
        "learning_rate": lr,
        "n_examples": len(examples),
        "total_steps": total_steps,
        "history": history,
        "adapter_name": adapter_name,
        "adapter_uri": adapter_uri,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
    }

    if output_dir is not None:
        write_json(metadata, output_dir / f"adapter_r{rank}.json")

    final_loss = history[-1]["mean_loss"]
    logger.info(
        "Stage 5 fine-tune: rank=%d done in %.1fs — final loss=%s, adapter=%s",
        rank, duration, final_loss, adapter_uri,
    )
    return metadata


def run_finetune(
    cfg: PipelineConfig,
    *,
    ranks: List[int] | None = None,
    epochs: int = 3,
    lr: float = 1e-4,
    base_model: str = DEFAULT_BASE_MODEL,
    corrective_file: str = "corrective_training_data.json",
) -> Dict[str, Any]:
    """Train one LoRA adapter per rank in `ranks`.

    ``corrective_file`` selects which corrective training set to use:

    * ``corrective_training_data.json``            — global (bottom-N) mode
    * ``corrective_training_data_stratified.json`` — proposal-aligned
      stratified mode (50-100 examples per weak cell, full optimal target)
    """
    started_at = datetime.now(timezone.utc)
    started_perf = time.time()

    stage5_dir = Path(cfg.output_dir) if cfg.output_dir else (DATA_DIR / "stage5")
    stage5_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = stage5_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    corrective_path = stage5_dir / corrective_file
    if not corrective_path.exists():
        raise FileNotFoundError(
            f"{corrective_path} not found. Build the corrective dataset first "
            "(run-stage5-corrective or run-stage5-corrective-stratified)."
        )
    examples = read_json(corrective_path)
    logger.info(
        "Stage 5 fine-tune: loaded %d corrective examples from %s",
        len(examples), corrective_path.name,
    )

    ranks = list(ranks) if ranks else list(DEFAULT_RANKS)
    results: Dict[int, Dict[str, Any]] = {}

    for r in ranks:
        results[r] = train_one_rank(
            examples=examples,
            rank=r,
            base_model=base_model,
            epochs=epochs,
            lr=lr,
            output_dir=adapters_dir,
        )

    duration = round(time.time() - started_perf, 3)
    finished_at = datetime.now(timezone.utc)

    manifest = build_manifest(
        stage="stage5_finetune",
        params={
            "base_model": base_model,
            "ranks": ranks,
            "epochs": epochs,
            "learning_rate": lr,
            "n_examples": len(examples),
            "input_corrective": "data/stage5/corrective_training_data.json",
            "output_adapters_dir": "data/stage5/adapters",
        },
        row_count=len(ranks),
        extra={
            "adapters": {
                str(r): {
                    "adapter_uri": meta["adapter_uri"],
                    "final_loss": meta["history"][-1]["mean_loss"],
                    "duration_seconds": meta["duration_seconds"],
                }
                for r, meta in results.items()
            },
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
        },
    )
    write_json(manifest, stage5_dir / "manifest_finetune.json")

    return {"results": results, "ranks": ranks, "manifest": manifest}
