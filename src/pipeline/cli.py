"""Command-line interface for the pedagogical adaptivity pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, dest="base_url")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument(
        "--requests-per-minute",
        "--rpm",
        type=int,
        default=None,
        dest="requests_per_minute",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=None, dest="checkpoint_every"
    )
    parser.add_argument(
        "--input",
        "--input-path",
        type=str,
        default=None,
        dest="input_path",
    )
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir")


def _add_stage4_args(parser: argparse.ArgumentParser) -> None:
    """Stage-4-specific flags (Gemini)."""
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        dest="max_output_tokens",
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default=None,
        dest="thinking_level",
        choices=["low", "medium", "high"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Pedagogical Adaptivity in LLM Tutors — research pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("run-stage1", help="Generate factorial sample (Stage 1)")
    _add_common_args(p1)

    p2 = sub.add_parser("run-stage2", help="Generate case narratives (Stage 2)")
    _add_common_args(p2)

    pa = sub.add_parser("run-all", help="Run Stage 1 then Stage 2")
    _add_common_args(pa)

    pr = sub.add_parser("repair", help="Re-generate failed cases from Stage 2")
    _add_common_args(pr)

    pd = sub.add_parser("dedup", help="Deduplicate narratives and prepare regeneration list")
    _add_common_args(pd)

    pg = sub.add_parser("regenerate", help="Regenerate cases flagged by dedup with diversity")
    _add_common_args(pg)

    pv = sub.add_parser("audit", help="Validate a JSONL dataset")
    pv.add_argument("file", nargs="?", default=None, help="Path to JSONL (default: cases_final.jsonl)")
    _add_common_args(pv)

    p4 = sub.add_parser(
        "run-stage4",
        help="Query Gemini 3.1 Pro Preview for activity plans (Stage 4)",
    )
    _add_common_args(p4)
    _add_stage4_args(p4)

    # ── Stage 5 ─────────────────────────────────────────────────────────
    p5s = sub.add_parser(
        "run-stage5-score",
        help="Score Stage 4 outputs with the PAI matrices (Stage 5 / scoring)",
    )
    _add_common_args(p5s)

    p5c = sub.add_parser(
        "run-stage5-corrective",
        help="Build the corrective LoRA training set from the lowest-PAI cases",
    )
    _add_common_args(p5c)
    p5c.add_argument(
        "--n", type=int, default=30,
        help="Number of lowest-PAI cases to include (default: 30)",
    )

    p5w = sub.add_parser(
        "run-stage5-weak-cells",
        help="Identify (DP × condition) cells with lowest PAI scores",
    )
    _add_common_args(p5w)
    p5w.add_argument(
        "--k", type=int, default=10,
        help="Number of weakest cells to flag (default: 10)",
    )

    p5cs = sub.add_parser(
        "run-stage5-corrective-stratified",
        help="Build stratified corrective set (50-100 examples per weak cell)",
    )
    _add_common_args(p5cs)
    p5cs.add_argument(
        "--per-cell", type=int, default=75, dest="per_cell",
        help="Examples per weak cell (proposal: 50-100, default: 75)",
    )
    p5cs.add_argument(
        "--target-dp-only", action="store_true", dest="target_dp_only",
        default=False,
        help="Correct only the weak cell's DP (target for per-DP isolated LoRAs)",
    )

    p5pdp = sub.add_parser(
        "run-stage5-per-dp-finetune",
        help="Train 5 isolated LoRA adapters (one per DP) for causal interference",
    )
    _add_common_args(p5pdp)
    p5pdp.add_argument("--rank", type=int, required=True,
                       help="LoRA rank for the per-DP adapters")
    p5pdp.add_argument("--epochs", type=int, default=3)
    p5pdp.add_argument("--lr", type=float, default=1e-4)
    p5pdp.add_argument("--base-model", type=str, default="Qwen/Qwen3-32B",
                       dest="base_model")

    p5ci = sub.add_parser(
        "run-stage5-causal-interference",
        help="Build the causal 5×5 interference matrix from per-DP adapters",
    )
    _add_common_args(p5ci)
    p5ci.add_argument(
        "--target", type=str, default="heldout",
        choices=["train", "heldout"],
        help="Case set to query (default: heldout)",
    )
    p5ci.add_argument("--base-model", type=str, default="Qwen/Qwen3-32B",
                      dest="base_model")
    p5ci.add_argument("--max-tokens-out", type=int, default=4096,
                      dest="max_tokens_out")
    p5ci.add_argument(
        "--skip-query", action="store_true", dest="skip_query",
        default=False,
        help="Reuse existing per_dp_query_*.json instead of re-querying",
    )

    p5f = sub.add_parser(
        "run-stage5-finetune",
        help="LoRA fine-tune Qwen3-32B on Tinker (Stage 5 Phase 1, Step 2)",
    )
    _add_common_args(p5f)
    p5f.add_argument("--rank", type=int, default=None,
                     help="Single LoRA rank (omit to use --all-ranks)")
    p5f.add_argument("--all-ranks", action="store_true", dest="all_ranks",
                     default=False, help="Train r=1, 4, 8, 16 in sequence")
    p5f.add_argument("--epochs", type=int, default=3)
    p5f.add_argument("--lr", type=float, default=1e-4)
    p5f.add_argument("--base-model", type=str, default="Qwen/Qwen3-32B",
                     dest="base_model")
    p5f.add_argument(
        "--corrective-file", type=str,
        default="corrective_training_data.json",
        dest="corrective_file",
        help=("Which corrective set to train on. Use "
              "corrective_training_data_stratified.json for the "
              "proposal-aligned stratified set (50-100 per weak cell)."),
    )

    p5q = sub.add_parser(
        "run-stage5-query",
        help="Query the LoRA-fine-tuned model and re-score with PAI",
    )
    _add_common_args(p5q)
    p5q.add_argument("--rank", type=int, default=None)
    p5q.add_argument("--all-ranks", action="store_true", dest="all_ranks",
                     default=False)
    p5q.add_argument("--base-model", type=str, default="Qwen/Qwen3-32B",
                     dest="base_model")
    p5q.add_argument("--max-tokens-out", type=int, default=4096,
                     dest="max_tokens_out")
    p5q.add_argument(
        "--target", type=str, default="train",
        choices=["train", "heldout"],
        help="Case set to re-query (default: train, legacy behavior)",
    )
    p5q.add_argument(
        "--corrective-file", type=str,
        default="corrective_training_data.json",
        dest="corrective_file",
        help=("Which corrective file defines the train case set "
              "(must match the file used for fine-tuning)."),
    )

    # ── Stage 5 (Phase 1, Fix) — held-out split + Qwen baseline ──────────
    p5h = sub.add_parser(
        "run-stage5-heldout",
        help="Build held-out eval set disjoint from the corrective train set",
    )
    _add_common_args(p5h)
    p5h.add_argument(
        "--k", type=int, default=30,
        help="Number of next-lowest-PAI cases to include (default: 30)",
    )

    p5b = sub.add_parser(
        "run-stage5-baseline",
        help="Query bare Qwen3-32B (no LoRA) on train and/or heldout cases",
    )
    _add_common_args(p5b)
    p5b.add_argument(
        "--target", type=str, default="heldout",
        choices=["train", "heldout", "both"],
        help="Case set(s) to baseline (default: heldout)",
    )
    p5b.add_argument("--base-model", type=str, default="Qwen/Qwen3-32B",
                     dest="base_model")
    p5b.add_argument("--max-tokens-out", type=int, default=4096,
                     dest="max_tokens_out")
    p5b.add_argument(
        "--corrective-file", type=str,
        default="corrective_training_data.json",
        dest="corrective_file",
    )

    p5i = sub.add_parser(
        "run-stage5-interference",
        help="Cross-dimensional interference analysis from post-intervention runs",
    )
    _add_common_args(p5i)

    # ── Stage 6 (Phase 2 — classroom) ───────────────────────────────────
    p6s = sub.add_parser(
        "run-stage6-select-cases",
        help="Select ~30 cases for Phase 2 classroom evaluation (3 strata)",
    )
    _add_common_args(p6s)
    p6s.add_argument("--rank", type=int, required=True,
                     help="Rank of the post-intervention file to draw from")
    p6s.add_argument(
        "--target", type=str, default="heldout",
        choices=["train", "heldout"],
        help="Post-intervention target to use (default: heldout)",
    )
    p6s.add_argument("--n-per-stratum", type=int, default=10, dest="n_per_stratum",
                     help="Cases per stratum (default: 10 → 30 total)")
    p6s.add_argument("--control-threshold", type=float, default=0.4, dest="control_threshold")
    p6s.add_argument("--modest-threshold", type=float, default=0.05, dest="modest_threshold")

    p6f = sub.add_parser(
        "run-stage6-build-forms",
        help="Generate per-student evaluation forms (pre+post, blinded, randomized)",
    )
    _add_common_args(p6f)
    p6f.add_argument("--n-students", type=int, default=30, dest="n_students")
    p6f.add_argument("--cases-per-student", type=int, default=5, dest="cases_per_student")
    p6f.add_argument("--raters-per-case", type=int, default=5, dest="raters_per_case")
    p6f.add_argument("--seed", type=int, default=42)

    p6a = sub.add_parser(
        "run-stage6-analyze",
        help="Statistical analysis of collected student ratings vs PAI deltas",
    )
    _add_common_args(p6a)

    return parser


def main(argv: list | None = None) -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    # Stage-5-specific args that aren't part of PipelineConfig — strip them
    # before building the config so they don't leak into setattr() calls.
    stage5_only = {
        "n", "k", "rank", "all_ranks", "epochs", "lr", "base_model",
        "max_tokens_out", "target", "per_cell", "target_dp_only",
        "skip_query", "corrective_file",
        # stage 6
        "n_per_stratum", "control_threshold", "modest_threshold",
        "n_students", "cases_per_student", "raters_per_case", "seed",
    }
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("command", "file") and k not in stage5_only
        and v is not None and v is not False
    }

    from src.pipeline.config import PipelineConfig
    from src.pipeline.logging_utils import setup_logging

    cfg = PipelineConfig.from_env_and_args(cli_overrides)
    setup_logging(cfg.logs_dir)

    if args.command in ("run-stage1", "run-all"):
        from src.stage1_factorial.pipeline import run_stage1

        run_stage1(cfg)

    if args.command in ("run-stage2", "run-all"):
        from src.stage2_generation.pipeline import run_stage2

        run_stage2(cfg)

    if args.command == "repair":
        from src.stage2_generation.pipeline import repair_stage2

        repair_stage2(cfg)

    if args.command == "dedup":
        from src.stage2_generation.dedup import run_dedup

        run_dedup(cfg.output_dir or cfg.stage2_dir)

    if args.command == "regenerate":
        from src.stage2_generation.regenerate import run_regenerate

        run_regenerate(cfg)

    if args.command == "audit":
        from src.stage2_generation.audit import run_audit

        audit_file = args.file or str(cfg.stage2_dir / "cases_final.jsonl")
        run_audit(Path(audit_file))

    if args.command == "run-stage4":
        from src.pipeline.config import DEFAULTS
        from src.stage4_query.pipeline import run_stage4

        # Apply Stage-4 defaults unless the user explicitly overrode them.
        if "model" not in cli_overrides:
            cfg.model = DEFAULTS["stage4_model"]
        if "temperature" not in cli_overrides:
            cfg.temperature = DEFAULTS["stage4_temperature"]
        if "requests_per_minute" not in cli_overrides:
            cfg.requests_per_minute = DEFAULTS["stage4_requests_per_minute"]
        if "checkpoint_every" not in cli_overrides:
            cfg.checkpoint_every = DEFAULTS["stage4_checkpoint_every"]
        if "retries" not in cli_overrides:
            cfg.retries = 3

        run_stage4(cfg)

    if args.command == "run-stage5-score":
        from src.stage5_scoring.pipeline import run_stage5_scoring

        run_stage5_scoring(cfg)

    if args.command == "run-stage5-corrective":
        from src.stage5_finetune.corrective_data import run_build_corrective_data

        run_build_corrective_data(cfg, n=args.n)

    if args.command == "run-stage5-finetune":
        from src.stage5_finetune.tinker_train import DEFAULT_RANKS, run_finetune

        if args.all_ranks:
            ranks = list(DEFAULT_RANKS)
        elif args.rank is not None:
            ranks = [args.rank]
        else:
            parser.error("run-stage5-finetune requires --rank N or --all-ranks")
        run_finetune(
            cfg,
            ranks=ranks,
            epochs=args.epochs,
            lr=args.lr,
            base_model=args.base_model,
            corrective_file=args.corrective_file,
        )

    if args.command == "run-stage5-query":
        from src.stage5_finetune.tinker_query import run_query_post_intervention

        if args.all_ranks:
            ranks = None  # discover from disk
        elif args.rank is not None:
            ranks = [args.rank]
        else:
            parser.error("run-stage5-query requires --rank N or --all-ranks")
        run_query_post_intervention(
            cfg,
            ranks=ranks,
            target=args.target,
            base_model=args.base_model,
            max_tokens=args.max_tokens_out,
            corrective_file=args.corrective_file,
        )

    if args.command == "run-stage5-heldout":
        from src.stage5_finetune.eval_split import run_build_heldout

        run_build_heldout(cfg, k=args.k)

    if args.command == "run-stage5-baseline":
        from src.stage5_finetune.tinker_query import run_query_baseline

        targets = ["train", "heldout"] if args.target == "both" else [args.target]
        for t in targets:
            run_query_baseline(
                cfg,
                target=t,
                base_model=args.base_model,
                max_tokens=args.max_tokens_out,
                corrective_file=args.corrective_file,
            )

    if args.command == "run-stage5-interference":
        from src.stage5_finetune.interference import run_interference_analysis

        run_interference_analysis(cfg)

    if args.command == "run-stage5-weak-cells":
        from src.stage5_finetune.weak_cells import run_identify_weak_cells

        run_identify_weak_cells(cfg, k=args.k)

    if args.command == "run-stage5-corrective-stratified":
        from src.stage5_finetune.corrective_data import (
            run_build_corrective_data_stratified,
        )

        run_build_corrective_data_stratified(
            cfg,
            per_cell=args.per_cell,
            target_dp_only=args.target_dp_only,
        )

    if args.command == "run-stage5-per-dp-finetune":
        from src.stage5_finetune.per_dp_train import run_per_dp_finetune

        run_per_dp_finetune(
            cfg,
            rank=args.rank,
            epochs=args.epochs,
            lr=args.lr,
            base_model=args.base_model,
        )

    if args.command == "run-stage5-causal-interference":
        from src.stage5_finetune.causal_interference import run_causal_interference

        run_causal_interference(
            cfg,
            target=args.target,
            base_model=args.base_model,
            max_tokens=args.max_tokens_out,
            skip_query=args.skip_query,
        )

    if args.command == "run-stage6-select-cases":
        from src.stage6_classroom.case_selection import run_select_phase2_cases

        run_select_phase2_cases(
            cfg,
            rank=args.rank,
            target=args.target,
            n_per_stratum=args.n_per_stratum,
            control_threshold=args.control_threshold,
            modest_threshold=args.modest_threshold,
        )

    if args.command == "run-stage6-build-forms":
        from src.stage6_classroom.forms import run_build_phase2_forms

        run_build_phase2_forms(
            cfg,
            n_students=args.n_students,
            cases_per_student=args.cases_per_student,
            raters_per_case=args.raters_per_case,
            seed=args.seed,
        )

    if args.command == "run-stage6-analyze":
        from src.stage6_classroom.analysis import run_phase2_analysis

        run_phase2_analysis(cfg)


if __name__ == "__main__":
    main()
