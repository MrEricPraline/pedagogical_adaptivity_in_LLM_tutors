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
        "--requests-per-minute", type=int, default=None, dest="requests_per_minute"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=None, dest="checkpoint_every"
    )
    parser.add_argument("--input-path", type=str, default=None, dest="input_path")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir")


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

    return parser


def main(argv: list | None = None) -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("command", "file") and v is not None and v is not False
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


if __name__ == "__main__":
    main()
