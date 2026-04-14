"""Stage 1 orchestrator: read existing CSV or generate -> export JSONL + manifest."""

import logging
from pathlib import Path

from src.common.io_utils import read_csv
from src.common.schemas import FACTORIAL_COLUMNS
from src.pipeline.config import PipelineConfig
from src.pipeline.manifests import build_manifest, save_manifest
from src.stage1_factorial.exporter import export_csv, export_jsonl_from_dicts
from src.stage1_factorial.generator import generate_factorial

logger = logging.getLogger("pipeline")


def run_stage1(cfg: PipelineConfig) -> None:
    output_dir = cfg.output_dir or cfg.stage1_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "factorial_sample.csv"

    if csv_path.exists():
        logger.info("Stage 1: existing CSV found at %s — importing …", csv_path)
        rows = read_csv(csv_path)
        method = "imported_csv"
    else:
        logger.info("Stage 1: no existing CSV — generating factorial sample …")
        factorial_rows = generate_factorial()
        rows = [r.to_dict() for r in factorial_rows]
        export_csv(factorial_rows, output_dir)
        logger.info("Stage 1: CSV  -> %s", csv_path)
        method = "full_factorial"

    logger.info("Stage 1: %d rows", len(rows))

    jsonl_path = export_jsonl_from_dicts(rows, output_dir)
    logger.info("Stage 1: JSONL -> %s", jsonl_path)

    manifest = build_manifest(
        stage="stage1_factorial",
        params={
            "method": method,
            "source": str(csv_path),
            "variables": [
                "bloom",
                "knowledge_state",
                "learning_stage",
                "learning_context",
                "subject",
            ],
        },
        row_count=len(rows),
    )
    manifest_path = save_manifest(manifest, output_dir)
    logger.info("Stage 1: manifest -> %s", manifest_path)
    logger.info("Stage 1: complete.")
