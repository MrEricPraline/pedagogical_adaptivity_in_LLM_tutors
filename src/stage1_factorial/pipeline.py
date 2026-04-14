"""Stage 1 orchestrator: generate factorial sample -> export -> manifest."""

import logging

from src.pipeline.config import PipelineConfig
from src.pipeline.manifests import build_manifest, save_manifest
from src.stage1_factorial.exporter import export_csv, export_jsonl
from src.stage1_factorial.generator import generate_factorial

logger = logging.getLogger("pipeline")


def run_stage1(cfg: PipelineConfig) -> None:
    output_dir = cfg.output_dir or cfg.stage1_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stage 1: generating factorial sample …")
    rows = generate_factorial()
    logger.info("Stage 1: generated %d rows", len(rows))

    csv_path = export_csv(rows, output_dir)
    logger.info("Stage 1: CSV  -> %s", csv_path)

    jsonl_path = export_jsonl(rows, output_dir)
    logger.info("Stage 1: JSONL -> %s", jsonl_path)

    manifest = build_manifest(
        stage="stage1_factorial",
        params={
            "method": "full_factorial",
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
