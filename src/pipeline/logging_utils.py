"""Logging setup: dual output to console and rotating log file."""

import logging
from pathlib import Path

_CONFIGURED = False


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED

    logger = logging.getLogger("pipeline")

    if _CONFIGURED:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _CONFIGURED = True
    return logger
