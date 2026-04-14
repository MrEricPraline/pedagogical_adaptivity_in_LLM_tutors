"""Centralised configuration with defaults, env-var overrides, and CLI merge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULTS = {
    "model": "grok-3-fast",
    "base_url": "https://api.x.ai/v1",
    "temperature": 0.8,
    "max_tokens": 300,
    "retries": 3,
    "requests_per_minute": 30,
    "checkpoint_every": 50,
    "word_count_min": 80,
    "word_count_max": 120,
}


@dataclass
class PipelineConfig:
    # xAI / generation
    xai_api_key: str = ""
    model: str = DEFAULTS["model"]
    base_url: str = DEFAULTS["base_url"]
    temperature: float = DEFAULTS["temperature"]
    max_tokens: int = DEFAULTS["max_tokens"]

    # execution
    start: int | None = None
    end: int | None = None
    resume: bool = False
    retries: int = DEFAULTS["retries"]
    requests_per_minute: int = DEFAULTS["requests_per_minute"]
    checkpoint_every: int = DEFAULTS["checkpoint_every"]

    # validation
    word_count_min: int = DEFAULTS["word_count_min"]
    word_count_max: int = DEFAULTS["word_count_max"]

    # paths
    stage1_dir: Path = field(default_factory=lambda: DATA_DIR / "stage1")
    stage2_dir: Path = field(default_factory=lambda: DATA_DIR / "stage2")
    logs_dir: Path = field(default_factory=lambda: DATA_DIR / "logs")
    input_path: Path | None = None
    output_dir: Path | None = None

    @classmethod
    def from_env_and_args(cls, cli_args: dict | None = None) -> PipelineConfig:
        """Build config by layering: defaults -> env vars -> CLI args."""
        cfg = cls()

        cfg.xai_api_key = os.getenv("XAI_API_KEY", "")
        cfg.model = os.getenv("MODEL", cfg.model)
        cfg.base_url = os.getenv("XAI_BASE_URL", cfg.base_url)
        cfg.temperature = float(os.getenv("TEMPERATURE", str(cfg.temperature)))
        cfg.max_tokens = int(os.getenv("MAX_TOKENS", str(cfg.max_tokens)))
        cfg.retries = int(os.getenv("RETRIES", str(cfg.retries)))
        cfg.requests_per_minute = int(
            os.getenv("REQUESTS_PER_MINUTE", str(cfg.requests_per_minute))
        )
        cfg.checkpoint_every = int(
            os.getenv("CHECKPOINT_EVERY", str(cfg.checkpoint_every))
        )

        if cli_args:
            for key, value in cli_args.items():
                if value is None:
                    continue
                if hasattr(cfg, key):
                    current = getattr(cfg, key)
                    if isinstance(current, Path):
                        setattr(cfg, key, Path(value))
                    else:
                        setattr(cfg, key, type(current)(value) if current is not None else value)

        if cfg.input_path is None:
            cfg.input_path = cfg.stage1_dir / "factorial_sample.csv"

        return cfg
