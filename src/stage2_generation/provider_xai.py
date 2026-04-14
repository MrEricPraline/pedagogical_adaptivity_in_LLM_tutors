"""xAI / Grok client with retry and rate limiting (OpenAI-compatible)."""

from __future__ import annotations

import logging
import time
from typing import List, Dict

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from src.pipeline.config import PipelineConfig

logger = logging.getLogger("pipeline")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class XAIProvider:
    """Thin wrapper around the OpenAI client targeting xAI endpoints."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if not cfg.xai_api_key:
            raise ValueError(
                "XAI_API_KEY is not set. Export it or add it to your .env file."
            )
        self.client = OpenAI(
            api_key=cfg.xai_api_key,
            base_url=cfg.base_url,
        )
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.retries = cfg.retries
        self._min_interval = 60.0 / cfg.requests_per_minute
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Call the chat completions endpoint with retry logic.

        Returns the generated text or raises after exhausting retries.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                self._wait_for_rate_limit()
                self._last_request_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                return (content or "").strip()

            except RateLimitError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Rate-limited (attempt %d/%d). Waiting %ds …",
                    attempt, self.retries, wait,
                )
                time.sleep(wait)

            except (APIError, APITimeoutError) as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status and status not in RETRYABLE_STATUS_CODES:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "API error %s (attempt %d/%d). Waiting %ds …",
                    exc, attempt, self.retries, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed after {self.retries} retries. Last error: {last_error}"
        )
