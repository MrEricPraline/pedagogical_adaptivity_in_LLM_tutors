"""Gemini 3.1 Pro Preview client for Stage 4 (target-model querying).

Wraps the official `google-genai` SDK with rate limiting, retries, and
structured-output configuration (response_mime_type + response_schema +
thinking_level).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("pipeline")


class GeminiProvider:
    """Thin wrapper around `google.genai.Client` for structured JSON output."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
        retries: int,
        requests_per_minute: int,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it or add it to your .env file. "
                "Stage 4 requires a Gemini API key with billing enabled "
                "(Gemini 3.1 Pro Preview is a paid model)."
            )

        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for Stage 4. "
                "Install it with: pip install 'google-genai>=1.51.0'"
            ) from exc

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.retries = max(1, retries)
        self._min_interval = 60.0 / max(requests_per_minute, 1)
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

    def _build_config(self, system_instruction: str, response_schema: Dict[str, Any]):
        from google.genai import types as genai_types  # type: ignore

        thinking_config = None
        try:
            thinking_config = genai_types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        except (TypeError, AttributeError):
            # Older SDKs may not expose `thinking_level` (or ThinkingConfig at
            # all). Fall back gracefully so the run does not crash.
            thinking_config = None

        kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if thinking_config is not None:
            kwargs["thinking_config"] = thinking_config

        return genai_types.GenerateContentConfig(**kwargs)

    def generate_json(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> str:
        """Call Gemini and return the raw JSON text. Raises on terminal failure."""
        config = self._build_config(system_instruction, response_schema)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                self._wait_for_rate_limit()
                self._last_request_time = time.time()

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )

                text = self._extract_text(response)
                if text:
                    return text

                raise RuntimeError(
                    "Empty response from Gemini (no text in candidates)."
                )

            except Exception as exc:  # noqa: BLE001 — we classify and back off
                last_error = exc
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self.retries, exc, wait,
                )
                if attempt < self.retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"Gemini call failed after {self.retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from a Gemini response object across SDK versions."""
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text
        return ""
