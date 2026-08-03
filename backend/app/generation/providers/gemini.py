"""Google Gemini provider.

Gemini is the configured default (see PRD §8.3, which names it as the sanctioned
alternate to Claude). Strong Indic-language handling makes it a good fit for the
Hindi and Hinglish registers this product answers in.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.core.logging import get_logger
from app.generation.providers.base import LLMProvider, LLMResponse, StreamChunk

log = get_logger(__name__)

# Reasoning depth maps onto Gemini's thinking budget (tokens).
_EFFORT_THINKING_BUDGET = {"low": 0, "medium": 1024, "high": 8192}


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.llm.resolved_model
        self._api_key = api_key or settings.google_api_key
        self._client = None

        if not self._api_key:
            raise LLMUnavailableError(
                "GOOGLE_API_KEY is not set. Run `make check-llm` to verify your key."
            )

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _config(self, system: str, max_tokens: int | None, temperature: float | None):
        from google.genai import types

        cfg = settings.llm
        kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens or cfg.max_tokens,
            "temperature": cfg.temperature if temperature is None else temperature,
        }

        # Thinking costs latency and output tokens. For grounded QA over
        # retrieved passages the reasoning is shallow, so it is off by default
        # (NFR-2) and opt-in via LLM__THINKING.
        budget = _EFFORT_THINKING_BUDGET.get(cfg.effort, 0) if cfg.thinking else 0
        # Older google-genai releases have no ThinkingConfig; the request is
        # still valid without it.
        with contextlib.suppress(Exception):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _usage(response) -> tuple[int, int]:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return 0, 0
        return (
            int(getattr(meta, "prompt_token_count", 0) or 0),
            int(getattr(meta, "candidates_token_count", 0) or 0),
        )

    async def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config=self._config(system, max_tokens, temperature),
            )
        except Exception as exc:
            raise LLMUnavailableError(_friendly_error(exc)) from exc

        input_tokens, output_tokens = self._usage(response)
        finish = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "") or None

        return LLMResponse(
            text=(response.text or "").strip(),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish,
        )

    async def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        input_tokens = output_tokens = 0

        try:
            stream = await client.aio.models.generate_content_stream(
                model=self.model,
                contents=user,
                config=self._config(system, max_tokens, temperature),
            )
            async for event in stream:
                piece = getattr(event, "text", None)
                if piece:
                    yield StreamChunk(text=piece)
                got_in, got_out = self._usage(event)
                # Usage arrives cumulatively; keep the latest non-zero reading.
                input_tokens = got_in or input_tokens
                output_tokens = got_out or output_tokens
        except Exception as exc:
            raise LLMUnavailableError(_friendly_error(exc)) from exc

        yield StreamChunk(
            text="",
            done=True,
            usage=LLMResponse(
                text="",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )


def _friendly_error(exc: Exception) -> str:
    """Turn vendor errors into something a user can act on (NFR-5)."""
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "The language model is rate-limited or out of quota. "
            "Check your Gemini plan, or run `make check-llm` to find a usable model."
        )
    if "404" in text and "model" in text.lower():
        return (
            "The configured model is not available to this API key. "
            "Run `make check-llm` to list the models you can reach."
        )
    if "API_KEY" in text.upper() or "401" in text or "403" in text:
        return "The Gemini API key was rejected. Check GOOGLE_API_KEY in .env."
    return f"Gemini request failed: {text[:200]}"
