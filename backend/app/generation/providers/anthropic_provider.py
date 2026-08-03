"""Anthropic Claude provider.

Kept fully implemented alongside Gemini so switching vendors is a `.env` change
(`LLM__PROVIDER=anthropic` plus `ANTHROPIC_API_KEY`) rather than a code change,
per §8.3 and NFR-9.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.core.logging import get_logger
from app.generation.providers.base import LLMProvider, LLMResponse, StreamChunk

log = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.llm.resolved_model
        self._api_key = api_key or settings.anthropic_api_key
        self._client = None

        if not self._api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=settings.llm.timeout_seconds,
                max_retries=settings.llm.max_retries,
            )
        return self._client

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
            message = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.llm.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMUnavailableError(f"Claude request failed: {str(exc)[:200]}") from exc

        # Safety classifiers can decline with a 200 and stop_reason "refusal";
        # `content` is then empty, so this must be checked before reading it.
        if getattr(message, "stop_reason", None) == "refusal":
            raise LLMUnavailableError("The model declined to answer this request.")

        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text.strip(),
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            finish_reason=message.stop_reason,
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
        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or settings.llm.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(text=text)
                final = await stream.get_final_message()
        except Exception as exc:
            raise LLMUnavailableError(f"Claude stream failed: {str(exc)[:200]}") from exc

        yield StreamChunk(
            text="",
            done=True,
            usage=LLMResponse(
                text="",
                model=self.model,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                finish_reason=final.stop_reason,
            ),
        )
