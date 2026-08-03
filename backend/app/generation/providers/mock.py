"""Deterministic LLM stub.

Lets the whole chat path — prompt assembly, citation parsing, grounding checks,
streaming, the API contract — be tested without a network call, an API key, or
a bill. It deliberately *obeys the prompt contract*: it cites the passages it
was given and emits the insufficient-context sentinel when it has none, so the
tests exercise real parsing rather than a canned string.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from app.generation.prompts import INSUFFICIENT_CONTEXT
from app.generation.providers.base import LLMProvider, LLMResponse, StreamChunk

_SOURCE_RE = re.compile(r"SOURCE:\s*(\[[^\]]+\])")


class MockLLMProvider(LLMProvider):
    name = "mock"

    def __init__(self, model: str = "mock-echo", *, force_insufficient: bool = False) -> None:
        self.model = model
        self.force_insufficient = force_insufficient

    def _compose(self, user: str) -> str:
        if self.force_insufficient:
            return INSUFFICIENT_CONTEXT

        labels = _SOURCE_RE.findall(user)
        if not labels:
            return INSUFFICIENT_CONTEXT

        question = ""
        if "QUESTION:" in user:
            question = user.split("QUESTION:", 1)[1].split("\n", 1)[0].strip()

        return (
            f"Based on the retrieved passages, here is the answer to "
            f"'{question}'. {labels[0]}"
            + (
                f" Additional detail appears in another passage. {labels[1]}"
                if len(labels) > 1
                else ""
            )
        )

    async def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        text = self._compose(user)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=len(user.split()),
            output_tokens=len(text.split()),
            finish_reason="stop",
        )

    async def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        text = self._compose(user)
        for word in text.split(" "):
            yield StreamChunk(text=word + " ")
        yield StreamChunk(
            text="",
            done=True,
            usage=LLMResponse(
                text="",
                model=self.model,
                input_tokens=len(user.split()),
                output_tokens=len(text.split()),
            ),
        )
