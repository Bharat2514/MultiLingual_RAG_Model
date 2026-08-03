"""LLM provider interface (§8.3, NFR-9).

The PRD keeps the LLM swappable so the system can move between vendors — or to
a local model — without touching the pipeline. Every provider takes the same
system/user prompt pair and returns the same envelope, so the generation
service never learns which vendor is behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class StreamChunk:
    """One increment of a streamed answer (FR-25)."""

    text: str
    done: bool = False
    usage: LLMResponse | None = None


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    async def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield partial text as it is produced, then a final chunk carrying usage."""

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Per-request cost for the NFR-4 budget tracking."""
        from app.core.config import settings

        pricing = settings.llm
        return round(
            input_tokens / 1_000_000 * pricing.input_cost_per_mtok
            + output_tokens / 1_000_000 * pricing.output_cost_per_mtok,
            6,
        )
