"""Chat orchestration: analyse → retrieve → ground → generate (§8.1).

The whole grounding contract is enforced here, not left to the model:

* nothing retrieved  → answer "not found" **without calling the LLM at all**
  (FR-23; also saves the token spend on a question we know we cannot answer);
* sentinel returned  → substitute the localised not-found message;
* answer produced    → resolve citations against the chunks we actually sent,
  and mark the answer ungrounded if none of them check out (§11.4).
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.core.logging import get_logger
from app.db.models import Language
from app.generation import prompts
from app.generation.citations import Citation, extract_citations, is_grounded
from app.generation.providers.base import LLMProvider
from app.generation.query import AnalyzedQuery, analyze_query
from app.retrieval.service import RetrievalService
from app.retrieval.types import RetrievalResult, SearchMode
from app.retrieval.vectorstore.base import SearchFilter

log = get_logger(__name__)

_provider: LLMProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> LLMProvider:
    """Process-wide provider singleton, selected from config (NFR-9)."""
    global _provider
    if _provider is not None:
        return _provider

    with _provider_lock:
        if _provider is not None:
            return _provider

        choice = settings.llm.provider
        if choice == "mock":
            from app.generation.providers.mock import MockLLMProvider

            _provider = MockLLMProvider()
        elif choice == "anthropic":
            from app.generation.providers.anthropic_provider import AnthropicProvider

            _provider = AnthropicProvider()
        else:
            from app.generation.providers.gemini import GeminiProvider

            _provider = GeminiProvider()

        log.info("llm_provider_selected", provider=_provider.name, model=_provider.model)
        return _provider


def reset_provider() -> None:
    global _provider
    _provider = None


@dataclass(slots=True)
class ChatResult:
    answer: str
    language_detected: Language
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    query: AnalyzedQuery | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    error_code: str | None = None

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in (self.retrieval.chunks if self.retrieval else [])]


class ChatService:
    def __init__(
        self, *, retrieval: RetrievalService | None = None, provider: LLMProvider | None = None
    ) -> None:
        self._retrieval = retrieval
        self._provider = provider

    @property
    def retrieval(self) -> RetrievalService:
        return self._retrieval or RetrievalService()

    @property
    def provider(self) -> LLMProvider:
        return self._provider or get_provider()

    # --- shared pre-flight ---------------------------------------------------

    async def _prepare(
        self,
        query: str,
        scope: SearchFilter,
        *,
        top_k: int | None,
        top_n: int | None,
        mode: SearchMode,
    ) -> tuple[AnalyzedQuery, RetrievalResult, dict[str, float]]:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        analyzed = analyze_query(query)
        timings["analyze"] = round((time.perf_counter() - started) * 1000, 2)

        retrieval = await self.retrieval.retrieve(
            analyzed.raw,
            scope,
            queries=analyzed.variants,
            top_k=top_k,
            top_n=top_n,
            mode=mode,
            # "Summarise this document" cannot be answered by ranking passages
            # against it — see app.generation.intent.
            coverage=analyzed.needs_coverage,
        )
        timings.update(retrieval.timings_ms)
        return analyzed, retrieval, timings

    def _not_found(
        self, analyzed: AnalyzedQuery, retrieval: RetrievalResult, timings: dict[str, float]
    ) -> ChatResult:
        log.info(
            "chat_not_found",
            language=str(analyzed.language),
            considered=retrieval.candidates_considered,
        )
        return ChatResult(
            answer=prompts.not_found_message(analyzed.language),
            language_detected=analyzed.language,
            grounded=False,
            retrieval=retrieval,
            query=analyzed,
            model=self.provider.model,
            latency_ms=timings,
            error_code="NO_RELEVANT_CONTEXT",
        )

    def _finalize(
        self,
        raw_answer: str,
        analyzed: AnalyzedQuery,
        retrieval: RetrievalResult,
        timings: dict[str, float],
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> ChatResult:
        if prompts.is_insufficient(raw_answer):
            result = self._not_found(analyzed, retrieval, timings)
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.cost_usd = self.provider.estimate_cost_usd(input_tokens, output_tokens)
            return result

        citations = extract_citations(raw_answer, retrieval.chunks)
        grounded = is_grounded(raw_answer, citations, retrieval.chunks)

        if not grounded:
            # Not an error — but the UI must be able to show that this answer
            # could not be tied back to a source (§11.4 post-check).
            log.warning("answer_not_grounded", language=str(analyzed.language))

        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 2)

        return ChatResult(
            answer=raw_answer,
            language_detected=analyzed.language,
            grounded=grounded,
            citations=citations,
            retrieval=retrieval,
            query=analyzed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.provider.estimate_cost_usd(input_tokens, output_tokens),
            model=self.provider.model,
            latency_ms=timings,
        )

    # --- non-streaming -------------------------------------------------------

    async def answer(
        self,
        query: str,
        scope: SearchFilter,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> ChatResult:
        analyzed, retrieval, timings = await self._prepare(
            query, scope, top_k=top_k, top_n=top_n, mode=mode
        )

        if retrieval.is_empty:
            timings["total"] = round(sum(timings.values()), 2)
            return self._not_found(analyzed, retrieval, timings)

        system = prompts.build_system_prompt(analyzed.language, analyzed.intent)
        user = prompts.build_user_prompt(analyzed.raw, retrieval.chunks, analyzed.intent)

        started = time.perf_counter()
        try:
            response = await self.provider.generate(system, user)
        except LLMUnavailableError:
            # NFR-5: surface the retrieved passages rather than nothing.
            timings["generation"] = round((time.perf_counter() - started) * 1000, 2)
            log.warning("llm_unavailable_returning_retrieval_only", exc_info=True)
            raise
        timings["generation"] = round((time.perf_counter() - started) * 1000, 2)

        return self._finalize(
            response.text,
            analyzed,
            retrieval,
            timings,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    # --- streaming (FR-25) ---------------------------------------------------

    async def answer_stream(
        self,
        query: str,
        scope: SearchFilter,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> AsyncIterator[tuple[str, ChatResult | None]]:
        """Yield `(delta, None)` while streaming, then `("", ChatResult)`.

        Citations and the grounding verdict need the complete answer, so they
        arrive in the final event rather than mid-stream.
        """
        analyzed, retrieval, timings = await self._prepare(
            query, scope, top_k=top_k, top_n=top_n, mode=mode
        )

        if retrieval.is_empty:
            timings["total"] = round(sum(timings.values()), 2)
            result = self._not_found(analyzed, retrieval, timings)
            yield result.answer, None
            yield "", result
            return

        system = prompts.build_system_prompt(analyzed.language, analyzed.intent)
        user = prompts.build_user_prompt(analyzed.raw, retrieval.chunks, analyzed.intent)

        started = time.perf_counter()
        parts: list[str] = []
        input_tokens = output_tokens = 0
        buffering_sentinel = True

        async for chunk in self.provider.stream(system, user):
            if chunk.done:
                if chunk.usage:
                    input_tokens = chunk.usage.input_tokens
                    output_tokens = chunk.usage.output_tokens
                break

            parts.append(chunk.text)

            # Hold back the opening tokens until we know this is not the
            # insufficient-context sentinel — the user must never see the raw
            # marker flash by before it is replaced.
            if buffering_sentinel:
                joined = "".join(parts)
                stripped = joined.strip().upper()
                if len(joined) < len(prompts.INSUFFICIENT_CONTEXT) and (
                    prompts.INSUFFICIENT_CONTEXT.startswith(stripped)
                ):
                    continue
                buffering_sentinel = False
                if prompts.is_insufficient(joined):
                    break
                yield joined, None
                continue

            yield chunk.text, None

        timings["generation"] = round((time.perf_counter() - started) * 1000, 2)
        raw_answer = "".join(parts)

        result = self._finalize(
            raw_answer,
            analyzed,
            retrieval,
            timings,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # The sentinel never reached the client, so send the replacement text.
        if result.error_code == "NO_RELEVANT_CONTEXT":
            yield result.answer, None

        yield "", result
