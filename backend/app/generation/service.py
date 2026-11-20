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
from app.generation.citations import (
    Citation,
    canonicalize_citations,
    extract_citations,
    is_grounded,
)
from app.generation.conversation import Resolution, Turn, history_block, resolve
from app.generation.providers.base import LLMProvider
from app.generation.query import AnalyzedQuery, analyze_query
from app.intelligence.claims import ClaimReport, verify_answer
from app.intelligence.corpus import should_redact
from app.intelligence.injection import is_unsafe
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
        elif choice == "ollama":
            from app.generation.providers.ollama import OllamaProvider

            _provider = OllamaProvider()
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
    #: Per-sentence verification of the answer against its cited passages.
    claims: ClaimReport | None = None
    #: The standalone question actually searched, when a follow-up was rewritten.
    resolution: Resolution | None = None
    #: Retrieved passages withheld from the model for carrying injected instructions.
    filtered_chunks: int = 0
    #: Why the answer is "not found", and what came closest.
    abstention: dict | None = None
    cached: bool = False

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in (self.retrieval.chunks if self.retrieval else [])]


@dataclass(slots=True)
class ChatOptions:
    """Per-request knobs beyond retrieval depth."""

    history: list[Turn] = field(default_factory=list)
    #: "default" or "simple" (plain-language mode).
    style: str = "default"
    #: Number of ready documents in scope — feeds the "not found" explanation.
    searched_documents: int | None = None


@dataclass(slots=True)
class _Prepared:
    analyzed: AnalyzedQuery
    retrieval: RetrievalResult
    timings: dict[str, float]
    resolution: Resolution
    filtered: int
    nearest: str | None


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
        options: ChatOptions,
    ) -> _Prepared:
        timings: dict[str, float] = {}

        # Follow-ups are resolved against the conversation *before* retrieval —
        # the search has to see what the user means, not "aur uska?".
        started = time.perf_counter()
        resolution = await resolve(query, options.history, self.provider)
        if resolution.rewritten:
            timings["rewrite"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        analyzed = analyze_query(resolution.query)
        if resolution.rewritten:
            # Reply in the register the user just used, not the rewrite's.
            own = analyze_query(query)
            if own.language != Language.UNKNOWN:
                analyzed.language = own.language
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

        nearest = retrieval.chunks[0].citation_label if retrieval.chunks else None

        # A passage that tries to instruct the model is withheld, even if it was
        # indexed before the ingestion-time scan existed.
        filtered = 0
        if settings.trust.filter_injected_context and retrieval.chunks:
            safe = [c for c in retrieval.chunks if not is_unsafe(c.text)]
            filtered = len(retrieval.chunks) - len(safe)
            if filtered:
                log.warning("context_passages_withheld", count=filtered)
                retrieval.chunks = safe

        return _Prepared(analyzed, retrieval, timings, resolution, filtered, nearest)

    def _abstention(self, prep: _Prepared, options: ChatOptions, reason: str) -> dict:
        searched = (
            options.searched_documents
            if options.searched_documents is not None
            else len({c.doc_id for c in prep.retrieval.chunks})
        )
        return {
            "reason": reason,
            "searched_documents": searched,
            "nearest_source": prep.nearest,
            "withheld_passages": prep.filtered,
            "hint": prompts.abstention_hint(
                prep.analyzed.language,
                searched=searched,
                nearest=prep.nearest,
                filtered=prep.filtered,
            ),
        }

    def _not_found(self, prep: _Prepared, options: ChatOptions, reason: str) -> ChatResult:
        log.info(
            "chat_not_found",
            language=str(prep.analyzed.language),
            considered=prep.retrieval.candidates_considered,
            reason=reason,
        )
        return ChatResult(
            answer=prompts.not_found_message(prep.analyzed.language),
            language_detected=prep.analyzed.language,
            grounded=False,
            retrieval=prep.retrieval,
            query=prep.analyzed,
            model=self.provider.model,
            latency_ms=prep.timings,
            error_code="NO_RELEVANT_CONTEXT",
            resolution=prep.resolution,
            filtered_chunks=prep.filtered,
            abstention=self._abstention(prep, options, reason),
        )

    def _prompts(self, prep: _Prepared, options: ChatOptions) -> tuple[str, str]:
        analyzed = prep.analyzed
        system = prompts.build_system_prompt(
            analyzed.language, analyzed.intent, style=options.style
        )
        user = prompts.build_user_prompt(
            analyzed.raw,
            prep.retrieval.chunks,
            analyzed.intent,
            history=history_block(options.history) if prep.resolution.rewritten else "",
            redact_pii=should_redact(self.provider),
        )
        return system, user

    def _finalize(
        self,
        raw_answer: str,
        prep: _Prepared,
        options: ChatOptions,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> ChatResult:
        analyzed, retrieval, timings = prep.analyzed, prep.retrieval, prep.timings

        if prompts.is_insufficient(raw_answer):
            result = self._not_found(prep, options, "model_declined")
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.cost_usd = self.provider.estimate_cost_usd(input_tokens, output_tokens)
            return result

        citations = extract_citations(raw_answer, retrieval.chunks)
        grounded = is_grounded(raw_answer, citations, retrieval.chunks)
        # One citation form downstream, whatever mix the model wrote: `[S2]`.
        answer = canonicalize_citations(raw_answer, retrieval.chunks)

        if not grounded:
            # Not an error — but the UI must be able to show that this answer
            # could not be tied back to a source (§11.4 post-check).
            log.warning("answer_not_grounded", language=str(analyzed.language))

        claims: ClaimReport | None = None
        if settings.trust.verify_claims:
            started = time.perf_counter()
            try:
                claims = verify_answer(answer, retrieval.chunks)
            except Exception:
                log.warning("claim_verification_failed", exc_info=True)
            timings["verify"] = round((time.perf_counter() - started) * 1000, 2)

        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 2)

        return ChatResult(
            answer=answer,
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
            claims=claims,
            resolution=prep.resolution,
            filtered_chunks=prep.filtered,
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
        options: ChatOptions | None = None,
    ) -> ChatResult:
        options = options or ChatOptions()
        prep = await self._prepare(
            query, scope, top_k=top_k, top_n=top_n, mode=mode, options=options
        )

        if prep.retrieval.is_empty:
            prep.timings["total"] = round(sum(prep.timings.values()), 2)
            reason = "no_documents" if options.searched_documents == 0 else "no_relevant_passages"
            return self._not_found(prep, options, reason)

        system, user = self._prompts(prep, options)

        started = time.perf_counter()
        try:
            response = await self.provider.generate(system, user)
        except LLMUnavailableError:
            # NFR-5: surface the retrieved passages rather than nothing.
            prep.timings["generation"] = round((time.perf_counter() - started) * 1000, 2)
            log.warning("llm_unavailable_returning_retrieval_only", exc_info=True)
            raise
        prep.timings["generation"] = round((time.perf_counter() - started) * 1000, 2)

        return self._finalize(
            response.text,
            prep,
            options,
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
        options: ChatOptions | None = None,
    ) -> AsyncIterator[tuple[str, ChatResult | None]]:
        """Yield `(delta, None)` while streaming, then `("", ChatResult)`.

        Citations and the grounding verdict need the complete answer, so they
        arrive in the final event rather than mid-stream.
        """
        options = options or ChatOptions()
        prep = await self._prepare(
            query, scope, top_k=top_k, top_n=top_n, mode=mode, options=options
        )

        if prep.retrieval.is_empty:
            prep.timings["total"] = round(sum(prep.timings.values()), 2)
            reason = "no_documents" if options.searched_documents == 0 else "no_relevant_passages"
            result = self._not_found(prep, options, reason)
            yield result.answer, None
            yield "", result
            return

        system, user = self._prompts(prep, options)

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

        prep.timings["generation"] = round((time.perf_counter() - started) * 1000, 2)
        raw_answer = "".join(parts)

        result = self._finalize(
            raw_answer,
            prep,
            options,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # The sentinel never reached the client, so send the replacement text.
        if result.error_code == "NO_RELEVANT_CONTEXT":
            yield result.answer, None

        yield "", result
