"""The chat endpoint (§9.2, §9.3, FR-24, FR-25)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_id
from app.core.config import settings
from app.core.context import get_request_id
from app.core.logging import get_logger
from app.db.models import QueryLog
from app.db.session import get_db
from app.generation.cache import answer_cache
from app.generation.conversation import Turn
from app.generation.service import ChatOptions, ChatResult, ChatService
from app.retrieval.vectorstore.base import SearchFilter
from app.schemas.chat import (
    AbstentionOut,
    ChatRequest,
    ChatResponse,
    CitationOut,
    ClaimOut,
    ClaimSummary,
)
from app.schemas.common import LatencyBreakdown, TokenUsage
from app.services.answers import build_snapshot
from app.services.scope import Scope, resolve_scope

router = APIRouter(tags=["chat"])
log = get_logger(__name__)


def _to_response(
    result: ChatResult,
    session_id: str,
    request_id: str,
    *,
    answer_id: str | None = None,
    scope: Scope | None = None,
) -> ChatResponse:
    retrieval = result.retrieval
    timings = result.latency_ms
    claims = result.claims
    return ChatResponse(
        request_id=request_id,
        session_id=session_id,
        answer=result.answer,
        language_detected=result.language_detected,
        grounded=result.grounded,
        citations=[
            CitationOut(
                doc_id=c.doc_id,
                filename=c.filename,
                page=c.page,
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                snippet=c.snippet,
                score=c.score,
                context_index=c.context_index,
            )
            for c in result.citations
        ],
        usage=TokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        ),
        latency_ms=LatencyBreakdown(
            retrieval=timings.get("retrieval", 0.0) + timings.get("embed", 0.0),
            rerank=timings.get("rerank", 0.0),
            generation=timings.get("generation", 0.0),
            total=timings.get("total", sum(timings.values())),
        ),
        model=result.model,
        retrieval_mode=retrieval.mode if retrieval else "hybrid",
        reranked=retrieval.reranked if retrieval else False,
        candidates_considered=retrieval.candidates_considered if retrieval else 0,
        query_variants=result.query.variants if result.query else [],
        answer_id=answer_id,
        claims=[ClaimOut(**c.to_dict()) for c in claims.claims] if claims else [],
        claim_summary=(
            ClaimSummary(counts=claims.counts, support_ratio=claims.support_ratio)
            if claims and claims.claims
            else None
        ),
        rewritten_query=(
            result.resolution.query if result.resolution and result.resolution.rewritten else None
        ),
        abstention=AbstentionOut(**result.abstention) if result.abstention else None,
        withheld_passages=result.filtered_chunks,
        excluded_documents=scope.excluded if scope else [],
        cached=result.cached,
    )


async def _record(
    db: AsyncSession, request_id: str, session_id: str, result: ChatResult
) -> str | None:
    """Persist the query trace (NFR-4, NFR-7) and the answer snapshot.

    Returns the snapshot's `answer_id`, or None if persistence failed — a lost
    trace must never fail the answer the user is waiting for.
    """
    timings = result.latency_ms
    try:
        db.add(
            QueryLog(
                request_id=request_id,
                session_id=session_id,
                query=result.query.raw if result.query else "",
                language_detected=str(result.language_detected),
                normalized_query=result.query.transliterated if result.query else None,
                doc_ids=None,
                retrieved_chunk_ids=result.chunk_ids,
                grounded=result.grounded,
                answer_chars=len(result.answer),
                citation_count=len(result.citations),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                latency_retrieval_ms=timings.get("retrieval", 0.0) + timings.get("embed", 0.0),
                latency_rerank_ms=timings.get("rerank", 0.0),
                latency_generation_ms=timings.get("generation", 0.0),
                latency_total_ms=timings.get("total", 0.0),
                error_code=result.error_code,
            )
        )
        snapshot = build_snapshot(request_id, session_id, result)
        db.add(snapshot)
        await db.commit()
        return snapshot.answer_id
    except Exception:
        await db.rollback()
        log.warning("query_log_write_failed", exc_info=True)
        return None


def _cache_key(payload: ChatRequest, session_id: str, scope: Scope) -> str | None:
    # A follow-up's meaning depends on the conversation, so it is never cached.
    if not settings.cache.enabled or not payload.use_cache or payload.history:
        return None
    return answer_cache.key(
        session=session_id,
        query=payload.query,
        docs=tuple(sorted(scope.doc_ids)) if scope.doc_ids is not None else None,
        corpus=scope.fingerprint,
        mode=str(payload.mode),
        top_k=payload.top_k,
        top_n=payload.top_n,
        style=payload.style,
        as_of=payload.as_of.isoformat() if payload.as_of else None,
        superseded=payload.include_superseded,
    )


def _options(payload: ChatRequest, scope: Scope) -> ChatOptions:
    return ChatOptions(
        history=[Turn(t.role, t.content) for t in payload.history],
        style=payload.style,
        searched_documents=scope.ready_count,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about your documents",
    response_description="A grounded answer with inline citations.",
)
async def chat(
    payload: ChatRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
):
    """Answer from the caller's documents only.

    When nothing relevant is retrieved the endpoint returns an explicit
    "not found in your documents" answer with `grounded: false` — a 200, not an
    error, because it is a valid and *desirable* outcome (FR-23, US4).
    """
    request_id = get_request_id()
    service = ChatService()

    scope = await resolve_scope(
        db,
        session_id,
        payload.doc_ids,
        as_of=payload.as_of,
        include_superseded=payload.include_superseded,
    )
    search_filter = SearchFilter(session_id=session_id, doc_ids=scope.doc_ids)
    top_k = payload.top_k or settings.retrieval.top_k
    top_n = payload.top_n or settings.retrieval.top_n
    cache_key = _cache_key(payload, session_id, scope)

    if payload.stream:
        return StreamingResponse(
            _sse(
                service,
                payload,
                search_filter,
                scope,
                session_id,
                request_id,
                db,
                top_k,
                top_n,
                cache_key,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Stops nginx buffering the stream into one lump.
                "X-Accel-Buffering": "no",
            },
        )

    cached = answer_cache.get(cache_key) if cache_key else None
    if cached is not None:
        result = dataclasses.replace(cached, cached=True)
    else:
        result = await service.answer(
            payload.query,
            search_filter,
            top_k=top_k,
            top_n=top_n,
            mode=payload.mode,
            options=_options(payload, scope),
        )
        if cache_key:
            answer_cache.put(cache_key, result)
    answer_id = await _record(db, request_id, session_id, result)

    log.info(
        "chat_answered",
        session_id=session_id,
        language=str(result.language_detected),
        grounded=result.grounded,
        citations=len(result.citations),
        cost_usd=result.cost_usd,
        cached=result.cached,
        latency_ms=result.latency_ms.get("total"),
    )
    return _to_response(result, session_id, request_id, answer_id=answer_id, scope=scope)


async def _sse(
    service: ChatService,
    payload: ChatRequest,
    search_filter: SearchFilter,
    scope: Scope,
    session_id: str,
    request_id: str,
    db: AsyncSession,
    top_k: int,
    top_n: int,
    cache_key: str | None,
) -> AsyncIterator[str]:
    """Server-sent events: `token` deltas, then one `done` with the metadata."""

    def event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield event("start", {"request_id": request_id, "session_id": session_id})

    async def finish(result: ChatResult) -> str:
        answer_id = await _record(db, request_id, session_id, result)
        return event(
            "done",
            _to_response(
                result, session_id, request_id, answer_id=answer_id, scope=scope
            ).model_dump(mode="json"),
        )

    try:
        cached = answer_cache.get(cache_key) if cache_key else None
        if cached is not None:
            result = dataclasses.replace(cached, cached=True)
            yield event("token", {"text": result.answer})
            yield await finish(result)
            return

        stream = service.answer_stream(
            payload.query,
            search_filter,
            top_k=top_k,
            top_n=top_n,
            mode=payload.mode,
            options=_options(payload, scope),
        )
        async for delta, final in stream:
            if final is None:
                if delta:
                    yield event("token", {"text": delta})
                continue

            if cache_key:
                answer_cache.put(cache_key, final)
            yield await finish(final)
    except Exception as exc:
        log.exception("chat_stream_failed", session_id=session_id)
        # Headers are already sent, so an HTTP error code is no longer possible;
        # the client learns about the failure from this event instead.
        yield event(
            "error",
            {
                "request_id": request_id,
                "code": getattr(exc, "code", "INTERNAL"),
                "message": getattr(exc, "message", str(exc))[:300],
            },
        )
