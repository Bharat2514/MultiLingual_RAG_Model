"""The chat endpoint (§9.2, §9.3, FR-24, FR-25)."""

from __future__ import annotations

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
from app.generation.service import ChatResult, ChatService
from app.retrieval.vectorstore.base import SearchFilter
from app.schemas.chat import ChatRequest, ChatResponse, CitationOut
from app.schemas.common import LatencyBreakdown, TokenUsage

router = APIRouter(tags=["chat"])
log = get_logger(__name__)


def _to_response(result: ChatResult, session_id: str, request_id: str) -> ChatResponse:
    retrieval = result.retrieval
    timings = result.latency_ms
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
    )


async def _record(db: AsyncSession, request_id: str, session_id: str, result: ChatResult) -> None:
    """Persist the per-query trace for cost and eval dashboards (NFR-4, NFR-7)."""
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
        await db.commit()
    except Exception:
        await db.rollback()
        log.warning("query_log_write_failed", exc_info=True)


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

    scope = SearchFilter(session_id=session_id, doc_ids=payload.doc_ids)
    top_k = payload.top_k or settings.retrieval.top_k
    top_n = payload.top_n or settings.retrieval.top_n

    if payload.stream:
        return StreamingResponse(
            _sse(service, payload, scope, session_id, request_id, db, top_k, top_n),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Stops nginx buffering the stream into one lump.
                "X-Accel-Buffering": "no",
            },
        )

    result = await service.answer(payload.query, scope, top_k=top_k, top_n=top_n, mode=payload.mode)
    await _record(db, request_id, session_id, result)

    log.info(
        "chat_answered",
        session_id=session_id,
        language=str(result.language_detected),
        grounded=result.grounded,
        citations=len(result.citations),
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms.get("total"),
    )
    return _to_response(result, session_id, request_id)


async def _sse(
    service: ChatService,
    payload: ChatRequest,
    scope: SearchFilter,
    session_id: str,
    request_id: str,
    db: AsyncSession,
    top_k: int,
    top_n: int,
) -> AsyncIterator[str]:
    """Server-sent events: `token` deltas, then one `done` with the metadata."""

    def event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield event("start", {"request_id": request_id, "session_id": session_id})

    try:
        stream = service.answer_stream(
            payload.query, scope, top_k=top_k, top_n=top_n, mode=payload.mode
        )
        async for delta, result in stream:
            if result is None:
                if delta:
                    yield event("token", {"text": delta})
                continue

            await _record(db, request_id, session_id, result)
            yield event(
                "done", _to_response(result, session_id, request_id).model_dump(mode="json")
            )
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
