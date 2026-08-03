"""Retrieval-only endpoint.

Not in the §9 API surface, but required by the evaluation plan: Recall@K, MRR,
and nDCG@N (§12.2) are properties of retrieval alone, and the §12.4 ablations
(dense vs hybrid vs hybrid+rerank) need to exercise each configuration without
paying for generation. It also lets the UI show "what did it find" independently
of "what did it say".
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_session_id
from app.core.config import settings
from app.core.context import get_request_id
from app.retrieval.service import RetrievalService
from app.retrieval.types import SearchMode
from app.retrieval.vectorstore.base import SearchFilter

router = APIRouter(tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] | None = None
    top_k: int = Field(default=0, ge=0, le=200, description="0 uses the configured default.")
    top_n: int = Field(default=0, ge=0, le=50, description="0 uses the configured default.")
    mode: SearchMode = SearchMode.HYBRID
    rerank: bool | None = None
    queries: list[str] | None = Field(
        default=None, description="Query variants to fuse (e.g. Hinglish + transliteration)."
    )


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    chunk_index: int
    language: str
    ocr_used: bool
    score: float
    retrieval_score: float | None = None
    rerank_score: float | None = None
    rank: int
    text: str


class SearchResponse(BaseModel):
    request_id: str
    session_id: str
    query: str
    mode: SearchMode
    reranked: bool
    candidates_considered: int
    hits: list[SearchHit]
    latency_ms: dict[str, float]


@router.post("/search", response_model=SearchResponse, summary="Retrieve chunks (no generation)")
async def search(
    payload: SearchRequest,
    session_id: str = Depends(get_session_id),
) -> SearchResponse:
    result = await RetrievalService().retrieve(
        payload.query,
        SearchFilter(session_id=session_id, doc_ids=payload.doc_ids),
        queries=payload.queries,
        top_k=payload.top_k or settings.retrieval.top_k,
        top_n=payload.top_n or settings.retrieval.top_n,
        mode=payload.mode,
        rerank=payload.rerank,
    )

    return SearchResponse(
        request_id=get_request_id(),
        session_id=session_id,
        query=payload.query,
        mode=result.mode,
        reranked=result.reranked,
        candidates_considered=result.candidates_considered,
        hits=[
            SearchHit(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
                chunk_index=c.chunk_index,
                language=c.language,
                ocr_used=c.ocr_used,
                score=round(c.score, 6),
                retrieval_score=round(c.retrieval_score, 6)
                if c.retrieval_score is not None
                else None,
                rerank_score=round(c.rerank_score, 6) if c.rerank_score is not None else None,
                rank=c.rank,
                text=c.text,
            )
            for c in result.chunks
        ],
        latency_ms=result.timings_ms,
    )
