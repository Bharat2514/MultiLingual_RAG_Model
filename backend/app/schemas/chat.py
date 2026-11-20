"""Chat request/response models (§9.2, §9.3)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Language
from app.retrieval.types import SearchMode
from app.schemas.common import LatencyBreakdown, TokenUsage


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "mujhe is document ka summary do",
                "doc_ids": ["doc_1"],
                "top_k": 20,
                "top_n": 5,
                "stream": False,
            }
        }
    )

    query: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents. Null searches all."
    )
    top_k: int = Field(default=0, ge=0, le=200, description="0 uses the configured default.")
    top_n: int = Field(default=0, ge=0, le=50, description="0 uses the configured default.")
    mode: SearchMode = SearchMode.HYBRID
    stream: bool = False
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=20,
        description="Earlier turns, oldest first. Lets follow-ups like 'aur uska penalty?' "
        "be resolved into a standalone question before retrieval.",
    )
    style: Literal["default", "simple"] = Field(
        default="default", description="'simple' answers in plain, jargon-free language."
    )
    as_of: date | None = Field(
        default=None,
        description="Answer only from documents in force on this date (see /documents/{id}/timeline).",
    )
    include_superseded: bool = Field(
        default=False, description="Also search documents that have been replaced by a newer one."
    )
    use_cache: bool = True


class CitationOut(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk_id: str
    chunk_index: int
    snippet: str
    score: float
    context_index: int = Field(
        default=0,
        description=(
            "1-based position of this passage in the context given to the model — "
            "what a bare `[2]` in the answer refers to. Citations are deduplicated "
            "and sorted by (filename, page), so list order does not resolve it."
        ),
    )


class ClaimOut(BaseModel):
    text: str
    verdict: Literal["supported", "partial", "unsupported", "uncited", "unchecked"]
    score: float | None = None
    method: str = "none"
    chunk_ids: list[str] = Field(default_factory=list)
    missing_numbers: list[str] = Field(default_factory=list)
    reason: str = ""


class ClaimSummary(BaseModel):
    counts: dict[str, int]
    support_ratio: float | None = Field(
        default=None, description="Supported share of the claims that could be judged."
    )


class AbstentionOut(BaseModel):
    reason: str = Field(description="no_documents | no_relevant_passages | model_declined")
    searched_documents: int
    nearest_source: str | None = None
    withheld_passages: int = 0
    hint: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req_789",
                "answer": "Is document mein ... [policy.pdf p.4]",
                "language_detected": "hinglish",
                "grounded": True,
                "citations": [
                    {
                        "doc_id": "doc_1",
                        "filename": "policy.pdf",
                        "page": 4,
                        "chunk_id": "c12",
                        "chunk_index": 12,
                        "snippet": "…source text…",
                        "score": 0.83,
                    }
                ],
                "usage": {"input_tokens": 1450, "output_tokens": 220, "cost_usd": 0.0009},
                "latency_ms": {
                    "retrieval": 180,
                    "rerank": 90,
                    "generation": 2600,
                    "total": 2980,
                },
            }
        }
    )

    request_id: str
    session_id: str
    answer: str
    language_detected: Language
    grounded: bool = Field(
        description="False when the answer could not be tied to a retrieved passage."
    )
    citations: list[CitationOut] = Field(default_factory=list)
    usage: TokenUsage
    latency_ms: LatencyBreakdown
    model: str
    # Retrieval diagnostics — used by the UI's "what did it find" panel and the
    # eval harness; harmless for ordinary API consumers.
    retrieval_mode: SearchMode = SearchMode.HYBRID
    reranked: bool = False
    candidates_considered: int = 0
    query_variants: list[str] = Field(default_factory=list)
    # --- beyond the citation contract ---
    answer_id: str | None = Field(
        default=None, description="Snapshot id; share it via POST /answers/{answer_id}/share."
    )
    claims: list[ClaimOut] = Field(default_factory=list)
    claim_summary: ClaimSummary | None = None
    rewritten_query: str | None = Field(
        default=None,
        description="The standalone question searched, when a follow-up was rewritten.",
    )
    abstention: AbstentionOut | None = None
    withheld_passages: int = Field(
        default=0, description="Retrieved passages withheld for carrying injected instructions."
    )
    excluded_documents: list[dict] = Field(
        default_factory=list, description="Documents left out as superseded or not yet in force."
    )
    cached: bool = False
