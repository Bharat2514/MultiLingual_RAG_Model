"""Session-scoped access to a document's chunks for the intelligence features.

These features read whole documents, not search results, so they load chunks
from Postgres (the metadata mirror) rather than Qdrant — no embedding model is
needed, which is also why they work on a machine without the ML stack.

Every query filters on `session_id` in SQL, the same isolation rule the rest of
the API follows: a document from another session is "not found", never
"forbidden".
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import DocumentNotReadyError
from app.db.models import Chunk, Document, DocumentInsight, DocumentStatus
from app.generation.providers.base import LLMProvider
from app.retrieval.coverage import select_coverage
from app.retrieval.types import ScoredChunk
from app.services import documents as doc_service


async def ready_document(db: AsyncSession, doc_id: str, session_id: str) -> Document:
    document = await doc_service.get_document(db, doc_id, session_id)
    if document.status != DocumentStatus.READY:
        raise DocumentNotReadyError(details={"doc_id": doc_id, "status": document.status})
    return document


async def load_chunks(db: AsyncSession, document: Document) -> list[ScoredChunk]:
    rows = await db.scalars(
        select(Chunk)
        .where(Chunk.doc_id == document.doc_id, Chunk.session_id == document.session_id)
        .order_by(Chunk.chunk_index.asc())
    )
    return [
        ScoredChunk(
            chunk_id=row.chunk_id,
            doc_id=row.doc_id,
            text=row.text,
            score=0.0,
            filename=document.filename,
            page_start=row.page_start,
            page_end=row.page_end,
            chunk_index=row.chunk_index,
            language=row.language,
            ocr_used=row.ocr_used,
            session_id=row.session_id,
        )
        for row in rows
    ]


def as_dicts(chunks: list[ScoredChunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "page": c.page_start,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "filename": c.filename,
        }
        for c in chunks
    ]


def sample(chunks: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
    """A reading-order sample spanning the document, for whole-document prompts."""
    return chunks if len(chunks) <= limit else select_coverage(chunks, limit=limit)


async def get_insight(
    db: AsyncSession, document: Document, kind: str, variant: str = ""
) -> DocumentInsight | None:
    return await db.scalar(
        select(DocumentInsight).where(
            DocumentInsight.doc_id == document.doc_id,
            DocumentInsight.session_id == document.session_id,
            DocumentInsight.kind == kind,
            DocumentInsight.variant == variant,
        )
    )


async def save_insight(
    db: AsyncSession,
    document: Document,
    kind: str,
    payload: dict,
    *,
    variant: str = "",
    model: str | None = None,
) -> DocumentInsight:
    insight = await get_insight(db, document, kind, variant)
    if insight is None:
        insight = DocumentInsight(
            doc_id=document.doc_id,
            session_id=document.session_id,
            kind=kind,
            variant=variant,
            payload=payload,
            model=model,
        )
        db.add(insight)
    else:
        insight.payload = payload
        insight.model = model
    await db.flush()
    return insight


def should_redact(provider: LLMProvider) -> bool:
    """Mask identifiers only when the context leaves the machine."""
    from app.core.config import settings

    return settings.trust.redact_pii_for_llm and provider.name != "ollama"
