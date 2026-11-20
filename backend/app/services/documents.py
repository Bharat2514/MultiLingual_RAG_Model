"""Document persistence and the ingestion state machine.

Split into async functions for the API and sync functions for the Celery
worker, over the same models. The duplication is deliberate: bridging an event
loop into a Celery process is a well-known source of deadlocks, and these two
call sites have genuinely different concurrency stories.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session as SyncSession

from app.core.errors import DocumentNotFoundError, TooManyFilesError
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus, IngestionJob, JobStage, JobStatus
from app.db.models import Session as SessionModel
from app.ingestion.types import IngestionResult

log = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --- async (API) -------------------------------------------------------------


async def ensure_session(db: AsyncSession, session_id: str) -> SessionModel:
    """Get or create the session row that scopes a user's documents."""
    session = await db.get(SessionModel, session_id)
    if session is None:
        session = SessionModel(session_id=session_id, last_seen_at=_utcnow())
        db.add(session)
        await db.flush()
    else:
        session.last_seen_at = _utcnow()
    return session


async def assert_capacity(db: AsyncSession, session_id: str, incoming: int, limit: int) -> None:
    """Enforce the per-session document cap (FR-2)."""
    current = await db.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.session_id == session_id, Document.status != DocumentStatus.FAILED)
    )
    current = current or 0
    if current + incoming > limit:
        raise TooManyFilesError(
            f"This session holds {current} documents; adding {incoming} would exceed the limit of {limit}.",
            details={"current": current, "incoming": incoming, "limit": limit},
        )


async def create_document(db: AsyncSession, **fields) -> Document:
    document = Document(status=DocumentStatus.QUEUED, **fields)
    db.add(document)
    await db.flush()
    return document


async def get_document(db: AsyncSession, doc_id: str, session_id: str) -> Document:
    """Fetch a document, enforcing session scoping (FR-19, NFR-6).

    Scoping is part of the query, not a check afterwards — a missing filter
    then cannot leak another session's document.
    """
    document = await db.scalar(
        select(Document).where(Document.doc_id == doc_id, Document.session_id == session_id)
    )
    if document is None:
        raise DocumentNotFoundError(details={"doc_id": doc_id})
    return document


async def list_documents(db: AsyncSession, session_id: str) -> list[Document]:
    result = await db.scalars(
        select(Document)
        .where(Document.session_id == session_id)
        .order_by(Document.created_at.desc())
    )
    return list(result)


async def list_jobs(db: AsyncSession, doc_id: str) -> list[IngestionJob]:
    result = await db.scalars(
        select(IngestionJob)
        .where(IngestionJob.doc_id == doc_id)
        .order_by(IngestionJob.created_at.asc())
    )
    return list(result)


async def delete_document(db: AsyncSession, document: Document) -> int:
    """Delete a document and its chunks. Vectors are removed by the caller."""
    chunk_count = await db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.doc_id == document.doc_id)
    )
    await db.execute(delete(Chunk).where(Chunk.doc_id == document.doc_id))
    await db.delete(document)
    return chunk_count or 0


# --- sync (worker) -----------------------------------------------------------


def sync_get_document(db: SyncSession, doc_id: str) -> Document:
    document = db.get(Document, doc_id)
    if document is None:
        raise DocumentNotFoundError(details={"doc_id": doc_id})
    return document


def sync_set_status(
    db: SyncSession,
    document: Document,
    status: DocumentStatus,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    document.status = status
    document.error_code = error_code
    document.error_message = error_message[:2000] if error_message else None
    db.flush()
    log.info("document_status", doc_id=document.doc_id, status=str(status), error=error_code)


def sync_record_stage(
    db: SyncSession,
    doc_id: str,
    stage: JobStage,
    status: str,
    meta: dict | None = None,
) -> None:
    """Upsert the per-stage job row that drives the UI progress display."""
    job = db.scalar(
        select(IngestionJob).where(IngestionJob.doc_id == doc_id, IngestionJob.stage == stage)
    )
    now = _utcnow()

    if job is None:
        job = IngestionJob(doc_id=doc_id, stage=str(stage), status=JobStatus.PENDING)
        db.add(job)

    if status == "running":
        job.status = JobStatus.RUNNING
        job.started_at = now
    elif status in ("succeeded", "skipped"):
        job.status = JobStatus.SUCCEEDED
        job.finished_at = now
        if job.started_at:
            # SQLite hands timestamps back naive even for timezone-aware
            # columns; Postgres does not. They are written as UTC either way.
            started = job.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            job.duration_ms = (now - started).total_seconds() * 1000
    elif status == "failed":
        job.status = JobStatus.FAILED
        job.finished_at = now
        job.error_message = str((meta or {}).get("error"))[:2000]

    # Stage detail rides in error_message for skipped/informational cases so the
    # UI can explain *why* a stage was skipped (e.g. "text layer sufficient").
    if status == "skipped" and meta and meta.get("reason"):
        job.error_message = str(meta["reason"])[:2000]

    db.flush()


def sync_persist_chunks(
    db: SyncSession, document: Document, result: IngestionResult
) -> list[Chunk]:
    """Replace a document's chunks with a fresh ingestion result."""
    db.execute(delete(Chunk).where(Chunk.doc_id == document.doc_id))

    import hashlib

    rows = [
        Chunk(
            doc_id=document.doc_id,
            session_id=document.session_id,
            user_id=document.user_id,
            chunk_index=chunk.index,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            token_count=chunk.token_count,
            language=chunk.language,
            section=chunk.section,
            ocr_used=chunk.ocr_used,
            text=chunk.text,
            text_hash=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        )
        for chunk in result.chunks
    ]
    db.add_all(rows)

    document.pages = result.page_count
    document.chunk_count = len(rows)
    document.language_primary = result.language_primary
    document.language_distribution = result.language_distribution
    document.ocr_used = result.ocr_used
    document.ocr_page_ratio = result.ocr_page_ratio
    # Flush so the generated chunk_ids exist — the vector points key off them.
    db.flush()
    return rows
