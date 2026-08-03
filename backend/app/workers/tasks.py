"""Ingestion tasks.

One task per document. Every stage transition is written to `ingestion_jobs`
so the UI can render real progress, and every failure lands as a terminal
`failed` status with a user-facing code rather than a lost job (NFR-5).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from celery import shared_task

from app.core.errors import ErrorCode, PolyDocError
from app.core.logging import get_logger
from app.db.models import DocumentStatus, JobStage
from app.db.session import sync_session_scope
from app.ingestion.pipeline import run_pipeline
from app.ingestion.storage import get_object_store
from app.ingestion.validation import FileKind
from app.retrieval.vectorstore.base import PendingChunk
from app.services import documents as doc_service

log = get_logger(__name__)


def _index_chunks(doc_id: str, points: list[PendingChunk], on_stage) -> int:
    """Embed and upsert a document's chunks (FR-10, FR-11, FR-12).

    The worker is synchronous while the vector store client is async, so the
    coroutine is driven with `asyncio.run` here. Each Celery task owns its
    process, so there is no running loop to conflict with.
    """
    from app.retrieval.service import RetrievalService

    on_stage(JobStage.EMBED, "running", {})
    started = time.perf_counter()
    try:
        indexed = asyncio.run(RetrievalService().index_chunks(points))
    except Exception as exc:
        on_stage(JobStage.EMBED, "failed", {"error": str(exc)})
        raise

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    on_stage(JobStage.EMBED, "succeeded", {"chunks": indexed})
    on_stage(JobStage.INDEX, "succeeded", {"vectors": indexed, "duration_ms": elapsed})
    log.info("chunks_indexed", doc_id=doc_id, vectors=indexed, duration_ms=elapsed)
    return indexed


# Which document status each pipeline stage corresponds to, so the status field
# tracks the pipeline instead of jumping straight from queued to ready.
_STAGE_STATUS = {
    JobStage.PARSE: DocumentStatus.PARSING,
    JobStage.OCR: DocumentStatus.OCR,
    JobStage.CLEAN: DocumentStatus.PARSING,
    JobStage.CHUNK: DocumentStatus.CHUNKING,
    JobStage.EMBED: DocumentStatus.INDEXING,
    JobStage.INDEX: DocumentStatus.INDEXING,
}


def ingest_document_sync(doc_id: str) -> dict:
    """Run ingestion for one document. Safe to call inline or from Celery."""
    started = time.perf_counter()

    with sync_session_scope() as db:
        document = doc_service.sync_get_document(db, doc_id)
        storage_key = document.storage_key
        mime_type = document.mime_type
        session_id = document.session_id
        doc_service.sync_set_status(db, document, DocumentStatus.PARSING)

    kind = FileKind.PDF if mime_type == "application/pdf" else FileKind.DOCX
    store = get_object_store()

    def on_stage(stage: JobStage, status: str, meta: dict) -> None:
        with sync_session_scope() as db:
            doc_service.sync_record_stage(db, doc_id, stage, status, meta)
            if status == "running" and stage in _STAGE_STATUS:
                document = doc_service.sync_get_document(db, doc_id)
                doc_service.sync_set_status(db, document, _STAGE_STATUS[stage])

    try:
        path = store.local_path(storage_key)
        result = run_pipeline(path, kind, on_stage=on_stage)

        with sync_session_scope() as db:
            document = doc_service.sync_get_document(db, doc_id)
            rows = doc_service.sync_persist_chunks(db, document, result)
            chunk_count = len(rows)
            # Snapshot what indexing needs before the session closes.
            points = [
                PendingChunk(
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    session_id=row.session_id,
                    user_id=row.user_id,
                    text=row.text,
                    filename=document.filename,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    chunk_index=row.chunk_index,
                    language=row.language,
                    ocr_used=row.ocr_used,
                )
                for row in rows
            ]

        _index_chunks(doc_id, points, on_stage)

        with sync_session_scope() as db:
            document = doc_service.sync_get_document(db, doc_id)
            document.indexed_at = datetime.now(UTC)
            doc_service.sync_set_status(db, document, DocumentStatus.READY)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "ingestion_succeeded",
            doc_id=doc_id,
            session_id=session_id,
            chunks=chunk_count,
            pages=result.page_count,
            language=result.language_primary,
            ocr_used=result.ocr_used,
            duration_ms=elapsed_ms,
        )
        return {
            "doc_id": doc_id,
            "status": str(DocumentStatus.READY),
            "chunks": chunk_count,
            "pages": result.page_count,
            "language": result.language_primary,
            "ocr_used": result.ocr_used,
            "duration_ms": elapsed_ms,
        }

    except PolyDocError as exc:
        _fail(doc_id, str(exc.code), exc.message)
        raise
    except Exception as exc:
        log.exception("ingestion_crashed", doc_id=doc_id)
        _fail(doc_id, str(ErrorCode.INTERNAL), f"{type(exc).__name__}: {exc}")
        raise


def _fail(doc_id: str, code: str, message: str) -> None:
    with sync_session_scope() as db:
        try:
            document = doc_service.sync_get_document(db, doc_id)
        except PolyDocError:
            return
        doc_service.sync_set_status(
            db, document, DocumentStatus.FAILED, error_code=code, error_message=message
        )


@shared_task(
    name="app.workers.tasks.ingest_document",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
    autoretry_for=(),  # retries are decided explicitly below
)
def ingest_document(self, doc_id: str) -> dict:
    try:
        return ingest_document_sync(doc_id)
    except PolyDocError:
        # A bad document will fail identically on every retry.
        raise
    except Exception as exc:
        raise self.retry(exc=exc) from exc
