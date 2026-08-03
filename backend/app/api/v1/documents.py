"""Document endpoints: upload, status, list, delete (FR-31, §9.1)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_id
from app.core.config import settings
from app.core.context import get_request_id
from app.core.errors import PolyDocError
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.db.session import get_db
from app.ingestion.storage import build_key, get_object_store
from app.ingestion.validation import validate_upload
from app.schemas.documents import (
    DeleteResponse,
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    IngestionStageStatus,
    UploadAccepted,
    UploadResponse,
)
from app.services import documents as doc_service

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)


def _dispatch_ingestion(doc_id: str, background: BackgroundTasks) -> str:
    """Send the document to Celery, or run it inline if no broker is up.

    The inline path keeps `make dev` usable with a single process, and means a
    Redis outage degrades throughput instead of dropping uploads (NFR-5).
    """
    from app.workers.celery_app import is_broker_reachable

    if is_broker_reachable():
        from app.workers.tasks import ingest_document

        ingest_document.delay(doc_id)
        return "celery"

    from app.workers.tasks import ingest_document_sync

    log.warning("celery_broker_unreachable_running_inline", doc_id=doc_id)
    background.add_task(ingest_document_sync, doc_id)
    return "inline"


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one or more documents",
)
async def upload_documents(
    background: BackgroundTasks,
    files: list[UploadFile] = File(..., description="PDF or DOCX files"),
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Accept files, persist them, and queue ingestion.

    Partial success is intentional: one bad file in a batch of ten should not
    reject the other nine, so rejects are reported per file (FR-1, FR-30).
    """
    await doc_service.ensure_session(db, session_id)
    await doc_service.assert_capacity(
        db, session_id, len(files), settings.limits.max_files_per_session
    )

    store = get_object_store()
    accepted: list[UploadAccepted] = []
    rejected: list[dict] = []
    to_dispatch: list[str] = []

    for upload in files:
        raw_name = upload.filename or "document"
        try:
            content = await upload.read()
            validated = validate_upload(raw_name, content)
        except PolyDocError as exc:
            rejected.append({"filename": raw_name, "code": str(exc.code), "message": exc.message})
            continue
        finally:
            await upload.close()

        # Re-uploading the same bytes into the same session is a no-op rather
        # than a duplicate index entry.
        existing = await db.scalar(
            select(Document).where(
                Document.session_id == session_id,
                Document.content_sha256 == validated.sha256,
                Document.status != DocumentStatus.FAILED,
            )
        )
        if existing is not None:
            accepted.append(
                UploadAccepted(
                    doc_id=existing.doc_id,
                    filename=existing.filename,
                    status=DocumentStatus(existing.status),
                    size_bytes=existing.size_bytes,
                    duplicate_of=existing.doc_id,
                )
            )
            continue

        document = await doc_service.create_document(
            db,
            session_id=session_id,
            filename=validated.filename,
            mime_type=validated.mime_type,
            storage_key="",  # set below, once we know the doc_id
            content_sha256=validated.sha256,
            size_bytes=validated.size_bytes,
        )
        key = build_key(session_id, document.doc_id, validated.extension)
        store.put(key, content)
        document.storage_key = key
        await db.flush()

        to_dispatch.append(document.doc_id)
        accepted.append(
            UploadAccepted(
                doc_id=document.doc_id,
                filename=document.filename,
                status=DocumentStatus.QUEUED,
                size_bytes=document.size_bytes,
            )
        )

    # Commit before dispatching so the worker cannot race an uncommitted row.
    await db.commit()

    dispatch = "none"
    for doc_id in to_dispatch:
        dispatch = _dispatch_ingestion(doc_id, background)

    log.info(
        "upload_received",
        session_id=session_id,
        accepted=len(accepted),
        rejected=len(rejected),
        dispatch=dispatch,
    )
    return UploadResponse(
        request_id=get_request_id(),
        session_id=session_id,
        accepted=accepted,
        rejected=rejected,
        dispatch=dispatch,
    )


@router.get("", response_model=DocumentListResponse, summary="List documents in this session")
async def list_documents(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    documents = await doc_service.list_documents(db, session_id)
    return DocumentListResponse(
        request_id=get_request_id(),
        session_id=session_id,
        documents=[DocumentSummary.from_orm_document(d) for d in documents],
        total=len(documents),
    )


@router.get("/{doc_id}", response_model=DocumentDetail, summary="Ingestion status and metadata")
async def get_document(
    doc_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    document = await doc_service.get_document(db, doc_id, session_id)
    jobs = await doc_service.list_jobs(db, doc_id)
    detail = DocumentDetail.model_validate(document)
    detail.stages = [IngestionStageStatus.from_orm_job(j) for j in jobs]
    return detail


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(
    doc_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> DeleteResponse:
    """Hard delete: vectors, file, and rows (NFR-6, §16)."""
    document = await doc_service.get_document(db, doc_id, session_id)
    storage_key = document.storage_key

    # Vectors first: a stale row is recoverable, an orphaned vector that can
    # still be retrieved is a data-leak.
    from app.retrieval.service import RetrievalService

    deleted_vectors = await RetrievalService().delete_document(doc_id, session_id)

    deleted_chunks = await doc_service.delete_document(db, document)
    await db.commit()

    if storage_key:
        try:
            get_object_store().delete(storage_key)
        except Exception:
            log.warning("storage_delete_failed", key=storage_key, exc_info=True)

    log.info(
        "document_deleted",
        doc_id=doc_id,
        session_id=session_id,
        chunks=deleted_chunks,
        vectors=deleted_vectors,
    )
    return DeleteResponse(
        request_id=get_request_id(),
        doc_id=doc_id,
        deleted_chunks=deleted_chunks,
        deleted_vectors=deleted_vectors,
    )
