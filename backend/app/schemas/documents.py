"""Request/response models for the document endpoints (§9)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Document, DocumentStatus, IngestionJob


class IngestionStageStatus(BaseModel):
    stage: str
    status: str
    duration_ms: float | None = None
    detail: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_orm_job(cls, job: IngestionJob) -> IngestionStageStatus:
        return cls(
            stage=job.stage,
            status=job.status,
            duration_ms=round(job.duration_ms, 2) if job.duration_ms else None,
            detail=job.error_message,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doc_id: str
    filename: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    pages: int | None = None
    chunk_count: int = 0
    language_primary: str
    language_distribution: dict[str, int] | None = None
    ocr_used: bool = False
    ocr_page_ratio: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm_document(cls, document: Document) -> DocumentSummary:
        return cls.model_validate(document)


class DocumentDetail(DocumentSummary):
    stages: list[IngestionStageStatus] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    request_id: str
    session_id: str
    documents: list[DocumentSummary]
    total: int


class UploadAccepted(BaseModel):
    """One entry per uploaded file (§9.1: returns job ids)."""

    doc_id: str
    filename: str
    status: DocumentStatus
    size_bytes: int
    duplicate_of: str | None = Field(
        default=None,
        description="Set when this file was already ingested in this session.",
    )


class UploadResponse(BaseModel):
    request_id: str
    session_id: str
    accepted: list[UploadAccepted]
    rejected: list[dict] = Field(default_factory=list)
    dispatch: str = Field(description="'celery' or 'inline' — how ingestion was scheduled.")


class DeleteResponse(BaseModel):
    request_id: str
    doc_id: str
    deleted_chunks: int
    deleted_vectors: int = 0
    deleted: bool = True
