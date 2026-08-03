"""Metadata store (§10.1).

Vectors live in Qdrant; this schema owns identity, ingestion state, and the
chunk metadata mirror used for citation rendering and isolation tests (NFR-6).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base, TimestampMixin, new_uuid

# JSONB on Postgres (indexable, typed); plain JSON elsewhere so SQLite dev works.
JSONColumn = JSON().with_variant(JSONB(), "postgresql")


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    OCR = "ocr"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (DocumentStatus.READY, DocumentStatus.FAILED)


class JobStage(StrEnum):
    VALIDATE = "validate"
    PARSE = "parse"
    OCR = "ocr"
    CLEAN = "clean"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Language(StrEnum):
    HI = "hi"
    EN = "en"
    HINGLISH = "hinglish"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    display_name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(320), unique=True)

    sessions: Mapped[list[Session]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Session(Base, TimestampMixin):
    """A scoping boundary for uploads and retrieval (FR-19, NFR-6)."""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_uuid)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str | None] = mapped_column(String(255))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User | None] = relationship(back_populates="sessions")
    documents: Mapped[list[Document]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_session_status", "session_id", "status"),
        CheckConstraint("size_bytes >= 0", name="size_bytes_non_negative"),
    )

    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    pages: Mapped[int | None] = mapped_column(Integer)
    language_primary: Mapped[str] = mapped_column(
        String(16), default=Language.UNKNOWN, nullable=False
    )
    language_distribution: Mapped[dict | None] = mapped_column(JSONColumn)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ocr_page_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.QUEUED, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[Session] = relationship(back_populates="documents")
    user: Mapped[User | None] = relationship(back_populates="documents")
    jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="IngestionJob.created_at"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    @property
    def is_ready(self) -> bool:
        return self.status == DocumentStatus.READY


class IngestionJob(Base, TimestampMixin):
    """Per-stage progress record so the UI can show a real pipeline (FR-26)."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_doc_stage", "doc_id", "stage"),)

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.PENDING, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)

    document: Mapped[Document] = relationship(back_populates="jobs")


class Chunk(Base, TimestampMixin):
    """Metadata mirror of a Qdrant point (§10.1).

    Text is stored here as well so citation snippets and the eval harness do not
    have to round-trip through the vector store.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="doc_chunk_index"),
        Index("ix_chunks_session_doc", "session_id", "doc_id"),
    )

    chunk_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(32), index=True)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    language: Mapped[str] = mapped_column(String(16), default=Language.UNKNOWN, nullable=False)
    section: Mapped[str | None] = mapped_column(String(512))
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class QueryLog(Base, TimestampMixin):
    """Per-query trace for the eval dashboards and cost tracking (NFR-4, NFR-7)."""

    __tablename__ = "query_logs"
    __table_args__ = (Index("ix_query_logs_session_created", "session_id", "created_at"),)

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    language_detected: Mapped[str] = mapped_column(
        String(16), default=Language.UNKNOWN, nullable=False
    )
    normalized_query: Mapped[str | None] = mapped_column(Text)

    doc_ids: Mapped[list | None] = mapped_column(JSONColumn)
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSONColumn)

    grounded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    answer_chars: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    citation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    latency_retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_rerank_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_generation_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_total_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    error_code: Mapped[str | None] = mapped_column(String(64))
