"""Vector store interface (NFR-8, NFR-9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.retrieval.types import Embedding, ScoredChunk, SearchMode


@dataclass(slots=True, kw_only=True)
class PendingChunk:
    """A chunk that has been extracted but not yet embedded.

    Separate from `ChunkPoint` so "not embedded yet" is a distinct type rather
    than a nullable field — the store cannot then be handed a vectorless point.
    """

    chunk_id: str
    doc_id: str
    session_id: str
    text: str
    filename: str
    page_start: int
    page_end: int
    chunk_index: int
    language: str
    ocr_used: bool
    user_id: str | None = None


@dataclass(slots=True, kw_only=True)
class ChunkPoint(PendingChunk):
    """A chunk plus its vectors, ready to upsert."""

    embedding: Embedding

    @classmethod
    def from_pending(cls, pending: PendingChunk, embedding: Embedding) -> ChunkPoint:
        return cls(
            chunk_id=pending.chunk_id,
            doc_id=pending.doc_id,
            session_id=pending.session_id,
            text=pending.text,
            filename=pending.filename,
            page_start=pending.page_start,
            page_end=pending.page_end,
            chunk_index=pending.chunk_index,
            language=pending.language,
            ocr_used=pending.ocr_used,
            user_id=pending.user_id,
            embedding=embedding,
        )


@dataclass(slots=True)
class SearchFilter:
    """Retrieval scope. `session_id` is mandatory — see `VectorStore.search`."""

    session_id: str
    doc_ids: list[str] | None = None
    user_id: str | None = None
    languages: list[str] | None = None


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self) -> None: ...

    @abstractmethod
    async def upsert_chunks(self, points: list[ChunkPoint]) -> int: ...

    @abstractmethod
    async def search(
        self,
        query: Embedding,
        scope: SearchFilter,
        *,
        limit: int,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[ScoredChunk]:
        """Retrieve candidates.

        Implementations **must** apply `scope.session_id` as a hard filter. It is
        a required field rather than an option precisely so that an unscoped
        search cannot be expressed (FR-19, NFR-6).
        """

    @abstractmethod
    async def list_chunks(self, scope: SearchFilter, *, limit: int) -> list[ScoredChunk]:
        """Return chunks in `scope` by document order, ignoring relevance.

        Aggregate queries ("summarise this document") have nothing for a ranker
        to rank against, so they need the document's *contents*, not its best
        matches. `scope.session_id` is a hard filter here exactly as in `search`.
        """

    @abstractmethod
    async def delete_document(self, doc_id: str, session_id: str) -> int: ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> int: ...

    @abstractmethod
    async def count(self, scope: SearchFilter | None = None) -> int: ...

    @abstractmethod
    async def health(self) -> bool: ...

    async def close(self) -> None:  # pragma: no cover
        return None
