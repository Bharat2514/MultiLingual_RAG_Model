"""Qdrant-backed hybrid vector store (§10.2).

One collection with two named vectors — `dense` (bge-m3) and `sparse` (bge-m3
lexical weights) — so a single point serves both halves of hybrid search and
they can never drift apart.

Fusion runs server-side via Qdrant's Query API (`prefetch` + RRF), which avoids
shipping two candidate lists to the client just to merge them.

Every query carries a mandatory `session_id` filter. Payload indexes on the
scoping fields keep that filter fast rather than a post-hoc scan.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

from app.core.config import settings
from app.core.errors import VectorStoreUnavailableError
from app.core.logging import get_logger
from app.retrieval.types import Embedding, ScoredChunk, SearchMode
from app.retrieval.vectorstore.base import ChunkPoint, SearchFilter, VectorStore

log = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Deterministic point ids: re-ingesting a document overwrites its points instead
# of duplicating them. Qdrant accepts UUIDs or unsigned ints only, so the
# hex chunk_id is folded into a UUID5.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        dense_dim: int | None = None,
    ) -> None:
        self.url = url or settings.qdrant.url
        self.api_key = api_key or settings.qdrant.api_key
        self.collection = collection or settings.qdrant.collection
        self.dense_dim = dense_dim or settings.embedding.dense_dim
        self._client = None
        # The event loop `_client` was created on; see `_get_client`.
        self._client_loop: object | None = None

    # --- client --------------------------------------------------------------

    def _get_client(self):
        """Return a client bound to the *currently running* event loop.

        The store is a process-wide singleton, but the async client underneath
        it owns connections tied to the loop that created them. The API runs one
        long-lived loop, whereas the Celery worker calls `asyncio.run()` once
        per task — a fresh loop every time. Reusing a client across loops does
        not error, it deadlocks: the second document ingested would hang
        forever. So we detect the loop change and rebuild.
        """
        import asyncio

        from qdrant_client import AsyncQdrantClient

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._client is not None and self._client_loop is not loop:
            # The old loop is gone; its connections cannot be closed cleanly and
            # will be reclaimed with it.
            log.debug("qdrant_client_rebound_to_new_event_loop")
            self._client = None

        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=settings.qdrant.timeout,
                prefer_grpc=settings.qdrant.prefer_grpc,
            )
            self._client_loop = loop

        return self._client

    async def close(self) -> None:
        if self._client is not None:
            # Closing a client whose loop already died is expected, not an error.
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None
            self._client_loop = None

    # --- schema --------------------------------------------------------------

    async def ensure_collection(self) -> None:
        from qdrant_client import models

        client = self._get_client()
        try:
            if await client.collection_exists(self.collection):
                return

            await client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

            # Without these, every scoped search degrades into a full scan.
            for field in ("session_id", "doc_id", "user_id", "language"):
                await client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

            log.info("qdrant_collection_created", collection=self.collection, dim=self.dense_dim)
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Qdrant unavailable: {exc}") from exc

    # --- write ---------------------------------------------------------------

    async def upsert_chunks(self, points: list[ChunkPoint]) -> int:
        if not points:
            return 0

        from qdrant_client import models

        await self.ensure_collection()
        client = self._get_client()

        records = []
        for point in points:
            vectors: dict = {DENSE_VECTOR: point.embedding.dense}
            if point.embedding.sparse and not point.embedding.sparse.is_empty:
                vectors[SPARSE_VECTOR] = models.SparseVector(
                    indices=point.embedding.sparse.indices,
                    values=point.embedding.sparse.values,
                )

            records.append(
                models.PointStruct(
                    id=point_id(point.chunk_id),
                    vector=vectors,
                    payload={
                        "chunk_id": point.chunk_id,
                        "doc_id": point.doc_id,
                        "session_id": point.session_id,
                        "user_id": point.user_id,
                        "filename": point.filename,
                        "page_start": point.page_start,
                        "page_end": point.page_end,
                        "chunk_index": point.chunk_index,
                        "language": point.language,
                        "ocr_used": point.ocr_used,
                        "text": point.text,
                    },
                )
            )

        try:
            await client.upsert(collection_name=self.collection, points=records, wait=True)
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Qdrant upsert failed: {exc}") from exc

        log.debug("qdrant_upserted", count=len(records), collection=self.collection)
        return len(records)

    # --- read ----------------------------------------------------------------

    @staticmethod
    def _build_filter(scope: SearchFilter):
        from qdrant_client import models

        # session_id is unconditional: this is the isolation boundary (NFR-6).
        must: list = [
            models.FieldCondition(key="session_id", match=models.MatchValue(value=scope.session_id))
        ]
        if scope.user_id:
            must.append(
                models.FieldCondition(key="user_id", match=models.MatchValue(value=scope.user_id))
            )
        if scope.doc_ids:
            must.append(
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(scope.doc_ids)))
            )
        if scope.languages:
            must.append(
                models.FieldCondition(
                    key="language", match=models.MatchAny(any=list(scope.languages))
                )
            )
        return models.Filter(must=must)

    @staticmethod
    def _to_scored_chunk(point, rank: int, *, score: float | None = None) -> ScoredChunk:
        payload = point.payload or {}
        # Scroll returns records without a score; search results carry one.
        score = float(score if score is not None else point.score)
        return ScoredChunk(
            chunk_id=payload.get("chunk_id", str(point.id)),
            doc_id=payload.get("doc_id", ""),
            text=payload.get("text", ""),
            score=score,
            retrieval_score=score,
            filename=payload.get("filename", ""),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            chunk_index=int(payload.get("chunk_index", 0)),
            language=payload.get("language", "unknown"),
            ocr_used=bool(payload.get("ocr_used", False)),
            session_id=payload.get("session_id", ""),
            rank=rank,
        )

    async def search(
        self,
        query: Embedding,
        scope: SearchFilter,
        *,
        limit: int,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[ScoredChunk]:
        from qdrant_client import models

        client = self._get_client()
        query_filter = self._build_filter(scope)

        # Bind to a local so the narrowing survives into the branches below.
        sparse = query.sparse if query.sparse and not query.sparse.is_empty else None
        if mode == SearchMode.SPARSE and sparse is None:
            log.warning("sparse_search_requested_without_sparse_vector")
            return []
        # Silently downgrading hybrid to dense is correct: a query of pure
        # stopwords can legitimately produce no lexical weights.
        effective_mode = SearchMode.DENSE if mode == SearchMode.HYBRID and sparse is None else mode

        try:
            if effective_mode == SearchMode.HYBRID:
                assert sparse is not None
                # Both arms run separately and are fused *client-side* so the
                # weights apply. Qdrant's built-in RRF is equal-weight, which
                # measured worse than dense alone on cross-lingual queries —
                # see RetrievalSettings.dense_weight for the numbers.
                prefetch_limit = max(limit * 2, limit + 10)
                dense_res, sparse_res = await asyncio.gather(
                    client.query_points(
                        collection_name=self.collection,
                        query=query.dense,
                        using=DENSE_VECTOR,
                        query_filter=query_filter,
                        limit=prefetch_limit,
                        with_payload=True,
                    ),
                    client.query_points(
                        collection_name=self.collection,
                        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=SPARSE_VECTOR,
                        query_filter=query_filter,
                        limit=prefetch_limit,
                        with_payload=True,
                    ),
                )

                from app.retrieval.fusion import reciprocal_rank_fusion

                return reciprocal_rank_fusion(
                    [
                        [self._to_scored_chunk(p, i) for i, p in enumerate(dense_res.points)],
                        [self._to_scored_chunk(p, i) for i, p in enumerate(sparse_res.points)],
                    ],
                    weights=[
                        settings.retrieval.dense_weight,
                        settings.retrieval.sparse_weight,
                    ],
                    limit=limit,
                )
            elif effective_mode == SearchMode.SPARSE:
                assert sparse is not None
                response = await client.query_points(
                    collection_name=self.collection,
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE_VECTOR,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            else:
                response = await client.query_points(
                    collection_name=self.collection,
                    query=query.dense,
                    using=DENSE_VECTOR,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Qdrant search failed: {exc}") from exc

        return [self._to_scored_chunk(p, i) for i, p in enumerate(response.points)]

    async def list_chunks(self, scope: SearchFilter, *, limit: int) -> list[ScoredChunk]:
        client = self._get_client()
        if not await self._collection_ready():
            return []

        records: list = []
        offset = None
        try:
            while len(records) < limit:
                page, offset = await client.scroll(
                    collection_name=self.collection,
                    scroll_filter=self._build_filter(scope),
                    limit=min(256, limit - len(records)),
                    offset=offset,
                    with_payload=True,
                    # The text is what we are after; the vectors would be pure
                    # transfer cost on a scan this wide.
                    with_vectors=False,
                )
                records.extend(page)
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Qdrant scroll failed: {exc}") from exc

        chunks = [self._to_scored_chunk(r, i, score=0.0) for i, r in enumerate(records)]
        chunks.sort(key=lambda c: (c.filename, c.doc_id, c.chunk_index))
        return chunks

    # --- delete --------------------------------------------------------------

    async def _delete_by_filter(self, query_filter) -> int:
        from qdrant_client import models

        client = self._get_client()
        try:
            before = await client.count(
                collection_name=self.collection, count_filter=query_filter, exact=True
            )
            await client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True,
            )
            return before.count
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Qdrant delete failed: {exc}") from exc

    async def delete_document(self, doc_id: str, session_id: str) -> int:
        """Delete a document's vectors, scoped to its session so a stray call
        cannot wipe another tenant's data."""
        from qdrant_client import models

        if not await self._collection_ready():
            return 0

        return await self._delete_by_filter(
            models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(
                        key="session_id", match=models.MatchValue(value=session_id)
                    ),
                ]
            )
        )

    async def delete_session(self, session_id: str) -> int:
        """Hard delete for a whole session (§16 retention controls)."""
        from qdrant_client import models

        if not await self._collection_ready():
            return 0

        return await self._delete_by_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id", match=models.MatchValue(value=session_id)
                    )
                ]
            )
        )

    async def _collection_ready(self) -> bool:
        try:
            return await self._get_client().collection_exists(self.collection)
        except Exception:  # noqa: BLE001
            return False

    # --- ops -----------------------------------------------------------------

    async def count(self, scope: SearchFilter | None = None) -> int:
        client = self._get_client()
        if not await self._collection_ready():
            return 0
        result = await client.count(
            collection_name=self.collection,
            count_filter=self._build_filter(scope) if scope else None,
            exact=True,
        )
        return result.count

    async def health(self) -> bool:
        try:
            await self._get_client().get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False
