"""Integration tests against a live Qdrant.

Marked `integration` — run with `make test-all` after `make infra`. These use
the deterministic mock embedder so they test the *store and retrieval wiring*
(filtering, fusion, isolation, deletion) without depending on model quality,
which is the eval harness's job.
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.embeddings.mock import MockEmbeddingProvider
from app.retrieval.rerank import NoOpReranker
from app.retrieval.service import RetrievalService
from app.retrieval.types import SearchMode
from app.retrieval.vectorstore.base import PendingChunk, SearchFilter
from app.retrieval.vectorstore.qdrant_store import QdrantVectorStore

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

CORPUS = [
    ("The annual compliance report must be filed within thirty days.", "compliance.pdf", 1),
    ("Officers who fail to comply face disciplinary proceedings.", "compliance.pdf", 2),
    ("भारत सरकार ने नई शिक्षा नीति की घोषणा की है।", "policy.pdf", 1),
    ("शिक्षा के क्षेत्र में सुधार के लिए यह नीति महत्वपूर्ण है।", "policy.pdf", 2),
    ("Clause 4.2 governs the renewal of district licences.", "licences.pdf", 1),
]


@pytest.fixture
async def store():
    """A throwaway collection per test, dropped afterwards."""
    embedder = MockEmbeddingProvider()
    collection = f"test_{uuid.uuid4().hex[:12]}"
    vector_store = QdrantVectorStore(collection=collection, dense_dim=embedder.dense_dim)

    if not await vector_store.health():
        pytest.skip("Qdrant is not reachable — run `make infra`")

    await vector_store.ensure_collection()
    try:
        yield vector_store
    finally:
        client = vector_store._get_client()
        try:
            await client.delete_collection(collection)
        finally:
            await vector_store.close()


def _pending(session_id: str, doc_suffix: str = "") -> list[PendingChunk]:
    return [
        PendingChunk(
            chunk_id=f"{session_id}_{i}{doc_suffix}",
            doc_id=f"doc_{filename}{doc_suffix}",
            session_id=session_id,
            text=text,
            filename=filename,
            page_start=page,
            page_end=page,
            chunk_index=i,
            language="hi" if any(ord(c) > 2300 for c in text) else "en",
            ocr_used=False,
        )
        for i, (text, filename, page) in enumerate(CORPUS)
    ]


@pytest.fixture
async def service(store):
    svc = RetrievalService(embedder=MockEmbeddingProvider(), store=store, reranker=NoOpReranker())
    await svc.index_chunks(_pending("sess_a"))
    return svc


class TestIndexing:
    async def test_indexes_every_chunk(self, store):
        indexed = await RetrievalService(
            embedder=MockEmbeddingProvider(), store=store, reranker=NoOpReranker()
        ).index_chunks(_pending("sess_a"))

        assert indexed == len(CORPUS)
        assert await store.count() == len(CORPUS)

    async def test_reindexing_overwrites_rather_than_duplicates(self, store):
        svc = RetrievalService(
            embedder=MockEmbeddingProvider(), store=store, reranker=NoOpReranker()
        )
        await svc.index_chunks(_pending("sess_a"))
        await svc.index_chunks(_pending("sess_a"))

        # Point ids are derived from chunk_id, so a re-ingest replaces points.
        assert await store.count() == len(CORPUS)


class TestSearchModes:
    @pytest.mark.parametrize("mode", [SearchMode.DENSE, SearchMode.SPARSE, SearchMode.HYBRID])
    async def test_each_mode_returns_hits(self, service, mode):
        result = await service.retrieve(
            "compliance report filing deadline",
            SearchFilter(session_id="sess_a"),
            mode=mode,
            rerank=False,
        )
        assert not result.is_empty

    async def test_lexical_term_is_retrievable(self, service):
        """Exact identifiers (clause ids, numbers) are what sparse retrieval
        exists for — pure dense search tends to miss them."""
        result = await service.retrieve(
            "Clause 4.2", SearchFilter(session_id="sess_a"), mode=SearchMode.SPARSE, rerank=False
        )
        assert any("4.2" in c.text for c in result.chunks)

    async def test_hybrid_reports_its_mode_and_timings(self, service):
        result = await service.retrieve(
            "compliance", SearchFilter(session_id="sess_a"), rerank=False
        )
        assert result.mode == SearchMode.HYBRID
        assert {"embed", "retrieval"} <= set(result.timings_ms)


class TestScoping:
    async def test_doc_id_filter_restricts_results(self, service):
        result = await service.retrieve(
            "report",
            SearchFilter(session_id="sess_a", doc_ids=["doc_compliance.pdf"]),
            rerank=False,
        )
        assert result.chunks
        assert all(c.doc_id == "doc_compliance.pdf" for c in result.chunks)

    async def test_language_filter(self, service):
        result = await service.retrieve(
            "नीति", SearchFilter(session_id="sess_a", languages=["hi"]), rerank=False
        )
        assert all(c.language == "hi" for c in result.chunks)


class TestSessionIsolation:
    """NFR-6 / R8 enforced at the vector layer, not just the API."""

    async def test_another_session_retrieves_nothing(self, service, store):
        await service.index_chunks(_pending("sess_b", doc_suffix="_b"))

        result = await service.retrieve(
            "compliance report", SearchFilter(session_id="sess_intruder"), rerank=False
        )

        assert result.is_empty

    async def test_results_never_cross_sessions(self, service, store):
        await service.index_chunks(_pending("sess_b", doc_suffix="_b"))

        for session_id in ("sess_a", "sess_b"):
            result = await service.retrieve(
                "compliance report disciplinary",
                SearchFilter(session_id=session_id),
                top_n=20,
                rerank=False,
            )
            assert result.chunks
            assert all(c.session_id == session_id for c in result.chunks), session_id


class TestDeletion:
    async def test_deletes_only_that_document(self, service, store):
        before = await store.count()

        removed = await service.delete_document("doc_compliance.pdf", "sess_a")

        assert removed == 2
        assert await store.count() == before - 2

    async def test_will_not_delete_another_sessions_document(self, service, store):
        """A doc_id alone must not be enough — the session must match too."""
        before = await store.count()

        removed = await service.delete_document("doc_compliance.pdf", "sess_wrong")

        assert removed == 0
        assert await store.count() == before

    async def test_deleted_content_is_no_longer_retrievable(self, service):
        await service.delete_document("doc_compliance.pdf", "sess_a")

        result = await service.retrieve(
            "compliance report", SearchFilter(session_id="sess_a"), rerank=False
        )
        assert all(c.doc_id != "doc_compliance.pdf" for c in result.chunks)

    async def test_delete_session_removes_everything(self, service, store):
        await service.index_chunks(_pending("sess_b", doc_suffix="_b"))

        removed = await store.delete_session("sess_a")

        assert removed == len(CORPUS)
        assert await store.count(SearchFilter(session_id="sess_a")) == 0
        assert await store.count(SearchFilter(session_id="sess_b")) == len(CORPUS)


class TestDualQueryFusion:
    async def test_fuses_query_variants(self, service):
        """FR-15: the Hinglish original and its transliteration are searched
        separately and their candidates fused."""
        result = await service.retrieve(
            "shiksha niti",
            SearchFilter(session_id="sess_a"),
            queries=["shiksha niti", "शिक्षा नीति"],
            rerank=False,
        )
        assert not result.is_empty


class TestEventLoopReuse:
    async def test_store_survives_a_new_event_loop(self, store):
        """Regression: the Celery worker calls `asyncio.run()` per task, so the
        cached async client meets a fresh loop each time."""
        import asyncio

        assert await store.health()

        def in_a_new_loop() -> bool:
            return asyncio.run(store.health())

        assert await asyncio.to_thread(in_a_new_loop)
