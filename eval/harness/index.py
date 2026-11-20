"""Build (or reuse) the Qdrant index the evaluation runs against.

Indexing 100+ chunks through bge-m3 costs about a minute on Metal, and a
chunk-size sweep re-indexes once per arm. The collection name therefore carries
the corpus fingerprint (content x chunking config), so:

* rerunning the same arm reuses the vectors instead of recomputing them;
* two sweep arms can never read each other's vectors, which would silently
  attribute one chunking config's recall to another.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.embeddings.base import EmbeddingProvider
from app.retrieval.rerank import NoOpReranker, Reranker
from app.retrieval.service import RetrievalService
from app.retrieval.vectorstore.qdrant_store import QdrantVectorStore

from .corpus import EvalCorpus


@dataclass(slots=True)
class Backends:
    """The models and store an evaluation run shares across every arm."""

    embedder: EmbeddingProvider
    reranker: Reranker
    store: QdrantVectorStore
    collection: str

    def service(self, *, rerank: bool) -> RetrievalService:
        return RetrievalService(
            embedder=self.embedder,
            store=self.store,
            reranker=self.reranker if rerank else NoOpReranker(),
        )


def build_embedder(kind: str) -> EmbeddingProvider:
    if kind == "mock":
        from app.retrieval.embeddings.mock import MockEmbeddingProvider

        return MockEmbeddingProvider()

    from app.retrieval.embeddings.bgem3 import BGEM3EmbeddingProvider

    embedder = BGEM3EmbeddingProvider()
    embedder.warmup()
    return embedder


def build_reranker(enabled: bool) -> Reranker:
    if not enabled:
        return NoOpReranker()

    from app.retrieval.rerank import BGEReranker

    reranker = BGEReranker()
    reranker.warmup()
    return reranker


async def ensure_index(
    corpus: EvalCorpus,
    embedder: EmbeddingProvider,
    *,
    prefix: str = "eval",
    force: bool = False,
) -> QdrantVectorStore:
    """Return a store whose collection holds exactly this corpus.

    Reuse is keyed on the corpus fingerprint *and* the chunk count, so a partial
    index left behind by an interrupted run is rebuilt rather than silently
    evaluated against.
    """
    collection = f"{prefix}_{embedder.name.replace('/', '_')}_{corpus.fingerprint}"
    store = QdrantVectorStore(collection=collection, dense_dim=embedder.dense_dim)

    if force:
        client = store._get_client()
        if await client.collection_exists(collection):
            await client.delete_collection(collection)

    await store.ensure_collection()
    existing = await store.count()

    if existing != len(corpus.chunks):
        if existing:
            client = store._get_client()
            await client.delete_collection(collection)
            await store.ensure_collection()
        service = RetrievalService(
            embedder=embedder, store=store, reranker=NoOpReranker()
        )
        await service.index_chunks(corpus.chunks)

    return store
