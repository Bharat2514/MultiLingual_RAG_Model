"""Component wiring (NFR-9).

Models are process-wide singletons: bge-m3 and the reranker are multi-gigabyte,
and loading one per request would exhaust memory. Everything is selected from
config so swapping the embedding model or disabling the reranker is a `.env`
change, never a code change.

Locking rule: each singleton has its **own** lock, and no initialiser calls
another `get_*` while holding one. A single shared `threading.Lock` deadlocked
here — `get_vector_store` needs the embedder's dimensionality, so it re-entered
a lock it already held and the first search of the process hung forever.
"""

from __future__ import annotations

import threading

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.embeddings.base import EmbeddingProvider
from app.retrieval.rerank import Reranker
from app.retrieval.vectorstore.base import VectorStore

log = get_logger(__name__)

_embedder: EmbeddingProvider | None = None
_reranker: Reranker | None = None
_store: VectorStore | None = None

_embedder_lock = threading.Lock()
_reranker_lock = threading.Lock()
_store_lock = threading.Lock()

#: Set by tests to force the deterministic stub without touching settings.
_force_mock_embeddings = False


def use_mock_embeddings(enabled: bool = True) -> None:
    global _force_mock_embeddings, _embedder, _store
    _force_mock_embeddings = enabled
    # The store caches the embedder's dimensionality, so it must be rebuilt too.
    _embedder = None
    _store = None


def get_embedder() -> EmbeddingProvider:
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        if _force_mock_embeddings or settings.embedding.model_name == "mock":
            from app.retrieval.embeddings.mock import MockEmbeddingProvider

            _embedder = MockEmbeddingProvider()
        else:
            from app.retrieval.embeddings.bgem3 import BGEM3EmbeddingProvider

            _embedder = BGEM3EmbeddingProvider()

        log.info("embedder_selected", provider=_embedder.name, dim=_embedder.dense_dim)
        return _embedder


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        if _reranker is not None:
            return _reranker

        from app.retrieval.rerank import BGEReranker, NoOpReranker

        if not settings.reranker.enabled:
            _reranker = NoOpReranker()
        else:
            try:
                _reranker = BGEReranker()
            except Exception:
                log.warning("reranker_unavailable_falling_back", exc_info=True)
                _reranker = NoOpReranker()

        log.info("reranker_selected", reranker=_reranker.name)
        return _reranker


def get_vector_store() -> VectorStore:
    global _store
    if _store is not None:
        return _store

    # Resolved *before* taking the store lock — see the module docstring.
    dense_dim = get_embedder().dense_dim

    with _store_lock:
        if _store is not None:
            return _store

        from app.retrieval.vectorstore.qdrant_store import QdrantVectorStore

        _store = QdrantVectorStore(dense_dim=dense_dim)
        return _store


def reset() -> None:
    """Drop cached singletons. Used by tests and by a config reload."""
    global _embedder, _reranker, _store
    _embedder = _reranker = _store = None
