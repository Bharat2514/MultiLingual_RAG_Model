"""Embedding provider interface (NFR-9).

Every implementation returns dense **and** sparse vectors from one call, because
that is what bge-m3 gives us natively and what hybrid search consumes. Providers
that cannot produce sparse vectors return `None` for that half, and the vector
store degrades to dense-only search rather than failing.

Documents and queries are embedded through separate methods: several
multilingual models (e5, and bge-m3 in some configurations) expect asymmetric
prefixes, and baking that into the provider keeps call sites honest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.types import Embedding


class EmbeddingProvider(ABC):
    """Base class for dense (+ optional sparse) text embedding."""

    #: Dimensionality of the dense vector.
    dense_dim: int = 1024
    #: Whether this provider emits learned sparse vectors.
    supports_sparse: bool = False
    #: Human-readable identifier recorded in eval runs and logs.
    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed chunk texts for indexing."""

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> list[Embedding]:
        """Embed user queries for search."""

    def embed_query(self, text: str) -> Embedding:
        return self.embed_queries([text])[0]

    def warmup(self) -> None:
        """Load weights eagerly so the first real request is not slow."""
        self.embed_queries(["warmup"])

    def close(self) -> None:  # pragma: no cover - most providers hold no handles
        return None
