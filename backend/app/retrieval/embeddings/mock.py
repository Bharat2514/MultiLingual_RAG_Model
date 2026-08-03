"""Deterministic embedding stub for tests and CI.

Not a semantic model, but not random either: vectors are derived from character
n-grams, so lexical overlap produces genuine similarity. That is enough to test
wiring, filtering, fusion, and isolation without downloading 2.5 GB of weights
or making the suite depend on a GPU.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from app.nlp.script import strip_to_words
from app.retrieval.embeddings.base import EmbeddingProvider
from app.retrieval.types import Embedding, SparseVector

_SPARSE_VOCAB = 2**16


def _token_id(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


class MockEmbeddingProvider(EmbeddingProvider):
    supports_sparse = True
    name = "mock"

    def __init__(self, dense_dim: int = 64) -> None:
        self.dense_dim = dense_dim

    def _dense(self, text: str) -> list[float]:
        """Hash character trigrams into a bag-of-ngrams vector, L2-normalised so
        cosine similarity behaves."""
        vector = [0.0] * self.dense_dim
        normalized = " ".join(strip_to_words(text.lower())) or text.lower()
        padded = f"  {normalized}  "
        for i in range(len(padded) - 2):
            trigram = padded[i : i + 3]
            slot = _token_id(trigram) % self.dense_dim
            vector[slot] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else [0.0] * self.dense_dim

    def _sparse(self, text: str) -> SparseVector | None:
        counts = Counter(strip_to_words(text.lower()))
        if not counts:
            return None
        # Deduplicate: two tokens can hash to the same slot.
        merged: dict[int, float] = {}
        for token, count in counts.items():
            slot = _token_id(token) % _SPARSE_VOCAB
            merged[slot] = merged.get(slot, 0.0) + float(count)
        indices = sorted(merged)
        return SparseVector(indices, [merged[i] for i in indices])

    def _embed(self, texts: list[str]) -> list[Embedding]:
        return [Embedding(dense=self._dense(t), sparse=self._sparse(t)) for t in texts]

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)

    def warmup(self) -> None:
        return None
