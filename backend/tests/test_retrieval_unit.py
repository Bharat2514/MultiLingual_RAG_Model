"""Retrieval units that need no vector store: fusion, reranking, wiring."""

from __future__ import annotations

import threading

import pytest

from app.retrieval import factory
from app.retrieval.embeddings.mock import MockEmbeddingProvider
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.rerank import NoOpReranker, Reranker
from app.retrieval.types import ScoredChunk


def chunk(chunk_id: str, score: float, text: str = "text") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text,
        score=score,
        filename="f.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )


class TestFactoryWiring:
    def test_vector_store_does_not_deadlock_on_first_call(self):
        """Regression: `get_vector_store` needs the embedder's dimensionality.
        With one shared non-reentrant lock it re-entered its own lock and the
        first search or delete of the process hung forever."""
        factory.reset()
        factory.use_mock_embeddings(True)

        done = threading.Event()

        def build():
            factory.get_vector_store()
            done.set()

        worker = threading.Thread(target=build, daemon=True)
        worker.start()

        assert done.wait(timeout=15), "get_vector_store() deadlocked"

    def test_singletons_are_reused(self):
        factory.reset()
        factory.use_mock_embeddings(True)
        assert factory.get_embedder() is factory.get_embedder()

    def test_mock_toggle_rebuilds_the_store(self):
        """The store caches the embedder's dimensionality, so switching the
        embedder must invalidate it or vectors get the wrong size."""
        factory.reset()
        factory.use_mock_embeddings(True)
        mock_store = factory.get_vector_store()
        assert mock_store.dense_dim == MockEmbeddingProvider().dense_dim

        factory.use_mock_embeddings(False)
        assert factory.get_vector_store() is not mock_store
        factory.reset()


class TestDeviceSelection:
    """Regression: Apple's Metal stack does not survive `fork()`. Celery's
    default prefork pool killed the worker with SIGABRT on the first MPS kernel
    launch — and because `resolve_device` was cached, the child inherited the
    parent's "mps" answer even after the guard was added."""

    def test_forked_child_never_selects_mps(self):
        import multiprocessing as mp

        from app.core.devices import resolve_device

        # Warm the parent's cache first — this is what made the bug subtle.
        resolve_device("auto")

        def child(queue):
            from app.core.devices import resolve_device as child_resolve

            queue.put((child_resolve("auto"), child_resolve("mps")))

        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        process = ctx.Process(target=child, args=(queue,))
        process.start()
        process.join(timeout=60)

        auto_device, explicit_device = queue.get(timeout=10)
        assert auto_device != "mps"
        assert explicit_device != "mps"

    def test_explicit_preference_is_honoured_in_the_main_process(self):
        from app.core.devices import resolve_device

        assert resolve_device("cpu") == "cpu"


class TestMockEmbeddings:
    def setup_method(self):
        self.provider = MockEmbeddingProvider()

    def test_dimension_is_stable(self):
        vectors = self.provider.embed_documents(["short", "a much longer piece of text here"])
        assert all(len(v.dense) == self.provider.dense_dim for v in vectors)

    def test_is_deterministic(self):
        first = self.provider.embed_query("same text")
        second = self.provider.embed_query("same text")
        assert first.dense == second.dense

    def test_similar_text_scores_higher_than_unrelated(self):
        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        query = self.provider.embed_query("annual compliance report").dense
        related = self.provider.embed_query("the annual compliance report summary").dense
        unrelated = self.provider.embed_query("zzz qqq vvv").dense

        assert cosine(query, related) > cosine(query, unrelated)

    def test_produces_sparse_vectors(self):
        embedding = self.provider.embed_query("clause 4.2 applies")
        assert embedding.sparse is not None
        assert len(embedding.sparse.indices) == len(embedding.sparse.values)


class TestReciprocalRankFusion:
    def test_agreement_across_lists_wins(self):
        """A chunk ranked well by both arms should beat one ranked first by a
        single arm — that is the point of fusion."""
        a = [chunk("x", 0.9), chunk("y", 0.8), chunk("z", 0.7)]
        b = [chunk("q", 0.95), chunk("y", 0.9), chunk("x", 0.5)]

        fused = reciprocal_rank_fusion([a, b])

        assert fused[0].chunk_id in {"x", "y"}
        assert {c.chunk_id for c in fused} == {"x", "y", "z", "q"}

    def test_rank_based_not_score_based(self):
        """Dense cosine and sparse lexical scores are on incomparable scales, so
        fusion must ignore magnitudes."""
        a = [chunk("a", 1000.0), chunk("b", 999.0)]
        b = [chunk("b", 0.002), chunk("a", 0.001)]

        fused = reciprocal_rank_fusion([a, b])
        assert abs(fused[0].score - fused[1].score) < 1e-9

    def test_single_list_passes_through(self):
        single = [chunk("a", 0.9), chunk("b", 0.8)]
        assert [c.chunk_id for c in reciprocal_rank_fusion([single])] == ["a", "b"]

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_respects_limit(self):
        a = [chunk(str(i), 1.0 - i / 10) for i in range(10)]
        assert len(reciprocal_rank_fusion([a, a], limit=3)) == 3

    def test_weights_shift_the_order(self):
        a = [chunk("a", 0.9)]
        b = [chunk("b", 0.9)]
        assert reciprocal_rank_fusion([a, b], weights=[5.0, 1.0])[0].chunk_id == "a"
        assert reciprocal_rank_fusion([a, b], weights=[1.0, 5.0])[0].chunk_id == "b"

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError, match="one entry per ranking"):
            reciprocal_rank_fusion([[chunk("a", 1.0)], [chunk("b", 1.0)]], weights=[1.0])


class TestDeduplicate:
    def test_keeps_the_best_occurrence(self):
        result = deduplicate([chunk("a", 0.5), chunk("a", 0.9), chunk("b", 0.7)])
        assert len(result) == 2
        assert next(c for c in result if c.chunk_id == "a").score == 0.9


class _ReversingReranker(Reranker):
    """Scores by inverse position, so reranking visibly reorders."""

    name = "test"

    def score(self, query: str, chunks: list[ScoredChunk]) -> list[float]:
        return [float(i) for i in range(len(chunks))]


class TestReranking:
    def test_reorders_and_preserves_the_original_score(self):
        candidates = [chunk("a", 0.9), chunk("b", 0.8), chunk("c", 0.7)]

        result = _ReversingReranker().rerank("q", candidates, top_n=3)

        assert [c.chunk_id for c in result] == ["c", "b", "a"]
        assert result[0].rerank_score == 2.0
        # The pre-rerank score survives for ablation comparison.
        assert result[0].retrieval_score == 0.7

    def test_truncates_to_top_n(self):
        candidates = [chunk(str(i), 1.0) for i in range(10)]
        assert len(_ReversingReranker().rerank("q", candidates, top_n=3)) == 3

    def test_ranks_are_renumbered(self):
        result = _ReversingReranker().rerank("q", [chunk("a", 1.0), chunk("b", 1.0)], top_n=2)
        assert [c.rank for c in result] == [0, 1]

    def test_threshold_never_empties_the_result(self):
        """The reranker must not decide 'not found' — that is the generation
        layer's grounding check (FR-23)."""
        result = _ReversingReranker().rerank(
            "q", [chunk("a", 1.0), chunk("b", 1.0)], top_n=5, min_score=1e9
        )
        assert len(result) == 2

    def test_threshold_falls_back_to_the_full_top_n(self):
        """When nothing clears the bar the model still gets top_n passages.

        Regression: falling back to a single chunk starved generation on any
        query the cross-encoder cannot score — document-level asks score around
        -8 — so it emitted the insufficient-context sentinel and the user was
        told their document did not contain its own summary.
        """
        candidates = [chunk(str(i), 1.0) for i in range(10)]
        result = _ReversingReranker().rerank("q", candidates, top_n=5, min_score=1e9)
        assert len(result) == 5

    def test_noop_reranker_preserves_order(self):
        candidates = [chunk("a", 0.9), chunk("b", 0.8)]
        result = NoOpReranker().rerank("q", candidates, top_n=5)
        assert [c.chunk_id for c in result] == ["a", "b"]

    def test_empty_candidates(self):
        assert _ReversingReranker().rerank("q", [], top_n=5) == []


class TestCitationLabels:
    def test_single_page(self):
        assert chunk("a", 1.0).citation_label == "[f.pdf p.1]"

    def test_page_range(self):
        c = chunk("a", 1.0)
        c.page_end = 3
        assert c.citation_label == "[f.pdf pp.1-3]"
