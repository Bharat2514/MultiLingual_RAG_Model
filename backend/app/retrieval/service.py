"""Retrieval orchestration: embed → hybrid search → rerank → top-N.

This is the §8.1 query path up to (not including) generation. It is written so
the ablation harness in §12.4 can disable each stage independently and quantify
what it contributes — `mode` selects dense/sparse/hybrid, and `rerank` toggles
the cross-encoder.
"""

from __future__ import annotations

import time

from app.core.config import settings
from app.core.errors import VectorStoreUnavailableError
from app.core.logging import get_logger
from app.retrieval import factory
from app.retrieval.coverage import merge_for_summary, select_coverage
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.types import RetrievalResult, ScoredChunk, SearchMode
from app.retrieval.vectorstore.base import ChunkPoint, PendingChunk, SearchFilter

log = get_logger(__name__)


class RetrievalService:
    """Stateless over the shared model singletons; safe to construct per request."""

    def __init__(self, *, embedder=None, store=None, reranker=None) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    @property
    def embedder(self):
        return self._embedder or factory.get_embedder()

    @property
    def store(self):
        return self._store or factory.get_vector_store()

    @property
    def reranker(self):
        return self._reranker or factory.get_reranker()

    async def retrieve(
        self,
        query: str,
        scope: SearchFilter,
        *,
        queries: list[str] | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        rerank: bool | None = None,
        coverage: bool = False,
    ) -> RetrievalResult:
        """Retrieve the chunks that should ground an answer.

        ``queries`` carries the Hinglish dual-query variants (FR-15): the
        original *and* its Devanagari transliteration are searched separately
        and their candidate lists fused, because either form alone can miss.

        ``coverage`` switches on the document-level path for aggregate queries
        (see `app.generation.intent`): the relevance hits are kept *and* joined
        by a sample spanning the documents in scope, because no ranking of
        passages can answer "what is this document about".
        """
        cfg = settings.retrieval
        k = top_k or cfg.top_k
        n = top_n or cfg.top_n
        should_rerank = cfg.hybrid_enabled if rerank is None else rerank
        timings: dict[str, float] = {}

        variants = [q for q in (queries or [query]) if q and q.strip()]
        if not variants:
            return RetrievalResult(mode=mode, timings_ms={})

        # --- embed -----------------------------------------------------------
        started = time.perf_counter()
        embeddings = self.embedder.embed_queries(variants)
        timings["embed"] = round((time.perf_counter() - started) * 1000, 2)

        # --- search ----------------------------------------------------------
        started = time.perf_counter()
        rankings: list[list[ScoredChunk]] = []
        for embedding in embeddings:
            rankings.append(await self.store.search(embedding, scope, limit=k, mode=mode))
        timings["retrieval"] = round((time.perf_counter() - started) * 1000, 2)

        candidates = (
            rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings, limit=k)
        )
        candidates = deduplicate(candidates)
        considered = len(candidates)

        if not candidates and not coverage:
            log.info("retrieval_empty", session_id=scope.session_id, mode=str(mode))
            return RetrievalResult(
                mode=mode, reranked=False, candidates_considered=0, timings_ms=timings
            )

        # --- rerank ----------------------------------------------------------
        reranked = False
        if candidates and should_rerank:
            started = time.perf_counter()
            try:
                candidates = self.reranker.rerank(
                    query, candidates, top_n=n, min_score=cfg.min_rerank_score
                )
                reranked = self.reranker.name != "none"
            except Exception:
                log.warning("rerank_failed_using_retrieval_order", exc_info=True)
                candidates = candidates[:n]
            timings["rerank"] = round((time.perf_counter() - started) * 1000, 2)
        else:
            candidates = candidates[:n]

        # --- coverage (document-level queries) --------------------------------
        if coverage:
            started = time.perf_counter()
            candidates = await self._add_coverage(candidates, scope)
            timings["coverage"] = round((time.perf_counter() - started) * 1000, 2)
            considered = max(considered, len(candidates))

        for position, chunk in enumerate(candidates):
            chunk.rank = position

        log.info(
            "retrieval_complete",
            session_id=scope.session_id,
            mode=str(mode),
            variants=len(variants),
            considered=considered,
            returned=len(candidates),
            reranked=reranked,
            coverage=coverage,
            timings_ms=timings,
        )
        return RetrievalResult(
            chunks=candidates,
            mode=mode,
            reranked=reranked,
            candidates_considered=considered,
            timings_ms=timings,
        )

    async def _add_coverage(
        self, relevant: list[ScoredChunk], scope: SearchFilter
    ) -> list[ScoredChunk]:
        """Join the relevance hits with a sample spanning the scoped documents.

        Best effort: if the scan fails we still have the relevance hits, and a
        thin summary beats no answer (NFR-5).
        """
        cfg = settings.retrieval
        try:
            everything = await self.store.list_chunks(scope, limit=cfg.summary_scan_limit)
        except Exception:
            log.warning("coverage_scan_failed_using_relevance_only", exc_info=True)
            return relevant

        if not everything:
            return relevant

        spanning = select_coverage(everything, limit=cfg.summary_max_chunks)
        merged = merge_for_summary(relevant, spanning, limit=cfg.summary_max_chunks)

        log.info(
            "coverage_selected",
            session_id=scope.session_id,
            scanned=len(everything),
            spanning=len(spanning),
            returned=len(merged),
        )
        return merged

    # --- indexing ------------------------------------------------------------

    async def index_chunks(self, pending: list[PendingChunk], batch_size: int = 32) -> int:
        """Embed and upsert chunks (FR-10, FR-11, FR-12)."""
        if not pending:
            return 0

        await self.store.ensure_collection()

        total = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            embeddings = self.embedder.embed_documents([p.text for p in batch])
            ready = [
                ChunkPoint.from_pending(chunk, embedding)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ]
            total += await self.store.upsert_chunks(ready)

        return total

    async def delete_document(self, doc_id: str, session_id: str) -> int:
        try:
            return await self.store.delete_document(doc_id, session_id)
        except VectorStoreUnavailableError:
            log.warning("vector_delete_failed", doc_id=doc_id, exc_info=True)
            return 0
