"""Reciprocal Rank Fusion.

Qdrant fuses the dense and sparse arms server-side, so this module is not in the
single-query path. It exists for the case Qdrant cannot see: merging results
from *two different queries* — the original Hinglish string and its Devanagari
transliteration (FR-15, §11.3) — where we need the union of candidates before
reranking.

RRF is rank-based on purpose. Dense cosine scores and sparse lexical scores live
on incomparable scales, so any weighted sum of raw scores is arbitrary; ranks
are comparable by construction.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.config import settings
from app.retrieval.types import ScoredChunk


def reciprocal_rank_fusion(
    rankings: Iterable[list[ScoredChunk]],
    *,
    k: int | None = None,
    weights: list[float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists.

    ``k`` damps the contribution of top ranks; the standard value of 60 stops a
    single list's first result from dominating the fused ordering.
    """
    smoothing = settings.retrieval.rrf_k if k is None else k
    materialised = list(rankings)

    if weights is not None and len(weights) != len(materialised):
        raise ValueError("weights must have one entry per ranking")

    # Drop empty rankings *with* their weights. An arm legitimately returns
    # nothing (a query with no lexical weights, a filter that excludes
    # everything), and dropping the list alone would desynchronise the pairs.
    paired = [
        (ranking, 1.0 if weights is None else weights[i])
        for i, ranking in enumerate(materialised)
        if ranking
    ]
    if not paired:
        return []

    lists = [ranking for ranking, _ in paired]
    weights = [weight for _, weight in paired]

    if len(lists) == 1 and all(w == weights[0] for w in weights):
        return lists[0][:limit] if limit else lists[0]

    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}

    for ranking, weight in zip(lists, weights, strict=True):
        for rank, chunk in enumerate(ranking):
            contribution = weight / (smoothing + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + contribution
            # Keep the instance with the strongest single-list score so the
            # per-arm score survives into the fused result for debugging.
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if limit:
        ordered = ordered[:limit]

    fused: list[ScoredChunk] = []
    for position, (chunk_id, fused_score) in enumerate(ordered):
        chunk = best[chunk_id]
        fused.append(
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=fused_score,
                retrieval_score=chunk.retrieval_score
                if chunk.retrieval_score is not None
                else chunk.score,
                rerank_score=chunk.rerank_score,
                filename=chunk.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_index=chunk.chunk_index,
                language=chunk.language,
                ocr_used=chunk.ocr_used,
                session_id=chunk.session_id,
                rank=position,
            )
        )
    return fused


def deduplicate(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Keep the highest-scoring occurrence of each chunk, preserving order."""
    seen: set[str] = set()
    unique: list[ScoredChunk] = []
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    return unique
