"""Document-spanning chunk selection for aggregate queries.

Relevance ranking answers "which passage answers this question". A summary needs
the opposite: passages that *span* the document, in the order they were written,
so the model sees its shape rather than five near-duplicate hits on whatever the
query happened to lexically resemble.

Selection is deterministic — same document, same sample — so a repeated summary
request does not silently produce a different answer.
"""

from __future__ import annotations

from app.retrieval.types import ScoredChunk

#: Documents open with the material that describes them — title page, abstract,
#: introduction, executive summary. Those chunks are worth far more per token to
#: a summary than an evenly-strided one, so the front of each document is taken
#: whole before striding the remainder.
HEAD_CHUNKS = 3


def _sample_document(chunks: list[ScoredChunk], quota: int) -> list[ScoredChunk]:
    """Take `quota` chunks from one document: its head, then an even stride."""
    if quota >= len(chunks):
        return chunks

    head_n = min(HEAD_CHUNKS, quota)
    picked = list(chunks[:head_n])

    rest = chunks[head_n:]
    remaining = quota - head_n
    if remaining > 0 and rest:
        # remaining <= len(rest) here (quota < len(chunks)), so the stride is
        # never below 1 and the indices cannot collide.
        step = len(rest) / remaining
        picked.extend(rest[min(len(rest) - 1, int(i * step))] for i in range(remaining))

    return picked


def select_coverage(chunks: list[ScoredChunk], *, limit: int) -> list[ScoredChunk]:
    """Pick up to `limit` chunks spanning every document in `chunks`.

    Budget is split across documents in proportion to their length — a 300-page
    contract and a one-page cover letter should not get the same number of
    passages — with a floor of one chunk so no document in scope goes unmentioned.
    Results come back in reading order.
    """
    if limit <= 0 or not chunks:
        return []

    by_doc: dict[str, list[ScoredChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c.chunk_index)

    if len(chunks) <= limit:
        return _in_reading_order(chunks)

    # Stable ordering so the allocation below is deterministic.
    docs = sorted(by_doc.items(), key=lambda kv: (by_doc[kv[0]][0].filename, kv[0]))
    total = len(chunks)

    quotas: dict[str, int] = {}
    for doc_id, doc_chunks in docs:
        quotas[doc_id] = max(1, round(limit * len(doc_chunks) / total))

    # Proportional rounding (and the floor of 1) can overshoot or undershoot the
    # budget; settle up against the largest documents, which absorb it best.
    by_size = sorted(docs, key=lambda kv: len(kv[1]), reverse=True)
    while sum(quotas.values()) > limit:
        for doc_id, _ in by_size:
            if sum(quotas.values()) <= limit:
                break
            if quotas[doc_id] > 1:
                quotas[doc_id] -= 1
    while sum(quotas.values()) < limit:
        for doc_id, doc_chunks in by_size:
            if sum(quotas.values()) >= limit:
                break
            if quotas[doc_id] < len(doc_chunks):
                quotas[doc_id] += 1
        else:
            break  # every document is fully included; the budget cannot be spent

    selected: list[ScoredChunk] = []
    for doc_id, doc_chunks in docs:
        selected.extend(_sample_document(doc_chunks, quotas[doc_id]))

    return _in_reading_order(selected)


def merge_for_summary(
    relevant: list[ScoredChunk], spanning: list[ScoredChunk], *, limit: int
) -> list[ScoredChunk]:
    """Combine relevance hits with the coverage sample, in reading order.

    Both halves earn their place: the coverage sample gives the document's
    shape, while the relevance hits keep a scoped request ("summarise the
    payment terms") anchored on the passages that actually discuss it. The
    relevance hits are added first so they survive the `limit` cut.
    """
    merged: dict[str, ScoredChunk] = {}
    for chunk in [*relevant, *spanning]:
        if len(merged) >= limit:
            break
        merged.setdefault(chunk.chunk_id, chunk)

    return _in_reading_order(list(merged.values()))


def _in_reading_order(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Document order, not score order — a summary must read front to back."""
    ordered = sorted(chunks, key=lambda c: (c.filename, c.doc_id, c.chunk_index))
    for position, chunk in enumerate(ordered):
        chunk.rank = position
    return ordered
