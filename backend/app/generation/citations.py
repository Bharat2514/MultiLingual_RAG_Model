"""Citation extraction and the grounding post-check (FR-22, §11.4).

The model is *asked* to cite; this module verifies that it did, and that what it
cited actually exists. A hallucinated citation is worse than none — it looks
verifiable and is not — so every extracted reference is matched back to a chunk
that was really in the context, and unmatched ones are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from app.core.logging import get_logger
from app.retrieval.types import ScoredChunk

log = get_logger(__name__)

# [policy.pdf p.4] / [policy.pdf pp.4-6] / [report 2024.pdf p. 12]
_CITATION_RE = regex.compile(
    r"\[\s*(?P<file>[^\[\]]+?)\s+pp?\.\s*(?P<start>\d+)\s*(?:[-–]\s*(?P<end>\d+))?\s*\]"
)

# A bare [1] / [2] — models fall back to the passage number despite the
# instruction, and that is still recoverable because we know the ordering.
#
# Two variations show up often enough that missing them costs real citations,
# and an answer with no resolvable citation is reported to the user as
# ungrounded even when every claim in it is sourced:
#
#   [1, 4]            one claim drawing on several passages — the norm in a
#                     summary, where each bullet synthesises more than one;
#   [2 p.5, 3 p.19]   the passage number *and* the page, a half-compliance with
#                     the instructed [file p.N] form.
_INDEX_ENTRY = r"\d{1,2}(?:\s*pp?\.\s*\d+(?:\s*[-–]\s*\d+)?)?"
_INDEX_RE = regex.compile(rf"\[\s*(?P<indexes>{_INDEX_ENTRY}(?:\s*,\s*{_INDEX_ENTRY})*)\s*\]")
_INDEX_ENTRY_RE = regex.compile(
    r"(?P<index>\d{1,2})(?:\s*pp?\.\s*(?P<page>\d+)(?:\s*[-–]\s*\d+)?)?"
)


@dataclass(slots=True)
class Citation:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    snippet: str
    score: float
    #: 1-based position of this chunk in the CONTEXT block the model was given —
    #: the number it writes when it cites `[2]` instead of `[file p.N]`. Without
    #: it a client has no way to resolve a bare index: this list is deduplicated
    #: and re-sorted by (filename, page), so the Nth citation is not the Nth
    #: passage, and mapping one to the other links the reader to the wrong source.
    context_index: int = 0

    @property
    def label(self) -> str:
        return f"[{self.filename} p.{self.page}]"


def _snippet(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def extract_citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Resolve the citations in `answer` against the chunks actually supplied.

    Only references that match a real chunk are returned; anything else is a
    fabricated source and is dropped rather than shown to the user.
    """
    if not answer or not chunks:
        return []

    by_filename: dict[str, list[ScoredChunk]] = {}
    for chunk in chunks:
        by_filename.setdefault(chunk.filename.casefold(), []).append(chunk)

    position_in_context = {chunk.chunk_id: i for i, chunk in enumerate(chunks, start=1)}
    resolved: dict[str, Citation] = {}

    def add(chunk: ScoredChunk, page: int) -> None:
        if chunk.chunk_id in resolved:
            return
        resolved[chunk.chunk_id] = Citation(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=page,
            chunk_index=chunk.chunk_index,
            snippet=_snippet(chunk.text),
            score=round(chunk.score, 4),
            context_index=position_in_context.get(chunk.chunk_id, 0),
        )

    for match in _CITATION_RE.finditer(answer):
        filename = match.group("file").strip().casefold()
        page = int(match.group("start"))
        candidates = by_filename.get(filename)
        if not candidates:
            # Try a suffix match — models sometimes shorten a long filename.
            candidates = [
                c
                for key, group in by_filename.items()
                if key.endswith(filename) or filename.endswith(key)
                for c in group
            ]
        if not candidates:
            log.debug("citation_unresolved", filename=filename, page=page)
            continue

        # Prefer the chunk whose page range actually contains the cited page.
        exact = [c for c in candidates if c.page_start <= page <= c.page_end]
        add((exact or candidates)[0], page)

    if not resolved:
        # Fall back to bare passage numbers, which index into the context we built.
        for match in _INDEX_RE.finditer(answer):
            for raw in match.group("indexes").split(","):
                entry = _INDEX_ENTRY_RE.fullmatch(raw.strip())
                if entry is None:
                    continue
                position = int(entry.group("index"))
                if 1 <= position <= len(chunks):
                    chunk = chunks[position - 1]
                    page = entry.group("page")
                    add(chunk, int(page) if page else chunk.page_start)

    return sorted(resolved.values(), key=lambda c: (c.filename, c.page))


def strip_citation_markers(answer: str) -> str:
    """Answer text with citation brackets removed (for length/word counts)."""
    return _INDEX_RE.sub("", _CITATION_RE.sub("", answer)).strip()


def is_grounded(answer: str, citations: list[Citation], chunks: list[ScoredChunk]) -> bool:
    """Best-effort grounding verdict (§11.4 post-check).

    Context was supplied and the model produced substantive prose, so an answer
    with no resolvable citation cannot be verified — we mark it ungrounded and
    let the caller surface that, rather than presenting it as sourced.
    """
    if not chunks:
        return False
    if citations:
        return True
    return len(strip_citation_markers(answer)) < 40
