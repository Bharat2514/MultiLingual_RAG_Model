"""Citation extraction and the grounding post-check (FR-22, §11.4).

The model is *asked* to cite; this module verifies that it did, and that what it
cited actually exists. A hallucinated citation is worse than none — it looks
verifiable and is not — so every extracted reference is matched back to a chunk
that was really in the context, and unmatched ones are discarded.

**The citation contract is a passage ID, not a filename.** Each CONTEXT passage
is labelled `[S1]`, `[S2]`, … and the model cites those. Asking a model to copy
a filename is fragile in exactly the ways users name files — `2601.20882v1.pdf`
came back as `[5.pdf p.22]` (the passage number glued to the extension), Hindi
names get translated, long ones get shortened, brackets in a name end the
marker early. A two-character ID has nothing to mangle, and the server already
knows which file and page each ID stands for.

Older and non-compliant forms still resolve, because models drift:

* ``[S2]`` / ``[S1, S3]`` / ``[S2 p.5]``      — the contract;
* ``[policy.pdf p.4]`` / ``[policy.pdf pp.4-6]`` — the previous contract;
* ``[5.pdf p.22]``  — passage number fused with an extension; when no document
  is really called ``5.pdf``, it is passage 5;
* ``[2.pdf p.3, 5.pdf p.22]`` — several citations in one bracket;
* ``[2]`` / ``[1, 4]`` / ``[2 p.5]`` — bare passage numbers. Only used when
  nothing else in the answer resolved: a bare number in brackets is also how
  documents write footnotes and list items, so it is the weakest signal.

After resolution, :func:`canonicalize_citations` rewrites every resolved marker
to the ``[S2]`` form, so the claim check, the answer cache, share links and the
UI all see one format whatever the model wrote.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from app.core.logging import get_logger
from app.retrieval.types import ScoredChunk

log = get_logger(__name__)

# Any bracketed span on one line — each is then tried as a citation.
_BRACKET_RE = regex.compile(r"\[([^\[\]\n]{1,200})\]")

# The page part shared by every form: "p.4", "pp. 4-6", "page 22", ", p.3".
_PAGE = r"(?:\s*,)?\s*(?:pp?\.|pages?)\s*(?P<page>\d+)(?:\s*[-–]\s*\d+)?"

# S2 / S 2 / S2 p.5 — the passage-ID contract.
_ID_ENTRY_RE = regex.compile(rf"S\s*(?P<index>\d{{1,3}})(?:{_PAGE})?", regex.IGNORECASE)
# 2 / 2 p.5 — a bare passage number.
_INDEX_ENTRY_RE = regex.compile(rf"(?P<index>\d{{1,2}})(?:{_PAGE})?", regex.IGNORECASE)
# policy.pdf p.4 / report 2024.pdf p. 12 / policy.pdf, page 4
_FILE_ENTRY_RE = regex.compile(rf"(?P<file>.+?){_PAGE}", regex.IGNORECASE)
# "5.pdf" / "S5.pdf" — a passage number wearing a file extension.
_NUMBERED_FILE_RE = regex.compile(r"S?(?P<index>\d{1,3})\.[a-z0-9]{2,5}", regex.IGNORECASE)

_ENTRY_SEPARATOR_RE = regex.compile(r"\s*[,;]\s*")

# For the length check in `is_grounded`: the forms that need no chunk list to
# recognise. Filename markers are matched loosely — this only measures prose.
_STRIP_RE = regex.compile(
    rf"\[\s*(?:S\s*)?\d{{1,3}}(?:{_PAGE})?(?:\s*[,;]\s*(?:S\s*)?\d{{1,3}}(?:{_PAGE})?)*\s*\]"
    rf"|\[[^\[\]\n]+?{_PAGE}\s*\]",
    regex.IGNORECASE,
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
    #: the N in the `[SN]` it cites. Without it a client has no way to resolve
    #: a marker: this list is deduplicated and re-sorted by (filename, page), so
    #: the Nth citation is not the Nth passage, and mapping one to the other
    #: links the reader to the wrong source.
    context_index: int = 0

    @property
    def label(self) -> str:
        return f"[{self.filename} p.{self.page}]"


def passage_id(position: int) -> str:
    """The ID a CONTEXT passage is labelled with, 1-based: ``S1``, ``S2``, …"""
    return f"S{position}"


def _snippet(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


# A resolved entry: (position in the context, 1-based; the page to show).
_Ref = tuple[int, int]


class _Resolver:
    """Resolves the text inside one ``[...]`` against the context passages."""

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks
        self.by_filename: dict[str, list[int]] = {}
        for position, chunk in enumerate(chunks, start=1):
            self.by_filename.setdefault(chunk.filename.casefold(), []).append(position)

    def _position(self, index: str) -> int | None:
        position = int(index)
        return position if 1 <= position <= len(self.chunks) else None

    def _page_for(self, position: int, page: str | None) -> int:
        # The page the model wrote is kept as the citation's page (a passage can
        # span several), falling back to where the passage starts.
        return int(page) if page else self.chunks[position - 1].page_start

    def _passage(self, match: regex.Match[str]) -> _Ref | None:
        position = self._position(match.group("index"))
        if position is None:
            return None
        return position, self._page_for(position, match.group("page"))

    def _file(self, name: str, page: int) -> _Ref | None:
        key = name.strip().strip("\"'“”‘’").casefold()
        if not key:
            return None
        positions = self.by_filename.get(key)

        if not positions:
            # "5.pdf" when no document has that name: the model fused passage 5
            # with an extension. Checked before the suffix match below, which
            # would otherwise pin "5.pdf" on any file that happens to end in 5.
            numbered = _NUMBERED_FILE_RE.fullmatch(key)
            if numbered:
                position = self._position(numbered.group("index"))
                if position is not None:
                    chunk = self.chunks[position - 1]
                    in_range = chunk.page_start <= page <= chunk.page_end
                    return position, page if in_range else chunk.page_start

            # Models sometimes shorten a long filename.
            positions = [
                p
                for name_key, group in self.by_filename.items()
                if name_key.endswith(key) or key.endswith(name_key)
                for p in group
            ]
        if not positions:
            log.debug("citation_unresolved", filename=key, page=page)
            return None

        # Prefer the chunk whose page range actually contains the cited page.
        exact = [
            p
            for p in positions
            if self.chunks[p - 1].page_start <= page <= self.chunks[p - 1].page_end
        ]
        return (exact or positions)[0], page

    def entry(self, text: str, *, allow_bare: bool) -> _Ref | None:
        text = text.strip()
        if not text:
            return None
        if match := _ID_ENTRY_RE.fullmatch(text):
            return self._passage(match)
        if allow_bare and (match := _INDEX_ENTRY_RE.fullmatch(text)):
            # Before the file form, which would read "2 p.5" as a file named "2".
            return self._passage(match)
        if match := _FILE_ENTRY_RE.fullmatch(text):
            return self._file(match.group("file"), int(match.group("page")))
        if (match := _NUMBERED_FILE_RE.fullmatch(text)) and text.casefold() not in self.by_filename:
            # "[5.pdf]" with no page at all.
            return self._passage(match)
        return None

    def bracket(self, inner: str, *, allow_bare: bool) -> list[_Ref]:
        # The whole bracket first: a real filename can contain a comma.
        whole = self.entry(inner, allow_bare=allow_bare)
        if whole is not None:
            return [whole]
        parts = _ENTRY_SEPARATOR_RE.split(inner)
        if len(parts) < 2:
            return []
        refs = [self.entry(part, allow_bare=allow_bare) for part in parts]
        return [r for r in refs if r is not None]


def _scan(answer: str, chunks: list[ScoredChunk]) -> list[tuple[regex.Match[str], list[_Ref]]]:
    """Every bracket in `answer` that resolves, with what it resolves to."""
    if not answer or not chunks:
        return []
    resolver = _Resolver(chunks)
    brackets = list(_BRACKET_RE.finditer(answer))

    found = [(m, resolver.bracket(m.group(1), allow_bare=False)) for m in brackets]
    found = [(m, refs) for m, refs in found if refs]
    if found:
        return found

    # Nothing in the instructed or file forms: fall back to bare passage numbers.
    found = [(m, resolver.bracket(m.group(1), allow_bare=True)) for m in brackets]
    return [(m, refs) for m, refs in found if refs]


def extract_citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Resolve the citations in `answer` against the chunks actually supplied.

    Only references that match a real chunk are returned; anything else is a
    fabricated source and is dropped rather than shown to the user.
    """
    resolved: dict[str, Citation] = {}
    for _, refs in _scan(answer, chunks):
        for position, page in refs:
            chunk = chunks[position - 1]
            if chunk.chunk_id in resolved:
                continue
            resolved[chunk.chunk_id] = Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page=page,
                chunk_index=chunk.chunk_index,
                snippet=_snippet(chunk.text),
                score=round(chunk.score, 4),
                context_index=position,
            )
    return sorted(resolved.values(), key=lambda c: (c.filename, c.page))


def canonicalize_citations(answer: str, chunks: list[ScoredChunk]) -> str:
    """Rewrite every resolved citation marker to the ``[S2]`` / ``[S1, S3]`` form.

    Whatever mix of forms the model produced, downstream consumers — the claim
    check, the cache, share links, the UI's chip renderer — then parse exactly
    one. Entries in a bracket that did not resolve are dropped from it (they
    point at nothing); brackets with no resolvable entry are left untouched,
    since "[sic]" and "[Rule 3(a)]" are ordinary text.
    """
    found = _scan(answer, chunks)
    if not found:
        return answer
    out: list[str] = []
    cursor = 0
    for match, refs in found:
        positions = list(dict.fromkeys(position for position, _ in refs))
        out.append(answer[cursor : match.start()])
        out.append("[" + ", ".join(passage_id(p) for p in positions) + "]")
        cursor = match.end()
    out.append(answer[cursor:])
    return "".join(out)


def strip_citation_markers(answer: str) -> str:
    """Answer text with citation brackets removed (for length/word counts)."""
    return _STRIP_RE.sub("", answer).strip()


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
