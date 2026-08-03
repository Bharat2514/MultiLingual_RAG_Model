"""Sentence-aware chunking (FR-8, §11.2).

Target 300–800 tokens with 10–15% overlap, never splitting mid-sentence where
avoidable. Chunks carry the page range and character span they came from, which
is what makes a citation clickable back to the source (FR-9, US3).

The packing rule: accumulate whole sentences until adding the next one would
exceed `max_tokens`; close the chunk once it is at least `target_tokens`. Only a
single sentence longer than `max_tokens` is ever hard-split, and then on word
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ChunkingSettings, settings
from app.core.logging import get_logger
from app.db.models import Language
from app.ingestion.types import PageText, TextChunk
from app.nlp.language import detect_language
from app.nlp.sentences import split_sentences
from app.nlp.tokens import count_tokens, count_tokens_cached

log = get_logger(__name__)

PAGE_SEPARATOR = "\n\n"


@dataclass(slots=True)
class _Unit:
    """A sentence positioned in the document-level text."""

    text: str
    start: int
    end: int
    page_no: int
    ocr_used: bool
    tokens: int


def _flatten(pages: list[PageText]) -> tuple[str, list[_Unit]]:
    """Concatenate pages into one string and segment it into positioned units.

    Offsets are document-level so a chunk's `char_span` is meaningful even when
    it straddles a page boundary.
    """
    parts: list[str] = []
    units: list[_Unit] = []
    cursor = 0

    for position, page in enumerate(pages):
        if position > 0:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)

        text = page.text
        parts.append(text)

        for sentence in split_sentences(text):
            units.append(
                _Unit(
                    text=sentence.text,
                    start=cursor + sentence.start,
                    end=cursor + sentence.end,
                    page_no=page.page_no,
                    ocr_used=page.ocr_used,
                    tokens=count_tokens_cached(sentence.text),
                )
            )
        cursor += len(text)

    return "".join(parts), units


def _split_oversized(unit: _Unit, max_tokens: int) -> list[_Unit]:
    """Hard-split a single sentence that exceeds the ceiling, on word bounds.

    Happens with unpunctuated OCR output and dense tabular rows.
    """
    words = unit.text.split()
    if len(words) <= 1:
        return [unit]

    pieces: list[_Unit] = []
    buffer: list[str] = []
    buffer_tokens = 0
    # Track position within the original sentence so spans stay accurate.
    offset = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens, offset
        if not buffer:
            return
        text = " ".join(buffer)
        start = unit.text.index(buffer[0], offset)
        end = start + len(text)
        pieces.append(
            _Unit(
                text,
                unit.start + start,
                unit.start + end,
                unit.page_no,
                unit.ocr_used,
                buffer_tokens,
            )
        )
        offset = end
        buffer = []
        buffer_tokens = 0

    for word in words:
        word_tokens = max(1, count_tokens_cached(word))
        if buffer and buffer_tokens + word_tokens > max_tokens:
            flush()
        buffer.append(word)
        buffer_tokens += word_tokens
    flush()

    return pieces or [unit]


def _overlap_units(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Trailing sentences of a chunk, totalling roughly `overlap_tokens`.

    Overlapping by whole sentences (rather than a fixed token window) keeps the
    guarantee that every chunk begins at a sentence boundary.
    """
    if overlap_tokens <= 0 or len(units) <= 1:
        return []

    carried: list[_Unit] = []
    total = 0
    # Never carry the whole chunk forward — that would fail to advance.
    for unit in reversed(units[1:]):
        if total >= overlap_tokens:
            break
        carried.insert(0, unit)
        total += unit.tokens
    return carried


def chunk_pages(
    pages: list[PageText],
    *,
    config: ChunkingSettings | None = None,
    detect_chunk_language: bool = True,
) -> list[TextChunk]:
    """Chunk cleaned pages into retrieval units."""
    cfg = config or settings.chunking
    overlap_tokens = int(cfg.target_tokens * cfg.overlap_ratio)

    _, units = _flatten(pages)
    if not units:
        return []

    # Expand any sentence that alone breaches the ceiling.
    expanded: list[_Unit] = []
    for unit in units:
        if unit.tokens > cfg.max_tokens:
            expanded.extend(_split_oversized(unit, cfg.max_tokens))
        else:
            expanded.append(unit)

    chunks: list[TextChunk] = []
    current: list[_Unit] = []
    current_tokens = 0

    def emit() -> None:
        nonlocal current, current_tokens
        if not current:
            return

        text = " ".join(u.text for u in current).strip()
        if not text:
            current, current_tokens = [], 0
            return

        language = Language.UNKNOWN
        if detect_chunk_language:
            # Documents are Hindi or English; Hinglish is a query-side register.
            language = detect_language(text, allow_hinglish=False).language

        chunks.append(
            TextChunk(
                index=len(chunks),
                text=text,
                page_start=min(u.page_no for u in current),
                page_end=max(u.page_no for u in current),
                char_start=current[0].start,
                char_end=current[-1].end,
                token_count=count_tokens(text),
                language=str(language),
                ocr_used=any(u.ocr_used for u in current),
            )
        )

        carried = _overlap_units(current, overlap_tokens)
        current = list(carried)
        current_tokens = sum(u.tokens for u in carried)

    for unit in expanded:
        # Close the chunk before overflowing, provided it is already usable.
        if current and current_tokens + unit.tokens > cfg.max_tokens:
            emit()
        current.append(unit)
        current_tokens += unit.tokens

        if current_tokens >= cfg.target_tokens:
            emit()

    # Flush the tail. A short remainder is merged back into the previous chunk
    # rather than emitted as a stub that would pollute retrieval.
    if current:
        tail_tokens = sum(u.tokens for u in current)
        carried_only = chunks and tail_tokens <= overlap_tokens
        if carried_only:
            current, current_tokens = [], 0
        elif tail_tokens < cfg.min_tokens and chunks:
            previous = chunks[-1]
            merged_text = f"{previous.text} {' '.join(u.text for u in current)}".strip()
            if count_tokens(merged_text) <= cfg.max_tokens * 1.25:
                chunks[-1] = TextChunk(
                    index=previous.index,
                    text=merged_text,
                    page_start=previous.page_start,
                    page_end=max(previous.page_end, max(u.page_no for u in current)),
                    char_start=previous.char_start,
                    char_end=current[-1].end,
                    token_count=count_tokens(merged_text),
                    language=previous.language,
                    ocr_used=previous.ocr_used or any(u.ocr_used for u in current),
                    section=previous.section,
                )
                current, current_tokens = [], 0
            else:
                emit()
        else:
            emit()

    log.debug(
        "chunked_document",
        pages=len(pages),
        sentences=len(expanded),
        chunks=len(chunks),
        avg_tokens=round(sum(c.token_count for c in chunks) / len(chunks), 1) if chunks else 0,
    )
    return chunks
