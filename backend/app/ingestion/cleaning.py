"""Text normalisation and cleanup (FR-7).

Four problems this solves, in the order they bite:

1. **Unicode form.** Devanagari has multiple encodings for the same grapheme
   (क + nukta vs क़). Without NFC the *same word* produces different embeddings
   and never matches. This is the single highest-impact fix for Hindi retrieval.
2. **Hyphenated line breaks.** PDF extraction yields "informa-\ntion", which
   tokenizes as two junk tokens.
3. **Repeated headers/footers.** Page furniture repeated 300 times pollutes
   every chunk and skews lexical retrieval.
4. **Ligature and whitespace noise** from PDF text layers.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

import regex

from app.core.logging import get_logger
from app.ingestion.types import PageText

log = get_logger(__name__)

# Zero-width and formatting characters that survive PDF extraction and break
# string equality. ZWNJ/ZWJ are excluded — they are *semantic* in Devanagari.
_INVISIBLE_RE = regex.compile(r"[­᠎​‎‏⁠﻿]")

# Latin ligatures PDF fonts emit as single codepoints.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

_NBSP_RE = regex.compile(r"[   ]")
_SPACE_RUN_RE = regex.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = regex.compile(r"\n{3,}")
_TRAILING_WS_RE = regex.compile(r"[ \t]+\n")

# "informa-\ntion" -> "information". Requires letters on both sides so that
# genuine compounds ("Indo-\nAryan") and numeric ranges survive.
_HYPHEN_BREAK_RE = regex.compile(r"(\p{Ll})[-‐‑]\s*\n\s*(\p{Ll})")

# A soft-wrapped line inside a paragraph: no terminator at the end, lowercase
# start on the next line. Joining these makes sentence splitting work.
_SOFT_WRAP_RE = regex.compile(r"(?<![.!?।॥:;\n])\n(?=[\p{Ll}ऀ-ॿ])")

_PAGE_NUMBER_RE = regex.compile(
    r"^\s*(?:page\s*)?[-–—\[(]*\s*(?:\d{1,4}|[ivxlcdm]{1,7}|[०-९]{1,4})\s*"
    r"(?:of\s*\d{1,4})?\s*[-–—\])]*\s*$",
    regex.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """NFC-normalise and strip invisible junk, preserving Devanagari ZWNJ/ZWJ."""
    if not text:
        return ""
    text = _INVISIBLE_RE.sub("", text)
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = _NBSP_RE.sub(" ", text)
    # NFC composes क + ़ into क़ so identical words hash and embed identically.
    return unicodedata.normalize("NFC", text)


def dehyphenate(text: str) -> str:
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def join_soft_wraps(text: str) -> str:
    """Reflow lines broken purely by column width."""
    return _SOFT_WRAP_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _SPACE_RUN_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def is_page_number(line: str) -> bool:
    """True for bare page numbers, including Devanagari digits."""
    stripped = line.strip()
    return bool(stripped) and len(stripped) <= 20 and bool(_PAGE_NUMBER_RE.match(stripped))


# Real headers/footers are short. Beyond this, a line is prose and must match
# *exactly* to be considered furniture — digit-blind matching on a sentence like
# "On page 4 the committee reviewed..." would classify body text as furniture
# and silently delete it.
_FURNITURE_MAX_WORDS_FOR_DIGIT_BLIND = 6
_FURNITURE_MAX_CHARS = 120

# If stripping would remove more than this share of a page, we assume the
# heuristic misfired and keep the page intact. Losing a header is cosmetic;
# losing the body is data loss.
_MAX_STRIP_RATIO = 0.5


def _normalize_for_comparison(line: str) -> str:
    """Comparison key for furniture matching.

    Short lines are compared digit-blind so "Page 4 of 300" matches "Page 5 of
    300". Longer lines keep their digits and must match exactly.
    """
    stripped = line.strip().casefold()
    if len(stripped.split()) <= _FURNITURE_MAX_WORDS_FOR_DIGIT_BLIND:
        return regex.sub(r"[\d०-९]+", "#", stripped)
    return stripped


def find_repeated_furniture(
    pages: list[PageText],
    *,
    edge_lines: int = 3,
    min_ratio: float = 0.4,
    min_pages: int = 4,
) -> set[str]:
    """Identify header/footer lines repeated across the document.

    Only the first and last few lines of each page are considered, so a phrase
    that legitimately recurs in body text is never stripped. Requires
    ``min_pages`` pages before running at all — on a 2-page document, "repeated
    on 40% of pages" is meaningless.
    """
    if len(pages) < min_pages:
        return set()

    counter: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        if not lines:
            continue
        candidates = lines[:edge_lines] + lines[-edge_lines:]
        # A line repeated twice on one page still counts once.
        for key in {
            _normalize_for_comparison(ln) for ln in candidates if len(ln) <= _FURNITURE_MAX_CHARS
        }:
            counter[key] += 1

    threshold = max(min_pages // 2, int(len(pages) * min_ratio))
    return {key for key, count in counter.items() if count >= threshold}


def strip_furniture(page: PageText, furniture: set[str], *, edge_lines: int = 3) -> str:
    """Remove known header/footer lines and bare page numbers from a page.

    Bails out if the result would discard most of the page — that means the
    repetition heuristic matched body text, and keeping a header is far cheaper
    than losing content.
    """
    lines = page.text.splitlines()
    if not lines:
        return page.text

    keep: list[str] = []
    last = len(lines) - 1
    for position, line in enumerate(lines):
        near_edge = position < edge_lines or position > last - edge_lines
        if (
            near_edge
            and line.strip()
            and len(line.strip()) <= _FURNITURE_MAX_CHARS
            and (_normalize_for_comparison(line) in furniture or is_page_number(line))
        ):
            continue
        keep.append(line)

    result = "\n".join(keep)

    original_len = len(page.text.strip())
    if original_len and len(result.strip()) < original_len * (1 - _MAX_STRIP_RATIO):
        log.warning(
            "furniture_strip_rejected_would_remove_body",
            page=page.page_no,
            original_chars=original_len,
            remaining_chars=len(result.strip()),
        )
        return page.text

    return result


def clean_page_text(text: str) -> str:
    """Full per-page normalisation pipeline."""
    text = normalize_unicode(text)
    text = dehyphenate(text)
    text = join_soft_wraps(text)
    return collapse_whitespace(text)


def clean_document(pages: list[PageText]) -> list[PageText]:
    """Clean every page and drop document-wide furniture.

    Returns new `PageText` objects; pages that become empty are kept (with an
    empty string) so page numbering stays aligned with the source PDF.
    """
    # Dehyphenate *before* furniture detection. A word broken across the last
    # line of a page ("informa-\ntion") otherwise presents as two short lines
    # at the page edge, and both halves get counted as footer candidates.
    normalized = [
        PageText(
            page_no=p.page_no,
            text=dehyphenate(normalize_unicode(p.text)),
            source=p.source,
            ocr_confidence=p.ocr_confidence,
            width=p.width,
            height=p.height,
        )
        for p in pages
    ]

    furniture = find_repeated_furniture(normalized)

    cleaned: list[PageText] = []
    for page in normalized:
        body = strip_furniture(page, furniture) if furniture else page.text
        body = join_soft_wraps(body)
        body = collapse_whitespace(body)
        cleaned.append(
            PageText(
                page_no=page.page_no,
                text=body,
                source=page.source,
                ocr_confidence=page.ocr_confidence,
                width=page.width,
                height=page.height,
            )
        )
    return cleaned
