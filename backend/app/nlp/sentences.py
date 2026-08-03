"""Sentence segmentation for mixed Devanagari / Latin text (§11.2).

Hindi terminates sentences with the danda (।) and double danda (॥), not the
full stop, so a Latin-only splitter treats an entire Hindi paragraph as one
sentence — which then blows past the chunk ceiling and gets hard-split
mid-clause. This splitter handles both, and guards the usual full-stop false
positives (abbreviations, decimals, initials, ellipses).
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from app.nlp.script import DANDA, DOUBLE_DANDA

# Abbreviations after which a period is almost never a sentence end.
# Single letters are deliberately absent: uppercase initials ("R. K. Narayan")
# are handled by a dedicated rule, and listing lowercase letters here would
# wrongly glue "...at 3.5 p.m. The meeting..." into one sentence.
_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "viz",
        "cf",
        "al",
        "fig",
        "figs",
        "no",
        "nos",
        "vol",
        "vols",
        "pp",
        "ch",
        "sec",
        "secs",
        "art",
        "arts",
        "para",
        "paras",
        "rs",
        "approx",
        "dept",
        "govt",
        "ltd",
        "pvt",
        "inc",
        "corp",
        "co",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
)

# A terminator run, then whitespace. Devanagari dandas always terminate; the
# Latin set is validated further in `_is_boundary`.
_CANDIDATE_RE = regex.compile(
    rf"(?P<term>[.!?{DANDA}{DOUBLE_DANDA}]+)(?P<space>[ \t]*\n?[ \t]*|\s+)"
)

# A hard paragraph break is a sentence boundary regardless of punctuation —
# headings and list items in real documents frequently carry none.
_PARAGRAPH_RE = regex.compile(r"\n\s*\n")

_TRAILING_TOKEN_RE = regex.compile(r"([\p{L}\p{M}\d]+)[\"'”’)\]]*$")


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence plus its span in the source text, so chunks can report
    accurate `char_span` metadata (FR-9)."""

    text: str
    start: int
    end: int

    def __len__(self) -> int:
        return len(self.text)


def _is_boundary(text: str, term_start: int, terminator: str, following: str) -> bool:
    if terminator[0] in (DANDA, DOUBLE_DANDA):
        return True
    if terminator[0] in "!?":
        return True

    # Ellipsis mid-sentence ("wait ... then") is usually not a boundary.
    if terminator.startswith("..") and following and following[0].islower():
        return False

    preceding = text[:term_start]
    match = _TRAILING_TOKEN_RE.search(preceding)
    if match:
        token = match.group(1)
        if token.lower() in _ABBREVIATIONS:
            return False
        # Single capital letter => initial ("R. K. Narayan").
        if len(token) == 1 and token.isupper():
            return False
        # Decimal number: digit before and after the dot.
        if token.isdigit() and following[:1].isdigit():
            return False

    # A boundary must be followed by something that can start a sentence:
    # a capital, a digit, or any non-ASCII character (Devanagari has no case).
    if not following:
        return True
    return following[0].isupper() or following[0].isdigit() or not following[0].isascii()


def split_sentences(text: str, *, min_length: int = 1) -> list[Sentence]:
    """Split `text`, preserving offsets into the original string."""
    if not text or not text.strip():
        return []

    boundaries: set[int] = set()

    for match in _CANDIDATE_RE.finditer(text):
        end = match.end("term")
        following = text[match.end() :].lstrip()
        if _is_boundary(text, match.start("term"), match.group("term"), following):
            boundaries.add(end)

    for match in _PARAGRAPH_RE.finditer(text):
        boundaries.add(match.start())

    boundaries.add(len(text))

    sentences: list[Sentence] = []
    cursor = 0
    for boundary in sorted(boundaries):
        if boundary <= cursor:
            continue
        raw = text[cursor:boundary]
        stripped = raw.strip()
        if len(stripped) >= min_length:
            offset = raw.index(stripped) if stripped else 0
            start = cursor + offset
            sentences.append(Sentence(stripped, start, start + len(stripped)))
        cursor = boundary

    return sentences
