"""Small text helpers shared by the intelligence modules."""

from __future__ import annotations

import unicodedata

import regex

#: Devanagari digits ०-९ map one-to-one onto ASCII, so translating them keeps
#: every character offset intact — detectors can run on the translated text and
#: report spans that are valid in the original.
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_WS_RE = regex.compile(r"\s+")
_NUMBER_RE = regex.compile(r"(?<![\p{L}\d])\d[\d,]*(?:\.\d+)?(?![\d])")
_WORD_RE = regex.compile(r"[\p{L}\p{M}\d]+")


def ascii_digits(text: str) -> str:
    return text.translate(DEVANAGARI_DIGITS)


def normalize(text: str) -> str:
    """NFC, ASCII digits, collapsed whitespace, casefolded — for comparisons."""
    text = unicodedata.normalize("NFC", text or "")
    return _WS_RE.sub(" ", ascii_digits(text)).strip().casefold()


def numbers_in(text: str) -> list[str]:
    """Numbers as canonical strings: `5,000` and `५०००` both become `5000`.

    Indian grouping (`1,00,000`) is handled by dropping separators rather than
    parsing them, which is exactly what makes it compare equal to `100000`.
    """
    out: list[str] = []
    for match in _NUMBER_RE.finditer(ascii_digits(text or "")):
        raw = match.group(0).replace(",", "").rstrip(".")
        if not raw:
            continue
        if "." in raw:
            raw = raw.rstrip("0").rstrip(".")
        out.append(raw.lstrip("0") or "0")
    return out


def words(text: str) -> list[str]:
    return [w.casefold() for w in _WORD_RE.findall(ascii_digits(text or ""))]


def contains_quote(haystack: str, quote: str, *, min_chars: int = 8) -> bool:
    """Whether `quote` appears in `haystack`, ignoring case and whitespace.

    Used to verify that a span the model claims to have read is really in the
    source. Very short quotes are rejected — "the" is in every document.
    """
    needle = normalize(quote)
    if len(needle) < min_chars:
        return False
    return needle in normalize(haystack)
