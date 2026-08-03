"""Script detection for Devanagari / Latin text.

Script is cheaper and far more reliable than language identification, and it is
what actually drives our routing decisions: Devanagari means Hindi, Latin means
English *or* Hinglish, and the mix ratio distinguishes the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import regex

# Devanagari (U+0900–U+097F), Devanagari Extended (U+A8E0–U+A8FF) and the Vedic
# Extensions (U+1CD0–U+1CFF) that occasionally appear in scanned Hindi texts.
DEVANAGARI_RANGES = r"ऀ-ॿ꣠-ꣿ᳐-᳿"

_DEVANAGARI_RE = regex.compile(f"[{DEVANAGARI_RANGES}]")
_LATIN_RE = regex.compile(r"[A-Za-z]")
_WORD_RE = regex.compile(r"[\p{L}\p{M}]+")

# The Devanagari danda and double danda are sentence terminators, not letters.
DANDA = "।"
DOUBLE_DANDA = "॥"


class Script(StrEnum):
    DEVANAGARI = "devanagari"
    LATIN = "latin"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScriptProfile:
    """Character-level script makeup of a span of text."""

    script: Script
    devanagari_chars: int
    latin_chars: int
    total_letters: int

    @property
    def devanagari_ratio(self) -> float:
        return self.devanagari_chars / self.total_letters if self.total_letters else 0.0

    @property
    def latin_ratio(self) -> float:
        return self.latin_chars / self.total_letters if self.total_letters else 0.0


def profile_script(text: str, *, mixed_threshold: float = 0.15) -> ScriptProfile:
    """Classify `text` by script.

    ``mixed_threshold`` is the minority-script share above which we stop calling
    the text single-script. 15% is deliberately generous: Hindi government
    documents routinely carry English clause IDs, dates, and acronyms without
    ceasing to be Hindi documents.
    """
    devanagari = len(_DEVANAGARI_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = devanagari + latin

    if total == 0:
        return ScriptProfile(Script.UNKNOWN, 0, 0, 0)

    minority = min(devanagari, latin) / total
    if minority >= mixed_threshold:
        script = Script.MIXED
    elif devanagari > latin:
        script = Script.DEVANAGARI
    else:
        script = Script.LATIN

    return ScriptProfile(script, devanagari, latin, total)


def detect_script(text: str) -> Script:
    return profile_script(text).script


def has_devanagari(text: str) -> bool:
    return _DEVANAGARI_RE.search(text) is not None


def strip_to_words(text: str) -> list[str]:
    """Letter-only tokens (marks kept, so Devanagari matras stay attached)."""
    return _WORD_RE.findall(text)
