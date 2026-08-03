"""Language identification for chunks and queries (FR-6, FR-14).

Script does most of the work; `lingua` disambiguates the Latin-script case,
which is where the interesting decision lives: English vs romanized Hindi
(Hinglish). Statistical LID alone is unreliable on short, code-mixed strings
like "mujhe iska summary do", so we combine it with a Hinglish function-word
lexicon.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from app.core.logging import get_logger
from app.db.models import Language
from app.nlp.script import Script, profile_script, strip_to_words

log = get_logger(__name__)

# High-frequency romanized Hindi function words. Function words are the right
# signal: they survive code-mixing, whereas content words are often English
# even in heavily Hinglish sentences.
HINGLISH_MARKERS = frozenset(
    [
        "hai",
        "hain",
        "tha",
        "the",
        "thi",
        "ho",
        "hoga",
        "hogi",
        "hona",
        "hote",
        "hota",
        "hoti",
        "raha",
        "rahi",
        "rahe",
        "kya",
        "kyu",
        "kyun",
        "kaise",
        "kaun",
        "kab",
        "kahan",
        "kitna",
        "kitne",
        "kitni",
        "konsa",
        "kaunsa",
        "mujhe",
        "muje",
        "mera",
        "meri",
        "mere",
        "hum",
        "hume",
        "humein",
        "tum",
        "tumhe",
        "aap",
        "aapko",
        "aapka",
        "aapki",
        "uska",
        "uski",
        "uske",
        "iska",
        "iski",
        "iske",
        "unka",
        "unki",
        "jinka",
        "jiska",
        "nahi",
        "nahin",
        "haan",
        "han",
        "bhi",
        "hi",
        "to",
        "toh",
        "se",
        "ka",
        "ki",
        "ke",
        "ko",
        "me",
        "mein",
        "par",
        "pe",
        "aur",
        "ya",
        "kar",
        "karo",
        "karna",
        "karne",
        "karta",
        "karti",
        "kiya",
        "kijiye",
        "dijiye",
        "dena",
        "do",
        "de",
        "diya",
        "batao",
        "bataye",
        "samjhao",
        "likho",
        "padho",
        "dekho",
        "sunao",
        "chahiye",
        "milta",
        "milti",
        "accha",
        "achha",
        "thik",
        "theek",
        "bilkul",
        "zyada",
        "kam",
        "thoda",
        "bahut",
        "sab",
        "kuch",
        "sabhi",
        "iske",
        "uske",
        "jaise",
        "waise",
        "lekin",
        "magar",
        "agar",
        "phir",
        "abhi",
        "kabhi",
        "hamesha",
    ]
)

# Words that are Hinglish markers but also ordinary English words. Counting
# these would classify plain English as Hinglish, so they are excluded.
_ENGLISH_COLLISIONS = frozenset({"hi", "to", "me", "the", "he", "so", "his", "him", "her", "no"})

_SAFE_MARKERS = HINGLISH_MARKERS - _ENGLISH_COLLISIONS

# Above this share of Hinglish markers, Latin-script text is romanized Hindi.
HINGLISH_MARKER_THRESHOLD = 0.12


@dataclass(frozen=True, slots=True)
class LanguageResult:
    language: Language
    script: Script
    confidence: float
    hinglish_ratio: float = 0.0

    @property
    def is_devanagari(self) -> bool:
        return self.script == Script.DEVANAGARI


@functools.lru_cache(maxsize=1)
def _detector():
    """Lazily build the lingua detector — construction loads language models."""
    try:
        from lingua import Language as LinguaLanguage
        from lingua import LanguageDetectorBuilder

        return (
            LanguageDetectorBuilder.from_languages(LinguaLanguage.ENGLISH, LinguaLanguage.HINDI)
            .with_preloaded_language_models()
            .build()
        )
    except Exception:
        log.warning("lingua_unavailable_using_script_heuristics", exc_info=True)
        return None


def hinglish_marker_ratio(text: str) -> float:
    """Share of tokens that are unambiguous romanized-Hindi function words."""
    words = [w.lower() for w in strip_to_words(text)]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _SAFE_MARKERS)
    return hits / len(words)


def detect_language(text: str, *, allow_hinglish: bool = True) -> LanguageResult:
    """Classify `text` as Hindi, English, Hinglish, or mixed.

    ``allow_hinglish`` is False for document chunks — a scanned circular is
    Hindi or English, never Hinglish — and True for user queries, where
    romanized input is the whole point (FR-14).
    """
    profile = profile_script(text)

    if profile.script == Script.UNKNOWN:
        return LanguageResult(Language.UNKNOWN, Script.UNKNOWN, 0.0)

    if profile.script == Script.DEVANAGARI:
        return LanguageResult(Language.HI, Script.DEVANAGARI, profile.devanagari_ratio)

    # Any Devanagari at all means the text is Hindi-bearing, even when the
    # character-level classifier called it Latin because Devanagari fell below
    # the mixed-script threshold. Falling through to the English/Hinglish branch
    # with Devanagari present would mislabel the text outright.
    if profile.script == Script.MIXED or profile.devanagari_chars > 0:
        return LanguageResult(Language.MIXED, Script.MIXED, profile.devanagari_ratio)

    # --- Latin script: English or Hinglish -----------------------------------
    ratio = hinglish_marker_ratio(text)
    if allow_hinglish and ratio >= HINGLISH_MARKER_THRESHOLD:
        return LanguageResult(Language.HINGLISH, Script.LATIN, min(1.0, ratio * 4), ratio)

    detector = _detector()
    if detector is not None:
        try:
            from lingua import Language as LinguaLanguage

            confidences = detector.compute_language_confidence_values(text)
            if confidences:
                best = confidences[0]
                if best.language == LinguaLanguage.HINDI and allow_hinglish:
                    # Romanized Hindi that our lexicon missed.
                    return LanguageResult(Language.HINGLISH, Script.LATIN, best.value, ratio)
                return LanguageResult(Language.EN, Script.LATIN, best.value, ratio)
        except Exception:
            log.warning("lingua_detection_failed", exc_info=True)

    return LanguageResult(Language.EN, Script.LATIN, profile.latin_ratio, ratio)


def dominant_language(counts: dict[str, int]) -> Language:
    """Document-level primary language from its per-chunk distribution."""
    meaningful = {k: v for k, v in counts.items() if k != Language.UNKNOWN and v > 0}
    if not meaningful:
        return Language.UNKNOWN
    return Language(max(meaningful, key=lambda k: meaningful[k]))
