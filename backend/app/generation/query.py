"""Query understanding: detect the register, build the search variants (FR-14/15).

Two jobs:

1. Work out what language and register the user wrote in, because the answer
   must come back in the same one (FR-21).
2. Produce the query variants to search. For Hinglish that means the original
   *and* its Devanagari transliteration — either alone misses content, so both
   are searched and the candidate lists fused (§11.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Language
from app.generation.intent import QueryIntent, detect_intent
from app.nlp.language import detect_language
from app.nlp.script import Script
from app.nlp.transliterate import lexicon_coverage, to_devanagari

log = get_logger(__name__)

# Below this share of recognised Hinglish words, a transliteration is mostly
# passthrough and adds a near-duplicate query for no recall — not worth the
# extra embedding call and search round trip.
MIN_LEXICON_COVERAGE = 0.15


@dataclass(slots=True)
class AnalyzedQuery:
    raw: str
    language: Language
    script: Script
    confidence: float
    variants: list[str] = field(default_factory=list)
    transliterated: str | None = None
    intent: QueryIntent = QueryIntent.EXTRACTIVE

    @property
    def is_hinglish(self) -> bool:
        return self.language == Language.HINGLISH

    @property
    def uses_dual_query(self) -> bool:
        return len(self.variants) > 1

    @property
    def needs_coverage(self) -> bool:
        """Whether this query must be grounded on the document, not on passages."""
        return self.intent == QueryIntent.SUMMARY


def analyze_query(query: str) -> AnalyzedQuery:
    """Classify a query and build its search variants."""
    text = (query or "").strip()
    if not text:
        return AnalyzedQuery(
            raw="", language=Language.UNKNOWN, script=Script.UNKNOWN, confidence=0.0
        )

    # allow_hinglish=True: romanized input is the whole point on the query side,
    # unlike document chunks where Hinglish is not a real category.
    result = detect_language(text, allow_hinglish=True)
    intent = detect_intent(text)
    variants = [text]
    transliterated: str | None = None

    if result.language == Language.HINGLISH and settings.retrieval.dual_query_hinglish:
        coverage = lexicon_coverage(text)
        candidate = to_devanagari(text)
        if candidate != text and coverage >= MIN_LEXICON_COVERAGE:
            transliterated = candidate
            variants.append(candidate)
            log.debug("hinglish_dual_query", coverage=round(coverage, 2), transliterated=candidate)
        else:
            log.debug("hinglish_transliteration_skipped", coverage=round(coverage, 2))

    return AnalyzedQuery(
        raw=text,
        language=result.language,
        script=result.script,
        confidence=result.confidence,
        variants=variants,
        transliterated=transliterated,
        intent=intent,
    )
