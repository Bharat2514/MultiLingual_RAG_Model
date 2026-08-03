"""Query intent: extractive vs document-level (companion to FR-14).

The retrieval pipeline ranks *chunks* by how well each one answers the query.
That is the right model for extractive questions — "notice period kitna hai"
lives in some passage, and the cross-encoder finds it.

It is the wrong model for **aggregate** questions. "Is document ka summary do"
is not answered by any single passage, so the cross-encoder scores every
candidate as irrelevant — measured on a real session, the best of 20 candidates
scored -8.1 against the -2.0 floor — and the model is handed one arbitrary chunk
plus a request it cannot satisfy from it. It then correctly emits the
insufficient-context sentinel, and the user is told their summary "was not found
in your documents".

Aggregate queries are detected here and routed down a coverage path instead of a
pure relevance path (`RetrievalService.retrieve(..., coverage=True)`), with a
prompt that permits summarising (§11.4 rule 5 otherwise forbids it).

Detection is lexical rather than a model call: the phrasings are a small, closed
set across all three registers we serve, and a regex costs nothing on the
latency budget that an extra LLM round trip would blow.
"""

from __future__ import annotations

from enum import StrEnum

import regex


class QueryIntent(StrEnum):
    #: Answered by specific passages — the default, and the pipeline's assumption.
    EXTRACTIVE = "extractive"
    #: Answered only by the document as a whole: summary, overview, key points.
    SUMMARY = "summary"


# Latin script: English, and Hinglish — which in practice borrows the English
# word ("summary do", "key points batao") far more often than it romanises the
# Hindi one, so both spellings have to be here.
_LATIN_PATTERNS = [
    r"\bsummar(?:y|ies|ise|ize|ised|ized|isation|ization)\b",
    r"\boverview\b",
    r"\btl;?dr\b",
    r"\bgist\b",
    r"\bsynopsis\b",
    r"\b(?:key|main|important|major)\s+(?:points?|ideas?|takeaways?|highlights?|findings?)\b",
    r"\btakeaways?\b",
    r"\bwhat(?:'s| is)\s+(?:this|the|it)\s+(?:\w+\s+)?about\b",
    # Hinglish: "ye document kis baare mein hai", "kis bare me hai"
    r"\bkis\s+ba+re?\s*(?:me|mein|men)\b",
    # Romanised Hindi for summary / in brief.
    r"\bsa+ra+nsh\b",
    r"\bsanksh(?:ep|ipt|iptikaran)\b",
    r"\bmukhya\s+(?:bindu|baat(?:ein|en)?)\b",
    r"\b(?:khaas|khas)\s+baat(?:ein|en)?\b",
]

# Devanagari: no \b word boundaries — Devanagari combining marks make them
# unreliable, and these forms are distinctive enough not to need them.
_DEVANAGARI_PATTERNS = [
    r"सारांश",
    r"संक्षेप",
    r"संक्षिप्त",
    r"रूपरेखा",
    r"मुख्य\s*(?:बिंदु|बिन्दु|बात[ेों]|बातें)",
    r"ख़?ास\s*बातें",
    r"किस\s*बारे\s*म[ेैं]",
]

_SUMMARY_RE = regex.compile(
    "|".join(_LATIN_PATTERNS + _DEVANAGARI_PATTERNS),
    regex.IGNORECASE | regex.UNICODE,
)


def detect_intent(query: str) -> QueryIntent:
    """Classify a query as document-level or extractive.

    Biased towards EXTRACTIVE: misrouting a factual question into the coverage
    path costs recall on the one passage that held the answer, whereas the
    reverse merely gives a summary slightly more context than it needed.
    """
    if not query or not query.strip():
        return QueryIntent.EXTRACTIVE
    return QueryIntent.SUMMARY if _SUMMARY_RE.search(query) else QueryIntent.EXTRACTIVE
