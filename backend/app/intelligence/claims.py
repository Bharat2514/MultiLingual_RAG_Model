"""Per-claim verification of a generated answer.

The citation check in `app.generation.citations` answers "does the answer cite
something real?". That is necessary and not sufficient: a sentence can cite a
genuine passage and still misstate it — the classic case being a number the
model got wrong while citing the right page.

This module splits the answer into claims and checks each one against the
passages *that claim* cites:

1. **Numbers are checked exactly.** Every amount, duration and date in a claim
   must appear in its cited passage. This is language-agnostic (`५०००` and
   `5,000` compare equal) and catches the most damaging hallucination there is.
2. **Wording is checked by the cross-encoder** when it is enabled — it scores
   (claim, passage) pairs across Hindi and English.
3. Otherwise a **lexical overlap** score is used, but only when claim and source
   share a script. A Hindi answer to an English passage cannot be judged by word
   overlap, and such claims are reported as `unchecked` rather than wrongly
   marked unsupported.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import regex

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Language
from app.generation.citations import extract_citations
from app.intelligence.textutil import numbers_in, words
from app.nlp.language import detect_language
from app.nlp.script import has_devanagari
from app.nlp.sentences import split_sentences
from app.retrieval.types import ScoredChunk

log = get_logger(__name__)

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
UNCITED = "uncited"
UNCHECKED = "unchecked"

_MARKER = r"\[[^\[\]\n]{1,120}\]"
# "…₹500. [policy.pdf p.4]" — move trailing markers before the terminator so the
# sentence splitter keeps a claim and its citation together.
_TRAILING_MARKERS_RE = regex.compile(rf"([.।!?])\s*((?:{_MARKER}\s*)+)")
_MARKERS_RE = regex.compile(_MARKER)
_PLACEHOLDER_RE = regex.compile("\ue000(\\d+)\ue001")
_LIST_PREFIX_RE = regex.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
_EMPHASIS_RE = regex.compile(r"[*_`#>]+")

_STOPWORDS = frozenset(
    """the a an and or of to in on for with by from at as is are was were be been this that
    these those it its which who whom will shall may must can not no any all each per such
    than then there their they has have had also under over into upon within
    का की के को में से है हैं था थी थे और या पर भी तो यह वह इस उस किया जाता जाती जाएगा
    गया गई होगा होगी होंगे लिए द्वारा तथा एवं""".split()  # noqa: SIM905 - readable as prose
)


@dataclass(slots=True)
class Claim:
    text: str
    verdict: str
    score: float | None = None
    method: str = "none"
    chunk_ids: list[str] = field(default_factory=list)
    missing_numbers: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ClaimReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out = {SUPPORTED: 0, PARTIAL: 0, UNSUPPORTED: 0, UNCITED: 0, UNCHECKED: 0}
        for claim in self.claims:
            out[claim.verdict] = out.get(claim.verdict, 0) + 1
        return out

    @property
    def support_ratio(self) -> float | None:
        """Supported share of the claims that could be judged at all."""
        c = self.counts
        judged = c[SUPPORTED] + c[PARTIAL] + c[UNSUPPORTED]
        if not judged:
            return None
        return round((c[SUPPORTED] + 0.5 * c[PARTIAL]) / judged, 3)


def _restore(text: str, markers: list[str]) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: markers[int(m.group(1))], text)


def _claims_from(answer: str) -> list[tuple[str, str]]:
    """(claim with markers, claim without markers) pairs."""
    out: list[tuple[str, str]] = []
    for line in answer.splitlines():
        line = _LIST_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        line = _TRAILING_MARKERS_RE.sub(r" \2\1", line)

        # The splitter would break inside "[policy.pdf p.4]" at "p.", so markers
        # are swapped for opaque placeholders while splitting and restored after.
        markers: list[str] = []

        def hide(match: regex.Match[str], store: list[str] = markers) -> str:
            store.append(match.group(0))
            return f"\ue000{len(store) - 1}\ue001"

        protected = _MARKERS_RE.sub(hide, line)
        for sentence in split_sentences(protected):
            raw = _restore(sentence.text, markers).strip()
            bare = _EMPHASIS_RE.sub("", _MARKERS_RE.sub("", raw)).strip(" .।:;-")
            # Headings and connective fragments ("Key points:") are not claims.
            if len(bare) < 15 or len(words(bare)) < 3:
                continue
            out.append((raw, bare))
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) >= 3 and not w.isdigit() and w not in _STOPWORDS}


def _lexically_comparable(claim: str, source: str) -> bool:
    claim_dev = has_devanagari(claim)
    if claim_dev != has_devanagari(source):
        return False
    if not claim_dev:
        # Hinglish shares the Latin script with English but not its vocabulary.
        return detect_language(claim).language != Language.HINGLISH
    return True


def _lexical_support(claim: str, sources: list[str]) -> float:
    wanted = _content_words(claim)
    if len(wanted) < 2:
        return 0.0
    have: set[str] = set()
    for source in sources:
        have |= _content_words(source)
    return len(wanted & have) / len(wanted)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _cross_encoder():
    if not settings.trust.verify_with_cross_encoder:
        return None
    try:
        from app.retrieval import factory

        reranker = factory.get_reranker()
    except Exception:  # pragma: no cover - model stack unavailable
        log.warning("claim_cross_encoder_unavailable", exc_info=True)
        return None
    return None if reranker.name == "none" else reranker


def verify_answer(answer: str, chunks: list[ScoredChunk], *, scorer=None) -> ClaimReport:
    """Check every claim in `answer` against the passages it cites.

    ``scorer`` overrides the cross-encoder (tests pass a stub); it must expose
    ``score(query, chunks) -> list[float]`` returning logits.
    """
    report = ClaimReport()
    if not answer or not chunks:
        return report

    by_id = {c.chunk_id: c for c in chunks}
    scorer = scorer if scorer is not None else _cross_encoder()
    support_t = settings.trust.support_threshold
    partial_t = settings.trust.partial_threshold

    for raw, bare in _claims_from(answer):
        cited = [by_id[c.chunk_id] for c in extract_citations(raw, chunks) if c.chunk_id in by_id]
        claim = Claim(text=bare, verdict=UNCITED, chunk_ids=[c.chunk_id for c in cited])

        if not cited:
            claim.reason = "No citation attached to this sentence."
            report.claims.append(claim)
            continue

        sources = [c.text for c in cited]
        source_numbers: set[str] = set()
        for text in sources:
            source_numbers.update(numbers_in(text))
        missing = [n for n in dict.fromkeys(numbers_in(bare)) if n not in source_numbers]

        if missing:
            claim.verdict = UNSUPPORTED
            claim.method = "numbers"
            claim.missing_numbers = missing
            claim.reason = "Numbers not found in the cited passage: " + ", ".join(missing)
            report.claims.append(claim)
            continue

        score: float | None = None
        if scorer is not None:
            try:
                logits = scorer.score(bare, cited)
                score = max(_sigmoid(float(x)) for x in logits)
                claim.method = "cross-encoder"
            except Exception:
                log.warning("claim_scoring_failed", exc_info=True)
        if score is None and all(_lexically_comparable(bare, s) for s in sources):
            score = _lexical_support(bare, sources)
            claim.method = "lexical"

        if score is None:
            claim.verdict = UNCHECKED
            claim.reason = (
                "Claim and source are in different languages; enable the "
                "cross-encoder verifier to check wording."
            )
        else:
            claim.score = round(score, 3)
            claim.verdict = (
                SUPPORTED if score >= support_t else PARTIAL if score >= partial_t else UNSUPPORTED
            )
            if claim.verdict != SUPPORTED:
                claim.reason = "Wording only partly matches the cited passage."
        report.claims.append(claim)

    return report
