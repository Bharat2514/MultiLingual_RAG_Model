"""End-to-end answer evaluation: §12.2's generation, language and ops metrics.

Runs the real `ChatService`, so what is measured is the shipped grounding
contract — the sentinel substitution, the citation resolution, and the
"answer without calling the LLM when nothing was retrieved" short circuit —
rather than a bare prompt call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.generation import prompts
from app.generation.service import ChatResult, ChatService
from app.nlp.language import detect_language
from app.retrieval.vectorstore.base import SearchFilter

from .corpus import EvalCorpus
from .goldset import GoldQuestion, GoldSet, QueryKind
from .index import Backends
from .judge import Judgement, judge_answer
from .metrics import mean, percentile


def _abstained(result: ChatResult) -> bool:
    """Whether the pipeline declined to answer.

    Checked on the *output contract*, not on an internal flag alone: the answer
    can become a not-found either by retrieving nothing (short circuit) or by the
    model emitting the sentinel, and both must count. Comparing against the
    localised message covers the second path even when a provider quirk loses the
    error code.
    """
    if result.error_code == "NO_RELEVANT_CONTEXT":
        return True
    expected = prompts.not_found_message(result.language_detected)
    return result.answer.strip() == expected.strip()


def _language_matches(answer: str, expected: str, *, abstained: bool) -> bool:
    """Language fidelity (FR-21): did the answer come back in the asked register?

    The not-found messages are short fixed strings; `lingua` cannot reliably
    classify a six-word sentence, and scoring the abstention path on detected
    language would measure the detector rather than the system. Those are
    compared against the localised template directly.
    """
    if abstained:
        return answer.strip() == prompts.not_found_message(expected).strip()
    detected = detect_language(answer, allow_hinglish=True)
    if str(detected.language) == expected:
        return True
    # A Hinglish answer that comes back in Devanagari-free English prose is a
    # genuine miss, but an English answer classified `mixed` because it quotes a
    # Devanagari clause is not. Accept `mixed` only where the expectation is a
    # code-mixed register.
    return expected == "hinglish" and str(detected.language) in {"mixed", "en"}


@dataclass(slots=True)
class AnswerScore:
    question_id: str
    kind: str
    lang: str
    answerable: bool
    abstained: bool
    language_ok: bool
    grounded: bool
    #: A citation resolved to a chunk that actually contains the answer.
    citation_correct: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    answer: str = ""
    error: str | None = None
    judgement: Judgement | None = None

    @property
    def abstention_correct(self) -> bool:
        """The anti-hallucination metric (§12.3), in both directions."""
        return self.abstained if not self.answerable else not self.abstained


@dataclass(slots=True)
class GenerationReport:
    scores: list[AnswerScore] = field(default_factory=list)

    @property
    def answered(self) -> list[AnswerScore]:
        return [s for s in self.scores if s.error is None]

    def _mean(self, attr: str, subset: list[AnswerScore] | None = None) -> float:
        pool = self.answered if subset is None else subset
        return mean([float(getattr(s, attr)) for s in pool])

    @property
    def language_fidelity(self) -> float:
        return self._mean("language_ok")

    @property
    def grounded_rate(self) -> float:
        return self._mean("grounded", [s for s in self.answered if s.answerable])

    @property
    def citation_accuracy(self) -> float:
        return self._mean("citation_correct", [s for s in self.answered if s.answerable])

    @property
    def abstention_accuracy(self) -> float:
        return self._mean("abstention_correct")

    @property
    def abstention_recall(self) -> float:
        """Of the questions that *should* be refused, how many were."""
        adversarial = [s for s in self.answered if not s.answerable]
        return mean([float(s.abstained) for s in adversarial])

    @property
    def abstention_precision(self) -> float:
        """Of the refusals issued, how many were correct.

        The counterweight to recall: a system that refuses everything scores a
        perfect anti-hallucination number and is useless. Reporting only recall
        is how that gets hidden.
        """
        refused = [s for s in self.answered if s.abstained]
        if not refused:
            return 1.0
        return mean([float(not s.answerable) for s in refused])

    @property
    def false_abstention(self) -> float:
        """Answerable questions the system wrongly refused."""
        pool = [s for s in self.answered if s.answerable]
        return mean([float(s.abstained) for s in pool])

    def judged(self, attr: str) -> float:
        pool = [s.judgement for s in self.answered if s.judgement and s.judgement.ok]
        return mean([float(getattr(j, attr)) for j in pool]) if pool else 0.0

    @property
    def judge_coverage(self) -> int:
        return len([s for s in self.answered if s.judgement and s.judgement.ok])

    def latency(self, p: float) -> float:
        return percentile([s.latency_ms for s in self.answered], p)

    @property
    def mean_cost_usd(self) -> float:
        return self._mean("cost_usd")

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.answered)

    @property
    def mean_tokens(self) -> tuple[float, float]:
        return self._mean("input_tokens"), self._mean("output_tokens")

    @property
    def errors(self) -> list[AnswerScore]:
        return [s for s in self.scores if s.error is not None]


async def run_generation(
    gold: GoldSet,
    corpus: EvalCorpus,
    backends: Backends,
    *,
    judge_provider=None,
    progress=None,
) -> GenerationReport:
    service = ChatService(retrieval=backends.service(rerank=True))
    scope = SearchFilter(session_id=corpus.session_id)
    report = GenerationReport()

    for question in gold:
        started = time.perf_counter()
        try:
            result = await service.answer(
                question.query,
                scope,
                top_k=settings.retrieval.top_k,
                top_n=settings.retrieval.top_n,
            )
        except LLMUnavailableError as exc:
            report.scores.append(_error_score(question, exc, started))
            if progress:
                progress(question, report)
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        abstained = _abstained(result)
        chunks = result.retrieval.chunks if result.retrieval else []

        score = AnswerScore(
            question_id=question.id,
            kind=str(question.kind),
            lang=str(question.lang),
            answerable=question.answerable,
            abstained=abstained,
            language_ok=_language_matches(
                result.answer, question.expect_language, abstained=abstained
            ),
            grounded=result.grounded,
            citation_correct=_citation_correct(question, result),
            latency_ms=elapsed_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            answer=result.answer,
        )

        if judge_provider is not None:
            score.judgement = await judge_answer(
                judge_provider, question.query, result.answer, chunks, question.answer
            )

        report.scores.append(score)
        if progress:
            progress(question, report)

    return report


def _citation_correct(question: GoldQuestion, result: ChatResult) -> bool:
    """Did at least one *cited* passage actually contain the answer?

    Stricter than `grounded`, which only asks whether a citation resolved to some
    chunk we sent. A citation that resolves to the wrong passage is still a
    citation the user will click and find nothing in.
    """
    if not question.answerable or question.kind is QueryKind.SUMMARY:
        return True
    if not result.citations:
        return False
    return any(c.chunk_id in question.relevant_chunk_ids for c in result.citations)


def _error_score(question: GoldQuestion, exc: Exception, started: float) -> AnswerScore:
    return AnswerScore(
        question_id=question.id,
        kind=str(question.kind),
        lang=str(question.lang),
        answerable=question.answerable,
        abstained=False,
        language_ok=False,
        grounded=False,
        citation_correct=False,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        error=f"{type(exc).__name__}: {exc}",
    )
