"""LLM-as-judge for faithfulness, answer relevancy and answer correctness.

Deliberately **not** RAGAS, though the PRD names it. RAGAS computes each metric
with its own multi-call chain — statement extraction, then per-statement NLI for
faithfulness; question generation, then embedding comparison for relevancy — so a
four-metric run over 150 questions is several thousand LLM calls. On the free
Gemini tier that is days of throttling, and on a billed key it costs more than
the rest of the project combined for numbers a single well-specified call
produces to within a few points.

So: one call per answer, returning all four judgements as JSON, with the same
definitions RAGAS uses. The trade is documented rather than hidden — the judge
prompt is right here, it is auditable, and `--judge-model` lets a reviewer rerun
it on a stronger model than the one that generated the answers, which is the
part that actually matters for judge validity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.generation.providers.base import LLMProvider
from app.retrieval.types import ScoredChunk

JUDGE_SYSTEM = """You are a strict evaluator for a retrieval-augmented question \
answering system. You score one answer at a time against the passages that were \
given to the model and a reference answer written by a human.

Score four dimensions, each from 0.0 to 1.0:

faithfulness — the fraction of the answer's factual claims that are directly \
supported by the CONTEXT passages. A claim that is true in the world but absent \
from the context is NOT faithful. An answer that correctly declines to answer \
because the context does not contain the fact is fully faithful (1.0).

answer_relevancy — how directly the answer addresses the question asked. Penalise \
padding, hedging and unrequested material. Do not reward or penalise correctness \
here.

correctness — agreement with the REFERENCE answer on the substance. Numbers, \
durations and amounts must match to score above 0.5. Wording, language and \
formatting differences are irrelevant. If REFERENCE is "null", the question is \
unanswerable from the corpus: score 1.0 only if the answer declines to answer.

context_precision — the fraction of the supplied CONTEXT passages that were \
actually needed to produce the answer.

Respond with JSON only, no prose, no code fences:
{"faithfulness": 0.0, "answer_relevancy": 0.0, "correctness": 0.0, \
"context_precision": 0.0, "reason": "one short sentence"}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
#: Cap the context sent to the judge. The judge only has to decide whether a
#: claim is supported, and paying to re-send five full 550-token chunks per
#: question doubles the cost of the eval for judgements that do not change.
MAX_PASSAGE_CHARS = 1200


@dataclass(slots=True)
class Judgement:
    faithfulness: float
    answer_relevancy: float
    correctness: float
    context_precision: float
    reason: str = ""
    ok: bool = True

    @classmethod
    def failed(cls, reason: str) -> Judgement:
        """A judge call that did not return usable JSON.

        Scored as `ok=False` and excluded from the means rather than counted as
        zero: a parse failure is missing data, and silently averaging it in would
        report a model quality problem that is actually a plumbing problem.
        """
        return cls(0.0, 0.0, 0.0, 0.0, reason=reason, ok=False)


def build_judge_prompt(
    question: str, answer: str, chunks: list[ScoredChunk], reference: str | None
) -> str:
    passages = (
        "\n\n".join(
            f"[{i}] ({c.filename} {c.page_label}) {c.text[:MAX_PASSAGE_CHARS]}"
            for i, c in enumerate(chunks, start=1)
        )
        or "(no passages were retrieved)"
    )

    return (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{passages}\n\n"
        f"REFERENCE:\n{reference or 'null'}\n\n"
        f"ANSWER:\n{answer}\n"
    )


def parse_judgement(raw: str) -> Judgement:
    match = _JSON_RE.search(raw or "")
    if not match:
        return Judgement.failed("judge returned no JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Judgement.failed(f"judge JSON invalid: {exc}")

    def clamp(key: str) -> float:
        try:
            return min(1.0, max(0.0, float(data.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    return Judgement(
        faithfulness=clamp("faithfulness"),
        answer_relevancy=clamp("answer_relevancy"),
        correctness=clamp("correctness"),
        context_precision=clamp("context_precision"),
        reason=str(data.get("reason", ""))[:200],
    )


async def judge_answer(
    provider: LLMProvider,
    question: str,
    answer: str,
    chunks: list[ScoredChunk],
    reference: str | None,
) -> Judgement:
    try:
        response = await provider.generate(
            JUDGE_SYSTEM,
            build_judge_prompt(question, answer, chunks, reference),
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - a judge outage must not kill the run
        return Judgement.failed(f"judge call failed: {type(exc).__name__}: {exc}")
    return parse_judgement(response.text)
