"""Run the §12.4 retrieval ablations over the gold set."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.generation.query import analyze_query
from app.retrieval.vectorstore.base import SearchFilter

from .arms import Arm
from .corpus import EvalCorpus
from .goldset import GoldQuestion, GoldSet, QueryKind
from .index import Backends
from .metrics import ArmSummary, QueryScore, score_query

#: Deep enough to compute @10 metrics; `top_k` stays at the production default so
#: the candidate set the reranker sees is the one production would hand it.
EVAL_TOP_N = 10


@dataclass(slots=True)
class ArmResult:
    arm: Arm
    scores: list[QueryScore] = field(default_factory=list)

    @property
    def summary(self) -> ArmSummary:
        return ArmSummary.from_scores(self.arm.id, self.scores)

    def slice_summary(self, label: str, predicate) -> ArmSummary | None:
        subset = [s for s in self.scores if predicate(s)]
        return ArmSummary.from_scores(label, subset) if subset else None

    @property
    def misses(self) -> list[QueryScore]:
        """Questions where no relevant chunk reached the top 5 — the error log."""
        return [s for s in self.scores if not s.found]


def retrieval_pool(gold: GoldSet) -> list[GoldQuestion]:
    """Questions that retrieval metrics are meaningful for.

    Adversarial questions have no relevant chunk by construction (they are scored
    by the abstention metric instead), and summary questions mark every chunk of
    their document relevant, so a Recall@5 over them measures nothing. Pooling
    either into the headline number would inflate or deflate it for a reason that
    has nothing to do with retrieval quality.
    """
    return [q for q in gold.answerable if q.kind is not QueryKind.SUMMARY]


async def run_arm(
    arm: Arm,
    gold: GoldSet,
    corpus: EvalCorpus,
    backends: Backends,
    *,
    top_k: int,
    progress=None,
) -> ArmResult:
    service = backends.service(rerank=arm.rerank)
    scope = SearchFilter(session_id=corpus.session_id)
    result = ArmResult(arm=arm)

    for question in retrieval_pool(gold):
        analyzed = analyze_query(question.query)
        variants = analyzed.variants if arm.dual_query else [analyzed.raw]

        started = time.perf_counter()
        retrieved = await service.retrieve(
            analyzed.raw,
            scope,
            queries=variants,
            top_k=top_k,
            top_n=EVAL_TOP_N,
            mode=arm.mode,
            rerank=arm.rerank,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        result.scores.append(
            score_query(
                question_id=question.id,
                kind=str(question.kind),
                lang=str(question.lang),
                retrieved=[c.chunk_id for c in retrieved.chunks],
                relevant=question.relevant_chunk_ids,
                latency_ms=elapsed_ms,
            )
        )
        if progress:
            progress(arm, question, result)

    return result
