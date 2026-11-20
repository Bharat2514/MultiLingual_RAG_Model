"""Retrieval metrics (§12.2) and the small statistics the report needs.

Everything here is binary-relevance: a chunk either contains the answer or it
does not (see `goldset` for why relevance is anchored on text). Graded relevance
would need a second annotation pass we have no rater budget for, and on a
single-answer lookup task the graded and binary orderings agree.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the relevant set found in the top k.

    Note this is *set* recall, not the "did we find any" hit rate: on questions
    whose answer legitimately spans two chunks, finding one of two scores 0.5.
    """
    if not relevant:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top k. What the answer actually needs."""
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    window = retrieved[:k]
    if not window:
        return 0.0
    return len(set(window) & relevant) / len(window)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG. Rewards putting the answer first, not merely inside k."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with 150 samples an interpolated P95 invents a
    latency that was never observed, and NFR-2 is a claim about observed requests.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion.

    A gold set of 150 questions puts roughly ±0.07 around a Recall@5 of 0.83.
    Reporting the point estimate alone invites reading a 2-point difference
    between two arms as a real one, which at this sample size it is not.
    """
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = (
        z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(slots=True)
class QueryScore:
    """Per-question retrieval outcome, kept so the report can slice by kind/lang."""

    question_id: str
    kind: str
    lang: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    latency_ms: float
    retrieved: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.hit_at_5 > 0


def score_query(
    question_id: str,
    kind: str,
    lang: str,
    retrieved: Sequence[str],
    relevant: set[str],
    latency_ms: float,
) -> QueryScore:
    return QueryScore(
        question_id=question_id,
        kind=kind,
        lang=lang,
        hit_at_1=hit_at_k(retrieved, relevant, 1),
        hit_at_3=hit_at_k(retrieved, relevant, 3),
        hit_at_5=hit_at_k(retrieved, relevant, 5),
        hit_at_10=hit_at_k(retrieved, relevant, 10),
        recall_at_5=recall_at_k(retrieved, relevant, 5),
        recall_at_10=recall_at_k(retrieved, relevant, 10),
        precision_at_5=precision_at_k(retrieved, relevant, 5),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg_at_5=ndcg_at_k(retrieved, relevant, 5),
        ndcg_at_10=ndcg_at_k(retrieved, relevant, 10),
        latency_ms=latency_ms,
        retrieved=list(retrieved),
        relevant=sorted(relevant),
    )


@dataclass(slots=True)
class ArmSummary:
    """Aggregate for one ablation arm."""

    arm: str
    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    recall_at_5_ci: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_scores(cls, arm: str, scores: Sequence[QueryScore]) -> ArmSummary:
        hits5 = [s.hit_at_5 for s in scores]
        latencies = [s.latency_ms for s in scores]
        return cls(
            arm=arm,
            n=len(scores),
            hit_at_1=mean([s.hit_at_1 for s in scores]),
            hit_at_3=mean([s.hit_at_3 for s in scores]),
            hit_at_5=mean(hits5),
            hit_at_10=mean([s.hit_at_10 for s in scores]),
            recall_at_5=mean([s.recall_at_5 for s in scores]),
            recall_at_10=mean([s.recall_at_10 for s in scores]),
            precision_at_5=mean([s.precision_at_5 for s in scores]),
            mrr=mean([s.mrr for s in scores]),
            ndcg_at_5=mean([s.ndcg_at_5 for s in scores]),
            ndcg_at_10=mean([s.ndcg_at_10 for s in scores]),
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            recall_at_5_ci=wilson_interval(round(sum(hits5)), len(scores)),
        )
