"""The metrics are the product here — if they are wrong, every number is wrong.

Each case is checked against a value computed by hand in the docstring rather
than against the implementation's own output, so a refactor that changes a
formula fails instead of re-baselining itself.
"""

from __future__ import annotations

import math

import pytest
from harness.metrics import (
    ArmSummary,
    hit_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
    wilson_interval,
)

RANKING = ["a", "b", "c", "d", "e", "f"]


def test_hit_at_k_is_binary_on_any_relevant_chunk():
    assert hit_at_k(RANKING, {"c"}, 3) == 1.0
    assert hit_at_k(RANKING, {"c"}, 2) == 0.0
    assert hit_at_k(RANKING, {"d", "z"}, 5) == 1.0


def test_hit_at_k_with_no_relevant_chunks_is_zero_not_one():
    """An adversarial question must not score as a hit for being unanswerable."""
    assert hit_at_k(RANKING, set(), 5) == 0.0


def test_recall_at_k_is_set_recall_not_hit_rate():
    """Two relevant chunks, one inside k -> 0.5, where hit@k would say 1.0."""
    assert recall_at_k(RANKING, {"a", "f"}, 3) == pytest.approx(0.5)
    assert hit_at_k(RANKING, {"a", "f"}, 3) == 1.0


def test_precision_at_k_divides_by_the_window_not_the_relevant_set():
    assert precision_at_k(RANKING, {"a", "b"}, 4) == pytest.approx(0.5)


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank(RANKING, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKING, {"z"}) == 0.0


def test_ndcg_rewards_rank_not_mere_presence():
    """One relevant chunk at rank 1 scores 1.0; the same chunk at rank 3 scores
    1/log2(4) = 0.5."""
    assert ndcg_at_k(RANKING, {"a"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(RANKING, {"c"}, 5) == pytest.approx(1 / math.log2(4))


def test_ndcg_ideal_is_capped_at_k():
    """With three relevant chunks and k=2 the ideal DCG uses two, not three —
    otherwise a perfect ranking could not score 1.0."""
    assert ndcg_at_k(RANKING, {"a", "b", "c"}, 2) == pytest.approx(1.0)


def test_percentile_is_nearest_rank_and_never_interpolates():
    """P95 must be a latency that was actually observed (see the docstring in
    metrics.percentile)."""
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 95) == 95.0
    assert percentile(values, 50) == 50.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) == 0.0


def test_wilson_interval_brackets_the_point_estimate_and_narrows_with_n():
    low_n = wilson_interval(8, 10)
    high_n = wilson_interval(800, 1000)
    assert low_n[0] < 0.8 < low_n[1]
    assert high_n[0] < 0.8 < high_n[1]
    assert (high_n[1] - high_n[0]) < (low_n[1] - low_n[0])


def test_wilson_interval_on_an_empty_sample_does_not_divide_by_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_arm_summary_averages_per_question_scores():
    scores = [
        score_query("q1", "numeric", "en", ["a", "x"], {"a"}, 10.0),
        score_query("q2", "numeric", "en", ["x", "y", "z", "w", "v"], {"a"}, 30.0),
    ]
    summary = ArmSummary.from_scores("test", scores)
    assert summary.n == 2
    assert summary.hit_at_1 == pytest.approx(0.5)
    assert summary.hit_at_5 == pytest.approx(0.5)
    assert summary.mrr == pytest.approx(0.5)
    assert summary.latency_p50_ms == pytest.approx(10.0)
