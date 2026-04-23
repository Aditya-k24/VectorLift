"""
Unit tests for pipelines/evaluation/metrics.py.

All expected values are computed analytically and verified by hand so that
numpy rounding is not the source of truth — the tests themselves document the
formula behaviour.
"""
from __future__ import annotations

import math
import pytest

from pipelines.evaluation.metrics import (
    average_precision,
    compute_metrics,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# dcg_at_k
# ---------------------------------------------------------------------------


class TestDcgAtK:
    """DCG formula: sum(rel_i / log2(i+1)) for i in [1, k]."""

    def test_perfect_ranking_single(self):
        # rel=[1], k=1 → 1/log2(2) = 1.0
        result = dcg_at_k([1], k=1)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_perfect_ranking_three(self):
        # rel=[3, 2, 1], k=3
        # = 3/log2(2) + 2/log2(3) + 1/log2(4)
        # = 3/1 + 2/1.585 + 1/2 ≈ 3 + 1.2619 + 0.5 = 4.7619
        expected = 3.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        result = dcg_at_k([3, 2, 1], k=3)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_cutoff_limits_computation(self):
        # Even though relevances has 5 items, k=2 should only use first 2
        result_k2 = dcg_at_k([1, 1, 1, 1, 1], k=2)
        result_k5 = dcg_at_k([1, 1, 1, 1, 1], k=5)
        assert result_k2 < result_k5

    def test_all_zeros(self):
        result = dcg_at_k([0, 0, 0], k=3)
        assert result == 0.0

    def test_k_zero_returns_zero(self):
        result = dcg_at_k([1, 2, 3], k=0)
        assert result == 0.0

    def test_k_exceeds_list_length(self):
        # Truncation at list length — should not raise
        result = dcg_at_k([1, 0], k=100)
        assert result > 0.0

    def test_empty_relevances(self):
        result = dcg_at_k([], k=10)
        assert result == 0.0

    def test_first_rank_highest_contribution(self):
        # Swapping first and last element should reduce DCG
        dcg_best = dcg_at_k([3, 0, 0], k=3)
        dcg_worst = dcg_at_k([0, 0, 3], k=3)
        assert dcg_best > dcg_worst


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    """NDCG = DCG / IDCG.  Must be in [0, 1] when there are relevant docs."""

    def test_perfect_ranking_is_1(self):
        # Perfect order → NDCG = 1.0
        relevances = [3, 2, 1, 0]
        assert math.isclose(ndcg_at_k(relevances, k=4), 1.0, rel_tol=1e-9)

    def test_reversed_ranking_less_than_1(self):
        # Worst order (all relevant at the end)
        relevances = [0, 0, 0, 1]
        ndcg = ndcg_at_k(relevances, k=4)
        assert 0.0 < ndcg < 1.0

    def test_all_zeros_returns_zero(self):
        assert ndcg_at_k([0, 0, 0], k=3) == 0.0

    def test_single_relevant_doc_at_rank1(self):
        # [1, 0, 0] — perfect because the only relevant doc is first
        assert math.isclose(ndcg_at_k([1, 0, 0], k=3), 1.0, rel_tol=1e-9)

    def test_single_relevant_doc_at_rank3(self):
        ndcg = ndcg_at_k([0, 0, 1], k=3)
        # DCG = 1/log2(4) = 0.5;  IDCG = 1/log2(2) = 1.0 → NDCG = 0.5
        assert math.isclose(ndcg, 0.5, rel_tol=1e-6)

    def test_ndcg_in_unit_interval(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            rels = [rng.randint(0, 3) for _ in range(10)]
            val = ndcg_at_k(rels, k=10)
            assert 0.0 <= val <= 1.0, f"NDCG out of range: {val} for rels={rels}"

    def test_k_zero_returns_zero(self):
        assert ndcg_at_k([1, 1, 1], k=0) == 0.0


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


class TestAveragePrecision:
    """AP = mean of precision-at-hit-ranks over all relevant docs."""

    def test_single_relevant_at_rank1(self):
        # P@1 = 1.0 → AP = 1.0
        assert math.isclose(average_precision([1, 0, 0]), 1.0, rel_tol=1e-9)

    def test_single_relevant_at_rank2(self):
        # P@2 = 1/2 → AP = 0.5
        assert math.isclose(average_precision([0, 1, 0]), 0.5, rel_tol=1e-9)

    def test_two_relevant_both_early(self):
        # hits at 1, 2: P@1=1, P@2=1 → AP = 1.0
        assert math.isclose(average_precision([1, 1, 0]), 1.0, rel_tol=1e-9)

    def test_two_relevant_interleaved(self):
        # hits at rank 1 and 3: P@1=1, P@3=2/3 → AP = (1 + 2/3)/2 = 5/6
        ap = average_precision([1, 0, 1, 0])
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert math.isclose(ap, expected, rel_tol=1e-9)

    def test_none_relevant(self):
        assert average_precision([0, 0, 0]) == 0.0

    def test_all_relevant(self):
        # Every position is a hit → P@i = 1 for all i → AP = 1.0
        assert math.isclose(average_precision([1, 1, 1, 1]), 1.0, rel_tol=1e-9)

    def test_relevance_treated_as_binary(self):
        # Values > 0 count as relevant
        ap_graded = average_precision([2, 0, 3])
        ap_binary = average_precision([1, 0, 1])
        assert math.isclose(ap_graded, ap_binary, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    def test_first_hit_at_rank1(self):
        assert math.isclose(reciprocal_rank([1, 0, 0]), 1.0, rel_tol=1e-9)

    def test_first_hit_at_rank2(self):
        assert math.isclose(reciprocal_rank([0, 1, 0]), 0.5, rel_tol=1e-9)

    def test_first_hit_at_rank3(self):
        assert math.isclose(reciprocal_rank([0, 0, 1]), 1.0 / 3.0, rel_tol=1e-9)

    def test_no_hit_returns_zero(self):
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_multiple_relevant_uses_first(self):
        # RR is 1 / rank_of_first_hit
        rr = reciprocal_rank([0, 1, 1])
        assert math.isclose(rr, 0.5, rel_tol=1e-9)

    def test_empty_list_returns_zero(self):
        assert reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        # 2 relevant in corpus, both in top-2
        assert math.isclose(recall_at_k([1, 1, 0], total_relevant=2, k=2), 1.0, rel_tol=1e-9)

    def test_partial_recall(self):
        # 3 relevant, only 1 in top-2
        assert math.isclose(recall_at_k([1, 0, 0], total_relevant=3, k=2), 1.0 / 3.0, rel_tol=1e-9)

    def test_k_less_than_total(self):
        # 3 relevant, top-2 has 2 → recall = 2/3
        assert math.isclose(recall_at_k([1, 1, 1], total_relevant=3, k=2), 2.0 / 3.0, rel_tol=1e-9)

    def test_k_ge_total_relevant_in_list(self):
        # k=5 ≥ length 3 — all 3 positions checked
        assert math.isclose(recall_at_k([1, 0, 1], total_relevant=2, k=5), 1.0, rel_tol=1e-9)

    def test_zero_total_relevant_returns_zero(self):
        assert recall_at_k([1, 1], total_relevant=0, k=2) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k([1, 1], total_relevant=2, k=0) == 0.0

    def test_no_relevant_retrieved(self):
        assert recall_at_k([0, 0, 0], total_relevant=3, k=3) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert math.isclose(precision_at_k([1, 1, 1], k=3), 1.0, rel_tol=1e-9)

    def test_none_relevant(self):
        assert precision_at_k([0, 0, 0], k=3) == 0.0

    def test_mixed_relevance(self):
        # 2 relevant out of first 4
        assert math.isclose(precision_at_k([1, 0, 1, 0, 1], k=4), 0.5, rel_tol=1e-9)

    def test_k_one(self):
        assert precision_at_k([1], k=1) == 1.0
        assert precision_at_k([0], k=1) == 0.0

    def test_k_zero_returns_zero(self):
        assert precision_at_k([1, 1], k=0) == 0.0

    def test_cutoff_respected(self):
        # Positions beyond k are ignored
        assert math.isclose(precision_at_k([1, 1, 0, 0], k=2), 1.0, rel_tol=1e-9)
        assert math.isclose(precision_at_k([1, 1, 0, 0], k=4), 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_metrics (integration)
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Test the full aggregation pipeline against hand-computed values."""

    @pytest.fixture
    def qrels(self) -> dict:
        # q1: doc "a" is relevant, "b" is not
        # q2: docs "c" and "d" are relevant, "e" is not
        return {
            "q1": {"a": 1, "b": 0},
            "q2": {"c": 1, "d": 1, "e": 0},
        }

    @pytest.fixture
    def results(self) -> dict:
        # q1: ranked [a, b, c] — "a" is at rank 1
        # q2: ranked [d, c, e] — both relevant docs in top-2
        return {
            "q1": ["a", "b", "c"],
            "q2": ["d", "c", "e"],
        }

    def test_returns_dict(self, qrels, results):
        metrics = compute_metrics(qrels, results, k_values=[5, 10])
        assert isinstance(metrics, dict)

    def test_expected_keys_present(self, qrels, results):
        metrics = compute_metrics(qrels, results, k_values=[5])
        for key in ["ndcg@5", "mrr@5", "map", "recall@5", "precision@5"]:
            assert key in metrics, f"Missing key: {key}"

    def test_ndcg_in_unit_interval(self, qrels, results):
        metrics = compute_metrics(qrels, results, k_values=[1, 5, 10])
        for k in [1, 5, 10]:
            val = metrics[f"ndcg@{k}"]
            assert 0.0 <= val <= 1.0, f"ndcg@{k} out of range: {val}"

    def test_perfect_ranking_ndcg_is_1(self):
        qrels = {"q1": {"a": 1, "b": 0}}
        results = {"q1": ["a", "b"]}
        metrics = compute_metrics(qrels, results, k_values=[1])
        assert math.isclose(metrics["ndcg@1"], 1.0, rel_tol=1e-9)

    def test_map_for_known_case(self, qrels, results):
        # q1: AP = 1.0 (hit at rank 1)
        # q2: hits at rank 1 and 2 → AP = (1 + 1) / 2 = 1.0
        # mean AP = 1.0
        metrics = compute_metrics(qrels, results, k_values=[10])
        assert math.isclose(metrics["map"], 1.0, rel_tol=1e-9)

    def test_empty_qrels_returns_empty(self):
        metrics = compute_metrics({}, {}, k_values=[10])
        assert metrics == {}

    def test_missing_query_in_results_handled_gracefully(self):
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        results = {"q1": ["a"]}  # q2 missing
        # Should not raise; q2 gets zero metrics
        metrics = compute_metrics(qrels, results, k_values=[1])
        assert "ndcg@1" in metrics

    def test_default_k_values(self, qrels, results):
        metrics = compute_metrics(qrels, results)
        for k in [1, 3, 5, 10, 100]:
            assert f"ndcg@{k}" in metrics

    def test_mrr_at_1_for_perfect(self):
        qrels = {"q1": {"a": 1}}
        results = {"q1": ["a"]}
        metrics = compute_metrics(qrels, results, k_values=[1])
        assert math.isclose(metrics["mrr@1"], 1.0, rel_tol=1e-9)
