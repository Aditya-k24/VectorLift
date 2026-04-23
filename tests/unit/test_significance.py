"""
Unit tests for pipelines/evaluation/significance.py.
"""
from __future__ import annotations

import math

import pytest

from pipelines.evaluation.significance import (
    SignificanceTestResult,
    compare_systems,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------


class TestPairedBootstrapTest:
    """Low-level bootstrap function."""

    def test_same_system_p_value_near_1(self):
        """When both score lists are identical the test should not reject H0."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.4, 0.55, 0.65, 0.75, 0.3, 0.9]
        result = paired_bootstrap_test(scores, scores, n_bootstrap=5_000, seed=42)
        # p-value should be very close to 1.0 — system never differs from itself
        assert result["p_value"] > 0.5, f"Expected p≈1, got {result['p_value']}"

    def test_clearly_better_system_p_below_threshold(self):
        """A system that scores 1.0 on every query should beat one scoring 0.0."""
        scores_a = [0.0] * 20
        scores_b = [1.0] * 20
        result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=5_000, seed=0)
        assert result["p_value"] < 0.05, f"Expected p<0.05, got {result['p_value']}"

    def test_returns_ci_lower_lt_mean_lt_upper(self):
        scores_a = [0.3, 0.4, 0.5, 0.6, 0.2, 0.45, 0.55, 0.35]
        scores_b = [0.5, 0.6, 0.7, 0.8, 0.4, 0.65, 0.75, 0.55]
        result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=5_000, seed=7)
        ci_low, ci_high = result["confidence_interval"]
        delta = result["delta"]
        assert ci_low <= delta <= ci_high, (
            f"Delta {delta} not within CI [{ci_low}, {ci_high}]"
        )

    def test_delta_equals_mean_b_minus_mean_a(self):
        scores_a = [0.2, 0.4, 0.6]
        scores_b = [0.3, 0.5, 0.7]
        result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=1_000, seed=1)
        expected_delta = sum(scores_b) / len(scores_b) - sum(scores_a) / len(scores_a)
        assert math.isclose(result["delta"], expected_delta, rel_tol=1e-9)

    def test_raises_on_different_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_test([0.1, 0.2], [0.3], n_bootstrap=100)

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            paired_bootstrap_test([], [], n_bootstrap=100)

    def test_is_significant_field_consistent_with_p_value(self):
        scores_a = [0.0] * 15
        scores_b = [1.0] * 15
        result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=2_000, alpha=0.05, seed=99)
        assert result["is_significant"] == (result["p_value"] < 0.05)

    def test_effect_size_positive_when_b_better(self):
        scores_a = [0.2, 0.3, 0.25, 0.35, 0.28]
        scores_b = [0.6, 0.7, 0.65, 0.75, 0.68]
        result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=1_000, seed=5)
        assert result["effect_size"] > 0, f"Expected positive Cohen's d, got {result['effect_size']}"

    def test_effect_size_negative_when_a_better(self):
        scores_a = [0.8, 0.9, 0.85]
        scores_b = [0.1, 0.2, 0.15]
        result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=1_000, seed=5)
        assert result["effect_size"] < 0


# ---------------------------------------------------------------------------
# compare_systems
# ---------------------------------------------------------------------------


class TestCompareSystems:
    """Higher-level helper that wraps paired_bootstrap_test."""

    @pytest.fixture
    def sys_a(self):
        return {"q1": 0.3, "q2": 0.4, "q3": 0.5, "q4": 0.35, "q5": 0.45}

    @pytest.fixture
    def sys_b_better(self):
        # Strictly better on all queries
        return {"q1": 0.8, "q2": 0.9, "q3": 0.85, "q4": 0.88, "q5": 0.92}

    def test_returns_significance_test_result(self, sys_a, sys_b_better):
        result = compare_systems(
            sys_a, sys_b_better, metric_name="ndcg@10",
            n_bootstrap=1_000, seed=42
        )
        assert isinstance(result, SignificanceTestResult)

    def test_correct_fields_populated(self, sys_a, sys_b_better):
        result = compare_systems(
            sys_a, sys_b_better, metric_name="ndcg@10",
            system_a_name="bm25", system_b_name="dense",
            n_bootstrap=1_000, seed=42
        )
        assert result.system_a == "bm25"
        assert result.system_b == "dense"
        assert result.metric_name == "ndcg@10"
        assert 0.0 <= result.p_value <= 1.0
        assert result.n_queries == 5
        assert result.n_bootstrap == 1_000

    def test_delta_positive_when_b_better(self, sys_a, sys_b_better):
        result = compare_systems(sys_a, sys_b_better, metric_name="map", n_bootstrap=500, seed=1)
        assert result.delta > 0, "System B is better, delta should be positive"

    def test_ci_contains_delta(self, sys_a, sys_b_better):
        result = compare_systems(sys_a, sys_b_better, metric_name="map", n_bootstrap=2_000, seed=3)
        assert result.ci_lower <= result.delta <= result.ci_upper

    def test_clearly_significant_result(self, sys_b_better):
        sys_worst = {q: 0.0 for q in sys_b_better}
        result = compare_systems(
            sys_worst, sys_b_better, metric_name="ndcg@10",
            n_bootstrap=5_000, seed=42
        )
        assert result.is_significant, "Trivially better system should be significant"
        assert result.p_value < 0.05

    def test_effect_size_positive_for_better_system(self, sys_a, sys_b_better):
        result = compare_systems(sys_a, sys_b_better, metric_name="ndcg@10", n_bootstrap=500, seed=1)
        assert result.effect_size > 0

    def test_raises_when_no_shared_queries(self):
        with pytest.raises(ValueError, match="No common queries"):
            compare_systems(
                {"q1": 0.5}, {"q2": 0.6}, metric_name="ndcg@10", n_bootstrap=100
            )

    def test_only_shared_queries_used(self):
        sys_a = {"q1": 0.5, "q2": 0.6, "only_a": 0.9}
        sys_b = {"q1": 0.55, "q2": 0.65, "only_b": 0.1}
        result = compare_systems(sys_a, sys_b, metric_name="map", n_bootstrap=500, seed=1)
        # Only q1 and q2 are shared
        assert result.n_queries == 2

    def test_to_dict_serialisable(self, sys_a, sys_b_better):
        result = compare_systems(sys_a, sys_b_better, metric_name="ndcg@10", n_bootstrap=200, seed=7)
        d = result.to_dict()
        assert isinstance(d, dict)
        for key in ["system_a", "system_b", "metric_name", "p_value", "delta", "is_significant"]:
            assert key in d
