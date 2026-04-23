"""
Unit tests for retrieval/hybrid/fusion.py.

Tests cover ScoreFusion (score_fusion) and ReciprocalRankFusion (rrf).
"""
from __future__ import annotations

import math

import pytest

from retrieval.hybrid.fusion import ReciprocalRankFusion, ScoreFusion
from retrieval.interfaces.base import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(entries: list[tuple[str, float, int]]) -> list[RetrievalResult]:
    """Create RetrievalResult objects from (id, score, rank) tuples."""
    return [
        RetrievalResult(
            passage_id=pid,
            text=f"Text for {pid}",
            title=f"Title {pid}",
            score=score,
            rank=rank,
        )
        for pid, score, rank in entries
    ]


# ---------------------------------------------------------------------------
# ScoreFusion
# ---------------------------------------------------------------------------


class TestScoreFusion:
    def test_merged_results_contain_all_unique_ids(self):
        bm25 = _make_results([("a", 0.9, 1), ("b", 0.5, 2)])
        dense = _make_results([("c", 0.8, 1), ("d", 0.4, 2)])
        fusion = ScoreFusion(alpha=0.5)
        merged = fusion.merge(bm25, dense)
        ids = {r.passage_id for r in merged}
        assert ids == {"a", "b", "c", "d"}

    def test_overlap_deduplication(self):
        bm25 = _make_results([("a", 0.9, 1), ("b", 0.5, 2)])
        dense = _make_results([("a", 0.8, 1), ("c", 0.3, 2)])
        fusion = ScoreFusion(alpha=0.5)
        merged = fusion.merge(bm25, dense)
        ids = [r.passage_id for r in merged]
        # "a" must appear only once
        assert ids.count("a") == 1

    def test_alpha_0_is_bm25_only(self):
        """alpha=0 → full weight on BM25; dense scores contribute nothing."""
        bm25 = _make_results([("a", 0.9, 1), ("b", 0.1, 2)])
        dense = _make_results([("b", 0.99, 1), ("a", 0.01, 2)])
        fusion = ScoreFusion(alpha=0.0)
        merged = fusion.merge(bm25, dense)
        # After normalisation BM25: a→1.0, b→0.0; dense: b→1.0, a→0.0
        # fused_score = 0 * dense + 1 * bm25
        # → a gets score 1.0, b gets 0.0 → "a" should be rank 1
        top = merged[0]
        assert top.passage_id == "a"

    def test_alpha_1_is_dense_only(self):
        """alpha=1 → full weight on dense scores."""
        bm25 = _make_results([("a", 0.9, 1), ("b", 0.1, 2)])
        dense = _make_results([("b", 0.99, 1), ("a", 0.01, 2)])
        fusion = ScoreFusion(alpha=1.0)
        merged = fusion.merge(bm25, dense)
        # dense: b→1.0, a→0.0 → "b" should be rank 1
        top = merged[0]
        assert top.passage_id == "b"

    def test_equal_weight_scores_in_range(self):
        bm25 = _make_results([("x", 10.0, 1), ("y", 5.0, 2)])
        dense = _make_results([("x", 0.9, 1), ("y", 0.3, 2)])
        fusion = ScoreFusion(alpha=0.5)
        merged = fusion.merge(bm25, dense)
        for r in merged:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0,1]"

    def test_empty_bm25_returns_dense_only(self):
        bm25: list[RetrievalResult] = []
        dense = _make_results([("a", 0.8, 1), ("b", 0.3, 2)])
        fusion = ScoreFusion(alpha=0.5)
        merged = fusion.merge(bm25, dense)
        ids = {r.passage_id for r in merged}
        assert ids == {"a", "b"}

    def test_empty_dense_returns_bm25_only(self):
        bm25 = _make_results([("a", 0.8, 1), ("b", 0.3, 2)])
        dense: list[RetrievalResult] = []
        fusion = ScoreFusion(alpha=0.5)
        merged = fusion.merge(bm25, dense)
        ids = {r.passage_id for r in merged}
        assert ids == {"a", "b"}

    def test_ranks_are_sequential_from_1(self):
        bm25 = _make_results([("a", 0.9, 1), ("b", 0.5, 2)])
        dense = _make_results([("c", 0.8, 1), ("d", 0.4, 2)])
        fusion = ScoreFusion(alpha=0.5)
        merged = fusion.merge(bm25, dense)
        ranks = sorted(r.rank for r in merged)
        assert ranks == list(range(1, len(merged) + 1))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ScoreFusion(alpha=1.5)

    def test_per_call_alpha_override(self):
        bm25 = _make_results([("a", 0.9, 1)])
        dense = _make_results([("a", 0.1, 1)])
        fusion = ScoreFusion(alpha=0.5)
        # Override alpha to 0 for this call
        merged = fusion.merge(bm25, dense, alpha=0.0)
        # With alpha=0 and only "a" in both → "a" score = bm25 normalised = 1.0
        assert math.isclose(merged[0].score, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# ReciprocalRankFusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_rrf_formula_single_list(self):
        """Score for rank r in k=60 RRF is exactly 1/(60+r)."""
        results = _make_results([("a", 0.9, 1), ("b", 0.5, 2), ("c", 0.2, 3)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([results])
        scores = {r.passage_id: r.score for r in merged}
        assert math.isclose(scores["a"], 1.0 / (60 + 1), rel_tol=1e-9)
        assert math.isclose(scores["b"], 1.0 / (60 + 2), rel_tol=1e-9)
        assert math.isclose(scores["c"], 1.0 / (60 + 3), rel_tol=1e-9)

    def test_rrf_deduplication(self):
        """Same passage_id in both lists → scores are summed, not duplicated."""
        list1 = _make_results([("a", 0.9, 1), ("b", 0.5, 2)])
        list2 = _make_results([("a", 0.8, 1), ("c", 0.3, 2)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([list1, list2])
        ids = [r.passage_id for r in merged]
        assert ids.count("a") == 1

    def test_rrf_score_additive_for_duplicate(self):
        """Passage appearing at rank 1 in both lists → score = 2/(60+1)."""
        list1 = _make_results([("a", 0.9, 1)])
        list2 = _make_results([("a", 0.5, 1)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([list1, list2])
        expected = 2.0 / (60 + 1)
        assert math.isclose(merged[0].score, expected, rel_tol=1e-9)

    def test_rrf_empty_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([])
        assert merged == []

    def test_rrf_empty_inner_list(self):
        list1: list[RetrievalResult] = []
        list2 = _make_results([("a", 0.9, 1)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([list1, list2])
        assert len(merged) == 1
        assert merged[0].passage_id == "a"

    def test_rrf_merged_contains_all_unique_ids(self):
        list1 = _make_results([("a", 0.9, 1), ("b", 0.5, 2)])
        list2 = _make_results([("c", 0.8, 1), ("d", 0.3, 2)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([list1, list2])
        ids = {r.passage_id for r in merged}
        assert ids == {"a", "b", "c", "d"}

    def test_rrf_ranks_sequential(self):
        list1 = _make_results([("a", 0.9, 1), ("b", 0.5, 2)])
        list2 = _make_results([("c", 0.8, 1)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([list1, list2])
        ranks = sorted(r.rank for r in merged)
        assert ranks == list(range(1, len(merged) + 1))

    def test_rrf_invalid_k_raises(self):
        with pytest.raises(ValueError):
            ReciprocalRankFusion(k=0)

    def test_rrf_per_call_k_override(self):
        """k=1 gives much higher scores than k=60."""
        results = _make_results([("a", 0.9, 1)])
        rrf = ReciprocalRankFusion(k=60)
        merged_k60 = rrf.merge([results], k=60)
        merged_k1 = rrf.merge([results], k=1)
        assert merged_k1[0].score > merged_k60[0].score

    def test_rrf_three_lists(self):
        list1 = _make_results([("a", 0.9, 1)])
        list2 = _make_results([("a", 0.8, 1)])
        list3 = _make_results([("a", 0.7, 1)])
        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.merge([list1, list2, list3])
        expected = 3.0 / (60 + 1)
        assert math.isclose(merged[0].score, expected, rel_tol=1e-9)
