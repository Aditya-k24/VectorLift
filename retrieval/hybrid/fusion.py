"""
Hybrid retrieval score-fusion strategies for VectorLift.

Two strategies are provided:

1. **ScoreFusion** – linearly combines min-max normalised BM25 and dense
   scores with a configurable alpha weight.

2. **ReciprocalRankFusion** – model-agnostic fusion based purely on rank
   positions.  Works well when the two score distributions are very
   different in magnitude (e.g. BM25 vs cosine similarity).

Both strategies deduplicate passages by ``passage_id`` and re-assign
sequential ranks in the merged output.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from retrieval.interfaces.base import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Min-max normalise the scores in *results* to [0, 1].

    Returns the same list with ``score`` fields replaced in place. If all
    scores are identical the normalised score is set to 1.0.
    """
    if not results:
        return results
    min_s = min(r.score for r in results)
    max_s = max(r.score for r in results)
    span = max_s - min_s
    if span == 0.0:
        for r in results:
            r.score = 1.0
    else:
        for r in results:
            r.score = (r.score - min_s) / span
    return results


def _assign_ranks(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Sort by descending score and assign 1-based ranks."""
    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results


# ---------------------------------------------------------------------------
# ScoreFusion
# ---------------------------------------------------------------------------

class ScoreFusion:
    """Weighted linear combination of normalised BM25 and dense scores.

    The merged score for a passage is::

        merged_score = alpha * dense_score + (1 - alpha) * bm25_score

    Passages that appear in only one result list receive a score of 0.0
    for the missing modality before combination.

    Args:
        alpha: Weight for the dense score (0 = pure BM25, 1 = pure dense).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha

    def merge(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        alpha: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Fuse BM25 and dense results.

        Args:
            bm25_results:  First-stage BM25 results (any order).
            dense_results: First-stage dense results (any order).
            alpha:         Per-call override for the dense weight.  Falls
                           back to ``self.alpha`` when ``None``.

        Returns:
            Merged and re-ranked list of :class:`RetrievalResult`.
        """
        w = alpha if alpha is not None else self.alpha
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {w}.")

        # Normalise each list independently
        norm_bm25 = _normalise(list(bm25_results))
        norm_dense = _normalise(list(dense_results))

        # Build lookup dicts: passage_id -> score
        bm25_map: Dict[str, float] = {r.passage_id: r.score for r in norm_bm25}
        dense_map: Dict[str, float] = {r.passage_id: r.score for r in norm_dense}

        # Payload lookup (prefer dense result if both lists have the passage)
        payload: Dict[str, RetrievalResult] = {}
        for r in norm_bm25:
            payload[r.passage_id] = r
        for r in norm_dense:
            payload[r.passage_id] = r

        all_ids = set(bm25_map) | set(dense_map)
        merged: List[RetrievalResult] = []

        for pid in all_ids:
            bm25_score = bm25_map.get(pid, 0.0)
            dense_score = dense_map.get(pid, 0.0)
            fused_score = w * dense_score + (1.0 - w) * bm25_score

            ref = payload[pid]
            merged.append(
                RetrievalResult(
                    passage_id=ref.passage_id,
                    text=ref.text,
                    title=ref.title,
                    score=fused_score,
                    rank=0,  # will be set by _assign_ranks
                    metadata=ref.metadata,
                )
            )

        _assign_ranks(merged)
        logger.debug(
            "ScoreFusion merged %d+%d → %d results (alpha=%.2f).",
            len(bm25_results),
            len(dense_results),
            len(merged),
            w,
        )
        return merged


# ---------------------------------------------------------------------------
# ReciprocalRankFusion
# ---------------------------------------------------------------------------

class ReciprocalRankFusion:
    """Rank-based fusion using the Reciprocal Rank Fusion formula.

    For each passage the RRF score is::

        rrf_score = sum(1 / (k + rank_i))

    where *k* is a smoothing constant (default 60) and ``rank_i`` is the
    1-based rank position in result list *i*.

    Passages not present in a list contribute 0 to that term.

    Args:
        k: Smoothing constant.  Larger values reduce the impact of
           high-ranked documents.
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        self.k = k

    def merge(
        self,
        result_lists: List[List[RetrievalResult]],
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Fuse an arbitrary number of ranked result lists.

        Args:
            result_lists: Each inner list is a ranked result list from one
                          retriever.  Lists may be of different lengths and
                          may overlap.
            k:            Per-call override for the smoothing constant.

        Returns:
            Merged and re-ranked list of :class:`RetrievalResult`.
        """
        smoothing = k if k is not None else self.k
        if smoothing <= 0:
            raise ValueError(f"k must be positive, got {smoothing}.")

        rrf_scores: Dict[str, float] = {}
        payload: Dict[str, RetrievalResult] = {}

        for result_list in result_lists:
            # Use the rank field if already set, otherwise assume list order
            for pos, result in enumerate(result_list, start=1):
                rank = result.rank if result.rank >= 1 else pos
                rrf_scores[result.passage_id] = (
                    rrf_scores.get(result.passage_id, 0.0) + 1.0 / (smoothing + rank)
                )
                payload[result.passage_id] = result  # last writer wins for payload

        merged: List[RetrievalResult] = []
        for pid, score in rrf_scores.items():
            ref = payload[pid]
            merged.append(
                RetrievalResult(
                    passage_id=ref.passage_id,
                    text=ref.text,
                    title=ref.title,
                    score=score,
                    rank=0,
                    metadata=ref.metadata,
                )
            )

        _assign_ranks(merged)
        logger.debug(
            "RRF merged %d list(s) → %d results (k=%d).",
            len(result_lists),
            len(merged),
            smoothing,
        )
        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES: Dict[str, Callable] = {
    "score": ScoreFusion().merge,
    "rrf": ReciprocalRankFusion().merge,
    "score_fusion": ScoreFusion().merge,
    "reciprocal_rank_fusion": ReciprocalRankFusion().merge,
}


def get_fusion_strategy(strategy_name: str) -> Callable:
    """Return a fusion callable by name.

    Recognised names (case-insensitive):
    - ``"score"`` / ``"score_fusion"`` – :class:`ScoreFusion`
    - ``"rrf"`` / ``"reciprocal_rank_fusion"`` – :class:`ReciprocalRankFusion`

    Args:
        strategy_name: Strategy identifier string.

    Returns:
        A callable with signature ``(list1, list2, **kwargs) -> list``.

    Raises:
        ValueError: If *strategy_name* is not recognised.
    """
    key = strategy_name.lower()
    if key not in _STRATEGIES:
        raise ValueError(
            f"Unknown fusion strategy '{strategy_name}'. "
            f"Choose from: {sorted(_STRATEGIES)}"
        )
    return _STRATEGIES[key]
