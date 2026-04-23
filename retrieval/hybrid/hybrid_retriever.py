"""
Hybrid retriever combining BM25 and dense retrieval with configurable fusion.

Retrieval proceeds in three steps:
  1. Run BM25 and dense retrievers concurrently.
  2. Fuse result lists using the configured fusion strategy.
  3. Return the top-k merged results.

Both retrievers are indexed together via :meth:`index` so callers only
need to maintain a single retriever object.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from retrieval.hybrid.fusion import ReciprocalRankFusion, ScoreFusion, get_fusion_strategy
from retrieval.interfaces.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines a BM25 retriever and a dense retriever using score fusion.

    Args:
        bm25_retriever:   Any :class:`~retrieval.interfaces.base.BaseRetriever`
                          that performs keyword (BM25) retrieval.
        dense_retriever:  Any :class:`~retrieval.interfaces.base.BaseRetriever`
                          that performs dense (vector) retrieval.
        fusion_strategy:  One of:
                          - ``"score"`` – weighted score fusion (default)
                          - ``"rrf"``   – reciprocal rank fusion
                          - A callable matching ``(bm25_list, dense_list, **kw)``
                            or ``([list, list], **kw)`` depending on arity.
        alpha:            Dense weight used by score fusion (ignored for RRF).
        rrf_k:            Smoothing constant used by RRF (ignored for score).
    """

    def __init__(
        self,
        bm25_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        fusion_strategy: str | Callable = "score",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k

        # Resolve fusion strategy
        if callable(fusion_strategy):
            self._fuser = fusion_strategy
            self._strategy_name = getattr(fusion_strategy, "__name__", "custom")
        else:
            strategy_name = str(fusion_strategy).lower()
            if strategy_name in ("score", "score_fusion"):
                fuser_obj = ScoreFusion(alpha=alpha)
                self._fuser = fuser_obj.merge
            elif strategy_name in ("rrf", "reciprocal_rank_fusion"):
                fuser_obj = ReciprocalRankFusion(k=rrf_k)
                self._fuser = fuser_obj.merge
            else:
                # Fall back to factory for any registered name
                self._fuser = get_fusion_strategy(strategy_name)
            self._strategy_name = strategy_name

        logger.info(
            "HybridRetriever initialised (strategy='%s', alpha=%.2f, rrf_k=%d).",
            self._strategy_name,
            alpha,
            rrf_k,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, passages: List[Dict[str, Any]]) -> None:
        """Index *passages* in both the BM25 and dense retrievers concurrently.

        Args:
            passages: List of passage dicts (``id``, ``text``, ``title``, …).
        """
        if not passages:
            logger.warning("HybridRetriever.index called with empty list.")
            return

        logger.info("HybridRetriever indexing %d passages in both backends…", len(passages))
        await asyncio.gather(
            self.bm25_retriever.index(passages),
            self.dense_retriever.index(passages),
        )
        logger.info("HybridRetriever indexing complete.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Retrieve from both backends and fuse results.

        Runs BM25 and dense retrieval concurrently with
        :func:`asyncio.gather` and fuses the two lists.

        Args:
            query:       Natural-language query string.
            top_k:       Number of results to return after fusion.
            bm25_top_k:  Results to fetch from BM25 before fusion.
                         Defaults to ``max(top_k * 3, 50)`` so fusion has
                         enough candidates.
            dense_top_k: Results to fetch from dense retriever before fusion.
                         Defaults to the same heuristic.

        Returns:
            Fused and re-ranked list of :class:`RetrievalResult` (length
            ≤ top_k).
        """
        if not query or not query.strip():
            logger.warning("Empty query received; returning empty results.")
            return []

        # Fetch more candidates than needed to give fusion room
        fetch_k = max(top_k * 3, 50)
        bk = bm25_top_k or fetch_k
        dk = dense_top_k or fetch_k

        bm25_results, dense_results = await asyncio.gather(
            self.bm25_retriever.retrieve(query, top_k=bk),
            self.dense_retriever.retrieve(query, top_k=dk),
            return_exceptions=False,
        )

        # Run fusion (synchronous – no I/O)
        fused = self._fuse(bm25_results, dense_results)

        # Truncate to top_k
        top_results = fused[:top_k]
        # Re-assign ranks after truncation
        for i, r in enumerate(top_results, start=1):
            r.rank = i

        logger.debug(
            "HybridRetriever: bm25=%d, dense=%d, fused=%d, returned=%d.",
            len(bm25_results),
            len(dense_results),
            len(fused),
            len(top_results),
        )
        return top_results

    def _fuse(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Dispatch to the configured fusion callable.

        Handles both the ``(bm25, dense)`` signature used by
        :class:`ScoreFusion` and the ``([list, list])`` signature used by
        :class:`ReciprocalRankFusion`.
        """
        try:
            # ScoreFusion.merge(bm25_results, dense_results, alpha=...)
            if self._strategy_name in ("score", "score_fusion"):
                return self._fuser(bm25_results, dense_results, alpha=self.alpha)

            # ReciprocalRankFusion.merge(result_lists, k=...)
            if self._strategy_name in ("rrf", "reciprocal_rank_fusion"):
                return self._fuser([bm25_results, dense_results], k=self.rrf_k)

            # Custom callable – try both signatures gracefully
            try:
                return self._fuser(bm25_results, dense_results)
            except TypeError:
                return self._fuser([bm25_results, dense_results])

        except Exception as exc:
            logger.error("Fusion failed: %s. Falling back to BM25 results.", exc)
            return bm25_results

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if both underlying retrievers are healthy."""
        bm25_ok, dense_ok = await asyncio.gather(
            self.bm25_retriever.health_check(),
            self.dense_retriever.health_check(),
        )
        if not bm25_ok:
            logger.warning("BM25 retriever health check failed.")
        if not dense_ok:
            logger.warning("Dense retriever health check failed.")
        return bool(bm25_ok and dense_ok)

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"strategy='{self._strategy_name}', "
            f"alpha={self.alpha}, "
            f"bm25={type(self.bm25_retriever).__name__}, "
            f"dense={type(self.dense_retriever).__name__})"
        )
