"""
End-to-end search pipeline for VectorLift.

Wires together a retriever (BM25, dense, or hybrid) with an optional
cross-encoder reranker and a Redis result cache.

Pipeline flow
-------------
1. Check Redis cache for (query, mode, top_k) → return immediately on hit.
2. Run the configured retriever (async, latency measured).
3. If a reranker is configured, rerank the candidates (async, latency measured).
4. Store result in Redis with TTL=300 s.
5. Return results + latency breakdown dict.

Cache key
---------
``vectorlift:search:{sha256(query + mode + str(top_k))}``

Requires:
    redis>=5.0  (async client via redis.asyncio)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from retrieval.interfaces.base import BaseReranker, BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# Cache TTL in seconds
_CACHE_TTL = 300

# Redis key namespace
_KEY_PREFIX = "vectorlift:search:"

# Supported retrieval modes
VALID_MODES = frozenset({"bm25", "dense", "hybrid"})


def _cache_key(query: str, mode: str, top_k: int) -> str:
    """Build a deterministic Redis cache key."""
    raw = f"{query.strip()}|{mode}|{top_k}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{_KEY_PREFIX}{digest}"


def _serialise_results(results: List[RetrievalResult]) -> str:
    """Serialise results to a JSON string for Redis storage."""
    return json.dumps([r.to_dict() for r in results])


def _deserialise_results(raw: str) -> List[RetrievalResult]:
    """Deserialise results from a Redis JSON string."""
    return [RetrievalResult.from_dict(d) for d in json.loads(raw)]


class SearchPipeline:
    """Full search pipeline: retrieve → (optionally) rerank → cache.

    Args:
        retriever:    A :class:`~retrieval.interfaces.base.BaseRetriever`
                      implementation (BM25, dense, or hybrid).
        reranker:     Optional :class:`~retrieval.interfaces.base.BaseReranker`.
                      When provided, first-stage results are reranked before
                      being returned.
        cache:        Optional ``redis.asyncio.Redis`` client.  When ``None``
                      caching is disabled.
        cache_ttl:    Redis TTL in seconds (default 300).
        default_mode: Default retrieval mode label stored in cache keys
                      (default ``"hybrid"``).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: Optional[BaseReranker] = None,
        cache: Optional[Any] = None,   # redis.asyncio.Redis
        cache_ttl: int = _CACHE_TTL,
        default_mode: str = "hybrid",
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.default_mode = default_mode

        logger.info(
            "SearchPipeline initialised (retriever=%s, reranker=%s, cache=%s).",
            type(retriever).__name__,
            type(reranker).__name__ if reranker else "None",
            "Redis" if cache else "disabled",
        )

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: int = 10,
        rerank_top_n: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], Dict[str, float]]:
        """Run the full search pipeline for *query*.

        Args:
            query:        Natural-language query string.
            mode:         Retrieval mode label (used only for cache key
                          differentiation). Defaults to ``self.default_mode``.
            top_k:        Number of results to return.
            rerank_top_n: Candidates to score with the reranker before
                          returning ``top_k``. Defaults to ``top_k * 3``.

        Returns:
            Tuple of:
            - ``results``: Final :class:`RetrievalResult` list (≤ top_k).
            - ``latency``: Dict with keys ``total_ms``, ``retrieval_ms``,
              ``rerank_ms``, ``cache_ms``.

        Raises:
            ValueError: If *query* is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        mode = (mode or self.default_mode).lower()
        if mode not in VALID_MODES:
            logger.warning(
                "Unknown mode '%s'; proceeding but cache key may be unexpected.", mode
            )

        rerank_candidates = rerank_top_n or (top_k * 3)
        latency: Dict[str, float] = {
            "total_ms": 0.0,
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "cache_ms": 0.0,
        }
        pipeline_start = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Cache lookup
        # ------------------------------------------------------------------
        cache_key = _cache_key(query, mode, top_k)
        cache_start = time.perf_counter()
        cached = await self._cache_get(cache_key)
        latency["cache_ms"] = (time.perf_counter() - cache_start) * 1000.0

        if cached is not None:
            latency["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
            logger.debug("Cache HIT for key '%s' (%.1f ms).", cache_key, latency["cache_ms"])
            return cached, latency

        logger.debug("Cache MISS for key '%s'.", cache_key)

        # ------------------------------------------------------------------
        # 2. First-stage retrieval
        # ------------------------------------------------------------------
        retrieval_start = time.perf_counter()
        # Fetch more candidates when reranking is enabled
        fetch_k = rerank_candidates if self.reranker else top_k
        results = await self.retriever.retrieve(query, top_k=fetch_k)
        latency["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000.0
        logger.debug(
            "Retrieval returned %d results in %.1f ms.",
            len(results),
            latency["retrieval_ms"],
        )

        # ------------------------------------------------------------------
        # 3. Optional reranking
        # ------------------------------------------------------------------
        if self.reranker and results:
            rerank_start = time.perf_counter()
            results = await self.reranker.rerank(query, results, top_n=top_k)
            latency["rerank_ms"] = (time.perf_counter() - rerank_start) * 1000.0
            logger.debug(
                "Reranking returned %d results in %.1f ms.",
                len(results),
                latency["rerank_ms"],
            )
        else:
            # Truncate to top_k without reranking
            results = results[:top_k]
            for i, r in enumerate(results, start=1):
                r.rank = i

        latency["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0

        # ------------------------------------------------------------------
        # 4. Store in cache
        # ------------------------------------------------------------------
        cache_write_start = time.perf_counter()
        await self._cache_set(cache_key, results)
        latency["cache_ms"] += (time.perf_counter() - cache_write_start) * 1000.0

        logger.info(
            "search('%s'…) → %d results | total=%.1f ms, retrieval=%.1f ms, "
            "rerank=%.1f ms, cache=%.1f ms.",
            query[:50],
            len(results),
            latency["total_ms"],
            latency["retrieval_ms"],
            latency["rerank_ms"],
            latency["cache_ms"],
        )
        return results, latency

    # ------------------------------------------------------------------
    # Cache warming
    # ------------------------------------------------------------------

    async def warm_cache(
        self,
        queries: List[str],
        mode: Optional[str] = None,
        top_k: int = 10,
        rerank_top_n: Optional[int] = None,
        concurrency: int = 8,
    ) -> Dict[str, Any]:
        """Pre-populate the cache for a list of common queries.

        Args:
            queries:     Queries to pre-compute.
            mode:        Retrieval mode (default: ``self.default_mode``).
            top_k:       Result count per query.
            rerank_top_n: Reranker candidate count.
            concurrency: Maximum concurrent search calls.

        Returns:
            Summary dict: ``{total, succeeded, failed, duration_s}``.
        """
        if not queries:
            logger.warning("warm_cache called with empty query list.")
            return {"total": 0, "succeeded": 0, "failed": 0, "duration_s": 0.0}

        semaphore = asyncio.Semaphore(concurrency)
        start = time.perf_counter()
        succeeded = 0
        failed = 0

        async def _warm_one(q: str) -> bool:
            async with semaphore:
                try:
                    await self.search(q, mode=mode, top_k=top_k, rerank_top_n=rerank_top_n)
                    return True
                except Exception as exc:
                    logger.warning("warm_cache failed for query '%s': %s", q[:60], exc)
                    return False

        outcomes = await asyncio.gather(*[_warm_one(q) for q in queries])
        succeeded = sum(outcomes)
        failed = len(outcomes) - succeeded
        duration = time.perf_counter() - start

        logger.info(
            "warm_cache: %d/%d queries succeeded in %.1f s.",
            succeeded,
            len(queries),
            duration,
        )
        return {
            "total": len(queries),
            "succeeded": succeeded,
            "failed": failed,
            "duration_s": duration,
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all components.

        Returns:
            Dict with keys ``retriever``, ``reranker``, ``cache``.
        """
        retriever_ok = await self.retriever.health_check()
        reranker_ok = await self.reranker.health_check() if self.reranker else True
        cache_ok = await self._ping_cache()

        return {
            "retriever": retriever_ok,
            "reranker": reranker_ok,
            "cache": cache_ok,
        }

    # ------------------------------------------------------------------
    # Redis helpers (graceful no-op when cache is None)
    # ------------------------------------------------------------------

    async def _cache_get(self, key: str) -> Optional[List[RetrievalResult]]:
        """Fetch and deserialise cached results; returns ``None`` on miss."""
        if self.cache is None:
            return None
        try:
            raw = await self.cache.get(key)
            if raw is None:
                return None
            return _deserialise_results(raw)
        except Exception as exc:
            logger.warning("Redis GET failed for key '%s': %s", key, exc)
            return None

    async def _cache_set(self, key: str, results: List[RetrievalResult]) -> None:
        """Serialise and store *results* in Redis with the configured TTL."""
        if self.cache is None:
            return
        try:
            serialised = _serialise_results(results)
            await self.cache.set(key, serialised, ex=self.cache_ttl)
        except Exception as exc:
            logger.warning("Redis SET failed for key '%s': %s", key, exc)

    async def _ping_cache(self) -> bool:
        """Return ``True`` if Redis responds to PING."""
        if self.cache is None:
            return True   # caching is intentionally disabled – not a failure
        try:
            await self.cache.ping()
            return True
        except Exception as exc:
            logger.error("Redis PING failed: %s", exc)
            return False
