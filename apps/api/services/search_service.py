"""
SearchService – thin orchestration layer between the API routers and the
underlying retrieval/reranking pipeline.

Responsibilities
----------------
* Route requests to the correct pipeline mode (bm25 / dense / hybrid / rerank).
* Convert raw ``RetrievalResult`` objects from the pipeline into API schemas.
* Record per-request Prometheus metrics.
* Provide structured logging with request IDs.
* Centralise error handling so that routers stay clean.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from prometheus_client import Counter, Histogram

from apps.api.schemas import (
    LatencyBreakdown,
    PassageResult,
    RerankCandidate,
    RerankRequest,
    RerankResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_SEARCH_REQUESTS = Counter(
    "vectorlift_search_requests_total",
    "Total number of search requests handled by SearchService.",
    ["mode", "status"],
)

_SEARCH_LATENCY = Histogram(
    "vectorlift_search_latency_seconds",
    "End-to-end search latency (seconds) by mode.",
    ["mode"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

_RERANK_REQUESTS = Counter(
    "vectorlift_rerank_requests_total",
    "Total number of standalone rerank requests.",
    ["status"],
)

_RERANK_LATENCY = Histogram(
    "vectorlift_rerank_latency_seconds",
    "Standalone rerank latency (seconds).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_schema(result: Any) -> PassageResult:
    """Convert a pipeline ``RetrievalResult`` dataclass to a Pydantic model."""
    return PassageResult(
        passage_id=result.passage_id,
        text=result.text,
        title=result.title,
        score=float(result.score),
        rank=result.rank,
        metadata=result.metadata or {},
    )


def _candidate_to_retrieval_result(candidate: RerankCandidate) -> Any:
    """Build a lightweight duck-typed object accepted by the reranker."""

    class _FakeResult:
        def __init__(self, c: RerankCandidate) -> None:
            self.passage_id = c.passage_id
            self.text = c.text
            self.title = c.title
            self.score = c.score
            self.rank = c.rank
            self.metadata = c.metadata

    return _FakeResult(candidate)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SearchService:
    """Wrap the :class:`~pipelines.SearchPipeline` for use by API routers.

    Parameters
    ----------
    pipeline:
        The application-level ``SearchPipeline`` singleton (stored in
        ``app.state.search_pipeline``).  The service never owns the pipeline's
        lifecycle.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_search(
        self,
        request: SearchRequest,
        *,
        request_id: str | None = None,
    ) -> SearchResponse:
        """Run a full search and return a :class:`~apps.api.schemas.SearchResponse`.

        Parameters
        ----------
        request:
            Validated search request.
        request_id:
            Correlation ID forwarded from the HTTP layer for structured logging.
        """
        rid = request_id or str(uuid.uuid4())
        mode_label = request.mode.value

        logger.info(
            "search.start",
            extra={
                "request_id": rid,
                "query": request.query[:120],
                "mode": mode_label,
                "top_k": request.top_k,
            },
        )

        t_start = time.perf_counter()
        retrieval_ms = 0.0
        rerank_ms = 0.0

        try:
            results, retrieval_ms, rerank_ms = await self._dispatch(request)
        except Exception as exc:
            _SEARCH_REQUESTS.labels(mode=mode_label, status="error").inc()
            logger.exception(
                "search.error",
                extra={"request_id": rid, "error": str(exc)},
            )
            raise

        total_ms = (time.perf_counter() - t_start) * 1_000
        _SEARCH_LATENCY.labels(mode=mode_label).observe(total_ms / 1_000)
        _SEARCH_REQUESTS.labels(mode=mode_label, status="ok").inc()

        logger.info(
            "search.done",
            extra={
                "request_id": rid,
                "result_count": len(results),
                "total_ms": round(total_ms, 2),
            },
        )

        return SearchResponse(
            request_id=rid,
            query=request.query,
            mode=request.mode,
            results=[_result_to_schema(r) for r in results],
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 2),
                rerank_ms=round(rerank_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            total_hits=len(results),
        )

    async def execute_rerank(
        self,
        request: RerankRequest,
        *,
        request_id: str | None = None,
    ) -> RerankResponse:
        """Run a standalone rerank over caller-supplied candidates."""
        rid = request_id or str(uuid.uuid4())
        logger.info(
            "rerank.start",
            extra={
                "request_id": rid,
                "query": request.query[:120],
                "candidates": len(request.candidates),
            },
        )

        t0 = time.perf_counter()
        try:
            raw_candidates = [_candidate_to_retrieval_result(c) for c in request.candidates]
            reranked = await self._pipeline.reranker.rerank(
                query=request.query,
                candidates=raw_candidates,
                top_n=request.top_n,
            )
        except Exception as exc:
            _RERANK_REQUESTS.labels(status="error").inc()
            logger.exception("rerank.error", extra={"request_id": rid, "error": str(exc)})
            raise

        rerank_ms = (time.perf_counter() - t0) * 1_000
        _RERANK_LATENCY.observe(rerank_ms / 1_000)
        _RERANK_REQUESTS.labels(status="ok").inc()

        logger.info(
            "rerank.done",
            extra={"request_id": rid, "rerank_ms": round(rerank_ms, 2)},
        )

        return RerankResponse(
            request_id=rid,
            query=request.query,
            results=[_result_to_schema(r) for r in reranked],
            rerank_ms=round(rerank_ms, 2),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _dispatch(
        self, request: SearchRequest
    ) -> tuple[list[Any], float, float]:
        """Route to the correct pipeline method and return (results, retrieval_ms, rerank_ms)."""
        mode = request.mode
        query = request.query
        top_k = request.top_k

        if mode == SearchMode.BM25:
            t0 = time.perf_counter()
            results = await self._pipeline.bm25_retriever.retrieve(query, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1_000
            return results, retrieval_ms, 0.0

        if mode == SearchMode.DENSE:
            t0 = time.perf_counter()
            results = await self._pipeline.dense_retriever.retrieve(query, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1_000
            return results, retrieval_ms, 0.0

        if mode == SearchMode.HYBRID:
            t0 = time.perf_counter()
            results = await self._pipeline.hybrid_retriever.retrieve(query, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1_000
            return results, retrieval_ms, 0.0

        if mode == SearchMode.RERANK:
            fetch_k = top_k * request.retrieval_multiplier
            t0 = time.perf_counter()
            candidates = await self._pipeline.hybrid_retriever.retrieve(query, top_k=fetch_k)
            retrieval_ms = (time.perf_counter() - t0) * 1_000

            t1 = time.perf_counter()
            results = await self._pipeline.reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=top_k,
            )
            rerank_ms = (time.perf_counter() - t1) * 1_000
            return results, retrieval_ms, rerank_ms

        msg = f"Unhandled search mode: {mode}"
        raise ValueError(msg)
