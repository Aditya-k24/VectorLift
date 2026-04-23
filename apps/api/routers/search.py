"""
Search router – exposes the primary retrieval and reranking endpoints.

Endpoints
---------
POST /search  – full pipeline search (bm25 / dense / hybrid / rerank).
POST /rerank  – standalone reranking of caller-supplied candidates.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from prometheus_client import Counter, Histogram

from apps.api.dependencies import SearchPipelineDep
from apps.api.schemas import (
    RerankRequest,
    RerankResponse,
    SearchRequest,
    SearchResponse,
)
from apps.api.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

# ---------------------------------------------------------------------------
# Router-level Prometheus metrics (in addition to service-level ones)
# ---------------------------------------------------------------------------

_HTTP_SEARCH_TOTAL = Counter(
    "vectorlift_http_search_total",
    "HTTP-level search request count.",
    ["mode", "http_status"],
)

_HTTP_SEARCH_LATENCY = Histogram(
    "vectorlift_http_search_latency_seconds",
    "HTTP-level search request latency.",
    ["mode"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

_HTTP_RERANK_TOTAL = Counter(
    "vectorlift_http_rerank_total",
    "HTTP-level rerank request count.",
    ["http_status"],
)


# ---------------------------------------------------------------------------
# Dependency: build a SearchService from the pipeline singleton
# ---------------------------------------------------------------------------


def _get_search_service(pipeline: SearchPipelineDep) -> SearchService:
    return SearchService(pipeline)


SearchServiceDep = Annotated[SearchService, Depends(_get_search_service)]


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=SearchResponse,
    summary="Run a search query",
    description=(
        "Execute a search using the specified mode. "
        "Supports BM25, dense, hybrid, and rerank (hybrid + cross-encoder) strategies."
    ),
    status_code=status.HTTP_200_OK,
)
async def search(
    body: SearchRequest,
    service: SearchServiceDep,
    http_request: Request,
) -> SearchResponse:
    request_id: str = getattr(http_request.state, "request_id", "")
    mode_label = body.mode.value

    try:
        response = await service.execute_search(body, request_id=request_id)
        _HTTP_SEARCH_TOTAL.labels(mode=mode_label, http_status="200").inc()
        return response
    except ValueError as exc:
        _HTTP_SEARCH_TOTAL.labels(mode=mode_label, http_status="400").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _HTTP_SEARCH_TOTAL.labels(mode=mode_label, http_status="500").inc()
        logger.exception(
            "Unhandled error in POST /search",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during search.",
        ) from exc


# ---------------------------------------------------------------------------
# POST /rerank
# ---------------------------------------------------------------------------


@router.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Rerank provided candidates",
    description=(
        "Apply the cross-encoder reranker to a caller-supplied list of candidates. "
        "Useful when retrieval has already been performed client-side."
    ),
    status_code=status.HTTP_200_OK,
)
async def rerank(
    body: RerankRequest,
    service: SearchServiceDep,
    http_request: Request,
) -> RerankResponse:
    request_id: str = getattr(http_request.state, "request_id", "")

    try:
        response = await service.execute_rerank(body, request_id=request_id)
        _HTTP_RERANK_TOTAL.labels(http_status="200").inc()
        return response
    except Exception as exc:
        _HTTP_RERANK_TOTAL.labels(http_status="500").inc()
        logger.exception(
            "Unhandled error in POST /rerank",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during reranking.",
        ) from exc
