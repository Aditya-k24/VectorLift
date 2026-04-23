"""
Health router – system and model status endpoints.

Endpoints
---------
GET /health       – deep health check of all external dependencies.
GET /model-info   – metadata about loaded models (names, dims, devices).
GET /metrics      – raw Prometheus text exposition.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from apps.api.schemas import HealthResponse, ModelInfo, ModelInfoResponse, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health & Observability"])


# ---------------------------------------------------------------------------
# Health probe helpers
# ---------------------------------------------------------------------------


async def _check_elasticsearch(es_client: Any | None) -> ServiceStatus:
    if es_client is None:
        return ServiceStatus(name="elasticsearch", healthy=False, detail="client not initialised")
    t0 = time.perf_counter()
    try:
        info = await asyncio.wait_for(es_client.ping(), timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1_000
        return ServiceStatus(
            name="elasticsearch",
            healthy=bool(info),
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        return ServiceStatus(name="elasticsearch", healthy=False, detail=str(exc))


async def _check_qdrant(qdrant_client: Any | None) -> ServiceStatus:
    if qdrant_client is None:
        return ServiceStatus(name="qdrant", healthy=False, detail="client not initialised")
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(qdrant_client.get_collections), timeout=5.0
        )
        latency_ms = (time.perf_counter() - t0) * 1_000
        return ServiceStatus(name="qdrant", healthy=True, latency_ms=round(latency_ms, 2))
    except Exception as exc:
        return ServiceStatus(name="qdrant", healthy=False, detail=str(exc))


async def _check_redis(redis_client: Any | None) -> ServiceStatus:
    if redis_client is None:
        return ServiceStatus(name="redis", healthy=False, detail="client not initialised")
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(redis_client.ping(), timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1_000
        return ServiceStatus(name="redis", healthy=True, latency_ms=round(latency_ms, 2))
    except Exception as exc:
        return ServiceStatus(name="redis", healthy=False, detail=str(exc))


async def _check_postgres(db_factory: Any | None) -> ServiceStatus:
    if db_factory is None:
        return ServiceStatus(name="postgres", healthy=False, detail="session factory not initialised")
    t0 = time.perf_counter()
    try:
        from sqlalchemy import text

        async with db_factory() as session:
            await asyncio.wait_for(session.execute(text("SELECT 1")), timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1_000
        return ServiceStatus(name="postgres", healthy=True, latency_ms=round(latency_ms, 2))
    except Exception as exc:
        return ServiceStatus(name="postgres", healthy=False, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Deep health check",
    description="Checks connectivity to Elasticsearch, Qdrant, Redis, Postgres and model readiness.",
)
async def health(request: Request) -> HealthResponse:
    state = request.app.state

    es_client = getattr(state, "es_client", None)
    qdrant_client = getattr(state, "qdrant_client", None)
    redis_client = getattr(state, "redis", None)
    db_factory = getattr(state, "db_session_factory", None)
    pipeline = getattr(state, "search_pipeline", None)

    service_checks = await asyncio.gather(
        _check_elasticsearch(es_client),
        _check_qdrant(qdrant_client),
        _check_redis(redis_client),
        _check_postgres(db_factory),
        return_exceptions=False,
    )

    services: list[ServiceStatus] = list(service_checks)
    models_loaded = pipeline is not None

    all_healthy = all(s.healthy for s in services) and models_loaded
    any_healthy = any(s.healthy for s in services)
    overall_status = (
        "healthy" if all_healthy else ("degraded" if any_healthy else "unhealthy")
    )

    logger.info(
        "health.check",
        extra={
            "status": overall_status,
            "services": {s.name: s.healthy for s in services},
        },
    )

    return HealthResponse(
        status=overall_status,
        services=services,
        models_loaded=models_loaded,
    )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Loaded model metadata",
    description="Returns the names, checkpoint paths and embedding dimensions of all loaded models.",
)
async def model_info(request: Request) -> ModelInfoResponse:
    pipeline = getattr(request.app.state, "search_pipeline", None)
    models: list[ModelInfo] = []

    if pipeline is not None:
        # Bi-encoder
        if hasattr(pipeline, "dense_retriever") and pipeline.dense_retriever is not None:
            encoder = getattr(pipeline.dense_retriever, "model", None)
            checkpoint = getattr(encoder, "name_or_path", "unknown")
            embedding_dim: int | None = None
            if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
                embedding_dim = encoder.config.hidden_size
            device_str = "cpu"
            if hasattr(encoder, "device"):
                device_str = str(encoder.device)
            models.append(
                ModelInfo(
                    name="bi-encoder",
                    checkpoint=checkpoint,
                    embedding_dim=embedding_dim,
                    device=device_str,
                )
            )

        # Cross-encoder / reranker
        if hasattr(pipeline, "reranker") and pipeline.reranker is not None:
            cross = getattr(pipeline.reranker, "model", None)
            checkpoint = getattr(cross, "name_or_path", "unknown")
            models.append(
                ModelInfo(
                    name="cross-encoder",
                    checkpoint=checkpoint,
                    device="cpu",
                )
            )

    if not models:
        # Return placeholder entries so the dashboard renders sensibly
        models = [
            ModelInfo(
                name="bi-encoder",
                checkpoint="not loaded",
                embedding_dim=None,
                device="n/a",
            ),
            ModelInfo(
                name="cross-encoder",
                checkpoint="not loaded",
                device="n/a",
            ),
        ]

    return ModelInfoResponse(models=models)


# ---------------------------------------------------------------------------
# GET /metrics  (Prometheus scrape endpoint)
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Raw Prometheus text format metrics for scraping.",
    include_in_schema=False,  # hide from OpenAPI docs to avoid confusion
)
async def prometheus_metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
