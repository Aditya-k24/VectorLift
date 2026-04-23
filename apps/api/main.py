"""
VectorLift FastAPI application entry point.

Startup sequence
----------------
1. Load configuration from environment.
2. Open connections to Elasticsearch, Qdrant, Redis and Postgres.
3. Instantiate the SearchPipeline (loads bi-encoder & cross-encoder weights).
4. Register all routers.
5. Attach middleware and exception handlers.

Shutdown sequence
-----------------
1. Drain in-flight requests (handled by uvicorn graceful shutdown).
2. Close all client connections.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram

from apps.api.middleware import RequestIDMiddleware, TimingMiddleware
from apps.api.routers import evaluation, health, search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus application-level metrics
# ---------------------------------------------------------------------------

_HTTP_REQUESTS_TOTAL = Counter(
    "vectorlift_http_requests_total",
    "Total HTTP requests received.",
    ["method", "path", "status_code"],
)

_HTTP_REQUEST_DURATION = Histogram(
    "vectorlift_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ---------------------------------------------------------------------------
# OpenAPI tags
# ---------------------------------------------------------------------------

TAGS_METADATA: list[dict[str, str]] = [
    {
        "name": "Search",
        "description": "Primary retrieval and reranking endpoints.",
    },
    {
        "name": "Evaluation",
        "description": "Trigger offline evaluation runs and inspect experiment results.",
    },
    {
        "name": "Health & Observability",
        "description": "Service health checks, model info and Prometheus metrics.",
    },
]


# ---------------------------------------------------------------------------
# Lifespan handler
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application-level resources across startup and shutdown."""
    logger.info("VectorLift API starting up …")

    # ------------------------------------------------------------------
    # 1. Settings
    # ------------------------------------------------------------------
    try:
        from pydantic_settings import BaseSettings

        class Settings(BaseSettings):
            elasticsearch_url: str = "http://localhost:9200"
            qdrant_url: str = "http://localhost:6333"
            redis_url: str = "redis://localhost:6379/0"
            postgres_dsn: str = "postgresql+asyncpg://vectorlift:vectorlift@localhost:5432/vectorlift"
            bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
            cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            log_level: str = "INFO"

            model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

        settings = Settings()
        app.state.settings = settings
        logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
        logger.info("Configuration loaded.")
    except Exception:
        logger.exception("Failed to load settings – using defaults.")
        app.state.settings = None
        settings = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Elasticsearch
    # ------------------------------------------------------------------
    es_client: Any = None
    try:
        from elasticsearch import AsyncElasticsearch

        es_url = getattr(settings, "elasticsearch_url", "http://localhost:9200")
        es_client = AsyncElasticsearch(hosts=[es_url])
        app.state.es_client = es_client
        logger.info("Elasticsearch client initialised: %s", es_url)
    except Exception:
        logger.warning("Elasticsearch client could not be initialised – skipping.")
        app.state.es_client = None

    # ------------------------------------------------------------------
    # 3. Qdrant
    # ------------------------------------------------------------------
    qdrant_client: Any = None
    try:
        from qdrant_client import QdrantClient

        qdrant_url = getattr(settings, "qdrant_url", "http://localhost:6333")
        qdrant_client = QdrantClient(url=qdrant_url, timeout=10)
        app.state.qdrant_client = qdrant_client
        logger.info("Qdrant client initialised: %s", qdrant_url)
    except Exception:
        logger.warning("Qdrant client could not be initialised – skipping.")
        app.state.qdrant_client = None

    # ------------------------------------------------------------------
    # 4. Redis
    # ------------------------------------------------------------------
    redis_client: Any = None
    try:
        import redis.asyncio as aioredis

        redis_url = getattr(settings, "redis_url", "redis://localhost:6379/0")
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        app.state.redis = redis_client
        logger.info("Redis client initialised: %s", redis_url)
    except Exception:
        logger.warning("Redis client could not be initialised – skipping.")
        app.state.redis = None

    # ------------------------------------------------------------------
    # 5. Postgres / SQLAlchemy
    # ------------------------------------------------------------------
    db_session_factory: Any = None
    try:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        pg_dsn = getattr(settings, "postgres_dsn", "")
        if pg_dsn:
            engine = create_async_engine(pg_dsn, pool_pre_ping=True, echo=False)
            db_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            app.state.db_session_factory = db_session_factory
            app.state.db_engine = engine
            logger.info("Postgres async engine created.")
        else:
            app.state.db_session_factory = None
            app.state.db_engine = None
    except Exception:
        logger.warning("Postgres session factory could not be created – skipping.")
        app.state.db_session_factory = None
        app.state.db_engine = None

    # ------------------------------------------------------------------
    # 6. Search pipeline (loads ML models)
    # ------------------------------------------------------------------
    try:
        app.state.search_pipeline = _load_pipeline(settings, es_client, qdrant_client)
        logger.info("Search pipeline loaded successfully.")
    except Exception:
        logger.exception("Search pipeline failed to load – service will return 503 for search.")
        app.state.search_pipeline = None

    logger.info("VectorLift API startup complete.")
    yield  # ← application runs here

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("VectorLift API shutting down …")

    if es_client is not None:
        try:
            await es_client.close()
        except Exception:
            pass

    if redis_client is not None:
        try:
            await redis_client.aclose()
        except Exception:
            pass

    if hasattr(app.state, "db_engine") and app.state.db_engine is not None:
        try:
            await app.state.db_engine.dispose()
        except Exception:
            pass

    logger.info("VectorLift API shut down cleanly.")


def _load_pipeline(settings: Any, es_client: Any, qdrant_client: Any) -> Any | None:
    """Attempt to instantiate the SearchPipeline.

    Returns ``None`` if required packages or models are unavailable so that
    the rest of the application still starts up.
    """
    try:
        # Lazy import – only attempt if sentence-transformers is installed
        from sentence_transformers import CrossEncoder, SentenceTransformer

        bi_encoder_name = getattr(
            settings, "bi_encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        cross_encoder_name = getattr(
            settings, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        logger.info("Loading bi-encoder: %s", bi_encoder_name)
        bi_encoder = SentenceTransformer(bi_encoder_name)

        logger.info("Loading cross-encoder: %s", cross_encoder_name)
        cross_encoder = CrossEncoder(cross_encoder_name)

        # Build a minimal pipeline duck-type accepted by SearchService
        class _MinimalPipeline:
            def __init__(self) -> None:
                from retrieval.dense import DenseRetriever
                from retrieval.hybrid import HybridRetriever
                from retrieval.bm25 import BM25Retriever
                from retrieval.reranker import CrossEncoderReranker

                self.bm25_retriever = BM25Retriever(es_client=es_client)
                self.dense_retriever = DenseRetriever(
                    model=bi_encoder, qdrant_client=qdrant_client
                )
                self.hybrid_retriever = HybridRetriever(
                    bm25=self.bm25_retriever, dense=self.dense_retriever
                )
                self.reranker = CrossEncoderReranker(model=cross_encoder)

        return _MinimalPipeline()
    except ImportError as exc:
        logger.warning("Could not import pipeline components (%s) – pipeline disabled.", exc)
        return None
    except Exception as exc:
        logger.warning("Pipeline construction failed: %s – pipeline disabled.", exc)
        return None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="VectorLift API",
        description=(
            "Production-grade Semantic Search and Ranking Engine. "
            "Supports BM25, dense, hybrid, and cross-encoder reranking."
        ),
        version="0.1.0",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # Middleware (order matters – outermost is applied first)
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ------------------------------------------------------------------
    # Prometheus instrumentation middleware
    # ------------------------------------------------------------------

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Any) -> Any:
        path = request.url.path
        method = request.method
        t0 = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - t0
        _HTTP_REQUESTS_TOTAL.labels(
            method=method, path=path, status_code=str(response.status_code)
        ).inc()
        _HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration)
        return response

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            "Validation error on %s %s",
            request.method,
            request.url.path,
            extra={"request_id": request_id, "errors": exc.errors()},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "request_id": request_id,
                "detail": exc.errors(),
                "body": None,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "request_id": request_id,
                "detail": "An internal server error occurred.",
            },
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(search.router)
    app.include_router(evaluation.router)
    app.include_router(health.router)

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn: apps.api.main:app)
# ---------------------------------------------------------------------------
app = create_app()
