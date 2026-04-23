"""
VectorLift — Prometheus Metrics
=================================
Defines all application-level Prometheus metrics and a helper function to
start a dedicated metrics HTTP server.

Metrics are registered as module-level singletons so they can be imported and
used from anywhere in the codebase without re-registration errors.

Usage
-----
    from core.metrics.prometheus import (
        SEARCH_REQUESTS_TOTAL,
        SEARCH_LATENCY_SECONDS,
        start_metrics_server,
    )

    # Record a search request
    SEARCH_REQUESTS_TOTAL.labels(mode="hybrid", status="success").inc()

    # Time a search
    with SEARCH_LATENCY_SECONDS.labels(mode="hybrid").time():
        results = do_search(query)

    # Expose metrics on port 9090 (call once at startup)
    start_metrics_server(port=9090)
"""

from __future__ import annotations

import threading
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from core.logging.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Histogram bucket configurations
# ---------------------------------------------------------------------------

# Fine-grained latency buckets for user-facing search (0ms → 10s)
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075,
    0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0,
)

# Coarser buckets for batch / background operations
_BATCH_LATENCY_BUCKETS = (
    0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

SEARCH_REQUESTS_TOTAL: Counter = Counter(
    name="vectorlift_search_requests_total",
    documentation=(
        "Total number of search requests received, partitioned by retrieval mode "
        "and terminal status (success / error)."
    ),
    labelnames=["mode", "status"],
)

INDEX_DOCUMENTS_TOTAL: Counter = Counter(
    name="vectorlift_index_documents_total",
    documentation=(
        "Cumulative number of documents successfully indexed, "
        "partitioned by index backend."
    ),
    labelnames=["backend"],  # elasticsearch | qdrant | bm25
)

RERANK_REQUESTS_TOTAL: Counter = Counter(
    name="vectorlift_rerank_requests_total",
    documentation="Total number of reranking operations performed.",
    labelnames=["status"],
)

EMBEDDING_REQUESTS_TOTAL: Counter = Counter(
    name="vectorlift_embedding_requests_total",
    documentation="Total number of embedding generation calls.",
    labelnames=["model", "status"],
)

CACHE_HITS_TOTAL: Counter = Counter(
    name="vectorlift_cache_hits_total",
    documentation="Number of Redis cache hits for query embeddings or results.",
    labelnames=["cache_type"],  # embedding | result
)

CACHE_MISSES_TOTAL: Counter = Counter(
    name="vectorlift_cache_misses_total",
    documentation="Number of Redis cache misses.",
    labelnames=["cache_type"],
)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

SEARCH_LATENCY_SECONDS: Histogram = Histogram(
    name="vectorlift_search_latency_seconds",
    documentation=(
        "End-to-end search request latency in seconds, "
        "partitioned by retrieval mode."
    ),
    labelnames=["mode"],
    buckets=_LATENCY_BUCKETS,
)

RERANK_LATENCY_SECONDS: Histogram = Histogram(
    name="vectorlift_rerank_latency_seconds",
    documentation="Cross-encoder reranking latency in seconds.",
    buckets=_LATENCY_BUCKETS,
)

EMBEDDING_GENERATION_SECONDS: Histogram = Histogram(
    name="vectorlift_embedding_generation_seconds",
    documentation=(
        "Time taken to generate embeddings for a single batch, "
        "partitioned by model identifier."
    ),
    labelnames=["model"],
    buckets=_BATCH_LATENCY_BUCKETS,
)

BM25_RETRIEVAL_LATENCY_SECONDS: Histogram = Histogram(
    name="vectorlift_bm25_retrieval_latency_seconds",
    documentation="BM25 sparse retrieval latency in seconds.",
    buckets=_LATENCY_BUCKETS,
)

DENSE_RETRIEVAL_LATENCY_SECONDS: Histogram = Histogram(
    name="vectorlift_dense_retrieval_latency_seconds",
    documentation="Dense ANN retrieval latency in seconds.",
    labelnames=["backend"],  # qdrant | faiss | elasticsearch
    buckets=_LATENCY_BUCKETS,
)

INDEXING_LATENCY_SECONDS: Histogram = Histogram(
    name="vectorlift_indexing_latency_seconds",
    documentation="Latency for indexing a batch of documents.",
    labelnames=["backend"],
    buckets=_BATCH_LATENCY_BUCKETS,
)


# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

ACTIVE_MODEL_INFO: Gauge = Gauge(
    name="vectorlift_active_model_info",
    documentation=(
        "Metadata about the currently loaded models. "
        "The gauge value is always 1; labels carry the actual info."
    ),
    labelnames=["model_type", "model_id", "device"],
)

INDEX_SIZE_DOCUMENTS: Gauge = Gauge(
    name="vectorlift_index_size_documents",
    documentation="Current number of documents in each index backend.",
    labelnames=["backend"],
)

CACHE_SIZE_BYTES: Gauge = Gauge(
    name="vectorlift_cache_size_bytes",
    documentation="Estimated memory usage of the embedding cache in bytes.",
)

ACTIVE_SEARCH_REQUESTS: Gauge = Gauge(
    name="vectorlift_active_search_requests",
    documentation="Number of search requests currently being processed.",
)

MODEL_LOAD_TIME_SECONDS: Gauge = Gauge(
    name="vectorlift_model_load_time_seconds",
    documentation="Wall-clock time taken to load each model at startup.",
    labelnames=["model_type"],
)


# ---------------------------------------------------------------------------
# Metrics server
# ---------------------------------------------------------------------------

_metrics_server_started = threading.Event()


def start_metrics_server(
    port: int | None = None,
    addr: str = "0.0.0.0",
    registry: CollectorRegistry | None = None,
) -> None:
    """
    Start a Prometheus metrics HTTP server on a background thread.

    The server is started at most once per process — subsequent calls are
    no-ops.  Metrics are served at ``http://<addr>:<port>/metrics``.

    Parameters
    ----------
    port:
        TCP port to listen on.  Defaults to ``settings.prometheus_port``
        (usually 9090).
    addr:
        Interface to bind to (default: all interfaces).
    registry:
        Custom Prometheus registry.  Defaults to the global ``REGISTRY``.
    """
    if _metrics_server_started.is_set():
        logger.debug("Prometheus metrics server already running — skipping start.")
        return

    if port is None:
        try:
            from core.config.settings import get_settings

            port = get_settings().prometheus_port
        except Exception:
            port = 9090

    effective_registry = registry or REGISTRY

    try:
        start_http_server(port=port, addr=addr, registry=effective_registry)
        _metrics_server_started.set()
        logger.info(
            "Prometheus metrics server started",
            extra={"port": port, "addr": addr},
        )
    except OSError as exc:
        logger.error(
            "Failed to start Prometheus metrics server",
            extra={"port": port, "addr": addr, "error": str(exc)},
        )
        raise


# ---------------------------------------------------------------------------
# Helper: record model info gauge
# ---------------------------------------------------------------------------


def record_model_info(
    model_type: str,
    model_id: str,
    device: str,
    load_time_seconds: float | None = None,
) -> None:
    """
    Update the ``vectorlift_active_model_info`` gauge for a loaded model.

    Call this after each model is loaded at startup.

    Parameters
    ----------
    model_type:        "biencoder" or "crossencoder"
    model_id:          Full model path or HF hub ID
    device:            "cpu", "cuda", "mps", etc.
    load_time_seconds: Optional wall-clock load time to record in the gauge.
    """
    ACTIVE_MODEL_INFO.labels(
        model_type=model_type,
        model_id=model_id,
        device=device,
    ).set(1)

    if load_time_seconds is not None:
        MODEL_LOAD_TIME_SECONDS.labels(model_type=model_type).set(load_time_seconds)

    logger.info(
        "Model loaded",
        extra={
            "model_type": model_type,
            "model_id": model_id,
            "device": device,
            "load_time_seconds": load_time_seconds,
        },
    )


# ---------------------------------------------------------------------------
# Context manager: track active requests
# ---------------------------------------------------------------------------


class ActiveRequestTracker:
    """
    Context manager that increments / decrements the active-requests gauge.

    Usage
    -----
        with ActiveRequestTracker():
            result = handle_search_request(...)
    """

    def __enter__(self) -> "ActiveRequestTracker":
        ACTIVE_SEARCH_REQUESTS.inc()
        return self

    def __exit__(self, *args: Any) -> None:
        ACTIVE_SEARCH_REQUESTS.dec()
