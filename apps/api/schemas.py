"""
Pydantic v2 request/response schemas for the VectorLift API.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SearchMode(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANK = "rerank"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class PassageResult(BaseModel):
    """Single passage returned by the search pipeline."""

    passage_id: str
    text: str
    title: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Wall-clock time consumed by each pipeline stage (milliseconds)."""

    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048, description="Natural-language query.")
    mode: SearchMode = Field(SearchMode.HYBRID, description="Retrieval / ranking strategy.")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return.")
    retrieval_multiplier: int = Field(
        5,
        ge=1,
        le=50,
        description="Over-fetch factor for reranker (ignored when mode != rerank).",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Metadata filters forwarded to the retriever."
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class SearchResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    mode: SearchMode
    results: list[PassageResult]
    latency: LatencyBreakdown
    total_hits: int = 0


# ---------------------------------------------------------------------------
# Rerank (standalone)
# ---------------------------------------------------------------------------


class RerankCandidate(BaseModel):
    passage_id: str
    text: str
    title: str = ""
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    candidates: list[RerankCandidate] = Field(..., min_length=1, max_length=200)
    top_n: int = Field(10, ge=1, le=200)


class RerankResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    results: list[PassageResult]
    rerank_ms: float


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class MetricSet(BaseModel):
    ndcg_at_10: float = 0.0
    mrr_at_10: float = 0.0
    map_score: float = 0.0
    recall_at_10: float = 0.0
    precision_at_10: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class ExperimentConfig(BaseModel):
    name: str = Field(..., description="Human-readable experiment name.")
    description: str = ""
    mode: SearchMode = SearchMode.HYBRID
    dataset: str = "ms_marco_dev_small"
    top_k: int = 10
    retrieval_multiplier: int = 5
    extra: dict[str, Any] = Field(default_factory=dict)


class SignificanceResult(BaseModel):
    metric: str
    p_value: float
    confidence_interval_low: float
    confidence_interval_high: float
    significant: bool
    delta: float


class ExperimentResult(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    config: ExperimentConfig
    metrics: MetricSet
    per_query_ndcg: list[float] = Field(default_factory=list)
    created_at: str = ""
    status: str = "completed"


class ExperimentComparison(BaseModel):
    baseline_id: str
    candidate_id: str
    baseline_metrics: MetricSet
    candidate_metrics: MetricSet
    significance_tests: list[SignificanceResult] = Field(default_factory=list)


class EvaluationJobResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ServiceStatus(BaseModel):
    name: str
    healthy: bool
    latency_ms: float | None = None
    detail: str = ""


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    services: list[ServiceStatus]
    models_loaded: bool


class ModelInfo(BaseModel):
    name: str
    checkpoint: str
    embedding_dim: int | None = None
    device: str = "cpu"
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    models: list[ModelInfo]
