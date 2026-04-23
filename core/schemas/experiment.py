"""
VectorLift — Experiment Tracking Schemas
==========================================
Pydantic v2 models for experiment configuration, evaluation metrics,
per-query results and statistical significance testing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from core.config.settings import DatasetMode, HybridFusionStrategy, RetrievalMode


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Snapshot of the model settings used during an experiment."""

    biencoder: str = Field(..., description="Bi-encoder model ID or path")
    crossencoder: str | None = Field(
        default=None,
        description="Cross-encoder model ID or path (None when reranking is disabled)",
    )
    embedding_dim: int = Field(default=768, ge=64)
    max_seq_length: int = Field(default=512, ge=32)
    biencoder_device: str = Field(default="cpu")
    crossencoder_device: str | None = Field(default=None)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model hyperparameters",
    )


class ExperimentConfig(BaseModel):
    """Full configuration for a single experiment run."""

    name: str = Field(..., min_length=1, max_length=256, description="Human-readable experiment name")
    retrieval_mode: RetrievalMode = Field(..., description="Retrieval pipeline used")
    dataset_mode: DatasetMode = Field(..., description="Dataset split used for evaluation")
    top_k: int = Field(default=10, ge=1, le=1000, description="Final result set size")
    rerank_top_n: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Number of first-stage candidates before reranking",
    )
    hybrid_bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="BM25 weight in hybrid fusion (only used in hybrid mode)",
    )
    hybrid_fusion_strategy: HybridFusionStrategy = Field(
        default=HybridFusionStrategy.RECIPROCAL_RANK_FUSION,
    )
    rrf_k: int = Field(default=60, ge=1)
    model_config: ModelConfig = Field(..., description="Model snapshot for this experiment")
    description: str | None = Field(default=None, max_length=2048)
    tags: list[str] = Field(default_factory=list, description="Arbitrary experiment tags")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def rerank_top_n_gte_top_k(self) -> "ExperimentConfig":
        if self.rerank_top_n < self.top_k:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) must be >= top_k ({self.top_k})"
            )
        return self


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MetricsResult(BaseModel):
    """
    Aggregate IR evaluation metrics for a retrieval experiment.

    Metric naming conventions
    -------------------------
    * ndcg_10  — NDCG@10 (Normalised Discounted Cumulative Gain)
    * mrr_10   — MRR@10 (Mean Reciprocal Rank)
    * map      — Mean Average Precision (MAP@top_k)
    * recall_k — Recall@k for various k values, e.g. {"10": 0.72, "100": 0.91}
    * precision_k — Precision@k for various k values
    * latency_* — percentile latencies in milliseconds
    """

    # Ranking quality
    ndcg_10: float = Field(..., ge=0.0, le=1.0, description="NDCG@10")
    mrr_10: float = Field(..., ge=0.0, le=1.0, description="MRR@10")
    map: float = Field(..., ge=0.0, le=1.0, description="Mean Average Precision")

    # Recall / Precision at multiple cut-offs
    recall_k: dict[str, float] = Field(
        ...,
        description="Recall@k — keys are string k values, e.g. {'10': 0.72, '100': 0.91}",
    )
    precision_k: dict[str, float] = Field(
        ...,
        description="Precision@k — keys are string k values",
    )

    # Latency percentiles (milliseconds)
    latency_p50: float = Field(..., ge=0.0, description="Median latency (ms)")
    latency_p95: float = Field(..., ge=0.0, description="95th-percentile latency (ms)")
    latency_p99: float = Field(..., ge=0.0, description="99th-percentile latency (ms)")

    # Throughput
    queries_per_second: float | None = Field(
        default=None,
        ge=0.0,
        description="Average queries processed per second during evaluation",
    )

    # Dataset info
    num_queries: int = Field(default=0, ge=0)
    num_passages: int = Field(default=0, ge=0)

    @field_validator("recall_k", "precision_k")
    @classmethod
    def values_in_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"All metric values must be in [0, 1]; got {val} for key '{key}'"
                )
        return v

    @model_validator(mode="after")
    def latency_percentile_order(self) -> "MetricsResult":
        if self.latency_p50 > self.latency_p95:
            raise ValueError("latency_p50 cannot exceed latency_p95")
        if self.latency_p95 > self.latency_p99:
            raise ValueError("latency_p95 cannot exceed latency_p99")
        return self


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


class PerQueryMetrics(BaseModel):
    """Evaluation metrics for a single query."""

    query_id: str = Field(..., description="Unique query identifier")
    query_text: str = Field(..., description="Query string")
    ndcg_10: float = Field(..., ge=0.0, le=1.0)
    mrr_10: float = Field(..., ge=0.0, le=1.0)
    recall_10: float = Field(..., ge=0.0, le=1.0)
    precision_10: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0, description="End-to-end query latency in ms")
    num_relevant: int = Field(default=0, ge=0, description="Number of relevant passages in corpus")
    num_retrieved: int = Field(default=0, ge=0, description="Number of passages retrieved")
    num_relevant_retrieved: int = Field(
        default=0, ge=0, description="Relevant passages that were retrieved"
    )


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------


class ExperimentResult(BaseModel):
    """Full result of an experiment run."""

    experiment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Auto-generated UUID for the experiment",
    )
    config: ExperimentConfig = Field(..., description="The configuration that was evaluated")
    metrics: MetricsResult = Field(..., description="Aggregate evaluation metrics")
    per_query_metrics: list[PerQueryMetrics] = Field(
        default_factory=list,
        description="Per-query breakdown (may be empty for large datasets)",
    )
    artifact_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Paths to saved artifacts (e.g. raw predictions JSON)",
    )
    notes: str | None = Field(default=None, max_length=4096)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def summary(self) -> dict[str, float]:
        """Return a flat summary dict for quick comparison."""
        return {
            "ndcg_10": self.metrics.ndcg_10,
            "mrr_10": self.metrics.mrr_10,
            "map": self.metrics.map,
            "latency_p50_ms": self.metrics.latency_p50,
            "latency_p95_ms": self.metrics.latency_p95,
        }


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------


class SignificanceTestResult(BaseModel):
    """
    Result of a paired statistical significance test between two systems.

    Supported tests: paired t-test, Wilcoxon signed-rank test.
    """

    system_a: str = Field(..., description="Experiment ID of system A")
    system_b: str = Field(..., description="Experiment ID of system B")
    metric: str = Field(..., description="Metric name being compared (e.g. 'ndcg_10')")
    test_name: str = Field(
        default="paired_t_test",
        description="Statistical test used (paired_t_test | wilcoxon)",
    )

    # Descriptive statistics
    mean_a: float = Field(..., description="Mean metric value for system A")
    mean_b: float = Field(..., description="Mean metric value for system B")
    delta: float = Field(..., description="mean_b - mean_a (positive = B is better)")

    # Significance
    p_value: float = Field(..., ge=0.0, le=1.0, description="Two-tailed p-value")
    confidence_interval: tuple[float, float] = Field(
        ...,
        description="95% confidence interval for the mean difference (delta)",
    )
    is_significant: bool = Field(
        ...,
        description="True when p_value < alpha (default alpha=0.05)",
    )
    alpha: float = Field(default=0.05, ge=0.0, le=1.0, description="Significance level")
    effect_size: float | None = Field(
        default=None,
        description="Cohen's d effect size",
    )

    @field_validator("confidence_interval")
    @classmethod
    def ci_ordering(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError(
                f"confidence_interval lower bound ({v[0]}) must be <= upper bound ({v[1]})"
            )
        return v


# ---------------------------------------------------------------------------
# Experiment comparison
# ---------------------------------------------------------------------------


class ExperimentComparison(BaseModel):
    """
    Side-by-side comparison of multiple experiment results with
    optional pairwise significance testing.
    """

    experiments: list[ExperimentResult] = Field(
        ...,
        min_length=2,
        description="At least two experiment results to compare",
    )
    significance_tests: list[SignificanceTestResult] = Field(
        default_factory=list,
        description="Pairwise significance test results (may be empty if not computed)",
    )
    baseline_experiment_id: str | None = Field(
        default=None,
        description="ID of the experiment considered the baseline",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def ranked_by_ndcg(self) -> list[ExperimentResult]:
        """Return experiments sorted by NDCG@10 descending."""
        return sorted(
            self.experiments,
            key=lambda e: e.metrics.ndcg_10,
            reverse=True,
        )

    @property
    def best_system(self) -> ExperimentResult:
        """Return the experiment with the highest NDCG@10."""
        return self.ranked_by_ndcg[0]
