"""
VectorLift — Search API Schemas
=================================
Pydantic v2 models that define the public contract for the search,
rerank and health-check endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from core.config.settings import RetrievalMode


# ---------------------------------------------------------------------------
# Passage — the atomic unit returned by retrieval
# ---------------------------------------------------------------------------


class Passage(BaseModel):
    """A single retrieved or reranked passage."""

    id: str = Field(..., description="Unique passage / document identifier")
    text: str = Field(..., description="Passage text content", min_length=1)
    title: str | None = Field(default=None, description="Document or section title")
    score: float = Field(..., description="Retrieval or reranking score")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (source, section, url, …)",
    )

    model_config = {"populate_by_name": True}

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError(f"score must be a finite number, got {v}")
        return v


# ---------------------------------------------------------------------------
# SearchRequest / SearchResponse
# ---------------------------------------------------------------------------


class SearchFilters(BaseModel):
    """Optional pre-filtering applied before retrieval."""

    source: str | None = Field(default=None, description="Filter by document source")
    language: str | None = Field(default=None, description="ISO 639-1 language code")
    date_from: str | None = Field(
        default=None,
        description="ISO 8601 date — only return passages published on or after this date",
    )
    date_to: str | None = Field(
        default=None,
        description="ISO 8601 date — only return passages published on or before this date",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="AND-filter: passage must carry all listed tags",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra filter key-value pairs passed to the backend",
    )


class SearchRequest(BaseModel):
    """Incoming search query payload."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Natural-language query string",
    )
    mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Which retrieval pipeline to use",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of final results to return",
    )
    rerank_top_n: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description=(
            "Number of first-stage candidates to fetch before reranking "
            "(only meaningful when mode=rerank or mode=hybrid)"
        ),
    )
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Optional metadata pre-filters",
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include passage metadata in the response",
    )
    explain: bool = Field(
        default=False,
        description="Include debug/explanation information in response metadata",
    )

    @model_validator(mode="after")
    def rerank_top_n_gte_top_k(self) -> "SearchRequest":
        if self.rerank_top_n < self.top_k:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) must be >= top_k ({self.top_k})"
            )
        return self


class LatencyBreakdown(BaseModel):
    """Per-stage latency in milliseconds."""

    total_ms: float = Field(..., ge=0)
    retrieval_ms: float | None = Field(default=None, ge=0)
    rerank_ms: float | None = Field(default=None, ge=0)
    embedding_ms: float | None = Field(default=None, ge=0)
    bm25_ms: float | None = Field(default=None, ge=0)
    fusion_ms: float | None = Field(default=None, ge=0)


class SearchResponse(BaseModel):
    """Full search response envelope."""

    query: str = Field(..., description="Echo of the original query")
    results: list[Passage] = Field(
        default_factory=list,
        description="Ordered list of retrieved passages (best first)",
    )
    latency: LatencyBreakdown = Field(..., description="Per-stage latency breakdown")
    mode: RetrievalMode = Field(..., description="Retrieval pipeline that was used")
    total_candidates: int = Field(
        default=0,
        ge=0,
        description="Number of candidates considered before final ranking",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Debug / explanation data (populated when explain=True)",
    )


# ---------------------------------------------------------------------------
# RerankRequest / RerankResponse  (standalone reranker endpoint)
# ---------------------------------------------------------------------------


class RerankRequest(BaseModel):
    """Request to rerank an externally supplied list of passages."""

    query: str = Field(..., min_length=1, max_length=2048)
    passages: list[Passage] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Candidate passages to rerank",
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="Return only the top-N reranked passages; None returns all",
    )

    @model_validator(mode="after")
    def top_n_lte_passages(self) -> "RerankRequest":
        if self.top_n is not None and self.top_n > len(self.passages):
            raise ValueError(
                f"top_n ({self.top_n}) cannot exceed the number of passages ({len(self.passages)})"
            )
        return self


class RerankResponse(BaseModel):
    """Response from the standalone reranker."""

    query: str
    results: list[Passage] = Field(
        ...,
        description="Reranked passages, ordered best-first, with updated scores",
    )
    latency_ms: float = Field(..., ge=0, description="Total reranking latency in ms")
    model_id: str = Field(
        ...,
        description="Identifier of the cross-encoder model that performed reranking",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class DependencyStatus(BaseModel):
    """Status of a single external dependency."""

    status: str = Field(..., description="healthy | degraded | down")
    latency_ms: float | None = Field(default=None, ge=0)
    detail: str | None = Field(default=None)


class HealthResponse(BaseModel):
    """API health-check response."""

    status: str = Field(..., description="healthy | degraded | down")
    version: str = Field(..., description="Application version string")
    dependencies: dict[str, DependencyStatus] = Field(
        default_factory=dict,
        description="Health status of each downstream dependency",
    )
    uptime_seconds: float | None = Field(default=None, ge=0)

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"
