"""
VectorLift — Async SQLAlchemy Database Layer
=============================================
Provides:
* Async SQLAlchemy engine + session factory (asyncpg driver)
* Declarative ``Base`` for all ORM models
* ``get_db()`` async generator for FastAPI dependency injection
* ORM table definitions:
    - ``Experiment``         — experiment configuration snapshots
    - ``ExperimentResult``   — aggregate metrics per run
    - ``PerQueryMetrics``    — per-query IR metrics for detailed analysis

The module lazily creates the engine on first import so that import-time
configuration errors are surfaced clearly.

Usage
-----
    # FastAPI dependency injection
    from core.common.database import get_db

    @app.get("/experiments")
    async def list_experiments(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Experiment))
        return result.scalars().all()

    # Standalone (e.g. in a Prefect task)
    from core.common.database import get_engine, AsyncSessionFactory

    async with AsyncSessionFactory() as session:
        session.add(experiment_row)
        await session.commit()
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from core.logging.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Shared declarative base for all VectorLift ORM models.

    Provides:
    * ``id``        — auto-generated UUID primary key
    * ``created_at`` — UTC timestamp set on insert
    * ``updated_at`` — UTC timestamp updated on every write
    """

    # Shared columns are defined on concrete subclasses via mixins below


class TimestampMixin:
    """Adds ``created_at`` and ``updated_at`` timestamp columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Row creation timestamp (UTC)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last row update timestamp (UTC)",
    )


class UUIDPrimaryKeyMixin:
    """Adds a UUID v4 primary key column."""

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        comment="Unique row identifier (UUID v4)",
    )


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class Experiment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Snapshot of an experiment configuration.

    One row per experiment run.  The ``config_json`` column stores the full
    ``ExperimentConfig`` Pydantic model serialised to JSON so that the exact
    configuration is preserved even if the schema evolves.
    """

    __tablename__ = "experiments"
    __table_args__ = (
        Index("ix_experiments_name", "name"),
        Index("ix_experiments_retrieval_mode", "retrieval_mode"),
        Index("ix_experiments_dataset_mode", "dataset_mode"),
        Index("ix_experiments_created_at", "created_at"),
        {"comment": "Experiment configuration snapshots"},
    )

    # Human-readable identifiers
    name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Human-readable experiment name",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional free-text description",
    )

    # Retrieval configuration (denormalised for fast filtering)
    retrieval_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Retrieval pipeline: bm25 | dense | hybrid | rerank",
    )
    dataset_mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Dataset split: dev | small | full",
    )
    top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        comment="Final result set size",
    )
    rerank_top_n: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        comment="Candidate pool size before reranking",
    )
    hybrid_bm25_weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.3,
        comment="BM25 weight in hybrid fusion",
    )
    fusion_strategy: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="reciprocal_rank_fusion",
        comment="Hybrid fusion strategy",
    )

    # Model identifiers (denormalised for filtering / grouping)
    biencoder_model: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Bi-encoder model ID or path",
    )
    crossencoder_model: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Cross-encoder model ID or path (NULL if not used)",
    )

    # Full config blob
    config_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment="Full ExperimentConfig JSON snapshot",
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="pending | running | completed | failed",
    )
    tags: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Arbitrary string tags",
    )

    # Relationships
    result: Mapped["ExperimentResult | None"] = relationship(
        "ExperimentResult",
        back_populates="experiment",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Experiment id={self.id} name={self.name!r} mode={self.retrieval_mode}>"
        )


class ExperimentResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Aggregate evaluation metrics for one experiment run.

    Each :class:`Experiment` has at most one ``ExperimentResult``.
    """

    __tablename__ = "experiment_results"
    __table_args__ = (
        UniqueConstraint("experiment_id", name="uq_experiment_results_experiment_id"),
        Index("ix_experiment_results_ndcg_10", "ndcg_10"),
        Index("ix_experiment_results_mrr_10", "mrr_10"),
        {"comment": "Aggregate IR metrics per experiment run"},
    )

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to experiments.id",
    )

    # Core IR metrics
    ndcg_10: Mapped[float] = mapped_column(Float, nullable=False, comment="NDCG@10")
    mrr_10: Mapped[float] = mapped_column(Float, nullable=False, comment="MRR@10")
    map_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Mean Average Precision",
    )

    # Recall / Precision at multiple cut-offs (stored as JSON)
    recall_k: Mapped[dict[str, float]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Recall@k — {k: value}",
    )
    precision_k: Mapped[dict[str, float]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Precision@k — {k: value}",
    )

    # Latency percentiles (milliseconds)
    latency_p50_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Median latency in ms",
    )
    latency_p95_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="95th-percentile latency in ms",
    )
    latency_p99_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="99th-percentile latency in ms",
    )

    # Throughput
    queries_per_second: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average queries per second during eval",
    )

    # Dataset statistics
    num_queries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    num_passages: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
    )

    # Full metrics blob (for forward compatibility)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Complete MetricsResult JSON snapshot",
    )

    # Optional notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_paths: Mapped[dict[str, str]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Paths to saved artifacts (raw predictions, etc.)",
    )

    # Relationship
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="result",
    )
    per_query_metrics: Mapped[list["PerQueryMetrics"]] = relationship(
        "PerQueryMetrics",
        back_populates="experiment_result",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentResult id={self.id} "
            f"experiment_id={self.experiment_id} "
            f"ndcg_10={self.ndcg_10:.4f}>"
        )


class PerQueryMetrics(UUIDPrimaryKeyMixin, Base):
    """
    Per-query IR metrics for a single experiment result.

    Stored separately from :class:`ExperimentResult` because there can be
    tens of thousands of queries per experiment.

    Notes
    -----
    ``created_at`` is not included to reduce row size; use the parent
    ``ExperimentResult.created_at`` for time-based queries.
    """

    __tablename__ = "per_query_metrics"
    __table_args__ = (
        Index("ix_per_query_metrics_result_id", "experiment_result_id"),
        Index("ix_per_query_metrics_query_id", "query_id"),
        Index("ix_per_query_metrics_ndcg_10", "ndcg_10"),
        {"comment": "Per-query IR metrics for detailed experiment analysis"},
    )

    experiment_result_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiment_results.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to experiment_results.id",
    )

    # Query identity
    query_id: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Dataset-assigned query identifier",
    )
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Query string",
    )

    # Per-query metrics
    ndcg_10: Mapped[float] = mapped_column(Float, nullable=False)
    mrr_10: Mapped[float] = mapped_column(Float, nullable=False)
    recall_10: Mapped[float] = mapped_column(Float, nullable=False)
    precision_10: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="End-to-end query latency in ms",
    )

    # Retrieved set statistics
    num_relevant: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of relevant passages in the corpus for this query",
    )
    num_retrieved: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of passages retrieved",
    )
    num_relevant_retrieved: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Intersection: relevant ∩ retrieved",
    )

    # Relationship
    experiment_result: Mapped["ExperimentResult"] = relationship(
        "ExperimentResult",
        back_populates="per_query_metrics",
    )

    def __repr__(self) -> str:
        return (
            f"<PerQueryMetrics id={self.id} "
            f"query_id={self.query_id!r} "
            f"ndcg_10={self.ndcg_10:.4f}>"
        )


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
AsyncSessionFactory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Return the application-wide async SQLAlchemy engine.

    Creates the engine on first call and caches it.  Thread-safe because
    the GIL protects the module-level assignment and SQLAlchemy's connection
    pool is itself thread-safe.
    """
    global _engine, AsyncSessionFactory

    if _engine is not None:
        return _engine

    from core.config.settings import get_settings

    settings = get_settings()
    db_url = settings.postgres.async_url

    _engine = create_async_engine(
        db_url,
        pool_size=settings.postgres.pool_size,
        max_overflow=settings.postgres.max_overflow,
        echo=settings.postgres.echo,
        # Recycle connections after 30 minutes to avoid stale connections
        pool_recycle=1_800,
        # Check connection health before returning from pool
        pool_pre_ping=True,
        # JSON serialiser
        json_serializer=_json_serialiser,
        json_deserializer=_json_deserialiser,
        future=True,
    )

    AsyncSessionFactory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autobegin=True,
    )

    logger.info("SQLAlchemy async engine created", extra={"url": _redact_url(db_url)})
    return _engine


# ---------------------------------------------------------------------------
# Dependency — get_db
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an :class:`AsyncSession`.

    Commits on success; rolls back on any exception; always closes the session.

    Usage
    -----
        @router.post("/experiments")
        async def create(db: AsyncSession = Depends(get_db)):
            ...
    """
    engine = get_engine()
    if AsyncSessionFactory is None:
        raise RuntimeError("AsyncSessionFactory is not initialised. Call get_engine() first.")

    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


async def create_all_tables() -> None:
    """
    Create all tables defined in this module against the live database.

    Intended for use in development and test setups.
    In production use Alembic migrations instead.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created.")


async def drop_all_tables() -> None:
    """
    Drop all tables.  Destructive — only use in testing.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped.")


async def dispose_engine() -> None:
    """Dispose the engine (close all pooled connections)."""
    global _engine, AsyncSessionFactory
    if _engine is not None:
        await _engine.dispose()
        logger.info("SQLAlchemy engine disposed.")
    _engine = None
    AsyncSessionFactory = None


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _redact_url(url: str) -> str:
    """Replace password in a database URL with ***."""
    import re

    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def _json_serialiser(obj: Any) -> str:
    import json

    def default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, uuid.UUID):
            return str(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=default)


def _json_deserialiser(data: str) -> Any:
    import json

    return json.loads(data)
