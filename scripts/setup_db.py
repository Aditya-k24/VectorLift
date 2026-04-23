"""
VectorLift — Database Setup Script
====================================
Creates all PostgreSQL tables defined in ``core.common.database`` and seeds
the database with initial experiment configuration records.

Usage
-----
    # From the repo root:
    python scripts/setup_db.py

    # Or as a module:
    python -m scripts.setup_db

Environment Variables
---------------------
All standard VectorLift env vars apply (read from .env or the environment):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

The script is idempotent — it can be run multiple times without
dropping or duplicating data.  Tables are created with ``checkfirst=True``
so pre-existing tables are left untouched.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run as a standalone script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.common.database import (
    AsyncSessionFactory,
    Experiment,
    ExperimentResult,
    PerQueryMetrics,
    create_all_tables,
    get_engine,
)
from core.config.settings import get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str, *, level: str = "INFO") -> None:
    colours = {"INFO": "\033[0;34m", "OK": "\033[0;32m", "WARN": "\033[1;33m", "ERROR": "\033[0;31m"}
    reset = "\033[0m"
    colour = colours.get(level, "")
    print(f"{colour}[{level:5s}]{reset} {msg}")


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

async def create_schema() -> None:
    """Create all ORM-defined tables if they do not already exist."""
    _print("Creating database schema (create_all — idempotent)...")
    await create_all_tables()
    _print("Schema is up to date.", level="OK")


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------

async def ensure_extensions(session: AsyncSession) -> None:
    """Ensure required PostgreSQL extensions are installed."""
    extensions = ["uuid-ossp", "pg_trgm"]
    for ext in extensions:
        try:
            await session.execute(
                text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
            )
            _print(f"Extension '{ext}' is available.", level="OK")
        except Exception as exc:
            _print(f"Could not create extension '{ext}': {exc}", level="WARN")
    await session.commit()


# ---------------------------------------------------------------------------
# Seed data — initial experiment configurations
# ---------------------------------------------------------------------------

_SEED_EXPERIMENTS: list[dict] = [
    {
        "name": "baseline-bm25",
        "description": (
            "BM25 baseline — pure keyword retrieval via Elasticsearch. "
            "No dense vectors, no reranking."
        ),
        "retrieval_mode": "bm25",
        "dataset_mode": "dev",
        "top_k": 10,
        "rerank_top_n": 100,
        "hybrid_bm25_weight": 1.0,
        "fusion_strategy": "reciprocal_rank_fusion",
        "biencoder_model": "N/A",
        "crossencoder_model": None,
        "status": "pending",
        "tags": ["baseline", "bm25", "dev"],
        "config_json": {
            "retrieval_mode": "bm25",
            "dataset_mode": "dev",
            "top_k": 10,
            "rerank_top_n": 100,
            "biencoder": "N/A",
            "crossencoder": None,
            "hybrid_bm25_weight": 1.0,
            "hybrid_dense_weight": 0.0,
            "fusion_strategy": "reciprocal_rank_fusion",
        },
    },
    {
        "name": "baseline-dense-msmarco-distilbert",
        "description": (
            "Dense retrieval baseline using msmarco-distilbert-base-tas-b "
            "bi-encoder and Qdrant vector store."
        ),
        "retrieval_mode": "dense",
        "dataset_mode": "dev",
        "top_k": 10,
        "rerank_top_n": 100,
        "hybrid_bm25_weight": 0.0,
        "fusion_strategy": "reciprocal_rank_fusion",
        "biencoder_model": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "crossencoder_model": None,
        "status": "pending",
        "tags": ["baseline", "dense", "distilbert", "dev"],
        "config_json": {
            "retrieval_mode": "dense",
            "dataset_mode": "dev",
            "top_k": 10,
            "rerank_top_n": 100,
            "biencoder": "sentence-transformers/msmarco-distilbert-base-tas-b",
            "crossencoder": None,
            "hybrid_bm25_weight": 0.0,
            "hybrid_dense_weight": 1.0,
            "fusion_strategy": "reciprocal_rank_fusion",
        },
    },
    {
        "name": "hybrid-rrf-30-70",
        "description": (
            "Hybrid retrieval with Reciprocal Rank Fusion. "
            "BM25 weight=0.3, Dense weight=0.7."
        ),
        "retrieval_mode": "hybrid",
        "dataset_mode": "dev",
        "top_k": 10,
        "rerank_top_n": 100,
        "hybrid_bm25_weight": 0.3,
        "fusion_strategy": "reciprocal_rank_fusion",
        "biencoder_model": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "crossencoder_model": None,
        "status": "pending",
        "tags": ["hybrid", "rrf", "dev"],
        "config_json": {
            "retrieval_mode": "hybrid",
            "dataset_mode": "dev",
            "top_k": 10,
            "rerank_top_n": 100,
            "biencoder": "sentence-transformers/msmarco-distilbert-base-tas-b",
            "crossencoder": None,
            "hybrid_bm25_weight": 0.3,
            "hybrid_dense_weight": 0.7,
            "fusion_strategy": "reciprocal_rank_fusion",
            "rrf_k": 60,
        },
    },
    {
        "name": "rerank-cross-encoder-minilm",
        "description": (
            "Full pipeline: hybrid retrieval (RRF 30/70) + cross-encoder "
            "reranking with ms-marco-MiniLM-L-6-v2. Best accuracy baseline."
        ),
        "retrieval_mode": "rerank",
        "dataset_mode": "dev",
        "top_k": 10,
        "rerank_top_n": 100,
        "hybrid_bm25_weight": 0.3,
        "fusion_strategy": "reciprocal_rank_fusion",
        "biencoder_model": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "crossencoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "status": "pending",
        "tags": ["rerank", "cross-encoder", "minilm", "dev"],
        "config_json": {
            "retrieval_mode": "rerank",
            "dataset_mode": "dev",
            "top_k": 10,
            "rerank_top_n": 100,
            "biencoder": "sentence-transformers/msmarco-distilbert-base-tas-b",
            "crossencoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "hybrid_bm25_weight": 0.3,
            "hybrid_dense_weight": 0.7,
            "fusion_strategy": "reciprocal_rank_fusion",
            "rrf_k": 60,
            "crossencoder_batch_size": 32,
            "crossencoder_max_length": 512,
        },
    },
    {
        "name": "hybrid-rrf-50-50",
        "description": (
            "Hybrid retrieval with equal BM25/Dense weighting (50/50). "
            "Ablation to compare against the default 30/70 split."
        ),
        "retrieval_mode": "hybrid",
        "dataset_mode": "dev",
        "top_k": 10,
        "rerank_top_n": 100,
        "hybrid_bm25_weight": 0.5,
        "fusion_strategy": "reciprocal_rank_fusion",
        "biencoder_model": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "crossencoder_model": None,
        "status": "pending",
        "tags": ["hybrid", "rrf", "ablation", "dev"],
        "config_json": {
            "retrieval_mode": "hybrid",
            "dataset_mode": "dev",
            "top_k": 10,
            "rerank_top_n": 100,
            "biencoder": "sentence-transformers/msmarco-distilbert-base-tas-b",
            "crossencoder": None,
            "hybrid_bm25_weight": 0.5,
            "hybrid_dense_weight": 0.5,
            "fusion_strategy": "reciprocal_rank_fusion",
            "rrf_k": 60,
        },
    },
]


async def seed_experiments(session: AsyncSession) -> None:
    """
    Insert seed experiment records if they do not already exist.

    Uses name as the uniqueness key — existing rows are skipped.
    """
    _print("Seeding initial experiment configurations...")

    # Fetch existing experiment names to avoid duplicates
    from sqlalchemy import select
    result = await session.execute(select(Experiment.name))
    existing_names: set[str] = {row[0] for row in result.fetchall()}

    inserted = 0
    skipped = 0

    for config in _SEED_EXPERIMENTS:
        name = config["name"]
        if name in existing_names:
            _print(f"  Experiment '{name}' already exists — skipping.", level="WARN")
            skipped += 1
            continue

        experiment = Experiment(
            id=uuid.uuid4(),
            name=name,
            description=config.get("description"),
            retrieval_mode=config["retrieval_mode"],
            dataset_mode=config["dataset_mode"],
            top_k=config["top_k"],
            rerank_top_n=config["rerank_top_n"],
            hybrid_bm25_weight=config["hybrid_bm25_weight"],
            fusion_strategy=config["fusion_strategy"],
            biencoder_model=config["biencoder_model"],
            crossencoder_model=config.get("crossencoder_model"),
            config_json=config["config_json"],
            status=config.get("status", "pending"),
            tags=config.get("tags", []),
        )
        session.add(experiment)
        _print(f"  Inserting experiment: '{name}'", level="INFO")
        inserted += 1

    await session.commit()
    _print(
        f"Seeding complete — {inserted} inserted, {skipped} skipped.",
        level="OK",
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

async def verify_schema(session: AsyncSession) -> None:
    """Run a quick sanity check that core tables are queryable."""
    _print("Verifying schema...")

    checks: list[tuple[str, str]] = [
        ("experiments", "SELECT COUNT(*) FROM experiments"),
        ("experiment_results", "SELECT COUNT(*) FROM experiment_results"),
        ("per_query_metrics", "SELECT COUNT(*) FROM per_query_metrics"),
    ]

    for table_name, query in checks:
        result = await session.execute(text(query))
        count = result.scalar_one()
        _print(f"  Table '{table_name}': {count} row(s).", level="OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    settings = get_settings()

    _print("=" * 60)
    _print("VectorLift — Database Setup")
    _print("=" * 60)
    _print(f"  Host     : {settings.postgres.host}:{settings.postgres.port}")
    _print(f"  Database : {settings.postgres.db}")
    _print(f"  User     : {settings.postgres.user}")
    _print(f"  Env      : {settings.app_env.value}")
    _print("-" * 60)

    # Initialise the engine (also sets AsyncSessionFactory)
    engine = get_engine()

    # 1. Create tables
    await create_schema()

    # 2. Ensure PostgreSQL extensions
    if AsyncSessionFactory is None:
        raise RuntimeError("AsyncSessionFactory not initialised after get_engine().")

    async with AsyncSessionFactory() as session:
        await ensure_extensions(session)

    # 3. Seed initial data
    async with AsyncSessionFactory() as session:
        await seed_experiments(session)

    # 4. Verify
    async with AsyncSessionFactory() as session:
        await verify_schema(session)

    # 5. Dispose engine
    await engine.dispose()

    _print("=" * 60)
    _print("Database setup complete!", level="OK")
    _print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
