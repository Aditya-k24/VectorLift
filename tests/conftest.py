"""
Root test fixtures shared across all test modules.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from typing import List, Dict
from unittest.mock import AsyncMock, MagicMock

from retrieval.interfaces.base import RetrievalResult


# ---------------------------------------------------------------------------
# Passage / corpus fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_passages() -> List[Dict]:
    return [
        {"id": "1", "text": "Cats are mammals that purr.", "title": "Cats"},
        {"id": "2", "text": "Dogs are loyal pets often called man's best friend.", "title": "Dogs"},
        {"id": "3", "text": "Machine learning is a subset of artificial intelligence.", "title": "ML"},
        {"id": "4", "text": "Neural networks are inspired by biological neurons.", "title": "Neural Networks"},
        {"id": "5", "text": "Information retrieval is the task of finding relevant documents.", "title": "IR"},
    ]


# ---------------------------------------------------------------------------
# Evaluation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_qrels() -> Dict[str, Dict[str, int]]:
    return {
        "q1": {"1": 1, "2": 0},
        "q2": {"3": 1, "4": 1, "5": 0},
    }


@pytest.fixture
def sample_results() -> Dict[str, List[str]]:
    return {
        "q1": ["1", "2", "3"],
        "q2": ["4", "3", "5"],
    }


# ---------------------------------------------------------------------------
# RetrievalResult fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_retrieval_results() -> List[RetrievalResult]:
    return [
        RetrievalResult(
            passage_id="1",
            text="Cats are mammals that purr.",
            title="Cats",
            score=0.95,
            rank=1,
        ),
        RetrievalResult(
            passage_id="2",
            text="Dogs are loyal pets often called man's best friend.",
            title="Dogs",
            score=0.87,
            rank=2,
        ),
        RetrievalResult(
            passage_id="3",
            text="Machine learning is a subset of artificial intelligence.",
            title="ML",
            score=0.76,
            rank=3,
        ),
        RetrievalResult(
            passage_id="4",
            text="Neural networks are inspired by biological neurons.",
            title="Neural Networks",
            score=0.65,
            rank=4,
        ),
        RetrievalResult(
            passage_id="5",
            text="Information retrieval is the task of finding relevant documents.",
            title="IR",
            score=0.54,
            rank=5,
        ),
    ]


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Return a mock Settings object with test-friendly values."""
    settings = MagicMock()
    settings.app_env = "dev"
    settings.debug = True
    settings.log_level = "DEBUG"
    settings.top_k_default = 10
    settings.rerank_top_n_default = 100
    settings.hybrid_bm25_weight = 0.3
    settings.hybrid_dense_weight = 0.7
    settings.rrf_k = 60

    # Nested sub-settings
    settings.elasticsearch.host = "localhost"
    settings.elasticsearch.port = 9200
    settings.elasticsearch.index = "test_passages"

    settings.qdrant.host = "localhost"
    settings.qdrant.port = 6333
    settings.qdrant.collection = "test_dense"

    settings.model.biencoder_pretrained = "sentence-transformers/all-MiniLM-L6-v2"
    settings.model.crossencoder_pretrained = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    settings.model.embedding_dim = 384
    settings.model.biencoder_device = "cpu"

    settings.api.host = "0.0.0.0"
    settings.api.port = 8000

    return settings


# ---------------------------------------------------------------------------
# Async HTTP client fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_client():
    """
    httpx AsyncClient pointed at the FastAPI test app.

    Uses httpx's ASGITransport so no real network socket is opened.
    The search_pipeline on app.state is replaced with a mock so tests
    do not require ML models or external services.
    """
    import httpx
    from apps.api.main import create_app

    app = create_app()

    # Build a minimal mock pipeline that satisfies SearchService._dispatch
    mock_result = RetrievalResult(
        passage_id="p1",
        text="Test passage text",
        title="Test Title",
        score=0.9,
        rank=1,
    )

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[mock_result])

    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(return_value=[mock_result])

    mock_pipeline = MagicMock()
    mock_pipeline.bm25_retriever = mock_retriever
    mock_pipeline.dense_retriever = mock_retriever
    mock_pipeline.hybrid_retriever = mock_retriever
    mock_pipeline.reranker = mock_reranker

    # Inject pipeline into app state before the lifespan kicks in
    app.state.search_pipeline = mock_pipeline

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
