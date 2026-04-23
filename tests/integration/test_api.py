"""
Integration tests for the FastAPI application.

These tests use httpx with ASGITransport to call the real FastAPI app
without a network socket.  The search_pipeline in app.state is replaced
with a lightweight mock so no ML models or external services are required.

All tests are marked with pytest.mark.integration.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from retrieval.interfaces.base import RetrievalResult

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared mock pipeline factory
# ---------------------------------------------------------------------------


def _build_mock_pipeline(results: list[RetrievalResult] | None = None) -> Any:
    """Return a duck-typed pipeline mock accepted by SearchService._dispatch."""
    if results is None:
        results = [
            RetrievalResult(
                passage_id="p1",
                text="Relevant passage about machine learning.",
                title="Machine Learning",
                score=0.93,
                rank=1,
            ),
            RetrievalResult(
                passage_id="p2",
                text="Another passage about neural networks.",
                title="Neural Nets",
                score=0.85,
                rank=2,
            ),
        ]

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=results)

    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(return_value=results[:1])

    pipeline = MagicMock()
    pipeline.bm25_retriever = mock_retriever
    pipeline.dense_retriever = mock_retriever
    pipeline.hybrid_retriever = mock_retriever
    pipeline.reranker = mock_reranker

    return pipeline


@pytest_asyncio.fixture
async def client():
    """Provide an httpx AsyncClient backed by the FastAPI test app."""
    from apps.api.main import create_app

    app = create_app()
    app.state.search_pipeline = _build_mock_pipeline()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] in ("healthy", "degraded", "unhealthy")


# ---------------------------------------------------------------------------
# POST /search — mode variants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_bm25_mode(client: httpx.AsyncClient):
    payload = {"query": "machine learning", "mode": "bm25", "top_k": 5}
    response = await client.post("/search", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert body["query"] == "machine learning"
    assert body["mode"] == "bm25"


@pytest.mark.asyncio
async def test_search_dense_mode(client: httpx.AsyncClient):
    payload = {"query": "neural networks", "mode": "dense", "top_k": 3}
    response = await client.post("/search", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "dense"
    assert len(body["results"]) <= 3


@pytest.mark.asyncio
async def test_search_hybrid_mode(client: httpx.AsyncClient):
    payload = {"query": "information retrieval", "mode": "hybrid", "top_k": 10}
    response = await client.post("/search", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"


@pytest.mark.asyncio
async def test_search_rerank_mode(client: httpx.AsyncClient):
    payload = {
        "query": "semantic search ranking",
        "mode": "rerank",
        "top_k": 5,
        "retrieval_multiplier": 3,
    }
    response = await client.post("/search", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "rerank"


@pytest.mark.asyncio
async def test_search_response_has_latency(client: httpx.AsyncClient):
    payload = {"query": "test", "mode": "bm25"}
    response = await client.post("/search", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "latency" in body
    latency = body["latency"]
    assert "total_ms" in latency
    assert latency["total_ms"] >= 0


@pytest.mark.asyncio
async def test_search_result_structure(client: httpx.AsyncClient):
    payload = {"query": "cats", "mode": "bm25"}
    response = await client.post("/search", json=payload)
    body = response.json()
    for result in body["results"]:
        assert "passage_id" in result
        assert "text" in result
        assert "title" in result
        assert "score" in result
        assert "rank" in result


# ---------------------------------------------------------------------------
# POST /search/rerank
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_endpoint(client: httpx.AsyncClient):
    payload = {
        "query": "machine learning basics",
        "candidates": [
            {
                "passage_id": "p1",
                "text": "Machine learning is a subset of AI.",
                "title": "ML",
                "score": 0.8,
                "rank": 1,
            },
            {
                "passage_id": "p2",
                "text": "Deep learning uses neural networks.",
                "title": "DL",
                "score": 0.7,
                "rank": 2,
            },
        ],
        "top_n": 2,
    }
    response = await client.post("/search/rerank", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert "rerank_ms" in body
    assert body["query"] == "machine learning basics"


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_info_endpoint(client: httpx.AsyncClient):
    response = await client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert isinstance(body["models"], list)
    assert len(body["models"]) >= 1
    for model in body["models"]:
        assert "name" in model
        assert "checkpoint" in model


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_search_request_missing_query(client: httpx.AsyncClient):
    payload = {"mode": "bm25", "top_k": 5}  # query is required
    response = await client.post("/search", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_search_mode_returns_422(client: httpx.AsyncClient):
    payload = {"query": "test", "mode": "invalid_mode"}
    response = await client.post("/search", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_top_k_too_large_returns_422(client: httpx.AsyncClient):
    payload = {"query": "test", "mode": "bm25", "top_k": 999}
    response = await client.post("/search", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rerank_with_no_candidates_returns_422(client: httpx.AsyncClient):
    payload = {"query": "test", "candidates": [], "top_n": 5}
    response = await client.post("/search/rerank", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Monkeypatch pipeline (demonstrating monkeypatch approach)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_uses_injected_pipeline(monkeypatch):
    """Verify that monkeypatching the pipeline in app.state works correctly."""
    from apps.api.main import create_app

    app = create_app()

    custom_result = RetrievalResult(
        passage_id="custom-id",
        text="Custom monkeypatched result",
        title="Custom",
        score=0.99,
        rank=1,
    )
    pipeline = _build_mock_pipeline(results=[custom_result])
    app.state.search_pipeline = pipeline

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/search", json={"query": "test", "mode": "bm25"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["passage_id"] == "custom-id"
