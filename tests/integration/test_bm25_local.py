"""
Integration tests for LocalBM25Retriever.

These tests run entirely in-process (no Elasticsearch required) and are
tagged pytest.mark.unit because they are fast and have no external deps.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest

from retrieval.bm25.local_retriever import LocalBM25Retriever
from retrieval.interfaces.base import RetrievalResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def passages():
    return [
        {"id": "1", "text": "Cats are small domesticated mammals that purr.", "title": "Cats"},
        {"id": "2", "text": "Dogs are loyal pets often called man's best friend.", "title": "Dogs"},
        {
            "id": "3",
            "text": "Machine learning is a subset of artificial intelligence.",
            "title": "ML",
        },
        {
            "id": "4",
            "text": "Neural networks are computational models inspired by the brain.",
            "title": "Neural Networks",
        },
        {
            "id": "5",
            "text": "Information retrieval systems find relevant documents for queries.",
            "title": "IR",
        },
    ]


@pytest.fixture
async def indexed_retriever(passages):
    retriever = LocalBM25Retriever()
    await retriever.index(passages)
    return retriever


# ---------------------------------------------------------------------------
# test_index_and_retrieve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_and_retrieve_returns_results(indexed_retriever):
    results = await indexed_retriever.retrieve("cat purr mammal")
    assert len(results) > 0
    assert isinstance(results[0], RetrievalResult)


@pytest.mark.asyncio
async def test_index_and_retrieve_relevant_at_top(indexed_retriever):
    results = await indexed_retriever.retrieve("machine learning artificial intelligence")
    # "ML" or "Neural Networks" passage should be top-ranked
    top_ids = [r.passage_id for r in results[:2]]
    assert "3" in top_ids or "4" in top_ids


@pytest.mark.asyncio
async def test_result_fields_populated(indexed_retriever):
    results = await indexed_retriever.retrieve("dog loyal pet")
    assert len(results) > 0
    top = results[0]
    assert top.passage_id != ""
    assert top.text != ""
    assert top.title != ""
    assert isinstance(top.score, float)
    assert top.rank == 1


@pytest.mark.asyncio
async def test_results_sorted_by_descending_score(indexed_retriever):
    results = await indexed_retriever.retrieve("neural network brain")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_ranks_are_1_based_sequential(indexed_retriever):
    results = await indexed_retriever.retrieve("information retrieval")
    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(ranks) + 1))


# ---------------------------------------------------------------------------
# test_top_k_limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_k_limit_returns_at_most_k(indexed_retriever):
    results = await indexed_retriever.retrieve("the", top_k=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_top_k_1_returns_single_result(indexed_retriever):
    results = await indexed_retriever.retrieve("cat", top_k=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_top_k_larger_than_corpus_returns_all(indexed_retriever, passages):
    results = await indexed_retriever.retrieve("the", top_k=1000)
    assert len(results) == len(passages)


# ---------------------------------------------------------------------------
# test_save_and_load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_load(indexed_retriever):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bm25_index.pkl"
        indexed_retriever.save(path)
        assert path.exists()

        loaded = LocalBM25Retriever.load(path)
        results = await loaded.retrieve("machine learning")
        assert len(results) > 0


@pytest.mark.asyncio
async def test_save_and_load_preserves_top_result(indexed_retriever):
    original_results = await indexed_retriever.retrieve("cat mammal purr", top_k=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.pkl"
        indexed_retriever.save(path)
        loaded = LocalBM25Retriever.load(path)
        loaded_results = await loaded.retrieve("cat mammal purr", top_k=1)
    assert original_results[0].passage_id == loaded_results[0].passage_id


@pytest.mark.asyncio
async def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        LocalBM25Retriever.load("/nonexistent/path/bm25.pkl")


def test_save_unindexed_retriever_raises():
    retriever = LocalBM25Retriever()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.pkl"
        with pytest.raises(RuntimeError, match="index"):
            retriever.save(path)


# ---------------------------------------------------------------------------
# test_empty_corpus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_corpus_index_does_not_raise():
    retriever = LocalBM25Retriever()
    # Indexing empty list should log a warning but not raise
    await retriever.index([])
    # Retriever should not be indexed
    assert not retriever._is_indexed


@pytest.mark.asyncio
async def test_retrieve_before_index_raises():
    retriever = LocalBM25Retriever()
    with pytest.raises(RuntimeError, match="index"):
        await retriever.retrieve("test")


@pytest.mark.asyncio
async def test_health_check_unindexed():
    retriever = LocalBM25Retriever()
    assert await retriever.health_check() is False


@pytest.mark.asyncio
async def test_health_check_indexed(indexed_retriever):
    assert await indexed_retriever.health_check() is True


# ---------------------------------------------------------------------------
# test_query_not_in_corpus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_not_in_corpus_returns_results_or_empty(indexed_retriever):
    """A query with no corpus matches should return empty or low-score results."""
    results = await indexed_retriever.retrieve("xyzzy qwerty flibbertigibbet")
    # All scores should be 0 or results should be empty
    if results:
        scores = [r.score for r in results]
        assert all(s == 0.0 for s in scores), f"Expected all-zero scores, got {scores}"


@pytest.mark.asyncio
async def test_empty_query_returns_empty(indexed_retriever):
    results = await indexed_retriever.retrieve("")
    assert results == []


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_empty(indexed_retriever):
    results = await indexed_retriever.retrieve("   ")
    assert results == []


# ---------------------------------------------------------------------------
# corpus_size property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corpus_size(indexed_retriever, passages):
    assert indexed_retriever.corpus_size == len(passages)


def test_corpus_size_unindexed():
    retriever = LocalBM25Retriever()
    assert retriever.corpus_size == 0
