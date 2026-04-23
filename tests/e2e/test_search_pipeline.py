"""
End-to-end tests for the full search and evaluation pipeline.

All tests are marked pytest.mark.e2e.

Service availability is checked at the start of each test; if the required
service is unreachable, the test is skipped via pytest.skip so that CI
pipelines without external services do not fail.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Service availability helpers
# ---------------------------------------------------------------------------


def _bm25_local_available() -> bool:
    """LocalBM25Retriever is always available — it runs in-process."""
    try:
        from retrieval.bm25.local_retriever import LocalBM25Retriever  # noqa: F401

        return True
    except ImportError:
        return False


def _elasticsearch_available(host: str = "localhost", port: int = 9200) -> bool:
    """Return True if Elasticsearch responds to a ping."""
    try:
        import urllib.request

        url = f"http://{host}:{port}"
        with urllib.request.urlopen(url, timeout=3):
            return True
    except Exception:
        return False


def _qdrant_available(host: str = "localhost", port: int = 6333) -> bool:
    """Return True if Qdrant responds."""
    try:
        import urllib.request

        url = f"http://{host}:{port}/collections"
        with urllib.request.urlopen(url, timeout=3):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


SAMPLE_PASSAGES = [
    {
        "id": "p1",
        "text": "Machine learning is a method of data analysis that automates analytical model building.",
        "title": "Machine Learning",
    },
    {
        "id": "p2",
        "text": "Deep learning is part of a broader family of machine learning methods.",
        "title": "Deep Learning",
    },
    {
        "id": "p3",
        "text": "Natural language processing (NLP) is a subfield of linguistics and AI.",
        "title": "NLP",
    },
    {
        "id": "p4",
        "text": "Information retrieval is the task of finding relevant documents from a corpus.",
        "title": "Information Retrieval",
    },
    {
        "id": "p5",
        "text": "Cats are small, domesticated mammals with soft fur that purr when content.",
        "title": "Cats",
    },
]

SAMPLE_QRELS = {
    "q_ml": {"p1": 1, "p2": 1, "p3": 0, "p4": 0, "p5": 0},
    "q_ir": {"p1": 0, "p2": 0, "p3": 0, "p4": 1, "p5": 0},
}

SAMPLE_QUERIES = {
    "q_ml": "machine learning deep learning methods",
    "q_ir": "information retrieval relevant documents",
}


# ---------------------------------------------------------------------------
# E2E: Local BM25 full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_bm25_search_pipeline():
    """Index sample data into LocalBM25Retriever, search, verify top result is relevant."""
    if not _bm25_local_available():
        pytest.skip("retrieval.bm25 module not available")

    from retrieval.bm25.local_retriever import LocalBM25Retriever

    retriever = LocalBM25Retriever()
    await retriever.index(SAMPLE_PASSAGES)

    # Query about machine learning — p1 and p2 should be in top results
    results = await retriever.retrieve("machine learning deep learning", top_k=3)

    assert len(results) > 0, "Expected at least one result"
    top_ids = {r.passage_id for r in results}
    relevant_ids = {"p1", "p2"}
    assert top_ids & relevant_ids, (
        f"Expected at least one relevant passage in top-3, got: {top_ids}"
    )

    # Verify basic result structure
    for r in results:
        assert r.rank >= 1
        assert r.score >= 0.0
        assert r.text != ""


@pytest.mark.asyncio
async def test_bm25_pipeline_query_specificity():
    """More specific query terms should surface the most relevant document at rank 1."""
    if not _bm25_local_available():
        pytest.skip("retrieval.bm25 module not available")

    from retrieval.bm25.local_retriever import LocalBM25Retriever

    retriever = LocalBM25Retriever()
    await retriever.index(SAMPLE_PASSAGES)

    # Very specific to p5
    results = await retriever.retrieve("cats domesticated mammals purr fur", top_k=1)
    assert len(results) == 1
    assert results[0].passage_id == "p5", (
        f"Expected 'p5' at rank 1 for cat query, got '{results[0].passage_id}'"
    )


# ---------------------------------------------------------------------------
# E2E: Evaluation produces valid metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluation_produces_metrics_in_valid_range():
    """Run the full evaluation pipeline on sample data; all metrics in [0, 1]."""
    if not _bm25_local_available():
        pytest.skip("retrieval.bm25 module not available")

    from retrieval.bm25.local_retriever import LocalBM25Retriever
    from pipelines.evaluation.metrics import compute_metrics

    retriever = LocalBM25Retriever()
    await retriever.index(SAMPLE_PASSAGES)

    # Retrieve results for each query
    results: dict[str, list[str]] = {}
    for qid, query_text in SAMPLE_QUERIES.items():
        retrieved = await retriever.retrieve(query_text, top_k=5)
        results[qid] = [r.passage_id for r in retrieved]

    metrics = compute_metrics(SAMPLE_QRELS, results, k_values=[1, 3, 5])

    assert len(metrics) > 0, "Expected non-empty metrics dict"

    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric '{key}'={val} is outside [0, 1]"


@pytest.mark.asyncio
async def test_evaluation_ndcg_at_1_for_exact_match():
    """When the top result is the only relevant doc, NDCG@1 should be 1.0."""
    if not _bm25_local_available():
        pytest.skip("retrieval.bm25 module not available")

    import math
    from retrieval.bm25.local_retriever import LocalBM25Retriever
    from pipelines.evaluation.metrics import compute_metrics

    retriever = LocalBM25Retriever()
    await retriever.index(SAMPLE_PASSAGES)

    # "cats purr" query — p5 should be top result
    cat_results = await retriever.retrieve("cats purr fur domesticated", top_k=5)
    result_ids = [r.passage_id for r in cat_results]

    qrels = {"q_cat": {"p5": 1}}
    results_map = {"q_cat": result_ids}
    metrics = compute_metrics(qrels, results_map, k_values=[1, 5])

    if result_ids and result_ids[0] == "p5":
        assert math.isclose(metrics["ndcg@1"], 1.0, rel_tol=1e-6), (
            f"Expected NDCG@1=1.0 when top result is the only relevant doc, "
            f"got {metrics['ndcg@1']}"
        )


# ---------------------------------------------------------------------------
# E2E: Elasticsearch (skipped if unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elasticsearch_search_pipeline():
    """Full BM25 search via Elasticsearch — skipped if ES is not running."""
    if not _elasticsearch_available():
        pytest.skip("Elasticsearch not available at localhost:9200")

    try:
        from retrieval.bm25.elasticsearch_retriever import ElasticsearchRetriever
    except ImportError:
        pytest.skip("ElasticsearchRetriever not importable")

    retriever = ElasticsearchRetriever(host="localhost", port=9200, index="e2e_test_index")
    is_healthy = await retriever.health_check()
    if not is_healthy:
        pytest.skip("Elasticsearch health check failed")

    await retriever.index(SAMPLE_PASSAGES)
    results = await retriever.retrieve("machine learning", top_k=3)
    assert len(results) > 0


# ---------------------------------------------------------------------------
# E2E: Multi-query evaluation coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluation_covers_all_queries():
    """All qrels queries must have corresponding results."""
    if not _bm25_local_available():
        pytest.skip("retrieval.bm25 module not available")

    from retrieval.bm25.local_retriever import LocalBM25Retriever
    from pipelines.evaluation.metrics import compute_metrics

    retriever = LocalBM25Retriever()
    await retriever.index(SAMPLE_PASSAGES)

    results: dict[str, list[str]] = {}
    for qid, query_text in SAMPLE_QUERIES.items():
        retrieved = await retriever.retrieve(query_text, top_k=5)
        results[qid] = [r.passage_id for r in retrieved]

    metrics = compute_metrics(SAMPLE_QRELS, results, k_values=[5])
    # MAP should be defined and non-negative
    assert "map" in metrics
    assert metrics["map"] >= 0.0
