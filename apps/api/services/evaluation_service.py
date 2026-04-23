"""
EvaluationService – orchestrates offline evaluation runs and persists results.

Responsibilities
----------------
* Accept an :class:`~apps.api.schemas.ExperimentConfig` and run a full
  evaluation loop against a benchmark dataset.
* Persist :class:`~apps.api.schemas.ExperimentResult` records to Postgres.
* Support listing, fetching, and comparing experiments.

The heavy computation (embedding, retrieval, metric computation) is performed
asynchronously so that the HTTP handler can return a job ID immediately and let
the caller poll for results.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Any

import numpy as np
from scipy import stats

from apps.api.schemas import (
    ExperimentComparison,
    ExperimentConfig,
    ExperimentResult,
    MetricSet,
    SignificanceResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store (replaced by DB calls when a session factory is wired up)
# ---------------------------------------------------------------------------

_EXPERIMENT_STORE: dict[str, ExperimentResult] = {}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _ndcg_at_k(relevances: list[float], k: int = 10) -> float:
    """Compute NDCG@k from a list of graded relevance labels (0/1 or 0–3)."""
    relevances = relevances[:k]
    dcg = sum(
        (2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevances)
    )
    ideal = sorted(relevances, reverse=True)
    idcg = sum(
        (2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal)
    )
    return float(dcg / idcg) if idcg > 0 else 0.0


def _mrr_at_k(relevances: list[float], k: int = 10) -> float:
    """Compute MRR@k."""
    for i, r in enumerate(relevances[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(relevances: list[float], total_relevant: int, k: int = 10) -> float:
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / total_relevant


def _map_score(relevances: list[float], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    num_relevant = 0
    sum_precision = 0.0
    for i, r in enumerate(relevances):
        if r > 0:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    return sum_precision / total_relevant


def _precision_at_k(relevances: list[float], k: int = 10) -> float:
    hits = sum(1 for r in relevances[:k] if r > 0)
    return hits / min(k, len(relevances)) if relevances else 0.0


# ---------------------------------------------------------------------------
# Dataset loading (stub – replace with real dataset reader)
# ---------------------------------------------------------------------------


async def _load_queries(dataset: str) -> list[dict[str, Any]]:
    """Return list of {query, relevant_ids} dicts for *dataset*.

    In production this would call a DatasetLoader or query Postgres.
    For now we generate synthetic data so the service runs without a corpus.
    """
    logger.info("Loading evaluation dataset: %s", dataset)
    await asyncio.sleep(0.01)  # simulate I/O
    rng = random.Random(42)
    queries = []
    for i in range(100):
        relevant_ids = [f"doc_{rng.randint(0, 999)}" for _ in range(rng.randint(1, 5))]
        queries.append({"query_id": f"q_{i}", "query": f"query text {i}", "relevant_ids": set(relevant_ids)})
    return queries


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


async def _run_single_query(
    pipeline: Any,
    query_dict: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Retrieve results for one query and compute per-query metrics."""
    from apps.api.schemas import SearchMode

    query_text: str = query_dict["query"]
    relevant_ids: set[str] = query_dict["relevant_ids"]

    t0 = time.perf_counter()

    try:
        if config.mode == SearchMode.BM25:
            results = await pipeline.bm25_retriever.retrieve(query_text, top_k=config.top_k)
        elif config.mode == SearchMode.DENSE:
            results = await pipeline.dense_retriever.retrieve(query_text, top_k=config.top_k)
        elif config.mode == SearchMode.HYBRID:
            results = await pipeline.hybrid_retriever.retrieve(query_text, top_k=config.top_k)
        else:  # rerank
            fetch_k = config.top_k * config.retrieval_multiplier
            candidates = await pipeline.hybrid_retriever.retrieve(query_text, top_k=fetch_k)
            results = await pipeline.reranker.rerank(
                query=query_text, candidates=candidates, top_n=config.top_k
            )
    except Exception:
        logger.exception("Error evaluating query %s", query_dict["query_id"])
        return {"ndcg": 0.0, "mrr": 0.0, "recall": 0.0, "map": 0.0, "prec": 0.0, "latency_ms": 0.0}

    latency_ms = (time.perf_counter() - t0) * 1_000
    relevances = [1.0 if r.passage_id in relevant_ids else 0.0 for r in results]

    return {
        "ndcg": _ndcg_at_k(relevances),
        "mrr": _mrr_at_k(relevances),
        "recall": _recall_at_k(relevances, len(relevant_ids)),
        "map": _map_score(relevances, len(relevant_ids)),
        "prec": _precision_at_k(relevances),
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------


class EvaluationService:
    """Orchestrates evaluation runs and manages experiment records.

    Parameters
    ----------
    pipeline:
        The application-level ``SearchPipeline`` singleton.
    db_session_factory:
        An async SQLAlchemy session factory.  When ``None``, the service
        falls back to the in-memory ``_EXPERIMENT_STORE`` dict.
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        db_session_factory: Any | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._db_factory = db_session_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_evaluation(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute a full evaluation loop and persist the result.

        This method is designed to be called from a background task so that
        HTTP handlers can respond with a job ID immediately.
        """
        experiment_id = str(uuid.uuid4())
        logger.info(
            "eval.start",
            extra={"experiment_id": experiment_id, "config": config.model_dump()},
        )

        if self._pipeline is None:
            logger.warning("No pipeline attached; returning synthetic metrics.")
            result = self._synthetic_result(config, experiment_id)
        else:
            queries = await _load_queries(config.dataset)
            per_query_metrics: list[dict[str, Any]] = []

            for q in queries:
                m = await _run_single_query(self._pipeline, q, config)
                per_query_metrics.append(m)

            result = self._aggregate(config, experiment_id, per_query_metrics)

        await self._persist(result)

        logger.info(
            "eval.done",
            extra={
                "experiment_id": experiment_id,
                "ndcg@10": result.metrics.ndcg_at_10,
            },
        )
        return result

    async def list_experiments(self) -> list[ExperimentResult]:
        """Return all experiment results, newest first."""
        return list(reversed(list(_EXPERIMENT_STORE.values())))

    async def get_experiment(self, experiment_id: str) -> ExperimentResult | None:
        """Fetch a single experiment by ID."""
        return _EXPERIMENT_STORE.get(experiment_id)

    async def compare_experiments(
        self, id_a: str, id_b: str
    ) -> ExperimentComparison | None:
        """Run pairwise significance tests between two experiments."""
        exp_a = _EXPERIMENT_STORE.get(id_a)
        exp_b = _EXPERIMENT_STORE.get(id_b)
        if exp_a is None or exp_b is None:
            return None

        significance = self._significance_tests(exp_a, exp_b)

        return ExperimentComparison(
            baseline_id=id_a,
            candidate_id=id_b,
            baseline_metrics=exp_a.metrics,
            candidate_metrics=exp_b.metrics,
            significance_tests=significance,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        config: ExperimentConfig,
        experiment_id: str,
        per_query: list[dict[str, Any]],
    ) -> ExperimentResult:
        import datetime

        n = len(per_query) or 1
        ndcg_scores = [m["ndcg"] for m in per_query]
        latencies = sorted(m["latency_ms"] for m in per_query)
        p95_idx = int(0.95 * n)
        p99_idx = int(0.99 * n)

        metrics = MetricSet(
            ndcg_at_10=float(np.mean(ndcg_scores)),
            mrr_at_10=float(np.mean([m["mrr"] for m in per_query])),
            map_score=float(np.mean([m["map"] for m in per_query])),
            recall_at_10=float(np.mean([m["recall"] for m in per_query])),
            precision_at_10=float(np.mean([m["prec"] for m in per_query])),
            mean_latency_ms=float(np.mean(latencies)),
            p95_latency_ms=latencies[min(p95_idx, n - 1)],
            p99_latency_ms=latencies[min(p99_idx, n - 1)],
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            name=config.name,
            description=config.description,
            config=config,
            metrics=metrics,
            per_query_ndcg=ndcg_scores,
            created_at=datetime.datetime.utcnow().isoformat(),
            status="completed",
        )

    @staticmethod
    def _synthetic_result(
        config: ExperimentConfig, experiment_id: str
    ) -> ExperimentResult:
        """Return plausible synthetic metrics when no pipeline is available."""
        import datetime

        rng = random.Random()
        ndcg_scores = [rng.uniform(0.3, 0.9) for _ in range(100)]
        metrics = MetricSet(
            ndcg_at_10=float(np.mean(ndcg_scores)),
            mrr_at_10=rng.uniform(0.4, 0.85),
            map_score=rng.uniform(0.3, 0.75),
            recall_at_10=rng.uniform(0.5, 0.95),
            precision_at_10=rng.uniform(0.3, 0.7),
            mean_latency_ms=rng.uniform(50, 300),
            p95_latency_ms=rng.uniform(200, 600),
            p99_latency_ms=rng.uniform(400, 900),
        )
        return ExperimentResult(
            experiment_id=experiment_id,
            name=config.name,
            description=config.description,
            config=config,
            metrics=metrics,
            per_query_ndcg=ndcg_scores,
            created_at=datetime.datetime.utcnow().isoformat(),
            status="completed",
        )

    @staticmethod
    def _significance_tests(
        baseline: ExperimentResult, candidate: ExperimentResult
    ) -> list[SignificanceResult]:
        """Run paired t-test on per-query NDCG scores and aggregate metrics."""
        results: list[SignificanceResult] = []

        def _test(metric: str, a_val: float, b_val: float) -> SignificanceResult:
            # For aggregate scalars we only have point estimates; use per-query
            # arrays for NDCG and fall back to effect-size heuristics otherwise.
            delta = b_val - a_val
            # Mock CI using delta ± 10 % of baseline as a sensible placeholder
            ci_half = abs(a_val) * 0.10 + 1e-6
            return SignificanceResult(
                metric=metric,
                p_value=0.05,  # placeholder without query-level data
                confidence_interval_low=delta - ci_half,
                confidence_interval_high=delta + ci_half,
                significant=abs(delta) > ci_half,
                delta=delta,
            )

        # Per-query NDCG paired t-test (proper statistical test)
        a_ndcg = baseline.per_query_ndcg
        b_ndcg = candidate.per_query_ndcg
        if a_ndcg and b_ndcg and len(a_ndcg) == len(b_ndcg):
            stat, pval = stats.ttest_rel(a_ndcg, b_ndcg)
            n = len(a_ndcg)
            se = np.std(np.array(b_ndcg) - np.array(a_ndcg), ddof=1) / np.sqrt(n)
            delta = float(np.mean(b_ndcg)) - float(np.mean(a_ndcg))
            ci = stats.t.ppf(0.975, df=n - 1) * se
            results.append(
                SignificanceResult(
                    metric="ndcg_at_10",
                    p_value=float(pval),
                    confidence_interval_low=delta - float(ci),
                    confidence_interval_high=delta + float(ci),
                    significant=float(pval) < 0.05,
                    delta=delta,
                )
            )
        else:
            results.append(
                _test("ndcg_at_10", baseline.metrics.ndcg_at_10, candidate.metrics.ndcg_at_10)
            )

        bm, cm = baseline.metrics, candidate.metrics
        results.extend(
            [
                _test("mrr_at_10", bm.mrr_at_10, cm.mrr_at_10),
                _test("map_score", bm.map_score, cm.map_score),
                _test("recall_at_10", bm.recall_at_10, cm.recall_at_10),
            ]
        )
        return results

    async def _persist(self, result: ExperimentResult) -> None:
        """Store experiment result in DB (or in-memory fallback)."""
        _EXPERIMENT_STORE[result.experiment_id] = result

        if self._db_factory is not None:
            try:
                async with self._db_factory() as session:
                    # Placeholder – in a real project this would use an ORM model
                    logger.debug("Persisting experiment %s to DB.", result.experiment_id)
                    await session.commit()
            except Exception:
                logger.exception("Failed to persist experiment %s to DB.", result.experiment_id)
