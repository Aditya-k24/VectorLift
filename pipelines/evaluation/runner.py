"""
Experiment runner for VectorLift evaluation pipeline.

Orchestrates retrieval + (optional) reranking across multiple system
configurations, computes retrieval metrics, persists results to PostgreSQL,
and emits markdown reports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.interfaces.base import BaseReranker, BaseRetriever

from pipelines.evaluation.metrics import compute_metrics, per_query_metrics
from pipelines.evaluation.significance import SignificanceTestResult, compare_all_systems

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes:
        experiment_id:    Auto-generated UUID4 if not supplied.
        name:             Human-readable experiment name.
        retrieval_mode:   One of 'bm25', 'dense', 'hybrid'.
        use_reranker:     Whether to apply the reranker on top of retrieval.
        top_k_retrieval:  Number of candidates fetched by the first-stage retriever.
        top_n_rerank:     Number of results after reranking (ignored if use_reranker=False).
        k_values:         Cutoffs for metrics evaluation.
        output_dir:       Where to persist result artifacts.
        tags:             Arbitrary key-value metadata (model names, hyperparams, …).
    """

    name: str
    retrieval_mode: str = "dense"  # 'bm25' | 'dense' | 'hybrid'
    use_reranker: bool = False
    top_k_retrieval: int = 100
    top_n_rerank: int = 10
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 100])
    output_dir: str = "experiments"
    tags: Dict[str, str] = field(default_factory=dict)
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        valid_modes = {"bm25", "dense", "hybrid"}
        if self.retrieval_mode not in valid_modes:
            raise ValueError(
                f"retrieval_mode must be one of {valid_modes}, got '{self.retrieval_mode}'"
            )


@dataclass
class ExperimentResult:
    """Holds all outputs from a completed experiment.

    Attributes:
        config:          The configuration used.
        metrics:         Aggregate metrics dict (ndcg@k, map, …).
        per_query:       Per-query breakdown (query_id -> {ndcg, rr, ap}).
        results:         Raw system results (query_id -> [doc_id, …]).
        elapsed_seconds: Total wall-clock time.
        timestamp:       UTC ISO-8601 timestamp.
        artifact_path:   Path to the saved JSON artifact.
    """

    config: ExperimentConfig
    metrics: Dict[str, float]
    per_query: Dict[str, Dict[str, float]]
    results: Dict[str, List[str]]
    elapsed_seconds: float
    timestamp: str
    artifact_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.config.experiment_id,
            "name": self.config.name,
            "config": asdict(self.config),
            "metrics": self.metrics,
            "per_query": self.per_query,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": self.timestamp,
            "artifact_path": self.artifact_path,
        }


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Orchestrates retrieval evaluation for a single system configuration.

    Args:
        retriever: First-stage retriever (BM25, dense, or hybrid).
        reranker:  Optional second-stage reranker.
        qrels:     Ground-truth relevance judgements –
                   query_id -> {doc_id: relevance_grade}.
        queries:   Query texts – query_id -> query_text.
        config:    Experiment configuration.
        db_session: Optional SQLAlchemy async session for persisting results.
                    If ``None``, DB persistence is skipped.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        qrels: Dict[str, Dict[str, int]],
        queries: Dict[str, str],
        config: ExperimentConfig,
        reranker: Optional[BaseReranker] = None,
        db_session: Optional[Any] = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.qrels = qrels
        self.queries = queries
        self.config = config
        self.db_session = db_session

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> ExperimentResult:
        """Execute the full evaluation pipeline.

        Steps:
          1. Retrieve candidates for every query.
          2. Optionally rerank.
          3. Compute aggregate metrics.
          4. Store results in DB (if session available).
          5. Save JSON artifact.

        Returns:
            :class:`ExperimentResult` with all outputs.
        """
        logger.info(
            "Starting experiment '%s' [%s] (reranker=%s)",
            self.config.name,
            self.config.experiment_id,
            self.config.use_reranker,
        )
        start = time.perf_counter()

        # Step 1 + 2: Retrieve (and optionally rerank)
        raw_results = await self._retrieve_all()

        # Step 3: Metrics
        metrics = compute_metrics(self.qrels, raw_results, k_values=self.config.k_values)
        pq_metrics = per_query_metrics(self.qrels, raw_results, k=10)

        elapsed = time.perf_counter() - start
        timestamp = datetime.now(timezone.utc).isoformat()

        result = ExperimentResult(
            config=self.config,
            metrics=metrics,
            per_query=pq_metrics,
            results=raw_results,
            elapsed_seconds=elapsed,
            timestamp=timestamp,
        )

        # Step 4: Persist to DB
        if self.db_session is not None:
            await self._store_in_db(result)

        # Step 5: Save artifact
        artifact_path = self._save_artifact(result)
        result.artifact_path = artifact_path

        logger.info(
            "Experiment '%s' finished in %.2fs. NDCG@10=%.4f  MAP=%.4f",
            self.config.name,
            elapsed,
            metrics.get("ndcg@10", float("nan")),
            metrics.get("map", float("nan")),
        )
        return result

    async def run_all_modes(self) -> List[ExperimentResult]:
        """Run all six standard evaluation configurations.

        Configurations:
          1. BM25 only
          2. Dense only
          3. Hybrid only
          4. BM25 + reranker
          5. Dense + reranker
          6. Hybrid + reranker

        Note: Configurations 4-6 are skipped if no reranker is provided.

        Returns:
            List of :class:`ExperimentResult`, one per configuration.
        """
        modes = ["bm25", "dense", "hybrid"]
        results: List[ExperimentResult] = []

        for mode in modes:
            for use_rerank in [False, True]:
                if use_rerank and self.reranker is None:
                    logger.warning(
                        "No reranker provided; skipping '%s+rerank' configuration.", mode
                    )
                    continue

                cfg = ExperimentConfig(
                    name=f"{mode}{'_rerank' if use_rerank else ''}",
                    retrieval_mode=mode,
                    use_reranker=use_rerank,
                    top_k_retrieval=self.config.top_k_retrieval,
                    top_n_rerank=self.config.top_n_rerank,
                    k_values=self.config.k_values,
                    output_dir=self.config.output_dir,
                    tags=self.config.tags,
                )

                # Clone runner for this config
                runner = ExperimentRunner(
                    retriever=self.retriever,
                    reranker=self.reranker,
                    qrels=self.qrels,
                    queries=self.queries,
                    config=cfg,
                    db_session=self.db_session,
                )
                try:
                    exp_result = await runner.run()
                    results.append(exp_result)
                except Exception as exc:
                    logger.error("Experiment '%s' failed: %s", cfg.name, exc, exc_info=True)

        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: List[ExperimentResult]) -> str:
        """Generate a markdown report comparing all experiment results.

        Args:
            results: List of completed :class:`ExperimentResult`.

        Returns:
            Markdown string ready to be written to a ``.md`` file.
        """
        if not results:
            return "# VectorLift Evaluation Report\n\n*No results to report.*\n"

        lines: List[str] = [
            "# VectorLift Evaluation Report",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## System Overview",
            "",
            f"Total configurations evaluated: **{len(results)}**",
            "",
            "## Aggregate Metrics",
            "",
        ]

        # Collect all metric keys
        all_metric_keys: List[str] = []
        for r in results:
            for k in r.metrics:
                if k not in all_metric_keys:
                    all_metric_keys.append(k)

        # Prioritise display order
        priority = ["ndcg@10", "ndcg@100", "mrr@10", "map", "recall@10", "precision@10"]
        ordered_keys = [k for k in priority if k in all_metric_keys]
        ordered_keys += [k for k in all_metric_keys if k not in ordered_keys]

        # Header row
        col_header = "| System | " + " | ".join(ordered_keys) + " | Time (s) |"
        col_sep = "|--------|" + "|".join(["--------"] * len(ordered_keys)) + "|----------|"
        lines.extend([col_header, col_sep])

        for r in results:
            vals = [f"{r.metrics.get(k, float('nan')):.4f}" for k in ordered_keys]
            row = f"| {r.config.name} | " + " | ".join(vals) + f" | {r.elapsed_seconds:.1f} |"
            lines.append(row)

        lines.extend(["", "## Per-Experiment Details", ""])

        for r in results:
            lines.extend([
                f"### {r.config.name}",
                "",
                f"- **Experiment ID**: `{r.config.experiment_id}`",
                f"- **Retrieval mode**: {r.config.retrieval_mode}",
                f"- **Reranker**: {'Yes' if r.config.use_reranker else 'No'}",
                f"- **Timestamp**: {r.timestamp}",
                f"- **Elapsed**: {r.elapsed_seconds:.2f}s",
                f"- **Artifact**: `{r.artifact_path or 'N/A'}`",
                "",
            ])
            for mk, mv in sorted(r.metrics.items()):
                lines.append(f"  - {mk}: **{mv:.4f}**")
            lines.append("")

        # Significance testing between all pairs (NDCG@10)
        if len(results) >= 2:
            lines.extend(["## Statistical Significance (NDCG@10, paired bootstrap)", ""])
            systems: Dict[str, Dict[str, float]] = {
                r.config.name: {qid: v["ndcg"] for qid, v in r.per_query.items()}
                for r in results
            }
            try:
                sig_results: List[SignificanceTestResult] = compare_all_systems(
                    systems, metric_name="ndcg@10"
                )
                if sig_results:
                    lines.extend([
                        "| System A | System B | Delta | p-value | CI | Significant |",
                        "|----------|----------|-------|---------|-----|-------------|",
                    ])
                    for sr in sig_results:
                        sig_str = "**Yes**" if sr.is_significant else "No"
                        lines.append(
                            f"| {sr.system_a} | {sr.system_b} | "
                            f"{sr.delta:+.4f} | {sr.p_value:.4f} | "
                            f"[{sr.ci_lower:.4f}, {sr.ci_upper:.4f}] | {sig_str} |"
                        )
                    lines.append("")
            except Exception as exc:
                logger.warning("Significance testing failed: %s", exc)
                lines.append(f"*Significance testing unavailable: {exc}*\n")

        lines.extend([
            "---",
            f"*VectorLift Evaluation Framework — {len(results)} systems compared.*",
            "",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _retrieve_all(self) -> Dict[str, List[str]]:
        """Run retrieval (and optional reranking) for all queries."""
        raw_results: Dict[str, List[str]] = {}

        # Run retrieval tasks concurrently, but with a semaphore to avoid
        # overwhelming the backend.
        semaphore = asyncio.Semaphore(20)

        async def _process_query(qid: str, query_text: str) -> None:
            async with semaphore:
                try:
                    candidates = await self.retriever.retrieve(
                        query_text, top_k=self.config.top_k_retrieval
                    )

                    if self.config.use_reranker and self.reranker is not None:
                        candidates = await self.reranker.rerank(
                            query_text, candidates, top_n=self.config.top_n_rerank
                        )

                    raw_results[qid] = [c.passage_id for c in candidates]
                except Exception as exc:
                    logger.error("Query '%s' failed: %s", qid, exc)
                    raw_results[qid] = []

        tasks = [
            asyncio.create_task(_process_query(qid, text))
            for qid, text in self.queries.items()
        ]
        await asyncio.gather(*tasks)
        return raw_results

    async def _store_in_db(self, result: ExperimentResult) -> None:
        """Persist experiment result to PostgreSQL via SQLAlchemy session."""
        try:
            await self.db_session.execute(
                # Raw SQL – production code should use ORM models instead
                "INSERT INTO experiment_results "
                "(experiment_id, name, config, metrics, elapsed_seconds, timestamp) "
                "VALUES (:eid, :name, :cfg, :metrics, :elapsed, :ts)",
                {
                    "eid": result.config.experiment_id,
                    "name": result.config.name,
                    "cfg": json.dumps(asdict(result.config)),
                    "metrics": json.dumps(result.metrics),
                    "elapsed": result.elapsed_seconds,
                    "ts": result.timestamp,
                },
            )
            await self.db_session.commit()
            logger.debug("Persisted experiment '%s' to DB.", result.config.experiment_id)
        except Exception as exc:
            logger.error("DB persistence failed for experiment '%s': %s", result.config.name, exc)

    def _save_artifact(self, result: ExperimentResult) -> str:
        """Write experiment result to a JSON file and return its path."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{result.config.name}_{result.config.experiment_id[:8]}.json"
        path = out_dir / fname

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)

        logger.debug("Saved experiment artifact to '%s'.", path)
        return str(path)
