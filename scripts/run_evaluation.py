"""
Evaluation runner script for VectorLift.

Loads MS MARCO qrels + queries, runs all six retrieval system configurations,
computes retrieval metrics with significance testing, and saves results to
JSON + a markdown report.

Usage:
    python scripts/run_evaluation.py \\
        --mode dev \\
        --top-k 100 \\
        --output-dir experiments/eval_$(date +%Y%m%d)

Options:
    --mode              Dataset mode: dev|small
    --top-k             Number of candidates per retrieval call
    --output-dir        Output directory for JSON + report
    --es-host           Elasticsearch host
    --es-port           Elasticsearch port
    --qdrant-host       Qdrant host
    --qdrant-port       Qdrant port
    --encoder-model     Bi-encoder model for dense retrieval
    --reranker-model    Cross-encoder model for reranking
    --skip-reranker     Skip the three +rerank configurations
    --cache-dir         Dataset cache directory
    --seed              Random seed
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="run-evaluation", add_completion=False, pretty_exceptions_enable=False)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def main(
    mode: Annotated[str, typer.Option("--mode", help="dev|small")] = "dev",
    top_k: Annotated[int, typer.Option("--top-k", help="Retrieval depth")] = 100,
    output_dir: Annotated[str, typer.Option("--output-dir")] = "experiments/eval",
    es_host: Annotated[str, typer.Option("--es-host")] = "localhost",
    es_port: Annotated[int, typer.Option("--es-port")] = 9200,
    qdrant_host: Annotated[str, typer.Option("--qdrant-host")] = "localhost",
    qdrant_port: Annotated[int, typer.Option("--qdrant-port")] = 6333,
    encoder_model: Annotated[str, typer.Option("--encoder-model")] = "sentence-transformers/all-MiniLM-L6-v2",
    reranker_model: Annotated[str, typer.Option("--reranker-model")] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    skip_reranker: Annotated[bool, typer.Option("--skip-reranker/--no-skip-reranker")] = False,
    cache_dir: Annotated[Optional[str], typer.Option("--cache-dir")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    n_queries: Annotated[int, typer.Option("--n-queries", help="Limit query count (0=all)")] = 0,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Run all six evaluation configurations and produce a report."""
    _setup_logging(log_level)
    asyncio.run(
        _async_main(
            mode=mode,
            top_k=top_k,
            output_dir=output_dir,
            es_host=es_host,
            es_port=es_port,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            encoder_model=encoder_model,
            reranker_model=reranker_model,
            skip_reranker=skip_reranker,
            cache_dir=cache_dir,
            seed=seed,
            n_queries=n_queries if n_queries > 0 else None,
        )
    )


async def _async_main(
    mode: str,
    top_k: int,
    output_dir: str,
    es_host: str,
    es_port: int,
    qdrant_host: str,
    qdrant_port: int,
    encoder_model: str,
    reranker_model: str,
    skip_reranker: bool,
    cache_dir: Optional[str],
    seed: int,
    n_queries: Optional[int],
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    console.rule(f"[bold blue]VectorLift Evaluation Run {run_id}")

    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    console.print("[cyan]Loading MS MARCO data …")
    from pipelines.ingestion.msmarco import MSMARCODataset

    ms = MSMARCODataset(cache_dir=cache_dir, seed=seed)
    qrels = ms.load_qrels("dev")
    queries_all = ms.load_queries("dev")

    # Only keep queries present in qrels
    queries: Dict[str, str] = {
        qid: text for qid, text in queries_all.items() if qid in qrels
    }

    if n_queries is not None:
        query_ids = sorted(queries.keys())[:n_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids}

    console.print(f"  Queries: {len(queries)}  |  Qrels: {len(qrels)}")

    # ------------------------------------------------------------------
    # 2. Build retrievers
    # ------------------------------------------------------------------
    console.print("[cyan]Initialising retrievers …")

    retrievers = await _build_retrievers(
        es_host=es_host,
        es_port=es_port,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        encoder_model=encoder_model,
        mode=mode,
        cache_dir=cache_dir,
    )
    reranker = None
    if not skip_reranker:
        reranker = _build_reranker(reranker_model)

    # ------------------------------------------------------------------
    # 3. Run experiments
    # ------------------------------------------------------------------
    from pipelines.evaluation.runner import ExperimentConfig, ExperimentResult, ExperimentRunner

    console.print("[cyan]Running experiments …")

    all_results: List[ExperimentResult] = []

    modes = ["bm25", "dense", "hybrid"]
    use_reranks = [False] + ([] if skip_reranker or reranker is None else [True])

    for ret_mode in modes:
        retriever = retrievers.get(ret_mode)
        if retriever is None:
            console.print(f"  [yellow]Retriever '{ret_mode}' unavailable; skipping.")
            continue

        for use_rerank in use_reranks:
            if use_rerank and reranker is None:
                continue

            config_name = f"{ret_mode}{'_rerank' if use_rerank else ''}"
            console.print(f"  Running [bold]{config_name}[/bold] …")

            cfg = ExperimentConfig(
                name=config_name,
                retrieval_mode=ret_mode,
                use_reranker=use_rerank,
                top_k_retrieval=top_k,
                top_n_rerank=10,
                k_values=[1, 3, 5, 10, 100],
                output_dir=str(out_path / "artifacts"),
            )
            runner = ExperimentRunner(
                retriever=retriever,
                reranker=reranker if use_rerank else None,
                qrels=qrels,
                queries=queries,
                config=cfg,
            )
            try:
                result = await runner.run()
                all_results.append(result)
                ndcg = result.metrics.get("ndcg@10", float("nan"))
                console.print(f"    NDCG@10={ndcg:.4f}")
            except Exception as exc:
                logger.error("Experiment '%s' failed: %s", config_name, exc, exc_info=True)
                console.print(f"    [red]FAILED: {exc}")

    # ------------------------------------------------------------------
    # 4. Save results JSON
    # ------------------------------------------------------------------
    results_json_path = out_path / f"results_{run_id}.json"
    all_results_dict = [r.to_dict() for r in all_results]
    with open(results_json_path, "w") as fh:
        json.dump(
            {
                "run_id": run_id,
                "mode": mode,
                "n_queries": len(queries),
                "top_k": top_k,
                "results": all_results_dict,
            },
            fh,
            indent=2,
        )
    console.print(f"\n[green]Results JSON saved to: {results_json_path}")

    # ------------------------------------------------------------------
    # 5. Generate markdown report
    # ------------------------------------------------------------------
    runner_for_report = ExperimentRunner(
        retriever=list(retrievers.values())[0],
        reranker=reranker,
        qrels=qrels,
        queries=queries,
        config=ExperimentConfig(name="report", output_dir=str(out_path)),
    )
    report_md = runner_for_report.generate_report(all_results)
    report_path = out_path / f"report_{run_id}.md"
    with open(report_path, "w") as fh:
        fh.write(report_md)
    console.print(f"[green]Markdown report saved to: {report_path}")

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    _print_summary_table(all_results)

    total_time = time.perf_counter() - total_start
    console.print(f"\n[bold green]Evaluation complete in {total_time:.1f}s")


def _print_summary_table(results: List["ExperimentResult"]) -> None:
    """Render a rich summary table to the console."""
    if not results:
        console.print("[yellow]No results to display.")
        return

    table = Table(title="Evaluation Summary", show_lines=True)
    table.add_column("System", style="bold cyan")

    metric_cols = ["ndcg@10", "mrr@10", "map", "recall@10", "precision@10"]
    for col in metric_cols:
        table.add_column(col, justify="right")
    table.add_column("Time (s)", justify="right")

    for r in results:
        row = [r.config.name]
        for col in metric_cols:
            val = r.metrics.get(col, float("nan"))
            row.append(f"{val:.4f}")
        row.append(f"{r.elapsed_seconds:.1f}")
        table.add_row(*row)

    console.print(table)


async def _build_retrievers(
    es_host: str,
    es_port: int,
    qdrant_host: str,
    qdrant_port: int,
    encoder_model: str,
    mode: str,
    cache_dir: Optional[str],
) -> Dict[str, "BaseRetriever"]:
    """Construct and health-check all three retriever backends."""
    from retrieval.interfaces.base import BaseRetriever

    retrievers: Dict[str, BaseRetriever] = {}

    # BM25
    try:
        from retrieval.bm25.elasticsearch_retriever import ElasticsearchRetriever

        es = ElasticsearchRetriever(host=es_host, port=es_port)
        if await es.health_check():
            retrievers["bm25"] = es
            logger.info("BM25 retriever ready.")
        else:
            logger.warning("Elasticsearch not healthy; BM25 retriever unavailable.")
    except Exception as exc:
        logger.warning("Could not build BM25 retriever: %s", exc)

    # Dense + Hybrid (require encoder)
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(encoder_model)
        dense = _build_dense_retriever(encoder, qdrant_host, qdrant_port)
        if dense is not None:
            retrievers["dense"] = dense
            logger.info("Dense retriever ready.")

        bm25_ret = retrievers.get("bm25")
        if dense is not None and bm25_ret is not None:
            hybrid = _build_hybrid_retriever(bm25_ret, dense)
            if hybrid is not None:
                retrievers["hybrid"] = hybrid
                logger.info("Hybrid retriever ready.")
    except Exception as exc:
        logger.warning("Dense/hybrid retriever setup failed: %s", exc)

    return retrievers


def _build_dense_retriever(encoder: Any, qdrant_host: str, qdrant_port: int) -> Optional[Any]:
    """Build a DenseRetriever if the retrieval.dense module exists."""
    try:
        from retrieval.dense import DenseRetriever  # type: ignore[import]
        return DenseRetriever(encoder=encoder, host=qdrant_host, port=qdrant_port)
    except ImportError:
        logger.warning("retrieval.dense.DenseRetriever not found; dense retrieval unavailable.")
        return None
    except Exception as exc:
        logger.warning("DenseRetriever init failed: %s", exc)
        return None


def _build_hybrid_retriever(bm25: Any, dense: Any) -> Optional[Any]:
    """Build a HybridRetriever if the retrieval.hybrid module exists."""
    try:
        from retrieval.hybrid import HybridRetriever  # type: ignore[import]
        return HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    except ImportError:
        logger.warning("retrieval.hybrid.HybridRetriever not found; hybrid retrieval unavailable.")
        return None
    except Exception as exc:
        logger.warning("HybridRetriever init failed: %s", exc)
        return None


def _build_reranker(model_name: str) -> Optional[Any]:
    """Build a CrossEncoderReranker if the retrieval.reranker module exists."""
    try:
        from retrieval.reranker import CrossEncoderReranker  # type: ignore[import]
        return CrossEncoderReranker(model=model_name)
    except ImportError:
        logger.warning("retrieval.reranker.CrossEncoderReranker not found; reranker unavailable.")
        return None
    except Exception as exc:
        logger.warning("CrossEncoderReranker init failed: %s", exc)
        return None


if __name__ == "__main__":
    app()
