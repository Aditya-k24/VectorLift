"""
Sample ingestion script for VectorLift.

Downloads MS MARCO sample passages, indexes them into Elasticsearch and Qdrant,
generates embeddings, and stores them in FAISS locally.

Usage:
    python scripts/ingest_sample.py --mode dev --es-host localhost --qdrant-host localhost

Options:
    --mode          dev (1k passages) or small (100k passages)
    --es-host       Elasticsearch host (default: localhost)
    --es-port       Elasticsearch port (default: 9200)
    --qdrant-host   Qdrant host (default: localhost)
    --qdrant-port   Qdrant port (default: 6333)
    --model         Encoder model (default: all-MiniLM-L6-v2)
    --batch-size    Encoding batch size (default: 256)
    --output-dir    Directory for FAISS index and embeddings
    --cache-dir     Dataset cache directory
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(name="ingest-sample", add_completion=False, pretty_exceptions_enable=False)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def main(
    mode: Annotated[str, typer.Option("--mode", help="dev|small")] = "dev",
    es_host: Annotated[str, typer.Option("--es-host")] = "localhost",
    es_port: Annotated[int, typer.Option("--es-port")] = 9200,
    qdrant_host: Annotated[str, typer.Option("--qdrant-host")] = "localhost",
    qdrant_port: Annotated[int, typer.Option("--qdrant-port")] = 6333,
    model: Annotated[str, typer.Option("--model")] = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 256,
    output_dir: Annotated[str, typer.Option("--output-dir")] = "data/faiss",
    cache_dir: Annotated[Optional[str], typer.Option("--cache-dir")] = None,
    index_name: Annotated[str, typer.Option("--index-name")] = "passages",
    collection_name: Annotated[str, typer.Option("--collection-name")] = "passages",
    skip_es: Annotated[bool, typer.Option("--skip-es/--no-skip-es")] = False,
    skip_qdrant: Annotated[bool, typer.Option("--skip-qdrant/--no-skip-qdrant")] = False,
    skip_faiss: Annotated[bool, typer.Option("--skip-faiss/--no-skip-faiss")] = False,
) -> None:
    """Ingest MS MARCO sample passages into ES, Qdrant, and FAISS."""
    _setup_logging()
    asyncio.run(
        _async_main(
            mode=mode,
            es_host=es_host,
            es_port=es_port,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            model=model,
            batch_size=batch_size,
            output_dir=output_dir,
            cache_dir=cache_dir,
            index_name=index_name,
            collection_name=collection_name,
            skip_es=skip_es,
            skip_qdrant=skip_qdrant,
            skip_faiss=skip_faiss,
        )
    )


async def _async_main(
    mode: str,
    es_host: str,
    es_port: int,
    qdrant_host: str,
    qdrant_port: int,
    model: str,
    batch_size: int,
    output_dir: str,
    cache_dir: Optional[str],
    index_name: str,
    collection_name: str,
    skip_es: bool,
    skip_qdrant: bool,
    skip_faiss: bool,
) -> None:
    total_start = time.perf_counter()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading MS MARCO passages (mode=%s) …", mode)
    from pipelines.ingestion.msmarco import MSMARCODataset

    ms = MSMARCODataset(cache_dir=cache_dir)
    t0 = time.perf_counter()
    passages = ms.load_passages(mode)
    load_time = time.perf_counter() - t0
    logger.info("Loaded %d passages in %.1fs.", len(passages), load_time)

    # ------------------------------------------------------------------
    # 2. Generate embeddings
    # ------------------------------------------------------------------
    logger.info("Loading encoder model '%s' …", model)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc

    encoder = SentenceTransformer(model)

    logger.info("Generating embeddings for %d passages …", len(passages))
    emb_start = time.perf_counter()
    texts = [p["text"] for p in passages]
    embeddings: np.ndarray = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    emb_time = time.perf_counter() - emb_start
    dim = embeddings.shape[1]
    logger.info(
        "Embeddings ready: shape=%s  dim=%d  time=%.1fs  %.0f docs/s",
        embeddings.shape, dim, emb_time, len(passages) / max(emb_time, 1e-6),
    )

    # Save raw embeddings for reuse
    emb_path = out_path / f"embeddings_{mode}.npz"
    np.savez_compressed(
        str(emb_path),
        embeddings=embeddings,
        ids=np.array([p["id"] for p in passages], dtype=str),
    )
    logger.info("Embeddings saved to '%s'.", emb_path)

    # ------------------------------------------------------------------
    # 3. FAISS index
    # ------------------------------------------------------------------
    faiss_path = out_path / f"faiss_{mode}.index"
    if not skip_faiss:
        _build_faiss_index(embeddings, [p["id"] for p in passages], faiss_path)
    else:
        logger.info("Skipping FAISS indexing.")

    # ------------------------------------------------------------------
    # 4. Elasticsearch indexing
    # ------------------------------------------------------------------
    if not skip_es:
        await _index_elasticsearch(passages, es_host, es_port, index_name)
    else:
        logger.info("Skipping Elasticsearch indexing.")

    # ------------------------------------------------------------------
    # 5. Qdrant indexing
    # ------------------------------------------------------------------
    if not skip_qdrant:
        await _index_qdrant(passages, embeddings, qdrant_host, qdrant_port, collection_name, dim, batch_size)
    else:
        logger.info("Skipping Qdrant indexing.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - total_start
    summary = {
        "mode": mode,
        "n_passages": len(passages),
        "embedding_dim": dim,
        "embedding_time_s": round(emb_time, 2),
        "total_time_s": round(total_time, 2),
        "throughput_docs_per_s": round(len(passages) / max(total_time, 1e-6), 1),
        "faiss_index": str(faiss_path) if not skip_faiss else "skipped",
        "embeddings_file": str(emb_path),
    }
    summary_path = out_path / f"ingest_summary_{mode}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    typer.echo("\n--- Ingestion Summary ---")
    for k, v in summary.items():
        typer.echo(f"  {k}: {v}")
    typer.echo(f"\nSummary saved to: {summary_path}")


def _build_faiss_index(
    embeddings: np.ndarray,
    ids: List[str],
    output_path: Path,
) -> None:
    """Build an L2-normalised FAISS flat index and save to disk."""
    try:
        import faiss  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install faiss-cpu: pip install faiss-cpu") from exc

    logger.info("Building FAISS flat (IP) index for %d vectors …", len(embeddings))
    faiss_start = time.perf_counter()

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-9)

    dim = normed.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Wrap in IDMap so we can map string IDs via integer handles
    # We store the string-to-int mapping separately as JSON
    id_index = faiss.IndexIDMap(index)

    int_ids = np.arange(len(ids), dtype=np.int64)
    id_index.add_with_ids(normed, int_ids)

    faiss.write_index(id_index, str(output_path))

    id_map_path = output_path.with_suffix(".id_map.json")
    with open(id_map_path, "w") as fh:
        json.dump({str(i): pid for i, pid in enumerate(ids)}, fh)

    faiss_time = time.perf_counter() - faiss_start
    logger.info(
        "FAISS index saved to '%s' (%.1fs). ID map: '%s'.",
        output_path, faiss_time, id_map_path,
    )


async def _index_elasticsearch(
    passages: List[dict],
    host: str,
    port: int,
    index_name: str,
) -> None:
    """Create ES index and bulk-index passages."""
    logger.info("Indexing %d passages into Elasticsearch %s:%d …", len(passages), host, port)
    try:
        from retrieval.bm25.elasticsearch_retriever import ElasticsearchRetriever
    except ImportError:
        logger.error("ElasticsearchRetriever not found; skipping ES indexing.")
        return

    t0 = time.perf_counter()
    async with ElasticsearchRetriever(
        host=host, port=port, index_name=index_name
    ) as es:
        healthy = await es.health_check()
        if not healthy:
            logger.warning("ES not healthy at %s:%d; creating index anyway.", host, port)
        await es.create_index(delete_if_exists=False)
        await es.index_passages(passages, batch_size=500)

    elapsed = time.perf_counter() - t0
    logger.info("ES indexing complete in %.1fs.", elapsed)


async def _index_qdrant(
    passages: List[dict],
    embeddings: np.ndarray,
    host: str,
    port: int,
    collection_name: str,
    dim: int,
    batch_size: int,
) -> None:
    """Create Qdrant collection and upsert all vectors."""
    logger.info("Indexing %d vectors into Qdrant %s:%d …", len(passages), host, port)
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install qdrant-client: pip install qdrant-client") from exc

    t0 = time.perf_counter()
    client = AsyncQdrantClient(host=host, port=port)

    try:
        collections = await client.get_collections()
        existing = [c.name for c in collections.collections]
        if collection_name not in existing:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'.", collection_name)

        from qdrant_client.models import PointStruct  # type: ignore[import]

        for batch_start in range(0, len(passages), batch_size):
            batch_passages = passages[batch_start : batch_start + batch_size]
            batch_embs = embeddings[batch_start : batch_start + batch_size]
            points = [
                PointStruct(
                    id=idx + batch_start,
                    vector=vec.tolist(),
                    payload={"id": p["id"], "text": p["text"], "title": p.get("title", "")},
                )
                for idx, (p, vec) in enumerate(zip(batch_passages, batch_embs))
            ]
            await client.upsert(collection_name=collection_name, points=points)

        elapsed = time.perf_counter() - t0
        logger.info("Qdrant indexing complete in %.1fs.", elapsed)
    finally:
        await client.close()


if __name__ == "__main__":
    app()
