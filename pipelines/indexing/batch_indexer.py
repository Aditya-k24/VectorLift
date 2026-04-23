"""
Batch indexing pipeline for VectorLift.

Indexes a passage corpus into both Elasticsearch (BM25) and Qdrant (dense
vectors) in parallel, with:
  - Progress bars via ``rich``
  - Retry-on-failure with exponential back-off (``tenacity``)
  - Skip-already-indexed logic based on a local manifest file
  - Throughput logging (docs/sec)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocol stubs so BatchIndexer does not depend on concrete
# retriever implementations directly (avoids circular imports).
# ---------------------------------------------------------------------------


class ESRetrieverProtocol(Protocol):
    async def index_passages(
        self, passages: List[Dict[str, Any]], batch_size: int = 500
    ) -> None: ...


class QdrantRetrieverProtocol(Protocol):
    async def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None: ...


class EncoderProtocol(Protocol):
    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress_bar: bool = False,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# BatchIndexer
# ---------------------------------------------------------------------------


class BatchIndexer:
    """Orchestrates bulk indexing into Elasticsearch and/or Qdrant.

    Args:
        es_retriever:     Elasticsearch retriever (``index_passages`` API).
        qdrant_retriever: Qdrant retriever (``upsert_vectors`` API).
        encoder:          Encoder with a ``encode(texts)`` method.
        batch_size:       Number of passages per micro-batch.
        manifest_path:    Path to a JSON manifest tracking indexed passage IDs.
                          Used to skip already-indexed documents on resume.
    """

    def __init__(
        self,
        es_retriever: ESRetrieverProtocol,
        qdrant_retriever: QdrantRetrieverProtocol,
        encoder: EncoderProtocol,
        batch_size: int = 500,
        manifest_path: Optional[str] = None,
    ) -> None:
        self.es_retriever = es_retriever
        self.qdrant_retriever = qdrant_retriever
        self.encoder = encoder
        self.batch_size = batch_size
        self._manifest_path = Path(manifest_path or ".vectorlift_index_manifest.json")
        self._indexed_ids: set[str] = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_corpus(
        self,
        passages: List[Dict[str, Any]],
        index_bm25: bool = True,
        index_dense: bool = True,
    ) -> None:
        """Index the passage corpus into ES and/or Qdrant.

        Already-indexed passage IDs (tracked in the manifest) are skipped
        automatically.

        Args:
            passages:    List of passage dicts with ``id``, ``text``, ``title``.
            index_bm25:  Whether to index into Elasticsearch.
            index_dense: Whether to generate embeddings and index into Qdrant.
        """
        # Filter already-indexed
        new_passages = [p for p in passages if str(p["id"]) not in self._indexed_ids]
        if not new_passages:
            logger.info("All %d passages already indexed; nothing to do.", len(passages))
            return

        logger.info(
            "Indexing %d new passages (skipping %d already indexed).",
            len(new_passages),
            len(passages) - len(new_passages),
        )

        try:
            from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
            _rich_available = True
        except ImportError:
            _rich_available = False

        tasks: List[Any] = []
        if index_bm25:
            tasks.append(self._index_bm25(new_passages, rich_available=_rich_available))
        if index_dense:
            tasks.append(self._index_dense(new_passages, rich_available=_rich_available))

        await asyncio.gather(*tasks)
        self._save_manifest()
        logger.info("Corpus indexing complete.")

    async def generate_and_store_embeddings(
        self,
        passages: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Generate embeddings and persist them to disk as a ``.npz`` file.

        The file contains:
          - ``embeddings``: float32 array of shape (N, D)
          - ``ids``:        string array of passage IDs (length N)

        Args:
            passages:    Source passage list with ``id`` and ``text`` keys.
            output_path: Destination ``.npz`` file path.
        """
        texts = [p["text"] for p in passages]
        ids = [str(p["id"]) for p in passages]

        logger.info("Generating embeddings for %d passages …", len(texts))
        start = time.perf_counter()

        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.encoder.encode(
                texts, batch_size=self.batch_size, show_progress_bar=True
            ),
        )
        elapsed = time.perf_counter() - start
        throughput = len(texts) / max(elapsed, 1e-6)

        logger.info(
            "Generated %d embeddings in %.1fs (%.0f docs/s).",
            len(texts), elapsed, throughput,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out),
            embeddings=embeddings.astype(np.float32),
            ids=np.array(ids, dtype=str),
        )
        logger.info("Saved embeddings to '%s'.", out)

    # ------------------------------------------------------------------
    # Private: BM25 indexing
    # ------------------------------------------------------------------

    async def _index_bm25(
        self, passages: List[Dict[str, Any]], rich_available: bool = True
    ) -> None:
        """Bulk-index passages into Elasticsearch in batches."""
        total = len(passages)
        start_time = time.perf_counter()
        indexed_count = 0

        _progress_ctx: Any = None
        _task_id: Any = None

        if rich_available:
            from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
            _progress_ctx = Progress(
                TextColumn("[bold blue]BM25 indexing"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            _progress_ctx.start()
            _task_id = _progress_ctx.add_task("indexing", total=total)

        try:
            for batch_start in range(0, total, self.batch_size):
                batch = passages[batch_start : batch_start + self.batch_size]
                await self._index_bm25_batch(batch)
                indexed_count += len(batch)

                # Track IDs in manifest
                for p in batch:
                    self._indexed_ids.add(str(p["id"]))

                if _progress_ctx and _task_id is not None:
                    _progress_ctx.update(_task_id, completed=indexed_count)

                elapsed = time.perf_counter() - start_time
                throughput = indexed_count / max(elapsed, 1e-6)
                logger.debug(
                    "BM25: %d/%d indexed  (%.0f docs/s)", indexed_count, total, throughput
                )
        finally:
            if _progress_ctx:
                _progress_ctx.stop()

        elapsed = time.perf_counter() - start_time
        logger.info(
            "BM25 indexing complete: %d passages in %.1fs (%.0f docs/s).",
            indexed_count, elapsed, indexed_count / max(elapsed, 1e-6),
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _index_bm25_batch(self, batch: List[Dict[str, Any]]) -> None:
        await self.es_retriever.index_passages(batch, batch_size=len(batch))

    # ------------------------------------------------------------------
    # Private: Dense indexing
    # ------------------------------------------------------------------

    async def _index_dense(
        self, passages: List[Dict[str, Any]], rich_available: bool = True
    ) -> None:
        """Generate embeddings and upsert into Qdrant in batches."""
        total = len(passages)
        start_time = time.perf_counter()
        indexed_count = 0

        _progress_ctx: Any = None
        _task_id: Any = None

        if rich_available:
            from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
            _progress_ctx = Progress(
                TextColumn("[bold green]Dense indexing"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            _progress_ctx.start()
            _task_id = _progress_ctx.add_task("embedding", total=total)

        try:
            for batch_start in range(0, total, self.batch_size):
                batch = passages[batch_start : batch_start + self.batch_size]
                texts = [p["text"] for p in batch]
                ids = [str(p["id"]) for p in batch]
                payloads = [
                    {"text": p["text"], "title": p.get("title", "")} for p in batch
                ]

                # Encode in thread pool to avoid blocking the event loop
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda t=texts: self.encoder.encode(
                        t, batch_size=self.batch_size, show_progress_bar=False
                    ),
                )

                vectors = embeddings.tolist()
                await self._upsert_qdrant_batch(ids, vectors, payloads)
                indexed_count += len(batch)

                if _progress_ctx and _task_id is not None:
                    _progress_ctx.update(_task_id, completed=indexed_count)

                elapsed = time.perf_counter() - start_time
                throughput = indexed_count / max(elapsed, 1e-6)
                logger.debug(
                    "Dense: %d/%d indexed  (%.0f docs/s)", indexed_count, total, throughput
                )
        finally:
            if _progress_ctx:
                _progress_ctx.stop()

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Dense indexing complete: %d passages in %.1fs (%.0f docs/s).",
            indexed_count, elapsed, indexed_count / max(elapsed, 1e-6),
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _upsert_qdrant_batch(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        await self.qdrant_retriever.upsert_vectors(ids, vectors, payloads)

    # ------------------------------------------------------------------
    # Manifest management
    # ------------------------------------------------------------------

    def _load_manifest(self) -> set[str]:
        if self._manifest_path.exists():
            with open(self._manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
            ids: set[str] = set(data.get("indexed_ids", []))
            logger.info("Manifest loaded: %d previously indexed IDs.", len(ids))
            return ids
        return set()

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"indexed_ids": sorted(self._indexed_ids)},
                fh,
                indent=2,
            )
        logger.debug(
            "Manifest saved with %d IDs to '%s'.",
            len(self._indexed_ids),
            self._manifest_path,
        )
