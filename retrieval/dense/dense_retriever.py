"""
High-level dense retriever that wires together a BiEncoder and a vector index.

Supports two index backends:
  - :class:`~retrieval.dense.faiss_index.FAISSIndex` – local in-process FAISS
  - :class:`~retrieval.dense.qdrant_retriever.QdrantRetriever` – remote Qdrant

Both backends share the same public :class:`DenseRetriever` API so the rest
of the pipeline is backend-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from retrieval.dense.encoder import BiEncoder
from retrieval.dense.faiss_index import FAISSIndex
from retrieval.dense.qdrant_retriever import QdrantRetriever
from retrieval.interfaces.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# Type alias for supported vector backends
VectorBackend = Union[FAISSIndex, QdrantRetriever]


class DenseRetriever(BaseRetriever):
    """Dense retriever combining a :class:`BiEncoder` with a vector index.

    When the backend is a :class:`FAISSIndex`, this class owns the full
    retrieval logic (encode → search → resolve IDs from passage_store).
    When the backend is a :class:`QdrantRetriever`, the Qdrant client
    handles both storage and search; this class delegates to it.

    Args:
        encoder:       Bi-encoder used to embed queries and passages.
        index:         Vector index backend (FAISS or Qdrant).
        passage_store: Mapping from ``passage_id`` → ``{text, title, metadata}``.
                       Required only when using FAISS (Qdrant stores payloads
                       internally).
    """

    def __init__(
        self,
        encoder: BiEncoder,
        index: VectorBackend,
        passage_store: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.passage_store: Dict[str, Dict[str, Any]] = passage_store or {}

        self._backend: str = (
            "qdrant" if isinstance(index, QdrantRetriever) else "faiss"
        )
        logger.info(
            "DenseRetriever initialised with backend='%s', "
            "encoder='%s', passage_store_size=%d.",
            self._backend,
            encoder.model_name_or_path,
            len(self.passage_store),
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, passages: List[Dict[str, Any]]) -> None:
        """Encode and index *passages* using the configured backend.

        Each passage dict must contain ``id``, ``text``, and ``title``.
        Additional keys are stored as metadata.

        For FAISS:  builds / updates the in-process index and the passage_store.
        For Qdrant: delegates to :meth:`QdrantRetriever.upsert_passages`.

        Args:
            passages: List of passage dicts.
        """
        if not passages:
            logger.warning("DenseRetriever.index called with empty passage list.")
            return

        if self._backend == "qdrant":
            await self.index.upsert_passages(passages)  # type: ignore[union-attr]
        else:
            await self._faiss_index(passages)

    async def _faiss_index(self, passages: List[Dict[str, Any]]) -> None:
        """Internal helper: encode passages and populate the FAISS index."""
        texts = [p.get("text", "") for p in passages]
        ids = [str(p["id"]) for p in passages]

        logger.info("Encoding %d passages for FAISS indexing…", len(passages))
        embeddings = self.encoder.encode_passages(texts, show_progress=True)

        faiss_index: FAISSIndex = self.index  # type: ignore[assignment]
        faiss_index.build(embeddings, ids)

        # Populate passage_store
        for p in passages:
            pid = str(p["id"])
            self.passage_store[pid] = {
                "text": p.get("text", ""),
                "title": p.get("title", ""),
                "metadata": {
                    k: v for k, v in p.items() if k not in {"id", "text", "title"}
                },
            }

        logger.info(
            "FAISS index built with %d vectors; passage_store has %d entries.",
            faiss_index.get_index_size(),
            len(self.passage_store),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Encode *query* and return the top-k nearest passages.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results.

        Returns:
            List of :class:`~retrieval.interfaces.base.RetrievalResult`
            sorted by descending similarity score.
        """
        if not query or not query.strip():
            logger.warning("Empty query received; returning empty results.")
            return []

        if self._backend == "qdrant":
            return await self.index.retrieve(query, top_k=top_k)  # type: ignore[union-attr]

        return await self._faiss_retrieve(query, top_k)

    async def _faiss_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Internal helper: encode query and search FAISS index."""
        faiss_index: FAISSIndex = self.index  # type: ignore[assignment]

        if faiss_index.get_index_size() == 0:
            logger.warning("FAISS index is empty; returning empty results.")
            return []

        query_vec = self.encoder.encode_query(query)
        ids, scores = faiss_index.search(query_vec, top_k)

        results: List[RetrievalResult] = []
        for rank, (pid, score) in enumerate(zip(ids, scores), start=1):
            stored = self.passage_store.get(pid, {})
            results.append(
                RetrievalResult(
                    passage_id=pid,
                    text=stored.get("text", ""),
                    title=stored.get("title", ""),
                    score=float(score),
                    rank=rank,
                    metadata=stored.get("metadata", {}),
                )
            )

        logger.debug(
            "FAISS retrieved %d results for query '%s'.", len(results), query[:80]
        )
        return results

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if the backend is ready to serve queries."""
        if self._backend == "qdrant":
            return await self.index.health_check()  # type: ignore[union-attr]

        faiss_index: FAISSIndex = self.index  # type: ignore[assignment]
        ready = faiss_index.get_index_size() > 0
        if not ready:
            logger.warning("FAISS index is empty – health check failed.")
        return ready

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of indexed passages."""
        if self._backend == "faiss":
            return self.index.get_index_size()  # type: ignore[union-attr]
        return len(self.passage_store)

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(backend='{self._backend}', "
            f"encoder='{self.encoder.model_name_or_path}', "
            f"corpus_size={self.corpus_size})"
        )
