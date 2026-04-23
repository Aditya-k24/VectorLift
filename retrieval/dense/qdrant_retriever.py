"""
Production Qdrant vector retriever for VectorLift.

Uses the official async Qdrant client.  Each passage is stored as a
Qdrant point whose payload mirrors the MSMARCO schema:
  {passage_id, text, title, metadata}

Requires:
    qdrant-client>=1.7
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from retrieval.dense.encoder import BiEncoder
from retrieval.interfaces.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

_DISTANCE_MAP: Dict[str, qmodels.Distance] = {
    "Cosine": qmodels.Distance.COSINE,
    "Dot": qmodels.Distance.DOT,
    "Euclid": qmodels.Distance.EUCLID,
}


class QdrantRetriever(BaseRetriever):
    """Dense retriever backed by a Qdrant vector database.

    Args:
        host:             Qdrant server hostname (default ``"localhost"``).
        port:             Qdrant gRPC port (default ``6333``).
        collection_name:  Name of the Qdrant collection.
        encoder:          A :class:`~retrieval.dense.encoder.BiEncoder` that
                          produces passage / query embeddings.
        grpc_port:        gRPC port (default 6334).  ``None`` disables gRPC.
        prefer_grpc:      Use gRPC for upserts / searches (faster).
        api_key:          Optional Qdrant Cloud API key.
        https:            Use TLS (default ``False`` for local).
        timeout:          Request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "passages",
        encoder: Optional[BiEncoder] = None,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None,
        https: bool = False,
        timeout: int = 30,
    ) -> None:
        self.collection_name = collection_name
        self.encoder = encoder or BiEncoder()

        self._client = AsyncQdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https,
            timeout=timeout,
        )

        logger.info(
            "QdrantRetriever initialised for collection '%s' at %s:%d.",
            collection_name,
            host,
            port,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collection(
        self,
        dim: Optional[int] = None,
        distance: str = "Cosine",
        *,
        delete_if_exists: bool = False,
    ) -> None:
        """Create the Qdrant collection with the appropriate vector config.

        Args:
            dim:              Embedding dimension.  Falls back to
                              ``encoder.embedding_dim`` when ``None``.
            distance:         Distance metric – ``"Cosine"``, ``"Dot"``,
                              or ``"Euclid"``.
            delete_if_exists: Drop and recreate any existing collection.
        """
        vector_dim = dim or self.encoder.embedding_dim
        qdrant_distance = _DISTANCE_MAP.get(distance, qmodels.Distance.COSINE)

        if delete_if_exists:
            try:
                await self._client.delete_collection(self.collection_name)
                logger.warning("Deleted existing collection '%s'.", self.collection_name)
            except Exception:
                pass  # collection may not exist yet

        try:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_dim,
                    distance=qdrant_distance,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10_000,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, distance=%s).",
                self.collection_name,
                vector_dim,
                distance,
            )
        except UnexpectedResponse as exc:
            if "already exists" in str(exc).lower():
                logger.info(
                    "Collection '%s' already exists – skipping creation.",
                    self.collection_name,
                )
            else:
                raise

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, passages: List[Dict[str, Any]]) -> None:
        """Index passages (alias for :meth:`upsert_passages`).

        Satisfies :class:`~retrieval.interfaces.base.BaseRetriever`.
        """
        await self.upsert_passages(passages)

    async def upsert_passages(
        self,
        passages: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Encode and upsert *passages* into Qdrant.

        Each passage dict must contain ``id``, ``text``, and ``title``.

        Args:
            passages:   List of passage dicts.
            batch_size: Number of passages per upsert batch.
        """
        if not passages:
            logger.warning("upsert_passages called with empty list.")
            return

        texts = [p.get("text", "") for p in passages]

        logger.info("Encoding %d passages for Qdrant upsert…", len(passages))
        embeddings = self.encoder.encode_passages(texts, show_progress=True)

        total = len(passages)
        upserted = 0

        for start in range(0, total, batch_size):
            batch_passages = passages[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]

            points: List[qmodels.PointStruct] = []
            for p, vec in zip(batch_passages, batch_embeddings):
                pid = str(p["id"])
                # Use a deterministic UUID based on passage_id so upserts
                # are idempotent.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pid))
                payload = {
                    "passage_id": pid,
                    "text": p.get("text", ""),
                    "title": p.get("title", ""),
                    "metadata": {
                        k: v
                        for k, v in p.items()
                        if k not in {"id", "text", "title"}
                    },
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

            await self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            upserted += len(points)
            logger.debug(
                "Upserted %d / %d passages to Qdrant.", upserted, total
            )

        logger.info(
            "Finished upserting %d passages into collection '%s'.",
            upserted,
            self.collection_name,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Encode *query* and search Qdrant for the nearest passage vectors.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~retrieval.interfaces.base.RetrievalResult`
            sorted by descending cosine similarity.
        """
        if not query or not query.strip():
            logger.warning("Empty query received; returning empty results.")
            return []

        query_vec = self.encoder.encode_query(query).tolist()

        hits = await self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

        results: List[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            results.append(
                RetrievalResult(
                    passage_id=payload.get("passage_id", str(hit.id)),
                    text=payload.get("text", ""),
                    title=payload.get("title", ""),
                    score=float(hit.score),
                    rank=rank,
                    metadata=payload.get("metadata", {}),
                )
            )

        logger.debug(
            "Qdrant search returned %d results for query '%s'.",
            len(results),
            query[:80],
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return the number of vectors in the collection."""
        try:
            info = await self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as exc:
            logger.error("Failed to count Qdrant points: %s", exc)
            return 0

    async def health_check(self) -> bool:
        """Return ``True`` if Qdrant is reachable and the collection exists."""
        try:
            collections = await self._client.get_collections()
            names = [c.name for c in collections.collections]
            exists = self.collection_name in names
            if not exists:
                logger.warning(
                    "Qdrant collection '%s' does not exist.", self.collection_name
                )
            return exists
        except Exception as exc:
            logger.error("Qdrant health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the underlying Qdrant HTTP client."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "QdrantRetriever":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
