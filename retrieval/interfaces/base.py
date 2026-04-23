"""
Abstract base classes for the VectorLift retrieval layer.

All retrievers and rerankers must implement these interfaces to ensure
consistent behaviour across BM25, dense, and hybrid backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieved passage with its relevance score and rank.

    Attributes:
        passage_id: Unique identifier for the passage (matches the indexed id).
        text:       Full passage text returned to the caller.
        title:      Document title the passage belongs to.
        score:      Relevance score assigned by the retriever / reranker.
                    Higher is always better (scores are normalised to [0, 1]
                    by convention, but retrievers may exceed this range before
                    normalisation).
        rank:       1-based position in the result list (1 = most relevant).
        metadata:   Arbitrary extra fields (e.g. url, source, date).
    """

    passage_id: str
    text: str
    title: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "passage_id": self.passage_id,
            "text": self.text,
            "title": self.title,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        return cls(
            passage_id=data["passage_id"],
            text=data["text"],
            title=data["title"],
            score=data["score"],
            rank=data["rank"],
            metadata=data.get("metadata", {}),
        )


class BaseRetriever(ABC):
    """Interface that every retriever backend must satisfy.

    Implementations include:
    - ``ElasticsearchRetriever``  – BM25 via Elasticsearch ≥ 8
    - ``LocalBM25Retriever``      – in-process BM25 via rank-bm25
    - ``DenseRetriever``          – bi-encoder + FAISS / Qdrant
    - ``HybridRetriever``         – BM25 + dense fusion
    """

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Return the top-k passages most relevant to *query*.

        Args:
            query:  Raw natural-language query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of :class:`RetrievalResult` objects sorted by descending
            score (rank 1 = best).
        """

    @abstractmethod
    async def index(self, passages: List[Dict[str, Any]]) -> None:
        """Index (or re-index) a collection of passages.

        Each passage dict must contain at minimum::

            {
                "id":    str,   # unique passage identifier
                "text":  str,   # full passage text
                "title": str,   # document title
            }

        Additional keys are stored as metadata.

        Args:
            passages: List of passage dicts.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the backend is reachable and ready."""

    # ------------------------------------------------------------------
    # Optional helpers with sensible defaults
    # ------------------------------------------------------------------

    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[RetrievalResult]]:
        """Retrieve results for a list of queries sequentially.

        Subclasses may override this with a more efficient batched
        implementation.
        """
        results: List[List[RetrievalResult]] = []
        for query in queries:
            results.append(await self.retrieve(query, top_k=top_k))
        return results


class BaseReranker(ABC):
    """Interface that every reranker must satisfy.

    Implementations include:
    - ``CrossEncoderReranker`` – transformer cross-encoder scoring
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int,
    ) -> List[RetrievalResult]:
        """Score and re-order *candidates* with respect to *query*.

        Args:
            query:      Raw natural-language query string.
            candidates: First-stage retrieval results (any order).
            top_n:      Return only the top-n after reranking.

        Returns:
            Re-ranked list of :class:`RetrievalResult` (length ≤ top_n),
            with updated ``score`` and ``rank`` fields.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the reranker model is loaded and ready."""
