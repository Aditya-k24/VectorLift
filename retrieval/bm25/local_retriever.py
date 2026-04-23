"""
In-process BM25 retriever powered by rank-bm25.

Designed for development, unit tests, and offline evaluation where
running Elasticsearch is not practical.  The full corpus is held in
memory and can be serialised to / deserialised from disk via pickle.

Requires:
    rank-bm25>=0.2
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from retrieval.interfaces.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> List[str]:
    """Lightweight tokeniser: lowercase + whitespace split."""
    return text.lower().split()


class LocalBM25Retriever(BaseRetriever):
    """BM25 retriever that runs entirely in the current Python process.

    Args:
        k1:  BM25Okapi *k1* parameter (term-frequency saturation).
        b:   BM25Okapi *b* parameter (length normalisation factor).
        epsilon: BM25Okapi *epsilon* parameter (floor for IDF).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # State populated by index()
        self._bm25: Optional[BM25Okapi] = None
        self._passage_ids: List[str] = []
        self._passage_texts: List[str] = []
        self._passage_titles: List[str] = []
        self._passage_metadata: List[Dict[str, Any]] = []
        self._is_indexed: bool = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, passages: List[Dict[str, Any]]) -> None:
        """Build a BM25 index from *passages* in memory.

        Each passage dict must contain ``id``, ``text``, and ``title``.
        Additional keys are stored as metadata.

        Args:
            passages: List of passage dicts to index.
        """
        if not passages:
            logger.warning("LocalBM25Retriever.index called with empty list.")
            return

        self._passage_ids = []
        self._passage_texts = []
        self._passage_titles = []
        self._passage_metadata = []

        tokenised_corpus: List[List[str]] = []

        for p in passages:
            pid = str(p["id"])
            text = str(p.get("text", ""))
            title = str(p.get("title", ""))
            meta = {k: v for k, v in p.items() if k not in {"id", "text", "title"}}

            self._passage_ids.append(pid)
            self._passage_texts.append(text)
            self._passage_titles.append(title)
            self._passage_metadata.append(meta)

            # Concatenate title + text so title contributes to BM25 scoring
            combined = f"{title} {text}" if title else text
            tokenised_corpus.append(_tokenise(combined))

        self._bm25 = BM25Okapi(
            tokenised_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )
        self._is_indexed = True
        logger.info("LocalBM25Retriever indexed %d passages.", len(passages))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Return the top-k passages by BM25 score for *query*.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results.

        Returns:
            List of :class:`~retrieval.interfaces.base.RetrievalResult`
            sorted by descending score.

        Raises:
            RuntimeError: If :meth:`index` has not been called yet.
        """
        if not self._is_indexed or self._bm25 is None:
            raise RuntimeError(
                "LocalBM25Retriever has not been indexed. Call index() first."
            )

        if not query or not query.strip():
            logger.warning("Empty query received; returning empty results.")
            return []

        tokenised_query = _tokenise(query)
        scores = self._bm25.get_scores(tokenised_query)

        # Pair up with indices, sort descending, take top_k
        k = min(top_k, len(scores))
        # argsort descending – avoid numpy dependency by using sorted()
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append(
                RetrievalResult(
                    passage_id=self._passage_ids[idx],
                    text=self._passage_texts[idx],
                    title=self._passage_titles[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    metadata=self._passage_metadata[idx],
                )
            )

        logger.debug(
            "LocalBM25 retrieved %d results for query '%s'.", len(results), query[:80]
        )
        return results

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if the retriever is indexed and ready."""
        ready = self._is_indexed and self._bm25 is not None
        if not ready:
            logger.warning("LocalBM25Retriever is not yet indexed.")
        return ready

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the retriever state to *path* using pickle.

        Args:
            path: Destination file path (e.g. ``"bm25_index.pkl"``).
        """
        if not self._is_indexed:
            raise RuntimeError("Cannot save an unindexed retriever. Call index() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "bm25": self._bm25,
            "passage_ids": self._passage_ids,
            "passage_texts": self._passage_texts,
            "passage_titles": self._passage_titles,
            "passage_metadata": self._passage_metadata,
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LocalBM25Retriever saved to '%s' (%d passages).", path, len(self._passage_ids))

    @classmethod
    def load(cls, path: str | Path) -> "LocalBM25Retriever":
        """Deserialise a retriever previously saved with :meth:`save`.

        Args:
            path: Path to the pickle file.

        Returns:
            A fully initialised :class:`LocalBM25Retriever`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with path.open("rb") as fh:
            state = pickle.load(fh)

        retriever = cls(
            k1=state["k1"],
            b=state["b"],
            epsilon=state["epsilon"],
        )
        retriever._bm25 = state["bm25"]
        retriever._passage_ids = state["passage_ids"]
        retriever._passage_texts = state["passage_texts"]
        retriever._passage_titles = state["passage_titles"]
        retriever._passage_metadata = state["passage_metadata"]
        retriever._is_indexed = True

        logger.info(
            "LocalBM25Retriever loaded from '%s' (%d passages).",
            path,
            len(retriever._passage_ids),
        )
        return retriever

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of indexed passages."""
        return len(self._passage_ids)

    def __repr__(self) -> str:
        return (
            f"LocalBM25Retriever(corpus_size={self.corpus_size}, "
            f"k1={self.k1}, b={self.b}, indexed={self._is_indexed})"
        )
