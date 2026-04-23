"""
Production Elasticsearch BM25 retriever for VectorLift.

Requires:
    elasticsearch>=8.0
    tenacity>=8.0

Index schema is designed for MSMARCO-style passage retrieval but is
general enough for any passage collection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from retrieval.interfaces.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default index settings
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
        "filter": {
            "english_stop": {
                "type": "stop",
                "stopwords": "_english_",
            },
            "english_stemmer": {
                "type": "stemmer",
                "language": "english",
            },
            "english_possessive_stemmer": {
                "type": "stemmer",
                "language": "possessive_english",
            },
        },
        "analyzer": {
            "english_analyzer": {
                "tokenizer": "standard",
                "filter": [
                    "english_possessive_stemmer",
                    "lowercase",
                    "english_stop",
                    "english_stemmer",
                ],
            },
        },
    },
}

_DEFAULT_MAPPINGS: Dict[str, Any] = {
    "properties": {
        "passage_id": {"type": "keyword"},
        "title": {
            "type": "text",
            "analyzer": "english_analyzer",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
        },
        "text": {
            "type": "text",
            "analyzer": "english_analyzer",
        },
        "metadata": {"type": "object", "dynamic": True},
    }
}


class ElasticsearchRetriever(BaseRetriever):
    """BM25 retriever backed by Elasticsearch ≥ 8.

    Args:
        host:        ES hostname (default ``"localhost"``).
        port:        ES port (default ``9200``).
        index_name:  Name of the ES index to read/write.
        settings:    Optional overrides merged into the default index
                     settings dict.
        username:    Optional HTTP basic auth username.
        password:    Optional HTTP basic auth password.
        use_ssl:     Whether to use HTTPS (default ``False`` for local dev).
        verify_certs: Verify SSL certificates (default ``False``).
        timeout:     Request timeout in seconds (default ``30``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "passages",
        settings: Optional[Dict[str, Any]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        verify_certs: bool = False,
        timeout: int = 30,
    ) -> None:
        self.index_name = index_name
        self._timeout = timeout

        scheme = "https" if use_ssl else "http"
        hosts = [{"host": host, "port": port, "scheme": scheme}]

        client_kwargs: Dict[str, Any] = {
            "hosts": hosts,
            "request_timeout": timeout,
            "verify_certs": verify_certs,
        }
        if username and password:
            client_kwargs["basic_auth"] = (username, password)

        self._client = AsyncElasticsearch(**client_kwargs)

        # Merge caller-supplied settings over defaults
        merged: Dict[str, Any] = {**_DEFAULT_SETTINGS}
        if settings:
            merged.update(settings)
        self._index_settings = merged

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def create_index(self, *, delete_if_exists: bool = False) -> None:
        """Create the passages index with BM25 settings and mappings.

        Args:
            delete_if_exists: If ``True`` drop any existing index first.
        """
        exists = await self._client.indices.exists(index=self.index_name)
        if exists:
            if delete_if_exists:
                logger.warning("Deleting existing index '%s'.", self.index_name)
                await self._client.indices.delete(index=self.index_name)
            else:
                logger.info("Index '%s' already exists – skipping creation.", self.index_name)
                return

        body: Dict[str, Any] = {
            "settings": self._index_settings,
            "mappings": _DEFAULT_MAPPINGS,
        }
        await self._client.indices.create(index=self.index_name, body=body)
        logger.info("Created Elasticsearch index '%s'.", self.index_name)

    async def delete_index(self) -> None:
        """Delete the index (irreversible)."""
        try:
            await self._client.indices.delete(index=self.index_name)
            logger.info("Deleted index '%s'.", self.index_name)
        except NotFoundError:
            logger.warning("Index '%s' not found; nothing to delete.", self.index_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, passages: List[Dict[str, Any]]) -> None:
        """Index passages (alias for :meth:`index_passages`).

        Satisfies :class:`~retrieval.interfaces.base.BaseRetriever`.
        """
        await self.index_passages(passages)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def index_passages(
        self,
        passages: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """Bulk-index *passages* into Elasticsearch.

        Each passage dict must contain ``id``, ``text``, and ``title``.
        Any extra keys are stored under ``metadata``.

        Args:
            passages:   List of passage dicts.
            batch_size: Number of documents per bulk request.
        """
        if not passages:
            logger.warning("index_passages called with empty list – nothing to do.")
            return

        total = len(passages)
        indexed = 0

        for start in range(0, total, batch_size):
            batch = passages[start : start + batch_size]
            actions = [
                {
                    "_index": self.index_name,
                    "_id": p["id"],
                    "_source": {
                        "passage_id": p["id"],
                        "text": p["text"],
                        "title": p.get("title", ""),
                        "metadata": {
                            k: v
                            for k, v in p.items()
                            if k not in {"id", "text", "title"}
                        },
                    },
                }
                for p in batch
            ]
            success, errors = await async_bulk(
                self._client,
                actions,
                chunk_size=batch_size,
                raise_on_error=False,
                raise_on_exception=False,
            )
            if errors:
                logger.error(
                    "Bulk indexing batch %d-%d produced %d errors: %s",
                    start,
                    start + len(batch),
                    len(errors),
                    errors[:3],
                )
            indexed += success
            logger.debug(
                "Indexed %d / %d passages (batch %d-%d).",
                indexed,
                total,
                start,
                start + len(batch),
            )

        # Refresh so documents are immediately searchable
        await self._client.indices.refresh(index=self.index_name)
        logger.info("Finished indexing %d passages into '%s'.", indexed, self.index_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=0.5, max=5),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """BM25 search over indexed passages.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of hits to return.

        Returns:
            List of :class:`~retrieval.interfaces.base.RetrievalResult`
            sorted by descending BM25 score.
        """
        if not query or not query.strip():
            logger.warning("Empty query received; returning empty results.")
            return []

        body: Dict[str, Any] = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^1.0", "title^0.5"],
                    "type": "best_fields",
                    "analyzer": "english_analyzer",
                }
            },
            "size": top_k,
            "_source": ["passage_id", "text", "title", "metadata"],
        }

        response = await self._client.search(index=self.index_name, body=body)

        hits = response["hits"]["hits"]
        results: List[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            results.append(
                RetrievalResult(
                    passage_id=src.get("passage_id", hit["_id"]),
                    text=src.get("text", ""),
                    title=src.get("title", ""),
                    score=float(hit["_score"]),
                    rank=rank,
                    metadata=src.get("metadata", {}),
                )
            )

        logger.debug("BM25 search returned %d results for query '%s'.", len(results), query[:80])
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def count_documents(self) -> int:
        """Return the number of documents currently in the index."""
        try:
            resp = await self._client.count(index=self.index_name)
            return int(resp["count"])
        except NotFoundError:
            return 0

    async def health_check(self) -> bool:
        """Return ``True`` if Elasticsearch is reachable and the index exists."""
        try:
            info = await self._client.info()
            version = info.get("version", {}).get("number", "unknown")
            logger.debug("Elasticsearch version: %s", version)
            exists = await self._client.indices.exists(index=self.index_name)
            return bool(exists)
        except Exception as exc:
            logger.error("Elasticsearch health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the underlying async HTTP transport."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ElasticsearchRetriever":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
