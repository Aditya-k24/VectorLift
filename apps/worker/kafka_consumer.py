"""
Kafka indexing consumer for VectorLift.

Consumes passage messages from a Kafka topic and indexes them into
Elasticsearch (BM25) and Qdrant (dense), with:
  - Exponential back-off retry on transient failures
  - Dead-letter logging for permanently failed messages
  - Manual offset commit after successful processing
  - Graceful shutdown on SIGTERM / SIGINT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols (avoid circular imports with retrieval layer)
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
        convert_to_numpy: bool = True,
    ) -> Any: ...  # returns np.ndarray


# ---------------------------------------------------------------------------
# IndexingConsumer
# ---------------------------------------------------------------------------


class IndexingConsumer:
    """Kafka consumer that indexes incoming passage documents.

    Args:
        bootstrap_servers:  Broker address(es) (comma-separated string or list).
        topic:              Kafka topic to consume.
        group_id:           Consumer group ID.
        es_retriever:       Elasticsearch retriever for BM25 indexing.
        qdrant_retriever:   Qdrant retriever for dense vector indexing.
        encoder:            Embedding encoder.
        max_retries:        Maximum per-message retry attempts before dead-lettering.
        base_backoff_ms:    Base back-off for the first retry (milliseconds).
        max_backoff_ms:     Cap for exponential back-off (milliseconds).
        dead_letter_path:   File path to log permanently failed messages.
        poll_timeout_ms:    How long each ``poll()`` call waits.
        auto_offset_reset:  ``"earliest"`` | ``"latest"``.
        enable_auto_commit: Must be ``False`` for exactly-once-at-least semantics.
    """

    def __init__(
        self,
        bootstrap_servers: str | List[str],
        topic: str,
        group_id: str,
        es_retriever: ESRetrieverProtocol,
        qdrant_retriever: QdrantRetrieverProtocol,
        encoder: EncoderProtocol,
        max_retries: int = 5,
        base_backoff_ms: int = 100,
        max_backoff_ms: int = 30_000,
        dead_letter_path: str = "dead_letters.jsonl",
        poll_timeout_ms: int = 1_000,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
    ) -> None:
        self.topic = topic
        self.group_id = group_id
        self.es_retriever = es_retriever
        self.qdrant_retriever = qdrant_retriever
        self.encoder = encoder
        self.max_retries = max_retries
        self.base_backoff_ms = base_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self.dead_letter_path = dead_letter_path
        self.poll_timeout_ms = poll_timeout_ms
        self._stop_event = asyncio.Event()

        servers = (
            bootstrap_servers
            if isinstance(bootstrap_servers, list)
            else [s.strip() for s in bootstrap_servers.split(",")]
        )

        try:
            from kafka import KafkaConsumer  # type: ignore[import]
            from kafka.errors import KafkaError  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install kafka-python: pip install kafka-python") from exc

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            # Prevent the consumer from being evicted during slow indexing
            session_timeout_ms=30_000,
            heartbeat_interval_ms=10_000,
            max_poll_interval_ms=300_000,
        )

        # Register SIGTERM / SIGINT handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_signal)
            except (OSError, ValueError):
                pass  # Not in main thread; skip

        logger.info(
            "IndexingConsumer ready. Group=%s  Topic=%s  Brokers=%s",
            group_id, topic, servers,
        )

    # ------------------------------------------------------------------
    # Main consume loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the consume-index loop.  Runs until ``stop()`` is called or a
        SIGTERM / SIGINT is received.
        """
        logger.info("IndexingConsumer starting consume loop …")
        processed = 0
        failed = 0
        start_time = time.perf_counter()

        loop = asyncio.get_event_loop()

        while not self._stop_event.is_set():
            # Run the blocking Kafka poll in a thread to avoid blocking the loop
            try:
                msg_pack = await loop.run_in_executor(
                    None,
                    lambda: self._consumer.poll(timeout_ms=self.poll_timeout_ms),
                )
            except Exception as exc:
                logger.error("poll() raised an exception: %s", exc)
                await asyncio.sleep(1)
                continue

            for tp, messages in msg_pack.items():
                for message in messages:
                    try:
                        await self.process_message(message)
                        # Commit offset immediately after successful processing
                        await loop.run_in_executor(None, self._consumer.commit)
                        processed += 1
                    except Exception as exc:
                        failed += 1
                        await self._handle_failure(message, exc)

            # Periodic throughput log
            elapsed = time.perf_counter() - start_time
            if elapsed > 0 and processed % 1000 == 0 and processed > 0:
                logger.info(
                    "Throughput: %d processed  %d failed  %.0f docs/s",
                    processed, failed, processed / elapsed,
                )

        # Graceful shutdown
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Consumer shutting down. Processed=%d  Failed=%d  Elapsed=%.1fs",
            processed, failed, elapsed,
        )
        self._consumer.close()

    def stop(self) -> None:
        """Signal the consumer to stop after the current batch."""
        logger.info("Stop requested for IndexingConsumer.")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    async def process_message(self, message: Any) -> None:
        """Deserialise a Kafka message and index it.

        Supported message types (``_type`` field):
          - ``"passage"``:      Index into ES + Qdrant.
          - ``"indexing_job"``: Log receipt and acknowledge (job dispatch handled elsewhere).

        Args:
            message: KafkaConsumer message record.

        Raises:
            Exception: Re-raised after logging; caller is responsible for retry.
        """
        payload: Dict[str, Any] = message.value
        msg_type = payload.get("_type", "passage")

        if msg_type == "passage":
            await self._index_passage(payload)
        elif msg_type == "indexing_job":
            logger.info(
                "Received indexing_job: %s",
                {k: v for k, v in payload.items() if k != "_type"},
            )
        else:
            logger.warning("Unknown message type '%s'; skipping.", msg_type)

    async def _index_passage(self, payload: Dict[str, Any]) -> None:
        """Index a single passage into Elasticsearch and Qdrant."""
        passage = {k: v for k, v in payload.items() if k != "_type"}

        # BM25 indexing
        await self.es_retriever.index_passages([passage], batch_size=1)

        # Dense indexing
        text = passage.get("text", "")
        pid = str(passage.get("id", ""))
        if text and pid:
            import numpy as np

            embedding: np.ndarray = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.encoder.encode(
                    [text], batch_size=1, show_progress_bar=False, convert_to_numpy=True
                ),
            )
            vector = embedding[0].tolist()
            payload_qdrant = {"text": text, "title": passage.get("title", "")}
            await self.qdrant_retriever.upsert_vectors(
                ids=[pid], vectors=[vector], payloads=[payload_qdrant]
            )
        else:
            logger.warning(
                "Passage missing 'text' or 'id'; skipped dense indexing. Keys: %s",
                list(passage.keys()),
            )

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    async def _handle_failure(self, message: Any, error: Exception) -> None:
        """Handle a permanently-failed message.

        Implements exponential back-off retry; if retries are exhausted, the
        message is written to the dead-letter log.

        Args:
            message: Original Kafka message.
            error:   Exception that caused the failure.
        """
        payload = getattr(message, "value", {})
        logger.warning(
            "Processing failed for message at offset %s: %s. Attempting retry …",
            getattr(message, "offset", "?"),
            error,
        )

        for attempt in range(1, self.max_retries + 1):
            backoff_ms = min(
                self.base_backoff_ms * (2 ** (attempt - 1)),
                self.max_backoff_ms,
            )
            await asyncio.sleep(backoff_ms / 1000.0)
            logger.info("Retry attempt %d/%d …", attempt, self.max_retries)
            try:
                await self.process_message(message)
                logger.info("Retry %d succeeded.", attempt)
                return
            except Exception as retry_exc:
                logger.warning("Retry %d failed: %s", attempt, retry_exc)
                error = retry_exc

        # All retries exhausted → dead-letter
        logger.error(
            "Message at offset %s permanently failed after %d retries. "
            "Writing to dead-letter log '%s'.",
            getattr(message, "offset", "?"),
            self.max_retries,
            self.dead_letter_path,
        )
        dead_letter_record = {
            "topic": getattr(message, "topic", self.topic),
            "partition": getattr(message, "partition", -1),
            "offset": getattr(message, "offset", -1),
            "payload": payload,
            "error": str(error),
            "timestamp": time.time(),
        }
        try:
            with open(self.dead_letter_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dead_letter_record, ensure_ascii=False) + "\n")
        except OSError as log_exc:
            logger.error("Could not write to dead-letter file: %s", log_exc)

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info("Received signal %d; initiating graceful shutdown.", signum)
        self._stop_event.set()
