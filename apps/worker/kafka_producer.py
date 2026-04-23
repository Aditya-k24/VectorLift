"""
Kafka document producer for VectorLift.

Serialises passage documents and indexing job configs to JSON and publishes
them to a Kafka topic.  Uses kafka-python's KafkaProducer.

Design:
  - All messages are UTF-8 JSON.
  - Each message has a ``_type`` field: ``passage`` | ``indexing_job``.
  - Delivery errors raise ``KafkaProducerError``.
  - The producer is thread-safe (kafka-python is thread-safe by design).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class KafkaProducerError(Exception):
    """Raised when a message delivery fails permanently."""


class DocumentProducer:
    """Kafka producer for VectorLift document messages.

    Args:
        bootstrap_servers: Kafka broker address(es), e.g. ``"localhost:9092"``
                           or a comma-separated list.
        topic:             Default topic to publish to.
        acks:              Producer acks setting (``"all"`` for durability).
        max_retries:       Internal kafka-python retries.
        linger_ms:         Batching window in milliseconds.
        batch_size:        Maximum bytes per batch.
        compression_type:  ``None`` | ``"gzip"`` | ``"snappy"`` | ``"lz4"``.
        on_delivery_error: Optional callback called with ``(message, exception)``
                           when an async delivery fails.
    """

    def __init__(
        self,
        bootstrap_servers: str | List[str],
        topic: str,
        acks: str | int = "all",
        max_retries: int = 5,
        linger_ms: int = 10,
        batch_size: int = 65_536,
        compression_type: Optional[str] = "gzip",
        on_delivery_error: Optional[Callable[[Dict[str, Any], Exception], None]] = None,
    ) -> None:
        try:
            from kafka import KafkaProducer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install kafka-python: pip install kafka-python"
            ) from exc

        servers = (
            bootstrap_servers
            if isinstance(bootstrap_servers, list)
            else [s.strip() for s in bootstrap_servers.split(",")]
        )

        self.topic = topic
        self._on_delivery_error = on_delivery_error

        self._producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=acks,
            retries=max_retries,
            linger_ms=linger_ms,
            batch_size=batch_size,
            compression_type=compression_type,
        )
        logger.info(
            "DocumentProducer ready. Brokers=%s  Topic=%s", servers, topic
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_passage(
        self,
        passage: Dict[str, Any],
        topic: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Serialise and send a single passage document.

        Args:
            passage: Passage dict with at minimum ``id``, ``text``, ``title``.
            topic:   Override the default topic.
            key:     Kafka message key (defaults to passage ``id``).

        Raises:
            KafkaProducerError: If the send fails after retries.
        """
        msg = {"_type": "passage", **passage}
        msg_key = key or str(passage.get("id", ""))
        self._send(msg, topic=topic or self.topic, key=msg_key)

    def send_batch(
        self,
        passages: List[Dict[str, Any]],
        topic: Optional[str] = None,
    ) -> None:
        """Send a list of passages in a tight loop and flush at the end.

        Args:
            passages: List of passage dicts.
            topic:    Override the default topic.
        """
        dest = topic or self.topic
        start = time.perf_counter()
        sent = 0

        for passage in passages:
            msg = {"_type": "passage", **passage}
            key = str(passage.get("id", ""))
            self._send(msg, topic=dest, key=key, flush=False)
            sent += 1

        self._producer.flush()
        elapsed = time.perf_counter() - start
        logger.info(
            "Sent batch of %d passages to '%s' in %.2fs (%.0f docs/s).",
            sent, dest, elapsed, sent / max(elapsed, 1e-6),
        )

    def send_indexing_job(
        self,
        job_config: Dict[str, Any],
        topic: Optional[str] = None,
    ) -> None:
        """Publish an indexing job configuration message.

        Indexing jobs carry a ``_type: indexing_job`` marker so consumers
        can route them appropriately.

        Args:
            job_config: Arbitrary dict describing the indexing job.
            topic:      Override the default topic.
        """
        msg = {"_type": "indexing_job", **job_config}
        self._send(msg, topic=topic or self.topic, key="indexing_job")
        logger.info("Sent indexing job config: %s", list(job_config.keys()))

    def close(self, timeout: float = 10.0) -> None:
        """Flush pending messages and close the producer.

        Args:
            timeout: Seconds to wait for in-flight messages.
        """
        logger.info("Closing DocumentProducer (flushing up to %.1fs) …", timeout)
        self._producer.flush(timeout=int(timeout * 1000))
        self._producer.close(timeout=int(timeout * 1000))
        logger.info("DocumentProducer closed.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send(
        self,
        message: Dict[str, Any],
        topic: str,
        key: Optional[str] = None,
        flush: bool = True,
    ) -> None:
        """Internal send with error handling."""
        try:
            future = self._producer.send(topic, value=message, key=key)
            if flush:
                record_metadata = future.get(timeout=10)
                logger.debug(
                    "Message delivered: topic=%s  partition=%d  offset=%d",
                    record_metadata.topic,
                    record_metadata.partition,
                    record_metadata.offset,
                )
        except Exception as exc:
            logger.error("Failed to deliver message to '%s': %s", topic, exc)
            if self._on_delivery_error:
                self._on_delivery_error(message, exc)
            else:
                raise KafkaProducerError(
                    f"Delivery failed to topic '{topic}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DocumentProducer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
