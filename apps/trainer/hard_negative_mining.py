"""
Hard negative mining for VectorLift bi-encoder training.

Uses a trained dense retriever to find hard negatives — passages that
are retrieved by the model for a query but are not actually relevant —
and saves them as (query, positive, hard_negative) triplets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for the dense retriever
# ---------------------------------------------------------------------------


class DenseRetrieverProtocol(Protocol):
    """Minimal interface expected from a dense retriever."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Any]: ...  # returns list of RetrievalResult-like objects


# ---------------------------------------------------------------------------
# HardNegativeMiner
# ---------------------------------------------------------------------------


class HardNegativeMiner:
    """Mines hard negatives using an existing dense retriever model.

    For each (query, positive_passage) pair, the miner:
      1. Retrieves top-k passages using the dense retriever.
      2. Filters out the known positive passage(s).
      3. Randomly samples one hard negative from the remaining candidates.

    This produces negatives that are "confusing" for the current model,
    which is essential for curriculum learning in a second training stage.

    Args:
        retriever:   A dense retriever with an ``async retrieve(query, top_k)`` method.
        batch_size:  Parallel retrieval concurrency (semaphore limit).
        seed:        Random seed for negative sampling.
    """

    def __init__(
        self,
        retriever: DenseRetrieverProtocol,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.retriever = retriever
        self.batch_size = batch_size
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine(
        self,
        queries: List[str],
        positive_passages: List[str],
        positive_ids: Optional[List[str]] = None,
        top_k: int = 30,
    ) -> List[Tuple[str, str, str]]:
        """Mine hard negatives for a list of (query, positive) pairs.

        Args:
            queries:           List of query strings (length N).
            positive_passages: List of positive passage texts matching each query (length N).
            positive_ids:      Optional list of passage IDs for the positives.
                               If provided, filtering is done by ID instead of text comparison.
            top_k:             Number of candidates to retrieve per query.

        Returns:
            List of (query, positive, hard_negative) tuples.  Queries for which
            no hard negative was found (all top-k are positives) are skipped
            with a warning.

        Raises:
            ValueError: If ``queries`` and ``positive_passages`` have different lengths.
        """
        if len(queries) != len(positive_passages):
            raise ValueError(
                f"queries and positive_passages must have the same length, "
                f"got {len(queries)} vs {len(positive_passages)}"
            )

        logger.info(
            "Mining hard negatives for %d queries (top_k=%d).", len(queries), top_k
        )
        start = time.perf_counter()

        semaphore = asyncio.Semaphore(self.batch_size)
        triplets: List[Tuple[str, str, str]] = []
        skipped = 0

        async def _process(idx: int) -> Optional[Tuple[str, str, str]]:
            query = queries[idx]
            positive = positive_passages[idx]
            pos_id = positive_ids[idx] if positive_ids else None

            async with semaphore:
                try:
                    candidates = await self.retriever.retrieve(query, top_k=top_k)
                except Exception as exc:
                    logger.warning("Retrieval failed for query '%s': %s", query[:80], exc)
                    return None

            # Filter positives
            negatives = []
            for c in candidates:
                c_id = getattr(c, "passage_id", None) or getattr(c, "id", None)
                c_text = getattr(c, "text", "")

                is_positive = False
                if pos_id is not None and c_id == pos_id:
                    is_positive = True
                elif _text_overlap(c_text, positive):
                    is_positive = True

                if not is_positive:
                    negatives.append(c_text)

            if not negatives:
                return None

            hard_neg = self._rng.choice(negatives)
            return (query, positive, hard_neg)

        tasks = [asyncio.create_task(_process(i)) for i in range(len(queries))]
        results = await asyncio.gather(*tasks)

        for res in results:
            if res is not None:
                triplets.append(res)
            else:
                skipped += 1

        elapsed = time.perf_counter() - start
        logger.info(
            "Hard negative mining complete: %d triplets mined, %d skipped, %.1fs.",
            len(triplets), skipped, elapsed,
        )
        return triplets

    def save_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        output_path: str,
    ) -> None:
        """Save triplets to a JSONL file.

        Each line is a JSON object with keys ``query``, ``positive``, ``negative``.

        Args:
            triplets:    List of (query, positive, negative) tuples.
            output_path: Destination file path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as fh:
            for query, positive, negative in triplets:
                record = {"query": query, "positive": positive, "negative": negative}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved %d hard-negative triplets to '%s'.", len(triplets), out)

    @staticmethod
    def load_triplets(path: str) -> List[Tuple[str, str, str]]:
        """Load triplets from a JSONL file produced by :meth:`save_triplets`.

        Args:
            path: Path to a JSONL file.

        Returns:
            List of (query, positive, negative) tuples.
        """
        triplets: List[Tuple[str, str, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                triplets.append((rec["query"], rec["positive"], rec["negative"]))
        logger.info("Loaded %d triplets from '%s'.", len(triplets), path)
        return triplets


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _text_overlap(a: str, b: str, threshold: float = 0.9) -> bool:
    """Heuristic: consider texts the same if one is a substring of the other
    or their normalised overlap exceeds the threshold."""
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if a_norm == b_norm:
        return True
    shorter, longer = (a_norm, b_norm) if len(a_norm) <= len(b_norm) else (b_norm, a_norm)
    if shorter and shorter in longer:
        return True
    # Token overlap
    a_toks = set(a_norm.split())
    b_toks = set(b_norm.split())
    if not a_toks or not b_toks:
        return False
    overlap = len(a_toks & b_toks) / len(a_toks | b_toks)
    return overlap >= threshold


async def _cli_mine(
    model_path: str,
    dataset_mode: str,
    output_path: str,
    top_k: int,
    max_queries: int,
    cache_dir: Optional[str],
) -> None:
    """Async main for the CLI command."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc

    from pipelines.ingestion.msmarco import MSMARCODataset

    ms = MSMARCODataset(cache_dir=cache_dir)
    triplets = ms.load_training_triplets(max_samples=max_queries)
    queries = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]

    # Build a minimal dense retriever around the SentenceTransformer for mining.
    # In production this would be a full DenseRetriever with Qdrant / FAISS backend.
    encoder = SentenceTransformer(model_path)

    class _SimpleRetriever:
        """Brute-force cosine retriever over the positive passage set for mining."""

        def __init__(self, passages: List[str], encoder: Any) -> None:
            import numpy as np
            logger.info("Encoding %d passages for hard negative mining …", len(passages))
            self._texts = passages
            self._embs: np.ndarray = encoder.encode(
                passages, batch_size=256, show_progress_bar=True, convert_to_numpy=True
            )
            # L2 normalise for cosine similarity
            norms = np.linalg.norm(self._embs, axis=1, keepdims=True)
            self._embs = self._embs / np.maximum(norms, 1e-9)

        async def retrieve(self, query: str, top_k: int = 30) -> List[Any]:
            import numpy as np
            from dataclasses import dataclass

            @dataclass
            class _Hit:
                passage_id: str
                text: str
                score: float

            q_emb = encoder.encode([query], convert_to_numpy=True)[0]
            q_emb = q_emb / max(float(np.linalg.norm(q_emb)), 1e-9)
            scores = self._embs @ q_emb
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [
                _Hit(passage_id=str(i), text=self._texts[i], score=float(scores[i]))
                for i in top_idx
            ]

    retriever = _SimpleRetriever(positives, encoder)
    miner = HardNegativeMiner(retriever=retriever, batch_size=64)
    hard_triplets = await miner.mine(queries, positives, top_k=top_k)
    miner.save_triplets(hard_triplets, output_path)


def _main_cli() -> None:
    import typer

    cli_app = typer.Typer(name="hard-negative-miner", add_completion=False)

    @cli_app.command()
    def mine(
        model_path: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--model"),
        dataset_mode: str = typer.Option("dev", "--dataset-mode"),
        output_path: str = typer.Option("data/hard_negatives.jsonl", "--output"),
        top_k: int = typer.Option(30, "--top-k"),
        max_queries: int = typer.Option(10_000, "--max-queries"),
        cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    ) -> None:
        """Mine hard negatives using a dense retriever."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        asyncio.run(_cli_mine(model_path, dataset_mode, output_path, top_k, max_queries, cache_dir))

    cli_app()


if __name__ == "__main__":
    _main_cli()
