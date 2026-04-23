"""
Embedding generation pipeline for VectorLift.

Wraps any BiEncoder model and provides:
  - Batched generation with rich progress bar
  - ETA estimation
  - Disk persistence (numpy .npz) and resumable generation
  - Save/load helpers compatible with BatchIndexer
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for the underlying encoder
# ---------------------------------------------------------------------------


class BiEncoderProtocol(Protocol):
    """Minimal interface expected from a bi-encoder model."""

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 256,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# EmbeddingGenerator
# ---------------------------------------------------------------------------


class EmbeddingGenerator:
    """High-level embedding generation pipeline.

    Args:
        encoder:    Any object satisfying :class:`BiEncoderProtocol`
                    (e.g. a ``SentenceTransformer`` instance).
        batch_size: Number of texts to encode per forward pass.
    """

    def __init__(self, encoder: BiEncoderProtocol, batch_size: int = 256) -> None:
        self.encoder = encoder
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts and return a float32 embedding matrix.

        Args:
            texts:         Input texts.
            show_progress: Show a ``rich`` progress bar with ETA.

        Returns:
            numpy array of shape ``(len(texts), embedding_dim)`` dtype float32.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        n = len(texts)
        logger.info("Generating embeddings for %d texts (batch_size=%d).", n, self.batch_size)

        all_embeddings: List[np.ndarray] = []
        start = time.perf_counter()
        processed = 0

        _progress_ctx = None
        _task_id = None

        if show_progress:
            try:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )
                _progress_ctx = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]Embedding"),
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                )
                _progress_ctx.start()
                _task_id = _progress_ctx.add_task("encoding", total=n)
            except ImportError:
                logger.warning("rich not installed; progress bar disabled.")

        try:
            for batch_start in range(0, n, self.batch_size):
                batch = texts[batch_start : batch_start + self.batch_size]

                batch_emb: np.ndarray = self.encoder.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.append(batch_emb.astype(np.float32))
                processed += len(batch)

                if _progress_ctx and _task_id is not None:
                    _progress_ctx.update(_task_id, completed=processed)

                # Log ETA every 10 batches
                if (batch_start // self.batch_size) % 10 == 0:
                    elapsed = time.perf_counter() - start
                    rate = processed / max(elapsed, 1e-6)
                    remaining = (n - processed) / max(rate, 1e-6)
                    logger.debug(
                        "Encoded %d/%d  rate=%.0f docs/s  ETA=%.0fs",
                        processed, n, rate, remaining,
                    )
        finally:
            if _progress_ctx:
                _progress_ctx.stop()

        embeddings = np.vstack(all_embeddings)
        elapsed = time.perf_counter() - start
        logger.info(
            "Embedding complete: shape=%s  dtype=%s  %.1fs  %.0f docs/s",
            embeddings.shape, embeddings.dtype, elapsed, n / max(elapsed, 1e-6),
        )
        return embeddings

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def generate_and_save(
        self,
        passages: List[dict],
        output_path: str,
        resume: bool = True,
    ) -> None:
        """Generate embeddings for a list of passages and save to a .npz file.

        Supports resuming: if ``output_path`` already exists and ``resume=True``,
        already-computed passage IDs are loaded and only the remaining passages
        are encoded.  The final file always contains all embeddings.

        Args:
            passages:    List of dicts with ``id`` and ``text`` keys.
            output_path: Destination path (will be saved as ``<output_path>.npz``
                         if the extension is not already ``.npz``).
            resume:      Skip passages whose IDs are already in the output file.
        """
        out_path = Path(output_path)
        if not out_path.suffix == ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing_emb: Optional[np.ndarray] = None
        existing_ids: List[str] = []

        if resume and out_path.exists():
            try:
                existing_emb, existing_ids = self.load_embeddings(str(out_path))
                logger.info(
                    "Resuming: found %d existing embeddings in '%s'.",
                    len(existing_ids), out_path,
                )
            except Exception as exc:
                logger.warning("Could not load existing embeddings (%s); starting fresh.", exc)
                existing_emb = None
                existing_ids = []

        done_ids: set[str] = set(existing_ids)
        remaining = [p for p in passages if str(p["id"]) not in done_ids]

        if not remaining:
            logger.info("All passages already embedded; nothing to do.")
            return

        logger.info(
            "Encoding %d passages (%d already done).", len(remaining), len(done_ids)
        )
        texts = [p["text"] for p in remaining]
        new_ids = [str(p["id"]) for p in remaining]

        new_emb = self.generate(texts, show_progress=True)

        # Merge with existing
        if existing_emb is not None and len(existing_ids) > 0:
            all_emb = np.vstack([existing_emb, new_emb])
            all_ids = existing_ids + new_ids
        else:
            all_emb = new_emb
            all_ids = new_ids

        np.savez_compressed(
            str(out_path),
            embeddings=all_emb.astype(np.float32),
            ids=np.array(all_ids, dtype=str),
        )
        logger.info(
            "Saved %d embeddings (dim=%d) to '%s'.",
            len(all_ids), all_emb.shape[1], out_path,
        )

    @staticmethod
    def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings from a .npz file.

        Args:
            path: Path to a ``.npz`` file produced by :meth:`generate_and_save`.

        Returns:
            Tuple of:
              - embeddings: float32 numpy array of shape (N, D)
              - ids:        list of passage ID strings (length N)

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError:          If the file does not contain the expected keys.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embedding file not found: '{path}'")

        data = np.load(str(p), allow_pickle=False)

        required_keys = {"embeddings", "ids"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f"Missing keys in '{path}': {missing}")

        embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        ids: List[str] = data["ids"].tolist()

        logger.info(
            "Loaded embeddings: shape=%s  n_ids=%d  from '%s'.",
            embeddings.shape, len(ids), path,
        )
        return embeddings, ids

    def estimate_time(self, n_texts: int) -> float:
        """Rough estimate of encoding time in seconds.

        Runs a tiny warm-up sample and extrapolates.

        Args:
            n_texts: Number of texts to estimate for.

        Returns:
            Estimated seconds (float).
        """
        sample_size = min(self.batch_size, n_texts, 32)
        sample = ["sample text for timing"] * sample_size

        t0 = time.perf_counter()
        self.encoder.encode(
            sample,
            batch_size=sample_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0

        rate = sample_size / max(elapsed, 1e-9)
        estimate = n_texts / rate
        logger.debug(
            "Time estimate for %d texts: %.1fs  (rate=%.0f docs/s from %d-sample warm-up)",
            n_texts, estimate, rate, sample_size,
        )
        return estimate
