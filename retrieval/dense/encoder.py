"""
Bi-encoder for query and passage embedding using SentenceTransformers.

Suitable for dense retrieval with FAISS or Qdrant backends.

Requires:
    sentence-transformers>=2.2
    torch>=2.0
    numpy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default model trained specifically for asymmetric passage retrieval on MSMARCO
_DEFAULT_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"

# Prefix tokens used by the TAS-B model for asymmetric encoding
_QUERY_PREFIX = ""   # TAS-B does not use explicit prefixes but subclasses may
_PASSAGE_PREFIX = ""


class BiEncoder:
    """Wraps a SentenceTransformer model to encode queries and passages
    separately, supporting mixed-precision inference.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device:             Torch device string (``"cpu"``, ``"cuda"``,
                            ``"mps"``). Auto-detected when ``None``.
        max_seq_length:     Override the model's maximum sequence length.
        batch_size:         Batch size used in :meth:`encode_batch`.
        normalize_embeddings: L2-normalise output vectors (required for
                              inner-product cosine similarity).
        use_fp16:           Run the model in half-precision on GPU/MPS.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 512,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Disable fp16 on CPU (not supported)
        self.use_fp16 = use_fp16 and self.device != "cpu"

        logger.info(
            "Loading BiEncoder model '%s' on device '%s' (fp16=%s).",
            model_name_or_path,
            self.device,
            self.use_fp16,
        )
        self._model = SentenceTransformer(model_name_or_path, device=self.device)

        # Override max sequence length if requested
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        self.max_seq_length = self._model.max_seq_length

        if self.use_fp16:
            self._model.half()

        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "BiEncoder ready – embedding dim=%d, max_seq_length=%d.",
            self.embedding_dim,
            self.max_seq_length,
        )

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: Raw query text.

        Returns:
            1-D numpy array of shape ``(embedding_dim,)``.
        """
        return self.encode_queries([query], show_progress=False)[0]

    def encode_queries(
        self,
        queries: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of query strings.

        Args:
            queries:       List of raw query strings.
            show_progress: Show a tqdm progress bar.

        Returns:
            2-D numpy array of shape ``(len(queries), embedding_dim)``.
        """
        return self.encode_batch(queries, is_query=True, show_progress=show_progress)

    def encode_passages(
        self,
        passages: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of passage strings.

        Args:
            passages:      List of raw passage texts.
            show_progress: Show a tqdm progress bar.

        Returns:
            2-D numpy array of shape ``(len(passages), embedding_dim)``.
        """
        return self.encode_batch(passages, is_query=False, show_progress=show_progress)

    def encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Core encoding method used by all public helpers.

        Applies the appropriate prompt prefix (if any) and runs inference
        in mini-batches using :attr:`batch_size`.

        Args:
            texts:         Input text strings.
            is_query:      If ``True`` apply query-side prefix; otherwise
                           apply passage-side prefix.
            show_progress: Show a tqdm progress bar.

        Returns:
            2-D numpy array of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        prefix = _QUERY_PREFIX if is_query else _PASSAGE_PREFIX
        prefixed = [f"{prefix}{t}" if prefix else t for t in texts]

        all_embeddings: List[np.ndarray] = []
        n_batches = (len(prefixed) + self.batch_size - 1) // self.batch_size

        iterator = range(n_batches)
        if show_progress and n_batches > 1:
            label = "Encoding queries" if is_query else "Encoding passages"
            iterator = tqdm(iterator, desc=label, total=n_batches, unit="batch")

        with torch.no_grad():
            for i in iterator:
                batch = prefixed[i * self.batch_size : (i + 1) * self.batch_size]
                emb: np.ndarray = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                )
                all_embeddings.append(emb.astype(np.float32))

        result = np.vstack(all_embeddings)
        logger.debug(
            "Encoded %d %s(s) → shape %s.",
            len(texts),
            "query" if is_query else "passage",
            result.shape,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the underlying SentenceTransformer to *path*.

        Args:
            path: Directory path to save the model artefacts.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info("BiEncoder model saved to '%s'.", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
    ) -> "BiEncoder":
        """Load a previously saved BiEncoder.

        Args:
            path:                 Directory saved by :meth:`save`.
            device:               Torch device (auto-detected when ``None``).
            batch_size:           Batch size for inference.
            normalize_embeddings: L2-normalise output vectors.
            use_fp16:             Mixed-precision inference.

        Returns:
            Loaded :class:`BiEncoder` instance.
        """
        return cls(
            model_name_or_path=str(path),
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BiEncoder(model='{self.model_name_or_path}', "
            f"dim={self.embedding_dim}, device='{self.device}', "
            f"fp16={self.use_fp16})"
        )
