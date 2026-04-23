"""
Cross-encoder reranker for VectorLift.

Scores query-passage pairs jointly using a transformer sequence classifier.
This is typically used as a second stage after a fast first-stage retriever.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MSMARCO passage re-ranking
  - ~22 M parameters – fast enough for production second-stage reranking
  - Outputs logits interpreted as relevance scores

Requires:
    transformers>=4.35
    torch>=2.0
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from retrieval.interfaces.base import BaseReranker, RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# A single process-level thread pool for CPU-bound model inference so the
# async event loop is never blocked.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")


class CrossEncoderReranker(BaseReranker):
    """Transformer-based cross-encoder reranker.

    Scores each (query, passage) pair independently and re-ranks the
    candidate list in descending score order.

    Args:
        model_name_or_path: HuggingFace model ID or local directory.
        device:             Torch device string (auto-detected when ``None``).
        max_length:         Maximum combined token length for query + passage.
        batch_size:         Number of pairs scored per forward pass.
        use_fp16:           Half-precision inference on GPU (faster, slightly
                            lower precision).
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        use_fp16: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.use_fp16 = use_fp16 and self.device != "cpu"

        logger.info(
            "Loading CrossEncoderReranker model '%s' on device '%s' (fp16=%s).",
            model_name_or_path,
            self.device,
            self.use_fp16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self._model.to(self.device)
        self._model.eval()

        if self.use_fp16:
            self._model.half()

        logger.info("CrossEncoderReranker ready.")

    # ------------------------------------------------------------------
    # Public reranking API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int,
    ) -> List[RetrievalResult]:
        """Score and re-order *candidates* with respect to *query*.

        The scoring runs in a thread-pool executor so the async event loop
        is not blocked during model inference.

        Args:
            query:      Natural-language query string.
            candidates: First-stage retrieval results to rerank.
            top_n:      Return only the top-n passages after reranking.

        Returns:
            Re-ranked list of :class:`~retrieval.interfaces.base.RetrievalResult`
            with updated ``score`` and ``rank`` fields (length ≤ top_n).
        """
        if not candidates:
            return []

        passages = [r.text for r in candidates]

        loop = asyncio.get_event_loop()
        scores: List[float] = await loop.run_in_executor(
            _INFERENCE_EXECUTOR,
            self._score_batch,
            query,
            passages,
        )

        # Pair each candidate with its new score and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (candidate, score) in enumerate(scored[:top_n], start=1):
            results.append(
                RetrievalResult(
                    passage_id=candidate.passage_id,
                    text=candidate.text,
                    title=candidate.title,
                    score=score,
                    rank=rank,
                    metadata=candidate.metadata,
                )
            )

        logger.debug(
            "CrossEncoderReranker: scored %d candidates, returned %d (top_n=%d).",
            len(candidates),
            len(results),
            top_n,
        )
        return results

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_batch(self, query: str, passages: List[str]) -> List[float]:
        """Score all (query, passage) pairs in mini-batches.

        Runs synchronously – call via :func:`asyncio.run_in_executor`.

        Args:
            query:    Query string.
            passages: List of passage texts.

        Returns:
            List of float scores (one per passage), sigmoid-normalised to
            (0, 1).
        """
        all_scores: List[float] = []
        n_batches = (len(passages) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in range(n_batches):
                batch_passages = passages[i * self.batch_size : (i + 1) * self.batch_size]
                pairs = [(query, p) for p in batch_passages]

                encoding = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                # Move tensors to the right device
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                if self.use_fp16:
                    with torch.autocast(device_type=self.device.split(":")[0]):
                        logits = self._model(**encoding).logits
                else:
                    logits = self._model(**encoding).logits

                # Binary classification: take score for class 1 (relevant)
                # or apply sigmoid to single logit
                if logits.shape[-1] == 1:
                    scores_tensor = torch.sigmoid(logits.squeeze(-1))
                elif logits.shape[-1] == 2:
                    scores_tensor = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    # Multi-class – use the maximum logit as a proxy score
                    scores_tensor = torch.sigmoid(logits.max(dim=-1).values)

                all_scores.extend(scores_tensor.float().cpu().tolist())

        return all_scores

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if the model is loaded and the device is available."""
        try:
            loop = asyncio.get_event_loop()
            is_ok: bool = await loop.run_in_executor(
                _INFERENCE_EXECUTOR, self._probe_model
            )
            return is_ok
        except Exception as exc:
            logger.error("CrossEncoderReranker health check failed: %s", exc)
            return False

    def _probe_model(self) -> bool:
        """Run a tiny forward pass to confirm the model is functional."""
        dummy_query = "health check"
        dummy_passage = "This is a test."
        try:
            _ = self._score_batch(dummy_query, [dummy_passage])
            return True
        except Exception as exc:
            logger.error("CrossEncoderReranker probe failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model and tokeniser to *path* (a directory).

        Args:
            path: Directory to write model artefacts into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        self._tokenizer.save_pretrained(str(path))
        logger.info("CrossEncoderReranker saved to '%s'.", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        use_fp16: bool = True,
    ) -> "CrossEncoderReranker":
        """Load a previously saved reranker.

        Args:
            path:       Directory saved by :meth:`save`.
            device:     Torch device (auto-detected when ``None``).
            max_length: Token length cap.
            batch_size: Inference batch size.
            use_fp16:   Mixed-precision inference.

        Returns:
            Loaded :class:`CrossEncoderReranker`.
        """
        return cls(
            model_name_or_path=str(path),
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            use_fp16=use_fp16,
        )

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model='{self.model_name_or_path}', "
            f"device='{self.device}', max_length={self.max_length}, "
            f"batch_size={self.batch_size})"
        )
