"""
Cross-encoder reranker training script for VectorLift.

Trains a sentence-transformers CrossEncoder for passage reranking on
MS MARCO binary relevance labels.

Usage:
    python -m apps.trainer.train_reranker \\
        --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \\
        --dataset-mode dev \\
        --epochs 3 \\
        --batch-size 32 \\
        --output-dir runs/reranker
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional, Tuple

import numpy as np
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(name="train-reranker", add_completion=False, pretty_exceptions_enable=False)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def main(
    model: Annotated[str, typer.Option("--model", help="HuggingFace model ID or local path")] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    dataset_mode: Annotated[str, typer.Option("--dataset-mode", help="dev|small|full")] = "dev",
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Training batch size")] = 32,
    lr: Annotated[float, typer.Option("--lr", help="Peak learning rate")] = 7e-6,
    warmup_ratio: Annotated[float, typer.Option("--warmup-ratio")] = 0.1,
    output_dir: Annotated[str, typer.Option("--output-dir")] = "runs/reranker",
    device: Annotated[str, typer.Option("--device", help="cpu|cuda|mps")] = "cpu",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    max_samples: Annotated[Optional[int], typer.Option("--max-samples")] = None,
    checkpoint_steps: Annotated[int, typer.Option("--checkpoint-steps")] = 1000,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 500,
    neg_ratio: Annotated[int, typer.Option("--neg-ratio", help="Negatives per positive")] = 4,
    cache_dir: Annotated[Optional[str], typer.Option("--cache-dir")] = None,
    fp16: Annotated[bool, typer.Option("--fp16/--no-fp16")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Train a cross-encoder reranker on MS MARCO binary labels."""
    _setup_logging(log_level)
    _set_seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Device resolution
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load cross-encoder dependencies
    # ------------------------------------------------------------------
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    except ImportError as exc:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc

    # ------------------------------------------------------------------
    # 2. Load MS MARCO data
    # ------------------------------------------------------------------
    logger.info("Loading MS MARCO data (mode=%s) …", dataset_mode)
    from pipelines.ingestion.msmarco import MSMARCODataset

    ms = MSMARCODataset(cache_dir=cache_dir, seed=seed)
    triplets = ms.load_training_triplets(max_samples=max_samples)
    logger.info("Loaded %d training triplets.", len(triplets))

    # ------------------------------------------------------------------
    # 3. Convert triplets to (sentence_pair, label) format
    # ------------------------------------------------------------------
    train_samples = _build_cross_encoder_samples(triplets, neg_ratio=neg_ratio)
    logger.info(
        "Built %d cross-encoder samples (%d pos + %d neg).",
        len(train_samples),
        sum(1 for _, lbl in train_samples if lbl == 1),
        sum(1 for _, lbl in train_samples if lbl == 0),
    )

    # ------------------------------------------------------------------
    # 4. Dev evaluator
    # ------------------------------------------------------------------
    dev_evaluator = None
    try:
        dev_evaluator = _build_dev_evaluator(ms, n_samples=500)
        logger.info("Dev evaluator ready.")
    except Exception as exc:
        logger.warning("Dev evaluator setup failed: %s. Training without eval.", exc)

    # ------------------------------------------------------------------
    # 5. Load model
    # ------------------------------------------------------------------
    logger.info("Loading cross-encoder model '%s' …", model)
    ce_model = CrossEncoder(
        model,
        num_labels=1,
        device=device,
        default_activation_function=torch.nn.Sigmoid(),
    )

    # ------------------------------------------------------------------
    # 6. Training hyperparameters
    # ------------------------------------------------------------------
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample

    train_examples = [
        InputExample(texts=[q, passage], label=float(label))
        for (q, passage), label in train_samples
    ]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
    )

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    logger.info(
        "Training: epochs=%d  steps_per_epoch=%d  total_steps=%d  warmup=%d",
        epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    # ------------------------------------------------------------------
    # 7. Fit
    # ------------------------------------------------------------------
    metrics_log: List[dict] = []

    def _eval_callback(score: float, epoch: int, steps: int) -> None:
        record = {
            "epoch": epoch,
            "steps": steps,
            "auc": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metrics_log.append(record)
        logger.info("Eval step %d: AUC=%.4f", steps, score)

    training_start = time.perf_counter()

    ce_model.fit(
        train_dataloader=train_dataloader,
        evaluator=dev_evaluator,
        epochs=epochs,
        evaluation_steps=eval_steps if dev_evaluator else 0,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(checkpoint_dir / "best_model"),
        use_amp=fp16,
        show_progress_bar=True,
        callback=_eval_callback if dev_evaluator else None,
    )

    elapsed = time.perf_counter() - training_start
    logger.info("Training complete in %.1fs.", elapsed)

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    final_path = out_path / "final_model"
    ce_model.save(str(final_path))
    logger.info("Final cross-encoder saved to '%s'.", final_path)

    training_config = {
        "model": model,
        "dataset_mode": dataset_mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "fp16": fp16,
        "neg_ratio": neg_ratio,
        "max_samples": max_samples,
        "n_triplets": len(triplets),
        "n_train_samples": len(train_samples),
        "total_steps": total_steps,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path / "training_config.json", "w") as fh:
        json.dump(training_config, fh, indent=2)

    with open(out_path / "training_metrics.json", "w") as fh:
        json.dump(metrics_log, fh, indent=2)

    typer.echo(f"Training complete. Model saved at: {final_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cross_encoder_samples(
    triplets: List[Tuple[str, str, str]],
    neg_ratio: int = 4,
) -> List[Tuple[Tuple[str, str], int]]:
    """Convert (query, positive, negative) triplets to (pair, label) samples.

    Args:
        triplets:  Training triplets.
        neg_ratio: Number of negative examples per positive.

    Returns:
        List of ((query, passage), binary_label) tuples.
    """
    samples: List[Tuple[Tuple[str, str], int]] = []
    for query, positive, negative in triplets:
        samples.append(((query, positive), 1))
        for _ in range(neg_ratio):
            # In a real pipeline, we would mine diverse negatives.
            # Here we repeat the single available negative as a baseline.
            samples.append(((query, negative), 0))
    random.shuffle(samples)
    return samples


def _build_dev_evaluator(
    ds: "MSMARCODataset",  # type: ignore[name-defined]  # noqa: F821
    n_samples: int = 500,
) -> "CEBinaryClassificationEvaluator":  # type: ignore[name-defined]  # noqa: F821
    """Build a dev evaluator from MS MARCO dev triplets."""
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    triplets = ds.load_training_triplets(max_samples=n_samples)
    sentence_pairs: List[List[str]] = []
    labels: List[int] = []
    for query, pos, neg in triplets:
        sentence_pairs.append([query, pos])
        labels.append(1)
        sentence_pairs.append([query, neg])
        labels.append(0)

    return CEBinaryClassificationEvaluator(
        sentence_pairs=sentence_pairs,
        labels=labels,
        name="msmarco-dev",
        show_progress_bar=False,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    app()
