"""
Bi-encoder training script for VectorLift.

Trains a SentenceTransformer bi-encoder with MultipleNegativesRankingLoss
on MS MARCO training triplets.

Usage:
    python -m apps.trainer.train_biencoder \\
        --model "microsoft/MiniLM-L12-H384-uncased" \\
        --dataset-mode dev \\
        --epochs 3 \\
        --batch-size 64 \\
        --output-dir runs/biencoder
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(name="train-biencoder", add_completion=False, pretty_exceptions_enable=False)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def main(
    model: Annotated[str, typer.Option("--model", help="HuggingFace model ID or local path")] = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_mode: Annotated[str, typer.Option("--dataset-mode", help="dev|small|full")] = "dev",
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Training batch size")] = 64,
    lr: Annotated[float, typer.Option("--lr", help="Peak learning rate")] = 2e-5,
    warmup_ratio: Annotated[float, typer.Option("--warmup-ratio", help="Fraction of steps for LR warmup")] = 0.1,
    output_dir: Annotated[str, typer.Option("--output-dir", help="Where to save the model")] = "runs/biencoder",
    device: Annotated[str, typer.Option("--device", help="cpu|cuda|mps")] = "cpu",
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    use_hard_negatives: Annotated[bool, typer.Option("--use-hard-negatives/--no-hard-negatives")] = False,
    max_samples: Annotated[Optional[int], typer.Option("--max-samples", help="Limit training samples")] = None,
    checkpoint_steps: Annotated[int, typer.Option("--checkpoint-steps", help="Save checkpoint every N steps")] = 1000,
    eval_steps: Annotated[int, typer.Option("--eval-steps", help="Evaluate every N steps")] = 500,
    cache_dir: Annotated[Optional[str], typer.Option("--cache-dir", help="Dataset cache directory")] = None,
    fp16: Annotated[bool, typer.Option("--fp16/--no-fp16", help="Mixed precision training")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Train a bi-encoder on MS MARCO using MultipleNegativesRankingLoss."""
    _setup_logging(log_level)
    _set_seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve device
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading MS MARCO training data (mode=%s) …", dataset_mode)
    from pipelines.ingestion.msmarco import MSMARCODataset

    ds = MSMARCODataset(cache_dir=cache_dir, seed=seed)
    triplets = ds.load_training_triplets(max_samples=max_samples)
    logger.info("Loaded %d training triplets.", len(triplets))

    # Hard negative mining: replace random negatives with model-retrieved ones
    if use_hard_negatives:
        logger.info(
            "Hard negatives requested but mining requires a trained model; "
            "using data-supplied negatives for this run. "
            "Run hard_negative_mining.py after first training pass."
        )

    # ------------------------------------------------------------------
    # 2. Build SentenceTransformer model
    # ------------------------------------------------------------------
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses, evaluation
    except ImportError as exc:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc

    logger.info("Loading model '%s' …", model)
    st_model = SentenceTransformer(model, device=device)

    # ------------------------------------------------------------------
    # 3. Prepare InputExample objects and DataLoader
    # ------------------------------------------------------------------
    from torch.utils.data import DataLoader

    train_examples: List[InputExample] = [
        InputExample(texts=[q, pos, neg]) for q, pos, neg in triplets
    ]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 4. Loss function
    # ------------------------------------------------------------------
    loss_fn = losses.MultipleNegativesRankingLoss(model=st_model)

    # ------------------------------------------------------------------
    # 5. Evaluator – NDCG@10 on a sample of dev queries
    # ------------------------------------------------------------------
    dev_evaluator = None
    try:
        dev_evaluator = _build_dev_evaluator(ds, st_model, device, n_queries=200, k=10)
    except Exception as exc:
        logger.warning("Dev evaluator setup failed (%s); training without eval.", exc)

    # ------------------------------------------------------------------
    # 6. Compute training steps
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    logger.info(
        "Training: epochs=%d  steps_per_epoch=%d  total_steps=%d  warmup=%d",
        epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    # ------------------------------------------------------------------
    # 7. Training
    # ------------------------------------------------------------------
    metrics_log: List[dict] = []
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    class _MetricsCallback:
        """Minimal callback to capture evaluator scores at each eval step."""
        def __call__(self, score: float, epoch: int, steps: int) -> None:
            record = {
                "epoch": epoch,
                "steps": steps,
                "ndcg@10": score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            metrics_log.append(record)
            logger.info("Eval step %d: NDCG@10=%.4f", steps, score)

            # Save checkpoint
            ckpt_path = checkpoint_dir / f"step_{steps:07d}"
            st_model.save(str(ckpt_path))
            logger.info("Checkpoint saved to '%s'.", ckpt_path)

    callback = _MetricsCallback() if dev_evaluator else None

    training_start = time.perf_counter()

    st_model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=dev_evaluator,
        evaluation_steps=eval_steps if dev_evaluator else 0,
        checkpoint_path=str(checkpoint_dir) if checkpoint_steps > 0 else None,
        checkpoint_save_steps=checkpoint_steps,
        optimizer_params={"lr": lr},
        use_amp=fp16,
        show_progress_bar=True,
        callback=callback,
    )

    elapsed = time.perf_counter() - training_start
    logger.info("Training complete in %.1fs.", elapsed)

    # ------------------------------------------------------------------
    # 8. Save final model and training config
    # ------------------------------------------------------------------
    final_model_path = out_path / "final_model"
    st_model.save(str(final_model_path))
    logger.info("Final model saved to '%s'.", final_model_path)

    training_config = {
        "model": model,
        "dataset_mode": dataset_mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "fp16": fp16,
        "use_hard_negatives": use_hard_negatives,
        "max_samples": max_samples,
        "n_triplets": len(triplets),
        "total_steps": total_steps,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path / "training_config.json", "w") as fh:
        json.dump(training_config, fh, indent=2)

    with open(out_path / "training_metrics.json", "w") as fh:
        json.dump(metrics_log, fh, indent=2)

    logger.info("Training artefacts saved to '%s'.", out_path)
    typer.echo(f"Training complete. Model at: {final_model_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _build_dev_evaluator(
    ds: "MSMARCODataset",  # type: ignore[name-defined]  # noqa: F821
    model: "SentenceTransformer",  # type: ignore[name-defined]  # noqa: F821
    device: str,
    n_queries: int = 200,
    k: int = 10,
) -> "SentenceTransformer":  # actually returns an evaluator
    """Build a simple information retrieval evaluator over a dev sample."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    logger.info("Building dev evaluator with %d queries …", n_queries)

    queries_dict = ds.load_queries("dev")
    qrels = ds.load_qrels("dev")

    # Only keep queries that have qrels
    query_ids = [qid for qid in list(queries_dict.keys())[:n_queries] if qid in qrels][:n_queries]
    queries = {qid: queries_dict[qid] for qid in query_ids}

    # Collect relevant doc IDs
    relevant_doc_ids: set[str] = set()
    for qid in query_ids:
        relevant_doc_ids.update(qrels[qid].keys())

    passages = ds.load_passages("dev")
    relevant_passages = {p["id"]: p["text"] for p in passages if p["id"] in relevant_doc_ids}

    # Build corpus + qrels in evaluator format
    corpus = {pid: {"title": "", "text": text} for pid, text in relevant_passages.items()}
    qrels_eval: dict[str, dict[str, int]] = {
        qid: {did: rel for did, rel in qrels[qid].items() if did in corpus}
        for qid in query_ids
    }

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels_eval,
        name="msmarco-dev",
        ndcg_at_k=[k],
        map_at_k=[k],
        mrr_at_k=[k],
        show_progress_bar=False,
    )
    logger.info(
        "Dev evaluator built: %d queries  %d corpus docs.", len(queries), len(corpus)
    )
    return evaluator


if __name__ == "__main__":
    app()
