# Training Guide

This document covers everything needed to train the bi-encoder and cross-encoder components of VectorLift from scratch or fine-tune existing checkpoints on MS MARCO.

---

## Overview

VectorLift uses a two-model architecture:

1. **Bi-encoder** — encodes queries and passages independently into dense vectors; used for first-stage ANN retrieval (fast, approximate)
2. **Cross-encoder** — encodes (query, passage) pairs jointly; used for reranking top-N candidates (slower, high precision)

Both models are optional: if no fine-tuned checkpoint is found, VectorLift falls back to the pretrained HuggingFace models specified in `.env`. Fine-tuning on MS MARCO data improves NDCG@10 significantly over the base pretrained checkpoints.

---

## Environment Setup

```bash
# Activate your virtual environment
source .venv/bin/activate

# Verify GPU availability (optional but recommended for training)
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU extras if needed
pip install -e ".[gpu]"
```

Training on CPU is supported but slow for `small` or `full` dataset modes. The `dev` mode (5K triplets) trains in under a minute on CPU and is suitable for verifying the pipeline works end-to-end.

---

## Dataset Preparation

MS MARCO data is downloaded automatically from HuggingFace on first use via the `datasets` library. No manual download is required.

```bash
# Pre-download and cache the dataset (optional, avoids download during training)
python -c "
from pipelines.ingestion.msmarco import MSMARCODataset
ds = MSMARCODataset()
ds.load_training_triplets(max_samples=1000)
print('Dataset cached.')
"
```

**Dataset structure**: Each training example is a triplet `(query, positive_passage, negative_passage)`. MS MARCO provides one positive and one (random) negative per training query.

**Dataset mode selection**:

```bash
# Set in .env or override per-command
DATASET_MODE=small  # 200K triplets, ~15 min on a single GPU
DATASET_MODE=full   # 503K triplets, ~45 min on a single GPU
```

---

## Bi-Encoder Training

### What It Does

`apps/trainer/train_biencoder.py` fine-tunes a `SentenceTransformer` model using `MultipleNegativesRankingLoss` (MNRL). MNRL treats all other positives in the same batch as hard negatives — for a batch of size N, each query has 1 explicit positive and N-1 in-batch negatives. This is efficient and effective: larger batches provide harder negatives.

The training objective is:

```
L = CrossEntropy(cosine_sim(q, p+), cosine_sim(q, p-_1), ..., cosine_sim(q, p-_{N-1}))
```

An `InformationRetrievalEvaluator` runs on a 200-query dev sample every `--eval-steps` steps, reporting NDCG@10 and saving checkpoints.

### Command Reference

```bash
python -m apps.trainer.train_biencoder \
  --model sentence-transformers/msmarco-distilbert-base-tas-b \
  --dataset-mode small \
  --epochs 3 \
  --batch-size 64 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --output-dir models/biencoder \
  --device cuda \
  --eval-steps 500 \
  --checkpoint-steps 1000 \
  --seed 42
```

Or via Make:

```bash
make train-biencoder
# Uses defaults from Makefile: data/training → models/biencoder, 3 epochs, batch 32
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `all-MiniLM-L6-v2` | HuggingFace model ID or path to a local SentenceTransformer |
| `--dataset-mode` | `dev` | `dev` / `small` / `full` |
| `--epochs` | `3` | Number of full passes over the training data |
| `--batch-size` | `64` | Per-device batch size; larger is better for MNRL (more in-batch negatives) |
| `--lr` | `2e-5` | Peak learning rate (AdamW optimizer) |
| `--warmup-ratio` | `0.1` | Fraction of total steps used for linear LR warmup |
| `--fp16` | off | Enable mixed-precision training (requires CUDA) |
| `--use-hard-negatives` | off | Use pre-mined hard negatives instead of MS MARCO random negatives |
| `--max-samples` | unlimited | Limit training samples (useful for quick experiments) |
| `--eval-steps` | `500` | Evaluate on dev NDCG@10 every N optimizer steps |
| `--checkpoint-steps` | `1000` | Save model checkpoint every N steps |
| `--device` | `cpu` | `cpu` / `cuda` / `mps` (Apple Silicon) |
| `--seed` | `42` | Random seed for reproducibility |
| `--output-dir` | `runs/biencoder` | Directory for model, checkpoints, and training logs |

### Output Files

```
models/biencoder/
├── final_model/           # SentenceTransformer format (config + weights)
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── pytorch_model.bin
│   └── sentence_bert_config.json
├── checkpoints/           # Intermediate checkpoints
│   ├── step_0000500/
│   ├── step_0001000/
│   └── ...
├── training_config.json   # All hyperparameters used
└── training_metrics.json  # NDCG@10 at each eval step
```

### Hyperparameter Recommendations

| Setting | Recommendation | Notes |
|---|---|---|
| Model | `msmarco-distilbert-base-tas-b` | Best NDCG/latency tradeoff for MS MARCO |
| Batch size | 64–128 | Larger batches = harder in-batch negatives = better training signal |
| Learning rate | 2e-5 | Standard for fine-tuning transformers; reduce to 1e-5 for very small datasets |
| Warmup ratio | 0.1 | 10% warmup is usually sufficient |
| Epochs | 2–3 | More epochs can overfit on `small`; 1 epoch on `full` is often sufficient |
| FP16 | Yes (if CUDA) | ~2x speedup, negligible accuracy loss |

### Monitoring Training

The evaluator logs NDCG@10 every `--eval-steps` steps. To plot the learning curve:

```python
import json
import matplotlib.pyplot as plt

with open("models/biencoder/training_metrics.json") as f:
    metrics = json.load(f)

steps = [m["steps"] for m in metrics]
ndcg = [m["ndcg@10"] for m in metrics]

plt.plot(steps, ndcg)
plt.xlabel("Steps")
plt.ylabel("NDCG@10 (dev)")
plt.title("Bi-encoder Training Progress")
plt.savefig("training_curve.png")
```

---

## Cross-Encoder Training

### What It Does

`apps/trainer/train_reranker.py` fine-tunes a cross-encoder (jointly-encoding query-passage model) as a binary classifier: positive pairs (query, relevant_passage) get label 1; negative pairs get label 0. This is a standard binary cross-entropy classification fine-tuning.

At inference, the model outputs a logit score for each (query, passage) pair. Higher logit = more relevant.

### Command Reference

```bash
python -m apps.trainer.train_reranker \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --dataset-mode small \
  --epochs 2 \
  --batch-size 32 \
  --output-dir models/reranker
```

Or via Make:

```bash
make train-reranker
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model ID |
| `--dataset-mode` | `dev` | `dev` / `small` / `full` |
| `--epochs` | `2` | Training epochs |
| `--batch-size` | `32` | Batch size (cross-encoder is memory-intensive; reduce if OOM) |
| `--lr` | `2e-5` | Peak learning rate |
| `--output-dir` | `runs/reranker` | Output directory |
| `--device` | `cpu` | Device |

### Hyperparameter Recommendations

| Setting | Recommendation | Notes |
|---|---|---|
| Model | `ms-marco-MiniLM-L-6-v2` | Fastest strong cross-encoder for MS MARCO |
| Batch size | 16–32 | Cross-encoders use more memory than bi-encoders |
| Epochs | 1–2 | The pretrained cross-encoder is already near-optimal on MS MARCO |
| LR | 2e-5 | Standard |

---

## Hard Negative Mining

### Why Hard Negatives?

The standard MS MARCO training triplets use **random negatives** — passages randomly sampled from the corpus that happen to not be relevant. These are easy negatives: the model distinguishes them from positives quickly. To improve the model further, we need **hard negatives** — passages that are retrieved by the current model for a query but are not actually relevant. These confuse the model and force it to learn more fine-grained distinctions.

Hard negative mining is a two-stage curriculum learning process:
1. Train Stage 1: fine-tune on random negatives → get a reasonable base model
2. Mine: use the Stage 1 model to retrieve top-k candidates per query, filter out positives, sample hard negatives
3. Train Stage 2: fine-tune Stage 1 model on hard negatives → improve discrimination

### Mining Hard Negatives

```bash
python -m apps.trainer.hard_negative_mining \
  --model models/biencoder/final_model \
  --dataset-mode small \
  --output data/hard_negatives.jsonl \
  --top-k 30 \
  --max-queries 50000
```

**What happens:**
1. Load queries and positives from MS MARCO training triplets
2. For each query, retrieve top-`k` passages using the current bi-encoder (brute-force cosine similarity over positives; in production, replace with Qdrant/FAISS)
3. Filter out known positives (by ID or text overlap)
4. Randomly sample one hard negative from the remaining candidates
5. Save `(query, positive, hard_negative)` triplets to JSONL

**Output format** (`data/hard_negatives.jsonl`):

```json
{"query": "what causes global warming", "positive": "Greenhouse gases trap heat...", "negative": "The sun has always driven..."}
{"query": "best treatment for diabetes", "positive": "Insulin therapy remains...", "negative": "Diet and exercise can..."}
```

### Stage 2 Training with Hard Negatives

```bash
python -m apps.trainer.train_biencoder \
  --model models/biencoder/final_model \
  --use-hard-negatives \
  --dataset-mode small \
  --epochs 1 \
  --batch-size 64 \
  --lr 1e-5 \
  --output-dir models/biencoder_v2
```

Notes:
- Use a lower learning rate for Stage 2 (1e-5 instead of 2e-5) to avoid catastrophic forgetting
- 1 epoch is usually sufficient — hard negatives are much more informative than random ones
- Compare Stage 1 vs Stage 2 with `make evaluate-all` to quantify the improvement

---

## Using Your Trained Models

After training, update `.env` to point to your checkpoints:

```bash
# Bi-encoder
BIENCODER_MODEL_PATH=models/biencoder/final_model

# Cross-encoder
CROSSENCODER_MODEL_PATH=models/reranker/final_model
```

Restart the API service to load the new models:

```bash
docker compose restart api
# or, if running locally:
make run-api
```

Verify via the model-info endpoint:

```bash
curl http://localhost:8000/model-info
```

---

## Troubleshooting

### Out of Memory (CUDA OOM)

- Reduce `--batch-size` (try 32, 16, 8)
- Enable `--fp16` to halve memory usage
- For cross-encoder training, reduce `--crossencoder-max-length` from 512 to 256 (small accuracy loss)

### Training is Very Slow

- Check device: `python -c "import torch; print(torch.cuda.is_available())"`
- Switch from `--device cpu` to `--device cuda` or `--device mps`
- Reduce `--dataset-mode` to `small` for initial experiments

### Dev Evaluator Shows No Improvement

- Check that the data pipeline is loading actual MS MARCO qrels (not empty)
- Ensure `--eval-steps` is not set too high relative to total steps
- For `dev` mode training (5K triplets), 3 epochs × 78 steps/epoch = 234 total steps; set `--eval-steps 50`

### Loss Not Decreasing

- Learning rate may be too high — try 1e-5
- Check that the DataLoader is shuffling (`shuffle=True` is set in the training script by default)
- Verify that positive and negative passages are not identical (text overlap check in hard negative mining)
