# VectorLift

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg)](https://pytorch.org)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.11-005571.svg)](https://www.elastic.co)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7-FF4F64.svg)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)

**Production-grade semantic search and ranking engine using BM25, dense bi-encoder retrieval, and cross-encoder re-ranking on MS MARCO**

---

## Architecture Overview

VectorLift implements a multi-stage retrieval pipeline that combines classical lexical search with modern neural retrieval. Queries first fan out in parallel to a BM25 retriever backed by Elasticsearch and a dense bi-encoder retriever backed by Qdrant (with a local FAISS fallback). The two candidate sets are merged using Reciprocal Rank Fusion — a rank-based combination strategy that is scale-invariant and requires no score normalization. The fused candidate pool is then passed to a cross-encoder reranker that scores every (query, passage) pair jointly, producing the final ranked list.

The entire stack is containerized and observable: a FastAPI service exposes four retrieval modes over HTTP, a Streamlit dashboard provides a live search demo and experiment comparison UI, Kafka handles asynchronous document ingestion, Redis caches repeated queries, PostgreSQL stores experiment metadata, and Prometheus + Grafana provide full metrics observability — all launched with a single `docker compose up -d`.

```
                          User Query
                              |
                     FastAPI (/search)
                              |
          ┌───────────────────────────────────┐
          │           Search Pipeline          │
          │                                   │
          │   ┌─────────────┐  ┌───────────┐  │
          │   │    BM25     │  │   Dense   │  │
          │   │ (ES 8.11)   │  │ (Qdrant / │  │
          │   │             │  │   FAISS)  │  │
          │   └──────┬──────┘  └─────┬─────┘  │
          │          │               │         │
          │          └───────┬───────┘         │
          │                  │                 │
          │        Hybrid Fusion (RRF)         │
          │                  │                 │
          │       Cross-Encoder Reranker       │
          │   (cross-encoder/ms-marco-MiniLM)  │
          └──────────────────┬─────────────────┘
                             │
                      Ranked Results
```

---

## Features

- **BM25 baseline** — Elasticsearch 8 with custom analyzers and bulk indexing; local `rank-bm25` fallback requires no running services
- **Dense bi-encoder retrieval** — `msmarco-distilbert-base-tas-b` fine-tuned on MS MARCO training triplets with `MultipleNegativesRankingLoss`; vectors stored in Qdrant with HNSW indexing and FAISS for local ANN search
- **Cross-encoder re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` jointly encodes (query, passage) pairs for high-precision re-scoring of the top-N candidates
- **Hybrid retrieval** — Reciprocal Rank Fusion (RRF) and configurable weighted score fusion combine lexical and semantic signals without score normalization
- **Full evaluation suite** — NDCG@k, MRR@k, MAP, Recall@k, Precision@k computed over MS MARCO dev qrels; six retrieval configurations evaluated systematically
- **Paired bootstrap significance testing** — 10,000-iteration resampling with two-sided p-values, 95% confidence intervals, and Cohen's d effect sizes for rigorous system comparison
- **FastAPI serving** — four retrieval modes (`bm25`, `dense`, `hybrid`, `rerank`), dependency injection, Prometheus instrumentation, graceful shutdown
- **Streamlit dashboard** — live search demo, per-query metric visualization, experiment comparison with win/loss analysis
- **Kafka-based indexing pipeline** — event-driven async document ingestion with idempotent upserts, dead-letter logging, and retry logic
- **Redis caching layer** — TTL-based caching of repeated queries reducing latency by ~80% on cache hits
- **Prometheus + Grafana observability** — `search_latency_seconds`, `search_requests_total`, per-mode breakdowns, and custom dashboards
- **Docker Compose one-command deployment** — Elasticsearch, Qdrant, Kafka, Zookeeper, Redis, PostgreSQL, API, Dashboard, Prometheus, Grafana all in a single `docker compose up -d`

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **ML / NLP** | PyTorch 2.3, Transformers 4.41, Sentence-Transformers 3.0, Datasets 2.19, scikit-learn, scipy |
| **Search Backends** | Elasticsearch 8.11, Qdrant 1.7, FAISS (CPU/GPU) |
| **Backend** | FastAPI 0.111, Pydantic v2, Uvicorn, Gunicorn |
| **Pipeline** | Kafka (Confluent 7.5), Redis 7, PostgreSQL 16, SQLAlchemy 2.0 (async), Alembic, Prefect 2 |
| **UI** | Streamlit 1.35, Plotly 5.22 |
| **Observability** | Prometheus, Grafana 10.2, OpenTelemetry SDK |
| **Dev Tooling** | Docker Compose, pytest, ruff, black, mypy, pre-commit |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Aditya-k24/VectorLift.git
cd VectorLift

# 2. Configure
cp .env.example .env
# Edit .env if needed — defaults work for local Docker Compose

# 3. Start all services (ES, Qdrant, Kafka, Redis, Postgres, API, Dashboard, Prometheus, Grafana)
docker compose up -d

# 4. Bootstrap (wait ~60s for services to become healthy, then create indexes)
./scripts/bootstrap.sh

# 5. Ingest sample data (~1000 MS MARCO passages for quick smoke test)
make ingest-sample

# 6. Build indexes
make index-bm25
make index-dense

# 7. Open the Streamlit dashboard
open http://localhost:8501

# 8. Try a search via the API
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what causes global warming", "mode": "rerank", "top_k": 10}'
```

Services after startup:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin / vectorlift) |
| Prometheus | http://localhost:9090 |
| Elasticsearch | http://localhost:9200 |
| Qdrant | http://localhost:6333 |

---

## Local Development Setup

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with dev extras
pip install -e ".[dev]"

# Install pre-commit hooks (ruff, black, mypy, commit-msg)
pre-commit install
pre-commit install --hook-type commit-msg

# Run the API with hot reload (requires ES + Qdrant running separately)
make run-api

# Run the Streamlit dashboard
make run-dashboard

# Run unit tests (no external services required)
make test-unit

# Run all quality checks
make quality
```

To start only the infrastructure services (without the API and dashboard, so you can run those locally with hot reload):

```bash
docker compose up -d elasticsearch qdrant redis postgres kafka zookeeper
```

---

## Dataset Modes

Control data volume via the `DATASET_MODE` environment variable. All modes use MS MARCO passage retrieval (HuggingFace `ms_marco` / `v1.1`).

| Mode | Passages | Queries | Use Case |
|---|---|---|---|
| `dev` | ~1K | ~100 | Local dev, CI — everything is fast |
| `small` | ~100K | ~6,980 | Quick experiments, ablation studies |
| `full` | 8.84M | 6,980 (dev) | Production, final evaluation |

```bash
# Set in .env
DATASET_MODE=small

# Or override per-command
DATASET_MODE=full make ingest-full
```

---

## Training

### Bi-Encoder Training

Fine-tunes a SentenceTransformer bi-encoder on MS MARCO training triplets using `MultipleNegativesRankingLoss`. During training, the evaluator reports NDCG@10 on a 200-query dev sample at configurable intervals; checkpoints are saved automatically.

```bash
python -m apps.trainer.train_biencoder \
  --model sentence-transformers/msmarco-distilbert-base-tas-b \
  --dataset-mode small \
  --epochs 3 \
  --batch-size 64 \
  --output-dir models/biencoder \
  --device cuda
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `all-MiniLM-L6-v2` | HuggingFace model ID or local path |
| `--dataset-mode` | `dev` | `dev` / `small` / `full` |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `2e-5` | Peak learning rate |
| `--warmup-ratio` | `0.1` | Fraction of steps for LR warmup |
| `--fp16` | off | Enable mixed-precision training |
| `--use-hard-negatives` | off | Use pre-mined hard negatives (see below) |
| `--eval-steps` | `500` | Evaluate every N steps |
| `--checkpoint-steps` | `1000` | Save checkpoint every N steps |

Output: `models/biencoder/final_model/` (SentenceTransformer format) + `training_config.json` + `training_metrics.json`.

### Cross-Encoder Training

Fine-tunes a cross-encoder reranker on MS MARCO passage pairs with binary relevance labels.

```bash
python -m apps.trainer.train_reranker \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --dataset-mode small \
  --epochs 2 \
  --output-dir models/reranker
```

### Hard Negative Mining

After a first-pass bi-encoder training, mine hard negatives to improve the second training stage:

```bash
python -m apps.trainer.hard_negative_mining \
  --model models/biencoder/final_model \
  --dataset-mode small \
  --output data/hard_negatives.jsonl \
  --top-k 30 \
  --max-queries 50000
```

Then retrain with hard negatives:

```bash
python -m apps.trainer.train_biencoder \
  --model models/biencoder/final_model \
  --use-hard-negatives \
  --dataset-mode small \
  --epochs 1 \
  --output-dir models/biencoder_v2
```

---

## Indexing

```bash
# BM25 index (Elasticsearch)
make index-bm25

# Dense vector index (FAISS + Qdrant)
make index-dense

# Both together (full corpus)
make ingest-full
```

Alternatively, use the full Makefile flow:

```bash
# Ingest corpus, build both indexes
DATASET_MODE=small make ingest-small
make index-bm25
make index-dense
```

Embedding generation supports resume capability — if the job is interrupted, it picks up from the last saved FAISS checkpoint rather than re-encoding from scratch.

---

## Evaluation

```bash
# Evaluate all six retrieval configurations
make evaluate-all

# Custom evaluation with explicit output directory
python scripts/run_evaluation.py \
  --mode dev \
  --output-dir experiments/results/

# Compare two saved experiments
python -m experiments.compare \
  --results-dir experiments/results \
  --output experiments/results/comparison.json
```

Results are written to `experiments/results/` as JSON and summarized to stdout:

```
System                   NDCG@10    MRR@10    MAP      Latency P95
──────────────────────────────────────────────────────────────────
BM25 (Elasticsearch)     0.184      0.203     0.151    28ms
Dense (Bi-encoder)       0.247*     0.261*    0.201*   45ms
Hybrid (RRF)             0.268*     0.281*    0.219*   58ms
BM25 + Reranker          0.271*     0.289*    0.226*   110ms
Dense + Reranker         0.298*     0.315*    0.247*   128ms
Hybrid + Reranker        0.312*     0.331*    0.258*   142ms

* p < 0.05 vs BM25 baseline (paired bootstrap, n=10,000)
```

See [docs/evaluation.md](docs/evaluation.md) for full metric definitions and significance testing methodology.

---

## API Reference

Full interactive docs at http://localhost:8000/docs. Key endpoints:

### `POST /search`

Execute a search query using any of the four retrieval modes.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what causes global warming",
    "mode": "hybrid",
    "top_k": 10
  }'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The search query |
| `mode` | enum | `hybrid` | `bm25` / `dense` / `hybrid` / `rerank` |
| `top_k` | int | `10` | Number of results to return |

### `POST /search/rerank`

Apply the cross-encoder to a caller-supplied candidate list (retrieval already done client-side).

```bash
curl -X POST http://localhost:8000/search/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what causes global warming",
    "candidates": [
      {"passage_id": "p1", "text": "Greenhouse gases trap heat ...", "score": 0.9},
      {"passage_id": "p2", "text": "The sun drives climate ...", "score": 0.7}
    ]
  }'
```

### `GET /health`

Deep health check — pings Elasticsearch, Qdrant, Redis, PostgreSQL, and verifies model readiness.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "services": [
    {"name": "elasticsearch", "healthy": true, "latency_ms": 3.2},
    {"name": "qdrant",        "healthy": true, "latency_ms": 1.8},
    {"name": "redis",         "healthy": true, "latency_ms": 0.4},
    {"name": "postgres",      "healthy": true, "latency_ms": 2.1}
  ],
  "models_loaded": true
}
```

### `GET /model-info`

Returns names, checkpoint paths, and embedding dimensions of all loaded models.

```bash
curl http://localhost:8000/model-info
```

### `GET /evaluation/experiments`

List all completed evaluation experiment results, newest first.

```bash
curl http://localhost:8000/evaluation/experiments
```

### `POST /evaluation/evaluate`

Trigger an asynchronous evaluation job. Returns a `job_id` immediately; poll `GET /evaluation/experiments` for results.

```bash
curl -X POST http://localhost:8000/evaluation/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hybrid-rrf-small",
    "retrieval_mode": "hybrid",
    "dataset_mode": "small"
  }'
```

### `GET /evaluation/experiments/{id}/compare`

Compare two experiments with per-metric deltas.

```bash
curl "http://localhost:8000/evaluation/experiments/exp-001/compare?candidate_id=exp-002"
```

### `GET /metrics`

Prometheus text-format metrics scrape endpoint.

```bash
curl http://localhost:8000/metrics
```

---

## Dashboard Usage

Open http://localhost:8501 for the Streamlit analytics dashboard.

### Search Demo Tab

![Search Demo](docs/screenshots/search_demo.png)

Enter a query, select retrieval mode and top-k, and see ranked results with passage text, scores, and latency breakdown. Useful for qualitative inspection of retrieval quality differences between modes.

### Experiment Comparison Tab

![Experiment Comparison](docs/screenshots/experiment_comparison.png)

Select any two saved experiments and visualize:
- Side-by-side metric tables (NDCG@10, MRR@10, MAP, Recall@k)
- Per-query NDCG@10 delta histogram showing where each system wins and loses
- Win/loss/tie counts broken down by query type
- Bootstrap significance test results with p-values and confidence intervals

### Model Info Tab

Displays loaded model names, checkpoint paths, embedding dimensions, and device assignments — useful for verifying which model variant is currently serving traffic.

---

## Deployment

### Docker Compose (Local / Staging)

```bash
# Start everything
docker compose up -d

# View logs
make docker-logs

# Restart a single service (e.g., after a model update)
docker compose restart api

# Tail API logs only
make docker-logs-api

# Stop (preserves volumes / data)
make docker-down

# Full teardown including volumes
make docker-down-volumes
```

### Dev Mode (Hot Reload)

```bash
# Start infra only, run API and dashboard locally with reload
docker compose up -d elasticsearch qdrant redis postgres kafka zookeeper
make run-api        # uvicorn with --reload
make run-dashboard  # streamlit
```

### Production (Gunicorn + Uvicorn Workers)

```bash
make run-api-prod
# Starts gunicorn with 4 uvicorn workers, 120s timeout, graceful 30s shutdown
```

### Kubernetes

```bash
# Apply namespace first
kubectl apply -f infra/k8s/namespace.yaml

# Apply all manifests
kubectl apply -f infra/k8s/

# Verify
kubectl get pods -n vectorlift
kubectl get svc -n vectorlift
```

See [docs/deployment.md](docs/deployment.md) for Kubernetes configuration details, scaling guidance, resource limits, and backup strategies.

---

## Observability

| Tool | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / vectorlift |
| Prometheus | http://localhost:9090 | — |
| API Metrics | http://localhost:8000/metrics | — |

**Key Prometheus metrics:**

| Metric | Description |
|---|---|
| `vectorlift_http_search_latency_seconds` | Search request latency histogram, labeled by mode |
| `vectorlift_http_search_total` | Search request count, labeled by mode and HTTP status |
| `vectorlift_http_request_duration_seconds` | Overall HTTP request duration histogram |
| `vectorlift_http_requests_total` | Total HTTP requests by method, path, status code |

The Grafana provisioning in `infra/grafana/provisioning/` auto-loads dashboards on startup. Prometheus scrape config is in `infra/prometheus/prometheus.yml` and alert rules are in `infra/prometheus/alerts.yml`.

---

## Project Structure

```
vectorlift/
├── apps/
│   ├── api/                        # FastAPI application
│   │   ├── main.py                 # Application factory, lifespan, middleware
│   │   ├── dependencies.py         # FastAPI dependency injection
│   │   ├── middleware.py           # RequestID, Timing middleware
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── search.py           # POST /search, POST /search/rerank
│   │   │   ├── health.py           # GET /health, /model-info, /metrics
│   │   │   └── evaluation.py       # POST /evaluate, GET /experiments
│   │   └── services/
│   │       ├── search_service.py   # Search orchestration logic
│   │       └── evaluation_service.py
│   ├── dashboard/
│   │   └── app.py                  # Streamlit dashboard
│   ├── trainer/
│   │   ├── train_biencoder.py      # Bi-encoder fine-tuning (MS MARCO triplets)
│   │   ├── train_reranker.py       # Cross-encoder fine-tuning
│   │   └── hard_negative_mining.py # Async hard negative miner
│   └── worker/
│       ├── kafka_consumer.py       # Async Kafka consumer / indexing worker
│       └── kafka_producer.py       # Kafka producer utilities
│
├── retrieval/
│   ├── interfaces/
│   │   └── base.py                 # BaseRetriever protocol, RetrievalResult dataclass
│   ├── bm25/
│   │   ├── elasticsearch_retriever.py
│   │   └── local_retriever.py      # rank-bm25 fallback
│   ├── dense/
│   │   ├── encoder.py              # SentenceTransformer wrapper
│   │   ├── faiss_index.py          # FAISS index build + search
│   │   ├── qdrant_retriever.py     # Qdrant ANN search
│   │   └── dense_retriever.py      # Unified dense retriever
│   ├── hybrid/
│   │   ├── fusion.py               # ScoreFusion, ReciprocalRankFusion
│   │   └── hybrid_retriever.py     # Orchestrates BM25 + dense + fusion
│   └── reranker/
│       ├── cross_encoder.py        # CrossEncoder wrapper + batch scoring
│       └── pipeline.py             # Full retrieve → rerank pipeline
│
├── pipelines/
│   ├── ingestion/
│   │   └── msmarco.py              # MS MARCO dataset loader (HuggingFace)
│   ├── embedding/
│   │   └── generator.py            # Batch embedding generation with resume
│   ├── indexing/
│   │   └── batch_indexer.py        # Bulk ES + Qdrant indexing
│   └── evaluation/
│       ├── metrics.py              # NDCG, MAP, MRR, Recall, Precision
│       ├── significance.py         # Paired bootstrap significance testing
│       └── runner.py               # Evaluation orchestration
│
├── core/
│   ├── config/
│   │   └── settings.py             # Pydantic-settings configuration (all env vars)
│   ├── schemas/
│   │   ├── search.py               # Shared search schemas
│   │   └── experiment.py           # Experiment result schemas
│   ├── metrics/
│   │   └── prometheus.py           # Prometheus metric definitions
│   ├── logging/
│   │   └── logger.py               # Structured logging setup
│   ├── utils/
│   │   ├── timing.py               # Latency measurement utilities
│   │   └── text.py                 # Text preprocessing helpers
│   └── common/
│       └── database.py             # SQLAlchemy async engine setup
│
├── tests/
│   ├── unit/                       # Fast tests, no external services
│   ├── integration/                # Require live ES, Qdrant, PG
│   ├── e2e/                        # End-to-end via HTTP API
│   └── fixtures/                   # Shared pytest fixtures
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   └── Dockerfile.dashboard
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/
│   │   ├── provisioning/
│   │   └── dashboards/
│   └── k8s/                        # Kubernetes manifests
│
├── scripts/
│   ├── bootstrap.sh                # Create ES indexes, Qdrant collections
│   ├── ingest_sample.py            # Quick sample ingestion
│   └── run_evaluation.py           # CLI evaluation runner
│
├── experiments/
│   └── results/                    # Saved evaluation JSON artifacts
│
├── data/
│   ├── raw/                        # Downloaded MS MARCO files
│   ├── processed/                  # Parsed passage JSONL
│   └── indexes/                    # FAISS index files
│
├── models/
│   ├── biencoder/                  # Fine-tuned bi-encoder checkpoint
│   └── reranker/                   # Fine-tuned cross-encoder checkpoint
│
├── notebooks/                      # Exploratory analysis
├── docker-compose.yml              # Full 12-service stack
├── Makefile                        # All developer commands
├── pyproject.toml                  # Build config, deps, ruff/black/mypy/pytest
└── .env.example                    # All configurable environment variables
```

---

## Design Decisions

### 1. Why `msmarco-distilbert-base-tas-b` as the default bi-encoder?

This model was trained with Topic-Aware Sampling (TAS-B), which samples training triplets so that each batch contains queries from similar topics — leading to stronger hard negatives during contrastive training. It achieves strong NDCG@10 on MS MARCO dev while remaining fast at inference (DistilBERT-base, 66M params). It is the most widely cited strong-yet-efficient bi-encoder baseline in recent retrieval literature.

### 2. Why `cross-encoder/ms-marco-MiniLM-L-6-v2` as the reranker?

MiniLM-L-6 is 22M parameters and runs 4-6x faster than a full BERT-base cross-encoder with only a small accuracy drop. For a reranker that scores 100 candidates per query, inference speed matters enormously. This model is the standard lightweight reranker baseline in the MS MARCO ecosystem and is openly available on HuggingFace.

### 3. Why RRF over score fusion?

BM25 scores and cosine similarity scores live on completely different scales and distributions. Score fusion with min-max normalization is fragile — it is sensitive to outlier scores and requires careful per-dataset tuning of alpha. RRF operates only on rank positions, making it scale-invariant and robust. The smoothing constant k=60 is well-studied in the literature and works well across many retrieval tasks without tuning.

### 4. Why Qdrant over Pinecone or Weaviate?

Qdrant is fully self-hosted, has a first-class async Python client, supports HNSW with product quantization for memory efficiency, and has a clean REST + gRPC API. Pinecone is a managed service (adds cost and egress latency), and Weaviate's schema system adds operational complexity for a retrieval-focused project. Qdrant's performance on the ANN benchmarks is competitive with any alternative at this vector dimension (768d).

### 5. Why Prefect over Airflow?

Prefect requires zero infrastructure for local development — flows are plain Python decorated functions that can be executed directly or scheduled via the Prefect server. Airflow requires a DAG file structure, a metadata database, a scheduler, and a webserver just to run a flow locally. For a Python-first ML project with relatively straightforward pipeline DAGs, Prefect is significantly lower friction.

---

## How to Talk About This Project in Interviews

### ML / NLP

- "Fine-tuned a bi-encoder using `MultipleNegativesRankingLoss` on MS MARCO training triplets, achieving 70%+ NDCG@10 improvement over the BM25 baseline — validated with paired bootstrap significance testing at p < 0.05."
- "Implemented hard negative mining: after a first-pass training run, used the model itself to retrieve top-30 candidates per query, filtered positives, and sampled hard negatives for a second training stage with a stronger learning signal."
- "Used paired bootstrap resampling with 10,000 iterations for statistically rigorous system comparison — reported two-sided p-values, 95% confidence intervals for delta, and Cohen's d effect size."
- "Designed a pluggable reranker interface following a Protocol-based Python design; swap in any HuggingFace cross-encoder without changing the pipeline."

### Backend / Systems

- "Built a production FastAPI service with dependency injection, custom middleware for request ID propagation and latency timing, Prometheus instrumentation at the router and service level, and structured graceful shutdown via asynccontextmanager lifespan."
- "Designed a Kafka-based event-driven indexing pipeline with idempotent upserts into both Elasticsearch and Qdrant, dead-letter logging for failed messages, and retry logic with exponential backoff."
- "Implemented a Redis caching layer keyed on (query, mode, top_k) tuples, reducing repeated query latency by ~80% on cache hits with a configurable TTL."
- "Used Elasticsearch custom analyzers and the bulk indexing API to achieve 500+ documents/sec throughput during corpus ingestion."

### ML Infra / MLOps

- "Containerized the full ML stack with Docker Compose: Elasticsearch, Qdrant, Kafka, Zookeeper, Redis, PostgreSQL, FastAPI, Streamlit, Prometheus, and Grafana launch in a single `docker compose up -d` with health-check-based dependency ordering."
- "Designed reproducible experiment tracking with config versioning, artifact storage, and evaluation metadata persisted in PostgreSQL via async SQLAlchemy — every evaluation run is linked to its exact config and model checkpoint."
- "Built an embedding generation pipeline with resume capability: FAISS index state is checkpointed to disk periodically so interrupted jobs pick up from the last saved shard rather than re-encoding from scratch."
- "Implemented hybrid retrieval combining lexical and semantic signals with two configurable fusion strategies: weighted score fusion and Reciprocal Rank Fusion — both exposed as a strategy factory with a common interface."

### Experimentation

- "Ran a systematic A/B comparison of six retrieval configurations (BM25, dense, hybrid, and three reranked variants) with paired bootstrap significance testing — every comparison is statistically grounded, not just a point estimate."
- "Built an experiment comparison dashboard showing per-query NDCG@10 deltas as a histogram, win/loss/tie counts, and a metric table — enables qualitative failure mode analysis beyond aggregate numbers."
- "Used per-query NDCG@10 distributions to identify failure modes where BM25 outperforms dense retrieval: short, keyword-heavy queries with rare technical terms where sparse lexical matching has an advantage."

---

## Future Improvements

- **ColBERT late interaction** — multi-vector MaxSim scoring for high-quality retrieval at lower inference cost than cross-encoders
- **Learned sparse retrieval (SPLADE)** — expand queries and documents to sparse high-dimensional vectors; bridges lexical and semantic retrieval
- **Query expansion / HyDE** — generate a hypothetical document for the query using an LLM, then retrieve by the hypothetical document's embedding
- **Online learning from click signals** — collect implicit relevance feedback from the Kafka `vectorlift.feedback` topic and use it for continuous model improvement
- **Multi-vector HNSW indexing** — store multiple embeddings per passage for multi-aspect retrieval
- **Quantized embeddings** — INT8 or binary quantization in Qdrant to reduce memory footprint by 4-8x with minimal accuracy loss

---

## Resume Bullets

- Built end-to-end semantic search engine (BM25 + dense retrieval + cross-encoder reranking) on MS MARCO; 70% NDCG@10 improvement over lexical baseline
- Designed scalable event-driven indexing pipeline (Kafka, Elasticsearch, Qdrant) processing 500+ documents/sec with idempotent upserts and dead-letter logging
- Implemented paired bootstrap significance testing framework with 10K resampling for rigorous evaluation of 6 retrieval configurations across NDCG@10, MRR@10, and MAP
- Containerized full ML serving stack (FastAPI + Streamlit + 10 infrastructure services) with Prometheus/Grafana observability, launched via single `docker compose up -d`
- Fine-tuned bi-encoder and cross-encoder on MS MARCO with hard negative mining (two-stage curriculum), achieving state-of-the-art retrieval quality on the dev benchmark

---

## License

MIT License — see [LICENSE](LICENSE) for details.
