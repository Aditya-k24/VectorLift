# Architecture Deep-Dive

This document describes the internal architecture of VectorLift: how components are structured, how data flows through the system at query time and index time, and the rationale behind key API and infrastructure decisions.

---

## Component Overview

VectorLift is composed of four logical layers:

1. **Retrieval Layer** (`retrieval/`) — the core search logic: BM25, dense, hybrid fusion, and cross-encoder reranking
2. **Application Layer** (`apps/`) — the FastAPI HTTP service, Streamlit dashboard, Kafka worker, and training scripts
3. **Pipeline Layer** (`pipelines/`) — batch ingestion, embedding generation, indexing, and evaluation orchestration
4. **Core Layer** (`core/`) — shared configuration, schemas, logging, Prometheus metrics, and database utilities

These layers have strict dependency ordering: `retrieval` and `core` have no knowledge of `apps`; `pipelines` orchestrate `retrieval` and `core` but not `apps`; `apps` depend on all three layers.

---

## Retrieval Layer

### Interface Contract

All retrievers implement the `BaseRetriever` protocol defined in `retrieval/interfaces/base.py`. The contract is:

```python
class BaseRetriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...
```

`RetrievalResult` is a dataclass with fields: `passage_id`, `text`, `title`, `score`, `rank`, `metadata`. This interface enables swapping backends (e.g., Qdrant → Pinecone, Elasticsearch → OpenSearch) without touching the pipeline or API layers.

### BM25 Retrieval

**Elasticsearch retriever** (`retrieval/bm25/elasticsearch_retriever.py`):
- Uses the `elasticsearch-py` async client
- Queries use `multi_match` over `text` and `title` fields with `BM25` similarity (ES default)
- Custom analyzers applied at index time: lowercase, stop-word removal, and optional stemming
- Bulk indexing via `helpers.async_bulk` for 500+ docs/sec throughput

**Local fallback** (`retrieval/bm25/local_retriever.py`):
- Uses the `rank-bm25` library (pure Python BM25Okapi implementation)
- Tokenizes on whitespace + punctuation strip
- Loaded from a pre-built pickle or built in-memory at startup
- Intended for CI, unit tests, and environments where Elasticsearch is unavailable

### Dense Retrieval

**Encoder** (`retrieval/dense/encoder.py`):
- Wraps `SentenceTransformer` with batched `.encode()` calls
- Supports `cpu`, `cuda`, and `mps` devices via the `BIENCODER_DEVICE` env var
- Returns L2-normalized vectors (cosine similarity = dot product on normalized vectors)

**Qdrant retriever** (`retrieval/dense/qdrant_retriever.py`):
- Uses `qdrant_client.QdrantClient` (HTTP mode; gRPC optional via `QDRANT_PREFER_GRPC`)
- Collection configured with HNSW index (`m=16`, `ef_construct=100` by default)
- `search()` returns `ScoredPoint` objects mapped to `RetrievalResult`

**FAISS index** (`retrieval/dense/faiss_index.py`):
- `IndexFlatIP` (inner product on normalized vectors = cosine similarity)
- Built from the processed corpus with batched encoding
- Serialized to `data/indexes/dense/faiss.index` for fast startup
- Resume-capable: saves progress shards so interrupted jobs continue cleanly

**Dense retriever** (`retrieval/dense/dense_retriever.py`):
- Unified wrapper that delegates to either Qdrant or FAISS depending on availability
- Qdrant preferred for production (ANN with recall guarantees); FAISS for local dev

### Hybrid Retrieval

**Fusion strategies** (`retrieval/hybrid/fusion.py`):

*Score Fusion*:
```
merged_score(p) = α × dense_score(p) + (1 − α) × bm25_score(p)
```
Both scores are min-max normalized to [0, 1] independently before combination. Passages present in only one list receive a score of 0.0 for the missing modality.

*Reciprocal Rank Fusion (RRF)*:
```
rrf_score(p) = Σ_i  1 / (k + rank_i(p))
```
where `k = 60` (smoothing constant). Passages not in a list contribute 0. RRF is scale-invariant — it ignores raw score magnitudes and works purely from rank positions, making it robust to the very different score distributions of BM25 and cosine similarity.

**Hybrid retriever** (`retrieval/hybrid/hybrid_retriever.py`):
- Fires BM25 and dense retrieval concurrently with `asyncio.gather`
- Applies the configured fusion strategy (RRF by default)
- Returns a merged, re-ranked list with new 1-based ranks and RRF scores

### Cross-Encoder Reranker

**Cross-encoder** (`retrieval/reranker/cross_encoder.py`):
- Wraps `sentence_transformers.CrossEncoder`
- `score(query, candidates)` tokenizes `[CLS] query [SEP] passage [SEP]`, runs a forward pass through all layers, and returns scalar logits for each pair
- Batched scoring with configurable `batch_size` (default 32) to bound GPU memory

**Rerank pipeline** (`retrieval/reranker/pipeline.py`):
1. First-stage retrieval: run the configured retriever (BM25, dense, or hybrid) for `top_n` candidates (default 100)
2. Cross-encoder scoring: score all `top_n` (query, passage) pairs
3. Re-sort by cross-encoder score, truncate to `top_k`

---

## Application Layer

### FastAPI Service

**Startup sequence** (in `apps/api/main.py` `lifespan` handler):
1. Load `Settings` from `.env` via Pydantic-settings
2. Initialize `AsyncElasticsearch` client
3. Initialize `QdrantClient`
4. Initialize `redis.asyncio` client
5. Create `async_sessionmaker` for PostgreSQL via SQLAlchemy
6. Instantiate `SearchPipeline` (loads bi-encoder and cross-encoder weights into memory)
7. Yield — application serves requests
8. On shutdown: gracefully close all client connections, dispose DB engine

**Middleware stack** (outermost first):
- `CORSMiddleware` — allows configurable origins
- `TimingMiddleware` — adds `X-Process-Time` response header
- `RequestIDMiddleware` — injects `X-Request-ID` for distributed tracing
- `prometheus_middleware` — records per-path latency and request count

**Dependency injection** (`apps/api/dependencies.py`):
- `SearchPipelineDep` — retrieves the singleton pipeline from `app.state`; raises 503 if not loaded
- `SearchServiceDep` — constructs a `SearchService` from the pipeline dependency per request

**Router structure:**
- `POST /search` → `SearchService.execute_search()`
- `POST /search/rerank` → `SearchService.execute_rerank()`
- `GET /health` → parallel async pings to all backends
- `GET /model-info` → introspects `app.state.search_pipeline`
- `GET /metrics` → `prometheus_client.generate_latest()`
- `POST /evaluation/evaluate` → background task via `BackgroundTasks`
- `GET /evaluation/experiments` → `EvaluationService.list_experiments()`
- `GET /evaluation/experiments/{id}/compare` → `EvaluationService.compare_experiments()`

### Kafka Worker

`apps/worker/kafka_consumer.py` runs as a separate Docker service (`vectorlift-worker`). It:
1. Subscribes to the `vectorlift.queries` and `vectorlift.feedback` topics
2. For each `document.index` event, deserializes the passage and writes it to both Elasticsearch (via bulk API) and Qdrant (via upsert)
3. For each failed message, logs to a dead-letter topic (`vectorlift.dlq`) with the original payload and error metadata
4. Commits offsets only after successful processing (at-least-once semantics with idempotent upserts)

### Training Scripts

Training is designed as standalone scripts rather than framework flows so that they can be run directly (`python -m apps.trainer.train_biencoder ...`) without a running server or orchestrator. Each script:
- Accepts all hyperparameters as CLI flags (via Typer)
- Persists `training_config.json` and `training_metrics.json` alongside the model checkpoint
- Uses `InformationRetrievalEvaluator` from `sentence-transformers` for NDCG@10 tracking during training

---

## Pipeline Layer

### Ingestion (`pipelines/ingestion/msmarco.py`)

`MSMARCODataset` wraps the HuggingFace `datasets` library:
- `load_passages(mode)` — streams the passage corpus, applying `DATASET_MODE` limits
- `load_queries(split)` — loads dev or train query sets
- `load_qrels(split)` — loads ground-truth relevance judgments
- `load_training_triplets(max_samples)` — loads (query, positive, negative) triplets for bi-encoder training

Dataset mode limits (approximate):

| Mode | Passages | Triplets |
|---|---|---|
| `dev` | 1,000 | 5,000 |
| `small` | 100,000 | 200,000 |
| `full` | 8,841,823 | 502,939 |

### Embedding Generation (`pipelines/embedding/generator.py`)

Batch embedding pipeline with resume capability:
1. Load unprocessed passages from `data/processed/`
2. Encode in batches of 512 using the bi-encoder
3. Every 10,000 passages, flush embeddings to a partial FAISS shard and record progress
4. On completion, merge shards into a final `faiss.index` file
5. If interrupted, resume from the last saved shard position

### Evaluation (`pipelines/evaluation/`)

**`metrics.py`** — per-query and aggregate IR metrics:
- `ndcg_at_k(relevances, k)` — standard TREC NDCG using log2 discounting
- `average_precision(relevances)` — area under the interpolated precision-recall curve
- `reciprocal_rank(relevances)` — inverse rank of the first relevant document
- `recall_at_k(relevances, total_relevant, k)`
- `compute_metrics(qrels, results, k_values)` — aggregate over all queries

**`significance.py`** — paired bootstrap testing:
- `paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000)` — resamples paired per-query scores with replacement, estimates two-sided p-value as the fraction of bootstrap deltas opposing the observed sign
- `compare_systems(...)` — higher-level wrapper producing a `SignificanceTestResult` dataclass with p-value, 95% CI for delta, and Cohen's d
- `compare_all_systems(systems)` — pairwise comparison of N systems, sorted by absolute effect size

**`runner.py`** — orchestrates a full evaluation run:
1. Load qrels and queries from MS MARCO
2. For each query, call the configured retriever and record the ranked passage IDs
3. Pass `qrels` and `results` to `compute_metrics`
4. Save results to `experiments/results/{experiment_id}.json`
5. Persist metadata to PostgreSQL for dashboard access

---

## Data Flow Diagrams

### Query-Time Flow

```
HTTP POST /search
     │
     ▼
RequestIDMiddleware (assign X-Request-ID)
     │
     ▼
TimingMiddleware (start timer)
     │
     ▼
PrometheusMiddleware (record request)
     │
     ▼
SearchRouter.search()
     │
     ▼
SearchService.execute_search(request)
     │
     ├─[cache hit]──► Redis GET ──► return cached response
     │
     ├─[mode=bm25]──► BM25Retriever.retrieve()
     │                     │
     │               ElasticsearchRetriever  (or LocalBM25)
     │
     ├─[mode=dense]─► DenseRetriever.retrieve()
     │                     │
     │               Encoder.encode(query)
     │                     │
     │               QdrantRetriever.search()  (or FAISS ANN)
     │
     ├─[mode=hybrid]─► HybridRetriever.retrieve()
     │                     │
     │               asyncio.gather(BM25, Dense)
     │                     │
     │               ReciprocalRankFusion.merge()
     │
     └─[mode=rerank]─► RerankPipeline.retrieve_and_rerank()
                           │
                       HybridRetriever (top 100)
                           │
                       CrossEncoder.score()
                           │
                       Sort + truncate to top_k
     │
     ▼
Redis SET (cache result with TTL)
     │
     ▼
SearchResponse → HTTP 200
```

### Index-Time Flow (Kafka Pipeline)

```
Kafka Producer
(scripts/ingest_sample.py or external system)
     │
     ▼
Topic: vectorlift.queries
     │
     ▼
Kafka Consumer (apps/worker/kafka_consumer.py)
     │
     ├─► Elasticsearch BulkAPI ──► BM25 index
     │
     ├─► Encoder.encode(passage_text)
     │        │
     │        ▼
     │   Qdrant.upsert(vector, payload)  ──► Dense index
     │
     └─► PostgreSQL (passage metadata)
```

---

## API Design Rationale

### Why four distinct retrieval modes instead of one configurable endpoint?

Different modes have fundamentally different latency and quality profiles. Exposing them as a named `mode` parameter gives clients explicit control and makes A/B testing straightforward: run the same query with `mode=bm25` and `mode=rerank` and compare. The `SearchService` dispatches to the appropriate retriever based on the mode enum, keeping routing logic out of the retrieval layer.

### Why return ranked results with scores and ranks rather than just text?

Downstream clients (the dashboard, evaluation scripts, re-ranking clients) need scores for visualization and further processing. Including `rank` explicitly makes pagination logic trivial. The `metadata` dict is extensible — future versions can include term match highlights, cluster labels, or confidence scores without breaking the response schema.

### Why async throughout?

All external service calls (Elasticsearch, Qdrant, Redis, PostgreSQL) are I/O-bound. Using `asyncio` for these operations allows the API to serve many concurrent requests on a small thread pool. The Qdrant sync client is wrapped in `asyncio.to_thread()` for the health check; all other clients have native async support.

### Why 202 Accepted for evaluation jobs?

Evaluation over even the `small` dataset can take 30-120 seconds. A synchronous HTTP endpoint would time out or hold connections for too long. The `POST /evaluation/evaluate` endpoint immediately returns a `job_id` and status `"pending"`, and the actual work runs in a `BackgroundTask`. The `GET /evaluation/experiments` endpoint is then polled to see when results are available.

---

## Model Choice Rationale

### Bi-encoder: `msmarco-distilbert-base-tas-b`

TAS-B (Topic-Aware Sampling for Bi-encoders) uses a sampling strategy during training that ensures each batch contains queries from similar topics, producing naturally harder in-batch negatives than random sampling. The DistilBERT backbone is 40% smaller than BERT-base with 97% of the performance, making it ideal for production inference where latency matters. This model is the de facto standard strong bi-encoder baseline for MS MARCO.

### Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

MiniLM-L6 has 6 transformer layers and 22M parameters — about 1/5 the size of BERT-base. For a reranker that processes 100 candidates per query (joint encoding), this size reduction is critical: MiniLM-L6 runs ~4x faster than a full BERT cross-encoder. The MS MARCO fine-tuned version achieves near-full BERT accuracy on the reranking task.

### Embedding dimension: 768

DistilBERT outputs 768-dimensional embeddings. This is a practical sweet spot: small enough to fit millions of vectors in a few GB of RAM (8.8M × 768 × 4 bytes ≈ 27 GB for full MS MARCO), large enough to encode rich semantic representations. For larger deployments, vector quantization (INT8 or binary) in Qdrant can reduce storage 4-32x.
