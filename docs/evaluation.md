# Evaluation Methodology

This document covers how VectorLift evaluates retrieval system quality: the metrics used, how statistical significance is established, what experiment configurations are compared, and how to interpret the results.

---

## Benchmark Dataset

All evaluations use **MS MARCO Passage Retrieval** (HuggingFace `ms_marco` / `v1.1`):

- **Corpus**: 8.84M passages extracted from web documents (Bing index)
- **Queries**: 6,980 dev queries with human relevance judgments
- **Qrels**: binary relevance (0 = not relevant, 1 = relevant); typically 1 relevant passage per query
- **Task**: given a query, rank the entire corpus by relevance

MS MARCO is the standard benchmark for passage retrieval and is used by every state-of-the-art neural retrieval paper, making it the natural choice for comparing VectorLift's systems to published baselines.

---

## Metrics

### NDCG@k (Normalized Discounted Cumulative Gain)

NDCG@k is the primary metric for ranked retrieval. It measures how well the system places relevant documents near the top of the result list, discounting by rank position.

**DCG@k** (Discounted Cumulative Gain):

```
DCG@k = Σ_{i=1}^{k}  rel_i / log2(i + 1)
```

where `rel_i` is the relevance grade of the document at rank `i`.

**Ideal DCG (IDCG@k)**: DCG of the perfect ranking (all relevant documents at the top).

**NDCG@k**:

```
NDCG@k = DCG@k / IDCG@k
```

NDCG@k is in [0, 1]; higher is better. It handles graded relevance naturally (though MS MARCO uses binary labels, so DCG reduces to the standard discount-at-rank formula). For MS MARCO with its one-relevant-passage-per-query structure, NDCG@10 captures whether the single relevant passage is in the top 10 and how high it appears.

**Implementation**: `pipelines/evaluation/metrics.py::ndcg_at_k(relevances, k)`

### MRR@k (Mean Reciprocal Rank)

MRR@k measures the rank of the first relevant document across queries, truncated at k.

```
MRR@k = (1/|Q|) Σ_{q∈Q}  1 / rank_q
```

where `rank_q` is the rank of the first relevant document for query `q` (if found in the top k; otherwise 0).

MRR@10 is particularly informative for MS MARCO because most queries have a single relevant passage — retrieving it at rank 1 vs rank 5 makes a big practical difference.

**Implementation**: `pipelines/evaluation/metrics.py::reciprocal_rank(relevances)`

### MAP (Mean Average Precision)

MAP is the mean over queries of Average Precision (AP), where AP is the area under the precision-recall curve:

```
AP = (1 / R)  Σ_{k: rel_k=1}  Precision@k
```

where `R` is the total number of relevant documents for the query and the sum is over all ranks where a relevant document is retrieved.

MAP penalizes systems that place relevant documents late in the list even when NDCG might not, making it a useful complement.

**Implementation**: `pipelines/evaluation/metrics.py::average_precision(relevances)`

### Recall@k

Fraction of all relevant documents for a query that appear in the top-k results:

```
Recall@k = |{rel docs in top k}| / |{all rel docs}|
```

Recall@100 (before reranking) is important because it measures how many relevant passages survive first-stage retrieval — passages not in the top 100 can never be found by the reranker.

**Implementation**: `pipelines/evaluation/metrics.py::recall_at_k(relevances, total_relevant, k)`

### Precision@k

Fraction of the top-k retrieved documents that are relevant:

```
Precision@k = |{rel docs in top k}| / k
```

**Implementation**: `pipelines/evaluation/metrics.py::precision_at_k(relevances, k)`

---

## Statistical Significance Testing

### Why Significance Testing Matters

It is easy to observe a higher NDCG@10 for System B vs System A on a given query set and conclude B is better. But if the improvement is within the noise of the evaluation, it is not reproducible — a different sample of queries or a slight perturbation would reverse the conclusion. Paired bootstrap resampling gives a principled answer to the question: "Is this improvement likely to reflect a true difference in system quality, or is it within the noise?"

### Paired Bootstrap Resampling

**Reference**: Efron & Tibshirani (1993); Sakai, "Statistical Significance, Power, and Sample Sizes" (SIGIR 2016).

**Procedure** (`pipelines/evaluation/significance.py::paired_bootstrap_test`):

1. Compute per-query metric values for System A and System B on the shared query set: `scores_a[i]` and `scores_b[i]` for each query `i`.
2. Compute the observed delta: `δ_obs = mean(scores_b) − mean(scores_a)`.
3. Repeat `n_bootstrap = 10,000` times:
   a. Sample `n` query indices with replacement (where `n` = number of queries).
   b. Compute `δ_boot = mean(scores_b[sample]) − mean(scores_a[sample])`.
4. **Two-sided p-value**: fraction of bootstrap samples where `δ_boot` has the opposite sign to `δ_obs`.
5. **95% confidence interval**: 2.5th and 97.5th percentile of the `δ_boot` distribution.
6. **Cohen's d**: `mean(scores_b − scores_a) / std(scores_b − scores_a)` — normalized effect size.

**Significance threshold**: `p < 0.05`.

**Why paired?** Both systems are evaluated on the same query set. Pairing removes between-query variance from the test — the same hard query appears in both systems' samples, so the test is sensitive to per-query improvements rather than being confused by which queries happened to be sampled.

**Implementation**:

```python
from pipelines.evaluation.significance import compare_systems

result = compare_systems(
    system_a_per_query={"q1": 0.5, "q2": 0.0, ...},
    system_b_per_query={"q1": 0.8, "q2": 0.2, ...},
    metric_name="ndcg@10",
    system_a_name="BM25",
    system_b_name="Hybrid+Rerank",
    n_bootstrap=10_000,
    alpha=0.05,
)
# result.p_value, result.delta, result.ci_lower, result.ci_upper, result.is_significant
```

---

## Experiment Configurations

VectorLift evaluates six retrieval configurations, each representing a step in the pipeline:

| System | First-Stage | Fusion | Second-Stage | Description |
|---|---|---|---|---|
| BM25 | Elasticsearch BM25 | — | — | Lexical-only baseline |
| Dense | Bi-encoder + Qdrant | — | — | Neural-only baseline |
| Hybrid (RRF) | BM25 + Dense | RRF (k=60) | — | Lexical + neural, no reranking |
| BM25 + Reranker | Elasticsearch BM25 | — | CrossEncoder | BM25 candidates reranked |
| Dense + Reranker | Bi-encoder + Qdrant | — | CrossEncoder | Dense candidates reranked |
| Hybrid + Reranker | BM25 + Dense | RRF (k=60) | CrossEncoder | Full pipeline |

All systems use `top_k = 10` for final results. Reranked systems retrieve `top_n = 100` candidates before reranking.

**Configuration for evaluation runs** (`scripts/run_evaluation.py`):

```bash
python scripts/run_evaluation.py \
  --mode dev \
  --systems bm25 dense hybrid bm25_rerank dense_rerank hybrid_rerank \
  --k-values 1 3 5 10 100 \
  --n-bootstrap 10000 \
  --output-dir experiments/results/
```

---

## Running Evaluations

### Quick Evaluation (dev mode, ~100 queries)

```bash
make evaluate
# or
python scripts/run_evaluation.py --mode dev --output-dir experiments/results/
```

### Full Evaluation Suite (all six systems)

```bash
make evaluate-all
# Equivalent to:
for mode in bm25 dense hybrid bm25_rerank dense_rerank hybrid_rerank; do
  python -m experiments.evaluate \
    --mode $mode \
    --dataset-mode small \
    --output-dir experiments/results/$mode
done
python -m experiments.compare \
  --results-dir experiments/results \
  --output experiments/results/comparison.json
```

### Via the API

```bash
# Trigger async evaluation job
curl -X POST http://localhost:8000/evaluation/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hybrid-rrf-small",
    "retrieval_mode": "hybrid",
    "dataset_mode": "small",
    "k_values": [1, 3, 5, 10, 100]
  }'

# Poll for results
curl http://localhost:8000/evaluation/experiments

# Compare two experiments
curl "http://localhost:8000/evaluation/experiments/exp-001/compare?candidate_id=exp-002"
```

---

## Result Format

Each experiment produces a JSON file in `experiments/results/{experiment_id}.json`:

```json
{
  "experiment_id": "hybrid-rrf-small-20240423",
  "config": {
    "retrieval_mode": "hybrid",
    "fusion_strategy": "reciprocal_rank_fusion",
    "dataset_mode": "small",
    "top_k": 10,
    "rerank_top_n": 100
  },
  "metrics": {
    "ndcg@1":  0.201,
    "ndcg@3":  0.251,
    "ndcg@5":  0.261,
    "ndcg@10": 0.268,
    "ndcg@100": 0.312,
    "mrr@10":  0.281,
    "map":     0.219,
    "recall@10":  0.318,
    "recall@100": 0.601,
    "precision@10": 0.044
  },
  "per_query_metrics": {
    "q1001": {"ndcg": 1.0, "rr": 1.0, "ap": 1.0, "precision": 0.1},
    "q1002": {"ndcg": 0.0, "rr": 0.0, "ap": 0.0, "precision": 0.0},
    ...
  },
  "latency_p50_ms": 42,
  "latency_p95_ms": 58,
  "n_queries": 6980,
  "timestamp": "2024-04-23T10:00:00Z"
}
```

---

## How to Interpret Results

### The Summary Table

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

- All neural systems significantly outperform BM25 at p < 0.05
- The Hybrid+Reranker achieves 69% relative NDCG@10 improvement over BM25 (0.312 vs 0.184)
- The latency cost of reranking is 2-3x: 58ms → 142ms for hybrid systems
- The hybrid first stage consistently beats either modality alone, confirming that BM25 and dense retrieval are complementary

### Per-Query Analysis

The experiment comparison dashboard (Streamlit) shows a histogram of per-query NDCG@10 deltas (System B − System A). Key patterns:
- **Dense fails on rare-term queries**: BM25 beats dense when queries contain rare technical terms or model numbers that the bi-encoder did not see frequently in training
- **BM25 fails on paraphrase queries**: Dense beats BM25 for queries where the relevant passage uses different vocabulary from the question
- **Hybrid captures both**: Most queries where either modality alone fails are recovered by the hybrid

### When to Accept Degraded Latency

The reranker adds ~80-90ms of latency (cross-encoder forward pass over 100 candidates). Whether to use it depends on the application:
- Real-time user-facing search: Hybrid (58ms P95) may be sufficient
- High-precision use cases (QA, document retrieval): Hybrid+Reranker is worth the latency

### Significance Test Interpretation

A result marked `*` means the system is statistically significantly better than BM25 at α=0.05. This does not mean the difference is practically significant — look at the delta magnitude (e.g., NDCG@10 +0.063 for Dense vs BM25) and the confidence interval.

A system that is significantly better on NDCG@10 but not on MRR@10 is likely retrieving the relevant document more consistently within the top 10 but not always at rank 1 — important context for choosing between systems.
