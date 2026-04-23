"""
Retrieval evaluation metrics for VectorLift.

Implements standard IR metrics:
  - DCG / NDCG @ k
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)
  - Recall @ k
  - Precision @ k

All metrics follow TREC conventions:
  - Binary relevance: 0 = not relevant, >= 1 = relevant
  - Graded relevance: supported for NDCG (higher = more relevant)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-query metric functions
# ---------------------------------------------------------------------------


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k.

    Args:
        relevances: Ordered list of graded relevance labels for retrieved docs.
                    relevances[0] is the top-ranked document.
        k:          Cutoff – only the first k entries are considered.

    Returns:
        DCG score (float >= 0).
    """
    if k <= 0:
        return 0.0
    relevances_k = np.asarray(relevances[:k], dtype=np.float64)
    if relevances_k.size == 0:
        return 0.0
    # Standard DCG: sum(rel_i / log2(i+2)) for i in [0, k)
    positions = np.arange(1, relevances_k.size + 1, dtype=np.float64)
    discounts = np.log2(positions + 1.0)
    return float(np.sum(relevances_k / discounts))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        relevances: Ordered list of graded relevance labels for retrieved docs.
        k:          Cutoff.

    Returns:
        NDCG in [0, 1].  Returns 0.0 if there are no relevant documents.
    """
    if k <= 0:
        return 0.0
    actual_dcg = dcg_at_k(relevances, k)
    # Ideal: sort all observed labels descending, compute DCG on that
    ideal_order = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, k)
    if ideal_dcg == 0.0:
        return 0.0
    return min(actual_dcg / ideal_dcg, 1.0)


def average_precision(relevances: List[int]) -> float:
    """Average Precision (AP) over the full ranked list.

    Args:
        relevances: Ordered list of binary relevance labels (0 or 1).
                    Values > 0 are treated as relevant.

    Returns:
        AP in [0, 1].  Returns 0.0 if there are no relevant documents.
    """
    rel_array = np.asarray([1 if r > 0 else 0 for r in relevances], dtype=np.float64)
    total_relevant = rel_array.sum()
    if total_relevant == 0:
        return 0.0

    precisions: List[float] = []
    running_hits = 0.0
    for i, rel in enumerate(rel_array):
        if rel > 0:
            running_hits += 1.0
            precisions.append(running_hits / (i + 1))

    return float(np.mean(precisions)) if precisions else 0.0


def reciprocal_rank(relevances: List[int]) -> float:
    """Reciprocal Rank – inverse of the rank of the first relevant document.

    Args:
        relevances: Ordered list of binary relevance labels (0 or 1).

    Returns:
        RR in (0, 1].  Returns 0.0 if no relevant document is found.
    """
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevances: List[int], total_relevant: int, k: int) -> float:
    """Recall at k.

    Args:
        relevances:     Ordered list of binary relevance labels.
        total_relevant: Total number of relevant documents in the corpus
                        (the denominator).
        k:              Cutoff.

    Returns:
        Recall in [0, 1].  Returns 0.0 if total_relevant == 0.
    """
    if total_relevant <= 0 or k <= 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r > 0)
    return float(hits) / float(total_relevant)


def precision_at_k(relevances: List[int], k: int) -> float:
    """Precision at k.

    Args:
        relevances: Ordered list of binary relevance labels.
        k:          Cutoff.

    Returns:
        Precision in [0, 1].  Returns 0.0 if k == 0.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r > 0)
    return float(hits) / float(k)


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------


def _build_relevance_list(
    qrel: Dict[str, int],
    ranked_docs: List[str],
) -> List[int]:
    """Map a ranked list of doc IDs to their relevance labels."""
    return [qrel.get(doc_id, 0) for doc_id in ranked_docs]


def compute_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[str]],
    k_values: List[int] | None = None,
) -> Dict[str, float]:
    """Compute all retrieval metrics averaged over queries.

    Args:
        qrels:    Mapping query_id -> {doc_id: relevance_grade}.
                  Relevance grades follow TREC convention (0 = not relevant).
        results:  Mapping query_id -> list of doc_ids ordered by descending rank
                  (index 0 = most relevant).
        k_values: Cutoffs for ndcg@k, recall@k, precision@k.
                  Defaults to [1, 3, 5, 10, 100].

    Returns:
        Dict with keys:
          - ndcg@{k}   for each k in k_values
          - mrr@{k}    for each k in k_values  (MRR truncated at k)
          - map         (untruncated MAP)
          - recall@{k} for each k in k_values
          - precision@{k} for each k in k_values
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 100]

    query_ids = list(qrels.keys())
    n_queries = len(query_ids)
    if n_queries == 0:
        logger.warning("compute_metrics called with empty qrels.")
        return {}

    # Accumulators
    ndcg_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    mrr_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    map_sum: float = 0.0
    recall_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    precision_sums: Dict[int, float] = {k: 0.0 for k in k_values}

    n_evaluated = 0
    for qid in query_ids:
        qrel = qrels[qid]
        ranked_docs = results.get(qid, [])
        if not ranked_docs and not qrel:
            continue

        relevances = _build_relevance_list(qrel, ranked_docs)
        total_rel = sum(1 for v in qrel.values() if v > 0)

        for k in k_values:
            ndcg_sums[k] += ndcg_at_k(relevances, k)
            # MRR@k: reciprocal rank considering only the top-k results
            mrr_sums[k] += reciprocal_rank(relevances[:k])
            recall_sums[k] += recall_at_k(relevances, total_rel, k)
            precision_sums[k] += precision_at_k(relevances, k)

        map_sum += average_precision(relevances)
        n_evaluated += 1

    if n_evaluated == 0:
        logger.warning("No queries evaluated; returning empty metrics.")
        return {}

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / n_evaluated
        metrics[f"mrr@{k}"] = mrr_sums[k] / n_evaluated
        metrics[f"recall@{k}"] = recall_sums[k] / n_evaluated
        metrics[f"precision@{k}"] = precision_sums[k] / n_evaluated
    metrics["map"] = map_sum / n_evaluated

    logger.info(
        "Evaluated %d queries. NDCG@10=%.4f  MAP=%.4f  MRR@10=%.4f",
        n_evaluated,
        metrics.get("ndcg@10", float("nan")),
        metrics["map"],
        metrics.get("mrr@10", float("nan")),
    )
    return metrics


def per_query_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[str]],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Return per-query NDCG@k and RR@k for detailed analysis.

    Args:
        qrels:   Mapping query_id -> {doc_id: relevance_grade}.
        results: Mapping query_id -> ranked list of doc_ids.
        k:       Cutoff for both metrics.

    Returns:
        Dict mapping query_id -> {"ndcg": float, "rr": float}.
    """
    per_query: Dict[str, Dict[str, float]] = {}
    for qid, qrel in qrels.items():
        ranked_docs = results.get(qid, [])
        relevances = _build_relevance_list(qrel, ranked_docs)
        per_query[qid] = {
            "ndcg": ndcg_at_k(relevances, k),
            "rr": reciprocal_rank(relevances[:k]),
            "ap": average_precision(relevances),
            "precision": precision_at_k(relevances, k),
        }
    return per_query
