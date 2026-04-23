"""
Paired bootstrap significance testing for retrieval system comparison.

Reference:
    Efron & Tibshirani, "An Introduction to the Bootstrap" (1993).
    Sakai, "Statistical Significance, Power, and Sample Sizes" (SIGIR 2016).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SignificanceTestResult:
    """Result of a paired bootstrap significance test.

    Attributes:
        system_a:            Name / identifier of system A.
        system_b:            Name / identifier of system B.
        metric_name:         The metric on which the test was performed.
        mean_a:              Mean metric value for system A.
        mean_b:              Mean metric value for system B.
        delta:               mean_b - mean_a  (positive = B is better).
        p_value:             Two-sided bootstrap p-value.
        ci_lower:            Lower bound of the 95% confidence interval for delta.
        ci_upper:            Upper bound of the 95% confidence interval for delta.
        is_significant:      True if p_value < alpha.
        alpha:               Significance level used.
        effect_size:         Cohen's d estimate for the per-query score differences.
        n_queries:           Number of queries used in the test.
        n_bootstrap:         Number of bootstrap samples drawn.
    """

    system_a: str
    system_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    alpha: float
    effect_size: float
    n_queries: int
    n_bootstrap: int
    extra: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        sig = "YES" if self.is_significant else "NO"
        return (
            f"[{self.metric_name}] {self.system_a} vs {self.system_b}: "
            f"delta={self.delta:+.4f}  p={self.p_value:.4f}  "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]  "
            f"significant={sig}  d={self.effect_size:.3f}"
        )

    def to_dict(self) -> Dict[str, float | str | bool | int]:
        return {
            "system_a": self.system_a,
            "system_b": self.system_b,
            "metric_name": self.metric_name,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "n_queries": self.n_queries,
            "n_bootstrap": self.n_bootstrap,
        }


# ---------------------------------------------------------------------------
# Core bootstrap routine
# ---------------------------------------------------------------------------


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float | bool | Tuple[float, float]]:
    """Paired bootstrap significance test between two systems.

    For each bootstrap sample, re-sample *pairs* (score_a_i, score_b_i) with
    replacement and measure whether the observed difference is maintained.
    The two-sided p-value is estimated as the fraction of bootstrap samples
    where the re-sampled delta has the opposite sign to the observed delta.

    Args:
        scores_a:    Per-query metric values for system A (length N).
        scores_b:    Per-query metric values for system B (length N).
        n_bootstrap: Number of bootstrap resamples.
        alpha:       Significance threshold (default 0.05).
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys:
          - p_value           (float)
          - confidence_interval  (Tuple[float, float]) – 95% CI for delta
          - is_significant    (bool)
          - effect_size       (float) – Cohen's d on per-query differences
          - delta             (float) – mean_b - mean_a
          - mean_a            (float)
          - mean_b            (float)

    Raises:
        ValueError: If the input lists have different lengths or are empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length, "
            f"got {len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) == 0:
        raise ValueError("Input score lists must not be empty.")

    rng = np.random.default_rng(seed)
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    n = len(arr_a)

    observed_delta = float(arr_b.mean() - arr_a.mean())

    # Bootstrap: resample pairs with replacement
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_a = arr_a[indices]  # shape (n_bootstrap, n)
    boot_b = arr_b[indices]
    boot_deltas = boot_b.mean(axis=1) - boot_a.mean(axis=1)  # shape (n_bootstrap,)

    # Two-sided p-value: proportion of samples where delta opposes the observed sign
    if observed_delta >= 0:
        # B is (on average) better; count how often bootstrap says otherwise
        p_value = float(np.mean(boot_deltas <= 0.0))
    else:
        # A is better; count how often bootstrap says B is better
        p_value = float(np.mean(boot_deltas >= 0.0))

    # 95% confidence interval for delta (percentile method)
    ci_level = 1.0 - alpha
    lo = float(np.percentile(boot_deltas, 100.0 * (1 - ci_level) / 2))
    hi = float(np.percentile(boot_deltas, 100.0 * (1 - (1 - ci_level) / 2)))

    # Cohen's d on per-query differences
    diffs = arr_b - arr_a
    pooled_std = float(np.std(diffs, ddof=1))
    effect_size = float(diffs.mean() / pooled_std) if pooled_std > 0.0 else 0.0

    return {
        "p_value": p_value,
        "confidence_interval": (lo, hi),
        "is_significant": p_value < alpha,
        "effect_size": effect_size,
        "delta": observed_delta,
        "mean_a": float(arr_a.mean()),
        "mean_b": float(arr_b.mean()),
    }


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------


def compare_systems(
    system_a_per_query: Dict[str, float],
    system_b_per_query: Dict[str, float],
    metric_name: str,
    system_a_name: str = "system_a",
    system_b_name: str = "system_b",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> SignificanceTestResult:
    """Full significance test between two systems on a shared set of queries.

    Only queries present in *both* systems are used; a warning is logged if
    there are queries with missing scores in either system.

    Args:
        system_a_per_query: query_id -> metric_value for system A.
        system_b_per_query: query_id -> metric_value for system B.
        metric_name:        Human-readable metric label (e.g. "ndcg@10").
        system_a_name:      Display name for system A.
        system_b_name:      Display name for system B.
        n_bootstrap:        Bootstrap resamples.
        alpha:              Significance level.
        seed:               Random seed.

    Returns:
        :class:`SignificanceTestResult` with all test statistics.

    Raises:
        ValueError: If the two systems share no common query IDs.
    """
    shared_queries = sorted(
        set(system_a_per_query.keys()) & set(system_b_per_query.keys())
    )
    only_a = set(system_a_per_query) - set(system_b_per_query)
    only_b = set(system_b_per_query) - set(system_a_per_query)

    if only_a:
        logger.warning(
            "%d queries only in system A (%s); excluded from test.", len(only_a), system_a_name
        )
    if only_b:
        logger.warning(
            "%d queries only in system B (%s); excluded from test.", len(only_b), system_b_name
        )

    if not shared_queries:
        raise ValueError(
            f"No common queries between '{system_a_name}' and '{system_b_name}'."
        )

    scores_a = [system_a_per_query[qid] for qid in shared_queries]
    scores_b = [system_b_per_query[qid] for qid in shared_queries]

    result = paired_bootstrap_test(
        scores_a, scores_b, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
    )

    ci: Tuple[float, float] = result["confidence_interval"]  # type: ignore[assignment]

    sr = SignificanceTestResult(
        system_a=system_a_name,
        system_b=system_b_name,
        metric_name=metric_name,
        mean_a=result["mean_a"],  # type: ignore[arg-type]
        mean_b=result["mean_b"],  # type: ignore[arg-type]
        delta=result["delta"],  # type: ignore[arg-type]
        p_value=result["p_value"],  # type: ignore[arg-type]
        ci_lower=ci[0],
        ci_upper=ci[1],
        is_significant=result["is_significant"],  # type: ignore[arg-type]
        alpha=alpha,
        effect_size=result["effect_size"],  # type: ignore[arg-type]
        n_queries=len(shared_queries),
        n_bootstrap=n_bootstrap,
    )
    logger.info("%s", sr)
    return sr


def compare_all_systems(
    systems: Dict[str, Dict[str, float]],
    metric_name: str,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> List[SignificanceTestResult]:
    """Pairwise significance testing for all systems.

    Args:
        systems:     system_name -> {query_id: metric_value}.
        metric_name: Metric label (e.g. "ndcg@10").
        n_bootstrap: Bootstrap resamples per pair.
        alpha:       Significance level.
        seed:        Random seed (incremented per pair for independence).

    Returns:
        List of :class:`SignificanceTestResult` for every A-B pair,
        sorted by descending |delta| (largest effect first).
    """
    system_names = list(systems.keys())
    if len(system_names) < 2:
        logger.warning("compare_all_systems requires at least 2 systems; got %d.", len(system_names))
        return []

    results: List[SignificanceTestResult] = []
    for pair_idx, (name_a, name_b) in enumerate(combinations(system_names, 2)):
        try:
            sr = compare_systems(
                system_a_per_query=systems[name_a],
                system_b_per_query=systems[name_b],
                metric_name=metric_name,
                system_a_name=name_a,
                system_b_name=name_b,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=seed + pair_idx,  # different seed per pair
            )
            results.append(sr)
        except ValueError as exc:
            logger.error("Skipping pair (%s, %s): %s", name_a, name_b, exc)

    # Sort by absolute effect size descending
    results.sort(key=lambda r: abs(r.delta), reverse=True)
    return results
