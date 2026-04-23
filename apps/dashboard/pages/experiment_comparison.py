"""
Experiment Comparison page – compare retrieval strategies across benchmark runs.

Features
--------
* Multi-select experiments from the API.
* Metrics table: NDCG@10, MRR@10, MAP, Recall@10, latency.
* Bar chart per metric (Plotly).
* Significance test results with p-values and confidence intervals.
* Green / gray colour-coding for significant / non-significant differences.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.api_client import VectorLiftClient, VectorLiftClientError

METRIC_LABELS: dict[str, str] = {
    "ndcg_at_10": "NDCG@10",
    "mrr_at_10": "MRR@10",
    "map_score": "MAP",
    "recall_at_10": "Recall@10",
    "mean_latency_ms": "Mean Latency (ms)",
}


def _extract_metrics(exp: dict[str, Any]) -> dict[str, float]:
    m = exp.get("metrics", {})
    return {k: m.get(k, 0.0) for k in METRIC_LABELS}


def _build_metrics_df(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        row: dict[str, Any] = {"Experiment": exp.get("name", exp.get("experiment_id", "?"))}
        row.update(_extract_metrics(exp))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Experiment")
    df.columns = [METRIC_LABELS.get(c, c) for c in df.columns]
    return df


def _metric_bar_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure(
        [
            go.Bar(
                x=df.index.tolist(),
                y=df[metric].tolist(),
                marker_color="#4C78A8",
                text=[f"{v:.4f}" for v in df[metric]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=metric,
        yaxis_title=metric,
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _significance_table(comparison: dict[str, Any]) -> pd.DataFrame:
    tests = comparison.get("significance_tests", [])
    rows = []
    for t in tests:
        rows.append(
            {
                "Metric": t.get("metric", ""),
                "Delta": round(t.get("delta", 0.0), 4),
                "p-value": round(t.get("p_value", 1.0), 4),
                "CI low": round(t.get("confidence_interval_low", 0.0), 4),
                "CI high": round(t.get("confidence_interval_high", 0.0), 4),
                "Significant": t.get("significant", False),
            }
        )
    return pd.DataFrame(rows)


def render(client: VectorLiftClient) -> None:
    st.header("Experiment Comparison")

    # ------------------------------------------------------------------
    # Load experiments
    # ------------------------------------------------------------------
    with st.spinner("Loading experiments …"):
        try:
            experiments: list[dict[str, Any]] = client.get_experiments()
        except VectorLiftClientError as exc:
            st.error(f"API error {exc.status_code}: {exc.detail}")
            return
        except Exception as exc:
            st.error(f"Could not connect to API: {exc}")
            return

    if not experiments:
        st.info("No experiments found.  Run an evaluation first.")
        return

    exp_names = [f"{e.get('name', '?')} ({e['experiment_id'][:8]})" for e in experiments]
    exp_by_label = dict(zip(exp_names, experiments))

    selected_labels = st.multiselect(
        "Select experiments to compare",
        options=exp_names,
        default=exp_names[:min(3, len(exp_names))],
    )

    if not selected_labels:
        st.info("Select at least one experiment above.")
        return

    selected_exps = [exp_by_label[l] for l in selected_labels]

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------
    st.subheader("Metrics Summary")
    df = _build_metrics_df(selected_exps)
    st.dataframe(df.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)

    # ------------------------------------------------------------------
    # Per-metric bar charts
    # ------------------------------------------------------------------
    st.subheader("Per-metric Comparison")
    chart_metrics = [v for v in METRIC_LABELS.values() if "Latency" not in v]
    cols = st.columns(2)
    for i, metric in enumerate(chart_metrics):
        with cols[i % 2]:
            if metric in df.columns:
                st.plotly_chart(_metric_bar_chart(df, metric), use_container_width=True)

    # ------------------------------------------------------------------
    # Pairwise significance tests
    # ------------------------------------------------------------------
    if len(selected_exps) >= 2:
        st.subheader("Significance Tests")
        baseline_label = st.selectbox(
            "Baseline experiment", options=selected_labels, index=0
        )
        candidate_label = st.selectbox(
            "Candidate experiment",
            options=[l for l in selected_labels if l != baseline_label],
        )
        baseline_id = exp_by_label[baseline_label]["experiment_id"]
        candidate_id = exp_by_label[candidate_label]["experiment_id"]

        if st.button("Run significance tests"):
            with st.spinner("Computing …"):
                try:
                    comparison = client.compare_experiments(baseline_id, candidate_id)
                except VectorLiftClientError as exc:
                    st.error(f"API error {exc.status_code}: {exc.detail}")
                    return
                except Exception as exc:
                    st.error(str(exc))
                    return

            sig_df = _significance_table(comparison)
            if sig_df.empty:
                st.info("No significance data returned.")
                return

            def _row_style(row: pd.Series) -> list[str]:
                color = "#c6efce" if row.get("Significant", False) else "#f2f2f2"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                sig_df.style.apply(_row_style, axis=1),
                use_container_width=True,
            )
            st.caption(
                "Green rows indicate statistically significant improvement "
                "(p < 0.05 with 95 % confidence interval not crossing zero)."
            )
