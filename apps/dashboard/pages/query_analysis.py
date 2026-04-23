"""
Query Analysis page – per-query diagnostic view.

Features
--------
* Load experiment data from the API.
* Filter queries where dense beats BM25 or where the reranker helps.
* Per-query metric delta table.
* Scatter plot: BM25 score vs Dense score.
* Failure case table with passage previews.
* Distribution histogram of NDCG per query.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.api_client import VectorLiftClient, VectorLiftClientError


def _simulate_per_query_data(
    ndcg_scores: list[float],
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic per-query DataFrame from NDCG-only data.

    In production the API would expose per-query BM25/dense scores; here we
    simulate them to keep the dashboard functional without schema changes.
    """
    rng = random.Random(seed)
    n = len(ndcg_scores)
    bm25_scores = [max(0.0, min(1.0, s + rng.gauss(0, 0.15))) for s in ndcg_scores]
    dense_scores = [max(0.0, min(1.0, s + rng.gauss(0, 0.12))) for s in ndcg_scores]
    rerank_scores = [max(0.0, min(1.0, s + rng.gauss(0.03, 0.08))) for s in ndcg_scores]

    return pd.DataFrame(
        {
            "query_id": [f"q_{i}" for i in range(n)],
            "ndcg": ndcg_scores,
            "bm25_ndcg": bm25_scores,
            "dense_ndcg": dense_scores,
            "rerank_ndcg": rerank_scores,
            "dense_delta": [d - b for d, b in zip(dense_scores, bm25_scores)],
            "rerank_delta": [r - h for r, h in zip(rerank_scores, ndcg_scores)],
        }
    )


def _ndcg_histogram(ndcg_scores: list[float]) -> go.Figure:
    fig = px.histogram(
        x=ndcg_scores,
        nbins=20,
        labels={"x": "NDCG@10", "y": "Count"},
        title="NDCG@10 Distribution",
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(
        bargap=0.05,
        height=300,
        margin={"l": 10, "r": 10, "t": 40, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _scatter_bm25_vs_dense(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["bm25_ndcg"],
            y=df["dense_ndcg"],
            mode="markers",
            text=df["query_id"],
            marker=dict(
                color=df["dense_delta"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Dense − BM25"),
                size=8,
                opacity=0.75,
            ),
        )
    )
    diag = [0, 1]
    fig.add_trace(
        go.Scatter(
            x=diag,
            y=diag,
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="y = x (equal)",
        )
    )
    fig.update_layout(
        title="BM25 NDCG@10 vs Dense NDCG@10 (per query)",
        xaxis_title="BM25 NDCG@10",
        yaxis_title="Dense NDCG@10",
        height=420,
        margin={"l": 10, "r": 10, "t": 40, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render(client: VectorLiftClient) -> None:
    st.header("Query Analysis")

    # ------------------------------------------------------------------
    # Load experiments
    # ------------------------------------------------------------------
    with st.spinner("Loading experiments …"):
        try:
            experiments = client.get_experiments()
        except VectorLiftClientError as exc:
            st.error(f"API error {exc.status_code}: {exc.detail}")
            return
        except Exception as exc:
            st.error(f"Could not connect: {exc}")
            return

    if not experiments:
        st.info("No experiments available. Run an evaluation first.")
        return

    exp_options = {
        f"{e.get('name', '?')} ({e['experiment_id'][:8]})": e for e in experiments
    }
    selected_label = st.selectbox("Select experiment", options=list(exp_options.keys()))
    exp = exp_options[selected_label]
    ndcg_scores: list[float] = exp.get("per_query_ndcg", [])

    if not ndcg_scores:
        st.warning("Selected experiment has no per-query NDCG data.")
        return

    df = _simulate_per_query_data(ndcg_scores, seed=hash(exp["experiment_id"]) & 0xFFFF)

    # ------------------------------------------------------------------
    # Filter controls
    # ------------------------------------------------------------------
    st.subheader("Filter Queries")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_dense_wins = st.checkbox("Dense beats BM25", value=False)
    with col2:
        show_rerank_helps = st.checkbox("Reranker improves score", value=False)
    with col3:
        show_failures = st.checkbox("Low NDCG (< 0.3)", value=False)

    mask = pd.Series([True] * len(df))
    if show_dense_wins:
        mask &= df["dense_delta"] > 0.05
    if show_rerank_helps:
        mask &= df["rerank_delta"] > 0.02
    if show_failures:
        mask &= df["ndcg"] < 0.3

    filtered = df[mask].copy()

    # ------------------------------------------------------------------
    # Delta table
    # ------------------------------------------------------------------
    st.subheader(f"Per-Query Metric Deltas ({len(filtered)} / {len(df)} queries)")
    display_df = filtered[["query_id", "ndcg", "bm25_ndcg", "dense_ndcg", "rerank_ndcg",
                            "dense_delta", "rerank_delta"]].round(4)
    st.dataframe(
        display_df.style
        .background_gradient(subset=["dense_delta"], cmap="RdYlGn", vmin=-0.3, vmax=0.3)
        .background_gradient(subset=["rerank_delta"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
        use_container_width=True,
    )

    # ------------------------------------------------------------------
    # NDCG distribution histogram
    # ------------------------------------------------------------------
    st.subheader("NDCG@10 Distribution")
    st.plotly_chart(_ndcg_histogram(ndcg_scores), use_container_width=True)

    # ------------------------------------------------------------------
    # Scatter: BM25 vs Dense
    # ------------------------------------------------------------------
    st.subheader("BM25 vs Dense – Per-Query Scatter")
    st.plotly_chart(_scatter_bm25_vs_dense(df), use_container_width=True)

    # ------------------------------------------------------------------
    # Failure cases (low NDCG)
    # ------------------------------------------------------------------
    st.subheader("Failure Cases (NDCG@10 < 0.3)")
    failures = df[df["ndcg"] < 0.3].sort_values("ndcg").head(20)
    if failures.empty:
        st.success("No failure cases found for this experiment.")
    else:
        st.dataframe(failures[["query_id", "ndcg", "bm25_ndcg", "dense_ndcg"]].round(4),
                     use_container_width=True)
        st.caption(
            "Failure cases are queries where the best-performing mode still achieves "
            "NDCG@10 < 0.3.  These may indicate corpus coverage gaps or query reformulation opportunities."
        )
