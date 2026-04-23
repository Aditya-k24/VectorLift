"""
Search Demo page – live interactive search against the VectorLift API.

Features
--------
* Query input, mode selector and top_k slider.
* Results displayed as expandable cards (passage text, score, rank).
* Latency breakdown as a horizontal Plotly bar chart.
* Side-by-side mode comparison: run the same query across all / selected modes.
"""

from __future__ import annotations

import time
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.api_client import VectorLiftClient, VectorLiftClientError

MODES = ["bm25", "dense", "hybrid", "rerank"]


def _make_latency_chart(latency: dict[str, float]) -> go.Figure:
    labels = ["Retrieval", "Reranking", "Total"]
    values = [
        latency.get("retrieval_ms", 0),
        latency.get("rerank_ms", 0),
        latency.get("total_ms", 0),
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#4C78A8", "#F58518", "#54A24B"],
            text=[f"{v:.1f} ms" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Latency breakdown",
        xaxis_title="milliseconds",
        height=200,
        margin={"l": 10, "r": 40, "t": 40, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("No results returned.")
        return
    for r in results:
        with st.expander(
            f"#{r.get('rank', '?')}  {r.get('title', '(no title)')}  "
            f"— score {r.get('score', 0):.4f}",
            expanded=r.get("rank", 99) <= 3,
        ):
            st.markdown(r.get("text", ""))
            meta = r.get("metadata", {})
            if meta:
                st.json(meta)


def render(client: VectorLiftClient) -> None:
    st.header("Search Demo")
    st.markdown(
        "Type a query and select a retrieval mode.  "
        "Use **Compare modes** to run the same query across all strategies side-by-side."
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    col_q, col_mode = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Query",
            value="What are the health benefits of exercise?",
            placeholder="Enter your search query …",
        )
    with col_mode:
        mode = st.selectbox("Mode", MODES, index=2)

    col_k, col_mult, col_compare = st.columns([1, 1, 1])
    with col_k:
        top_k = st.slider("top_k", min_value=1, max_value=50, value=10)
    with col_mult:
        multiplier = st.slider(
            "Retrieval multiplier (rerank only)", min_value=1, max_value=20, value=5
        )
    with col_compare:
        compare = st.checkbox("Compare all modes", value=False)

    run = st.button("Search", type="primary", use_container_width=True)

    if not run:
        return

    if not query.strip():
        st.warning("Please enter a query.")
        return

    # ------------------------------------------------------------------
    # Single-mode search
    # ------------------------------------------------------------------
    if not compare:
        with st.spinner(f"Searching ({mode}) …"):
            try:
                t0 = time.perf_counter()
                result = client.search(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    retrieval_multiplier=multiplier,
                )
                wall_ms = (time.perf_counter() - t0) * 1_000
            except VectorLiftClientError as exc:
                st.error(f"API error {exc.status_code}: {exc.detail}")
                return
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                return

        latency = result.get("latency", {})
        st.success(
            f"Returned {len(result.get('results', []))} results in "
            f"{latency.get('total_ms', wall_ms):.1f} ms"
        )
        st.plotly_chart(_make_latency_chart(latency), use_container_width=True)
        _render_results(result.get("results", []))
        return

    # ------------------------------------------------------------------
    # Side-by-side comparison across all modes
    # ------------------------------------------------------------------
    columns = st.columns(len(MODES))
    for col, m in zip(columns, MODES):
        with col:
            st.subheader(m.upper())
            with st.spinner(f"{m} …"):
                try:
                    r = client.search(
                        query=query,
                        mode=m,
                        top_k=top_k,
                        retrieval_multiplier=multiplier,
                    )
                    lat = r.get("latency", {})
                    st.caption(f"Total: {lat.get('total_ms', 0):.1f} ms")
                    _render_results(r.get("results", []))
                except VectorLiftClientError as exc:
                    st.error(f"{exc.status_code}: {exc.detail}")
                except Exception as exc:
                    st.error(str(exc))
