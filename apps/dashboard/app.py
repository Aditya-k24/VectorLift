"""
VectorLift Dashboard – main Streamlit entry point.

Navigation
----------
Search Demo         – live query interface with mode comparison.
Experiment Comparison – side-by-side metric tables and significance tests.
Query Analysis      – per-query diagnostics and failure analysis.
System Health       – service status badges and model info (auto-refresh).
Training Artifacts  – model checkpoints, loss curves and config viewer.

Run with:
    streamlit run apps/dashboard/app.py
"""

from __future__ import annotations

import os

import streamlit as st

from apps.dashboard.api_client import VectorLiftClient
from apps.dashboard.pages import (
    experiment_comparison,
    query_analysis,
    search_demo,
    system_health,
    training_artifacts,
)

# ---------------------------------------------------------------------------
# Page configuration (must be called before any other Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VectorLift Dashboard",
    page_icon="magnifying-glass",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/isomerai/vectorlift",
        "Report a bug": "https://github.com/isomerai/vectorlift/issues",
        "About": "VectorLift – Production Semantic Search & Ranking Engine",
    },
)

# ---------------------------------------------------------------------------
# Session state: shared API client
# ---------------------------------------------------------------------------

_DEFAULT_API_URL = os.environ.get("VECTORLIFT_API_URL", "http://localhost:8000")

if "api_client" not in st.session_state:
    st.session_state["api_client"] = VectorLiftClient(base_url=_DEFAULT_API_URL)

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = _DEFAULT_API_URL


# ---------------------------------------------------------------------------
# Sidebar – branding + navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("VectorLift")
    st.caption("Semantic Search & Ranking Engine")
    st.divider()

    # API URL override
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state["api_base_url"],
        help="URL of the running FastAPI backend.",
    )
    if api_url != st.session_state["api_base_url"]:
        st.session_state["api_base_url"] = api_url
        # Recreate client with new URL
        old_client: VectorLiftClient = st.session_state["api_client"]
        old_client.close()
        st.session_state["api_client"] = VectorLiftClient(base_url=api_url)
        st.success("API URL updated.")

    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "Search Demo",
            "Experiment Comparison",
            "Query Analysis",
            "System Health",
            "Training Artifacts",
        ],
        label_visibility="collapsed",
    )

client: VectorLiftClient = st.session_state["api_client"]

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "Search Demo":
    search_demo.render(client)

elif page == "Experiment Comparison":
    experiment_comparison.render(client)

elif page == "Query Analysis":
    query_analysis.render(client)

elif page == "System Health":
    system_health.render(client)

elif page == "Training Artifacts":
    training_artifacts.render(client)
