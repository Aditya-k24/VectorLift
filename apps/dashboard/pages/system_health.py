"""
System Health page – live status dashboard for all VectorLift dependencies.

Features
--------
* Status badges (green / red) for Elasticsearch, Qdrant, Redis, Postgres.
* Loaded model names, checkpoint paths and embedding dims.
* Index size metrics.
* Auto-refreshes every 30 seconds using st.rerun.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from apps.dashboard.api_client import VectorLiftClient, VectorLiftClientError

_REFRESH_INTERVAL_S = 30

# Colour map for service status
_STATUS_COLOUR = {True: "#28a745", False: "#dc3545"}
_STATUS_LABEL = {True: "Healthy", False: "Unhealthy"}


def _status_badge(healthy: bool) -> str:
    colour = _STATUS_COLOUR[healthy]
    label = _STATUS_LABEL[healthy]
    return (
        f'<span style="'
        f"display:inline-block;padding:3px 10px;"
        f"border-radius:12px;background:{colour};"
        f'color:#fff;font-weight:600;font-size:0.85em;">'
        f"{label}</span>"
    )


def _render_services(services: list[dict[str, Any]]) -> None:
    cols = st.columns(len(services) or 1)
    for col, svc in zip(cols, services):
        with col:
            st.markdown(
                f"**{svc.get('name', '?').capitalize()}**  "
                + _status_badge(svc.get("healthy", False)),
                unsafe_allow_html=True,
            )
            lat = svc.get("latency_ms")
            if lat is not None:
                st.caption(f"Latency: {lat:.1f} ms")
            detail = svc.get("detail", "")
            if detail:
                st.caption(f"Detail: {detail}")


def _render_models(models: list[dict[str, Any]]) -> None:
    for m in models:
        with st.expander(m.get("name", "unknown model"), expanded=True):
            st.write(f"**Checkpoint:** `{m.get('checkpoint', 'n/a')}`")
            dim = m.get("embedding_dim")
            if dim:
                st.write(f"**Embedding dim:** {dim}")
            st.write(f"**Device:** `{m.get('device', 'n/a')}`")
            extra = m.get("extra", {})
            if extra:
                st.json(extra)


def render(client: VectorLiftClient) -> None:
    st.header("System Health")

    # Auto-refresh counter stored in session state
    if "health_last_refresh" not in st.session_state:
        st.session_state["health_last_refresh"] = time.time()

    placeholder = st.empty()

    with placeholder.container():
        # ------------------------------------------------------------------
        # Service health
        # ------------------------------------------------------------------
        st.subheader("Service Status")
        try:
            health = client.get_health()
            overall = health.get("status", "unknown")
            colour_map = {"healthy": "#28a745", "degraded": "#ffc107", "unhealthy": "#dc3545"}
            colour = colour_map.get(overall, "#6c757d")
            st.markdown(
                f"**Overall status:** "
                f'<span style="color:{colour};font-weight:700;">{overall.upper()}</span>',
                unsafe_allow_html=True,
            )
            services = health.get("services", [])
            if services:
                _render_services(services)
            else:
                st.info("No service data returned.")

            models_loaded = health.get("models_loaded", False)
            st.markdown(
                "**ML models:** " + _status_badge(models_loaded),
                unsafe_allow_html=True,
            )
        except VectorLiftClientError as exc:
            st.error(f"API error {exc.status_code}: {exc.detail}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

        st.divider()

        # ------------------------------------------------------------------
        # Model info
        # ------------------------------------------------------------------
        st.subheader("Loaded Models")
        try:
            model_info = client.get_model_info()
            models = model_info.get("models", [])
            if models:
                _render_models(models)
            else:
                st.info("No model info returned.")
        except VectorLiftClientError as exc:
            st.error(f"API error {exc.status_code}: {exc.detail}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

        st.divider()

        # ------------------------------------------------------------------
        # Refresh controls
        # ------------------------------------------------------------------
        last_refresh = st.session_state["health_last_refresh"]
        elapsed = time.time() - last_refresh
        remaining = max(0, _REFRESH_INTERVAL_S - int(elapsed))

        col_time, col_btn = st.columns([3, 1])
        with col_time:
            st.caption(f"Auto-refresh in {remaining}s (every {_REFRESH_INTERVAL_S}s)")
        with col_btn:
            manual_refresh = st.button("Refresh now")

        if manual_refresh or elapsed >= _REFRESH_INTERVAL_S:
            st.session_state["health_last_refresh"] = time.time()
            time.sleep(0.05)
            st.rerun()
