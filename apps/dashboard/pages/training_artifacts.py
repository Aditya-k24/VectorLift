"""
Training Artifacts page – inspect model checkpoints and training logs.

Features
--------
* List available model checkpoints discovered on the filesystem.
* Show training/validation loss curves parsed from JSON log files (Plotly).
* Display experiment configs (JSON).
* Download buttons for reports and configs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Default root path; override via VECTORLIFT_ARTIFACTS_DIR env var
_DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("VECTORLIFT_ARTIFACTS_DIR", "experiments"))


def _discover_checkpoints(root: Path) -> list[Path]:
    """Find all directories that look like model checkpoints."""
    candidates: list[Path] = []
    if not root.exists():
        return candidates
    for p in sorted(root.rglob("config.json")):
        candidates.append(p.parent)
    for p in sorted(root.rglob("pytorch_model.bin")):
        if p.parent not in candidates:
            candidates.append(p.parent)
    for p in sorted(root.rglob("*.safetensors")):
        if p.parent not in candidates:
            candidates.append(p.parent)
    return candidates


def _load_trainer_log(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Load Hugging Face Trainer ``trainer_state.json`` or any JSON log file."""
    candidates = [
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "training_log.json",
        checkpoint_dir / "logs.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                with path.open() as f:
                    data = json.load(f)
                # HF trainer_state.json has a "log_history" list
                if isinstance(data, dict) and "log_history" in data:
                    return data["log_history"]  # type: ignore[return-value]
                if isinstance(data, list):
                    return data  # type: ignore[return-value]
            except Exception:
                pass
    return []


def _loss_curve(log_history: list[dict[str, Any]]) -> go.Figure | None:
    """Build a training / validation loss figure from log_history entries."""
    train_steps: list[float] = []
    train_loss: list[float] = []
    eval_steps: list[float] = []
    eval_loss: list[float] = []

    for entry in log_history:
        step = entry.get("step") or entry.get("epoch", 0)
        if "loss" in entry:
            train_steps.append(float(step))
            train_loss.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_steps.append(float(step))
            eval_loss.append(float(entry["eval_loss"]))

    if not train_loss and not eval_loss:
        return None

    fig = go.Figure()
    if train_loss:
        fig.add_trace(
            go.Scatter(
                x=train_steps,
                y=train_loss,
                name="Train loss",
                mode="lines+markers",
                line=dict(color="#4C78A8"),
            )
        )
    if eval_loss:
        fig.add_trace(
            go.Scatter(
                x=eval_steps,
                y=eval_loss,
                name="Eval loss",
                mode="lines+markers",
                line=dict(color="#F58518"),
            )
        )

    fig.update_layout(
        title="Loss curves",
        xaxis_title="Step / Epoch",
        yaxis_title="Loss",
        height=380,
        margin={"l": 10, "r": 10, "t": 40, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def _load_config(checkpoint_dir: Path) -> dict[str, Any] | None:
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        try:
            with config_path.open() as f:
                return json.load(f)  # type: ignore[return-value]
        except Exception:
            pass
    return None


def render(_client: Any = None) -> None:  # noqa: ANN001
    st.header("Training Artifacts")

    # ------------------------------------------------------------------
    # Artifacts directory picker
    # ------------------------------------------------------------------
    artifacts_dir_input = st.text_input(
        "Artifacts root directory",
        value=str(_DEFAULT_ARTIFACTS_DIR.resolve()),
        help="Absolute path to the directory containing model checkpoints.",
    )
    artifacts_root = Path(artifacts_dir_input)

    if not artifacts_root.exists():
        st.warning(
            f"Directory `{artifacts_root}` does not exist.  "
            "Adjust the path above or set the `VECTORLIFT_ARTIFACTS_DIR` environment variable."
        )
        return

    # ------------------------------------------------------------------
    # Discover checkpoints
    # ------------------------------------------------------------------
    checkpoints = _discover_checkpoints(artifacts_root)

    if not checkpoints:
        st.info(
            f"No model checkpoints found under `{artifacts_root}`.  "
            "Checkpoints are detected by the presence of `config.json`, "
            "`pytorch_model.bin`, or `*.safetensors` files."
        )
        return

    checkpoint_labels = [str(p.relative_to(artifacts_root)) for p in checkpoints]
    selected_label = st.selectbox(
        f"Select checkpoint ({len(checkpoints)} found)",
        options=checkpoint_labels,
    )
    selected_dir = checkpoints[checkpoint_labels.index(selected_label)]
    st.caption(f"Full path: `{selected_dir}`")

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------
    st.subheader("Training Curves")
    log_history = _load_trainer_log(selected_dir)
    if log_history:
        fig = _loss_curve(log_history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log file found but no loss values detected.")
    else:
        st.info(
            "No training log found in this checkpoint directory.  "
            "Place a `trainer_state.json` or `training_log.json` file there."
        )

    # ------------------------------------------------------------------
    # Experiment config
    # ------------------------------------------------------------------
    st.subheader("Model Config")
    config = _load_config(selected_dir)
    if config:
        st.json(config)
    else:
        st.info("No `config.json` found.")

    # ------------------------------------------------------------------
    # File listing & download buttons
    # ------------------------------------------------------------------
    st.subheader("Files in Checkpoint")
    try:
        files = sorted(selected_dir.iterdir())
    except PermissionError:
        st.error("Permission denied when reading checkpoint directory.")
        return

    for f in files:
        size_kb = f.stat().st_size / 1024 if f.is_file() else 0
        col_name, col_size, col_dl = st.columns([3, 1, 1])
        with col_name:
            st.text(f.name)
        with col_size:
            if f.is_file():
                st.text(f"{size_kb:.1f} KB")
        with col_dl:
            if f.is_file() and f.stat().st_size < 50 * 1024 * 1024:  # limit 50 MB
                with f.open("rb") as fh:
                    st.download_button(
                        label="Download",
                        data=fh,
                        file_name=f.name,
                        key=str(f),
                    )
