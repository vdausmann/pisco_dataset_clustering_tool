#!/usr/bin/env python3
"""
PISCO Pipeline Control + Embedding Explorer
-------------------------------------------
A single Dash app that lets you configure, run, and explore the full
create_validation_dataset pipeline without touching the command line.

Run:
  python pipeline_app.py --host 0.0.0.0 --port 8060

Then open http://localhost:8060 in your browser.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

# ── Import the reusable dataset/clustering tool ──────────────────────────────
from pisco_dataset_clustering_tool import (
    run_pipeline,
    export_ecotaxa_zips,
    count_export_particles,
    discover_recent_runs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global pipeline state
# ─────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_log_lines: list[str] = []
_pipeline_state: dict = {
    "running": False,
    "error": None,
    "result": [],
    "stop_requested": False,
}
_count_state: dict = {
    "running": False,
    "message": "Use 'Count export' to preview how many particles will be exported.",
}


class _UILogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        with _state_lock:
            _log_lines.append(self.format(record))
            if len(_log_lines) > 3000:
                del _log_lines[:500]


_ui_handler = _UILogHandler()
_ui_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_ui_handler)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level explorer cache  (output_dir → DataFrame)
# ─────────────────────────────────────────────────────────────────────────────
_explorer_cache: dict[str, pd.DataFrame] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_pth(output_dir: str) -> str:
    if not output_dir or not os.path.isdir(output_dir):
        return ""
    files = sorted(
        Path(output_dir).glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(files[0]) if files else ""


def _badge(exists: bool, label: str) -> html.Span:
    color = "#16a34a" if exists else "#9ca3af"
    return html.Span(
        label,
        style={
            "background": color, "color": "white", "borderRadius": "3px",
            "padding": "1px 6px", "fontSize": "10px", "marginLeft": "5px",
            "fontWeight": "700", "whiteSpace": "nowrap",
        },
    )


def _file_status_row(label: str, path: str) -> html.Div:
    exists = bool(path) and os.path.isfile(path)
    if exists:
        sz = os.path.getsize(path)
        size_str = (
            f"{sz/1e9:.2f} GB" if sz > 1e9
            else f"{sz/1e6:.1f} MB" if sz > 1e6
            else f"{sz/1e3:.0f} KB"
        )
        badge = _badge(True, f"✓ {size_str}")
    else:
        badge = _badge(False, "✗ missing")
    return html.Div(
        [
            html.Span(
                label,
                style={"fontSize": "11.5px", "minWidth": "190px",
                       "display": "inline-block", "color": "#374151"},
            ),
            badge,
        ],
        style={"marginBottom": "3px", "display": "flex", "alignItems": "center"},
    )


def _status_file_rows(output_dir: str) -> list:
    od = output_dir or ""

    def p(name: str) -> str:
        return os.path.join(od, name) if od else ""

    rows = [
        _file_status_row("Metadata cache (.pkl)",       p("all_crops_metadata.pkl")),
        _file_status_row("Sample index (.csv)",         p("sample_index.csv")),
        _file_status_row("Features (.h5)",              p("features_nonliving.h5")),
        _file_status_row("SSL checkpoint (.pth)",       _find_latest_pth(od)),
        _file_status_row("UMAP 2D (.csv)",              p("umap_2d.csv")),
        _file_status_row("Cluster assignments (.csv)",  p("cluster_assignments.csv")),
    ]
    if od and os.path.isdir(od):
        for z in sorted(Path(od).glob("*.zip")):
            rows.append(_file_status_row(f"  {z.name}", str(z)))
    return rows


def _discover_recent_runs(anchor_output: str, limit: int = 20) -> list[dict]:
    return discover_recent_runs(anchor_output, limit=limit)


def _load_export_inputs(output_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load sample dataframe and cluster label map from a finished run directory."""
    if not output_dir:
        raise ValueError("Output directory is empty.")

    sample_path = os.path.join(output_dir, "sample_index.csv")
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Missing sample index: {sample_path}")
    df_sample = pd.read_csv(sample_path, low_memory=False)

    cluster_map: dict[str, str] = {}
    assignments_path = os.path.join(output_dir, "cluster_assignments.csv")
    if os.path.isfile(assignments_path):
        df_assign = pd.read_csv(assignments_path, low_memory=False)
        if {"object_id", "cluster_label"}.issubset(df_assign.columns):
            cluster_map = dict(
                zip(
                    df_assign["object_id"].astype(str),
                    df_assign["cluster_label"].astype(str),
                )
            )
    return df_sample, cluster_map


def _run_config_path(output_dir: str) -> str:
    return os.path.join(output_dir, "run_config.json")


def _save_run_config(output_dir: str, config: dict) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(_run_config_path(output_dir), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)


def _load_run_config(output_dir: str) -> Optional[dict]:
    path = _run_config_path(output_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


@lru_cache(maxsize=8192)
def _thumb_uri(path: str, size: int = 220) -> str:
    if not path or not os.path.isfile(path) or PILImage is None:
        return ""
    try:
        with PILImage.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _load_run_data(run_dir: str) -> Optional[pd.DataFrame]:
    if not run_dir or not os.path.isdir(run_dir):
        return None
    umap_path = Path(run_dir) / "umap_2d.csv"
    if not umap_path.exists():
        return None
    df = pd.read_csv(umap_path)
    if not {"object_id", "umap_x", "umap_y"}.issubset(df.columns):
        return None
    assign_path = Path(run_dir) / "cluster_assignments.csv"
    if assign_path.exists():
        da = pd.read_csv(assign_path)
        if "object_id" in da.columns:
            keep = [
                c for c in
                ["object_id", "image_path", "cluster_raw", "cluster_final", "cluster_label"]
                if c in da.columns
            ]
            df = df.merge(
                da[keep].drop_duplicates("object_id"),
                on="object_id", how="left", suffixes=("", "_a"),
            )
    if "cluster_label" not in df.columns:
        df["cluster_label"] = (
            df["cluster_final"].apply(
                lambda x: f"non_living/cluster_{int(x):02d}" if pd.notna(x) else "unknown"
            )
            if "cluster_final" in df.columns else "unknown"
        )
    if "image_path" not in df.columns:
        df["image_path"] = ""
    df["object_id"] = df["object_id"].astype(str)
    df["cluster_label"] = df["cluster_label"].astype(str)
    return df


def _make_umap_fig(df: pd.DataFrame, color_by: str, max_pts: int = 40000, seed: int = 42):
    plot_df = df if len(df) <= max_pts else df.sample(n=max_pts, random_state=seed)
    col = color_by if color_by in plot_df.columns else "cluster_label"
    hover: dict = {"object_id": True, "cluster_label": True, "umap_x": ":.3f", "umap_y": ":.3f"}
    for c in ["cluster_raw", "cluster_final", "image_path"]:
        hover[c] = c in plot_df.columns
    fig = px.scatter(
        plot_df, x="umap_x", y="umap_y", color=col,
        hover_data=hover, render_mode="webgl", opacity=0.80, height=680,
    )
    fig.update_traces(marker={"size": 5})
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"itemsizing": "constant"},
        uirevision="stable",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI primitives
# ─────────────────────────────────────────────────────────────────────────────
_LBL = {"fontSize": "11.5px", "fontWeight": "600", "display": "block",
        "marginBottom": "2px", "color": "#374151"}
_INP = {"width": "100%", "fontSize": "12px", "padding": "3px 7px",
        "borderRadius": "4px", "border": "1px solid #d1d5db", "boxSizing": "border-box"}
_G2 = {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px 12px"}
_SEC = {"border": "1px solid #e5e7eb", "borderRadius": "7px",
        "padding": "9px 13px 11px", "marginBottom": "9px", "background": "#f9fafb"}


def lbl(text: str) -> html.Label:
    return html.Label(text, style=_LBL)


def inp(id_: str, val, t: str = "text", step=None) -> dcc.Input:
    kw = {"step": step} if step is not None else {}
    return dcc.Input(id=id_, type=t, value=val, style=_INP, debounce=True, **kw)


def chk(id_: str, label: str, checked: bool = False) -> dcc.Checklist:
    return dcc.Checklist(
        id=id_,
        options=[{"label": f" {label}", "value": "yes"}],
        value=["yes"] if checked else [],
        style={"fontSize": "11.5px", "marginTop": "2px"},
    )


def drp(id_: str, opts: list[str], val: str) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=id_,
        options=[{"label": o, "value": o} for o in opts],
        value=val, clearable=False,
        style={**_INP, "padding": "0"},
    )


def row(*ch) -> html.Div:
    return html.Div(list(ch), style=_G2)


def col(*ch) -> html.Div:
    return html.Div(list(ch))


def sec(title: str, *ch, opened: bool = True) -> html.Details:
    return html.Details(
        [
            html.Summary(
                title,
                style={"fontWeight": "700", "fontSize": "12px", "cursor": "pointer",
                       "marginBottom": "7px", "userSelect": "none"},
            ),
            *ch,
        ],
        open=opened,
        style=_SEC,
    )


def gap(px: int = 5) -> html.Div:
    return html.Div(style={"height": f"{px}px"})


# ─────────────────────────────────────────────────────────────────────────────
# Settings panel
# ─────────────────────────────────────────────────────────────────────────────

def _settings_panel(d: dict) -> html.Div:
    return html.Div(
        [
            # I/O
            sec("📁  I / O",
                col(lbl("Source root  (--source)"),
                    inp("cfg-source", d.get("source", ""))),
                gap(),
                col(lbl("Output directory  (--output)"),
                    inp("cfg-output", d.get("output", ""))),
                gap(4),
                col(
                    lbl("Recent runs"),
                    dcc.Dropdown(
                        id="cfg-recent-runs",
                        options=_discover_recent_runs(d.get("output", "")),
                        value=None,
                        clearable=True,
                        placeholder="Select a previous run directory",
                        style={"fontSize": "12px"},
                    ),
                ),
                gap(4),
                html.Div(
                    [
                        html.Button(
                            "⟳ Refresh runs",
                            id="btn-refresh-runs",
                            n_clicks=0,
                            style={"fontSize": "11px", "padding": "2px 10px", "cursor": "pointer"},
                        ),
                        html.Button(
                            "Use selected run",
                            id="btn-use-run",
                            n_clicks=0,
                            style={"fontSize": "11px", "padding": "2px 10px", "cursor": "pointer", "marginLeft": "8px"},
                        ),
                        html.Span(
                            id="recent-runs-info",
                            children="",
                            style={"fontSize": "11px", "color": "#6b7280", "marginLeft": "8px"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                ),
                gap(),
                row(
                    col(lbl("Metadata choice"),
                        drp("cfg-meta-choice", ["auto", "crops", "ecotaxa"], "auto")),
                    col(lbl("Crop path token"),
                        inp("cfg-crop-token", "Deconv_crops")),
                ),
            ),

            # Sampling & Caching
            sec("🎲  Sampling & Caching",
                row(
                    col(lbl("# samples"), inp("cfg-n-samples", 100000, "number")),
                    col(lbl("Random seed"), inp("cfg-seed", 42, "number")),
                ),
                gap(4),
                html.Div(
                    [
                        chk("cfg-no-reuse-meta-cache", "Re-scan metadata (ignore cache)"),
                        chk("cfg-refresh-meta-cache",  "Rebuild metadata cache"),
                        chk("cfg-reuse-sample-cache",  "Reuse sample index cache"),
                        chk("cfg-refresh-sample-cache","Refresh sample index cache"),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1px 8px"},
                ),
            ),

            # SSL Training
            sec("🧠  SSL Training",
                chk("cfg-no-train", "No training — use existing checkpoint / ImageNet"),
                gap(),
                col(lbl("Checkpoint path  (blank = auto-train or ImageNet fallback)"),
                    inp("cfg-ssl-model", d.get("ssl_model_path", "") or "")),
                gap(),
                row(
                    col(lbl("Epochs"),     inp("cfg-ssl-epochs",     200, "number")),
                    col(lbl("Batch size"), inp("cfg-ssl-batch-size", 256, "number")),
                ),
                gap(4),
                row(
                    col(lbl("Learning rate"),  inp("cfg-ssl-lr", 1e-3, "number", step=1e-4)),
                    col(lbl("Input size (px)"), inp("cfg-ssl-input-size", 224, "number")),
                ),
                gap(4),
                row(
                    col(lbl("Workers"),          inp("cfg-ssl-workers",   4,      "number")),
                    col(lbl("Max train images"), inp("cfg-ssl-max-imgs", 200000, "number")),
                ),
                gap(),
                row(
                    col(lbl("Augmentation profile"),
                        drp("cfg-aug-profile", ["default", "sparse_grayscale", "size_aware_grayscale"], "default")),
                    col(lbl("Positive consistency weight"),
                        inp("cfg-pos-cons", 0.0, "number", step=0.01)),
                ),
                gap(4),
                chk("cfg-ssl-size-aware-canvas", "Size-aware canvas (preserve relative pixel size)"),
                row(
                    col(lbl("SSL canvas size (px)"), inp("cfg-ssl-canvas-size", 512, "number")),
                    col(gap(2)),
                ),
                gap(4),
                chk("cfg-fg-crop", "Foreground-aware pre-crop  (--ssl-foreground-crop)"),
                row(
                    col(lbl("Foreground threshold (0–255)"), inp("cfg-fg-thresh", 245, "number")),
                    col(lbl("Foreground min pixels"),        inp("cfg-fg-min-px", 8,   "number")),
                ),
            ),

            # Feature Extraction
            sec("🔬  Feature Extraction",
                row(
                    col(lbl("Extract batch size"), inp("cfg-extract-bs", 512, "number")),
                    col(lbl("Device  (blank = auto)"), inp("cfg-device", "")),
                ),
                gap(3),
                html.Small(
                    "⚡ Features are auto-reused from features_nonliving.h5 if it exists. "
                    "Use ⟳ Re-cluster only to skip training entirely.",
                    style={"color": "#6b7280", "fontSize": "11px", "lineHeight": "1.4"},
                ),
            ),

            # UMAP & Clustering
            sec("🗺️  UMAP & Clustering",
                row(
                    col(lbl("UMAP dims"),   inp("cfg-umap-dims", 20,  "number")),
                    col(lbl("n_neighbors"), inp("cfg-umap-nn",   30,  "number")),
                ),
                gap(4),
                row(
                    col(lbl("min_dist"),              inp("cfg-umap-md",  0.0, "number", step=0.01)),
                    col(lbl("HDBSCAN min cluster sz"), inp("cfg-hdb-mcs", 20,  "number")),
                ),
                gap(4),
                row(
                    col(lbl("min samples  (blank=auto)"),    inp("cfg-hdb-ms",  "", "number")),
                    col(lbl("cluster_sel_epsilon"),          inp("cfg-hdb-eps", 0.0,"number", step=0.01)),
                ),
                gap(4),
                row(
                    col(lbl("cluster_sel_method"),
                        drp("cfg-hdb-method", ["eom", "leaf"], "eom")),
                    col(lbl("KMeans fallback K  (blank=off)"),
                        inp("cfg-kmeans-k", "", "number")),
                ),
                gap(4),
                row(
                    col(lbl("Size feature mode"),
                        drp("cfg-size-feature-mode", ["off", "append_log_area"], "off")),
                    col(lbl("Size feature weight"),
                        inp("cfg-size-feature-weight", 0.0, "number", step=0.05)),
                ),
                gap(4),
                row(
                    col(lbl("Max clusters  (blank=off)"), inp("cfg-max-clusters", "", "number")),
                    col(gap(8), chk("cfg-keep-noise", "Keep noise as non_living/noise")),
                ),
            ),

            # Non-living labels & export (collapsed by default)
            sec("🏷️  Non-living Labels & Export",
                col(lbl("Non-living substrings  (space-separated)"),
                    inp("cfg-nl-classes", "non_living detritus")),
                gap(),
                col(lbl("Max ZIP size (MB)"), inp("cfg-max-zip", 500.0, "number")),
                gap(),
                row(
                    col(lbl("Export filter"),
                        drp("cfg-export-filter", ["all", "living", "non_living"], "all")),
                    col(lbl("Split export sets"),
                        drp("cfg-export-split", ["none", "living_non_living"], "none")),
                ),
                gap(4),
                row(
                    col(lbl("ESD min (px, optional)"), inp("cfg-export-esd-min", "", "number")),
                    col(lbl("ESD max (px, optional)"), inp("cfg-export-esd-max", "", "number")),
                ),
                gap(4),
                col(lbl("Export subfolder"), inp("cfg-export-subdir", "ecotaxa_zips_manual")),
                gap(4),
                html.Div(
                    id="export-count-info",
                    children="Use 'Count export' to preview how many particles will be exported.",
                    style={"fontSize": "11px", "color": "#374151", "lineHeight": "1.4"},
                ),
                opened=False,
            ),
        ],
        style={
            "overflowY": "auto",
            "height": "calc(100vh - 58px)",
            "paddingRight": "3px",
            "paddingBottom": "20px",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build Dash app
# ─────────────────────────────────────────────────────────────────────────────

def build_app(defaults: dict) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "PISCO Pipeline"

    def _btn(text: str, id_: str, color: str, extra: dict | None = None) -> html.Button:
        return html.Button(
            text, id=id_, n_clicks=0,
            style={
                "padding": "5px 16px", "borderRadius": "5px", "border": "none",
                "fontWeight": "700", "fontSize": "12px", "cursor": "pointer",
                "background": color, "color": "white",
                **(extra or {}),
            },
        )

    # ── Tab styles ────────────────────────────────────────────────────────────
    _TAB_ACTIVE = {
        "border": "none", "borderBottom": "3px solid #2563eb",
        "background": "transparent", "fontWeight": "700", "fontSize": "13px",
        "cursor": "pointer", "padding": "6px 18px", "color": "#2563eb",
    }
    _TAB_IDLE = {
        "border": "none", "borderBottom": "3px solid transparent",
        "background": "transparent", "fontWeight": "600", "fontSize": "13px",
        "cursor": "pointer", "padding": "6px 18px", "color": "#6b7280",
    }

    app.layout = html.Div(
        [
            # ── Top bar ──────────────────────────────────────────────────
            html.Div(
                [
                    html.Span(
                        "PISCO Pipeline",
                        style={"fontWeight": "800", "fontSize": "17px", "color": "#1e3a8a"},
                    ),
                    html.Div(
                        [
                            _btn("▶ Run pipeline",    "btn-run",       "#2563eb"),
                            _btn("⟳ Re-cluster only", "btn-recluster", "#16a34a",
                                 {"marginLeft": "8px"}),
                            _btn("⏹ Stop", "btn-stop", "#dc2626",
                                 {"marginLeft": "8px"}),
                           _btn("📦 Export", "btn-export", "#7c3aed",
                               {"marginLeft": "8px"}),
                           _btn("🔢 Count export", "btn-count-export", "#475569",
                               {"marginLeft": "8px"}),
                            html.Span(
                                id="pipeline-badge",
                                children="● idle",
                                style={"marginLeft": "14px", "fontSize": "12px",
                                       "fontWeight": "700", "color": "#6b7280"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "center", "padding": "8px 16px",
                    "background": "#eff6ff", "borderBottom": "2px solid #bfdbfe",
                    "position": "sticky", "top": "0", "zIndex": "100",
                },
            ),

            # ── Body ─────────────────────────────────────────────────────
            html.Div(
                [
                    # ── Left sidebar: settings ────────────────────────────
                    html.Div(
                        _settings_panel(defaults),
                        style={
                            "width": "330px", "minWidth": "280px",
                            "borderRight": "1px solid #e5e7eb",
                            "padding": "10px 8px 0 10px",
                            "background": "#ffffff",
                            "flexShrink": "0",
                        },
                    ),

                    # ── Right: tab bar + content ──────────────────────────
                    html.Div(
                        [
                            # Tab bar
                            html.Div(
                                [
                                    html.Button("📊 Status & Log", id="tab-btn-status",
                                                n_clicks=0, style=_TAB_ACTIVE),
                                    html.Button("🗺️ Explorer", id="tab-btn-explorer",
                                                n_clicks=0, style=_TAB_IDLE),
                                ],
                                style={"borderBottom": "1px solid #e5e7eb",
                                       "background": "#f9fafb"},
                            ),

                            # ── Status & Log (always in DOM) ──────────────
                            html.Div(
                                id="panel-status",
                                children=[
                                    html.Div(
                                        id="status-file-list",
                                        children=_status_file_rows(""),
                                        style={"padding": "10px 14px",
                                               "background": "#f9fafb",
                                               "borderBottom": "1px solid #e5e7eb"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("Log output",
                                                              style={"fontWeight": "700",
                                                                     "fontSize": "12px"}),
                                                    html.Button(
                                                        "Clear", id="btn-clear-log", n_clicks=0,
                                                        style={"marginLeft": "10px",
                                                               "fontSize": "11px",
                                                               "padding": "1px 8px",
                                                               "cursor": "pointer",
                                                               "border": "1px solid #d1d5db",
                                                               "borderRadius": "4px"},
                                                    ),
                                                ],
                                                style={"marginBottom": "5px"},
                                            ),
                                            html.Pre(
                                                id="log-output",
                                                children="",
                                                style={
                                                    "background": "#111827",
                                                    "color": "#a7f3d0",
                                                    "fontSize": "11px",
                                                    "lineHeight": "1.5",
                                                    "padding": "9px 12px",
                                                    "borderRadius": "5px",
                                                    "overflowY": "auto",
                                                    "height": "calc(100vh - 228px)",
                                                    "fontFamily": "monospace",
                                                    "whiteSpace": "pre-wrap",
                                                    "wordBreak": "break-all",
                                                },
                                            ),
                                        ],
                                        style={"padding": "10px 14px"},
                                    ),
                                ],
                                style={"display": "block"},
                            ),

                            # ── Explorer (always in DOM, initially hidden) ──
                            html.Div(
                                id="panel-explorer",
                                children=[
                                    # Explorer toolbar
                                    html.Div(
                                        [
                                            _btn("⟳ Load / Reload from output dir",
                                                 "btn-load-explorer", "#2563eb",
                                                 {"padding": "4px 12px", "fontSize": "11.5px",
                                                  "marginRight": "12px"}),
                                            html.Label(
                                                "Color by:",
                                                style={"fontSize": "12px", "fontWeight": "600",
                                                       "marginRight": "5px"},
                                            ),
                                            dcc.Dropdown(
                                                id="explorer-color-by",
                                                options=[{"label": "cluster_label",
                                                          "value": "cluster_label"}],
                                                value="cluster_label",
                                                clearable=False,
                                                style={"width": "200px", "fontSize": "12px"},
                                            ),
                                            html.Span(
                                                id="explorer-info",
                                                style={"marginLeft": "12px", "fontSize": "11.5px",
                                                       "color": "#6b7280"},
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "center",
                                               "padding": "8px 12px",
                                               "borderBottom": "1px solid #e5e7eb",
                                               "background": "#f9fafb"},
                                    ),
                                    # UMAP + side panel
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="umap-graph",
                                                figure=px.scatter(
                                                    title=(
                                                        "Click 'Load / Reload' to display the UMAP"
                                                    )
                                                ),
                                                style={"height": "calc(100vh - 108px)",
                                                       "flex": "1"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Selection",
                                                        style={"margin": "0 0 8px",
                                                               "fontSize": "13px"},
                                                    ),
                                                    html.Img(
                                                        id="preview-img", src="",
                                                        style={"maxWidth": "100%",
                                                               "border": "1px solid #ddd",
                                                               "borderRadius": "4px",
                                                               "marginBottom": "8px"},
                                                    ),
                                                    html.Pre(
                                                        id="selection-meta",
                                                        children="Hover or click a point.",
                                                        style={
                                                            "whiteSpace": "pre-wrap",
                                                            "fontSize": "11px",
                                                            "color": "#374151",
                                                            "overflowY": "auto",
                                                            "maxHeight": "calc(100vh - 350px)",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "width": "220px", "flexShrink": "0",
                                                    "padding": "10px 10px",
                                                    "borderLeft": "1px solid #e5e7eb",
                                                    "overflowY": "auto",
                                                    "height": "calc(100vh - 108px)",
                                                },
                                            ),
                                        ],
                                        style={"display": "flex"},
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                        ],
                        style={
                            "flex": "1", "minWidth": "0",
                            "display": "flex", "flexDirection": "column",
                            "overflow": "hidden",
                        },
                    ),
                ],
                style={"display": "flex", "height": "calc(100vh - 46px)", "overflow": "hidden"},
            ),

            # ── Hidden stores + polling interval ─────────────────────────
            dcc.Store(id="store-log-idx", data=0),
            dcc.Interval(id="interval-tick", interval=1200, n_intervals=0),
        ],
        style={"fontFamily": "system-ui, -apple-system, sans-serif",
               "margin": "0", "padding": "0"},
    )

    # ── Tab switch ───────────────────────────────────────────────────────────
    @app.callback(
        Output("panel-status",    "style"),
        Output("panel-explorer",  "style"),
        Output("tab-btn-status",  "style"),
        Output("tab-btn-explorer","style"),
        Input("tab-btn-status",   "n_clicks"),
        Input("tab-btn-explorer", "n_clicks"),
    )
    def switch_tab(_s, _e):
        ctx = callback_context
        to_explorer = (
            ctx.triggered
            and "tab-btn-explorer" in ctx.triggered[0]["prop_id"]
        )
        if to_explorer:
            return {"display": "none"}, {"display": "block"}, _TAB_IDLE, _TAB_ACTIVE
        return {"display": "block"}, {"display": "none"}, _TAB_ACTIVE, _TAB_IDLE

    @app.callback(
        Output("cfg-recent-runs", "options"),
        Output("cfg-recent-runs", "value"),
        Input("btn-refresh-runs", "n_clicks"),
        Input("cfg-output", "value"),
        State("cfg-recent-runs", "value"),
    )
    def refresh_recent_runs(_refresh_clicks, output_dir, selected_value):
        options = _discover_recent_runs(output_dir or "")
        values = {opt["value"] for opt in options}
        if selected_value in values:
            chosen = selected_value
        else:
            chosen = options[0]["value"] if options else None
        return options, chosen

    @app.callback(
        Output("cfg-output", "value"),
        Output("recent-runs-info", "children"),
        Output("cfg-umap-dims", "value"),
        Output("cfg-umap-nn", "value"),
        Output("cfg-umap-md", "value"),
        Output("cfg-hdb-mcs", "value"),
        Output("cfg-hdb-ms", "value"),
        Output("cfg-hdb-eps", "value"),
        Output("cfg-hdb-method", "value"),
        Output("cfg-size-feature-mode", "value"),
        Output("cfg-size-feature-weight", "value"),
        Output("cfg-keep-noise", "value"),
        Output("cfg-kmeans-k", "value"),
        Output("cfg-max-clusters", "value"),
        Output("cfg-aug-profile", "value"),
        Output("cfg-ssl-size-aware-canvas", "value"),
        Output("cfg-ssl-canvas-size", "value"),
        Input("btn-use-run", "n_clicks"),
        State("cfg-recent-runs", "value"),
        State("cfg-recent-runs", "options"),
        prevent_initial_call=True,
    )
    def load_recent_run(_n, selected_run, options):
        use_path = selected_run
        if not use_path and options:
            use_path = options[0]["value"]
        if not use_path:
            return (
                no_update, "No previous runs found.",
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update,
                no_update, no_update,
                no_update, no_update, no_update,
            )
        if not os.path.isdir(str(use_path)):
            return (
                no_update, f"Run path does not exist: {use_path}",
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update,
                no_update, no_update,
                no_update, no_update, no_update,
            )

        cfg = _load_run_config(str(use_path)) or {}
        keep_noise_val = ["yes"] if bool(cfg.get("keep_noise_as_noise", False)) else []
        ssl_size_canvas_val = ["yes"] if bool(cfg.get("ssl_size_aware_canvas", False)) else []

        return (
            str(use_path),
            f"Loaded run: {use_path}",
            cfg.get("umap_dims", 20),
            cfg.get("umap_n_neighbors", 30),
            cfg.get("umap_min_dist", 0.0),
            cfg.get("min_cluster_size", 20),
            cfg.get("hdbscan_min_samples", ""),
            cfg.get("hdbscan_cluster_selection_epsilon", 0.0),
            cfg.get("hdbscan_cluster_selection_method", "eom"),
            cfg.get("size_feature_mode", "off"),
            cfg.get("size_feature_weight", 0.0),
            keep_noise_val,
            cfg.get("fallback_kmeans_k", ""),
            cfg.get("max_clusters", ""),
            cfg.get("ssl_aug_profile", "default"),
            ssl_size_canvas_val,
            cfg.get("ssl_canvas_size", 512),
        )

    # ── Run pipeline ─────────────────────────────────────────────────────────
    @app.callback(
        Output("btn-run",       "disabled"),
        Output("btn-recluster", "disabled"),
        Input("btn-run",        "n_clicks"),
        Input("btn-recluster",  "n_clicks"),
        State("cfg-source",              "value"),
        State("cfg-output",              "value"),
        State("cfg-meta-choice",         "value"),
        State("cfg-crop-token",          "value"),
        State("cfg-no-reuse-meta-cache", "value"),
        State("cfg-refresh-meta-cache",  "value"),
        State("cfg-reuse-sample-cache",  "value"),
        State("cfg-refresh-sample-cache","value"),
        State("cfg-n-samples",           "value"),
        State("cfg-seed",                "value"),
        State("cfg-no-train",            "value"),
        State("cfg-ssl-model",           "value"),
        State("cfg-ssl-epochs",          "value"),
        State("cfg-ssl-batch-size",      "value"),
        State("cfg-ssl-lr",              "value"),
        State("cfg-ssl-input-size",      "value"),
        State("cfg-ssl-workers",         "value"),
        State("cfg-ssl-max-imgs",        "value"),
        State("cfg-aug-profile",         "value"),
        State("cfg-fg-crop",             "value"),
        State("cfg-fg-thresh",           "value"),
        State("cfg-fg-min-px",           "value"),
        State("cfg-pos-cons",            "value"),
        State("cfg-ssl-size-aware-canvas", "value"),
        State("cfg-ssl-canvas-size", "value"),
        State("cfg-extract-bs",          "value"),
        State("cfg-device",              "value"),
        State("cfg-umap-dims",           "value"),
        State("cfg-umap-nn",             "value"),
        State("cfg-umap-md",             "value"),
        State("cfg-hdb-mcs",             "value"),
        State("cfg-hdb-ms",              "value"),
        State("cfg-hdb-eps",             "value"),
        State("cfg-hdb-method",          "value"),
        State("cfg-size-feature-mode",   "value"),
        State("cfg-size-feature-weight", "value"),
        State("cfg-kmeans-k",            "value"),
        State("cfg-max-clusters",        "value"),
        State("cfg-keep-noise",          "value"),
        State("cfg-nl-classes",          "value"),
        State("cfg-max-zip",             "value"),
        prevent_initial_call=True,
    )
    def handle_run(
        _nr, _nc,
        source, output_dir, meta_choice, crop_token,
        no_reuse_meta, refresh_meta, reuse_sample, refresh_sample,
        n_samples, seed,
        no_train_chk, ssl_model,
        ssl_epochs, ssl_bs, ssl_lr, ssl_input, ssl_workers, ssl_max_imgs,
        aug_profile, fg_crop, fg_thresh, fg_min_px, pos_cons,
        ssl_size_canvas_chk, ssl_canvas_size,
        extract_bs, device_val,
        umap_dims, umap_nn, umap_md,
        hdb_mcs, hdb_ms, hdb_eps, hdb_method,
        size_feature_mode, size_feature_weight,
        kmeans_k, max_clusters, keep_noise,
        nl_classes_str, max_zip,
    ):
        with _state_lock:
            if _pipeline_state["running"]:
                return True, True

        ctx = callback_context
        is_recluster = (
            ctx.triggered
            and "btn-recluster" in ctx.triggered[0]["prop_id"]
        )

        if not source or not output_dir:
            logging.getLogger().warning(
                "Cannot start: source and output directories must be set."
            )
            return False, False

        def _i(v, d=None):
            try:
                return int(v) if v not in (None, "") else d
            except Exception:
                return d

        def _f(v, d=0.0):
            try:
                return float(v) if v not in (None, "") else d
            except Exception:
                return d

        def _c(v):
            return bool(v)

        no_train = _c(no_train_chk) or is_recluster

        # For re-cluster, auto-discover latest checkpoint in output dir
        _ssl_model = ssl_model or ""
        if is_recluster and not _ssl_model:
            _ssl_model = _find_latest_pth(output_dir)

        kwargs = dict(
            source_root=source,
            output_dir=output_dir,
            metadata_choice=meta_choice or "auto",
            crop_path_token=crop_token or None,
            reuse_metadata_cache=not _c(no_reuse_meta),
            refresh_metadata_cache=_c(refresh_meta),
            reuse_sample_index_cache=_c(reuse_sample),
            refresh_sample_index_cache=_c(refresh_sample),
            n_samples=_i(n_samples, 100000),
            seed=_i(seed, 42),
            non_living_classes=tuple((nl_classes_str or "non_living detritus").split()),
            ssl_model_path=_ssl_model or None,
            no_train=no_train,
            ssl_epochs=_i(ssl_epochs, 200),
            ssl_batch_size=_i(ssl_bs, 256),
            ssl_lr=_f(ssl_lr, 1e-3),
            ssl_input_size=_i(ssl_input, 224),
            ssl_num_workers=_i(ssl_workers, 4),
            ssl_train_max_images=_i(ssl_max_imgs) or None,
            ssl_aug_profile=aug_profile or "default",
            ssl_foreground_crop=_c(fg_crop),
            ssl_foreground_threshold=_i(fg_thresh, 245),
            ssl_foreground_min_pixels=_i(fg_min_px, 8),
            ssl_positive_consistency_weight=_f(pos_cons, 0.0),
            ssl_size_aware_canvas=_c(ssl_size_canvas_chk),
            ssl_canvas_size=_i(ssl_canvas_size, 512),
            umap_dims=_i(umap_dims, 20),
            umap_n_neighbors=_i(umap_nn, 30),
            umap_min_dist=_f(umap_md, 0.0),
            min_cluster_size=_i(hdb_mcs, 20),
            hdbscan_min_samples=_i(hdb_ms),
            hdbscan_cluster_selection_epsilon=_f(hdb_eps, 0.0),
            hdbscan_cluster_selection_method=hdb_method or "eom",
            size_feature_mode=size_feature_mode or "off",
            size_feature_weight=_f(size_feature_weight, 0.0),
            fallback_kmeans_k=_i(kmeans_k),
            keep_noise_as_noise=_c(keep_noise),
            max_clusters=_i(max_clusters),
            max_zip_mb=_f(max_zip, 500.0),
            do_export=False,
            device=device_val or None,
            extract_batch_size=_i(extract_bs, 512),
        )

        run_config = dict(
            metadata_choice=meta_choice or "auto",
            crop_path_token=crop_token or None,
            n_samples=_i(n_samples, 100000),
            seed=_i(seed, 42),
            ssl_aug_profile=aug_profile or "default",
            ssl_size_aware_canvas=_c(ssl_size_canvas_chk),
            ssl_canvas_size=_i(ssl_canvas_size, 512),
            umap_dims=_i(umap_dims, 20),
            umap_n_neighbors=_i(umap_nn, 30),
            umap_min_dist=_f(umap_md, 0.0),
            min_cluster_size=_i(hdb_mcs, 20),
            hdbscan_min_samples=_i(hdb_ms),
            hdbscan_cluster_selection_epsilon=_f(hdb_eps, 0.0),
            hdbscan_cluster_selection_method=hdb_method or "eom",
            size_feature_mode=size_feature_mode or "off",
            size_feature_weight=_f(size_feature_weight, 0.0),
            fallback_kmeans_k=_i(kmeans_k),
            max_clusters=_i(max_clusters),
            keep_noise_as_noise=_c(keep_noise),
        )
        try:
            _save_run_config(output_dir, run_config)
        except Exception as exc:
            logging.getLogger().warning(f"Could not save run config: {exc}")

        def _is_stop_requested() -> bool:
            with _state_lock:
                return bool(_pipeline_state.get("stop_requested", False))

        def _run():
            with _state_lock:
                _pipeline_state.update(running=True, error=None, result=[], stop_requested=False)
            label = "Re-cluster only" if is_recluster else "Full pipeline"
            logging.getLogger().info(f"=== {label} started ===")
            try:
                zips = run_pipeline(stop_requested=_is_stop_requested, **kwargs)
                with _state_lock:
                    _pipeline_state["result"] = zips or []
                logging.getLogger().info("=== Pipeline finished (export not run) ===")
            except InterruptedError as exc:
                logging.getLogger().info(f"=== Pipeline stopped by user: {exc} ===")
                with _state_lock:
                    _pipeline_state["error"] = None
            except Exception as exc:
                import traceback
                logging.getLogger().error(
                    f"Pipeline FAILED: {exc}\n{traceback.format_exc()}"
                )
                with _state_lock:
                    _pipeline_state["error"] = str(exc)
            finally:
                with _state_lock:
                    _pipeline_state["running"] = False
                    _pipeline_state["stop_requested"] = False

        threading.Thread(target=_run, daemon=True).start()
        return True, True

    # ── Polling tick ─────────────────────────────────────────────────────────
    @app.callback(
        Output("log-output",       "children"),
        Output("store-log-idx",    "data"),
        Output("status-file-list", "children"),
        Output("pipeline-badge",   "children"),
        Output("pipeline-badge",   "style"),
        Output("btn-run",          "disabled", allow_duplicate=True),
        Output("btn-recluster",    "disabled", allow_duplicate=True),
        Input("interval-tick",     "n_intervals"),
        State("store-log-idx",     "data"),
        State("log-output",        "children"),
        State("cfg-output",        "value"),
        prevent_initial_call=True,
    )
    def tick(_, log_idx, cur_log, output_dir):
        with _state_lock:
            running   = _pipeline_state["running"]
            error     = _pipeline_state["error"]
            stop_req  = bool(_pipeline_state.get("stop_requested", False))
            new_lines = _log_lines[log_idx:]
            new_idx   = len(_log_lines)

        appended = (cur_log or "") + (
            "\n".join(new_lines) + "\n" if new_lines else ""
        )

        if running and stop_req:
            badge_txt   = "⏹ stop requested…"
            badge_style = {"marginLeft": "14px", "fontSize": "12px",
                           "fontWeight": "700", "color": "#dc2626"}
        elif running:
            badge_txt   = "⏳ running…"
            badge_style = {"marginLeft": "14px", "fontSize": "12px",
                           "fontWeight": "700", "color": "#d97706"}
        elif error:
            badge_txt   = "❌ error"
            badge_style = {"marginLeft": "14px", "fontSize": "12px",
                           "fontWeight": "700", "color": "#dc2626"}
        else:
            badge_txt   = "● idle"
            badge_style = {"marginLeft": "14px", "fontSize": "12px",
                           "fontWeight": "700", "color": "#6b7280"}

        return (
            appended, new_idx,
            _status_file_rows(output_dir or ""),
            badge_txt, badge_style,
            running, running,
        )

    @app.callback(
        Output("btn-export", "disabled"),
        Input("interval-tick", "n_intervals"),
    )
    def sync_export_button(_n):
        with _state_lock:
            return bool(_pipeline_state["running"])

    @app.callback(
        Output("btn-stop", "disabled"),
        Input("interval-tick", "n_intervals"),
    )
    def sync_stop_button(_n):
        with _state_lock:
            return not bool(_pipeline_state["running"])

    @app.callback(
        Output("pipeline-badge", "children", allow_duplicate=True),
        Input("btn-stop", "n_clicks"),
        prevent_initial_call=True,
    )
    def request_stop(_n):
        with _state_lock:
            if not _pipeline_state.get("running", False):
                return no_update
            _pipeline_state["stop_requested"] = True
        logging.getLogger().info("Stop requested by user. Waiting for current step to interrupt safely ...")
        return "⏹ stop requested…"

    # ── Clear log ────────────────────────────────────────────────────────────
    @app.callback(
        Output("log-output",    "children", allow_duplicate=True),
        Output("store-log-idx", "data",     allow_duplicate=True),
        Input("btn-clear-log",  "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_log(_):
        with _state_lock:
            _log_lines.clear()
        return "", 0

    # ── Export count preview ────────────────────────────────────────────────
    @app.callback(
        Output("export-count-info", "children"),
        Output("btn-count-export", "disabled"),
        Input("btn-count-export", "n_clicks"),
        State("cfg-output", "value"),
        State("cfg-export-filter", "value"),
        State("cfg-export-esd-min", "value"),
        State("cfg-export-esd-max", "value"),
        State("cfg-nl-classes", "value"),
        prevent_initial_call=True,
    )
    def preview_export_count(_n, output_dir, export_filter, esd_min, esd_max, nl_classes_str):
        with _state_lock:
            if _count_state["running"]:
                return "Count already running…", True

        if not output_dir:
            return "Set an output directory first.", False

        classes = tuple((nl_classes_str or "non_living detritus").split())

        def _run_count():
            with _state_lock:
                _count_state["running"] = True
                _count_state["message"] = "Counting export candidates…"
            try:
                df_sample, cluster_map = _load_export_inputs(output_dir)
                counts = count_export_particles(
                    df_sample=df_sample,
                    cluster_labels_map=cluster_map,
                    non_living_classes=classes,
                    category_filter=export_filter or "all",
                    esd_min=float(esd_min) if esd_min not in (None, "") else None,
                    esd_max=float(esd_max) if esd_max not in (None, "") else None,
                    check_source_exists=False,
                )
                with _state_lock:
                    _count_state["message"] = (
                        f"Will export {counts['total']:,} particles "
                        f"(living={counts['living']:,}, non_living={counts['non_living']:,}) "
                        f"for filter='{export_filter or 'all'}', esd_min={esd_min}, esd_max={esd_max}. "
                        f"(fast preview; missing-file check is skipped)"
                    )
            except Exception as exc:
                with _state_lock:
                    _count_state["message"] = f"Count failed: {exc}"
            finally:
                with _state_lock:
                    _count_state["running"] = False

        threading.Thread(target=_run_count, daemon=True).start()
        return "Counting export candidates…", True

    @app.callback(
        Output("export-count-info", "children", allow_duplicate=True),
        Output("btn-count-export", "disabled", allow_duplicate=True),
        Input("interval-tick", "n_intervals"),
        prevent_initial_call=True,
    )
    def sync_export_count_feedback(_n):
        with _state_lock:
            msg = str(_count_state.get("message", ""))
            running = bool(_count_state.get("running", False))
        return msg, running

    # ── Manual export ───────────────────────────────────────────────────────
    @app.callback(
        Output("export-count-info", "children", allow_duplicate=True),
        Input("btn-export", "n_clicks"),
        State("cfg-output", "value"),
        State("cfg-export-subdir", "value"),
        State("cfg-max-zip", "value"),
        State("cfg-export-filter", "value"),
        State("cfg-export-split", "value"),
        State("cfg-export-esd-min", "value"),
        State("cfg-export-esd-max", "value"),
        State("cfg-nl-classes", "value"),
        prevent_initial_call=True,
    )
    def run_manual_export(
        _n,
        output_dir,
        export_subdir,
        max_zip_mb,
        export_filter,
        split_mode,
        esd_min,
        esd_max,
        nl_classes_str,
    ):
        with _state_lock:
            if _pipeline_state["running"]:
                return "Another background task is running. Please wait."

        if not output_dir:
            return "Set an output directory first."

        export_dir = os.path.join(output_dir, export_subdir or "ecotaxa_zips_manual")
        classes = tuple((nl_classes_str or "non_living detritus").split())

        def _run_export():
            with _state_lock:
                _pipeline_state.update(running=True, error=None)
            try:
                logging.getLogger().info("=== Manual export started ===")
                df_sample, cluster_map = _load_export_inputs(output_dir)
                zip_paths = export_ecotaxa_zips(
                    df_sample=df_sample,
                    cluster_labels_map=cluster_map,
                    output_dir=export_dir,
                    max_mb=float(max_zip_mb) if max_zip_mb not in (None, "") else 500.0,
                    non_living_classes=classes,
                    category_filter=export_filter or "all",
                    esd_min=float(esd_min) if esd_min not in (None, "") else None,
                    esd_max=float(esd_max) if esd_max not in (None, "") else None,
                    split_mode=split_mode or "none",
                )
                logging.getLogger().info(
                    f"=== Manual export finished — {len(zip_paths)} ZIP(s) in {export_dir} ==="
                )
            except Exception as exc:
                import traceback
                logging.getLogger().error(
                    f"Manual export FAILED: {exc}\n{traceback.format_exc()}"
                )
                with _state_lock:
                    _pipeline_state["error"] = str(exc)
            finally:
                with _state_lock:
                    _pipeline_state["running"] = False

        threading.Thread(target=_run_export, daemon=True).start()
        return "Manual export started. Check Log output for progress and final ZIP count."

    # ── Load / update UMAP ───────────────────────────────────────────────────
    @app.callback(
        Output("umap-graph",        "figure"),
        Output("explorer-color-by", "options"),
        Output("explorer-color-by", "value"),
        Output("explorer-info",     "children"),
        Input("btn-load-explorer",  "n_clicks"),
        Input("explorer-color-by",  "value"),
        State("cfg-output",         "value"),
        prevent_initial_call=True,
    )
    def update_explorer(n_load, color_by, output_dir):
        ctx = callback_context
        is_load = ctx.triggered and "btn-load-explorer" in ctx.triggered[0]["prop_id"]

        if is_load:
            df = _load_run_data(output_dir or "")
            if df is None:
                return (
                    px.scatter(title="No umap_2d.csv found — run the pipeline first"),
                    [{"label": "cluster_label", "value": "cluster_label"}],
                    "cluster_label",
                    "Not loaded",
                )
            _explorer_cache[output_dir] = df
            cols = [c for c in ["cluster_label", "cluster_final", "cluster_raw"]
                    if c in df.columns]
            val  = cols[0] if cols else "cluster_label"
            info = (
                f"{len(df):,} points  |  "
                f"{df['cluster_label'].nunique()} clusters  |  "
                f"{output_dir}"
            )
            return (
                _make_umap_fig(df, val),
                [{"label": c, "value": c} for c in (cols or ["cluster_label"])],
                val,
                info,
            )

        df = _explorer_cache.get(output_dir or "")
        if df is None:
            return no_update, no_update, no_update, no_update
        return _make_umap_fig(df, color_by or "cluster_label"), no_update, no_update, no_update

    # ── Image preview ────────────────────────────────────────────────────────
    @app.callback(
        Output("preview-img",    "src"),
        Output("selection-meta", "children"),
        Input("umap-graph",  "clickData"),
        Input("umap-graph",  "hoverData"),
        State("cfg-output",  "value"),
        prevent_initial_call=True,
    )
    def show_point(click_data, hover_data, output_dir):
        data = click_data if click_data and click_data.get("points") else hover_data
        if not data or not data.get("points"):
            return "", "Hover or click a point."

        point  = data["points"][0]
        custom = point.get("customdata", [])
        df     = _explorer_cache.get(output_dir or "")

        image_path = cluster_label = cluster_raw = cluster_final = object_id = ""
        if df is not None and custom:
            object_id = str(custom[0]) if len(custom) > 0 else ""
            if object_id:
                m = df[df["object_id"] == object_id]
                if not m.empty:
                    r = m.iloc[0]
                    image_path    = str(r.get("image_path",   "") or "")
                    cluster_label = str(r.get("cluster_label",""))
                    cluster_raw   = str(r.get("cluster_raw",  ""))
                    cluster_final = str(r.get("cluster_final",""))

        meta = (
            f"object_id:     {object_id}\n"
            f"cluster_label: {cluster_label}\n"
            f"cluster_raw:   {cluster_raw}\n"
            f"cluster_final: {cluster_final}\n"
            f"umap_x:        {point.get('x', 0):.4f}\n"
            f"umap_y:        {point.get('y', 0):.4f}\n"
            f"image_exists:  {os.path.isfile(image_path) if image_path else False}\n"
            f"image_path:\n  {image_path}"
        )
        return _thumb_uri(image_path), meta

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PISCO Pipeline Control + Explorer")
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    type=int, default=8060)
    parser.add_argument("--debug",   action="store_true")
    parser.add_argument("--source",  default="", help="Pre-fill source directory")
    parser.add_argument("--output",  default="", help="Pre-fill output directory")
    parser.add_argument("--run-dir", default="", help="Alias for --output (Explorer compat)")
    args = parser.parse_args()

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.WARNING)
    werkzeug_logger.propagate = False

    defaults = {"source": args.source, "output": args.output or args.run_dir}
    app = build_app(defaults)
    print(f"\n  Open  http://{args.host}:{args.port}  in your browser\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
