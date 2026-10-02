# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generate benchmark comparison charts using matplotlib."""

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Theme dicts (passed to plt.style.context) ─────────
_DARK_THEME: dict[str, Any] = {
    "figure.facecolor": "#181818",
    "axes.facecolor": "#212121",
    "axes.edgecolor": "#3a3a3a",
    "axes.labelcolor": "#b0b0b0",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#e0e0e0",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "grid.color": "#2e2e2e",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.prop_cycle": "cycler('color', ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#8b5cf6'])",
}

_LIGHT_THEME: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#ccc",
    "axes.labelcolor": "#333",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#222",
    "xtick.color": "#555",
    "ytick.color": "#555",
    "grid.color": "#eee",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.prop_cycle": "cycler('color', ['#16a34a', '#2563eb', '#d97706', '#dc2626', '#9333ea', '#0891b2', '#ea580c', '#db2777', '#0d9488', '#7c3aed'])",
}

_DARK_THEMES = {"dark", "unsloth-dark"}
_LIGHT_THEMES = {"light"}

# Color palettes for bars/lines (used when theme doesn't set prop_cycle)
_DARK_PALETTE = [
    "#22c55e",
    "#3b82f6",
    "#f59e0b",
    "#ef4444",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#ec4899",
    "#14b8a6",
    "#8b5cf6",
]
_LIGHT_PALETTE = [
    "#16a34a",
    "#2563eb",
    "#d97706",
    "#dc2626",
    "#9333ea",
    "#0891b2",
    "#ea580c",
    "#db2777",
    "#0d9488",
    "#7c3aed",
]


def _get_palette(theme: str) -> list[str]:
    """Return a color palette for built-in themes."""
    if theme in _LIGHT_THEMES:
        return _LIGHT_PALETTE
    return _DARK_PALETTE


def _resolve_palette(theme: str, n: int) -> list[str]:
    """Get n colors for the theme."""
    if theme in _DARK_THEMES or theme in _LIGHT_THEMES:
        palette = _get_palette(theme)
        return [palette[i % len(palette)] for i in range(n)]
    # For ggplot, bmh, seaborn, etc — read from current rcParams color cycle
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    if hasattr(prop_cycle, "by_key"):
        colors = prop_cycle.by_key().get("color", _DARK_PALETTE)
    else:
        colors = _DARK_PALETTE
    return [colors[i % len(colors)] for i in range(n)]


def _make_label(run: dict[str, Any]) -> str:
    """Create a short label from a run summary."""
    task = run.get("task", "???")
    model = run.get("model", "???")
    short_model = model.split("/")[-1] if "/" in model else model
    if len(short_model) > 20:
        short_model = f"{short_model[:18]}…"
    created = run.get("created_at", "")
    date_str = created[:10] if created else ""
    return f"{task}\n{short_model}\n{date_str}"


def _get_metric_score(metrics: list[dict], metric_name: str) -> float:
    """Find the score for a given metric name.

    Returns the score as a value suitable for display on a percentage axis.
    If the raw score is in ratio form (0–1), it is scaled to 0–100.
    If it is already in percentage form (>1), it is returned as-is.
    """
    raw = None
    lower = metric_name.lower()
    for m in metrics:
        if m.get("name") == metric_name:
            raw = m.get("score", 0.0)
            break
    if raw is None:
        for m in metrics:
            if lower in m.get("name", "").lower():
                raw = m.get("score", 0.0)
                break
    if raw is None:
        return 0.0
    if 0.0 <= raw <= 1.0:
        return raw * 100.0
    return raw


def _resolve_accuracy(run: dict[str, Any]) -> float:
    """Return the primary metric score for a run (0–100 scale)."""
    default = run.get("default_metric", "")
    if default:
        for m in run.get("metrics", []):
            if m.get("name") == default:
                raw = m.get("score", 0.0)
                return raw * 100.0 if 0.0 <= raw <= 1.0 else raw
    return 0.0


def _get_score_for_metric(run: dict[str, Any], metric_name: str) -> float:
    """Get the score for a run, handling the special __accuracy__ metric."""
    if metric_name == "__accuracy__":
        return _resolve_accuracy(run)
    return _get_metric_score(run.get("metrics", []), metric_name)


def _resolve_theme_context(theme: str):
    """Return a list of style sources for plt.style.context()."""
    if theme in _DARK_THEMES:
        return [_DARK_THEME]
    if theme in _LIGHT_THEMES:
        return [_LIGHT_THEME]
    # Built-in matplotlib style or custom string — pass through
    return [theme]


def _render_chart(
    fn, runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> bytes:
    """Apply a theme and render a chart to PNG bytes."""
    ctx = _resolve_theme_context(theme)
    with plt.style.context(ctx):
        fig, result = fn(runs, metric_name, width, height, theme)

    buf = io.BytesIO()
    fig.savefig(buf, format = "png", dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _bar_chart_inner(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize = (width, height))

    labels = [_make_label(r) for r in runs]
    scores = [_get_score_for_metric(r, metric_name) for r in runs]
    colors = _resolve_palette(theme, len(runs))

    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color = colors, width = 0.6, edgecolor = "none", zorder = 3)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{score:.1f}%",
            ha = "center",
            va = "bottom",
            fontsize = 9,
            fontweight = "bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize = 8, ha = "center")
    ax.set_ylabel("Score (%)", fontsize = 10)
    title = "Accuracy" if metric_name == "__accuracy__" else metric_name
    ax.set_title(f"Benchmark Comparison — {title}", fontsize = 12, fontweight = "bold", pad = 12)
    ax.set_ylim(0, max(max(scores) * 1.15, 10) if scores else 10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    return fig, ax


def generate_bar_chart(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> bytes:
    """Generate a bar chart comparing runs."""
    return _render_chart(_bar_chart_inner, runs, metric_name, width, height, theme)


def _line_chart_inner(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize = (width, height))

    labels = [_make_label(r) for r in runs]
    scores = [_get_score_for_metric(r, metric_name) for r in runs]
    colors = _resolve_palette(theme, len(runs))

    x = np.arange(len(labels))
    ax.plot(x, scores, marker = "o", color = colors[0], linewidth = 2, markersize = 8, zorder = 3)

    for i, score in enumerate(scores):
        ax.annotate(
            f"{score:.1f}%",
            (x[i], score),
            textcoords = "offset points",
            xytext = (0, 10),
            ha = "center",
            fontsize = 9,
            fontweight = "bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize = 8, ha = "center")
    ax.set_ylabel("Score (%)", fontsize = 10)
    title = "Accuracy" if metric_name == "__accuracy__" else metric_name
    ax.set_title(f"Benchmark Trend — {title}", fontsize = 12, fontweight = "bold", pad = 12)
    ax.set_ylim(
        min(min(scores) * 0.9, 0) if scores else 0, max(max(scores) * 1.15, 10) if scores else 10
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    return fig, ax


def generate_line_chart(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> bytes:
    """Generate a line chart comparing runs."""
    return _render_chart(_line_chart_inner, runs, metric_name, width, height, theme)


def _short_model(model: str) -> str:
    """Short model name for legend labels."""
    short = model.split("/")[-1] if "/" in model else model
    return short if len(short) <= 25 else f"{short[:23]}…"


def _grouped_bar_chart_inner(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> tuple[Any, Any]:
    """Grouped bar: tasks on x-axis, models as grouped bars within each task."""
    fig, ax = plt.subplots(figsize = (width, height))

    # Build task → model → score mapping
    task_model: dict[str, dict[str, float]] = {}
    for r in runs:
        task = r.get("task", "???")
        model = _short_model(r.get("model", "???"))
        score = _get_score_for_metric(r, metric_name)
        task_model.setdefault(task, {})[model] = score

    tasks = list(task_model.keys())
    models = list({m for d in task_model.values() for m in d})
    colors = _resolve_palette(theme, len(models))

    n_tasks = len(tasks)
    n_models = len(models)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(n_tasks)

    for j, model in enumerate(models):
        scores = [task_model[t].get(model, 0.0) for t in tasks]
        offset = (j - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            scores,
            width = bar_width,
            color = colors[j],
            label = model,
            edgecolor = "none",
            zorder = 3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize = 8, ha = "center")
    ax.set_ylabel("Score (%)", fontsize = 10)
    ax.set_title("Benchmark Comparison — All Metrics", fontsize = 12, fontweight = "bold", pad = 12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize = 7, loc = "upper right", framealpha = 0.8)
    fig.tight_layout()
    return fig, ax


def generate_grouped_bar_chart(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> bytes:
    """Generate a grouped bar chart with all metrics per run."""
    return _render_chart(_grouped_bar_chart_inner, runs, metric_name, width, height, theme)


def _radar_chart_inner(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> tuple[Any, Any]:
    """Radar: tasks as axes, each model as its own ring."""
    # Build task → model → score mapping
    task_model: dict[str, dict[str, float]] = {}
    for r in runs:
        task = r.get("task", "???")
        model = _short_model(r.get("model", "???"))
        score = _get_score_for_metric(r, metric_name)
        task_model.setdefault(task, {})[model] = score

    tasks = list(task_model.keys())
    models = list({m for d in task_model.values() for m in d})

    if len(tasks) < 3:
        raise ValueError(
            f"Radar charts need at least 3 different tasks, but only {len(tasks)} found. "
            "Use a bar or grouped_bar chart for fewer tasks."
        )

    fig, ax = plt.subplots(figsize = (width, height), subplot_kw = {"polar": True})

    n = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n, endpoint = False).tolist()
    angles += angles[:1]

    colors = _resolve_palette(theme, len(models))
    for i, model in enumerate(models):
        scores = [task_model[t].get(model, 0.0) for t in tasks]
        scores += scores[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, scores, marker = "o", linewidth = 2, label = model, color = color)
        ax.fill(angles, scores, alpha = 0.1, color = color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize = 7)
    ax.set_title("Benchmark Radar Comparison", fontsize = 12, fontweight = "bold", pad = 20)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize = 7, loc = "upper right", bbox_to_anchor = (1.3, 1.1), framealpha = 0.8)
    fig.tight_layout()
    return fig, ax


def generate_radar_chart(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> bytes:
    """Generate a radar/spider chart comparing runs across all metrics."""
    return _render_chart(_radar_chart_inner, runs, metric_name, width, height, theme)


_CHART_GENERATORS = {
    "bar": generate_bar_chart,
    "line": generate_line_chart,
    "grouped_bar": generate_grouped_bar_chart,
    "radar": generate_radar_chart,
}


def generate_chart(
    runs: list[dict[str, Any]],
    chart_type: str,
    metric: str,
    width: float,
    height: float,
    theme: str,
) -> bytes:
    """Generate a benchmark comparison chart.

    Args:
        runs: List of run summary dicts.
        chart_type: One of 'bar', 'line', 'grouped_bar', 'radar'.
        metric: Metric name to plot, or '__accuracy__' for raw accuracy.
        width: Chart width in inches.
        height: Chart height in inches.
        theme: Theme name.

    Returns:
        PNG image bytes.
    """
    generator = _CHART_GENERATORS.get(chart_type, generate_bar_chart)
    return generator(runs, metric, width, height, theme)
