# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generate benchmark comparison charts using matplotlib."""

import io
import re
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


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


# Mirrors isRatioScore in frontend features/benchmark/lib/compare-metrics.ts so the
# PNG agrees with the results table: perplexity / bits_per_byte keep their own units.
_ACCURACY_LIKE_RE = re.compile(r"(em|exact|strict|flexible|acc|f1|match|score)", re.IGNORECASE)


def _is_ratio_score(name: str, score: float) -> bool:
    return bool(_ACCURACY_LIKE_RE.search(name)) and 0.0 <= score <= 1.0


def _is_number(raw: Any) -> bool:
    return isinstance(raw, (int, float)) and not isinstance(raw, bool)


def _as_percent(raw: Any) -> Optional[float]:
    if not _is_number(raw):
        return None
    return raw * 100.0 if 0.0 <= raw <= 1.0 else float(raw)


# A chart point: (value, is_percent); value None when the run lacks the metric.
_Point = tuple[Optional[float], bool]


def _metric_point(name: str, raw: Any) -> _Point:
    if not _is_number(raw):
        return None, False
    if _is_ratio_score(name, raw):
        return raw * 100.0, True
    return float(raw), False


def _get_metric_point(metrics: list[dict], metric_name: str) -> _Point:
    """Find the score for a metric name (exact, else substring match).

    Ratio metrics (accuracy-like names scored 0-1) are scaled to percent; anything
    else keeps its own unit. None when the run does not report the metric:
    charting it as 0 would read as measured.
    """
    name, raw = metric_name, None
    lower = metric_name.lower()
    for m in metrics:
        if m.get("name") == metric_name:
            raw = m.get("score")
            break
    if raw is None:
        for m in metrics:
            if lower in m.get("name", "").lower():
                name, raw = m.get("name", ""), m.get("score")
                break
    return _metric_point(name, raw)


def _resolve_accuracy(run: dict[str, Any]) -> Optional[float]:
    """Return the primary metric score for a run (0–100 scale), None if it has none."""
    default = run.get("default_metric", "")
    if default:
        for m in run.get("metrics", []):
            if m.get("name") == default:
                return _as_percent(m.get("score"))
    return None


def _measured(scores: list[Optional[float]]) -> list[float]:
    return [s for s in scores if s is not None]


def _plotted(scores: list[Optional[float]]) -> list[float]:
    # NaN leaves a gap where a run lacks the metric instead of drawing a 0% bar or point.
    return [np.nan if s is None else s for s in scores]


def _get_point_for_metric(run: dict[str, Any], metric_name: str) -> _Point:
    """Get (score, is_percent) for a run, handling the special __accuracy__ metric."""
    if metric_name == "__accuracy__":
        return _resolve_accuracy(run), True
    return _get_metric_point(run.get("metrics", []), metric_name)


def _get_score_for_metric(run: dict[str, Any], metric_name: str) -> Optional[float]:
    return _get_point_for_metric(run, metric_name)[0]


def _format_point(score: Optional[float], is_percent: bool) -> str:
    """Point label, matching formatMetricWithUnit in the frontend."""
    if score is None:
        return "n/a"
    if is_percent:
        return f"{score:.1f}%"
    if float(score).is_integer():
        return str(int(score))
    return f"{score:.4f}" if score < 1 else f"{score:.1f}"


def _percent_axis(points: list[_Point]) -> bool:
    """A % axis only when every measured value is a percent."""
    measured = [p for p in points if p[0] is not None]
    return not measured or all(is_pct for _, is_pct in measured)


def _apply_score_axis(
    ax,
    percent: bool,
    label: bool = True,
) -> None:
    if label:
        ax.set_ylabel("Score (%)" if percent else "Score", fontsize = 10)
    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())


def _upper_limit(measured: list[float], percent: bool) -> float:
    if not measured:
        return 10 if percent else 1
    top = max(measured) * 1.15
    # Percent keeps a 10% floor; raw units (perplexity, bits_per_byte) scale to the data.
    return max(top, 10) if percent else (top if top > 0 else 1)


def _resolve_theme_context(theme: str):
    """Return a list of style sources for plt.style.context()."""
    if theme in _DARK_THEMES:
        return [_DARK_THEME]
    if theme in _LIGHT_THEMES:
        return [_LIGHT_THEME]
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
    points = [_get_point_for_metric(r, metric_name) for r in runs]
    scores = [p[0] for p in points]
    percent = _percent_axis(points)
    colors = _resolve_palette(theme, len(runs))

    x = np.arange(len(labels))
    bars = ax.bar(x, _plotted(scores), color = colors, width = 0.6, edgecolor = "none", zorder = 3)

    for bar, (score, is_pct) in zip(bars, points):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            (score or 0.0) + (0.5 if percent else 0.0),
            _format_point(score, is_pct),
            ha = "center",
            va = "bottom",
            fontsize = 9,
            fontweight = "bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize = 8, ha = "center")
    _apply_score_axis(ax, percent)
    title = "Accuracy" if metric_name == "__accuracy__" else metric_name
    ax.set_title(f"Benchmark Comparison — {title}", fontsize = 12, fontweight = "bold", pad = 12)
    ax.set_ylim(0, _upper_limit(_measured(scores), percent))
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
    points = [_get_point_for_metric(r, metric_name) for r in runs]
    scores = [p[0] for p in points]
    percent = _percent_axis(points)
    colors = _resolve_palette(theme, len(runs))

    x = np.arange(len(labels))
    ax.plot(x, _plotted(scores), marker = "o", color = colors[0], linewidth = 2, markersize = 8, zorder = 3)

    for i, (score, is_pct) in enumerate(points):
        ax.annotate(
            _format_point(score, is_pct),
            (x[i], score or 0.0),
            textcoords = "offset points",
            xytext = (0, 10),
            ha = "center",
            fontsize = 9,
            fontweight = "bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize = 8, ha = "center")
    _apply_score_axis(ax, percent)
    title = "Accuracy" if metric_name == "__accuracy__" else metric_name
    ax.set_title(f"Benchmark Trend — {title}", fontsize = 12, fontweight = "bold", pad = 12)
    measured = _measured(scores)
    ax.set_ylim(
        min(min(measured) * 0.9, 0) if measured else 0,
        _upper_limit(measured, percent),
    )
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


def _legend_labels(models: list[str]) -> dict[str, str]:
    """Short legend text per full model string, kept distinct: two quants of one
    model share a truncated prefix, so colliding names show their tail instead."""
    labels = {m: _short_model(m) for m in models}
    taken = [labels[m] for m in models]
    for m in models:
        if taken.count(labels[m]) > 1:
            base = m.split("/")[-1]
            labels[m] = base if len(base) <= 25 else f"…{base[-23:]}"
    seen: dict[str, int] = {}
    for m in models:
        seen[labels[m]] = seen.get(labels[m], 0) + 1
        if seen[labels[m]] > 1:
            labels[m] = f"{labels[m]} ({seen[labels[m]]})"
    return labels


def _grouped_bar_chart_inner(
    runs: list[dict[str, Any]], metric_name: str, width: float, height: float, theme: str
) -> tuple[Any, Any]:
    """Grouped bar: tasks on x-axis, models as grouped bars within each task."""
    fig, ax = plt.subplots(figsize = (width, height))

    task_model: dict[str, dict[str, Optional[float]]] = {}
    points: list[_Point] = []
    for r in runs:
        task = r.get("task", "???")
        model = str(r.get("model", "???"))
        point = _get_point_for_metric(r, metric_name)
        points.append(point)
        task_model.setdefault(task, {})[model] = point[0]

    tasks = list(task_model.keys())
    models = list(dict.fromkeys(m for d in task_model.values() for m in d))
    legend = _legend_labels(models)
    colors = _resolve_palette(theme, len(models))

    n_tasks = len(tasks)
    n_models = len(models)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(n_tasks)

    for j, model in enumerate(models):
        scores = [task_model[t].get(model) for t in tasks]
        offset = (j - n_models / 2 + 0.5) * bar_width
        for i, score in enumerate(scores):
            if score is None:
                ax.text(x[i] + offset, 0.5, "n/a", ha = "center", va = "bottom", fontsize = 7)
        ax.bar(
            x + offset,
            _plotted(scores),
            width = bar_width,
            color = colors[j],
            label = legend[model],
            edgecolor = "none",
            zorder = 3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize = 8, ha = "center")
    _apply_score_axis(ax, _percent_axis(points))
    ax.set_title("Benchmark Comparison — All Metrics", fontsize = 12, fontweight = "bold", pad = 12)
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
    task_model: dict[str, dict[str, Optional[float]]] = {}
    points: list[_Point] = []
    for r in runs:
        task = r.get("task", "???")
        model = str(r.get("model", "???"))
        point = _get_point_for_metric(r, metric_name)
        points.append(point)
        task_model.setdefault(task, {})[model] = point[0]

    tasks = list(task_model.keys())
    models = list(dict.fromkeys(m for d in task_model.values() for m in d))
    legend = _legend_labels(models)

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
        scores = [task_model[t].get(model) for t in tasks]
        scores += scores[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, _plotted(scores), marker = "o", linewidth = 2, label = legend[model], color = color)
        if None not in scores:
            ax.fill(angles, scores, alpha = 0.1, color = color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize = 7)
    ax.set_title("Benchmark Radar Comparison", fontsize = 12, fontweight = "bold", pad = 20)
    _apply_score_axis(ax, _percent_axis(points), label = False)
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
