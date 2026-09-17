# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Growth exponents, fitted WITHIN ONE SESSION, and the severity ranking.

`log(self_time_f) = a + b * log(L)` across the length ladder. The exponent `b`
is what separates a frame that is merely expensive from a frame that is the
reason long threads get worse: `b ~ 0` is a fixed cost, `b ~ 1` is linear in
thread length, `b ~ 2` is the quadratic re-parse.

CROSS-SESSION FITS ARE VOID AND THIS MODULE REFUSES TO PRODUCE ONE. The same
cell drifts about 8% between sessions on the same machine. Two rungs measured in
different sessions can therefore differ by 8% for no reason at all, and across a
ladder spanning one decade of length that manufactures an exponent of roughly
log(1.08)/log(10) = 0.03 out of pure drift, or far more when the ladder is
short. Every point carries a `session` tag and `fit_loglog` raises if the tags
are not all equal. This is not a warning that can be waved through, because a
fit is exactly the kind of number that looks authoritative once it is in a
table.

Ranking is

    severity = self_ms(L_max) * max(0, b_frame - b_task)

Absolute cost at the top rung, weighted by how much FASTER the frame grows than
total task time. The clamp at zero matters: a frame that grows more slowly than
the total is getting relatively cheaper as threads lengthen, so however large it
is, it is not the reason the curve bends. Multiplying rather than adding means a
frame must be both big and steepening; either alone scores nothing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import CellFailure


@dataclass(frozen = True)
class Point:
    """One rung of one series, tagged with the session that produced it."""

    length: float  # the treatment axis, e.g. thread tokens or characters
    value: float  # the measured quantity, e.g. self ms
    session: str  # opaque session identity; fits refuse to mix these
    rung: str = ""


@dataclass(frozen = True)
class Fit:
    a: float  # intercept in log space
    b: float  # exponent
    r2: float
    n: int
    session: str
    b_ci: tuple[float, float] | None = None
    x_min: float = 0.0
    x_max: float = 0.0

    def predict(self, length: float) -> float:
        return math.exp(self.a + self.b * math.log(length))

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "exponent_b": round(self.b, 4),
            "intercept_a": round(self.a, 4),
            "r2": round(self.r2, 4),
            "points": self.n,
            "session": self.session,
        }
        if self.b_ci is not None:
            row["b_ci95"] = [round(self.b_ci[0], 4), round(self.b_ci[1], 4)]
        return row


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise CellFailure("fit_degenerate", "every point sits at the same length; no slope exists")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return a, b, r2


def fit_loglog(
    points: Sequence[Point],
    *,
    bootstrap: int = 2000,
    seed: int = 20260819,
    min_points: int = 3,
) -> Fit:
    """Fit `log(value) = a + b * log(length)` over one session's ladder.

    Zero and negative values are DROPPED, not floored. A frame that did not run
    at a rung has no logarithm, and substituting an epsilon would invent a data
    point at whatever exponent the epsilon implies. The number of points
    actually used is reported so a fit over two surviving rungs is visible as
    such.
    """
    sessions = {p.session for p in points}
    if len(sessions) > 1:
        raise CellFailure(
            "cross_session_fit",
            f"points span {len(sessions)} sessions ({sorted(sessions)}). The same cell "
            "drifts about 8% between sessions, which alone manufactures a nonzero "
            "exponent, so a cross-session fit is void.",
        )
    usable = [p for p in points if p.length > 0 and p.value > 0]
    if len(usable) < min_points:
        raise CellFailure(
            "fit_underpowered",
            f"{len(usable)} usable points (need {min_points}); "
            f"{len(points) - len(usable)} were dropped for non-positive length or value",
        )
    xs = [math.log(p.length) for p in usable]
    ys = [math.log(p.value) for p in usable]
    a, b, r2 = _ols(xs, ys)

    ci: tuple[float, float] | None = None
    if bootstrap and len(usable) >= 4:
        rng = random.Random(seed)
        slopes: list[float] = []
        idx = range(len(usable))
        for _ in range(bootstrap):
            pick = [rng.choice(idx) for _ in idx]
            bx = [xs[i] for i in pick]
            by = [ys[i] for i in pick]
            try:
                slopes.append(_ols(bx, by)[1])
            except CellFailure:
                continue
        if len(slopes) >= 100:
            slopes.sort()
            lo = slopes[int(0.025 * len(slopes))]
            hi = slopes[min(len(slopes) - 1, int(0.975 * len(slopes)))]
            ci = (lo, hi)

    return Fit(
        a = a,
        b = b,
        r2 = r2,
        n = len(usable),
        session = next(iter(sessions)) if sessions else "",
        b_ci = ci,
        x_min = min(p.length for p in usable),
        x_max = max(p.length for p in usable),
    )


@dataclass
class FrameGrowth:
    """One call frame's cost and growth across a ladder, plus its severity."""

    frame_label: str
    frame_key: tuple[str, str, int, int]
    points: list[Point] = field(default_factory = list)
    fit: Fit | None = None
    self_ms_at_max: float = 0.0
    severity: float = 0.0
    bridged_name: str | None = None
    exact_call_count: int | None = None

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "frame": self.bridged_name or self.frame_label,
            "raw_frame": self.frame_label,
            "self_ms_at_L_max": round(self.self_ms_at_max, 3),
            "severity": round(self.severity, 4),
            "bridged": self.bridged_name is not None,
        }
        if self.exact_call_count is not None:
            row["exact_call_count"] = self.exact_call_count
        if self.fit is not None:
            row.update(self.fit.as_row())
        return row


def severity(self_ms_at_max: float, b_frame: float, b_task: float) -> float:
    """Absolute cost weighted by how much faster the frame grows than the total.

    Clamped at zero on the exponent difference: a frame growing more slowly than
    total task time is becoming a smaller share of the problem as threads
    lengthen, so it cannot be the reason the curve bends, no matter how many
    milliseconds it costs today.
    """
    return self_ms_at_max * max(0.0, b_frame - b_task)


def rank_frames(
    series: dict[tuple[str, str, int, int], list[Point]],
    labels: dict[tuple[str, str, int, int], str],
    task_total_points: Sequence[Point],
    *,
    bootstrap: int = 2000,
    min_points: int = 3,
) -> tuple[list[FrameGrowth], dict[str, Any]]:
    """Fit every frame, fit the task total, rank by severity.

    Returns the ranking and a diagnostics block naming every frame that could
    not be fitted and why. A frame dropped for having too few rungs is a fact
    about coverage of the ladder, not a fact about the frame, and silently
    omitting it would make the ranking look more complete than it is.
    """
    task_fit = fit_loglog(task_total_points, bootstrap = bootstrap, min_points = min_points)
    rows: list[FrameGrowth] = []
    skipped: dict[str, str] = {}
    for key, pts in series.items():
        label = labels.get(key, str(key))
        try:
            f = fit_loglog(pts, bootstrap = bootstrap, min_points = min_points)
        except CellFailure as exc:
            skipped[label] = exc.detail
            continue
        at_max = max(pts, key = lambda p: p.length)
        g = FrameGrowth(
            frame_label = label,
            frame_key = key,
            points = list(pts),
            fit = f,
            self_ms_at_max = at_max.value,
        )
        g.severity = severity(g.self_ms_at_max, f.b, task_fit.b)
        rows.append(g)
    rows.sort(key = lambda g: -g.severity)
    diagnostics = {
        "task_total_fit": task_fit.as_row(),
        "frames_fitted": len(rows),
        "frames_skipped": skipped,
        "session": task_fit.session,
    }
    return rows, diagnostics


def growth_is_superlinear(fit: Fit, *, margin: float = 0.15) -> bool:
    """Is the exponent above 1 by more than the fit's own uncertainty?

    Uses the bootstrap lower bound when there is one, because "b = 1.4" from
    three noisy points is not evidence of superlinearity and reading it as such
    is how an O(n) mechanism gets reported as O(n^2).
    """
    if fit.b_ci is not None:
        return fit.b_ci[0] > 1.0
    return fit.b > 1.0 + margin


def collect_series(
    per_rung: Iterable[tuple[str, float, dict[tuple[str, str, int, int], float]]], session: str
) -> dict[tuple[str, str, int, int], list[Point]]:
    """Reshape per-rung frame tables into per-frame ladders.

    Input is (rung label, rung length, {frame key: self ms}). A frame absent at
    a rung contributes no point rather than a zero, since a zero has no
    logarithm and inventing one at the bottom of the ladder tilts every
    exponent upward.
    """
    out: dict[tuple[str, str, int, int], list[Point]] = {}
    for rung, length, table in per_rung:
        for key, value in table.items():
            out.setdefault(key, []).append(
                Point(length = length, value = value, session = session, rung = rung)
            )
    return out
