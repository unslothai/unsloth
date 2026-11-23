# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Three headline frame numbers (time_in_jank_pct, jank_index, max_frame_ms), never one alone."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import Measure

JANK_FRAME_MS = 100.0

FLOOR_FRAME_MS = 0.5
FLOOR_TIME_IN_JANK_PCT = 0.1
FLOOR_JANK_INDEX = 0.05

REFRESH_MIN_MS = 3.0
REFRESH_MAX_MS = 60.0
REFRESH_FALLBACK_MS = 1000.0 / 60.0

HISTOGRAM_EDGES_MS: tuple[float, ...] = (
    0.0,
    4.0,
    8.0,
    16.7,
    25.0,
    33.3,
    50.0,
    75.0,
    100.0,
    150.0,
    250.0,
    400.0,
    650.0,
    1000.0,
    1600.0,
    2500.0,
    4000.0,
)


@dataclass
class FrameStats:
    """Everything derivable from one window of inter-frame deltas."""

    frames_total: int
    window_ms: float
    budget_ms: float
    refresh_source: str
    time_in_jank_pct: Measure
    jank_index: Measure
    max_frame_ms: Measure
    p50_frame_ms: Measure
    p95_frame_ms: Measure
    p99_frame_ms: Measure
    dropped_frames: Measure
    effective_fps: Measure
    histogram: list[dict[str, Any]] = field(default_factory = list)
    no_frames_recorded: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "frames_total": int(self.frames_total),
            "window_ms": float(self.window_ms),
            "budget_ms": float(self.budget_ms),
            "refresh_source": self.refresh_source,
            "no_frames_recorded": bool(self.no_frames_recorded),
            "time_in_jank_pct": self.time_in_jank_pct.to_json(),
            "jank_index": self.jank_index.to_json(),
            "max_frame_ms": self.max_frame_ms.to_json(),
            "p50_frame_ms": self.p50_frame_ms.to_json(),
            "p95_frame_ms": self.p95_frame_ms.to_json(),
            "p99_frame_ms": self.p99_frame_ms.to_json(),
            "dropped_frames": self.dropped_frames.to_json(),
            "effective_fps": self.effective_fps.to_json(),
            "histogram": list(self.histogram),
        }


def measure_refresh_interval_ms(deltas: Sequence[float]) -> tuple[float, str]:
    """Median of the fastest quartile, not the minimum, which picks up coalesced or duplicated callbacks."""

    usable = [float(d) for d in deltas if d is not None and math.isfinite(d) and d > 0]
    if len(usable) < 8:
        return REFRESH_FALLBACK_MS, "fallback_too_few_frames"
    usable.sort()
    fast_quartile = usable[: max(2, len(usable) // 4)]
    candidate = _percentile(fast_quartile, 50.0)
    if not (REFRESH_MIN_MS <= candidate <= REFRESH_MAX_MS):
        return REFRESH_FALLBACK_MS, f"fallback_out_of_range({candidate:.2f}ms)"
    return candidate, "measured"


def compute_frame_stats(
    deltas: Sequence[float],
    window_ms: float,
    *,
    declared_refresh_ms: float | None = None,
    attempted: bool = True,
    not_attempted_reason: str | None = None,
) -> FrameStats:
    """attempted=False means no recorder was installed; a recorder that saw no frames is not zero jank."""

    if not attempted:
        reason = not_attempted_reason or "frame recorder not installed"
        return FrameStats(
            frames_total = 0,
            window_ms = float(window_ms),
            budget_ms = REFRESH_FALLBACK_MS,
            refresh_source = "not_attempted",
            time_in_jank_pct = Measure.not_attempted("%", reason),
            jank_index = Measure.not_attempted("ms", reason),
            max_frame_ms = Measure.not_attempted("ms", reason),
            p50_frame_ms = Measure.not_attempted("ms", reason),
            p95_frame_ms = Measure.not_attempted("ms", reason),
            p99_frame_ms = Measure.not_attempted("ms", reason),
            dropped_frames = Measure.not_attempted("frames", reason),
            effective_fps = Measure.not_attempted("fps", reason),
            histogram = [],
            no_frames_recorded = False,
        )

    usable = [float(d) for d in deltas if d is not None and math.isfinite(d) and d >= 0]
    window_ms = float(window_ms)

    if declared_refresh_ms is not None and REFRESH_MIN_MS <= declared_refresh_ms <= REFRESH_MAX_MS:
        budget_ms, refresh_source = float(declared_refresh_ms), "declared"
    else:
        budget_ms, refresh_source = measure_refresh_interval_ms(usable)

    if not usable:
        # The rAF-unscheduled trap: a hidden compositor stops scheduling callbacks, which is not zero jank.
        reason = "frame recorder produced no frames (rAF may be unscheduled)"
        return FrameStats(
            frames_total = 0,
            window_ms = window_ms,
            budget_ms = budget_ms,
            refresh_source = refresh_source,
            time_in_jank_pct = Measure.failed("%", reason),
            jank_index = Measure.failed("ms", reason),
            max_frame_ms = Measure.failed("ms", reason),
            p50_frame_ms = Measure.failed("ms", reason),
            p95_frame_ms = Measure.failed("ms", reason),
            p99_frame_ms = Measure.failed("ms", reason),
            dropped_frames = Measure.failed("frames", reason),
            effective_fps = Measure.failed("fps", reason),
            histogram = [],
            no_frames_recorded = True,
        )

    ordered = sorted(usable)
    jank_time_ms = sum(d for d in usable if d > JANK_FRAME_MS)
    denominator = window_ms if window_ms > 0 else sum(usable)
    over_budget_sq = sum((d - budget_ms) ** 2 for d in usable if d > budget_ms)
    dropped = sum(1 for d in usable if d > budget_ms * 1.5)

    return FrameStats(
        frames_total = len(usable),
        window_ms = window_ms,
        budget_ms = budget_ms,
        refresh_source = refresh_source,
        time_in_jank_pct = Measure.read(
            100.0 * jank_time_ms / denominator, "%", floor = FLOOR_TIME_IN_JANK_PCT
        ),
        jank_index = Measure.read(over_budget_sq / denominator, "ms", floor = FLOOR_JANK_INDEX),
        max_frame_ms = Measure.read(max(usable), "ms", floor = FLOOR_FRAME_MS),
        p50_frame_ms = Measure.read(_percentile(ordered, 50.0), "ms", floor = FLOOR_FRAME_MS),
        p95_frame_ms = Measure.read(_percentile(ordered, 95.0), "ms", floor = FLOOR_FRAME_MS),
        p99_frame_ms = Measure.read(_percentile(ordered, 99.0), "ms", floor = FLOOR_FRAME_MS),
        dropped_frames = Measure.read(float(dropped), "frames", floor = None),
        effective_fps = Measure.read(1000.0 * len(usable) / denominator, "fps", floor = None),
        histogram = build_histogram(usable),
        no_frames_recorded = False,
    )


def build_histogram(deltas: Sequence[float]) -> list[dict[str, Any]]:
    """Log-spaced frame histogram. Always present in the payload, never summarised away."""

    edges = list(HISTOGRAM_EDGES_MS)
    buckets = [
        {"lo_ms": edges[i], "hi_ms": edges[i + 1], "bucket_count": 0} for i in range(len(edges) - 1)
    ]
    buckets.append({"lo_ms": edges[-1], "hi_ms": None, "bucket_count": 0})
    for delta in deltas:
        placed = False
        for bucket in buckets[:-1]:
            if bucket["lo_ms"] <= delta < bucket["hi_ms"]:
                bucket["bucket_count"] += 1
                placed = True
                break
        if not placed:
            buckets[-1]["bucket_count"] += 1
    return buckets


def _percentile(ordered: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile over an already sorted sequence."""

    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    weight = rank - low
    return float(ordered[low]) * (1.0 - weight) + float(ordered[high]) * weight
