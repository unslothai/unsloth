# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frame statistics that catch jank in both directions, plus the histogram behind them.

There are two completely different ways a UI is unpleasant and a single summary number always
hides one of them:

    * UNIFORM MEDIOCRITY -- every frame takes 120 ms. The mean is bad but no individual frame is
      remarkable, and `max` says 130 ms, which sounds survivable.
    * THE SINGLE STALL -- 5,000 frames at 8 ms and one at 3.4 s. The mean is excellent, the p95
      is excellent, and the app was visibly frozen for three and a half seconds.

So this module always produces THREE headline frame numbers and the report is not allowed to
quote one alone:

    time_in_jank_pct  fraction of WALL TIME spent inside frames longer than 100 ms. Catches
                      uniform mediocrity, because it goes to 100% when every frame is bad.
    jank_index        sum(max(0, d - budget)^2) / window_ms. Squaring makes one 3.4 s frame
                      dominate a thousand 40 ms ones, which is also how a user remembers it.
    max_frame_ms      the single worst frame, unsummarised.

plus the full histogram in the payload, because every summary above is a lossy view of it and
disagreements between reviewers are settled from the histogram, not from an argument about which
percentile is fairest.

THE BUDGET IS MEASURED, NOT ASSUMED. A 16.7 ms constant silently mis-scores every 120 Hz laptop
(everything looks fine) and every 30 Hz remote desktop (everything looks broken). `budget_ms`
comes from the observed inter-frame distribution of the window itself; `refresh_source` records
where it came from so a reader can see when it fell back.

The 100 ms long-frame threshold is NOT scaled by refresh rate on purpose: it is a claim about
human perception (RAIL's "feels like a break in continuity"), not about the display, and a
120 Hz screen does not make a 100 ms freeze feel shorter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import Measure

#:Perceptual constant. A frame longer than this reads as an interruption regardless of display.
JANK_FRAME_MS = 100.0

#:Detection floors. Below these the recorder cannot distinguish a reading from its own noise.
FLOOR_FRAME_MS = 0.5
FLOOR_TIME_IN_JANK_PCT = 0.1
FLOOR_JANK_INDEX = 0.05

#:A refresh interval outside this range is not a display, it is a broken recorder.
REFRESH_MIN_MS = 3.0
REFRESH_MAX_MS = 60.0
REFRESH_FALLBACK_MS = 1000.0 / 60.0

#: Log-spaced histogram edges, in ms. Chosen so the interesting region (one frame, a few frames, a
#: visible hitch, a freeze) each gets its own bucket rather than all landing in ">33".
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
    """Recover the display's frame interval from the window's own fastest frames.

    The fast tail of the distribution is the display cadence: a frame cannot be delivered faster
    than the compositor presents, so the low quantiles pile up on the refresh interval no matter
    how badly the main thread is behaving. The median of the fastest quartile is used rather than
    the minimum, because the minimum picks up coalesced or duplicated callbacks.
    """

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
    """Turn one window of inter-frame deltas into the three-headline block.

    `attempted=False` is for a window where the frame recorder was deliberately not installed.
    It is NOT for a window where the recorder ran and saw nothing: that case is a reading of
    "no frames", which is a symptom (an unscheduled rAF loop) and never a score of zero jank.
    """

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
        # The recorder was installed and produced nothing: the rAF-unscheduled trap, where a compositor
        # that decided nothing is visible stops SCHEDULING callbacks and the naive reading is "zero dropped
        # frames". Every metric here reads as a failed measurement so the tri-clock gate upstream decides
        # whether the window survives.
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
