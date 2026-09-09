# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The declared anchor table: the only place a judgement about "good" is written down.

Every scalar this benchmark produces is a chain of arbitrary choices, and the way that chain
becomes dishonest is by being spread across six files so nobody can see it. So all of it is here,
in one table, and the table is HASHED into `weights_id`. A report that compares two runs with
different `weights_id` values is refused rather than rendered, because a scoring change and a
performance change are indistinguishable in the output.

WHY LOG ANCHORS. Latency is perceived multiplicatively: 20 ms -> 40 ms is the same felt step as
200 ms -> 400 ms, and a linear map makes the entire interesting range (20-200 ms) occupy 4% of
the scale while 2 s -> 4 s occupies half of it. That is the second way naive AUC lies: the curve
goes flat exactly where users start to hurt. Each metric therefore declares two anchors, `good`
(scores 100) and `bad` (scores 0), and interpolates in log space between them.

The anchors are not measurements, they are targets, and they are argued for here rather than
tuned until the numbers look nice:

    keystroke_p95_ms   good 20 / bad 500. 20 ms is one frame at 50 Hz, the point at which typing
                       stops feeling like typing at all. 500 ms is the point at which characters
                       arrive after you have stopped looking at the keyboard. This metric carries
                       the largest weight because the shipped complaint is about typing.
    time_in_jank_pct   good 0.5 / bad 60. Half a percent of wall time inside 100 ms+ frames is
                       one hitch a minute. 60% is a UI that is unresponsive more often than not.
    jank_index         good 0.1 / bad 50. Squared over-budget ms per ms of window.
    max_frame_ms       good 33 / bad 2000. Two frames at 60 Hz, versus a freeze long enough that
                       a user reaches for the window close button.
    scroll_settle_ms   good 100 / bad 3000.
    menu_open_ms       good 50 / bad 1500. A menu that takes a second and a half to open reads
                       as a click that did not register, and gets clicked again.

WHY THESE SIX AND NOT THE FRAME NUMBERS ALONE. `time_in_jank_pct` catches uniform mediocrity and
`jank_index` plus `max_frame_ms` catch the single stall; neither is allowed to be a headline on
its own, and both are in the mean so a build cannot trade one for the other.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping


@dataclass(frozen = True)
class MetricAnchor:
    """One perceptual metric, its two anchors, its weight and its direction."""

    key: str
    unit: str
    good: float
    bad: float
    weight: float
    lower_is_better: bool = True
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.good <= 0 or self.bad <= 0:
            raise ValueError(
                f"{self.key}: log anchors require positive values; got good={self.good}, "
                f"bad={self.bad}"
            )
        if self.lower_is_better and not self.bad > self.good:
            raise ValueError(f"{self.key}: lower_is_better needs bad > good")
        if not self.lower_is_better and not self.good > self.bad:
            raise ValueError(f"{self.key}: higher_is_better needs good > bad")
        if self.weight <= 0:
            raise ValueError(f"{self.key}: weight must be positive")


METRIC_ANCHORS: tuple[MetricAnchor, ...] = (
    MetricAnchor(
        key = "keystroke_p95_ms",
        unit = "ms",
        good = 20.0,
        bad = 500.0,
        weight = 0.25,
        rationale = "the shipped complaint is typing latency, so it carries the most weight",
    ),
    MetricAnchor(
        key = "time_in_jank_pct",
        unit = "%",
        good = 0.5,
        bad = 60.0,
        weight = 0.20,
        rationale = "catches uniform mediocrity, which max and p95 both hide",
    ),
    MetricAnchor(
        key = "jank_index",
        unit = "ms",
        good = 0.1,
        bad = 50.0,
        weight = 0.15,
        rationale = "squared over-budget time, so one long freeze dominates many small ones",
    ),
    MetricAnchor(
        key = "max_frame_ms",
        unit = "ms",
        good = 33.0,
        bad = 2000.0,
        weight = 0.15,
        rationale = "the single stall, unsummarised",
    ),
    MetricAnchor(
        key = "scroll_settle_ms",
        unit = "ms",
        good = 100.0,
        bad = 3000.0,
        weight = 0.15,
        rationale = "scrolling a long thread is the second half of the shipped complaint",
    ),
    MetricAnchor(
        key = "menu_open_ms",
        unit = "ms",
        good = 50.0,
        bad = 1500.0,
        weight = 0.10,
        rationale = "a slow menu reads as a click that did not register",
    ),
)

METRIC_BY_KEY: Mapping[str, MetricAnchor] = {m.key: m for m in METRIC_ANCHORS}

#: The rung ladder, in tokens of thread content. Log-spaced on purpose: aggregation integrates over
#: log(tokens), so evenly spaced rungs on the log axis give evenly spaced evidence.
RUNG_TOKENS: tuple[int, ...] = (1_000, 10_000, 100_000, 500_000, 1_000_000)

#: A rung counts as usable when its score clears this AND no single metric is catastrophic. The
#: onset rung, the largest usable rung, is the human headline, because it survives being carried to
#: a different machine in a way a 0-100 score does not.
ONSET_SCORE_THRESHOLD = 50.0
ONSET_METRIC_FLOOR = 10.0

#: A rung whose measured metrics do not cover at least this fraction of the declared weight is
#: INCOMPLETE, which scores 0. It does not drop out: a build that crashes at 500K must not outscore
#: one that limps through it.
MIN_WEIGHT_COVERAGE = 0.60

#: Ratios inside this band are indistinguishable from run-to-run noise. Anything an A/B claims must
#: clear it, and the NULL calibration arm must land inside it.
DEFAULT_NOISE_FLOOR_PCT = 5.0


def _canonical_table() -> dict:
    return {
        "metrics": [asdict(m) for m in METRIC_ANCHORS],
        "onset_score_threshold": ONSET_SCORE_THRESHOLD,
        "onset_metric_floor": ONSET_METRIC_FLOOR,
        "min_weight_coverage": MIN_WEIGHT_COVERAGE,
        "default_noise_floor_pct": DEFAULT_NOISE_FLOOR_PCT,
    }


def weights_id() -> str:
    """Stable hash of every judgement in this file. Changing any of it changes the id."""

    blob = json.dumps(_canonical_table(), sort_keys = True, separators = (",", ":"))
    return "w-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def rung_ladder_id(rungs: Iterable[int] = RUNG_TOKENS) -> str:
    blob = json.dumps(sorted(int(r) for r in rungs), separators = (",", ":"))
    return "r-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
