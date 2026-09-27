# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-metric scores, per-rung scores, the aggregate, and the onset rung.

THREE WAYS A NAIVE AREA-UNDER-CURVE SCORE LIES, AND WHAT IS DONE ABOUT EACH.

1. INTEGRATING OVER A LINEAR RUNG AXIS MAKES THE TOP RUNG THE WHOLE SCORE.
   Rungs are 1K, 10K, 100K, 500K, 1M tokens. On a linear axis the 1M rung is half the width of
   the entire integral and the 1K rung is 0.1% of it, so a build that is unusable at every size a
   human actually uses scores well by being no worse than the competition at 1M. Integration
   happens over log(tokens) instead, where the rungs are near-evenly spaced, which is also the
   axis on which the underlying cost is believed to be a power law.

2. INTEGRATING A RAW METRIC IS FLAT EXACTLY WHERE USERS HURT.
   Milliseconds are unbounded above, so the curve is dominated by whichever rung is worst, and
   the region between 20 ms and 200 ms -- where typing goes from fine to unpleasant -- is a
   rounding error next to a 4 s stall at the top rung. Every metric is mapped through its log
   anchors to a bounded [0, 100] perceptual score BEFORE any aggregation.

3. AN INCOMPLETE RUNG THAT DROPS OUT MAKES CRASHING BETTER THAN LIMPING.
   If a rung that fails to complete is skipped, a build that kills the renderer at 500K is scored
   over 1K/10K/100K only -- its three best rungs -- and beats a build that finishes 500K slowly.
   An incomplete rung scores 0 and keeps its weight. It never drops out. This is why
   `RungScore.complete` and `MIN_WEIGHT_COVERAGE` exist.

WHY THE PER-RUNG MEAN IS GEOMETRIC. A UI that types fine but cannot be scrolled is broken, and an
arithmetic mean lets four good metrics rescue one catastrophic one: 100, 100, 100, 100, 0 averages
to 80, which reads as a good build. The weighted geometric mean of that set is 0. A zero anywhere
zeroes the rung, by construction and on purpose, because that is what the user experiences.

WHY THE HEADLINE FOR HUMANS IS THE ONSET RUNG. A 0-100 score does not travel: run it on a
different laptop and every number moves, so two testers cannot compare notes. "It is still usable
at 100K and not at 500K" does travel, and it is the same shape as "highest playable settings",
which people already know how to reason about. Ceiling shifts (the onset rung moving) are reported
SEPARATELY and are never folded into the scalar, because a build that moves the ceiling by one
rung and a build that shaves 8% off every metric are different kinds of win and averaging them
produces a number that describes neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .anchors import (
    METRIC_ANCHORS,
    METRIC_BY_KEY,
    MIN_WEIGHT_COVERAGE,
    ONSET_METRIC_FLOOR,
    ONSET_SCORE_THRESHOLD,
    RUNG_TOKENS,
    MetricAnchor,
    rung_ladder_id,
    weights_id,
)
from .schema import Measure


@dataclass
class MetricScore:
    """One metric's bounded [0, 100] score, or an explicit refusal to score it."""

    key: str
    score: float | None
    scored: bool
    weight: float
    measure: Measure
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": None if self.score is None else round(float(self.score), 3),
            "scored": bool(self.scored),
            "weight": float(self.weight),
            "measure": self.measure.to_json(),
            "reason": self.reason,
        }


def score_metric(anchor: MetricAnchor, measure: Measure) -> MetricScore:
    """Map one reading onto [0, 100] by log interpolation between the declared anchors.

    A measurement with no reading is NOT scored. It does not become a 0 (which would claim the
    build is catastrophic at something we did not measure) and it does not become a 100 (which
    would reward a build for an instrument that failed). It is excluded from the mean and
    subtracted from the rung's weight coverage, which is the thing that decides whether the rung
    is complete enough to score at all.
    """

    if not measure.has_reading:
        reason = measure.note or ("not attempted" if not measure.attempted else "no reading")
        return MetricScore(
            key = anchor.key,
            score = None,
            scored = False,
            weight = anchor.weight,
            measure = measure,
            reason = reason,
        )

    value = float(measure.value)
    # A sub-floor reading is at least as good as the floor; scoring the raw value would let instrument
    # noise on a fast machine invent a difference between two perfect builds.
    if measure.sub_floor and measure.floor is not None and anchor.lower_is_better:
        value = min(abs(value), float(measure.floor))

    good, bad = float(anchor.good), float(anchor.bad)
    if value <= 0:
        # Log space has no zero. A non-positive reading on a positive-only metric is either perfect (below
        # every floor) or a broken instrument, and the floor logic above has handled the honest case, so
        # clamp to the good anchor and say so.
        value = good if anchor.lower_is_better else bad

    span = math.log(bad) - math.log(good)
    fraction = (math.log(bad) - math.log(value)) / span
    score = 100.0 * max(0.0, min(1.0, fraction))
    return MetricScore(
        key = anchor.key,
        score = score,
        scored = True,
        weight = anchor.weight,
        measure = measure,
    )


@dataclass
class RungScore:
    """One rung of the ladder: its metrics, its geometric mean, and whether it is usable."""

    tokens: int
    score: float
    complete: bool
    usable: bool
    weight_coverage: float
    metric_scores: list[MetricScore] = field(default_factory = list)
    incomplete_reason: str | None = None
    zeroed_by: list[str] = field(default_factory = list)

    def to_json(self) -> dict[str, Any]:
        return {
            "tokens": int(self.tokens),
            "score": round(float(self.score), 3),
            "complete": bool(self.complete),
            "usable": bool(self.usable),
            "weight_coverage": round(float(self.weight_coverage), 4),
            "incomplete_reason": self.incomplete_reason,
            "zeroed_by": list(self.zeroed_by),
            "metric_scores": [m.to_json() for m in self.metric_scores],
        }


def score_rung(
    tokens: int,
    metrics: Mapping[str, Measure],
    *,
    completed: bool = True,
    failure_mode: str | None = None,
) -> RungScore:
    """Score one rung. An incomplete rung scores 0 and keeps its weight in the aggregate.

    `completed=False` covers every way a rung can fail to produce a session: a renderer crash, a
    `goto` timeout, an out-of-memory kill. Each of those is a first-class RESULT about the build,
    not a missing data point, and the score that describes it is 0.
    """

    metric_scores = [
        score_metric(anchor, metrics.get(anchor.key) or _absent(anchor))
        for anchor in METRIC_ANCHORS
    ]

    if not completed:
        return RungScore(
            tokens = int(tokens),
            score = 0.0,
            complete = False,
            usable = False,
            weight_coverage = 0.0,
            metric_scores = metric_scores,
            incomplete_reason = failure_mode or "rung did not complete",
        )

    scored = [m for m in metric_scores if m.scored]
    total_weight = sum(a.weight for a in METRIC_ANCHORS)
    coverage = (sum(m.weight for m in scored) / total_weight) if total_weight else 0.0

    if coverage < MIN_WEIGHT_COVERAGE:
        return RungScore(
            tokens = int(tokens),
            score = 0.0,
            complete = False,
            usable = False,
            weight_coverage = coverage,
            metric_scores = metric_scores,
            incomplete_reason = (
                f"only {coverage:.0%} of the declared metric weight produced a reading, "
                f"below the {MIN_WEIGHT_COVERAGE:.0%} floor"
            ),
        )

    zeroed_by = [m.key for m in scored if float(m.score) <= 0.0]
    if zeroed_by:
        # Geometric mean of a set containing zero IS zero. Written out rather than left to log(0) so the
        # reason travels with the number.
        rung_score = 0.0
    else:
        numerator = sum(m.weight * math.log(float(m.score)) for m in scored)
        denominator = sum(m.weight for m in scored)
        rung_score = math.exp(numerator / denominator)

    worst = min((float(m.score) for m in scored), default = 0.0)
    usable = rung_score >= ONSET_SCORE_THRESHOLD and worst >= ONSET_METRIC_FLOOR

    return RungScore(
        tokens = int(tokens),
        score = rung_score,
        complete = True,
        usable = usable,
        weight_coverage = coverage,
        metric_scores = metric_scores,
        zeroed_by = zeroed_by,
    )


def _absent(anchor: MetricAnchor) -> Measure:
    return Measure.not_attempted(anchor.unit, f"{anchor.key} absent from the payload")


def log_rung_weights(rungs: Sequence[int]) -> list[float]:
    """Trapezoid weights on the log(tokens) axis, normalised to sum to 1.

    This is the AUC over log(tokens) written as a weighted mean. A single rung gets weight 1.
    Interior rungs get half the span on each side; the two ends get their one half-span, which is
    why the top rung does not silently become the whole score.
    """

    ordered = sorted(int(r) for r in rungs)
    if not ordered:
        return []
    if len(ordered) == 1:
        return [1.0]
    logs = [math.log(r) for r in ordered]
    widths: list[float] = []
    for i, _ in enumerate(logs):
        lo = logs[i - 1] if i > 0 else logs[i]
        hi = logs[i + 1] if i + 1 < len(logs) else logs[i]
        widths.append((hi - lo) / 2.0)
    total = sum(widths)
    if total <= 0:
        return [1.0 / len(ordered)] * len(ordered)
    return [w / total for w in widths]


@dataclass
class LadderScore:
    """The aggregate over the whole rung ladder, plus the headline that travels."""

    aggregate: float
    onset_rung_tokens: int | None
    onset_reason: str
    non_monotonic: bool
    rungs: list[RungScore] = field(default_factory = list)
    rung_weights: list[float] = field(default_factory = list)
    weights_id: str = ""
    rung_ladder_id: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "aggregate": round(float(self.aggregate), 3),
            "onset_rung_tokens": self.onset_rung_tokens,
            "onset_reason": self.onset_reason,
            "non_monotonic": bool(self.non_monotonic),
            "weights_id": self.weights_id,
            "rung_ladder_id": self.rung_ladder_id,
            "rung_weights": [round(w, 5) for w in self.rung_weights],
            "rungs": [r.to_json() for r in self.rungs],
        }


def score_ladder(rungs: Sequence[RungScore]) -> LadderScore:
    """Aggregate scored rungs, and pick the onset rung.

    Every rung on the declared ladder must be present, complete or not. A caller that only ran
    three of five rungs must still hand over the other two as incomplete, because silently
    aggregating over what was attempted is the crash-beats-limp bug in a different costume.
    """

    ordered = sorted(rungs, key = lambda r: r.tokens)
    weights = log_rung_weights([r.tokens for r in ordered])
    aggregate = sum(w * float(r.score) for w, r in zip(weights, ordered))

    usable = [r for r in ordered if r.usable]
    onset = usable[-1].tokens if usable else None
    if onset is None:
        smallest = ordered[0].tokens if ordered else None
        onset_reason = (
            f"no rung was usable; the smallest rung on the ladder ({smallest:,} tokens) already "
            "fails the usability gate"
            if smallest is not None
            else "no rungs were scored"
        )
    else:
        above = [r for r in ordered if r.tokens > onset]
        if above:
            onset_reason = (
                f"usable at {onset:,} tokens; the next rung "
                f"({above[0].tokens:,}) scores {above[0].score:.1f}"
            )
        else:
            onset_reason = (
                f"usable at {onset:,} tokens, the top of the declared ladder; the true ceiling "
                "is above what this ladder measures"
            )

    # Usability is expected to be monotone in thread size. When it is not, something other than thread
    # size moved (throttling, a background process, an unstable machine) and the onset rung is not
    # trustworthy on its own.
    non_monotonic = False
    seen_unusable = False
    for rung in ordered:
        if not rung.usable:
            seen_unusable = True
        elif seen_unusable:
            non_monotonic = True

    return LadderScore(
        aggregate = aggregate,
        onset_rung_tokens = onset,
        onset_reason = onset_reason,
        non_monotonic = non_monotonic,
        rungs = list(ordered),
        rung_weights = weights,
        weights_id = weights_id(),
        rung_ladder_id = rung_ladder_id([r.tokens for r in ordered] or RUNG_TOKENS),
    )
