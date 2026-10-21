# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An incomplete rung scores 0 and keeps its weight; metrics map through log anchors onto [0, 100]."""

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
    """A missing reading is neither 0 nor 100, but is excluded from the mean and counted against
    coverage."""

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
    # A sub-floor reading scores as the floor, so instrument noise cannot separate perfect builds.
    if measure.sub_floor and measure.floor is not None and anchor.lower_is_better:
        value = min(abs(value), float(measure.floor))

    good, bad = float(anchor.good), float(anchor.bad)
    if value <= 0:
        # Log space has no zero: clamp a non-positive reading to the good anchor.
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
    """An incomplete rung is a result about the build, not missing data, so it scores 0."""

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
    """Trapezoid weights over log(tokens), so the top rung does not silently become the whole score."""

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
    """Every declared rung must be passed in, complete or not, or a crash-beats-limp ladder slips
    through."""

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

    # Usability should be monotone in thread size; if not, the onset rung is untrustworthy.
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
