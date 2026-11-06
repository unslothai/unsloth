# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paired A/B within one session; a failed null control voids the run; a win needs a CI clearing 1.0."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .anchors import DEFAULT_NOISE_FLOOR_PCT, METRIC_BY_KEY
from .schema import Measure


class IncomparableRuns(AssertionError):
    """Raised when two runs cannot be put in the same table."""


@dataclass(frozen = True)
class RunIdentity:
    """Everything that has to match before two sets of numbers may be compared."""

    bench_version: str
    corpus_hash: str
    rung_ladder_id: str
    weights_id: str
    session_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "bench_version": self.bench_version,
            "corpus_hash": self.corpus_hash,
            "rung_ladder_id": self.rung_ladder_id,
            "weights_id": self.weights_id,
            "session_id": self.session_id,
        }


#: `session_id` is checked separately: a mismatch there is drift, not a different meaning.
COMPARABILITY_FIELDS = ("bench_version", "corpus_hash", "rung_ladder_id", "weights_id")


def assert_comparable(base: RunIdentity, treatment: RunIdentity) -> None:
    problems = [
        f"{field_name}: base={getattr(base, field_name)!r} treatment={getattr(treatment, field_name)!r}"
        for field_name in COMPARABILITY_FIELDS
        if getattr(base, field_name) != getattr(treatment, field_name)
    ]
    if problems:
        raise IncomparableRuns(
            "refusing to render an A/B table; these differ between the two sides:\n  "
            + "\n  ".join(problems)
        )
    if base.session_id != treatment.session_id:
        raise IncomparableRuns(
            "refusing to render an A/B table across sessions: base session "
            f"{base.session_id!r} != treatment session {treatment.session_id!r}. "
            "Cross-session drift measured 8% on this machine, which is larger than most real "
            "wins. Interleave both sides inside one session."
        )


@dataclass
class Pair:
    """One matched base/treatment reading at one rung for one metric."""

    rung_tokens: int
    metric_key: str
    base: Measure
    treatment: Measure

    @staticmethod
    def _divisible(measure: Measure) -> float | None:
        """A measured zero is a reading; a sub-floor value uses the floor, giving a bound rather
        than infinity."""
        if not measure.has_reading:
            return None
        value = float(measure.value)
        if measure.sub_floor and measure.floor is not None and value >= 0:
            return float(measure.floor)
        return value if value > 0 else None

    @property
    def usable(self) -> bool:
        return (
            self._divisible(self.base) is not None and self._divisible(self.treatment) is not None
        )

    @property
    def bounded(self) -> bool:
        """True when either arm was sub-floor, so the ratio is a bound and not a point estimate."""
        return self.usable and (self.base.sub_floor or self.treatment.sub_floor)

    @property
    def ratio(self) -> float:
        """treatment / base. Below 1 is faster for a lower-is-better metric."""

        return float(self._divisible(self.treatment)) / float(self._divisible(self.base))

    def to_json(self) -> dict[str, Any]:
        return {
            "rung_tokens": int(self.rung_tokens),
            "metric_key": self.metric_key,
            "base": self.base.to_json(),
            "treatment": self.treatment.to_json(),
            "ratio": self.ratio if self.usable else None,
            "usable": self.usable,
            "bounded": self.bounded,
        }


@dataclass
class MetricComparison:
    """One metric's paired result, with the range across rungs and a bootstrap CI."""

    metric_key: str
    n_pairs: int
    ratio_geomean: float | None
    ratio_min: float | None
    ratio_max: float | None
    ci_low: float | None
    ci_high: float | None
    verdict: str = "no_reading"
    beyond_noise: bool = False
    #: The paired geomean 95% CI contains 1.0, so the effect size is unresolved.
    ci_spans_no_effect: bool = False
    #: A contributing pair had a sub-floor arm, so the ratio is only a bound.
    bounded: bool = False

    @property
    def ci_rules_out_no_effect(self) -> bool:
        """An interval exists AND it lies entirely on one side of 1.0."""
        if self.ci_low is None or self.ci_high is None:
            return False
        return not (self.ci_low <= 1.0 <= self.ci_high)

    @property
    def withheld(self) -> bool:
        """Only the better side is withheld; an unresolved regression still counts toward the headline."""
        return self.verdict == "inconclusive"

    @property
    def unresolved(self) -> bool:
        """Past the floor but no CI clears 1.0; a missing CI counts as unresolved, never as permission."""
        return self.beyond_noise and not self.ci_rules_out_no_effect

    @property
    def resolves_direction(self) -> bool:
        """Cleared the floor AND its own CI, so this metric can speak for a headline."""
        return self.verdict in ("improved", "regressed")

    def to_json(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "n_pairs": int(self.n_pairs),
            "ratio_geomean": self.ratio_geomean,
            "ratio_range": [self.ratio_min, self.ratio_max],
            "ci95": [self.ci_low, self.ci_high],
            "verdict": self.verdict,
            "beyond_noise": bool(self.beyond_noise),
            "ci_spans_no_effect": bool(self.ci_spans_no_effect),
            "bounded": bool(self.bounded),
        }


def bootstrap_geomean_ci(
    ratios: Sequence[float],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> tuple[float | None, float | None]:
    """Resamples over pairs, the unit that was randomised; resampling readings would discard the pairing."""

    usable = [float(r) for r in ratios if r is not None and r > 0 and math.isfinite(r)]
    if len(usable) < 3:
        return None, None
    rng = random.Random(bootstrap_seed)
    logs = [math.log(r) for r in usable]
    draws: list[float] = []
    n = len(logs)
    for _ in range(iterations):
        sample = [logs[rng.randrange(n)] for _ in range(n)]
        draws.append(math.exp(sum(sample) / n))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lo = draws[max(0, int(math.floor(tail * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1.0 - tail) * len(draws))) - 1)]
    return lo, hi


def _geomean(values: Sequence[float]) -> float | None:
    usable = [float(v) for v in values if v is not None and v > 0 and math.isfinite(v)]
    if not usable:
        return None
    return math.exp(sum(math.log(v) for v in usable) / len(usable))


@dataclass
class AbResult:
    """The whole comparison, including the reason it may not be quoted."""

    label: str
    identity_base: RunIdentity
    identity_treatment: RunIdentity
    noise_floor_pct: float
    noise_floor_source: str
    metrics: list[MetricComparison] = field(default_factory = list)
    pairs: list[Pair] = field(default_factory = list)
    void: bool = False
    void_reason: str | None = None
    regressions: list[str] = field(default_factory = list)
    headline_ratio: float | None = None
    is_null_control: bool = False

    @property
    def verdict(self) -> str:
        if self.void:
            return "VOID"
        if self.regressions:
            return "FAIL"
        if self.headline_ratio is None:
            # Unresolved metrics are excluded from the headline, so this is INCONCLUSIVE, not NO READING.
            if any(m.unresolved for m in self.metrics):
                return "INCONCLUSIVE"
            return "NO READING"
        # Dropping an unresolved mover can leave only flat metrics; do not claim no difference then.
        if any(m.unresolved for m in self.metrics) and not any(
            m.resolves_direction for m in self.metrics
        ):
            return "INCONCLUSIVE"
        if abs(self.headline_ratio - 1.0) * 100.0 <= self.noise_floor_pct:
            return "NO DIFFERENCE"
        return "IMPROVED" if self.headline_ratio < 1.0 else "REGRESSED"

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "is_null_control": bool(self.is_null_control),
            "identity_base": self.identity_base.to_json(),
            "identity_treatment": self.identity_treatment.to_json(),
            "noise_floor_pct": float(self.noise_floor_pct),
            "noise_floor_source": self.noise_floor_source,
            "void": bool(self.void),
            "void_reason": self.void_reason,
            "verdict": self.verdict,
            "regressions": list(self.regressions),
            "headline_ratio": self.headline_ratio,
            "metrics": [m.to_json() for m in self.metrics],
            "pairs": [p.to_json() for p in self.pairs],
        }


def compare(
    label: str,
    pairs: Sequence[Pair],
    identity_base: RunIdentity,
    identity_treatment: RunIdentity,
    *,
    noise_floor_pct: float = DEFAULT_NOISE_FLOOR_PCT,
    noise_floor_source: str = "declared default",
    is_null_control: bool = False,
    bootstrap_seed: int = 0,
) -> AbResult:
    """Raises on identity mismatch; returns a VOID result, not an error, when data cannot support a
    claim."""

    assert_comparable(identity_base, identity_treatment)

    result = AbResult(
        label = label,
        identity_base = identity_base,
        identity_treatment = identity_treatment,
        noise_floor_pct = float(noise_floor_pct),
        noise_floor_source = noise_floor_source,
        pairs = list(pairs),
        is_null_control = is_null_control,
    )

    by_metric: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_metric.setdefault(pair.metric_key, []).append(pair)

    weighted_logs: list[tuple[float, float]] = []
    for metric_key, metric_pairs in sorted(by_metric.items()):
        usable = [p for p in metric_pairs if p.usable]
        ratios = [p.ratio for p in usable]
        geo = _geomean(ratios)
        lo, hi = bootstrap_geomean_ci(ratios, bootstrap_seed = bootstrap_seed)
        comparison = MetricComparison(
            metric_key = metric_key,
            n_pairs = len(usable),
            ratio_geomean = geo,
            ratio_min = min(ratios) if ratios else None,
            ratio_max = max(ratios) if ratios else None,
            ci_low = lo,
            ci_high = hi,
            bounded = any(p.bounded for p in usable),
        )
        if geo is None:
            comparison.verdict = "no_reading"
        else:
            delta_pct = (geo - 1.0) * 100.0
            anchor = METRIC_BY_KEY.get(metric_key)
            lower_is_better = anchor.lower_is_better if anchor else True
            worse = delta_pct > 0 if lower_is_better else delta_pct < 0
            comparison.beyond_noise = abs(delta_pct) > noise_floor_pct
            comparison.ci_spans_no_effect = lo is not None and hi is not None and lo <= 1.0 <= hi
            if not comparison.beyond_noise:
                comparison.verdict = "within noise"
            elif worse:
                # Deliberately asymmetric: an unresolved win is withheld, an unresolved loss keeps counting.
                comparison.verdict = (
                    "regressed (unresolved)" if comparison.ci_spans_no_effect else "regressed"
                )
                unresolved = (
                    f"; 95% CI {lo:.3f}-{hi:.3f} spans no effect, so the size is unresolved"
                    if comparison.ci_spans_no_effect
                    else ""
                )
                result.regressions.append(
                    f"{metric_key}: {abs(delta_pct):.1f}% worse "
                    f"(noise floor {noise_floor_pct:.1f}%{unresolved})"
                )
            else:
                comparison.verdict = "inconclusive" if comparison.unresolved else "improved"
            # Unresolved metrics are excluded from the headline geomean so a wide-CI mover cannot dominate it.
            if not comparison.withheld:
                weight = anchor.weight if anchor else 1.0
                weighted_logs.append((weight, math.log(geo)))
        result.metrics.append(comparison)

    if weighted_logs:
        total_weight = sum(w for w, _ in weighted_logs)
        result.headline_ratio = math.exp(sum(w * lg for w, lg in weighted_logs) / total_weight)

    if is_null_control:
        # A null control showing a difference means the harness moved; nothing measured alongside counts.
        offenders = [
            f"{m.metric_key}: {abs((m.ratio_geomean - 1.0) * 100):.1f}%"
            for m in result.metrics
            if m.ratio_geomean is not None and abs(m.ratio_geomean - 1.0) * 100.0 > noise_floor_pct
        ]
        if offenders:
            result.void = True
            result.void_reason = (
                "the null-treatment control (base vs base) moved beyond its own noise floor of "
                f"{noise_floor_pct:.1f}%: " + ", ".join(offenders)
            )
        result.regressions = []

    return result


def noise_floor_from_null_control(
    null_control: AbResult, *, minimum_pct: float = 1.0
) -> tuple[float, str]:
    """Measures spread only: bounded ratios are excluded, since they state a bound, not a deviation."""

    bounded = sum(1 for m in null_control.metrics if m.ratio_geomean is not None and m.bounded)
    deviations = [
        abs(m.ratio_geomean - 1.0) * 100.0
        for m in null_control.metrics
        if m.ratio_geomean is not None and not m.bounded
    ]
    if not deviations:
        why = (
            f"null control produced only bounded ratios ({bounded} metric(s) under an instrument "
            "floor)"
            if bounded
            else "null control produced no ratios"
        )
        return DEFAULT_NOISE_FLOOR_PCT, f"declared default ({why})"
    floor = max(minimum_pct, max(deviations))
    note = f" ({bounded} bounded metric(s) excluded)" if bounded else ""
    return floor, (
        f"measured from the null-treatment control over {len(deviations)} metrics "
        f"(worst deviation {max(deviations):.2f}%){note}"
    )


def pairs_from_cells(
    base_cells: Mapping[int, Mapping[str, Measure]],
    treatment_cells: Mapping[int, Mapping[str, Measure]],
    metric_keys: Iterable[str] | None = None,
) -> list[Pair]:
    """Dropping unmatched readings is right only because they have no partner to form a ratio."""

    keys = list(metric_keys) if metric_keys is not None else list(METRIC_BY_KEY)
    out: list[Pair] = []
    for rung in sorted(set(base_cells) & set(treatment_cells)):
        for key in keys:
            base = base_cells[rung].get(key)
            treatment = treatment_cells[rung].get(key)
            if base is None or treatment is None:
                continue
            out.append(Pair(rung_tokens = int(rung), metric_key = key, base = base, treatment = treatment))
    return out
