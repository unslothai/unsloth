# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A/B comparison: paired ratios, a bootstrap CI, and four ways to refuse to answer.

The comparison is PAIRED AND INTERLEAVED WITHIN ONE SESSION. Cross-session drift on the measured
machine ran to 8%: the same build, measured an hour apart, differs by more than most real wins.
Two runs one after the other therefore cannot be subtracted, and this module will not do it. What
it compares is base and treatment cells that were recorded alternately inside a single browser
session, matched by (rung, metric), so whatever drifted drifted through both sides.

THE NULL-TREATMENT CONTROL RUNS FIRST AND CAN VOID THE WHOLE THING. Before any real comparison,
base is compared against base under two different arm ids. Whatever spread that produces IS the
noise floor for this machine on this day; it is measured, not assumed. If the null control itself
shows a difference outside its own band, the harness is not currently capable of resolving a
difference and every number downstream of it is unquotable. That is reported as VOID, not as a
result with a caveat.

A REGRESSION BEYOND THE NOISE FLOOR IS A FAIL REGARDLESS OF THE HEADLINE. A change that improves
the aggregate by 12% while tripling the worst frame is not a win, and a single headline number is
exactly the instrument that would let it ship. Every metric is checked individually.

A POINT ESTIMATE PAST THE NOISE FLOOR IS NOT YET AN EFFECT. The noise floor answers "could this
machine resolve a difference of this size at all", which is a question about the harness; it says
nothing about whether these repetitions agree. Ratios of 0.7, 0.7, 1.2, 1.2 have a geometric mean
of 0.917, clear of a 5% floor, and a bootstrap CI of 0.700-1.200 that contains 1.0: the pairs do
not agree on the sign, and the honest reading is that nothing was resolved. So a direction is only
claimed when the CI clears 1.0, and an interval that does not exist cannot clear it: below three
usable pairs there is no bootstrap CI at all, and that reads as unresolved rather than as
permission. The refusal is one-sided: an unresolved WIN is withheld from the headline, while an
unresolved regression keeps both its FAIL and its place in the aggregate, where it can only pull
the number toward worse. Withholding a win costs a headline; withholding a loss ships it. `sweep/floor_table.py` applies the same rule to its own pooled ratios under
"VOID (pairs disagree on sign)".

FOUR REFUSALS. Rendering is refused outright when `bench_version`, `corpus_hash`, `rung_ladder_id`
or `weights_id` differ between the two sides. Each of those changes what the numbers mean, and a
table that prints them side by side is not a comparison, it is a category error with column
headers.
"""

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


#: The fields that must match. `session_id` is separate because its failure mode is different:
#: mismatched ids mean different meanings, a mismatched session means an 8% drift term.
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
        """The value this arm contributes to a ratio, or None when it cannot contribute one.

        A MEASURED ZERO IS A READING, AND `> 0` THREW THAT AWAY. `time_in_jank_pct` and
        `jank_index` are 0.0 on any arm smooth enough to have no over-budget frames, which is the
        ordinary state of a healthy base. Requiring both values to be strictly positive dropped
        the pair, so a treatment that introduced 5% time-in-jank over a zero-jank base reported
        `no reading` -- a false statement about two arms that both read -- kept the regression out
        of the table, the headline and `result.regressions`, and, as a null control, neither voided
        the run nor contributed to the noise floor derived from it. That last one is the worst of
        the three: the floor is what every later comparison is judged against, so a null control
        blind to the jank it introduced silently shrinks the effect size anything else can claim.

        The rule is `score.py`'s, which the ladder scorer has always applied to exactly this case:
        a sub-floor reading is at least as good as the floor, so use the floor rather than the raw
        value. Dividing by it yields a BOUND on the ratio -- understating the regression, never
        overstating it -- instead of an infinity or a silence. Two sub-floor arms give floor/floor
        = 1.0, which is the honest answer and the one score.py's comment is about: instrument noise
        on a fast machine must not invent a difference between two perfect builds.

        WHAT THIS DELIBERATELY DOES NOT ADMIT. It keys on `has_reading`, so a measure that was
        never attempted or that was attempted and failed still contributes nothing: those carry
        `value is None` and are a different thing from a measured zero, which is the distinction
        `frames.py` makes when it refuses to score an unscheduled rAF loop as zero jank. A zero
        with no declared floor stays unusable too, because nothing bounds it. So this admits
        readings that were taken and still excludes readings that were not.
        """
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
    #: The 95% CI of the paired geometric mean contains 1.0, so the data cannot rule out "no
    #: effect" however far the point estimate landed from the noise floor. Carried so the table
    #: can say the size is unresolved rather than print a direction as a finding.
    ci_spans_no_effect: bool = False
    #: At least one contributing pair had a sub-floor arm, so the ratio is a BOUND: the true
    #: magnitude is larger. Carried so the table can say so rather than let a number derived from
    #: an instrument floor be quoted as a measurement.
    bounded: bool = False

    @property
    def ci_rules_out_no_effect(self) -> bool:
        """An interval exists AND it lies entirely on one side of 1.0."""
        if self.ci_low is None or self.ci_high is None:
            return False
        return not (self.ci_low <= 1.0 <= self.ci_high)

    @property
    def withheld(self) -> bool:
        """An unresolved BETTER-side metric, whose magnitude is kept out of the headline.

        Only the better side is withheld. An unresolved regression keeps contributing, where it
        can only pull the aggregate toward worse; dropping it would make the headline read rosier
        than the run actually was, which is the one direction this table must never round toward.
        """
        return self.verdict == "inconclusive"

    @property
    def unresolved(self) -> bool:
        """Moved past the floor without an interval that rules out no effect.

        Covers two cases, and the second is the one that failed open. An interval can straddle
        1.0, or there can be no interval at all: `bootstrap_geomean_ci` returns `(None, None)`
        below three usable pairs, which a short ladder or a partially measured metric reaches
        easily. A rule that claims a direction only when the CI clears 1.0 cannot be satisfied
        by a CI that does not exist, so an absent one has to read as unresolved rather than as
        permission. Two pairs at 0.5 used to print a 50% win with no interval behind it.
        """
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
    """Percentile bootstrap CI of the geometric mean of paired ratios.

    Resampling is over PAIRS, which is the unit that was randomised. Resampling over individual
    readings would treat base and treatment as independent samples and throw away the pairing
    that is doing all the work here.
    """

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
            # UNRESOLVED IS NOT UNMEASURED. `compare` keeps unresolved metrics out of the headline
            # entirely, so a run whose every moving metric straddled 1.0 arrives here with no
            # ratio at all. That is not the same as an empty payload: the data exists and says
            # nothing, which is INCONCLUSIVE, while NO READING means there was nothing to read.
            if any(m.unresolved for m in self.metrics):
                return "INCONCLUSIVE"
            return "NO READING"
        # NO DIFFERENCE IS A CLAIM, AND A STRONGER ONE THAN THIS DATA SUPPORTS. Excluding an
        # unresolved mover from the headline can leave only flat metrics behind, putting the
        # aggregate back inside the noise floor. Reporting "no difference" there would assert the
        # change did nothing, when in fact one metric moved and could not resolve its own sign.
        # Only say that once some metric actually resolved a direction.
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
    """Build one A/B result from interleaved paired cells.

    Refuses (raises) on identity mismatch. Produces a VOID result, rather than raising, when the
    data is present but cannot support a claim: that distinction matters because the first is a
    caller bug and the second is a fact about the machine that belongs in the report.
    """

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
                # THE ASYMMETRY IS THE POINT, ON BOTH THE LABEL AND THE HEADLINE. Withholding an
                # unresolved win costs a headline; withholding an unresolved loss ships the
                # regression. So the worse side keeps counting and keeps contributing to the
                # aggregate, where it can only drag the number toward worse, which is the
                # conservative direction. A single bounded pair over a zero base is the case that
                # proves it: `_divisible` exists because that regression once vanished from the
                # table, the headline and this list at once, and it must not vanish again.
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
            # AN UNRESOLVED METRIC LENDS ITS MAGNITUDE TO NOTHING. The headline is a weighted
            # geometric mean of point estimates and carries no interval of its own, so a metric
            # that moved a long way but could not resolve its own sign would otherwise supply most
            # of the number that gets quoted. keystroke_p95_ms at 0.2, 0.2, 1.5, 1.5 (geomean
            # 0.548, CI 0.200-1.500) beside a resolved menu_open_ms of 0.900 produced a headline
            # of 0.631 and the word IMPROVED: a quoted 36.9% win almost entirely made of data the
            # table itself labels inconclusive. Metrics still within noise keep contributing --
            # they pull the headline toward 1.0, which is the conservative direction.
            if not comparison.withheld:
                weight = anchor.weight if anchor else 1.0
                weighted_logs.append((weight, math.log(geo)))
        result.metrics.append(comparison)

    if weighted_logs:
        total_weight = sum(w for w, _ in weighted_logs)
        result.headline_ratio = math.exp(sum(w * lg for w, lg in weighted_logs) / total_weight)

    if is_null_control:
        # A null control that shows a difference is the harness moving, not the build. Whatever
        # it reports, no comparison run on the same machine at the same time can be believed.
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
        # A null control never counts as a regression; it is a measurement of the harness.
        result.regressions = []

    return result


def noise_floor_from_null_control(
    null_control: AbResult, *, minimum_pct: float = 1.0
) -> tuple[float, str]:
    """Derive this machine's noise floor from the null control it just ran.

    The floor is the largest absolute per-metric deviation the null control showed, never below
    `minimum_pct`. Using the measured spread rather than a constant is the difference between
    "this machine can resolve 3%" and "we hope every machine can resolve 5%".

    A BOUNDED RATIO IS NOT A DEVIATION and is excluded here. A metric whose base fell under its
    instrument floor contributes the floor to the ratio, so the result says "at least this much"
    rather than "this much": a null control that moved from no measurable jank to 5% yields a
    ratio of 50 and would publish a 4,900% noise floor, which would then swallow every real effect
    on that machine. Such a control has already set `void` on the same evidence -- the movement is
    real and nothing measured beside it can be believed -- and that is the outcome that belongs to
    it. The floor is a question about SPREAD, and only point estimates can answer it.
    """

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
    """Match base and treatment readings by (rung, metric). Unmatched readings are dropped.

    Dropping is correct here and only here: an unmatched cell has no partner, so there is no
    ratio to compute. It is NOT the same as dropping an incomplete rung from a score, where the
    absence is itself the result.
    """

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
