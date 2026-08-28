# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The telescoping ladder: how a correlation becomes a cause, and where the risk goes.

A set of independent on/off ablations produces a set of differences that do not add up to the
total, and the gap between them is a residual. A residual is not a measurement of anything. It is
the part nobody can name, and it grows every time an arm is added, because overlapping arms
double-count the same work.

A TELESCOPING LADDER removes the residual by construction. Arms are nested,

    A0 (shipping)  superset  A1  superset  ...  superset  An (floor)

with adjacent pairs differing by exactly one mechanism. Then

    cost(A0) - cost(An)  =  sum of the adjacent differences

IDENTICALLY. Not approximately, not up to a residual: the interior terms cancel algebraically, so
the identity is a property of the arithmetic and any failure of it is a failure of MEASUREMENT
(a missing cell, a voided arm) that the code reports rather than absorbs.

THE RISK DOES NOT VANISH, IT MOVES. What was "an unnamed remainder" becomes "a possibly
mislabelled step". That is a strict improvement for one reason: a step is testable. If step 3 is
labelled "layout geometry" and someone doubts it, they can attack step 3 directly with a
different knob. Nobody can attack a residual.

TWO RULES, ENFORCED IN CODE:

  1. NO ARM IS EVER QUOTED ALONE. `LadderRoute.quote_arm()` raises. A single arm's absolute cost
     is not a measurement of a mechanism; it is a measurement of that arm's whole configuration
     against nothing. Only adjacent differences are quotable.
  2. TWO LADDERS MUST REACH THE SAME FLOOR BY DIFFERENT ROUTES. Where they disagree about the
     same mechanism, that disagreement is the INTERACTION TERM and it is reported as such. It is
     never averaged away: two routes disagreeing by 3 ms means the mechanisms are not additive,
     which is a finding, and the mean of the two hides exactly that finding.

FUSED STEPS. Some mechanisms cannot be separated by any knob available at runtime: `display:none`
removes layout geometry AND sibling count in one move, and there is no knob that removes one
without the other. A step may therefore declare more than one mechanism ONLY when it is marked
`fused` with a reason, and the report prints it as fused. That is honest; pretending it is one
mechanism is how a step gets mislabelled, which is the one risk this design accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..scoring.schema import Measure
from .manifest import ArmOutcome, ArmStatus

#: The named mechanisms this ladder can remove. A step must name mechanisms from this set, so a
#: typo becomes an error rather than a new, unexplained row in the report.
MECHANISMS: tuple[str, ...] = (
    "paint_raster",
    "offscreen_style_layout",
    "layout_geometry",
    "sibling_count",
    "autoscroll_forced_layout",
    "stabilizer_style_invalidation",
    "react_reconciliation",
)

MECHANISM_FIX: Mapping[str, str] = {
    "paint_raster": (
        "the cost is painting and rasterising completed messages. Fix: stop painting what is not "
        "on screen (containment, or a virtualised list)"
    ),
    "offscreen_style_layout": (
        "the cost is style and layout of off-screen content. The obvious fix, content-visibility: "
        "auto, is NOT available: the shipped override that disables it is kept for a height "
        "flicker during stream finalisation and because WebKit below Safari 26 cannot find "
        "skipped content with find-in-page and Unsloth has no in-thread search. So a win here "
        "names the cost and still needs a different mechanism to remove it"
    ),
    "layout_geometry": (
        "the cost is layout geometry of retained messages. Fix: virtualise the message list; "
        "thread.tsx:1741 renders a bare ThreadPrimitive.Messages and nothing is virtualised"
    ),
    "sibling_count": (
        "the cost is proportional to the number of siblings React must walk. Fix: reduce the flat "
        "sibling count (grouping, windowing) or stop the parent update that forces the walk"
    ),
    "autoscroll_forced_layout": (
        "the cost is forced synchronous layout inside the autoscroll MutationObserver. Fix: stop "
        "reading scrollHeight in the observer callback, or scope the observer off the subtree"
    ),
    "stabilizer_style_invalidation": (
        "the cost is style invalidation of the whole subtree from writing an inherited custom "
        "property. Fix: stop writing --aui-scroll-stabilizer on the scroll container per "
        "mutation. Note this is the mechanism the shipped index.css comment ALREADY names -- it "
        "says what grows with thread length is inherited-property style recalc, not layout -- so "
        "a large step here confirms a claim that is currently asserted without a number next to "
        "it, and a small one contradicts it"
    ),
    "react_reconciliation": (
        "the cost is React subscriptions and reconciliation, not DOM. Fix: cut the update rate or "
        "the fibre count reached per update; memo does not help, because bailout still clones one "
        "work-in-progress fibre per sibling when childLanes is set"
    ),
}


class LadderError(AssertionError):
    """Raised when a ladder is malformed or used in a way the design forbids."""


def arms_key(arms: Iterable[str]) -> str:
    """Canonical key for a set of simultaneously applied arms."""

    ordered = sorted(set(arms))
    return "+".join(ordered) if ordered else "shipping"


@dataclass(frozen = True)
class Step:
    """One rung-to-rung transition: which arms are added, which mechanisms that removes."""

    arms_before: frozenset[str]
    arms_after: frozenset[str]
    mechanisms: tuple[str, ...]
    fused: bool = False
    fused_reason: str = ""

    def __post_init__(self) -> None:
        unknown = [m for m in self.mechanisms if m not in MECHANISMS]
        if unknown:
            raise LadderError(f"unknown mechanisms {unknown}; add them to MECHANISMS")
        if not self.mechanisms:
            raise LadderError("a step must remove at least one named mechanism")
        if len(self.mechanisms) > 1 and not self.fused:
            raise LadderError(
                f"step {arms_key(self.arms_before)} -> {arms_key(self.arms_after)} removes "
                f"{len(self.mechanisms)} mechanisms but is not marked fused. Adjacent pairs "
                "differ by exactly one mechanism unless no available knob can separate them, in "
                "which case say so explicitly"
            )
        if self.fused and not self.fused_reason:
            raise LadderError("a fused step must say why the mechanisms cannot be separated")
        if not self.arms_before < self.arms_after:
            raise LadderError(
                f"a ladder step must ADD arms: {arms_key(self.arms_before)} is not a strict "
                f"subset of {arms_key(self.arms_after)}. A telescoping ladder is nested; two "
                "arms that merely differ are two experiments, not a ladder"
            )

    @property
    def label(self) -> str:
        return f"{arms_key(self.arms_before)} -> {arms_key(self.arms_after)}"

    def to_json(self) -> dict[str, Any]:
        return {
            "arms_before": sorted(self.arms_before),
            "arms_after": sorted(self.arms_after),
            "label": self.label,
            "mechanisms": list(self.mechanisms),
            "fused": bool(self.fused),
            "fused_reason": self.fused_reason,
        }


@dataclass
class StepResult:
    """One adjacent difference, which is the only kind of number a ladder may quote."""

    step: Step
    difference: Measure
    quotable: bool
    reason: str
    bound_only: bool = False

    def quote(self) -> str:
        if not self.quotable:
            return f"NOT QUOTABLE ({self.reason})"
        prefix = "<= " if self.bound_only else ""
        return f"{prefix}{self.difference.display()}"

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step.to_json(),
            "difference": self.difference.to_json(),
            "quotable": bool(self.quotable),
            "bound_only": bool(self.bound_only),
            "reason": self.reason,
            "quote": self.quote(),
            "implies_fix": [MECHANISM_FIX[m] for m in self.step.mechanisms],
        }


@dataclass
class LadderRoute:
    """One nested route from the shipping build down to the floor."""

    route_id: str
    name: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise LadderError(f"{self.route_id}: a route needs at least one step")
        for earlier, later in zip(self.steps, self.steps[1:]):
            if earlier.arms_after != later.arms_before:
                raise LadderError(
                    f"{self.route_id}: steps are not contiguous; "
                    f"{arms_key(earlier.arms_after)} != {arms_key(later.arms_before)}. A gap in "
                    "the chain reintroduces the residual this design exists to remove"
                )
        seen: list[str] = []
        for step in self.steps:
            for mechanism in step.mechanisms:
                if mechanism in seen:
                    raise LadderError(
                        f"{self.route_id}: mechanism {mechanism!r} is removed twice. A mechanism "
                        "removed twice is counted twice, which is the residual wearing a hat"
                    )
                seen.append(mechanism)

    @property
    def top(self) -> frozenset[str]:
        return self.steps[0].arms_before

    @property
    def floor(self) -> frozenset[str]:
        return self.steps[-1].arms_after

    @property
    def mechanisms(self) -> tuple[str, ...]:
        return tuple(m for step in self.steps for m in step.mechanisms)

    def quote_arm(self, arms: Iterable[str]) -> str:
        """Refuse. An arm's absolute cost is not a measurement of a mechanism."""

        raise LadderError(
            f"refusing to quote arm {arms_key(arms)} on its own. A single arm's cost is the cost "
            "of its whole configuration against nothing; only adjacent differences on this "
            "ladder isolate a mechanism. Use differences()"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "name": self.name,
            "top": sorted(self.top),
            "floor": sorted(self.floor),
            "steps": [s.to_json() for s in self.steps],
        }


@dataclass
class RouteResult:
    """A whole route's differences, plus the telescoping identity check."""

    route: LadderRoute
    steps: list[StepResult] = field(default_factory = list)
    total: Measure | None = None
    sum_of_steps: Measure | None = None
    residual_ms: float = 0.0
    identity_holds: bool = False
    identity_note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "route": self.route.to_json(),
            "steps": [s.to_json() for s in self.steps],
            "total": self.total.to_json() if self.total else None,
            "sum_of_steps": self.sum_of_steps.to_json() if self.sum_of_steps else None,
            "residual_ms": float(self.residual_ms),
            "identity_holds": bool(self.identity_holds),
            "identity_note": self.identity_note,
        }


def _cost_of(
    outcomes: Mapping[str, ArmOutcome], arms: frozenset[str]
) -> tuple[Measure | None, str, bool]:
    """Look up one rung's cost, returning (measure, reason, bound_only)."""

    key = arms_key(arms)
    outcome = outcomes.get(key)
    if outcome is None:
        return None, f"no cell was recorded for rung {key}", False
    if outcome.status is ArmStatus.VOIDED:
        return None, f"rung {key} is VOIDED: {outcome.reason}", False
    if outcome.status is ArmStatus.NOT_RUN:
        return None, f"rung {key} reads NOT RUN: {outcome.reason}", False
    if outcome.status is ArmStatus.UNAVAILABLE:
        return None, f"rung {key} is unavailable: {outcome.reason}", False
    if not outcome.cost.has_reading:
        return None, f"rung {key} produced no reading: {outcome.cost.note}", False
    return outcome.cost, "", outcome.status is ArmStatus.BOUND


def differences(
    route: LadderRoute,
    outcomes: Mapping[str, ArmOutcome],
    *,
    detection_floor_ms: float | None = None,
) -> RouteResult:
    """Compute every adjacent difference on a route and check the telescoping identity.

    `outcomes` is keyed by `arms_key(...)`, so the shipping rung is `"shipping"` and a rung with
    arms C and D applied is `"C+D"`.

    The identity is checked only when EVERY rung on the route produced a reading. A route with a
    voided or missing rung has no identity to check, and reporting `residual = 0` for it would be
    a claim about arithmetic that was never performed.
    """

    result = RouteResult(route = route)
    all_readable = True

    for step in route.steps:
        before, before_reason, before_bound = _cost_of(outcomes, step.arms_before)
        after, after_reason, after_bound = _cost_of(outcomes, step.arms_after)
        if before is None or after is None:
            all_readable = False
            result.steps.append(
                StepResult(
                    step = step,
                    difference = Measure.failed("ms", before_reason or after_reason),
                    quotable = False,
                    reason = before_reason or after_reason,
                )
            )
            continue
        delta = float(before.value) - float(after.value)
        measure = Measure.read(delta, before.unit, floor = detection_floor_ms)
        bound_only = before_bound or after_bound
        result.steps.append(
            StepResult(
                step = step,
                difference = measure,
                quotable = True,
                bound_only = bound_only,
                reason = (
                    "one or both rungs are DOM-changing, so this difference bounds the mechanism"
                    if bound_only
                    else "both rungs held their invariance and fired"
                ),
            )
        )

    top, top_reason, _ = _cost_of(outcomes, route.top)
    floor, floor_reason, _ = _cost_of(outcomes, route.floor)
    if top is not None and floor is not None:
        result.total = Measure.read(
            float(top.value) - float(floor.value), top.unit, floor = detection_floor_ms
        )
    else:
        result.total = Measure.failed("ms", top_reason or floor_reason)

    readable_steps = [s for s in result.steps if s.quotable]
    if readable_steps:
        result.sum_of_steps = Measure.read(
            sum(float(s.difference.value) for s in readable_steps),
            readable_steps[0].difference.unit,
            floor = detection_floor_ms,
        )

    if (
        all_readable
        and result.total is not None
        and result.total.has_reading
        and result.sum_of_steps
    ):
        result.residual_ms = float(result.total.value) - float(result.sum_of_steps.value)
        # This is an arithmetic identity, not an empirical claim: the interior terms cancel. A
        # non-zero residual here means a bug in this function, not a finding about the app.
        result.identity_holds = abs(result.residual_ms) < 1e-9
        result.identity_note = (
            "top minus floor equals the sum of the adjacent differences, identically, because the "
            "interior terms cancel. There is no residual to attribute"
            if result.identity_holds
            else (
                f"residual {result.residual_ms:.9f} ms. This is an arithmetic identity, so a "
                "non-zero value is a bug in the ladder code, not a property of the app"
            )
        )
    else:
        result.identity_holds = False
        result.identity_note = (
            "the identity was not checked: at least one rung on this route produced no usable "
            "reading, so there is no complete chain to telescope"
        )
    return result


@dataclass
class InteractionTerm:
    """One mechanism, measured on two routes, and how much they disagree."""

    mechanism: str
    route_a: str
    route_b: str
    value_a: Measure
    value_b: Measure
    disagreement_ms: float | None
    disagreement_pct: float | None
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "route_a": self.route_a,
            "route_b": self.route_b,
            "value_a": self.value_a.to_json(),
            "value_b": self.value_b.to_json(),
            "disagreement_ms": self.disagreement_ms,
            "disagreement_pct": self.disagreement_pct,
            "note": self.note,
        }


def interaction_terms(
    result_a: RouteResult,
    result_b: RouteResult,
    *,
    detection_floor_ms: float | None = None,
) -> list[InteractionTerm]:
    """Compare two routes mechanism by mechanism. Disagreement is reported, never averaged.

    Two routes reaching the same floor must agree about each mechanism if the mechanisms are
    additive. Where they do not, the mechanisms interact: removing the autoscroll observer first
    makes the paint step look cheaper, because some of that paint was being forced by the
    observer. That is a real property of the system and it is the most interesting thing a
    two-route ladder can find. Averaging the two numbers produces one that describes neither
    route and hides the finding completely.
    """

    if result_a.route.floor != result_b.route.floor:
        raise LadderError(
            "the two routes do not reach the same floor: "
            f"{arms_key(result_a.route.floor)} vs {arms_key(result_b.route.floor)}. Two ladders "
            "that end somewhere different are not two routes to one answer"
        )

    def by_mechanism(result: RouteResult) -> dict[str, StepResult]:
        out: dict[str, StepResult] = {}
        for step_result in result.steps:
            for mechanism in step_result.step.mechanisms:
                out[mechanism] = step_result
        return out

    a_map, b_map = by_mechanism(result_a), by_mechanism(result_b)
    terms: list[InteractionTerm] = []
    for mechanism in sorted(set(a_map) & set(b_map)):
        step_a, step_b = a_map[mechanism], b_map[mechanism]
        value_a, value_b = step_a.difference, step_b.difference
        if not (value_a.has_reading and value_b.has_reading):
            terms.append(
                InteractionTerm(
                    mechanism = mechanism,
                    route_a = result_a.route.route_id,
                    route_b = result_b.route.route_id,
                    value_a = value_a,
                    value_b = value_b,
                    disagreement_ms = None,
                    disagreement_pct = None,
                    note = "one route produced no reading for this mechanism",
                )
            )
            continue
        delta = float(value_a.value) - float(value_b.value)
        base = max(abs(float(value_a.value)), abs(float(value_b.value)))
        pct = (abs(delta) / base * 100.0) if base > 0 else None
        if detection_floor_ms is not None and abs(delta) < detection_floor_ms:
            note = (
                "the two routes agree to within the detection floor, so the mechanisms are "
                "additive as far as this instrument can tell"
            )
        elif step_a.step.fused or step_b.step.fused:
            note = (
                "the routes disagree and at least one side measures this mechanism inside a "
                "fused step, so part of the disagreement may be the fusion rather than an "
                "interaction"
            )
        else:
            note = (
                "the routes disagree beyond the detection floor: these mechanisms are not "
                "additive. Reported as an interaction term, not averaged"
            )
        terms.append(
            InteractionTerm(
                mechanism = mechanism,
                route_a = result_a.route.route_id,
                route_b = result_b.route.route_id,
                value_a = value_a,
                value_b = value_b,
                disagreement_ms = delta,
                disagreement_pct = pct,
                note = note,
            )
        )
    return terms


# ---------------------------------------------------------------------------------------
# the two declared routes
# ---------------------------------------------------------------------------------------

_SHIPPING: frozenset[str] = frozenset()

#: Route 1 walks down the rendering pipeline first: stop painting, then stop laying out
#: off-screen, then stop occupying layout at all, and only then touch the observers and React.
ROUTE_VISUAL_FIRST = LadderRoute(
    route_id = "visual_first",
    name = "paint, then off-screen layout, then geometry, then observers, then React",
    steps = (
        Step(
            arms_before = _SHIPPING,
            arms_after = frozenset({"A"}),
            mechanisms = ("paint_raster",),
        ),
        Step(
            arms_before = frozenset({"A"}),
            arms_after = frozenset({"A", "B"}),
            mechanisms = ("offscreen_style_layout",),
        ),
        Step(
            arms_before = frozenset({"A", "B"}),
            arms_after = frozenset({"A", "B", "C"}),
            mechanisms = ("layout_geometry", "sibling_count"),
            fused = True,
            fused_reason = (
                "display:none removes the element from layout AND from the sibling sequence React "
                "walks. No runtime knob removes one without the other, so the step is quoted as "
                "the pair rather than mislabelled as either one"
            ),
        ),
        Step(
            arms_before = frozenset({"A", "B", "C"}),
            arms_after = frozenset({"A", "B", "C", "D"}),
            mechanisms = ("autoscroll_forced_layout",),
        ),
        Step(
            arms_before = frozenset({"A", "B", "C", "D"}),
            arms_after = frozenset({"A", "B", "C", "D", "E"}),
            mechanisms = ("stabilizer_style_invalidation",),
        ),
        Step(
            arms_before = frozenset({"A", "B", "C", "D", "E"}),
            arms_after = frozenset({"A", "B", "C", "D", "E", "F"}),
            mechanisms = ("react_reconciliation",),
        ),
    ),
)

#: Route 2 reaches the identical floor from the other end: kill the observers and React first,
#: then walk down the rendering pipeline. If the mechanisms are additive both routes report the
#: same per-mechanism numbers. Where they do not, that gap is the interaction term.
ROUTE_SCHEDULER_FIRST = LadderRoute(
    route_id = "scheduler_first",
    name = "observers, then React, then paint, then off-screen layout, then geometry",
    steps = (
        Step(
            arms_before = _SHIPPING,
            arms_after = frozenset({"D"}),
            mechanisms = ("autoscroll_forced_layout",),
        ),
        Step(
            arms_before = frozenset({"D"}),
            arms_after = frozenset({"D", "E"}),
            mechanisms = ("stabilizer_style_invalidation",),
        ),
        Step(
            arms_before = frozenset({"D", "E"}),
            arms_after = frozenset({"D", "E", "F"}),
            mechanisms = ("react_reconciliation",),
        ),
        Step(
            arms_before = frozenset({"D", "E", "F"}),
            arms_after = frozenset({"A", "D", "E", "F"}),
            mechanisms = ("paint_raster",),
        ),
        Step(
            arms_before = frozenset({"A", "D", "E", "F"}),
            arms_after = frozenset({"A", "B", "D", "E", "F"}),
            mechanisms = ("offscreen_style_layout",),
        ),
        Step(
            arms_before = frozenset({"A", "B", "D", "E", "F"}),
            arms_after = frozenset({"A", "B", "C", "D", "E", "F"}),
            mechanisms = ("layout_geometry", "sibling_count"),
            fused = True,
            fused_reason = (
                "same fusion as the other route: display:none is the only knob that removes "
                "layout geometry, and it removes the sibling from the sequence at the same time"
            ),
        ),
    ),
)

DECLARED_ROUTES: tuple[LadderRoute, ...] = (ROUTE_VISUAL_FIRST, ROUTE_SCHEDULER_FIRST)


def required_rungs(routes: Sequence[LadderRoute] = DECLARED_ROUTES) -> list[frozenset[str]]:
    """Every distinct arm-combination the declared routes need a cell for."""

    seen: dict[str, frozenset[str]] = {}
    for route in routes:
        for step in route.steps:
            seen[arms_key(step.arms_before)] = step.arms_before
            seen[arms_key(step.arms_after)] = step.arms_after
    return [seen[key] for key in sorted(seen)]
