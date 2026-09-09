# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dose-response: the only design in this file set that makes a NULL result informative.

An on/off arm that reads "no difference" is almost worthless. It is consistent with the mechanism
being absent, with the arm not firing, with the instrument being blind, and with the mechanism
being real but smaller than the noise. Those four are not distinguishable from one number.

A DOSE-RESPONSE is different. The same content is split across 4, 40, 400 and 4,000 memoised
siblings. Content is held FIXED, so the amount of text laid out, highlighted and painted does not
change; the only thing that changes is how many siblings React has to walk. If the cost is
O(children) -- one cloned work-in-progress fibre per sibling per render, which is what
`cloneChildFibers` does when `childLanes` is set and `bailoutOnAlreadyFinishedWork` cannot return
null -- then the cost must be a STRAIGHT LINE THROUGH THE ORIGIN in the number of siblings.

That makes both outcomes informative:

  * a straight line through the origin, with a slope well above the minimum detectable slope, is
    a positive identification of an O(children) term, and the slope is its per-child cost;
  * a FLAT line is a real negative, and it comes with a number: "any O(children) term is below
    X microseconds per child", where X is set by the detection floor and the largest dose. An
    on/off arm can never produce that sentence.

WHAT MAKES IT FAIL. A large intercept with a flat slope means the cost is there but is not
proportional to children, so it belongs to some other mechanism. A curve (better fit with a
quadratic than a line) means something superlinear, which at these sizes usually means the
allocator or a cache boundary rather than the walk. Both are reported rather than forced into a
line.

MEMOISED IS LOAD-BEARING. The siblings must be `memo`-wrapped and not re-rendering, because the
claim under test is that React reaches a child it does not render. If the children re-render, the
slope measures rendering, which nobody doubts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..scoring.schema import Measure

#: The declared doses. Four points spanning three orders of magnitude, because a two-point "line"
#: cannot be distinguished from a step and a three-point one cannot show curvature.
DOSES: tuple[int, ...] = (4, 40, 400, 4000)

#: A fit this far from the through-origin model is called nonlinear rather than reported as a
#: slope. Loose, because the question is "is this O(children)", not the exact constant.
MIN_R2_FOR_LINE = 0.90

#: An intercept larger than this fraction of the largest-dose cost means most of the cost is not
#: proportional to children, whatever the slope says.
MAX_INTERCEPT_FRACTION = 0.35


@dataclass
class DosePoint:
    """One dose and its measured cost, at fixed content."""

    dose: int
    cost: Measure
    content_chars: int
    memoised: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "dose": int(self.dose),
            "cost": self.cost.to_json(),
            "content_chars": int(self.content_chars),
            "memoised": bool(self.memoised),
        }


@dataclass
class DoseFit:
    """The two fits, the minimum detectable slope, and the verdict."""

    slope_through_origin: float | None
    r2_through_origin: float | None
    slope_free: float | None
    intercept_free: float | None
    r2_free: float | None
    min_detectable_slope: Measure
    verdict: str
    note: str
    points: list[DosePoint] = field(default_factory = list)
    content_varied: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "slope_through_origin": self.slope_through_origin,
            "r2_through_origin": self.r2_through_origin,
            "slope_free": self.slope_free,
            "intercept_free": self.intercept_free,
            "r2_free": self.r2_free,
            "min_detectable_slope": self.min_detectable_slope.to_json(),
            "verdict": self.verdict,
            "note": self.note,
            "content_varied": bool(self.content_varied),
            "points": [p.to_json() for p in self.points],
        }

    def render(self) -> str:
        lines = ["DOSE-RESPONSE (memoised siblings at fixed content)"]
        for point in self.points:
            lines.append(
                f"  {point.dose:>6,} siblings   {point.cost.display():>26}   "
                f"{point.content_chars:,} chars"
            )
        lines.append("")
        if self.slope_through_origin is not None:
            lines.append(
                f"  through-origin slope  {self.slope_through_origin * 1000:.4f} us per child "
                f"(R2 {self.r2_through_origin:.4f})"
            )
        if self.slope_free is not None:
            lines.append(
                f"  free fit              slope {self.slope_free * 1000:.4f} us per child, "
                f"intercept {self.intercept_free:.4f} ms (R2 {self.r2_free:.4f})"
            )
        lines.append(f"  minimum detectable slope  {self.min_detectable_slope.display()}")
        lines.append(f"  VERDICT: {self.verdict} -- {self.note}")
        return "\n".join(lines)


def _r2(xs: Sequence[float], ys: Sequence[float], predict) -> float | None:
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - predict(x)) ** 2 for x, y in zip(xs, ys))
    if ss_tot <= 0:
        # Every reading identical. R2 is undefined, and reporting 1.0 for a flat line is exactly the wrong
        # way round: this is the case with the least information.
        return None
    return 1.0 - (ss_res / ss_tot)


def fit_dose_response(
    points: Sequence[DosePoint], *, detection_floor_ms: float | None = None
) -> DoseFit:
    """Fit both models and decide what the shape says.

    `detection_floor_ms` sets the minimum detectable slope: below `floor / max_dose` per child,
    no arrangement of these doses could have seen the term, and a flat result is UNDERPOWERED
    rather than a negative. Printing a null without that number is how a real O(children) term
    gets declared absent.
    """

    usable = [p for p in points if p.cost.has_reading]
    contents = {p.content_chars for p in points}
    content_varied = len(contents) > 1

    max_dose = max((p.dose for p in points), default = 0)
    if detection_floor_ms is not None and max_dose > 0:
        min_slope = Measure.read(detection_floor_ms / max_dose, "ms/child")
    else:
        min_slope = Measure.not_attempted(
            "ms/child", "no detection floor was supplied, so a null result cannot be bounded"
        )

    if len(usable) < 3:
        return DoseFit(
            slope_through_origin = None,
            r2_through_origin = None,
            slope_free = None,
            intercept_free = None,
            r2_free = None,
            min_detectable_slope = min_slope,
            verdict = "NO FIT",
            note = (
                f"only {len(usable)} of {len(points)} doses produced a reading; three points is "
                "the minimum that can distinguish a line from a step"
            ),
            points = list(points),
            content_varied = content_varied,
        )

    if content_varied:
        return DoseFit(
            slope_through_origin = None,
            r2_through_origin = None,
            slope_free = None,
            intercept_free = None,
            r2_free = None,
            min_detectable_slope = min_slope,
            verdict = "INVALID",
            note = (
                "content length is not fixed across the doses "
                f"({sorted(contents)}), so a slope in siblings is confounded with a slope in "
                "content and the whole design is void"
            ),
            points = list(points),
            content_varied = True,
        )

    xs = [float(p.dose) for p in usable]
    ys = [float(p.cost.value) for p in usable]

    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    slope_origin = sum_xy / sum_xx if sum_xx > 0 else None
    r2_origin = _r2(xs, ys, lambda x: slope_origin * x) if slope_origin is not None else None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator > 0:
        slope_free = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        intercept_free = mean_y - slope_free * mean_x
        r2_free = _r2(xs, ys, lambda x: slope_free * x + intercept_free)
    else:
        slope_free = intercept_free = r2_free = None

    largest_cost = max(ys)
    floor = detection_floor_ms
    # Detectability is judged on the FREE slope, not the through-origin one: a perfectly flat series
    # still has a large through-origin slope, because forcing the line through zero makes the
    # constant offset look like a per-child cost, which turns every null result into a positive
    # identification.
    detectable = (
        slope_free is not None
        and floor is not None
        and max_dose > 0
        and abs(slope_free) > (floor / max_dose)
    )

    if slope_origin is None:
        verdict, note = "NO FIT", "the doses carry no variation to fit"
    elif not detectable and floor is not None:
        verdict = "UNDERPOWERED NULL"
        note = (
            "the fitted slope is below the minimum this batch could detect. Any O(children) term "
            f"is smaller than {min_slope.display()}, which is a real bound, but this is not "
            "evidence that the term is absent"
        )
    elif not detectable:
        verdict = "NULL, UNBOUNDED"
        note = (
            "the slope is indistinguishable from flat and no detection floor was supplied, so "
            "the result cannot be turned into a bound. Run the calibration arms"
        )
    elif (
        intercept_free is not None
        and r2_free is not None
        and r2_free >= MIN_R2_FOR_LINE
        and largest_cost > 0
        and abs(intercept_free) / largest_cost > MAX_INTERCEPT_FRACTION
    ):
        # Checked BEFORE the through-origin R2, because a straight line with a big offset fits a free line
        # perfectly and a through-origin line badly. Reporting that as NONLINEAR would be backwards: the
        # data is as linear as data gets, it just does not pass through zero, and the offset is the
        # finding.
        verdict = "MOSTLY FIXED COST"
        note = (
            f"the free fit puts {abs(intercept_free) / largest_cost:.0%} of the largest-dose cost "
            "in the intercept, at R2 "
            f"{r2_free:.4f}. Most of what is being measured does not scale with children, "
            "whatever the slope is"
        )
    elif r2_origin is not None and r2_origin < MIN_R2_FOR_LINE:
        verdict = "NONLINEAR"
        note = (
            f"a line through the origin fits at R2 {r2_origin:.3f}, below {MIN_R2_FOR_LINE}. The "
            "cost changes with sibling count but not proportionally, so it is not the simple "
            "per-child walk"
        )
    else:
        verdict = "LINEAR THROUGH ORIGIN"
        note = (
            f"cost is proportional to sibling count at {slope_origin * 1000:.3f} us per child "
            f"(R2 {r2_origin:.4f}), which is the signature of a per-child walk at fixed content"
        )

    return DoseFit(
        slope_through_origin = slope_origin,
        r2_through_origin = r2_origin,
        slope_free = slope_free,
        intercept_free = intercept_free,
        r2_free = r2_free,
        min_detectable_slope = min_slope,
        verdict = verdict,
        note = note,
        points = list(points),
        content_varied = False,
    )
