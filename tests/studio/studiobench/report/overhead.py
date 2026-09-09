# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`overhead_growth_with_length`: the gate that disqualifies an instrument level.

An instrument that costs a constant 3 ms per window is an annoyance. An instrument whose cost
GROWS WITH THREAD LENGTH is a catastrophe, because the entire question this tool exists to answer
is whether cost grows with thread length. Such an instrument manufactures the symptom, and it
manufactures it in exactly the shape everyone is looking for, which is the worst possible way to
be wrong: the result confirms the hypothesis, the hypothesis is plausible, and nobody re-runs it.

So overhead is not just recorded, it is REGRESSED AGAINST THE TREATMENT. Every instrument at
level 1 or above declares `overhead_ms` per cell. If that number climbs across the rung ladder by
more than the tolerance, the LEVEL is disqualified: the numbers gathered at it may still be read
for structure (which function ran how many times), but no growth claim may rest on them.

Headline numbers come from level 0 for exactly this reason, and this gate is what keeps that from
being a promise rather than a check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..scoring.schema import Measure

#: Overhead may vary this much across the ladder before the level is disqualified. Generous: some
#: growth is unavoidable (more DOM means a longer snapshot), and a tight bound would disqualify
#: every level on every machine.
MAX_OVERHEAD_GROWTH_RATIO = 1.5

#: Growth below this many milliseconds is not worth acting on however large the ratio looks: a level
#: whose overhead goes from 0.01 ms to 0.05 ms has a ratio of 5 and does not matter.
MIN_ABSOLUTE_GROWTH_MS = 1.0


@dataclass
class OverheadVerdict:
    """One instrument level, judged on whether its cost tracks the treatment."""

    level: int
    instrument: str
    disqualified: bool
    reason: str
    growth_ratio: float | None
    growth_ms: Measure
    by_rung: dict[int, Measure] = field(default_factory = dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "instrument": self.instrument,
            "disqualified": bool(self.disqualified),
            "reason": self.reason,
            "growth_ratio": self.growth_ratio,
            "growth_ms": self.growth_ms.to_json(),
            "by_rung": {str(k): v.to_json() for k, v in sorted(self.by_rung.items())},
        }

    def render(self) -> str:
        lines = [
            f"INSTRUMENT OVERHEAD: {self.instrument} at level {self.level}",
        ]
        for rung, measure in sorted(self.by_rung.items()):
            lines.append(f"  {rung:>10,} tokens   {measure.display()}")
        lines.append(f"  growth        {self.growth_ms.display()}")
        if self.growth_ratio is not None:
            lines.append(f"  growth ratio  {self.growth_ratio:.2f}x across the ladder")
        lines.append(
            f"  VERDICT: {'DISQUALIFIED' if self.disqualified else 'usable'} -- {self.reason}"
        )
        return "\n".join(lines)


def overhead_growth_gate(
    instrument: str,
    level: int,
    by_rung: Mapping[int, Measure],
    *,
    max_ratio: float = MAX_OVERHEAD_GROWTH_RATIO,
    min_absolute_growth_ms: float = MIN_ABSOLUTE_GROWTH_MS,
) -> OverheadVerdict:
    """Judge one instrument at one level across the rung ladder.

    Disqualification needs BOTH a ratio above `max_ratio` and an absolute growth above
    `min_absolute_growth_ms`. Either alone is a false positive generator: a ratio on a tiny number
    is noise, and an absolute growth on a huge constant overhead is not correlation with the
    treatment, it is a big instrument.
    """

    readings = {int(rung): m for rung, m in by_rung.items() if m.has_reading}
    verdict = OverheadVerdict(
        level = level,
        instrument = instrument,
        disqualified = False,
        reason = "",
        growth_ratio = None,
        growth_ms = Measure.not_attempted("ms", "overhead was not measured across rungs"),
        by_rung = {int(rung): m for rung, m in by_rung.items()},
    )

    if level <= 0:
        verdict.reason = (
            "level 0 carries the headline numbers and declares no overhead; there is nothing to "
            "disqualify it against"
        )
        return verdict

    if len(readings) < 2:
        verdict.reason = (
            f"only {len(readings)} rung(s) reported an overhead reading, so the gate could not be "
            "evaluated. An unevaluated gate is not a passed gate"
        )
        return verdict

    smallest_rung = min(readings)
    largest_rung = max(readings)
    low = float(readings[smallest_rung].value)
    high = float(readings[largest_rung].value)
    growth = high - low
    verdict.growth_ms = Measure.read(growth, "ms")
    verdict.growth_ratio = (high / low) if low > 0 else None

    if growth <= 0:
        verdict.reason = (
            "overhead does not grow with thread length, so it cannot be manufacturing a slope"
        )
        return verdict

    ratio_bad = verdict.growth_ratio is not None and verdict.growth_ratio > max_ratio
    absolute_bad = growth > min_absolute_growth_ms
    if ratio_bad and absolute_bad:
        verdict.disqualified = True
        verdict.reason = (
            f"overhead grows {verdict.growth_ratio:.2f}x ({growth:.2f} ms) between the "
            f"{smallest_rung:,} and {largest_rung:,} token rungs. This instrument's own cost "
            "tracks the treatment, so it manufactures the very slope the run is looking for. No "
            "growth claim may rest on level "
            f"{level}"
        )
        return verdict

    if ratio_bad:
        verdict.reason = (
            f"the ratio is {verdict.growth_ratio:.2f}x but the absolute growth is only "
            f"{growth:.3f} ms, below the {min_absolute_growth_ms:g} ms that would matter"
        )
    else:
        verdict.reason = (
            f"overhead grows {growth:.2f} ms across the ladder, within the {max_ratio:g}x "
            "tolerance"
        )
    return verdict


def render_overhead_section(verdicts: list[OverheadVerdict]) -> str:
    if not verdicts:
        return (
            "INSTRUMENT OVERHEAD\n"
            "  no instrument above level 0 ran, so no overhead gate was evaluated"
        )
    blocks = [verdict.render() for verdict in verdicts]
    disqualified = [v for v in verdicts if v.disqualified]
    if disqualified:
        blocks.append(
            "DISQUALIFIED LEVELS: "
            + ", ".join(f"{v.instrument}@L{v.level}" for v in disqualified)
            + "\n  Their structural output (counts, names) is still usable. Their timings are "
            "not, and nothing about growth may be quoted from them."
        )
    return "\n\n".join(blocks)


def log_growth_slope(by_rung: Mapping[int, Measure]) -> float | None:
    """Least-squares slope of overhead against log(tokens), for the payload.

    Reported alongside the ratio because the ratio only looks at the two ends. An instrument that
    is flat at both ends and spikes in the middle has a ratio of 1.0 and a visible problem.
    """

    points = [
        (math.log(float(rung)), float(m.value))
        for rung, m in by_rung.items()
        if m.has_reading and float(rung) > 0
    ]
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
