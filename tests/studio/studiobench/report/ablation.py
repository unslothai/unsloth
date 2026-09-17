# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rendering the ablation batch, in an order that cannot be read out of sequence.

CALIBRATION IS PRINTED FIRST AND CAN END THE SECTION. If the batch is not quotable, the only
thing printed is why. There is no table with a warning above it, because the table is what gets
screenshotted into a thread and the warning is what gets left behind.

ARMS ARE NEVER QUOTED ALONE. Only adjacent differences on a route appear, which is enforced
upstream by `LadderRoute.quote_arm` raising; here it shows up as the absence of a per-arm cost
column. What IS printed per arm is its VERDICT, because a voided arm and an arm that did not fire
are findings about the experiment that a reader needs even though their numbers are not usable.

THE TWO ROUTES ARE PRINTED SEPARATELY AND THEN COMPARED. Nothing anywhere averages them. Where
they disagree, the disagreement is the interaction term and gets its own block with its own
sentence about what it means.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..arms.batch import BatchResult
from ..arms.knobs import render_decision_table
from ..arms.ladder import MECHANISM_FIX, RouteResult
from ..arms.manifest import ArmStatus


def render_route(result: RouteResult) -> str:
    lines = [
        f"ROUTE {result.route.route_id}: {result.route.name}",
        "",
    ]
    header = f"  {'step':<28} {'mechanism removed':<44} difference"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for step_result in result.steps:
        mechanisms = ", ".join(step_result.step.mechanisms)
        if step_result.step.fused:
            mechanisms += "  [FUSED]"
        lines.append(f"  {step_result.step.label:<28} {mechanisms:<44} {step_result.quote()}")
    lines.append("")
    if result.total is not None:
        lines.append(f"  top minus floor      {result.total.display()}")
    if result.sum_of_steps is not None:
        lines.append(f"  sum of the steps     {result.sum_of_steps.display()}")
    lines.append(f"  residual             {result.residual_ms:.9f} ms")
    lines.append(f"  {result.identity_note}")
    fused = [s for s in result.steps if s.step.fused]
    if fused:
        lines.append("")
        for step_result in fused:
            lines.append(f"  FUSED STEP {step_result.step.label}: {step_result.step.fused_reason}")
    return "\n".join(lines)


def render_interactions(result: BatchResult) -> str:
    lines = ["INTERACTION TERMS (two routes, same floor, compared mechanism by mechanism)"]
    if not result.interactions:
        lines.append(
            "  none computed. Two routes are needed and at least one of them did not produce a "
            "complete set of differences"
        )
        return "\n".join(lines)
    header = f"  {'mechanism':<32} {'route A':>12} {'route B':>12} {'gap':>12}"
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for term in result.interactions:
        gap = f"{term.disagreement_ms:+.3f}" if term.disagreement_ms is not None else "n/a"
        lines.append(
            f"  {term.mechanism:<32} {term.value_a.display():>12} "
            f"{term.value_b.display():>12} {gap:>12}"
        )
    lines.append("")
    for term in result.interactions:
        lines.append(f"  {term.mechanism}: {term.note}")
    lines.append("")
    lines.append(
        "  Nothing here is averaged. Two routes disagreeing about the same mechanism is a "
        "finding about additivity, and a mean of the two describes neither route."
    )
    return "\n".join(lines)


def render_arm_verdicts(result: BatchResult) -> str:
    """Every arm's status, including the ones whose numbers are not usable."""

    lines = ["ARM VERDICTS (no arm's absolute cost is quoted; only adjacent differences are)"]
    if not result.outcomes:
        lines.append("  no arms were run")
        return "\n".join(lines)
    for key in sorted(result.outcomes):
        outcome = result.outcomes[key]
        lines.append(f"  {key:<20} {outcome.status.value:<12} {outcome.reason}")
    voided = result.voided_arms()
    not_run = result.not_run_arms()
    if voided:
        lines.append("")
        lines.append(
            "  VOIDED arms produced a number and it is not being used. They changed the rendered "
            "output, so their difference is not attributable to the mechanism."
        )
    if not_run:
        lines.append("")
        lines.append(
            "  NOT RUN arms are not evidence of no effect. Their potency counter never moved, "
            "which means the treatment was not applied at all."
        )
    return "\n".join(lines)


def render_batch(result: BatchResult, *, include_decision_table: bool = True) -> str:
    """The whole ablation section for one rung, in the only order it may be read."""

    sections: list[str] = []
    title = f"ABLATION at {result.rung_tokens:,} tokens"
    sections.append(title + "\n" + "=" * len(title))
    sections.append(result.calibration.render())

    if not result.quotable:
        sections.append(
            "NO ABLATION NUMBERS ARE PRINTED FOR THIS BATCH.\n"
            f"  {result.calibration.reason}\n"
            "  The arms ran and produced numbers. They are in the payload and they are not "
            "quoted here, because a batch that cannot tell two identical builds apart cannot "
            "tell two different ones apart either."
        )
        sections.append(render_arm_verdicts(result))
        return "\n\n".join(sections) + "\n"

    if result.armpack is not None:
        sections.append(result.armpack.render())

    if include_decision_table:
        sections.append(render_decision_table().rstrip())

    for route in result.routes:
        sections.append(render_route(route))

    sections.append(render_interactions(result))
    sections.append(render_arm_verdicts(result))

    if result.dose is not None:
        sections.append(result.dose.render())
    if result.recovery is not None:
        sections.append(result.recovery.render())

    if result.plan_notes:
        sections.append("NOTES\n" + "\n".join(f"  {note}" for note in result.plan_notes))

    return "\n\n".join(sections) + "\n"


def render_fix_implications(result: BatchResult, *, top_n: int = 3) -> str:
    """Which fix the measured steps point at, ranked by the size of the step.

    This is the decision table applied to the numbers rather than printed as a table: the largest
    adjacent difference names the layer the cost lives in, and the mechanism's declared fix is
    printed next to it. A step that is only a bound is labelled as such and cannot be ranked
    above a point estimate of the same size.
    """

    by_mechanism: dict[str, list[tuple[str, float, bool]]] = {}
    for route in result.routes:
        for step_result in route.steps:
            if not (step_result.quotable and step_result.difference.has_reading):
                continue
            for mechanism in step_result.step.mechanisms:
                by_mechanism.setdefault(mechanism, []).append(
                    (
                        route.route.route_id,
                        float(step_result.difference.value),
                        step_result.bound_only,
                    )
                )
    if not by_mechanism:
        return "FIX IMPLICATIONS\n  none: no step on either route produced a usable difference"

    ranked = sorted(
        by_mechanism.items(),
        key = lambda item: (-max(value for _, value, _ in item[1]), item[0]),
    )
    lines = [
        "FIX IMPLICATIONS (ranked by the largest measured step, which is a RANGE when the two "
        "routes disagree)"
    ]
    for mechanism, measurements in ranked[:top_n]:
        values = [value for _, value, _ in measurements]
        bound_only = any(bound for _, _, bound in measurements)
        prefix = "<= " if bound_only else ""
        if len(values) > 1 and max(values) - min(values) > 1e-9:
            # Printing only the largest would let a reader take the most flattering route as the
            # answer. The spread IS the interaction term and it belongs next to the number.
            spread = ", ".join(
                f"{route_id} {value:.3f}" for route_id, value, _ in sorted(measurements)
            )
            lines.append(f"  {prefix}{min(values):.3f} to {max(values):.3f} ms  {mechanism}")
            lines.append(f"      routes disagree: {spread}")
        else:
            lines.append(f"  {prefix}{values[0]:.3f} ms  {mechanism}")
        lines.append(f"      {MECHANISM_FIX[mechanism]}")
    return "\n".join(lines)
