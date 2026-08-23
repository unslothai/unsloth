# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rendering: the rules about what may and may not be printed next to what.

This module is where the report's editorial policy lives, and it is enforced in code rather than
left to whoever writes the next summary:

  * THE HEADLINE FOR HUMANS IS THE ONSET RUNG. The largest thread size still usable on this
    machine. It is printed first, in plain words, because it is the only number that survives
    being carried to a different laptop. The aggregate score is printed under it, labelled as
    machine-local.
  * NO SINGLE FRAME SUMMARY MAY BE A HEADLINE. `time_in_jank_pct` and `jank_index` catch opposite
    failure shapes; quoting either alone is how a build with one three-second freeze gets called
    smooth. `render_frame_health()` always prints all three, and `assert_headline_pair()` fails
    a caller that tries to quote one.
  * CEILING SHIFTS ARE NEVER FOLDED INTO THE SCALAR. Moving the onset rung is a different kind of
    win from shaving 8% off every metric, and a single number that mixes them describes neither.
  * `excluded_cells` IS ALWAYS RENDERED, INCLUDING WHEN EMPTY. An empty block is the claim "we
    dropped nothing"; a missing block is a question nobody asked.
  * THE HARNESS-BIAS CELL IS PRINTED AT THE TOP AND NEVER SUBTRACTED. Knowing the instrument
    costs 4% is information; quietly removing 4% from every number is a second, unvalidated
    measurement pretending to be a correction.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..scoring.ab import AbResult
from ..scoring.anchors import METRIC_BY_KEY
from ..scoring.frames import FrameStats
from ..scoring.schema import Measure
from ..scoring.score import LadderScore, RungScore
from .payload import excluded_totals

#: The three frame numbers that must appear together or not at all.
HEADLINE_FRAME_METRICS = ("time_in_jank_pct", "jank_index", "max_frame_ms")


class HeadlinePolicyError(AssertionError):
    """Raised when a caller tries to quote one frame summary as the headline."""


def assert_headline_pair(keys: Iterable[str]) -> None:
    """Refuse a headline that quotes one frame summary without its counterpart.

    `time_in_jank_pct` answers "was it bad most of the time" and `jank_index` plus `max_frame_ms`
    answer "was it catastrophic once". A build can be terrible by either route while looking fine
    by the other, so a headline containing one and not the others is not a summary, it is a
    selection.
    """

    chosen = {k for k in keys if k in HEADLINE_FRAME_METRICS}
    if chosen and chosen != set(HEADLINE_FRAME_METRICS):
        missing = sorted(set(HEADLINE_FRAME_METRICS) - chosen)
        raise HeadlinePolicyError(
            "a frame headline must quote "
            + ", ".join(HEADLINE_FRAME_METRICS)
            + f"; missing {', '.join(missing)}"
        )


def render_frame_health(stats: FrameStats, *, indent: str = "") -> str:
    """All three frame headlines plus the histogram pointer. Never one of them alone."""

    lines = [
        f"{indent}frames            {stats.frames_total} over {stats.window_ms:.0f} ms "
        f"(budget {stats.budget_ms:.2f} ms, {stats.refresh_source})",
        f"{indent}time in jank      {stats.time_in_jank_pct.display()}   "
        "(wall time inside frames over 100 ms)",
        f"{indent}jank index        {stats.jank_index.display()}   "
        "(sum of squared over-budget ms per ms of window)",
        f"{indent}worst frame       {stats.max_frame_ms.display()}",
        f"{indent}p50 / p95 / p99   {stats.p50_frame_ms.display()} / "
        f"{stats.p95_frame_ms.display()} / {stats.p99_frame_ms.display()}",
    ]
    if stats.no_frames_recorded:
        lines.append(
            f"{indent}NOTE: the frame recorder produced no frames. That is not zero jank, it is "
            "no measurement: an unscheduled rAF loop reads exactly like a perfectly smooth page."
        )
    lines.append(f"{indent}histogram         {len(stats.histogram)} buckets, in the payload")
    return "\n".join(lines)


def render_headline(ladder: LadderScore) -> str:
    """The onset rung, in words, followed by the machine-local scalar."""

    lines: list[str] = []
    if ladder.onset_rung_tokens is None:
        lines.append("ONSET RUNG: none. This build is not usable at any measured thread size.")
    else:
        lines.append(
            f"ONSET RUNG: usable up to {ladder.onset_rung_tokens:,} tokens of thread content."
        )
    lines.append(f"  {ladder.onset_reason}")
    if ladder.non_monotonic:
        lines.append(
            "  WARNING: usability is not monotone in thread size on this run. A smaller rung "
            "failed while a larger one passed, which thread size cannot cause. Something else "
            "moved (thermal throttling, another process, an unstable machine) and the onset rung "
            "should not be quoted from this run alone."
        )
    lines.append("")
    lines.append(
        f"  aggregate score {ladder.aggregate:.1f}/100 (machine-local; does not travel between "
        "machines, unlike the onset rung above)"
    )
    lines.append(f"  weights {ladder.weights_id}   ladder {ladder.rung_ladder_id}")
    return "\n".join(lines)


def render_rung_table(ladder: LadderScore) -> str:
    header = f"{'tokens':>10}  {'score':>6}  {'weight':>6}  {'usable':>6}  status"
    rows = [header, "-" * len(header)]
    for rung, weight in zip(ladder.rungs, ladder.rung_weights):
        status = "complete"
        if not rung.complete:
            status = f"INCOMPLETE: {rung.incomplete_reason}"
        elif rung.zeroed_by:
            status = "zeroed by " + ", ".join(rung.zeroed_by)
        rows.append(
            f"{rung.tokens:>10,}  {rung.score:>6.1f}  {weight:>6.3f}  "
            f"{'yes' if rung.usable else 'no':>6}  {status}"
        )
    rows.append("")
    rows.append(
        "An incomplete rung scores 0 and keeps its weight. It is not dropped: dropping it is how "
        "a build that crashes at 500K outscores one that limps through."
    )
    rows.append(
        "Weights are the trapezoid widths on the log(tokens) axis, so two rungs close together "
        "in log space SHARE the weight of that region rather than each getting a full share. "
        "That is why the top rung is not the whole score, and why 500K and 1M together weigh "
        "about what one decade-spaced rung does."
    )
    return "\n".join(rows)


def render_rung_metrics(rung: RungScore, *, indent: str = "  ") -> str:
    lines = [f"{indent}{rung.tokens:,} tokens -- score {rung.score:.1f}"]
    if not rung.complete:
        # Printing six "not attempted" lines for a rung that never ran buries the one fact that
        # matters, which is why it did not run.
        lines.append(f"{indent}  INCOMPLETE: {rung.incomplete_reason}")
        lines.append(
            f"{indent}  scores 0 and keeps its weight; no metric was measured at this rung"
        )
        return "\n".join(lines)
    for metric in rung.metric_scores:
        anchor = METRIC_BY_KEY.get(metric.key)
        anchor_text = f"[{anchor.good:g} -> 100, {anchor.bad:g} -> 0]" if anchor is not None else ""
        if metric.scored:
            lines.append(
                f"{indent}  {metric.key:<20} {metric.measure.display():>28}   "
                f"score {float(metric.score):5.1f}  {anchor_text}"
            )
        else:
            lines.append(f"{indent}  {metric.key:<20} {'NOT SCORED':>28}   {metric.reason}")
    return "\n".join(lines)


def render_excluded(payload: Mapping[str, Any]) -> str:
    """Always printed. An empty block is a claim, an absent block is a hole."""

    cells = payload.get("excluded_cells")
    if cells is None:
        raise AssertionError("excluded_cells is mandatory and must not be null")
    lines = ["EXCLUDED CELLS"]
    if not cells:
        lines.append("  none. Every cell that was measured is in the numbers above.")
        return "\n".join(lines)
    totals = excluded_totals(payload)
    for reason, count in sorted(totals.items(), key = lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {count:>4}  {reason}")
    lines.append("")
    for cell in cells:
        detail = f" -- {cell['detail']}" if cell.get("detail") else ""
        lines.append(f"  {cell['cell_id']}: {cell['reason']} x{cell.get('count', 1)}{detail}")
    return "\n".join(lines)


def render_harness_bias(bias: Mapping[str, Any] | None) -> str:
    """The cost of the harness itself, printed at the top, never subtracted."""

    lines = ["HARNESS BIAS (dist-shipping vs dist-armed at control)"]
    if not bias:
        lines.append(
            "  not measured on this run. Every number below therefore includes an unknown "
            "amount of instrument cost."
        )
        return "\n".join(lines)
    for key, value in bias.items():
        rendered = value.display() if isinstance(value, Measure) else str(value)
        lines.append(f"  {key:<24} {rendered}")
    lines.append(
        "  This is NOT subtracted from anything below. Knowing the instrument costs 4% is "
        "information; removing 4% everywhere is a second unvalidated measurement."
    )
    return "\n".join(lines)


def render_ab_table(result: AbResult) -> str:
    """The A/B table, or the reason there is no A/B table.

    A void result prints its reason and no numbers at all. Printing a table with a warning above
    it does not work: the table gets screenshotted and the warning does not.
    """

    title = f"A/B: {result.label}"
    if result.is_null_control:
        title += "   (NULL-TREATMENT CONTROL, base vs base)"
    lines = [title, "=" * len(title)]

    if result.void:
        lines.append("")
        lines.append("VOID. No numbers are quotable from this comparison.")
        lines.append(f"  {result.void_reason}")
        # The paragraph under the reason explains the NULL CONTROL, and printing it under an
        # incomplete plan named the wrong cause for the void. A void has more than one cause; the
        # reason line carries which one, and only a null control gets the null-control paragraph.
        if result.is_null_control:
            lines.append(
                "  The null-treatment control measures whether this machine can currently tell "
                "two identical builds apart. When it cannot, nothing measured alongside it can "
                "be believed, so nothing is printed."
            )
        else:
            lines.append(
                "  A comparison is all of its pairs. A cell that did not complete takes its "
                "healthy partner out of the table with it, so the pairs that remain are a "
                "selection and no verdict is printed over them."
            )
        return "\n".join(lines)

    lines.append(f"noise floor {result.noise_floor_pct:.2f}%  ({result.noise_floor_source})")
    lines.append("")
    header = f"{'metric':<20} {'pairs':>5}  {'ratio':>7}  {'range':>17}  {'ci95':>17}  verdict"
    lines.append(header)
    lines.append("-" * len(header))
    for metric in result.metrics:
        if metric.ratio_geomean is None:
            lines.append(
                f"{metric.metric_key:<20} {metric.n_pairs:>5}  "
                f"{'--':>7}  {'--':>17}  {'--':>17}  no reading"
            )
            continue
        rng = f"{metric.ratio_min:.3f}-{metric.ratio_max:.3f}"
        ci = (
            f"{metric.ci_low:.3f}-{metric.ci_high:.3f}"
            if metric.ci_low is not None
            else "too few pairs"
        )
        # A BOUND IS NOT A MEASUREMENT. An arm under its instrument floor contributes the floor,
        # so the ratio understates the true magnitude and must not be quoted as a point estimate.
        # Marked in the ratio cell rather than footnoted, for the reason the void path gives: the
        # table gets screenshotted and the note does not.
        ratio_cell = (
            f">={metric.ratio_geomean:.3f}"
            if metric.bounded and metric.ratio_geomean >= 1.0
            else f"<={metric.ratio_geomean:.3f}"
            if metric.bounded
            else f"{metric.ratio_geomean:.3f}"
        )
        lines.append(
            f"{metric.metric_key:<20} {metric.n_pairs:>5}  {ratio_cell:>7}  "
            f"{rng:>17}  {ci:>17}  {metric.verdict}"
        )
    lines.append("")
    if result.headline_ratio is not None:
        direction = "faster" if result.headline_ratio < 1.0 else "slower"
        lines.append(
            f"headline ratio {result.headline_ratio:.3f} "
            f"({abs(1.0 - result.headline_ratio) * 100:.1f}% {direction}, weighted)"
        )
    lines.append(f"VERDICT: {result.verdict}")
    unresolved = [m for m in result.metrics if m.withheld]
    if unresolved:
        lines.append("")
        lines.append(
            "These metrics cleared the noise floor without a 95% CI that clears 1.0, either "
            "because the interval contains it or because there were too few pairs to compute "
            "one, so no direction is claimed for them. They are excluded from the headline "
            "ratio above, which would otherwise quote their size as a measured win:"
        )
        for metric in unresolved:
            why = (
                f"ci95 {metric.ci_low:.3f}-{metric.ci_high:.3f} contains 1.0"
                if metric.ci_low is not None and metric.ci_high is not None
                else f"no ci95 from {metric.n_pairs} usable pair(s)"
            )
            lines.append(f"  {metric.metric_key}: ratio {metric.ratio_geomean:.3f}, {why}")
    if result.regressions:
        lines.append("")
        lines.append(
            "FAIL. A per-metric regression beyond the noise floor is a fail regardless of the "
            "headline:"
        )
        for regression in result.regressions:
            lines.append(f"  {regression}")
    lines.append("")
    lines.append(
        "Ratios are paired within one session and interleaved. Ceiling shifts (the onset rung "
        "moving) are reported separately and are not in this table."
    )
    return "\n".join(lines)


def render_ceiling_shift(base: LadderScore, treatment: LadderScore) -> str:
    """Ceiling shift, reported on its own and never folded into the scalar."""

    lines = ["CEILING SHIFT (reported separately, never folded into the score)"]
    base_onset = base.onset_rung_tokens
    treat_onset = treatment.onset_rung_tokens
    lines.append(
        f"  base onset rung      {base_onset:,} tokens"
        if base_onset
        else "  base onset rung      none"
    )
    lines.append(
        f"  treatment onset rung {treat_onset:,} tokens"
        if treat_onset
        else "  treatment onset rung none"
    )
    if base_onset and treat_onset:
        if treat_onset > base_onset:
            lines.append(f"  the ceiling MOVED UP: {base_onset:,} -> {treat_onset:,}")
        elif treat_onset < base_onset:
            lines.append(f"  the ceiling MOVED DOWN: {base_onset:,} -> {treat_onset:,}")
        else:
            lines.append("  the ceiling did not move")
    return "\n".join(lines)


def render_summary(
    payload: Mapping[str, Any],
    ladder: LadderScore | None = None,
    *,
    harness_bias: Mapping[str, Any] | None = None,
    frame_stats_by_rung: Mapping[int, FrameStats] | None = None,
    extra_sections: Sequence[str] = (),
) -> str:
    """Assemble the human-facing summary in the order the policy above demands."""

    sections: list[str] = []
    sections.append("studiobench summary\n===================")

    if not payload.get("complete", False):
        sections.append(
            "RUN DID NOT FINISH. "
            + str(payload.get("incomplete_note", "no footer record was written."))
        )
        crashes = payload.get("crashes") or []
        for crash in crashes:
            sections.append(
                f"  crash: {crash.get('where', 'unknown')} "
                f"{crash.get('error_type', '')} {crash.get('error', '')}".rstrip()
            )

    sections.append(render_harness_bias(harness_bias))

    if ladder is not None:
        sections.append(render_headline(ladder))
        sections.append(render_rung_table(ladder))
        sections.append(
            "PER-RUNG METRICS\n" + "\n".join(render_rung_metrics(rung) for rung in ladder.rungs)
        )

    if frame_stats_by_rung:
        blocks = ["FRAME HEALTH"]
        for tokens in sorted(frame_stats_by_rung):
            blocks.append(f"  {tokens:,} tokens")
            blocks.append(render_frame_health(frame_stats_by_rung[tokens], indent = "    "))
        sections.append("\n".join(blocks))

    sections.extend(extra_sections)
    sections.append(render_excluded(payload))
    return "\n\n".join(sections) + "\n"
