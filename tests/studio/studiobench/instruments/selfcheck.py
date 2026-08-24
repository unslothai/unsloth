# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integrity gates. These run BEFORE any cell counts, and a failure ABORTS instead of reporting.

Every gate here exists because the corresponding defect was live, produced a plausible table, and
was believed for a while.

  120 ms STALL, SEEN WITHIN +/-20 ms
      A frame recorder that cannot see a stall the harness injected itself cannot see a stall the
      app produced. The failure this catches is a recorder that is running but attributing its
      samples to the wrong window, which reads as a clean run rather than as a broken one.

  400 ms INPUT DELAY, MOVING KEYSTROKE p95 BY AT LEAST 350 ms
      The input path is measured through `page.keyboard` so `latencyInfo` is real. A harness that
      writes the textarea value directly and dispatches a synthetic `input` event measures the
      DOM setter and nothing else, and that measurement does not move when the app gets slower.

  A HEAVY SCENE AND A TRIVIAL SCENE DIFFERING BY MORE THAN 20%
      If the instrument cannot separate a deliberately heavy page from a deliberately trivial
      one, it is BLIND, and every "no significant difference" it produces is meaningless. This is
      the gate that would have caught a fixture rendering a fifth of the content it claimed.

  longtask READ FROM supportedEntryTypes, NEVER FROM WHETHER observe() THROWS
      `PerformanceObserver.observe({type:"longtask"})` does not throw on WebKit or Firefox. It is
      accepted and then never fires. A try/catch gate therefore reports both engines as supported
      and then reports zero long tasks, which is a fabricated number, not a missing one.

  THE _clock_pair CONTROL RATIO WITHIN 10%
      The page's wall clock against the driver's monotonic clock. It must be flat across a ladder
      by construction, so when it moves, the measurement moved and not the page. The driver half
      is the MIDPOINT of two readings taken either side of the round trip: a single post-evaluate
      reading charges the whole blocked main thread to the driver and produced 2.5% of pure skew
      at the top of a ladder, in a quantity whose entire job is to be flat.

  THREE-CLOCK AGREEMENT, >20% DISAGREEMENT EXCLUDES THE WINDOW
      rAF callbacks, CDP `Page.startScreencast` presented frames, and a 1 ms timer, each
      independently measuring the same window's elapsed time. This gate exists for one specific
      reason: rAF STOPS BEING SCHEDULED when the compositor decides nothing is visible. A window
      in that state reports no dropped frames, because it reports no frames, and "no dropped
      frames" is how a completely unmeasured window looks in every table. A window whose clocks
      disagree is marked `clock_disagreement` and EXCLUDED from scoring rather than believed.

The pure evaluators below take numbers and return verdicts, so they are unit-testable without a
browser. The driver functions underneath them are the thin part that talks to Playwright.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..scoring.schema import ExcludedCell, Measure

STALL_TOLERANCE_MS = 20.0
INJECTED_STALL_MS = 120.0
#: Milliseconds of main-thread time burned per SSE chunk when the streaming-cost injection is
#: armed. Sized against what it has to stand in for: preprocessLaTeX over a 96,000 character reply
#: is projected at 246 ms across a whole stream, so a per-chunk figure in the low single digits is
#: the same ORDER as the effect the metric is meant to resolve, not a boulder it cannot miss.
INJECTED_STREAM_COST_MS = 3.0
#: The share of injected cost `stream_cost` must read back. Not 1.0: the metric measures the task
#: chain from the chunk to the event loop draining, and a burn queued as a microtask lands inside
#: that chain but the chain also contains the app's own work, so recovery is bounded below by
#: attribution and above by nothing. Under-recovery is the failure that matters, because a metric
#: that recovers a quarter of a known cost will under-report an unknown one by the same factor.
MIN_STREAM_COST_RECOVERY = 0.70
INJECTED_INPUT_DELAY_MS = 400.0
MIN_INPUT_P95_SHIFT_MS = 350.0
MIN_SCENE_CONTRAST_PCT = 20.0
CLOCK_PAIR_TOLERANCE_PCT = 10.0
TRI_CLOCK_TOLERANCE_PCT = 20.0


class SelfCheckFailure(AssertionError):
    """Raised when an integrity gate fails. The run aborts; no numbers are produced."""


@dataclass
class Gate:
    """One integrity gate: what it measured, what it required, and whether it held."""

    name: str
    passed: bool
    measured: Measure
    expected: str
    detail: str
    fatal: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "measured": self.measured.to_json(),
            "expected": self.expected,
            "detail": self.detail,
            "fatal": bool(self.fatal),
        }

    def render(self) -> str:
        mark = "pass" if self.passed else ("FAIL" if self.fatal else "warn")
        return (
            f"  [{mark}] {self.name:<28} {self.measured.display():>24}   "
            f"expected {self.expected}\n         {self.detail}"
        )


@dataclass
class SelfCheckReport:
    gates: list[Gate] = field(default_factory = list)

    @property
    def ok(self) -> bool:
        return all(gate.passed for gate in self.gates if gate.fatal)

    @property
    def failures(self) -> list[Gate]:
        return [gate for gate in self.gates if gate.fatal and not gate.passed]

    def raise_if_failed(self) -> None:
        """Abort the run. A failed gate means the numbers would be about the instrument."""

        if self.ok:
            return
        detail = "\n".join(f"  {gate.name}: {gate.detail}" for gate in self.failures)
        raise SelfCheckFailure(
            "integrity gates failed; no cells will be measured and nothing will be reported.\n"
            + detail
            + "\nThis is deliberate. A run that reports numbers from a blind instrument is worse "
            "than a run that reports nothing, because the numbers get quoted and the blindness "
            "does not."
        )

    def render(self) -> str:
        lines = ["INTEGRITY GATES (run before any cell counts; failure aborts)"]
        lines.extend(gate.render() for gate in self.gates)
        lines.append("")
        lines.append(f"  VERDICT: {'all gates held' if self.ok else 'ABORT'}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "gates": [gate.to_json() for gate in self.gates]}


# ---------------------------------------------------------------------------------------
# pure evaluators
# ---------------------------------------------------------------------------------------


def evaluate_stall_gate(
    observed_ms: float | None,
    *,
    injected_ms: float = INJECTED_STALL_MS,
    tolerance_ms: float = STALL_TOLERANCE_MS,
) -> Gate:
    """The frame recorder must see an injected stall, at roughly the size it was injected."""

    if observed_ms is None:
        return Gate(
            name = "injected_stall_seen",
            passed = False,
            measured = Measure.failed("ms", "the frame recorder reported no stall at all"),
            expected = f"{injected_ms:g} +/- {tolerance_ms:g} ms",
            detail = (
                "a stall this harness injected itself was not observed. The recorder is not "
                "watching the window it thinks it is"
            ),
        )
    error = abs(float(observed_ms) - injected_ms)
    return Gate(
        name = "injected_stall_seen",
        passed = error <= tolerance_ms,
        measured = Measure.read(float(observed_ms), "ms"),
        expected = f"{injected_ms:g} +/- {tolerance_ms:g} ms",
        detail = (
            f"observed longest frame is {error:.1f} ms from the injected stall"
            + ("" if error <= tolerance_ms else "; the recorder is mis-attributing samples")
        ),
    )


def evaluate_input_delay_gate(
    baseline_p95_ms: float | None,
    delayed_p95_ms: float | None,
    *,
    injected_ms: float = INJECTED_INPUT_DELAY_MS,
    min_shift_ms: float = MIN_INPUT_P95_SHIFT_MS,
) -> Gate:
    """A 400 ms input delay must move keystroke p95 by at least 350 ms."""

    if baseline_p95_ms is None or delayed_p95_ms is None:
        return Gate(
            name = "input_delay_seen",
            passed = False,
            measured = Measure.failed("ms", "one of the two keystroke measurements is missing"),
            expected = f">= {min_shift_ms:g} ms shift for a {injected_ms:g} ms delay",
            detail = (
                "without both readings there is no shift to check, and an input path that is not "
                "verified is an input path that may be measuring the DOM setter"
            ),
        )
    shift = float(delayed_p95_ms) - float(baseline_p95_ms)
    return Gate(
        name = "input_delay_seen",
        passed = shift >= min_shift_ms,
        measured = Measure.read(shift, "ms"),
        expected = f">= {min_shift_ms:g} ms shift for a {injected_ms:g} ms delay",
        detail = (
            f"keystroke p95 moved {shift:.1f} ms"
            + (
                ""
                if shift >= min_shift_ms
                else "; the keystroke path is not going through the real input pipeline, so it "
                "will not move when the app gets slower either"
            )
        ),
    )


def evaluate_scene_contrast_gate(
    heavy_ms: float | None,
    trivial_ms: float | None,
    *,
    min_pct: float = MIN_SCENE_CONTRAST_PCT,
) -> Gate:
    """A deliberately heavy scene and a trivial one must differ, or the instrument is blind."""

    if heavy_ms is None or trivial_ms is None or trivial_ms <= 0:
        return Gate(
            name = "scene_contrast",
            passed = False,
            measured = Measure.failed("%", "one of the two contrast scenes produced no reading"),
            expected = f"> {min_pct:g}% difference",
            detail = "a contrast check with a missing side proves nothing",
        )
    contrast = (float(heavy_ms) - float(trivial_ms)) / float(trivial_ms) * 100.0
    return Gate(
        name = "scene_contrast",
        passed = contrast > min_pct,
        measured = Measure.read(contrast, "%"),
        expected = f"> {min_pct:g}% difference",
        detail = (
            f"the heavy scene is {contrast:.1f}% slower than the trivial one"
            if contrast > min_pct
            else (
                f"only {contrast:.1f}% between a deliberately heavy scene and a trivial one. The "
                "instrument is BLIND, and every null result it produces is uninformative"
            )
        ),
    )


def evaluate_longtask_support(supported_entry_types: Sequence[str] | None) -> Gate:
    """Read support from `supportedEntryTypes`. Never from whether `observe()` throws."""

    if supported_entry_types is None:
        return Gate(
            name = "longtask_support",
            passed = False,
            measured = Measure.failed("bool", "supportedEntryTypes was not readable"),
            expected = "supportedEntryTypes readable",
            detail = (
                "the support list itself could not be read, so support cannot be established by "
                "the only method that works"
            ),
            fatal = False,
        )
    supported = "longtask" in list(supported_entry_types)
    return Gate(
        name = "longtask_support",
        passed = True,  # not supporting longtask is a fact about the engine, not a failure
        measured = Measure.read(1.0 if supported else 0.0, "bool"),
        expected = "read from supportedEntryTypes",
        detail = (
            "longtask entries are available on this engine"
            if supported
            else (
                "this engine does not support longtask. Long-task numbers will read as NOT "
                "ATTEMPTED rather than as zero. observe() would have accepted the type and then "
                "never fired, which is why support is read from the list and not from a try/catch"
            )
        ),
        fatal = False,
    )


def evaluate_clock_pair(
    page_ms: float | None,
    driver_ms: float | None,
    *,
    tolerance_pct: float = CLOCK_PAIR_TOLERANCE_PCT,
) -> Gate:
    """The control ratio between the page's wall clock and the driver's monotonic clock."""

    if page_ms is None or driver_ms is None or driver_ms <= 0:
        return Gate(
            name = "clock_pair_control",
            passed = False,
            measured = Measure.failed("ratio", "a clock pair reading is missing"),
            expected = f"1.0 +/- {tolerance_pct:g}%",
            detail = "without the control ratio there is nothing holding the ladder flat",
        )
    ratio = float(page_ms) / float(driver_ms)
    drift_pct = abs(ratio - 1.0) * 100.0
    return Gate(
        name = "clock_pair_control",
        passed = drift_pct <= tolerance_pct,
        measured = Measure.read(ratio, "ratio"),
        expected = f"1.0 +/- {tolerance_pct:g}%",
        detail = (
            f"the control ratio is {drift_pct:.2f}% from unity"
            + (
                ""
                if drift_pct <= tolerance_pct
                else ". This quantity is flat by construction, so a move means the MEASUREMENT "
                "moved, not the page"
            )
        ),
    )


@dataclass
class TriClockVerdict:
    """Three clocks over one window, and whether the window survives."""

    agreed: bool
    wall_ms: float
    spans: dict[str, float | None]
    worst_clock: str | None
    worst_disagreement_pct: float | None
    reason: str

    def excluded_cell(self, cell_id: str) -> ExcludedCell | None:
        if self.agreed:
            return None
        return ExcludedCell(
            cell_id = cell_id,
            reason = "clock_disagreement",
            count = 1,
            detail = self.reason,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "agreed": bool(self.agreed),
            "wall_ms": float(self.wall_ms),
            "spans": dict(self.spans),
            "worst_clock": self.worst_clock,
            "worst_disagreement_pct": self.worst_disagreement_pct,
            "reason": self.reason,
        }


def evaluate_tri_clock(
    *,
    wall_ms: float,
    raf_span_ms: float | None,
    screencast_span_ms: float | None,
    timer_span_ms: float | None,
    raf_frames: int | None = None,
    tolerance_pct: float = TRI_CLOCK_TOLERANCE_PCT,
) -> TriClockVerdict:
    """Compare each clock's own measurement of the window against wall time.

    Each of the three clocks independently spans the window: the rAF loop by summing its frame
    gaps, the screencast by the interval between its first and last presented frame, the 1 ms
    timer by summing its tick gaps. All three should equal the wall duration, and a clock that
    covers materially less than the window did not observe part of it.

    THE CASE THIS IS FOR. When the compositor decides nothing is visible it stops scheduling rAF
    callbacks entirely. The frame recorder then reports very few frames and no dropped ones, and
    a report reads that as a smooth window. Here it shows up as an rAF span far short of wall
    time while the timer clock covers the whole window, which is the correct conclusion: the
    window was not measured.
    """

    spans: dict[str, float | None] = {
        "raf": None if raf_span_ms is None else float(raf_span_ms),
        "screencast": None if screencast_span_ms is None else float(screencast_span_ms),
        "timer": None if timer_span_ms is None else float(timer_span_ms),
    }

    if wall_ms <= 0:
        return TriClockVerdict(
            agreed = False,
            wall_ms = float(wall_ms),
            spans = spans,
            worst_clock = None,
            worst_disagreement_pct = None,
            reason = "the window has no positive wall duration, so no clock can be checked",
        )

    if raf_frames is not None and raf_frames <= 0:
        return TriClockVerdict(
            agreed = False,
            wall_ms = float(wall_ms),
            spans = spans,
            worst_clock = "raf",
            worst_disagreement_pct = 100.0,
            reason = (
                "the rAF loop produced no frames at all. That is not a smooth window, it is an "
                "unmeasured one: rAF stops being scheduled when the compositor decides nothing "
                "is visible"
            ),
        )

    disagreements: dict[str, float] = {}
    for name, span in spans.items():
        if span is None:
            continue
        disagreements[name] = abs(span - float(wall_ms)) / float(wall_ms) * 100.0

    if len(disagreements) < 2:
        return TriClockVerdict(
            agreed = False,
            wall_ms = float(wall_ms),
            spans = spans,
            worst_clock = None,
            worst_disagreement_pct = None,
            reason = (
                f"only {len(disagreements)} of three clocks reported. Agreement between fewer "
                "than two clocks is not agreement"
            ),
        )

    worst_clock = max(disagreements, key = lambda k: disagreements[k])
    worst = disagreements[worst_clock]
    agreed = worst <= tolerance_pct
    return TriClockVerdict(
        agreed = agreed,
        wall_ms = float(wall_ms),
        spans = spans,
        worst_clock = worst_clock,
        worst_disagreement_pct = worst,
        reason = (
            f"all reporting clocks are within {tolerance_pct:g}% of wall time "
            f"(worst: {worst_clock} at {worst:.1f}%)"
            if agreed
            else (
                f"the {worst_clock} clock is {worst:.1f}% away from wall time, beyond the "
                f"{tolerance_pct:g}% tolerance. The window is marked clock_disagreement and "
                "excluded from scoring"
            )
        ),
    )


# ---------------------------------------------------------------------------------------
# browser-side snippets and drivers
# ---------------------------------------------------------------------------------------

#: Burns a known amount of main-thread time once, on the next frame. A busy wait, not a sleep:
#: a sleep is recovered through a different scheduler path than a blocked main thread, which is
#: the thing being calibrated.
STALL_INJECT_JS = """
(stallMs) => {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      const started = performance.now();
      while (performance.now() - started < stallMs) { /* spin */ }
      requestAnimationFrame(() => resolve(performance.now() - started));
    });
  });
}
"""

#: Burns a known amount of main-thread time PER SSE CHUNK, inside the task chain that chunk
#: starts. This is the streaming analogue of STALL_INJECT_JS: a stall injected once tests whether
#: the frame recorder is watching the right window, and this tests whether the streaming-cost
#: accumulator integrates a cost spread thinly across a whole stream, which is the shape of every
#: effect it was built for and the shape a single-action metric cannot see.
#:
#: THE BURN IS QUEUED AS A MICROTASK, not run inline in the decode wrapper, and that is what makes
#: the check independent of wrapper order. `add_init_script` runs scripts in the order they were
#: added, and the instrument's own TextDecoder wrapper is installed by `Instrument.attach` after
#: the scripts assembled in __main__. Whichever wrapper ends up outermost, a microtask queued
#: during the decode runs after the current task's synchronous code and before the MessageChannel
#: macrotask that closes the measured chain -- so the burn is inside the chain either way. Burning
#: inline would land it before the accumulator timestamps the chunk under one ordering and after
#: it under the other, and the check would silently measure nothing.
STREAM_COST_INJECT_JS = """
(() => {
  if (window.__sbStreamCostInject) { return; }
  const burnMs = %(burn_ms)f;
  const S = { burnMs: burnMs, chunks: 0, burnedMs: 0 };
  window.__sbStreamCostInject = S;
  const nativeDecode = TextDecoder.prototype.decode;
  TextDecoder.prototype.decode = function (input, options) {
    const out = nativeDecode.call(this, input, options);
    if (typeof out === "string" && out.length > 0 && out.length <= 65536
        && out.indexOf("data:") >= 0) {
      S.chunks += 1;
      queueMicrotask(() => {
        const started = performance.now();
        while (performance.now() - started < burnMs) { /* spin */ }
        S.burnedMs += performance.now() - started;
      });
    }
    return out;
  };
})();
"""


def stream_cost_injection_init_script(burn_ms: float = INJECTED_STREAM_COST_MS) -> str:
    return STREAM_COST_INJECT_JS % {"burn_ms": float(burn_ms)}


def evaluate_stream_cost_recovery_gate(
    base_ms_per_kchar: float | None,
    injected_ms_per_kchar: float | None,
    injected_total_ms: float | None,
    streamed_chars: int | None,
    *,
    min_recovery: float = MIN_STREAM_COST_RECOVERY,
) -> Gate:
    """`stream_cost` must read back a cost this harness injected into the stream itself.

    A metric that cannot see a known cost cannot see an unknown one, and this is the only check
    that separates "the change did nothing" from "the metric is not watching". The RECOVERY
    FRACTION is the output that matters and it is reported whether or not the gate passes: a
    metric recovering 40% of what was injected is not broken, but every number it produces is
    four tenths of the truth and a reader has to be told the multiplier.
    """
    missing = [
        name
        for name, value in (
            ("the base rate", base_ms_per_kchar),
            ("the injected rate", injected_ms_per_kchar),
            ("the injected total", injected_total_ms),
            ("the streamed character count", streamed_chars),
        )
        if value is None
    ]
    if missing or not streamed_chars or not injected_total_ms:
        return Gate(
            name = "injected_stream_cost_recovered",
            passed = False,
            measured = Measure.failed(
                "fraction", "missing: " + ", ".join(missing or ["a non-zero denominator"])
            ),
            expected = f">= {min_recovery:.0%} of the injected cost",
            detail = (
                "the streaming-cost metric produced no reading on one side, so nothing can be "
                "said about whether it can see an injected cost"
            ),
        )
    observed_ms = (injected_ms_per_kchar - base_ms_per_kchar) * streamed_chars / 1000.0
    recovery = observed_ms / injected_total_ms
    return Gate(
        name = "injected_stream_cost_recovered",
        passed = recovery >= min_recovery,
        measured = Measure.read(round(recovery, 3), "fraction"),
        expected = f">= {min_recovery:.0%} of the injected cost",
        detail = (
            f"injected {injected_total_ms:.0f} ms across {streamed_chars:,} streamed characters, "
            f"read back {observed_ms:.0f} ms ({recovery:.0%})"
            + ("" if recovery >= min_recovery else "; the accumulator is under-attributing")
        ),
    )


#: Adds a fixed delay to every keydown before the app sees it, by installing a capturing listener
#: that blocks. Installed pre-boot so it sits ahead of every app handler.
INPUT_DELAY_INIT_JS = """
(() => {
  if (window.__sbInputDelay) { return; }
  const delayMs = %(delay_ms)f;
  const S = { delayMs: delayMs, events: 0, burnedMs: 0, armed: false };
  window.__sbInputDelay = S;
  const block = () => {
    if (!S.armed) { return; }
    const started = performance.now();
    while (performance.now() - started < delayMs) { /* spin */ }
    S.events += 1;
    S.burnedMs += performance.now() - started;
  };
  window.addEventListener("keydown", block, true);
  S.arm = () => { S.armed = true; };
  S.disarm = () => { S.armed = false; };
})();
"""

#: Reads `supportedEntryTypes` directly. Never a try/catch around observe().
LONGTASK_SUPPORT_JS = """
() => {
  try {
    return Array.from(PerformanceObserver.supportedEntryTypes || []);
  } catch (e) {
    return null;
  }
}
"""


def input_delay_init_script(delay_ms: float = INJECTED_INPUT_DELAY_MS) -> str:
    return INPUT_DELAY_INIT_JS % {"delay_ms": float(delay_ms)}


def clock_pair(page: Any) -> tuple[float, float]:
    """The page's wall clock and the driver's monotonic clock, read around one round trip.

    The driver half is the MIDPOINT of readings taken either side of the evaluate. The page can
    only answer while its main thread is free, so a single reading taken after the call charges
    the entire blocked main thread to the driver's clock. That mistake produced 2.5% of pure
    round-trip skew at the top of a ladder, in the one quantity whose job is to be flat.
    """

    before = time.monotonic()
    page_ms = page.evaluate("() => Date.now()")
    after = time.monotonic()
    return float(page_ms), (before + after) / 2.0


def read_longtask_support(page: Any) -> list[str] | None:
    try:
        return page.evaluate(LONGTASK_SUPPORT_JS)
    except Exception:
        return None


def run_gates(
    *,
    stall_observed_ms: float | None,
    keystroke_p95_baseline_ms: float | None,
    keystroke_p95_delayed_ms: float | None,
    heavy_scene_ms: float | None,
    trivial_scene_ms: float | None,
    supported_entry_types: Sequence[str] | None,
    clock_pair_page_ms: float | None,
    clock_pair_driver_ms: float | None,
    injected_stall_ms: float = INJECTED_STALL_MS,
    injected_input_delay_ms: float = INJECTED_INPUT_DELAY_MS,
) -> SelfCheckReport:
    """Assemble every gate from already-collected readings.

    Pure: the caller does the driving, this does the judging. That split is what lets the whole
    gate set be unit-tested, including the cases where it must FAIL, which is the half that never
    gets exercised when the checks live inline in the driver.
    """

    report = SelfCheckReport()
    report.gates.append(evaluate_stall_gate(stall_observed_ms, injected_ms = injected_stall_ms))
    report.gates.append(
        evaluate_input_delay_gate(
            keystroke_p95_baseline_ms,
            keystroke_p95_delayed_ms,
            injected_ms = injected_input_delay_ms,
        )
    )
    report.gates.append(evaluate_scene_contrast_gate(heavy_scene_ms, trivial_scene_ms))
    report.gates.append(evaluate_longtask_support(supported_entry_types))
    report.gates.append(evaluate_clock_pair(clock_pair_page_ms, clock_pair_driver_ms))
    return report


def guard(report: SelfCheckReport, *, on_abort: Callable[[str], None] | None = None) -> None:
    """Abort the run if any fatal gate failed. Called before the first cell is measured."""

    if report.ok:
        return
    message = report.render()
    if on_abort is not None:
        on_abort(message)
    report.raise_if_failed()
