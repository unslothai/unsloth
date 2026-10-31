# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integrity gates abort the run; each guards a defect that once gave a plausible but wrong table."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..scoring.schema import ExcludedCell, Measure

STALL_TOLERANCE_MS = 20.0
INJECTED_STALL_MS = 120.0
# Same order as the real per-chunk streaming cost the metric must resolve.
INJECTED_STREAM_COST_MS = 3.0
# Below 1.0 since app work shares the chain; under-recovery is the failure that matters.
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
        passed = True,  # longtask unsupported is an engine fact, not a failure
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
    """A clock spanning materially less than wall time missed part of the window, which is unmeasured."""

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


# Busy wait, not sleep: a sleep is recovered through a different scheduler path.

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

# Burns main-thread time per SSE chunk, queued as a microtask so wrapper order does not matter.
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
    """A metric blind to an injected cost cannot be trusted on real ones; report the recovery fraction."""
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
    """Driver time is the midpoint around evaluate; a post-call reading charges the page's stall to it."""

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
    """Judges already-collected readings without driving the browser, so failing cases can be unit-
    tested."""

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
