# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The line between a NAMING and an `unexplained_hot_frame`.

A hot frame with a bridged name is not yet an explanation. `Zk` resolving to
`cloneChildFibers` tells you what the code is called; it does not tell you why
it ran 4,110 times, and without that the next step is guesswork.

A NAMING requires the frame's EXACT invocation count, from precise coverage, to
equal a STRUCTURAL quantity measured independently: blocks x renders,
subscribers x notifies, siblings x updates. Independently means the structural
quantity was counted from the DOM or from the app's own instrumentation, not
derived from the same trace. When those two integers agree, the mechanism is
identified, the fix is implied, and the prediction is falsifiable at the next
rung.

When nothing matches, the frame is emitted as `unexplained_hot_frame` carrying
its bridged name, its exponent and its exact count. That is an honest partial
result and it is enormously more than a residual: someone can read it, recognise
the name, and know where to look. What it must never do is get quietly rounded
into whichever oracle is closest.

NEAR MISSES ARE REPORTED, NOT ACCEPTED. An exact 2x is the signature of React's
StrictMode double invoke; an exact ratio of (n+1)/n is the signature of counting
a root along with its children. Those are diagnoses in their own right, so the
integer ratio is printed. It never promotes a match to a naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Sequence

NAMING = "naming"
NEAR_MISS = "near_miss"
UNEXPLAINED = "unexplained_hot_frame"
NOT_MEASURED = "not_measured"

# Integer ratios worth naming when a count is off by a clean factor. Each has a specific mechanical
# meaning, so reporting the ratio hands over a hypothesis rather than a mystery.
_KNOWN_RATIOS: dict[Fraction, str] = {
    Fraction(
        2, 1
    ): "exactly 2x predicted: React StrictMode double invoke, or the render ran twice per commit",
    Fraction(
        1, 2
    ): "exactly half predicted: the structural count is double counting, or only one of two passes is instrumented",
    Fraction(3, 1): "exactly 3x predicted: three passes over the same structure",
}


@dataclass(frozen = True)
class StructuralQuantity:
    """A count measured from the app, independently of the profile.

    `source` must say where the number came from. An oracle whose structural
    side was derived from the same trace as the frame count proves nothing and
    the field exists to make that visible in the report.
    """

    name: str
    value: int
    source: str
    components: dict[str, int] = field(default_factory = dict)

    def describe(self) -> str:
        if self.components:
            parts = " x ".join(f"{v} {k}" for k, v in self.components.items())
            return f"{self.value} ({parts}, from {self.source})"
        return f"{self.value} (from {self.source})"


@dataclass
class OracleVerdict:
    frame: str
    verdict: str
    exact_call_count: int | None
    quantity: StructuralQuantity | None
    detail: str
    ratio: str | None = None

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "frame": self.frame,
            "verdict": self.verdict,
            "detail": self.detail,
        }
        if self.exact_call_count is not None:
            row["exact_call_count"] = self.exact_call_count
        if self.quantity is not None:
            row["structural_quantity"] = self.quantity.describe()
            row["structural_name"] = self.quantity.name
        if self.ratio:
            row["integer_ratio"] = self.ratio
        return row

    @property
    def is_naming(self) -> bool:
        return self.verdict == NAMING


def blocks_times_renders(blocks: int, renders: int, source: str) -> StructuralQuantity:
    """M1's prediction: one work-in-progress fiber cloned per sibling per render.

    `memo` stops a child RENDERING but not React REACHING it.
    `bailoutOnAlreadyFinishedWork` returns null only when `childLanes` is clear,
    so an update anywhere in the subtree still walks every sibling and clones
    one fiber each. With a flat, unvirtualised message list that is
    blocks x renders clones per chunk, and it is invisible to `<Profiler>`
    because it is fiber bookkeeping and not component render.

    NAME THE RIGHT FUNCTION. `cloneChildFibers` DOES NOT EXIST IN REACT 19.2.
    Grepping `react-dom@19.2.4`'s development bundle finds zero occurrences of
    it; it was inlined at some point before 19. The functions that do exist, and
    that a symbol bridge built against a real 19.2.4 profiling bundle does
    resolve, are `createWorkInProgress` (10 occurrences) and
    `bailoutOnAlreadyFinishedWork` (10). Point this oracle at those. An oracle
    aimed at a function that does not exist returns `not_measured` forever and
    reads as "M1 is not happening", which is the wrong conclusion drawn from a
    stale function name.
    """
    return StructuralQuantity(
        name = "blocks_x_renders",
        value = blocks * renders,
        source = source,
        components = {"blocks": blocks, "renders": renders},
    )


def subscribers_times_notifies(subscribers: int, notifies: int, source: str) -> StructuralQuantity:
    """An external-store subscription fanout: every store notify hits every subscriber."""
    return StructuralQuantity(
        name = "subscribers_x_notifies",
        value = subscribers * notifies,
        source = source,
        components = {"subscribers": subscribers, "notifies": notifies},
    )


def chars_times_deltas(chars: int, deltas: int, source: str) -> StructuralQuantity:
    """M2's prediction: the cumulative buffer is re-parsed once per delta.

    The count that matters for the re-parse is characters rescanned, so this is
    a quantity to compare against a character counter rather than against a call
    count; it is here so the M2 oracle has the same shape as the others.
    """
    return StructuralQuantity(
        name = "chars_x_deltas",
        value = chars * deltas,
        source = source,
        components = {"chars": chars, "deltas": deltas},
    )


def mutations_times_thread_nodes(mutations: int, nodes: int, source: str) -> StructuralQuantity:
    """M3's prediction: one observer callback per mutation, each reading layout over the whole thread."""
    return StructuralQuantity(
        name = "mutations_x_thread_nodes",
        value = mutations * nodes,
        source = source,
        components = {"mutations": mutations, "thread_nodes": nodes},
    )


def _ratio_note(measured: int, predicted: int) -> str | None:
    if predicted <= 0 or measured <= 0:
        return None
    fr = Fraction(measured, predicted)
    known = _KNOWN_RATIOS.get(fr)
    if known:
        return known
    if fr.denominator == 1 and 1 < fr.numerator <= 16:
        return f"exactly {fr.numerator}x predicted"
    if fr.numerator == 1 and 1 < fr.denominator <= 16:
        return f"exactly 1/{fr.denominator} of predicted"
    # Off by one structural unit, e.g. counting a root alongside its children.
    if abs(measured - predicted) <= 2:
        return f"off by {measured - predicted:+d}, within one structural unit"
    return None


def check(
    frame: str, exact_call_count: int | None, quantities: Sequence[StructuralQuantity]
) -> OracleVerdict:
    """Compare one frame's exact count against every candidate structural quantity.

    EXACT equality is required for a naming. No tolerance, no rounding. The
    whole value of an exact count is that it is exact; a count oracle with a 5%
    tolerance is a correlation with extra steps, and it will match something
    eventually.
    """
    if exact_call_count is None:
        return OracleVerdict(
            frame = frame,
            verdict = NOT_MEASURED,
            exact_call_count = None,
            quantity = None,
            detail = (
                "no precise-coverage count for this frame. Without an exact integer "
                "there is nothing to match against a structural quantity, so this "
                "frame cannot be named however hot it is."
            ),
        )

    for q in quantities:
        if q.value == exact_call_count:
            return OracleVerdict(
                frame = frame,
                verdict = NAMING,
                exact_call_count = exact_call_count,
                quantity = q,
                detail = f"ran exactly {exact_call_count} times = {q.describe()}",
            )

    best: tuple[StructuralQuantity, str] | None = None
    for q in quantities:
        note = _ratio_note(exact_call_count, q.value)
        if note is not None:
            best = (q, note)
            break

    if best is not None:
        q, note = best
        return OracleVerdict(
            frame = frame,
            verdict = NEAR_MISS,
            exact_call_count = exact_call_count,
            quantity = q,
            ratio = note,
            detail = (
                f"ran exactly {exact_call_count} times against a predicted {q.describe()}; "
                f"{note}. A near miss is a hypothesis about the discrepancy, not a naming."
            ),
        )

    return OracleVerdict(
        frame = frame,
        verdict = UNEXPLAINED,
        exact_call_count = exact_call_count,
        quantity = None,
        detail = (
            f"ran exactly {exact_call_count} times, matching none of "
            f"{[q.describe() for q in quantities] or 'any supplied quantity'}. "
            "Reported with its bridged name and exponent so it can be looked up; "
            "this is a partial result, not a residual."
        ),
    )


def check_all(
    frames: Sequence[tuple[str, int | None]], quantities: Sequence[StructuralQuantity]
) -> dict[str, Any]:
    verdicts = [check(name, count, quantities) for name, count in frames]
    namings = [v for v in verdicts if v.is_naming]
    return {
        "namings": [v.as_row() for v in namings],
        "near_misses": [v.as_row() for v in verdicts if v.verdict == NEAR_MISS],
        "unexplained_hot_frames": [v.as_row() for v in verdicts if v.verdict == UNEXPLAINED],
        "not_measured": [v.as_row() for v in verdicts if v.verdict == NOT_MEASURED],
        "named_at_least_one_frame": bool(namings),
    }


def predicted_next_rung(
    quantity_fn: Callable[[int], StructuralQuantity], next_structural_input: int
) -> int:
    """The count a naming PREDICTS at the next rung.

    A naming that cannot predict forward is a coincidence that has not been
    caught yet. Emitting the prediction before the next rung runs is what makes
    it falsifiable.
    """
    return quantity_fn(next_structural_input).value


# M2 and M3: oracle shapes over PAGE-SIDE counters
# M1 is settled with a call count from precise coverage, because the mechanism lives inside
# react-dom where invocations can be counted. M2 and M3 do not work that way: their cost is in how
# much text is rescanned and how much layout is forced, and those must be counted IN THE PAGE.
# Layer 3 owns `instruments/layoutcost.js` and already counts `scrollHeight` reads, `scrollTop`
# writes and MutationObserver callbacks. This section does NOT duplicate it: it fixes the KEY
# NAMES that file should emit and supplies the oracles that consume them, so counting and
# interpretation are owned separately.
# ONE HONEST DIFFERENCE: the M1 oracle is an EXACT INTEGER MATCH that either holds or does not,
# while the M2 oracle is a REGIME TEST, since SSE deltas do not arrive in equal sizes, so the
# quadratic prediction is exact only for a uniform stream and must be compared as a ratio with a
# refusal band. That is weaker evidence and is labelled as such everywhere it is emitted.

REGIME_QUADRATIC = "cumulative_reparse_quadratic"
REGIME_LINEAR = "incremental_parse_linear"
REGIME_UNDECIDED = "undecided"

# The counters Layer 3's page-side instrument should emit, per streamed reply. Every one is an
# integer; none is a duration.
PAGE_COUNTER_CONTRACT: dict[str, dict[str, str]] = {
    "m2_reparse": {
        "parse_calls": "invocations of parseAssistantContent during the reply",
        "chars_rescanned": "SUM of the input length over those invocations; the whole point",
        "deltas_received": "SSE delta events applied to the cumulative buffer",
        "final_content_chars": "length of the cumulative buffer when the reply completed",
        "think_tracker_calls": "invocations of createThinkTagTracker over the same window",
    },
    "m3_forced_layout": {
        "observer_callbacks": "MutationObserver callback invocations on the viewport subtree",
        "forced_layouts": "reads of scrollHeight/offsetHeight/getBoundingClientRect that flushed layout",
        "scroll_writes": "scrollTo / scrollTop writes issued by the autoscroll path",
        "stabilizer_writes": "writes of the --aui-scroll-stabilizer custom property",
        "thread_nodes": "DOM node count inside the scroll container at the end of the window",
    },
}


def cumulative_reparse_chars(
    final_content_chars: int, deltas: int, source: str
) -> StructuralQuantity:
    """M2 under the CUMULATIVE hypothesis: the whole buffer is re-parsed per delta.

    If `parseAssistantContent(cumulativeText)` runs on every delta, the i-th
    delta rescans roughly `i * final/deltas` characters, so the total is

        final * (deltas + 1) / 2

    which is quadratic in reply length at fixed delta size. Exact only for a
    uniform stream; see the module note about this being a regime test.
    """
    value = int(final_content_chars * (deltas + 1) / 2) if deltas > 0 else 0
    return StructuralQuantity(
        name = "cumulative_reparse_chars",
        value = value,
        source = source,
        components = {"final_chars": final_content_chars, "deltas": deltas},
    )


def incremental_parse_chars(final_content_chars: int, source: str) -> StructuralQuantity:
    """M2 under the NULL hypothesis: each delta is parsed once, so total = final length."""
    return StructuralQuantity(
        name = "incremental_parse_chars",
        value = int(final_content_chars),
        source = source,
        components = {"final_chars": final_content_chars},
    )


def reparse_regime(
    chars_rescanned: int,
    final_content_chars: int,
    deltas: int,
    *,
    refusal_band: float = 2.0,
) -> dict[str, Any]:
    """Which regime is the measured rescan count in?

    Compares the measurement against the linear and quadratic predictions in log
    space and refuses to call it when the two predictions are within
    `refusal_band` of each other, which happens on short replies where a handful
    of deltas makes the quadratic prediction indistinguishable from the linear
    one. Refusing on a short reply is correct: the mechanism is real or not
    regardless, but that reply cannot show it.
    """
    quad = cumulative_reparse_chars(final_content_chars, deltas, "prediction").value
    lin = incremental_parse_chars(final_content_chars, "prediction").value
    out: dict[str, Any] = {
        "chars_rescanned": int(chars_rescanned),
        "linear_prediction": lin,
        "quadratic_prediction": quad,
        "evidence_class": "regime_test",
        "evidence_note": (
            "a regime verdict is WEAKER evidence than a naming: it is a ratio "
            "comparison with a refusal band, not an exact integer match"
        ),
    }
    if lin <= 0 or quad <= 0 or chars_rescanned <= 0:
        out["regime"] = REGIME_UNDECIDED
        out["reason"] = "a prediction or the measurement was non-positive; nothing to compare"
        return out
    if quad < lin * refusal_band:
        out["regime"] = REGIME_UNDECIDED
        out["reason"] = (
            f"the two predictions are only {quad / lin:.2f}x apart, below the {refusal_band}x "
            "refusal band. This reply is too short to distinguish the regimes; use a longer rung."
        )
        return out
    # Closer in log space wins; the ratio is reported either way so a reader can see how decisive the call was.
    r_lin = chars_rescanned / lin
    r_quad = chars_rescanned / quad
    out["ratio_to_linear"] = round(r_lin, 3)
    out["ratio_to_quadratic"] = round(r_quad, 3)
    import math

    if abs(math.log(r_quad)) < abs(math.log(r_lin)):
        out["regime"] = REGIME_QUADRATIC
        out["reason"] = (
            f"rescanned {chars_rescanned} chars, {r_quad:.2f}x the cumulative-reparse "
            f"prediction and {r_lin:.1f}x the incremental one"
        )
    else:
        out["regime"] = REGIME_LINEAR
        out["reason"] = (
            f"rescanned {chars_rescanned} chars, {r_lin:.2f}x the incremental prediction; "
            "the cumulative re-parse is NOT firing on this path"
        )
    return out


def forced_layout_per_callback(
    observer_callbacks: int, forced_layouts: int, source: str
) -> OracleVerdict:
    """M3's exact oracle: one forced layout per observer callback.

    This one IS an exact integer match, unlike M2. `stabilize()` reads
    `scrollHeight` synchronously inside the MutationObserver callback, so if the
    mechanism is live the two counters are equal. A forced-layout count BELOW
    the callback count means the read is being skipped or batched on some
    callbacks, which is a different and much cheaper story; ABOVE means
    something else is also forcing layout and M3 is not the whole cost.
    """
    return check(
        "autoscroll MutationObserver forced layout",
        forced_layouts,
        [
            StructuralQuantity(
                name = "observer_callbacks",
                value = observer_callbacks,
                source = source,
                components = {"observer_callbacks": observer_callbacks},
            )
        ],
    )


def forced_layout_cost_quantity(
    forced_layouts: int, thread_nodes: int, source: str
) -> StructuralQuantity:
    """M3's cost shape: each forced layout is proportional to the whole thread.

    The mechanism is invisible to a React Profiler, to markdown timing and to a
    DOM census, and it grows with THREAD SIZE rather than with reply length,
    which is what distinguishes it from M2. Compare against the growth exponent
    of layout time, not against a call count.
    """
    return StructuralQuantity(
        name = "forced_layouts_x_thread_nodes",
        value = forced_layouts * thread_nodes,
        source = source,
        components = {"forced_layouts": forced_layouts, "thread_nodes": thread_nodes},
    )


def evaluate_page_counters(counters: dict[str, Any]) -> dict[str, Any]:
    """Run the M2 and M3 oracles over one window's page-side counter block.

    `counters` is the dict Layer 3's page instrument emits, keyed as in
    `PAGE_COUNTER_CONTRACT`. Missing groups produce an explicit skip with a
    reason, never a silent absence, because "M3 did not fire" and "nobody
    counted" must not look the same in a report.
    """
    out: dict[str, Any] = {}

    m2 = counters.get("m2_reparse")
    if not isinstance(m2, dict):
        out["m2"] = {"skipped": True, "reason": "no m2_reparse counter block was emitted"}
    else:
        missing = [
            k for k in ("chars_rescanned", "final_content_chars", "deltas_received") if k not in m2
        ]
        if missing:
            out["m2"] = {"skipped": True, "reason": f"m2_reparse is missing {missing}"}
        else:
            out["m2"] = reparse_regime(
                int(m2["chars_rescanned"]),
                int(m2["final_content_chars"]),
                int(m2["deltas_received"]),
            )

    m3 = counters.get("m3_forced_layout")
    if not isinstance(m3, dict):
        out["m3"] = {"skipped": True, "reason": "no m3_forced_layout counter block was emitted"}
    else:
        missing = [k for k in ("observer_callbacks", "forced_layouts") if k not in m3]
        if missing:
            out["m3"] = {"skipped": True, "reason": f"m3_forced_layout is missing {missing}"}
        else:
            verdict = forced_layout_per_callback(
                int(m3["observer_callbacks"]),
                int(m3["forced_layouts"]),
                source = "page counters",
            )
            block = verdict.as_row()
            if "thread_nodes" in m3:
                block["cost_quantity"] = forced_layout_cost_quantity(
                    int(m3["forced_layouts"]),
                    int(m3["thread_nodes"]),
                    source = "page counters",
                ).describe()
            out["m3"] = block
    return out
