# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A frame is named only when its exact count equals an independently measured structural quantity."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Sequence

NAMING = "naming"
NEAR_MISS = "near_miss"
UNEXPLAINED = "unexplained_hot_frame"
NOT_MEASURED = "not_measured"

# Each ratio has a specific mechanical meaning, so naming it hands over a hypothesis.
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
    """source records where the number came from, so an oracle derived from the same trace is visible."""

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
    """One clone per sibling per render; React 19.2 has no cloneChildFibers; use createWorkInProgress."""
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
    """M2's count is characters rescanned, so compare it against a character counter, not a call count."""
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
    if abs(measured - predicted) <= 2:
        return f"off by {measured - predicted:+d}, within one structural unit"
    return None


def check(
    frame: str, exact_call_count: int | None, quantities: Sequence[StructuralQuantity]
) -> OracleVerdict:
    """No tolerance: a naming needs exact integer equality, or a tolerant oracle matches eventually."""
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
    """Predicts the next rung before it runs, so a naming that cannot predict forward is caught."""
    return quantity_fn(next_structural_input).value


# M2 and M3: oracles over page-side counters emitted by `instruments/layoutcost.js`. Unlike M1's
# exact integer match, M2 is a regime test: SSE deltas are uneven, so it is weaker evidence.

REGIME_QUADRATIC = "cumulative_reparse_quadratic"
REGIME_LINEAR = "incremental_parse_linear"
REGIME_UNDECIDED = "undecided"

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
    """Chars rescanned if the whole buffer is re-parsed per delta; exact only for uniform delta streams."""
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
    """Refuses to name a regime when linear and quadratic predictions lie within refusal_band."""
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
    """Exact match expected: stabilize() forces a layout per callback by reading scrollHeight."""
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
    """Scales with thread size, not reply length: compare against the layout-time growth exponent."""
    return StructuralQuantity(
        name = "forced_layouts_x_thread_nodes",
        value = forced_layouts * thread_nodes,
        source = source,
        components = {"forced_layouts": forced_layouts, "thread_nodes": thread_nodes},
    )


def evaluate_page_counters(counters: dict[str, Any]) -> dict[str, Any]:
    """Missing groups are skipped with a reason, so 'did not fire' differs from 'nobody counted'."""
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
