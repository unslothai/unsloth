# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming phase is detected from SSE traffic, not window kind, which tags idle gaps as stream."""

from __future__ import annotations

from typing import Optional

from ..runtime.types import Cell, Window
from . import register_instrument
from .pagejs import _PageInstrument


@register_instrument(name = "stream_cost", level = 0)
def _stream_cost():
    return StreamCostInstrument()


class StreamCostInstrument(_PageInstrument):
    """Per-window streaming cost, and the streamed characters to divide it by."""

    name = "stream_cost"
    level = 0
    script_name = "streamcost.js"

    def __init__(self) -> None:
        super().__init__()
        self._chars_open: Optional[int] = None
        self._integrity_open: dict = {}
        self._overhead_ms = 0.0

    def start_cell(self, cell: Cell) -> None:
        # Per cell: one instance serves the session, and a carried total fakes growth across rungs.
        super().start_cell(cell)
        self._overhead_ms = 0.0
        self._chars_open = None

    def open(self, window: Window) -> None:
        # Drain first in case the previous close did not run.
        self._eval("() => window.__sb.streamcost && window.__sb.streamcost.reset()")
        self._chars_open = self._eval(
            "() => window.__sb.streamcost && window.__sb.streamcost.replyChars()"
        )
        # Sample integrity at both ends: a parse failure or unterminated frame makes the delta unscoreable.
        self._integrity_open = (
            self._eval("() => window.__sb.streamcost && window.__sb.streamcost.wireIntegrity()")
            or {}
        )

    def close(self, window: Window) -> Optional[dict]:
        if self.unavailable:
            return {"unavailable": self.unavailable, "stream_cost_attempted": False}
        elapsed_ms = window.duration_ms
        out = self._eval("(ms) => window.__sb.streamcost.read(ms)", elapsed_ms)
        if out is None:
            return {
                "unavailable": self.unavailable or "the page did not answer",
                "stream_cost_attempted": False,
            }
        chars_close = self._eval("() => window.__sb.streamcost.replyChars()")
        out["stream_cost_attempted"] = True
        out["reply_chars_open"] = self._chars_open
        out["reply_chars_close"] = chars_close
        out["reply_chars_source"] = "sse_wire"
        # Ask about the decoder pending at the open; buffer and flush must belong to the same decoder.
        integrity = (
            self._eval(
                "(id) => window.__sb.streamcost.wireIntegrity(id)",
                self._integrity_open.get("decoder_id"),
            )
            or {}
        )
        failures = (integrity.get("failures") or 0) - (self._integrity_open.get("failures") or 0)
        residual = integrity.get("pending_chars") or 0
        # Refuse a window that opened on a buffered frame only if that frame completed inside it.
        # An aborted response's half frame never completes and must not refuse the window.
        pending_at_open = self._integrity_open.get("pending_chars") or 0
        carried = (integrity.get("carried_flushes") or 0) - (
            self._integrity_open.get("carried_flushes") or 0
        )
        out["wire_parse_failures_in_window"] = failures
        out["wire_pending_chars_at_close"] = residual
        out["wire_pending_chars_at_open"] = pending_at_open
        out["wire_carried_frames_counted_in_window"] = carried
        if failures > 0 or residual > 0 or (pending_at_open > 0 and carried > 0):
            out["reply_chars_scoreable"] = False
            out["reply_chars_unscoreable_reason"] = (
                f"{failures} SSE frame(s) failed to parse inside this window, "
                f"{pending_at_open} character(s) of an unterminated frame were already buffered "
                f"when it opened and {carried} of those frames were completed and counted inside "
                f"it, and {residual} character(s) were still buffered at its close, "
                "so the wire character count over this window is short by an unknown amount at "
                "one end or carries a frame that began before the other, and any cost-per-"
                "character derived from it would be wrong"
            )
        else:
            out["reply_chars_scoreable"] = True
        out["reply_chars_source_note"] = (
            "counted from the SSE deltas in the decode hook, O(the chunk). Previously read from "
            "the DOM with a querySelectorAll, which is O(the document) and therefore cheaper on "
            "an arm that mounts fewer nodes"
        )

        if self._chars_open is None or chars_close is None:
            out["reply_chars_delta"] = None
            out["reply_chars_delta_reason"] = (
                "the reply's length was not read at one end of this window, either because no "
                "assistant message was on screen or because the stream had been finished longer "
                "than the idle gap when the window opened"
            )
        elif chars_close < self._chars_open:
            # Unreachable since the counter is monotonic; going backwards means the instrument was reset.
            out["reply_chars_delta"] = None
            out["reply_chars_delta_reason"] = (
                f"the wire character counter went backwards, from {self._chars_open} to "
                f"{chars_close}. It is cumulative and monotonic, so this means the page was "
                "reloaded or the instrument was reinstalled inside the window"
            )
        else:
            out["reply_chars_delta"] = chars_close - self._chars_open
            out["reply_chars_delta_attempted"] = True

        self._overhead_ms += float(out.get("overhead_ms") or 0.0)

        # read() resets overhead_ms, so drain the close-side scan cost here or it is never counted.
        tail = self._eval("() => window.__sb.streamcost.read(0)")
        close_scan_ms = float(tail.get("overhead_ms") or 0.0) if isinstance(tail, dict) else 0.0
        self._overhead_ms += close_scan_ms
        out["close_scan_overhead_ms"] = round(close_scan_ms, 2)

        self._chars_open = None
        self._integrity_open = {}
        return out

    def end_cell(self, cell: Cell) -> Optional[dict]:
        # The one DOM read, between cells so no window or arm is charged for it.
        wire = self._eval("() => window.__sb.streamcost.wireStats()") or {}
        dom_chars = self._eval("() => window.__sb.streamcost.replyCharsDom(true)")
        return {
            "overhead_ms": round(self._overhead_ms, 2),
            "overhead_attempted": True,
            "overhead_note": (
                "measured inside the decode hook and at the window boundaries, not estimated. "
                "O(1) per SSE chunk and O(the chunk) for the wire character count; nothing here "
                "is proportional to the document, the rung or the arm. The reply-length read USED "
                "to be a querySelectorAll -- O(the whole DOM), 38.8 ms per cell at 10K and "
                "289.6 ms at 100K -- and was justified on the grounds that it cancels in a paired "
                "ratio. It does not cancel against an arm that changes the size of the document, "
                "so it was removed from the paired path entirely"
            ),
            "wire": wire,
            "last_message_chars_in_dom": dom_chars,
            "last_message_chars_note": (
                "read once, after the film, purely as a cross-check against the wire count. Never "
                "inside a measured window and never used as a denominator"
            ),
        }
