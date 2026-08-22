# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registers `stream_cost`: the streaming-phase cost accumulator.

WHAT THIS ADDS THAT THE HARNESS DID NOT HAVE. Not an integral -- `time_in_jank_pct`, `jank_index`
and `max_frame_ms` already integrate, and `_frame_measures` already pools the streaming windows
into them. What it adds is SEPARATION and a DENOMINATOR:

  SEPARATION. The three frame metrics collapse one film into one number per cell, and the action
  windows dominate it. Measured on a 100K null control, `action:reasoning_toggle` contributes
  2,865 ms of blocked time at 99.3% busy with a 1,866 ms worst frame, and `action:select_all_copy`
  3,017 ms at 97.7% with a 2,102 ms worst frame, while the streaming stretch beside them runs at
  3.6% busy with a 100 ms worst frame. A change to the streaming path moves the second and is
  scored against the first.

  The window KIND cannot make that separation, and this is the trap worth naming because the name
  actively misleads. `SceneRunner._gap_window` opens EVERY inter-slot gap as `kind = "stream"`. On
  the standard film that is eighteen windows called `stream:gapN` of which only the first four
  contain any streaming, plus `stream:drain`, which on a measured 100K cell was 7 ms long because
  the stream had finished forty seconds earlier. Filtering on `kind == "stream"` selects mostly
  post-stream idle. The phase is therefore detected from the SSE traffic itself.

  A DENOMINATOR. Cost per streamed character, not per cell. This is what makes two rungs
  comparable on cost per unit of work, which is the claim the whole effort is testing: that a
  thread twice as long costs more to stream one character into. Worth being honest about what it
  buys and where: WITHIN one rung the denominator is very nearly a constant, because the pacer is
  deficit-scheduled and the tail is pinned, and measured across twelve null-control pairs at 100K
  the streamed character count varies by 0.0%. It earns its keep ACROSS rungs, where it is the
  only way to compare 10K with 100K at all.

WHY IT IS LEVEL 0. Its per-event work is O(1) and its per-window work is O(the reply being
streamed), never O(the thread). The two hooks are a `TextDecoder.prototype.decode` wrapper that
runs about fourteen times a second at field cadence, and a 1 ms timer of exactly the kind
frames.js already runs and documents as costing nothing at about 150 ticks a second. Nothing here
is proportional to the rung, which is what `overhead_growth_with_length` exists to catch, and
`overhead_ms` is measured inside the hooks rather than asserted so the gate has something real to
read.
"""

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
        self._overhead_ms = 0.0

    def start_cell(self, cell: Cell) -> None:
        # PER CELL, like every other instrument that declares overhead (heap.py, tracing.py,
        # coverage.py all zero theirs here). One instrument instance serves the whole session, so
        # an accumulator carried across cells reports cell k's overhead as the sum of cells 1..k.
        # The rung ladder is run in ascending order, so that sum climbs with the rung and
        # `overhead_growth_with_length` -- the gate whose entire job is to catch an instrument
        # whose cost tracks the treatment -- would read manufactured growth off a flat instrument.
        super().start_cell(cell)
        self._overhead_ms = 0.0
        self._chars_open = None

    def open(self, window: Window) -> None:
        # Drain first, so the window starts from zero even if the previous close did not run
        # (an instrument that raised is disabled for the rest of the cell, and the accumulator
        # would otherwise carry that cell's remaining traffic into this window).
        self._eval("() => window.__sb.streamcost && window.__sb.streamcost.reset()")
        self._chars_open = self._eval(
            "() => window.__sb.streamcost && window.__sb.streamcost.replyChars()"
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
        # Forced when the window carried SSE traffic: the open-read is skipped on windows long
        # past the stream, and a window that unexpectedly streamed must not be scored from a
        # half-taken pair.
        saw_traffic = bool(out.get("streaming_observed"))
        chars_close = self._eval("(f) => window.__sb.streamcost.replyChars(f)", saw_traffic)
        out["stream_cost_attempted"] = True
        out["reply_chars_open"] = self._chars_open
        out["reply_chars_close"] = chars_close

        if self._chars_open is None or chars_close is None:
            out["reply_chars_delta"] = None
            out["reply_chars_delta_reason"] = (
                "the reply's length was not read at one end of this window, either because no "
                "assistant message was on screen or because the stream had been finished longer "
                "than the idle gap when the window opened"
            )
        elif chars_close < self._chars_open:
            # THE REPLY WAS REPLACED, NOT EXTENDED. `send_turn` starts a new assistant message and
            # `thread_reopen` rebuilds the thread, so the last message can be SHORTER at the close
            # of a window than it was at the open. That is not negative streaming; it is a
            # different message. A signed delta here would subtract from the denominator and
            # inflate the cost per character without anything in the payload saying why.
            out["reply_chars_delta"] = None
            out["reply_chars_delta_reason"] = (
                f"the last assistant message shrank from {self._chars_open} to {chars_close} "
                "characters, so it is a different message and its growth is not measurable here"
            )
        else:
            out["reply_chars_delta"] = chars_close - self._chars_open
            out["reply_chars_delta_attempted"] = True

        self._overhead_ms += float(out.get("overhead_ms") or 0.0)

        # THE CLOSE-SIDE SCAN IS NOT FREE AND WAS NOT COUNTED. `read()` snapshots `overhead_ms`
        # into its result and then calls `reset()`, so the `replyChars()` above -- which at close
        # is FORCED whenever the window carried traffic, and is the `querySelectorAll` that is
        # O(the whole DOM) rather than O(the reply) -- accumulated into a page-side total that
        # nothing ever read: the next `open()` begins by resetting it. Exactly half of every
        # window's boundary scans were therefore missing from the one number whose job is to make
        # the level 0 claim checkable from the payload rather than from a docstring, and the half
        # that was missing is the rung-dependent half. Drained here, after the scan, rather than
        # by reordering the pair: the close read needs `force`, and `force` is
        # `streaming_observed`, which only `read()` can answer. Nothing else is lost by the second
        # drain -- every other accumulator it clears was already snapshotted above, and `open()`
        # clears them again before the next window.
        tail = self._eval("() => window.__sb.streamcost.read(0)")
        close_scan_ms = float(tail.get("overhead_ms") or 0.0) if isinstance(tail, dict) else 0.0
        self._overhead_ms += close_scan_ms
        out["close_scan_overhead_ms"] = round(close_scan_ms, 2)

        self._chars_open = None
        return out

    def end_cell(self, cell: Cell) -> Optional[dict]:
        # Declared even though level 0 is not obliged to: the whole argument for calling this
        # level 0 is that its cost does not grow with the rung, and a claim like that should be
        # checkable from the payload rather than from this docstring.
        return {
            "overhead_ms": round(self._overhead_ms, 2),
            "overhead_attempted": True,
            "overhead_note": (
                "measured inside the decode hook and at the window boundaries, not estimated. "
                "O(1) per SSE chunk, but the reply-length read is a querySelectorAll and so is "
                "O(the whole DOM): 38.8 ms per cell at 10K and 289.6 ms at 100K before the "
                "post-stream windows were skipped. It is identical on both arms of an A/B and "
                "cancels in a paired ratio; it does NOT cancel across rungs"
            ),
        }
