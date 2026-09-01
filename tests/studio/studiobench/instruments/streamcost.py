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
        self._integrity_open: dict = {}
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
        # O(1), off the wire. This used to be a `querySelectorAll` over the whole document at
        # both ends of every window; see the long note in streamcost.js for why that had to go.
        self._chars_open = self._eval(
            "() => window.__sb.streamcost && window.__sb.streamcost.replyChars()"
        )
        # THE DENOMINATOR'S INTEGRITY, SAMPLED AT BOTH ENDS. A frame that cannot be parsed --
        # unrelated `TextDecoder` traffic appended while `pending` holds a split SSE frame, say --
        # increments a diagnostic and leaves `wireChars` short by an amount nobody can recover.
        # Counting the failures was not enough on its own: nothing consulted the count, so the
        # affected `reply_chars_delta` was still accepted as a denominator and every cost-per-
        # character derived from it came out inflated by an unknown factor. A delta spanning a new
        # failure, or ending with an unterminated frame still buffered, is now marked unscoreable
        # at the window that contains it.
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
        # No `force` argument any more. The old DOM read was skipped on windows long past the
        # stream because it was expensive; this one is a counter read, so it is taken on every
        # window unconditionally and the "the open-read was skipped, so the pair is half-taken"
        # case cannot arise.
        chars_close = self._eval("() => window.__sb.streamcost.replyChars()")
        out["stream_cost_attempted"] = True
        out["reply_chars_open"] = self._chars_open
        out["reply_chars_close"] = chars_close
        # WHERE THE DENOMINATOR CAME FROM, in the payload rather than in this file. A reader
        # comparing a run recorded before this change with one recorded after is comparing two
        # different quantities -- the growth of the last mounted message against the characters
        # delivered to the page -- and nothing else in the row would tell them so.
        out["reply_chars_source"] = "sse_wire"
        # ABOUT THE DECODER THAT WAS PENDING AT THE OPEN, named rather than assumed. The refusal
        # below pairs a buffer with a flush, and the two were read at different scopes: the buffer
        # belonged to one decoder, the flush counter was page-wide. Since `send_turn` follows
        # `stop_generation` in every shipped schedule, an abort's orphaned half frame was routinely
        # paired with a carried flush produced by the NEW response's own split, refusing a window
        # that had delivered every one of its characters. Naming the open's decoder makes the pair
        # a statement about one response.
        integrity = (
            self._eval(
                "(id) => window.__sb.streamcost.wireIntegrity(id)",
                self._integrity_open.get("decoder_id"),
            )
            or {}
        )
        failures = (integrity.get("failures") or 0) - (self._integrity_open.get("failures") or 0)
        residual = integrity.get("pending_chars") or 0
        # AND THE OTHER END OF THE SAME SPLIT. `pending` is not cleared by `reset()` -- it cannot
        # be, it holds half a frame -- so a frame the socket cut across a window boundary is still
        # buffered when the NEXT window opens. Its suffix arrives inside that window, the parser
        # adds the WHOLE frame's characters there and empties the buffer, and the close reading
        # then sees no failures and no residual and calls the window scoreable. Part of its
        # denominator was delivered before it opened. The window that closed on the split is
        # already refused by `residual`; this refuses its partner, and the two together are what
        # make `reply_chars_delta` mean "characters delivered IN this window" rather than
        # "characters counted in this window".
        #
        # AND IT IS THE FLUSH, NOT THE BUFFER, THAT MAKES THE DELTA WRONG. `open()` samples the
        # integrity before the action has created the response it is about to measure, so the
        # buffer it sees can belong to a response that is already over: `stop_generation` exists to
        # cut a socket mid-frame and `send_turn` follows it in every shipped schedule. That half
        # frame is never completed and never counted anywhere, so it takes nothing out of this
        # window's delta -- refusing on its presence alone threw away the one stream-cost reading
        # the send_turn window has. `carried_flushes` counts the completions of frames that were
        # already buffered when the decode began, so the pair "a buffer was pending at the open"
        # and "a carried frame was counted inside the window" is what the refusal now needs.
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
            # NOW UNREACHABLE, AND KEPT ANYWAY. The wire counter is cumulative and monotonic, so
            # it cannot go backwards; if it ever does, the instrument has been reset underneath
            # itself and the delta is meaningless. Under the old DOM reading this branch fired
            # legitimately and often: `send_turn` starts a new assistant message and
            # `thread_reopen` rebuilds the thread, so the LAST message was routinely shorter at
            # the close of a window than at its open, and those windows lost their denominator
            # for a reason that had nothing to do with streaming.
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
        self._integrity_open = {}
        return out

    def end_cell(self, cell: Cell) -> Optional[dict]:
        # Declared even though level 0 is not obliged to: the whole argument for calling this
        # level 0 is that its cost does not grow with the rung, and a claim like that should be
        # checkable from the payload rather than from this docstring.
        #
        # THE ONE DOM READ, and it happens HERE, which is after the last window of the cell has
        # closed and before the next cell's first one opens. Its cost is charged to no window and
        # therefore to no arm. It exists because the wire count and the DOM count answer different
        # questions -- what the app was sent, and what it rendered -- and their disagreement is
        # the cheapest available check that a windowed arm is dropping text rather than merely
        # not mounting it.
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
            # Outside every window, so this number is not in anybody's frame rate.
            "last_message_chars_in_dom": dom_chars,
            "last_message_chars_note": (
                "read once, after the film, purely as a cross-check against the wire count. Never "
                "inside a measured window and never used as a denominator"
            ),
        }
