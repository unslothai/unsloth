# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A DENOMINATOR THAT IS SHORT BY AN UNKNOWN AMOUNT MUST NOT BE SCORED.

`streamcost` counts the characters delivered on the SSE wire and uses the delta across a window as
the denominator of every cost-per-character figure. A frame that cannot be parsed -- unrelated
`TextDecoder` traffic appended while `pending` holds a split SSE frame, a truncated stream, a
provider that emits something other than the expected shape -- increments a diagnostic counter and
leaves `wireChars` short.

Counting the failures was never the problem. NOTHING CONSULTED THE COUNT: not `streamcost.py`, not
`scoring/from_payload.py`. So the affected delta was still accepted as a denominator and every
cost-per-character derived from it came out inflated by an unknown factor, silently, in the
direction that makes the app look more expensive per character than it is.

Two ways to be short, and both are checked here: a frame that failed to parse INSIDE the window,
and an unterminated frame still sitting in the buffer at the window's close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

_STREAMCOST_JS = _STUDIO_TESTS / "studiobench" / "instruments" / "streamcost.js"


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


@pytest.fixture(scope = "module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    pg = browser.new_page(viewport = {"width": 900, "height": 600})
    pg.set_content("<!doctype html><meta charset=utf-8><body></body>")
    pg.add_script_tag(content = _STREAMCOST_JS.read_text(encoding = "utf-8"))
    yield pg
    pg.close()


def _feed(page, text: str) -> None:
    """Push bytes through the real `TextDecoder.prototype.decode` hook, AS ONE RESPONSE.

    The decoder is reused across calls because that is what the app does: `chat-api.ts` builds one
    `TextDecoder` per streaming request and calls `decode(value, {stream: true})` on it for every
    chunk of that response. A split frame is therefore always two chunks through the SAME decoder,
    which is what the reassembly tests below mean by a split. Building a fresh decoder per chunk
    would model a socket rather than a response, and reassembly state is scoped per decoder.
    """
    page.evaluate(
        """(text) => {
             const bytes = new TextEncoder().encode(text);
             if (!window.__testDecoder) window.__testDecoder = new TextDecoder();
             window.__testDecoder.decode(bytes);
           }""",
        text,
    )


def _feed_other(page, text: str) -> None:
    """Push bytes through a DIFFERENT decoder, as any other component of the page does.

    `TextDecoder.prototype.decode` is hooked page-wide, so every decoder in the document arrives
    here: the app builds one per streaming request and has six other read loops besides the chat
    one. A decoder that is not carrying the relay's framing is not the stream being measured.
    """
    page.evaluate(
        """(text) => {
             const bytes = new TextEncoder().encode(text);
             if (!window.__otherDecoder) window.__otherDecoder = new TextDecoder();
             window.__otherDecoder.decode(bytes);
           }""",
        text,
    )


def _end_response(page) -> None:
    """Abandon this response's decoder, as the app does when a stream ends or is aborted.

    The next `_feed` builds a new one, exactly as the next `send_turn` does.
    """
    page.evaluate("() => { window.__testDecoder = null; }")


def _frame(content: str) -> str:
    return 'data: {"choices":[{"delta":{"content":' + __import__("json").dumps(content) + "}}]}\n\n"


def test_a_clean_stream_is_scoreable_and_counts_every_character(page):
    """The control. Without this the failure tests below could pass on an instrument that marks
    everything unscoreable and counts nothing."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("hello") + _frame(" world"))
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("hello world")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got == {
        "failures": 0,
        "pending_chars": 0,
        "carried_flushes": got["carried_flushes"],
        # Not pinned to a value, on the same footing as `carried_flushes` beside it: the
        # id is which decoder the two numbers above are about, and the exact integer
        # depends on how many decoders the page has built before this assertion.
        "decoder_id": got["decoder_id"],
    }, got


def test_an_unparseable_frame_is_visible_at_the_window_boundary(page):
    """The counter has to be readable at O(1) cost at both ends of a window, or a window cannot
    tell a failure that happened inside it from one that happened before it."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("good"))
    before = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    _feed(page, "data: {this is not json}\n\n")
    after = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert after["failures"] > before["failures"], (before, after)


def test_an_unterminated_frame_is_reported_as_still_buffered(page):
    """The other way to be short. The frame never completed, so its characters were never counted,
    and at the close of the window the count is missing them with nothing to say so."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, 'data: {"choices":[{"delta":{"content":"half a fra')
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got["pending_chars"] > 0, got


# ── the consumer, which is where the defect actually lived ──────────


def _instrument(page):
    from studiobench.instruments.streamcost import StreamCostInstrument

    inst = StreamCostInstrument()
    inst._eval = lambda script, *a: page.evaluate(script, *a)  # noqa: SLF001
    inst.unavailable = None
    return inst


class _Window:
    duration_ms = 1000.0


def test_a_window_containing_a_parse_failure_is_marked_unscoreable(page):
    """THE DEFECT. Before the fix this window's `reply_chars_delta` was published with nothing to
    indicate that the wire count underneath it had lost an unknown number of characters, and every
    cost-per-character derived from it was inflated."""
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    _feed(page, _frame("counted"))
    _feed(page, "data: {this is not json}\n\n")
    out = inst.close(_Window())
    assert out["reply_chars_scoreable"] is False, out
    assert out["wire_parse_failures_in_window"] >= 1
    assert "short by an unknown amount" in out["reply_chars_unscoreable_reason"]


def test_a_window_ending_mid_frame_is_marked_unscoreable(page):
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    _feed(page, _frame("counted"))
    _feed(page, 'data: {"choices":[{"delta":{"content":"unterminat')
    out = inst.close(_Window())
    assert out["reply_chars_scoreable"] is False, out
    assert out["wire_pending_chars_at_close"] > 0


def test_a_clean_window_stays_scoreable(page):
    """The fix must not mark everything unscoreable; a check that never passes is as useless as one
    that never fails."""
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    _feed(page, _frame("all") + _frame(" good"))
    out = inst.close(_Window())
    assert out["reply_chars_scoreable"] is True, out
    assert out["wire_parse_failures_in_window"] == 0
    assert out["reply_chars_delta"] == len("all good")


# ── a frame split INSIDE the "data:" prefix ─────────────────────────
#
# The hook starts buffering when a chunk contains a complete `data:`. The socket does not respect
# that boundary: it can cut a frame anywhere, including between the "da" and the "ta:". Neither
# half then contains the marker and neither completes a buffered frame, so both were discarded --
# and the frame's characters left the denominator WITHOUT incrementing `wireParseFailures`, so the
# window-integrity check above could not see it either. Later frames make the window look
# scoreable while the count underneath it is short, which inflates every cost-per-character
# derived from it, at exactly the moment the instrument exists to measure: chunks arrive ragged
# when the renderer is jammed.


def _halves(text: str, at: int) -> tuple:
    return text[:at], text[at:]


def test_the_counter_survives_a_split_inside_the_data_prefix(page):
    """THE DEFECT. Two chunks, neither of which contains `data:`, and one whole frame between
    them."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    head, tail = _halves(_frame("split marker"), 2)
    assert "data:" not in head and "data:" not in tail, (head, tail)
    _feed(page, head)
    _feed(page, tail)
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("split marker")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got == {
        "failures": 0,
        "pending_chars": 0,
        "carried_flushes": got["carried_flushes"],
        # Not pinned to a value, on the same footing as `carried_flushes` beside it: the
        # id is which decoder the two numbers above are about, and the exact integer
        # depends on how many decoders the page has built before this assertion.
        "decoder_id": got["decoder_id"],
    }, got


def test_a_held_marker_fragment_is_reported_as_buffered_rather_than_lost(page):
    """While the second half has not arrived, the frame is not counted -- so a window closing here
    has a short denominator and has to say so. Dropping the fragment reported `pending_chars: 0`,
    which is the instrument stating that nothing was outstanding while a frame was."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("orphan")[:3])
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got["pending_chars"] > 0, got
    assert got["failures"] == 0, got


def test_a_marker_split_three_ways_is_still_one_frame(page):
    """The socket is under no obligation to cut in a convenient place, and a fix that only handles
    a two-way split is a fix for the example rather than for the defect."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    text = _frame("three ways")
    for part in (text[:1], text[1:2], text[2:4], text[4:]):
        _feed(page, part)
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("three ways")


def test_unrelated_text_ending_in_a_marker_letter_does_not_corrupt_the_next_frame(page):
    """THE WAY THIS FIX COULD HAVE BEEN WORSE THAN THE BUG. Any page traffic may end in "d", "da"
    or "data", and gluing that onto the front of a real frame makes "ddata: {...}", which no longer
    starts with the marker and would be skipped in silence."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, "an unrelated chunk that ends in a d")
    _feed(page, _frame("counted anyway"))
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("counted anyway")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got == {
        "failures": 0,
        "pending_chars": 0,
        "carried_flushes": got["carried_flushes"],
        # Not pinned to a value, on the same footing as `carried_flushes` beside it: the
        # id is which decoder the two numbers above are about, and the exact integer
        # depends on how many decoders the page has built before this assertion.
        "decoder_id": got["decoder_id"],
    }, got


def test_the_speculative_buffer_cannot_grow_with_unrelated_traffic(page):
    """The memory bound, asserted rather than argued. A fix that buffered every chunk in the page
    on the chance that one of them was an SSE frame would be worse than the bug it fixes: a tail
    short enough to be a partial `data:` is at most four characters."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    for i in range(50):
        _feed(page, f"chunk {i} of unrelated traffic, ending in data")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got["pending_chars"] <= 4, got
    assert got["failures"] == 0, got


def test_a_window_whose_only_frame_was_split_in_the_marker_still_counts_it(page):
    """THE CONSEQUENCE at the window boundary, which is where the number is used. The window used
    to close scoreable, with a `reply_chars_delta` of zero over a frame that really was delivered:
    a denominator short by an unknown amount wearing a clean bill of health."""
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    head, tail = _halves(_frame("inside the window"), 3)
    _feed(page, head)
    _feed(page, tail)
    out = inst.close(_Window())
    assert out["reply_chars_delta"] == len("inside the window"), out
    assert out["reply_chars_scoreable"] is True, out


def test_a_failure_before_the_window_does_not_taint_it(page):
    """Attribution matters: a window is scoreable or not on its OWN evidence. Counting total
    failures rather than the delta would condemn every window after the first bad frame."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, "data: {this is not json}\n\n")
    inst = _instrument(page)
    inst.open(_Window())
    _feed(page, _frame("clean"))
    out = inst.close(_Window())
    assert out["reply_chars_scoreable"] is True, out


# ── the OTHER end of a split, which is the window that opens on it ──


def test_a_window_that_opens_on_a_half_delivered_frame_is_not_scoreable(page):
    """THE DEFECT, one window to the right of the one already covered above.

    `reset()` cannot clear `pending` -- it holds half a frame whose other half has not arrived --
    so a frame the socket cut across a window boundary is still buffered when the NEXT window
    opens. Its suffix lands inside that window, the parser adds the WHOLE frame's characters
    there and empties the buffer, and the close reading then sees zero failures and zero residual
    and calls the window scoreable. Part of its denominator was delivered before it opened, so
    `reply_chars_delta` is not "characters delivered in this window" and the cost per character
    above it is wrong in a direction nothing in the row discloses.
    """
    page.evaluate("() => window.__sb.streamcost.reset()")
    head, tail = _halves(_frame("straddles the boundary"), 30)
    assert "\n\n" not in head, head
    # The first window closes mid-frame: already refused, by `wire_pending_chars_at_close`.
    first = _instrument(page)
    first.open(_Window())
    _feed(page, head)
    closed = first.close(_Window())
    assert closed["reply_chars_scoreable"] is False, closed
    assert closed["wire_pending_chars_at_close"] > 0, closed

    # The second opens holding that frame, and is handed its whole character count.
    second = _instrument(page)
    second.open(_Window())
    _feed(page, tail)
    out = second.close(_Window())
    assert out["wire_parse_failures_in_window"] == 0, out
    assert out["wire_pending_chars_at_close"] == 0, out
    assert out["wire_pending_chars_at_open"] > 0, out
    # The whole frame was charged here, including the part delivered in the window before it.
    assert out["reply_chars_delta"] == len("straddles the boundary"), out
    assert out["reply_chars_scoreable"] is False, out
    assert "already buffered" in out["reply_chars_unscoreable_reason"], out


def test_a_window_that_opens_on_an_empty_buffer_is_still_scoreable(page):
    """The positive control. A refusal that fires on every window measures nothing at all."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("before"))
    inst = _instrument(page)
    inst.open(_Window())
    _feed(page, _frame("during"))
    out = inst.close(_Window())
    assert out["wire_pending_chars_at_open"] == 0, out
    assert out["reply_chars_scoreable"] is True, out
    assert out["reply_chars_delta"] == len("during"), out


def test_a_marker_fragment_held_at_the_open_does_not_cost_the_window_its_reading(page):
    """THE BOUNDARY OF THE REFUSAL, and it is a boundary rather than a hole.

    A decoder whose first chunk is "dat" is holding a marker fragment, so `wireIntegrity` reports
    it as buffered -- deliberately, and the reason is above `markerHold`. But that decoder is NOT
    `active` yet, so `decoder_id` names whoever was, and when the fragment's frame completes
    inside the window the carried flush lands on the new decoder and the delta stays zero. The
    window is scoreable.

    That is correct, and the quantity is why. `markerTail` is at most four characters of the
    literal `data:` and is only ever set while `pending` is empty, so a held fragment carries
    ZERO denominator characters: every character `reply_chars_delta` counts here was delivered
    inside the window. The refusal exists for a buffer whose characters were delivered before the
    window and counted inside it, which is the half-frame case above, and that one still names
    the decoder holding it and still fires.
    """
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("earlier"))
    _end_response(page)
    # The next response begins with a chunk that is only part of the marker.
    _feed(page, "dat")
    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] == 3, inst._integrity_open

    _feed(page, 'a: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
    out = inst.close(_Window())
    assert out["wire_pending_chars_at_open"] == 3, out
    assert out["wire_pending_chars_at_close"] == 0, out
    assert out["wire_parse_failures_in_window"] == 0, out
    # Every counted character arrived inside the window: the fragment held none of them.
    assert out["reply_chars_delta"] == len("hello"), out
    assert out["reply_chars_scoreable"] is True, out


# ── an aborted response must not poison the next one ──────────────────────────


def test_an_aborted_frame_does_not_follow_the_stream_that_replaces_it(page):
    """REGRESSION. `stop_generation` cuts a socket mid-frame, which is what it is for.

    The reassembly buffer used to be one page-wide variable, so the abandoned JSON tail waited
    there for the next response. `send_turn` follows `stop_generation` in all three shipped
    schedules, and its first chunk was glued behind that tail: the merged part no longer began
    `data:`, so it was skipped by `continue` WITHOUT counting a parse failure, and the denominator
    went quietly short -- the one outcome this file exists to prevent. The residue never cleared
    either, so `pending_chars` stayed above zero and every later window was refused as well.
    """
    page.evaluate("() => window.__sb.streamcost.reset()")
    # A response aborted in the middle of a frame.
    _feed(page, 'data: {"choices":[{"delta":{"content":"half a re')
    assert page.evaluate("() => window.__sb.streamcost.wireIntegrity()")["pending_chars"] > 0
    _end_response(page)

    # The next turn is a new response, and it is intact.
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("hello"))
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("hello")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got == {
        "failures": 0,
        "pending_chars": 0,
        "carried_flushes": got["carried_flushes"],
        # Not pinned to a value, on the same footing as `carried_flushes` beside it: the
        # id is which decoder the two numbers above are about, and the exact integer
        # depends on how many decoders the page has built before this assertion.
        "decoder_id": got["decoder_id"],
    }, got


def test_a_split_inside_one_response_still_reassembles_after_an_abort(page):
    """The scoping must not cost a genuine split its repair: same decoder, still one frame."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, 'data: {"choices":[{"delta":{"content":"abandoned')
    _end_response(page)

    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, 'data: {"choices":[{"delta":{"con')
    _feed(page, 'tent":"rejoined"}}]}\n\n')
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("rejoined")
    assert page.evaluate("() => window.__sb.streamcost.wireIntegrity()")["pending_chars"] == 0


def test_an_unrelated_decoder_does_not_hide_a_frame_the_stream_is_holding(page):
    """THE DEFECT. `active` moved to whichever decoder decoded last, before anything had looked at
    the chunk, so any other TextDecoder in the page took the report away from the stream.

    Its buffer is empty, so `wireIntegrity` answered "nothing outstanding" while the SSE decoder
    held half a frame: the window closing there published a denominator short by that frame with a
    clean bill of health, and the window the suffix landed in was handed the whole frame -- both
    ends of the split accepted, which is the one outcome this file exists to prevent."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    head, tail = _halves(_frame("straddles the boundary"), 30)
    assert "\n\n" not in head, head

    first = _instrument(page)
    first.open(_Window())
    _feed(page, head)
    _feed_other(page, "an unrelated chunk of page traffic")
    closed = first.close(_Window())
    assert closed["wire_pending_chars_at_close"] > 0, closed
    assert closed["reply_chars_scoreable"] is False, closed

    second = _instrument(page)
    second.open(_Window())
    _feed(page, tail)
    out = second.close(_Window())
    assert out["wire_pending_chars_at_open"] > 0, out
    assert out["reply_chars_delta"] == len("straddles the boundary"), out
    assert out["reply_chars_scoreable"] is False, out


def test_a_window_opening_after_an_abort_keeps_the_next_response_scoreable(page):
    """THE OTHER HALF OF THE SAME SCOPING, and a reading that was being thrown away.

    `open()` samples the integrity BEFORE the action has created the response it will measure, so
    the buffer it sees is still the aborted one. That half frame is never completed and never
    counted anywhere, so it takes nothing out of this window's delta -- but `pending_chars > 0` at
    the open refused the window on its own, and `send_turn` follows `stop_generation` in all three
    shipped schedules. Every socket split at the abort cost the next window its denominator."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, 'data: {"choices":[{"delta":{"content":"half a re')
    _end_response(page)

    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] > 0, inst._integrity_open  # noqa: SLF001
    _feed(page, _frame("a clean reply"))
    out = inst.close(_Window())
    assert out["reply_chars_delta"] == len("a clean reply"), out
    assert out["wire_carried_frames_counted_in_window"] == 0, out
    assert out["reply_chars_scoreable"] is True, out


def test_an_abort_does_not_cost_the_next_response_its_reading_when_that_one_is_split(page):
    """THE RESIDUAL ON THAT FIX, and it survived because the two halves of the refusal were read at
    two different scopes. `pending_at_open` belongs to ONE decoder; `carriedFlushes` was a counter
    on `S` that any decoder could move. So the abort's orphaned half frame was paired with a
    carried flush produced by the NEW response's own split, and the window was refused for a buffer
    that never flushed -- the same false refusal, surviving in the fragmented case.

    It is the fragmented case that matters most. Reads go ragged when the renderer is jammed, which
    is exactly the window worth measuring, so the loss is biased against the expensive windows.
    Measured before this change: this window delivered all 13 of its characters and was refused,
    while the same traffic with no abort in front of it was accepted."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, 'data: {"choices":[{"delta":{"content":"half a re')
    _end_response(page)

    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] > 0, inst._integrity_open  # noqa: SLF001
    frame = _frame("a clean reply")
    head, tail = _halves(frame, frame.index("a clean") + 4)
    assert "\n\n" not in head, head
    _feed(page, head)
    _feed(page, tail)
    out = inst.close(_Window())
    assert out["reply_chars_delta"] == len("a clean reply"), out
    # The new response DID carry a frame across a decode boundary, so the counter really moved:
    # this window is accepted despite that, not because nothing was counted.
    assert out["wire_carried_frames_counted_in_window"] == 0, out
    assert out["reply_chars_scoreable"] is True, out


def test_the_carried_counter_still_moves_for_the_decoder_that_owns_the_split(page):
    """THE CONTROL FOR THE ONE ABOVE. Without it that test passes just as well against an
    instrument that has stopped counting carried flushes altogether, which would take the genuine
    straddle refusal down with it. Same fragmentation, no abort in front of it.

    ASKED OF THE DECODER THAT OWNS THE SPLIT, which is the whole point of the change. The window's
    own `wire_carried_frames_counted_in_window` reads 0 here and that is correct rather than
    convenient: nothing was pending when it opened, so the decoder it names is not the one that
    went on to carry a frame, and its count is the honest answer to the question the refusal asks.
    The counter itself has to be shown to have moved, and only the decoder holding it can say so."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] == 0, inst._integrity_open  # noqa: SLF001
    # Read BEFORE the close, which clears the open sample.
    opened_on = inst._integrity_open["decoder_id"]  # noqa: SLF001
    frame = _frame("a clean reply")
    head, tail = _halves(frame, frame.index("a clean") + 4)
    _feed(page, head)
    _feed(page, tail)
    out = inst.close(_Window())
    live = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert live["carried_flushes"] == 1, live
    assert live["decoder_id"] != opened_on, (live, opened_on)
    assert out["wire_carried_frames_counted_in_window"] == 0, out
    assert out["reply_chars_delta"] == len("a clean reply"), out
    assert out["reply_chars_scoreable"] is True, out


def test_a_frame_that_really_did_straddle_the_open_still_refuses_its_window(page):
    """THE REFUSAL THIS MUST NOT REMOVE, in the shape that separates the two candidate fixes.

    The decoder that was pending at the open completes its carried frame INSIDE the window, so part
    of the delta really was delivered before the window opened -- and then another decoder becomes
    the active one before the close. A fix that compared the open's decoder id with the close's
    would see two different ids, discard the carry it is looking for, and accept a window whose
    denominator is wrong. Asking `wireIntegrity` about the NAMED decoder answers regardless of who
    is active now."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    frame = _frame("straddles the open")
    head, tail = _halves(frame, 30)
    assert "\n\n" not in head, head
    _feed(page, head)

    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] > 0, inst._integrity_open  # noqa: SLF001
    _feed(page, tail)
    # Some other decoder takes over as active before the close.
    _feed_other(page, "unrelated page traffic")
    out = inst.close(_Window())
    assert out["wire_carried_frames_counted_in_window"] == 1, out
    assert out["reply_chars_scoreable"] is False, out


# ── the tail of a split frame is stream traffic on the numerator too ─────────
#
# `noteSse` was gated on the marker while the character counter was gated on `looksSse ||
# pending`, so the two halves of one frame were attributed to two different things: the tail's
# characters went into the denominator and its render work went nowhere. The bias is downward on
# `stream_delta_cost_ms_per_kchar` and it grows with fragmentation, which is the regime the
# instrument exists to measure.


def test_the_tail_of_a_split_frame_is_counted_as_stream_traffic(page):
    """THE DEFECT, at the quantity the numerator is built from.

    The frame is cut inside its JSON body, so the head carries `data:` and the tail carries no
    marker at all -- it is the rest of the body and the blank line. The tail still completes the
    frame and its characters are counted, so a chunk that the instrument scores the cost of has
    to be a chunk the instrument charges the cost to."""
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    frame = _frame("cut inside the body")
    head, tail = _halves(frame, frame.index("cut") + 3)
    assert "data:" in head and "data:" not in tail, (head, tail)
    _feed(page, head)
    _feed(page, tail)
    out = inst.close(_Window())

    # Both chunks carried this response's bytes, so both are stream chunks.
    assert out["sse_chunks"] == 2, out
    assert out["reply_chars_delta"] == len("cut inside the body"), out
    assert out["reply_chars_scoreable"] is True, out


def test_unrelated_traffic_is_still_not_stream_traffic(page):
    """The control the widened condition must not break. A decoder carrying no marker and holding
    no frame of its own is not the stream, and feeding it must not add a chunk."""
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    _feed(page, _frame("real"))
    _feed_other(page, "nothing to do with the relay")
    out = inst.close(_Window())

    assert out["sse_chunks"] == 1, out
    assert out["reply_chars_delta"] == len("real"), out
