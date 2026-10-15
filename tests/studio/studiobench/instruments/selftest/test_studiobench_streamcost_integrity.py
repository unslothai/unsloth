# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows whose wireChars is short by unparsed or unterminated frames must be refused, not scored."""

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
    """Feeds one response through one reused decoder, as chat-api.ts does, so split frames reassemble."""
    page.evaluate(
        """(text) => {
             const bytes = new TextEncoder().encode(text);
             if (!window.__testDecoder) window.__testDecoder = new TextDecoder();
             window.__testDecoder.decode(bytes);
           }""",
        text,
    )


def _feed_other(page, text: str) -> None:
    """Feeds an unrelated decoder, as other page components do; it is not the stream being measured."""
    page.evaluate(
        """(text) => {
             const bytes = new TextEncoder().encode(text);
             if (!window.__otherDecoder) window.__otherDecoder = new TextDecoder();
             window.__otherDecoder.decode(bytes);
           }""",
        text,
    )


def _end_response(page) -> None:
    """Drops the response's decoder, as the app does on end or abort; the next _feed builds a fresh one."""
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
        # Not pinned: the exact id depends on how many decoders the page built.
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


# A socket can cut a frame inside "data:", so neither half contains the marker.


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
        # Not pinned: the exact id depends on how many decoders the page built.
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
        # Not pinned: the exact id depends on how many decoders the page built.
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


def test_a_window_that_opens_on_a_half_delivered_frame_is_not_scoreable(page):
    """A window opening on a half-delivered frame is not scoreable: part of its denominator came earlier."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    head, tail = _halves(_frame("straddles the boundary"), 30)
    assert "\n\n" not in head, head
    first = _instrument(page)
    first.open(_Window())
    _feed(page, head)
    closed = first.close(_Window())
    assert closed["reply_chars_scoreable"] is False, closed
    assert closed["wire_pending_chars_at_close"] > 0, closed

    second = _instrument(page)
    second.open(_Window())
    _feed(page, tail)
    out = second.close(_Window())
    assert out["wire_parse_failures_in_window"] == 0, out
    assert out["wire_pending_chars_at_close"] == 0, out
    assert out["wire_pending_chars_at_open"] > 0, out
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
    """A held marker fragment carries zero denominator characters, so its window stays scoreable."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("earlier"))
    _end_response(page)
    _feed(page, "dat")
    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] == 3, inst._integrity_open

    _feed(page, 'a: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
    out = inst.close(_Window())
    assert out["wire_pending_chars_at_open"] == 3, out
    assert out["wire_pending_chars_at_close"] == 0, out
    assert out["wire_parse_failures_in_window"] == 0, out
    assert out["reply_chars_delta"] == len("hello"), out
    assert out["reply_chars_scoreable"] is True, out


def test_an_aborted_frame_does_not_follow_the_stream_that_replaces_it(page):
    """Aborted frame residue must not join the next response, so the reassembly buffer is per response."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, 'data: {"choices":[{"delta":{"content":"half a re')
    assert page.evaluate("() => window.__sb.streamcost.wireIntegrity()")["pending_chars"] > 0
    _end_response(page)

    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("hello"))
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("hello")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got == {
        "failures": 0,
        "pending_chars": 0,
        "carried_flushes": got["carried_flushes"],
        # Not pinned: the exact id depends on how many decoders the page built.
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
    """An unrelated decoder must not make the stream inactive, or wireIntegrity misses the held frame."""
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
    """A half frame left by an abort must not refuse the next window at open; it never enters its delta."""
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
    """Refusal must pair a carried flush with the decoder that owned the split, not a shared counter."""
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
    # The counter did move: the window is accepted despite a carried frame.
    assert out["wire_carried_frames_counted_in_window"] == 0, out
    assert out["reply_chars_scoreable"] is True, out


def test_the_carried_counter_still_moves_for_the_decoder_that_owns_the_split(page):
    """The carried-flush counter must still move for the split's own decoder, or real straddles pass."""
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
    """Ask wireIntegrity about the decoder named at open, not whichever decoder is active at close."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    frame = _frame("straddles the open")
    head, tail = _halves(frame, 30)
    assert "\n\n" not in head, head
    _feed(page, head)

    inst = _instrument(page)
    inst.open(_Window())
    assert inst._integrity_open["pending_chars"] > 0, inst._integrity_open  # noqa: SLF001
    _feed(page, tail)
    _feed_other(page, "unrelated page traffic")
    out = inst.close(_Window())
    assert out["wire_carried_frames_counted_in_window"] == 1, out
    assert out["reply_chars_scoreable"] is False, out


# The tail of a split frame must count on the numerator too, or cost per char biases down.


def test_the_tail_of_a_split_frame_is_counted_as_stream_traffic(page):
    """The tail of a split frame must be counted as stream traffic, so its cost is charged to the stream."""
    inst = _instrument(page)
    page.evaluate("() => window.__sb.streamcost.reset()")
    inst.open(_Window())
    frame = _frame("cut inside the body")
    head, tail = _halves(frame, frame.index("cut") + 3)
    assert "data:" in head and "data:" not in tail, (head, tail)
    _feed(page, head)
    _feed(page, tail)
    out = inst.close(_Window())

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
