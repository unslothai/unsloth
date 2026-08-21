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
    """Push bytes through the real `TextDecoder.prototype.decode` hook."""
    page.evaluate(
        """(text) => {
             const bytes = new TextEncoder().encode(text);
             new TextDecoder().decode(bytes);
           }""",
        text,
    )


def _frame(content: str) -> str:
    return 'data: {"choices":[{"delta":{"content":' + __import__("json").dumps(content) + "}}]}\n\n"


def test_a_clean_stream_is_scoreable_and_counts_every_character(page):
    """The control. Without this the failure tests below could pass on an instrument that marks
    everything unscoreable and counts nothing."""
    page.evaluate("() => window.__sb.streamcost.reset()")
    _feed(page, _frame("hello") + _frame(" world"))
    assert page.evaluate("() => window.__sb.streamcost.replyChars()") == len("hello world")
    got = page.evaluate("() => window.__sb.streamcost.wireIntegrity()")
    assert got == {"failures": 0, "pending_chars": 0}, got


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
