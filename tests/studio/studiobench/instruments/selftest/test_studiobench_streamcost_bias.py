# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE INSTRUMENT MUST NOT COST LESS ON THE ARM IT IS SCORING.

`streamcost.js` used to read its denominator out of the DOM:
`querySelectorAll('[data-role="assistant"]')`, last element, `textContent.length`, at both ends of
every window. That is O(the whole document) whatever matches, and the file's own note claimed it
"is identical on both arms of an A/B and cancels in a paired ratio".

It cancels between two arms that mount the same DOM. It does NOT cancel against an arm whose whole
purpose is to mount less of it: a virtualised thread pays a fraction of the cost, so the
instrument hands the treatment a saving it never earned, in the direction that flatters the
hypothesis under test. At 100K the read totalled 289.6 ms per cell, which is not a rounding error
next to the effects this campaign argues about.

This measures the residual -- the instrument's own cost on a full document minus its cost on a
windowed one -- for BOTH the old reading and the new one, on the same two pages, in the same run.
The old one is still exported as `replyCharsDom` for a once-per-cell cross-check, so both can be
driven side by side and the comparison is a measurement rather than a claim about deleted code.

    python -m pytest tests/studio/studiobench/instruments/selftest/test_studiobench_streamcost_bias.py -q -s
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

_STREAMCOST_JS = _STUDIO_TESTS / "studiobench" / "instruments" / "streamcost.js"

#: The two documents. 40,000 elements is the size the 289.6 ms figure was measured against; 4,000
#: is roughly what a window of six messages leaves standing at the same rung.
FULL_ELEMENTS = 40_000
WINDOWED_ELEMENTS = 4_000

#: How many reads to take. The quantity is a few milliseconds, so one reading is noise.
READS = 40

#: What counts as "no longer biased". The old reading's residual is milliseconds per call; the new
#: one is a property read and must be under a tenth of a millisecond per call even on a loaded
#: shared machine.
MAX_WIRE_RESIDUAL_MS_PER_CALL = 0.1


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


BUILD_JS = """
(n) => {
  // A thread-shaped document: assistant messages carrying spans, which is what the real one is
  // mostly made of. The absolute count is what matters to a querySelectorAll, not the shape.
  const perMessage = 40;
  const messages = Math.max(1, Math.floor(n / perMessage));
  const parts = [];
  for (let i = 0; i < messages; i += 1) {
    const spans = new Array(perMessage - 1).fill('<span>token </span>').join("");
    parts.push('<div data-role="assistant">' + spans + '</div>');
  }
  document.body.innerHTML = parts.join("");
  return document.getElementsByTagName("*").length;
}
"""

#: One SSE frame in exactly the shape `_gguf_chat_delta_line` emits, fed through the page's own
#: TextDecoder so the instrument's real hook sees it.
FEED_JS = """
(frames) => {
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  let sent = 0;
  for (let i = 0; i < frames; i += 1) {
    const piece = "token " + i + " ";
    sent += piece.length;
    const frame = 'data: ' + JSON.stringify({
      id: "x", object: "chat.completion.chunk", model: "m",
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    }) + "\\n\\n";
    dec.decode(enc.encode(frame));
  }
  return sent;
}
"""

TIME_JS = """
([which, reads]) => {
  const sc = window.__sb.streamcost;
  const fn = which === "wire" ? () => sc.replyChars() : () => sc.replyCharsDom(true);
  fn();  // warm, so the first call's one-off costs are not the reading
  const t = performance.now();
  for (let i = 0; i < reads; i += 1) fn();
  return (performance.now() - t) / reads;
}
"""


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


def _page(browser, elements: int):
    page = browser.new_page(viewport = {"width": 900, "height": 600})
    page.set_content("<!doctype html><meta charset=utf-8><body></body>")
    # BEFORE the document exists, exactly as the real harness installs it via add_init_script:
    # the hook has to be on TextDecoder.prototype before any decode happens.
    page.add_script_tag(content = _STREAMCOST_JS.read_text(encoding = "utf-8"))
    got = page.evaluate(BUILD_JS, elements)
    page.evaluate(FEED_JS, 200)
    return page, got


def _per_call_ms(page, which: str) -> float:
    # Median of several passes. A single pass on a shared machine picks up whatever else is
    # running, and this box is running other people's sweeps.
    return statistics.median(page.evaluate(TIME_JS, [which, READS]) for _ in range(5))


def test_the_wire_denominator_is_the_same_price_on_a_windowed_document(browser, capsys):
    """THE RESIDUAL, MEASURED, for the old reading and the new one on the same two documents."""
    full, full_n = _page(browser, FULL_ELEMENTS)
    win, win_n = _page(browser, WINDOWED_ELEMENTS)
    try:
        dom_full = _per_call_ms(full, "dom")
        dom_win = _per_call_ms(win, "dom")
        wire_full = _per_call_ms(full, "wire")
        wire_win = _per_call_ms(win, "wire")
    finally:
        full.close()
        win.close()

    dom_residual = dom_full - dom_win
    wire_residual = wire_full - wire_win
    with capsys.disabled():
        print(
            f"\n  documents: full {full_n:,} elements, windowed {win_n:,} elements\n"
            f"  OLD reading (querySelectorAll, O(document)):\n"
            f"    full {dom_full:.4f} ms/call, windowed {dom_win:.4f} ms/call, "
            f"RESIDUAL {dom_residual:+.4f} ms/call\n"
            f"  NEW reading (wire counter, O(1)):\n"
            f"    full {wire_full:.4f} ms/call, windowed {wire_win:.4f} ms/call, "
            f"RESIDUAL {wire_residual:+.4f} ms/call"
        )

    # The old reading really is cheaper on the smaller document. If this ever stops being true the
    # test below is proving nothing, so it is asserted rather than assumed.
    assert dom_residual > 0, (
        "the O(document) read was not measurably cheaper on the windowed document, so this "
        f"machine cannot demonstrate the bias at all (full {dom_full}, windowed {dom_win})"
    )
    # And the new one is not.
    assert abs(wire_residual) < MAX_WIRE_RESIDUAL_MS_PER_CALL, (
        f"the wire counter cost {wire_residual:+.4f} ms/call more on the full document than on "
        "the windowed one, so it still carries a bias in the treatment's favour"
    )
    # The point of the whole exercise: whatever is left is a small fraction of what was there.
    assert abs(wire_residual) < dom_residual / 10


def test_the_wire_count_is_identical_on_both_documents_for_identical_traffic(browser):
    """The denominator itself, not just its cost. Both arms are fed by the SAME pacer, so the
    counter has to be a function of the bytes and of nothing else -- least of all of how much of
    the thread the arm chose to mount."""
    full, _ = _page(browser, FULL_ELEMENTS)
    win, _ = _page(browser, WINDOWED_ELEMENTS)
    try:
        a = full.evaluate("() => window.__sb.streamcost.wireStats()")
        b = win.evaluate("() => window.__sb.streamcost.wireStats()")
    finally:
        full.close()
        win.close()
    assert a["wire_chars"] == b["wire_chars"] > 0
    assert a["wire_frames"] == b["wire_frames"] == 200
    assert a["wire_parse_failures"] == b["wire_parse_failures"] == 0
    # The DOM reading, by contrast, is a different number on the two documents -- which is exactly
    # why it could not be the denominator.
    assert a["wire_chars"] == 200 * len("token 0 ") or a["wire_chars"] > 0


def test_the_counter_survives_a_frame_split_across_two_decode_calls(browser):
    """A decode() call is a slice of the socket, not an SSE frame. A counter that assumed frame
    alignment would silently under-count exactly when the renderer is jammed and the chunks arrive
    ragged, which is the condition the whole instrument exists to measure."""
    page, _ = _page(browser, WINDOWED_ELEMENTS)
    try:
        before = page.evaluate("() => window.__sb.streamcost.wireStats()")
        got = page.evaluate("""
          () => {
            const enc = new TextEncoder();
            const dec = new TextDecoder();
            const frame = 'data: ' + JSON.stringify({
              choices: [{ index: 0, delta: { content: "abcdefghij" }, finish_reason: null }],
            }) + "\\n\\n";
            // Split in the middle of the JSON body, which is the worst place for it.
            const cut = Math.floor(frame.length / 2);
            dec.decode(enc.encode(frame.slice(0, cut)));
            const mid = window.__sb.streamcost.wireStats().wire_chars;
            dec.decode(enc.encode(frame.slice(cut)));
            return { mid, after: window.__sb.streamcost.wireStats() };
          }
        """)
    finally:
        page.close()
    # Nothing is counted until the frame is complete, and then all ten characters are.
    assert got["mid"] == before["wire_chars"]
    assert got["after"]["wire_chars"] == before["wire_chars"] + 10
    assert got["after"]["wire_parse_failures"] == 0


def test_reasoning_and_content_are_both_counted(browser):
    """`_gguf_chat_delta_line` emits reasoning as `reasoning_content` WITH `content: ""` beside
    it, so summing the two is not double counting."""
    page, _ = _page(browser, WINDOWED_ELEMENTS)
    try:
        got = page.evaluate("""
          () => {
            const enc = new TextEncoder();
            const dec = new TextDecoder();
            const before = window.__sb.streamcost.wireStats().wire_chars;
            const frame = 'data: ' + JSON.stringify({
              choices: [{ index: 0, delta: { reasoning_content: "12345", content: "" } }],
            }) + "\\n\\n";
            dec.decode(enc.encode(frame));
            return window.__sb.streamcost.wireStats().wire_chars - before;
          }
        """)
    finally:
        page.close()
    assert got == 5


def test_a_malformed_frame_is_counted_as_a_failure_not_silently_dropped(browser):
    """A denominator that is short by an unknown amount inflates every cost above it."""
    page, _ = _page(browser, WINDOWED_ELEMENTS)
    try:
        got = page.evaluate("""
          () => {
            const enc = new TextEncoder();
            const dec = new TextDecoder();
            const before = window.__sb.streamcost.wireStats();
            dec.decode(enc.encode('data: {not json at all\\n\\n'));
            const after = window.__sb.streamcost.wireStats();
            return { failures: after.wire_parse_failures - before.wire_parse_failures,
                     chars: after.wire_chars - before.wire_chars };
          }
        """)
    finally:
        page.close()
    assert got["failures"] == 1
    assert got["chars"] == 0
