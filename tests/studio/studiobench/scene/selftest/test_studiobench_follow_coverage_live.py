# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE FOLLOW VERDICT COVERED THIRTEEN PERCENT OF THE STREAM AND READ AS A PASS.

`follows_the_stream` exists to stop a windowed renderer producing a flattering streaming cost: if
the thread stops following, the streamed message leaves the viewport, a virtualizer unmounts it,
and the renderer is no longer rendering the thing being measured. The frame rate that comes out is
excellent and is about nothing.

The sampler splits its samples into two phases. ATTACHED, before the harness has scrolled
anywhere: the thread must stay pinned as content arrives, and this is what `pinned_fraction`
scores. DETACHED, after a deliberate scroll: the thread must NOT come back on its own.

`detached` latched on the FIRST `suspend()` and was never cleared. Every scene in the suite calls
`scroll_during_generation` about 1.5 seconds into an opening stream that runs for roughly 18
seconds, and two more streamed turns follow it. So from that first scroll onwards every sample
went to the detached branch, `running_samples` stopped growing, and the verdict was computed from
the first few seconds. Measured on a real 100K cell: `running_samples` 11, `detached_samples` 72.
Thirteen percent coverage, reported as 100% pinned, and quoted as evidence that the arm follows
the stream.

Two fixes, both pinned here. Coming back to the end RE-ATTACHES, because the contract is about
intent and intent is re-expressed by returning. And the coverage travels with the verdict, so the
fraction cannot be read without knowing how much of the stream it describes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"

#: A running thread: the stop button is what `isRunning()` looks for. The viewport is scrollable so
#: "at the bottom" is a real question.
#:
#: The jump-to-bottom control is present and its `invisible` class is kept in sync with the scroll
#: position by a listener, because that class IS the app's own answer to "are we at the bottom" and
#: `appSaysAtBottom()` reads nothing else. Omitting the control makes every sample `running_unknown`
#: and `pinned_fraction` null, which is correct behaviour on a build that renders no control and
#: useless as a fixture for scoring one.
FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 300px; overflow-y: auto; }
  .filler { height: 3000px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp">
    <div class="filler"></div>
    <div data-role="assistant">last</div>
  </div>
  <button aria-label="Stop generating">stop</button>
  <div class="aui-thread-scroll-to-bottom invisible"></div>
</div>
<script>
  // The app's contract, modelled directly: the control's `invisible` class reflects whether the
  // viewport is at the end. Driven by an explicit call rather than a scroll listener, because a
  // programmatic `scrollTop` assignment does not reliably deliver a scroll event to a listener in
  // this fixture, and a test whose setup silently does not run is worse than no test.
  window.__syncJump = () => {
    const vp = document.getElementById("vp");
    const jump = document.querySelector(".aui-thread-scroll-to-bottom");
    const distance = vp.scrollHeight - vp.clientHeight - vp.scrollTop;
    jump.classList.toggle("invisible", distance <= 2);
  };
  window.__syncJump();
</script>
"""


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
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    pg.set_content(FIXTURE)
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.evaluate("() => window.__sb.follow.reset()")
    yield pg
    pg.close()


def _to_bottom(page) -> None:
    page.evaluate(
        "() => { const v = document.getElementById('vp'); v.scrollTop = v.scrollHeight;"
        " window.__syncJump(); }"
    )


def _to_top(page) -> None:
    page.evaluate("() => { document.getElementById('vp').scrollTop = 0; window.__syncJump(); }")


def _settle(page, ms: int = 700) -> None:
    """Long enough for several ticks at FOLLOW_TICK_MS (250ms)."""
    page.wait_for_timeout(ms)


def _read(page) -> dict:
    return page.evaluate("() => window.__sb.follow.read()")


def _start_at_bottom(page) -> None:
    """Position the viewport, THEN zero the counters.

    The sampler ticks every 250ms from the moment the script loads, so a test that scrolls after
    the first tick has already recorded a sample of the pre-scroll state and its `pinned_fraction`
    is 0.5 for reasons that have nothing to do with the code under test.
    """
    _to_bottom(page)
    page.wait_for_timeout(60)
    page.evaluate("() => window.__sb.follow.reset()")


def test_the_sampler_scores_a_pinned_stream(page):
    """The control. Without it the tests below could pass on a sampler that never counts."""
    _start_at_bottom(page)
    _settle(page)
    got = _read(page)
    assert got["running_samples"] > 0, got
    assert got["pinned_fraction"] == 1, got
    assert got["attached_fraction_of_stream"] == 1, got


def test_returning_to_the_end_reattaches_so_the_rest_of_the_stream_is_still_scored(page):
    """THE DEFECT. A deliberate scroll used to detach the sampler permanently, so every sample for
    the remainder of the stream -- and for every later stream in the cell -- was scored against the
    second half of the contract and contributed nothing to the verdict."""
    _start_at_bottom(page)
    _settle(page)
    before = _read(page)["running_samples"]

    # A deliberate gesture that ends back at the bottom, which is what `scroll_after` does and what
    # `scroll_during_generation` does when the user returns.
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    _to_bottom(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page)

    got = _read(page)
    assert got["reattachments"] == 1, got
    assert got["running_samples"] > before, (
        "the sampler stayed detached after a gesture that ended at the bottom, so the rest of the "
        "stream was never scored"
    )
    assert got["attached_fraction_of_stream"] > 0.5, got


def test_a_gesture_that_ends_scrolled_up_does_NOT_reattach(page):
    """The other half of the contract has to survive the fix. A user who scrolled away and stayed
    away is detached, and any pin after that is the app yanking them down."""
    _start_at_bottom(page)
    _settle(page)
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page)
    got = _read(page)
    assert got["reattachments"] == 0, got
    assert got["detached_samples"] > 0, got


def test_the_app_pulling_the_viewport_down_on_its_own_is_not_laundered_into_a_reattachment(page):
    """Re-attachment is only ever evaluated on the way out of a DELIBERATE gesture. If it were
    evaluated on every tick, an app that yanks a scrolled-up user back to the bottom would clear
    `detached` and the yank would be scored as the user following again -- the sampler would
    reward exactly the behaviour it exists to catch."""
    _start_at_bottom(page)
    _settle(page)
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page, 400)
    # The app, not the user, returns the viewport to the end.
    _to_bottom(page)
    _settle(page)
    got = _read(page)
    assert got["reattachments"] == 0, got
    assert got["yanked_back_samples"] > 0, got
    assert got["yanked_after_scroll"] is True, got


def test_the_coverage_travels_with_the_verdict(page):
    """A fraction computed over the attached phases cannot be read without knowing how much of the
    streaming time those phases covered. Reported beside it so the one cannot be quoted alone."""
    _start_at_bottom(page)
    _settle(page, 400)
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page, 1200)
    got = _read(page)
    assert got["pinned_fraction"] == 1, got
    assert got["stream_samples"] == got["running_samples"] + got["detached_samples"], got
    assert got["attached_fraction_of_stream"] < 0.5, (
        got,
        "this cell was detached for most of its stream and the coverage must say so",
    )


# ── the run the user started is a fresh intent to be at the end ─────
#
# THE GESTURE ON THE REAL FILM NEVER ENDS AT THE BOTTOM. `scene/actions.py::SCROLL_JS` jumps to
# the bottom and then steps 14 x 420px away from it, so on any thread taller than 5,880px the
# gesture ends thousands of pixels up and the `resume()` re-attachment above cannot fire. The film
# then starts two more runs of its own -- `stop_generation` and `send_turn` both submit a turn --
# and the app pins to the bottom for them, which `runtime/session.py` already documents as
# intended rather than a violation.
#
# Measured at head over every 100K payload in `outputs/`: attached_fraction_of_stream 0.07 to 0.15
# with reattachments 0, on the BASE arm as well as the treatment and on pure null controls, so the
# gate's FOLLOW_MIN_STREAM_COVERAGE of 0.50 failed every 100K cell of every run -- two copies of
# the shipped build included -- and a failed gate excludes the cell from scoring entirely. It
# passed only on the 1K smoke film, where the thread is short enough that the gesture's reversal
# lands back at the bottom by accident: a verdict about the thread's height, not about the app.


def _end_run(page) -> None:
    """The reply finishes: the stop control goes, so `isRunning()` is false."""
    page.evaluate(
        "() => { const b = document.querySelector('button[aria-label=\"Stop generating\"]');"
        " if (b) b.setAttribute('aria-label', 'idle'); }"
    )


def _start_run(page) -> None:
    """The user submits another turn: a NEW run, which the app pins to the bottom for."""
    page.evaluate(
        "() => { const b = document.querySelector('button[aria-label=\"idle\"]');"
        " if (b) b.setAttribute('aria-label', 'Stop generating'); }"
    )


def test_a_run_the_user_started_reattaches_when_the_app_pins_for_it(page):
    """THE DEFECT. The verdict covered the first few seconds of the opening stream and nothing
    else, because every later turn of the film was scored against the yank half of the contract."""
    _start_at_bottom(page)
    _settle(page, 400)

    # The harness scrolls away mid-stream and stays away, exactly as SCROLL_JS leaves it.
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page, 500)
    assert _read(page)["reattachments"] == 0, _read(page)

    # That reply finishes; the user sends another turn and the app pins to the bottom for it.
    _end_run(page)
    _settle(page, 300)
    _start_run(page)
    _to_bottom(page)
    _settle(page, 800)

    got = _read(page)
    assert got["reattachments"] == 1, got
    assert got["attached_fraction_of_stream"] > 0.5, (
        got,
        "the follow-up turn streamed with the thread pinned at the bottom and none of it was "
        "scored, so the coverage gate fails a build that followed perfectly",
    )


def test_a_pin_inside_the_SAME_run_is_still_a_yank_and_not_a_reattachment(page):
    """The other half has to survive. No new run started, so the app returning the viewport to the
    bottom on its own is the behaviour the sampler exists to catch, not the user coming back."""
    _start_at_bottom(page)
    _settle(page)
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page, 400)
    _to_bottom(page)
    _settle(page)
    got = _read(page)
    assert got["reattachments"] == 0, got
    assert got["yanked_back_samples"] > 0, got
    assert got["yanked_after_scroll"] is True, got


def test_a_new_run_the_app_does_NOT_pin_for_leaves_the_sampler_detached(page):
    """A run start is not a free pass. The re-attachment needs the viewport to actually BE at the
    end, so an app that leaves a scrolled-up user where they are is scored exactly as before."""
    _start_at_bottom(page)
    _settle(page, 400)
    page.evaluate("() => window.__sb.follow.suspend()")
    _to_top(page)
    page.evaluate("() => window.__sb.follow.resume()")
    _settle(page, 400)
    _end_run(page)
    _settle(page, 300)
    _start_run(page)
    _settle(page, 800)
    got = _read(page)
    assert got["reattachments"] == 0, got
    assert got["attached_fraction_of_stream"] < 0.5, got
