# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE POSITIVE CONTROL ON THE STREAMING PROBE, read off a real DOM instead of a fixture dict.

`in_flight_unplaced` says "the app says a reply is running and not one message published a
streaming state, so the probe has gone blind". It is checked offline in
fixture/selftest/test_studiobench_parity_streamed.py by handing `compare()` a capture with the flag
already set, which proves what the flag DOES and nothing at all about when `capture()` sets it --
and when it is set is the whole of its value, because a control that fires on an ordinary settled
thread costs exactly the coverage it was added to protect.

The state that fires it wrongly is QUEUED-IDLE: a prompt waiting in the queue while the thread is
doing nothing. `ComposerRightControls` renders `aria-label="Queue message"` there, under
`isQueueRunning && !thread.isRunning`, and it renders the same button on a RUNNING thread whose
composer has text -- `queueDisabled` swaps Stop for Queue. So the button alone cannot tell a
waiting queue from a live stream, and `dom.isRunning()`, which accepts it because every action that
asks "may I send now" needs the broad reading, cannot either.

Three states, all of them rendered from the shipped markup, and the claim is about which pairs of
them a reader can tell apart:

    STREAMING     a reply is being written; the last message publishes `data-status="running"`.
    QUEUED-IDLE   a prompt waits in the queue, nothing is being written, every message is settled.
    BLIND         a reply really is being written and the `data-status` hook was renamed, so
                  `streamingMessages()` matches nothing. This is what the control exists for.

QUEUED-IDLE and BLIND are the pair that matters. Reading the queue button as "running" makes them
the SAME observation -- both `in_flight_unplaced: true` -- and `compare()` then refuses a settled
queued-idle pair, in front of its settled digests, as a broken instrument. That refusal is not a
pass, but it is not a reading either, and a real difference elsewhere in the thread is lost with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.analysis import parity as P  # noqa: E402

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"
_PARITY_JS = _STUDIO_TESTS / "studiobench" / "scene" / "parity.js"
_THREAD_TSX = (
    _STUDIO_TESTS.parents[1] / "studio" / "frontend" / "src" / "components" / "assistant-ui"
    / "thread.tsx"
)

#: The running branch: `AuiIf thread.isRunning` renders Queue instead of Stop once the composer
#: holds a queueable prompt, wrapped in a positioning div.
_QUEUE_BUTTON_RUNNING = """
<div class="ml-1.5 flex items-center">
  <button class="aui-composer-send size-9 rounded-full" aria-label="Queue message">
    <span class="aui-sr-only">Queue message</span>
  </button>
</div>
"""

#: The queued branch: `isQueueRunning && !thread.isRunning`, with the active item undispatched.
_QUEUE_BUTTON_IDLE = """
<button class="aui-composer-send ml-1.5 size-9 rounded-full" aria-label="Queue message">
  <span class="aui-sr-only">Queue message</span>
</button>
"""

_STOP_BUTTON = """
<button class="aui-composer-cancel size-9 rounded-full" aria-label="Stop generating">
  <span class="aui-sr-only">Stop generating</span>
</button>
"""

#: `PromptQueueStack`, which renders inside the composer root whenever the run has an item left to
#: show. An undispatched active item is always one of them, so this surface is up in every state
#: that renders the queued-idle Queue button.
_QUEUE_STACK = """
<div aria-label="Prompt queue, 1 of 2">
  <div>a prompt that has not been dispatched</div>
</div>
"""


def _page(*, control: str, queue_stack: bool, statuses: list[str | None], tail: str) -> str:
    """A thread, a composer, and one composer control. `statuses` is per assistant message, and
    None renames the hook the probe walks -- which is what going blind looks like."""
    messages = []
    for i, status in enumerate(statuses):
        role = "user" if i % 2 == 0 else "assistant"
        body = "the prompt" if role == "user" else f"reply {i} {tail if i == len(statuses) - 1 else ''}"
        attr = 'data-state="running"' if status is None else f'data-status="{status}"'
        messages.append(f'<div data-role="{role}"><div {attr}>{body}</div></div>')
    return f"""<!doctype html><meta charset="utf-8">
<div class="aui-thread-root">
  <div class="aui-thread-viewport">{"".join(messages)}</div>
  <div class="aui-composer-root">
    {_QUEUE_STACK if queue_stack else ""}
    <textarea aria-label="Message input">typed while it ran</textarea>
    {control}
  </div>
</div>"""


STREAMING = dict(
    control = _QUEUE_BUTTON_RUNNING,
    queue_stack = False,
    statuses = ["complete", "complete", "complete", "running"],
)
QUEUED_IDLE = dict(
    control = _QUEUE_BUTTON_IDLE,
    queue_stack = True,
    statuses = ["complete", "complete", "complete", "complete"],
)
BLIND = dict(
    control = _STOP_BUTTON,
    queue_stack = False,
    statuses = [None, None, None, None],
)
#: The cost of reading the queue surface, pinned rather than left for a later reader to find: a
#: queue run with a prompt still waiting, a reply genuinely streaming, and text in the composer.
#: The surface is up and the Queue button is the only control, so the control is not armed.
QUEUED_AND_STREAMING_BLIND = dict(
    control = _QUEUE_BUTTON_RUNNING,
    queue_stack = True,
    statuses = [None, None, None, None],
)


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
    pg = browser.new_page(viewport = {"width": 900, "height": 700})
    yield pg
    pg.close()


def _capture(page, state: dict, tail: str = "settled") -> dict:
    page.set_content(_page(tail = tail, **state))
    # After the content, not before it: `set_content` does not reliably run init scripts, and the
    # symptom is `window.__sb` simply not existing, which reads like a broken instrument.
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    got = page.evaluate("() => window.__sb.parity.capture()")
    assert got.get("parity_attempted") is True, got
    return got


def _reading(cap: dict) -> dict:
    return {k: cap.get(k) for k in ("streaming", "in_flight", "in_flight_unplaced", "queued_idle")}


# ── the two states the instrument has to tell apart ──────────────────


def test_a_waiting_queue_is_not_read_as_a_blind_probe(page):
    """THE REGRESSION. Same button, same empty in-flight list, opposite meanings."""
    idle = _capture(page, QUEUED_IDLE)
    blind = _capture(page, BLIND)
    # Both refuse a fresh send, which is what `isRunning()` is asked and why it cannot decide this.
    assert idle["streaming"] is True and blind["streaming"] is True
    assert idle["in_flight"] == [] and blind["in_flight"] == []
    # And they are still two different readings.
    assert _reading(idle) != _reading(blind)
    assert idle["in_flight_unplaced"] is False, idle
    assert idle["queued_idle"] is True, idle
    assert blind["in_flight_unplaced"] is True, blind
    assert blind["queued_idle"] is False, blind


def test_a_settled_queued_idle_pair_is_scored_rather_than_refused(page):
    """What the conflation cost, at the level of the verdict.

    Two queued-idle arms whose settled threads genuinely differ. Reading the queue button as a
    running reply sets the control on both, and `streaming_probe` refuses the pair inside
    `compare()` BEFORE the per-message digests are reached, so the difference is never localised.
    """
    base = _capture(page, QUEUED_IDLE, tail = "alpha")
    treat = _capture(page, QUEUED_IDLE, tail = "omega")
    assert base["digest"] != treat["digest"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any(m.startswith("msg3(") for m in got["moved"]), got["moved"]
    # The pair the control IS for still refuses, so this is a narrowing and not a removal.
    refused = P.compare(base, _capture(page, BLIND))
    assert refused["verdict"] == P.NOT_COMPARABLE
    assert "could not be identified" in refused["reason"]


def test_a_real_stream_is_still_placed_and_still_scored(page):
    """The coverage this must not cost: the running branch renders the SAME button."""
    cap = _capture(page, STREAMING)
    assert cap["streaming"] is True
    assert cap["in_flight"] == [3], cap
    assert cap["in_flight_unplaced"] is False and cap["queued_idle"] is False


def test_what_reading_the_queue_surface_gives_up(page):
    """A queue with a prompt still waiting, a live stream, and text in the composer.

    The surface is up and Stop is not rendered, so this reads as queued-idle and the control is not
    armed for this capture. Under-claiming rather than over-claiming, and bounded: a probe goes
    blind by a renamed selector, which is global, so the control still fires on the run's other
    captures. Pinned so the cost is a known one.
    """
    cap = _capture(page, QUEUED_AND_STREAMING_BLIND)
    assert cap["in_flight"] == []
    assert cap["in_flight_unplaced"] is False, cap
    assert cap["queued_idle"] is True, cap


# ── the fixtures are the app's markup, not the test's ────────────────


def test_the_shipped_composer_still_renders_the_two_queue_buttons():
    """The fixtures above are hand-written, so they can drift into asserting themselves.

    What makes them a claim about Studio is that the app still renders BOTH Queue buttons and still
    names the queue surface, so this reads that out of the shipped TSX. If Studio stops rendering
    the queued-idle button the conflation is gone and this file should go with it; if it renames
    the queue surface, `dom.promptQueue()` goes quiet and the conflation is back.
    """
    if not _THREAD_TSX.exists():
        pytest.skip(f"the shipped composer is not in this checkout: {_THREAD_TSX}")
    src = _THREAD_TSX.read_text(encoding = "utf-8")
    assert src.count('aria-label="Queue message"') == 2, (
        "ComposerRightControls no longer renders the Queue button in exactly two places; "
        "re-read which of them can appear on an idle thread"
    )
    assert "aria-label={`Prompt queue, ${current} of ${total}`}" in src, (
        "PromptQueueStack no longer names itself, so dom.promptQueue() matches nothing and the "
        "queued-idle interval is indistinguishable again"
    )
    assert 'aria-label="Stop queued message"' in src
