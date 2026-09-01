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
    _STUDIO_TESTS.parents[1]
    / "studio"
    / "frontend"
    / "src"
    / "components"
    / "assistant-ui"
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

#: The DISPATCHED queued branch: `isQueueRunning && !thread.isRunning` with `queueEntry.dispatched`,
#: which renders a stop control the thread does not report itself running behind. Neither
#: `stopButton()` nor `queueButton()` matches it, so on its own it reads exactly like a settled
#: composer. `getPromptQueueUIItemsForRun` drops dispatched items, so the queue surface can be gone
#: here too -- which is why this state has to be recognised from its own control.
_STOP_QUEUED_BUTTON = """
<button class="aui-composer-cancel ml-1.5 size-9 rounded-full" aria-label="Stop queued message">
  <span class="aui-sr-only">Stop queued message</span>
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


def _page(
    *,
    control: str,
    queue_stack: bool,
    statuses: list[str | None],
    tail: str,
    overlay: str = "",
) -> str:
    """A thread, a composer, and one composer control. `statuses` is per assistant message, and
    None renames the hook the probe walks -- which is what going blind looks like.

    THE HOOK IS ON ASSISTANT ROWS ONLY, and the rename is too, because that is where it lives:
    only assistant parts are rendered through `MarkdownText`, which is the single component that
    emits `<div data-status={status.type}>` (thread.tsx `ASSISTANT_PART_COMPONENTS`), and
    `dom.statusHookPresent` is scoped to assistant messages for exactly that reason. A fixture that
    also moved the attribute on the USER rows would make going blind change a surface the real
    rename cannot touch, and any rule that reads settled user rows would score the fixture's own
    artefact rather than the build.
    """
    messages = []
    for i, status in enumerate(statuses):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "user":
            messages.append('<div data-role="user"><div>the prompt</div></div>')
            continue
        body = f"reply {i} {tail if i == len(statuses) - 1 else ''}"
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
</div>{overlay}"""


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


def _capture(
    page,
    state: dict,
    tail: str = "settled",
) -> dict:
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

    What makes them a claim about Unsloth is that the app still renders BOTH Queue buttons and still
    names the queue surface, so this reads that out of the shipped TSX. If Unsloth stops rendering
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


# ── what the blind-probe refusal may NOT take out with it ────────────

#: An overlay is walked from `document`, OUTSIDE `.aui-thread-root`. Its digest therefore carries
#: neither the streamed message nor the composer, which is what makes it readable on a pair whose
#: stream could not be placed.
_MENU = '<div role="menu"><div class="item">Rename</div></div>'
_MENU_CHANGED = '<div role="menu"><div class="item">Rename thread</div></div>'
#: The composer of a thread that is NOT generating. `_STOP_BUTTON` is the same composer generating.
_SEND_BUTTON = (
    '<button class="aui-composer-send" aria-label="Send message">'
    '<span class="aui-sr-only">Send message</span></button>'
)


def _capture_html(page, html: str) -> dict:
    """`_capture`, for a page built outside the `STATE` dictionaries."""
    page.set_content(html)
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    got = page.evaluate("() => window.__sb.parity.capture()")
    assert got.get("parity_attempted") is True, got
    return got


def test_an_overlay_difference_survives_the_blind_probe_refusal(page):
    """A menu that changed while the stream could not be placed is still a finding.

    Without this the refusal took it out, and `structural_report` buckets a refusal as blind and
    never consults it for the exit code, so a real menu or dialog regression went green.
    """
    base = _capture_html(page, _page(tail = "same", overlay = _MENU, **BLIND))
    treat = _capture_html(page, _page(tail = "same", overlay = _MENU_CHANGED, **BLIND))
    assert base["in_flight_unplaced"] is True and treat["in_flight_unplaced"] is True
    assert len(base["overlays"]) == len(treat["overlays"]) == 1
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert len(got["moved"]) == 1 and got["moved"][0].startswith('overlay0[[role="menu"]]'), got
    assert "walked outside the thread root" in got["reason"]


def test_a_settled_user_row_survives_the_blind_probe_refusal(page):
    """A user row cannot be the reply being written, so the refusal may not take it out either.

    Read off the real DOM rather than a fixture dict, and on the state the control is actually for:
    both arms generating with the hook renamed. Only assistant parts publish `data-status`, so the
    rename does not touch this row -- what moved it is the build.

    Without this it left as NOT COMPARABLE with an empty `moved`, and `report` buckets a refusal as
    blind and never consults it for the exit code, so the run went green on it.
    """
    base = _capture_html(page, _page(tail = "same", **BLIND))
    treat = _capture_html(
        page,
        _page(tail = "same", **BLIND).replace(
            "<div>the prompt</div>",
            "<div>the prompt, rendered differently</div>",
            1,
        ),
    )
    assert base["in_flight_unplaced"] is True and treat["in_flight_unplaced"] is True
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any(m.startswith("msg0(user)") for m in got["moved"]), got["moved"]


def test_matching_overlays_still_leave_the_blind_pair_refused(page):
    """The narrowing is not a hole: with nothing independent to say, the refusal stands."""
    base = _capture_html(page, _page(tail = "alpha", overlay = _MENU, **BLIND))
    treat = _capture_html(page, _page(tail = "omega", overlay = _MENU, **BLIND))
    assert base["digest"] != treat["digest"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []


def test_the_scaffold_is_not_an_independent_surface_and_here_is_why(page):
    """WHY the scaffold is deliberately NOT consulted beside the overlays.

    `ThreadPrimitive.Root` wraps `ThreadComposerDock` (thread.tsx), so the composer is inside
    `.aui-thread-root` and inside the scaffold, and the composer is exactly what changes when a
    reply starts and stops. On the pair the refusal is about -- one arm generating with a quiet
    hook, the other finished -- the scaffold therefore differs BECAUSE one arm is generating, and
    reporting that as a rendering difference would manufacture the wall-clock false alarm this file
    exists to remove.

    Two threads with identical messages, differing only in the composer control.
    """
    settled = dict(QUEUED_IDLE, control = _SEND_BUTTON, queue_stack = False)
    generating = dict(settled, control = _STOP_BUTTON)
    a = _capture_html(page, _page(tail = "same", **settled))
    b = _capture_html(page, _page(tail = "same", **generating))
    assert [m["digest"] for m in a["messages"]] == [m["digest"] for m in b["messages"]]
    assert a["digest_scaffold"] != b["digest_scaffold"], (
        "if this ever holds, the composer has left the thread root and the scaffold may be "
        "consulted beside the overlays"
    )


# ── the composer is not a rendering difference ───────────────────────
#
# The pair this whole change is about: one arm has finished its reply, the other is still writing
# it. Its messages are withheld correctly. Its COMPOSER was not, because the dock is inside
# `.aui-thread-root` and the scaffold therefore carries Stop on one arm and Send on the other.

_SETTLED_STATUSES = ["complete", "complete", "complete", "complete"]
_STREAMING_STATUSES = ["complete", "complete", "complete", "running"]


def _finished(**kw):
    return dict(control = _SEND_BUTTON, queue_stack = False, statuses = _SETTLED_STATUSES, **kw)


def _writing(**kw):
    return dict(control = _STOP_BUTTON, queue_stack = False, statuses = _STREAMING_STATUSES, **kw)


def test_a_scaffold_only_difference_across_a_finished_and_a_running_arm_is_refused(page):
    """THE REGRESSION, on the pair this mode exists for.

    Every settled message row is byte-identical; the only thing that moved is the composer, and it
    moved because one arm was generating. Reported as DIFFER this read as a rendering change, with
    the single claim `thread scaffolding outside any message (373->381c)`.
    """
    base = _capture_html(page, _page(tail = "arrived at last", **_finished()))
    treat = _capture_html(page, _page(tail = "arr", **_writing()))
    assert base["streaming"] is False and treat["streaming"] is True
    assert treat["in_flight"] == [3] and treat["in_flight_unplaced"] is False
    assert [m["digest"] for m in base["messages"][:3]] == [
        m["digest"] for m in treat["messages"][:3]
    ]
    assert base["digest_scaffold"] != treat["digest_scaffold"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "composer dock is inside the thread root" in got["reason"]
    assert got["moved"] == []


def test_the_same_two_stream_positions_with_both_arms_running_are_unchanged(page):
    """WHY THE NULL COULD NOT SEE IT. One build against itself at two points in one stream: both
    arms render Stop, the scaffolds match, and the bias cancels inside the control."""
    a = _capture_html(page, _page(tail = "arrived at last", **_writing()))
    b = _capture_html(page, _page(tail = "arr", **_writing()))
    assert a["digest_scaffold"] == b["digest_scaffold"]
    assert P.compare(a, b)["verdict"] == P.NOT_COMPARABLE


def test_a_real_message_difference_is_still_reported_across_a_generation_disagreement(page):
    """The withholding is not a blanket. It applies only when the scaffold is the ONLY thing that
    moved; a settled message that differs is reported exactly as before."""
    base = _capture_html(page, _page(tail = "arrived at last", **_finished()))
    treat_statuses = list(_STREAMING_STATUSES)
    treat = _capture_html(
        page,
        _page(
            tail = "arr",
            control = _STOP_BUTTON,
            queue_stack = False,
            statuses = treat_statuses,
            overlay = "",
        ),
    )
    # Change a SETTLED message on the treatment arm as well.
    treat2 = _capture_html(
        page,
        _page(
            tail = "arr",
            control = _STOP_BUTTON,
            queue_stack = False,
            statuses = treat_statuses,
        ).replace("reply 1 ", "reply 1 rewritten "),
    )
    assert treat["messages"][1]["digest"] != treat2["messages"][1]["digest"]
    got = P.compare(base, treat2)
    assert got["verdict"] == P.DIFFER, got
    assert any(m.startswith("msg1(") for m in got["moved"]), got["moved"]


def test_a_scaffold_difference_with_both_arms_agreeing_is_still_a_difference(page):
    """And the coverage this must not cost: when the arms agree about generation, the composer is
    comparable again and a scaffolding change is reported."""
    base = _capture_html(page, _page(tail = "same", **_finished()))
    treat = _capture_html(
        page,
        _page(tail = "same", **_finished()).replace(
            ">typed while it ran<",
            ">a different draft left in the box<",
        ),
    )
    assert base["streaming"] == treat["streaming"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == [
        "thread scaffolding outside any message (%d->%dc)"
        % (base["chars_scaffold"], treat["chars_scaffold"])
    ], got["moved"]


def test_the_blind_branch_reads_the_scaffold_when_the_arms_agree_about_generation(page):
    """The correct form of the scaffold half of the earlier review item.

    Refused there because the composer is a function of generation. When both arms are generating,
    that objection does not apply and a scaffolding change is a finding even though the stream
    could not be placed.
    """
    base = _capture_html(page, _page(tail = "same", **BLIND))
    treat = _capture_html(
        page,
        _page(tail = "same", **BLIND).replace(
            '<textarea aria-label="Message input">typed while it ran</textarea>',
            '<textarea aria-label="Message input">typed while it ran, differently</textarea>',
        ),
    )
    assert base["in_flight_unplaced"] is True and treat["in_flight_unplaced"] is True
    assert base["streaming"] == treat["streaming"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any("thread scaffolding" in m for m in got["moved"]), got["moved"]


#: The composer regression this branch has to report rather than excuse: the run-state slot is
#: empty because the treatment dropped its Send button. `runStateControl` finds none of the six
#: controls and returns "", so `composer_control` differs while the thread's own run state does
#: not.
_NO_CONTROL = '<div class="ml-1.5 flex items-center"></div>'


def test_a_composer_regression_between_two_settled_arms_is_reported(page):
    """THE CIRCLE. `generation_disagrees` reads `composer_control`, so a composer that regressed
    supplied its own excuse: the refusal said the arms were at different points in the turn on the
    authority of the very surface whose difference was in question.

    Both arms here are settled. Nothing is generating, nothing is queued, every message and overlay
    agrees, and the treatment simply has no Send button. That is as plain a rendering regression as
    this tool can be shown, and NOT COMPARABLE would take it out of the exit code entirely, since
    `report` files a refusal under `blind` and scores only `stable_bad or one_sided`.
    """
    base = _capture_html(page, _page(tail = "same", **_finished()))
    treat = _capture_html(
        page,
        _page(tail = "same", **dict(_finished(), control = _NO_CONTROL)),
    )
    # The run state agrees on both independent readings; only the composer moved.
    assert base["streaming"] is False and treat["streaming"] is False
    assert bool(base["queued_idle"]) is False and bool(treat["queued_idle"]) is False
    assert base["composer_control"] == "Send message" and treat["composer_control"] == ""
    assert P.generation_disagrees(base, treat) is True
    assert base["digest_scaffold"] != treat["digest_scaffold"]
    assert [m["digest"] for m in base["messages"]] == [m["digest"] for m in treat["messages"]]

    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any("scaffolding" in m for m in got["moved"]), got["moved"]


def test_a_finished_against_a_running_arm_is_still_refused_after_that(page):
    """And the suppression this must not cost: `streaming` disagrees, so the run state itself says
    the two arms were at different points in one turn and the composer is not evidence of a
    change."""
    base = _capture_html(page, _page(tail = "arrived at last", **_finished()))
    treat = _capture_html(page, _page(tail = "arr", **_writing()))
    assert base["streaming"] is False and treat["streaming"] is True
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "composer dock is inside the thread root" in got["reason"]


def test_a_dispatched_queue_wait_is_a_run_state_not_a_rendering_difference(page):
    """THE TRANSIENT THAT READ AS IDLE.

    Once a queued entry is dispatched, `ComposerRightControls` renders "Stop queued message" under
    `isQueueRunning && !thread.isRunning`, so the thread says it is not running and neither control
    `isRunning()` matches is present. Both `streaming` and `queued_idle` therefore came back false,
    which is exactly how a settled Send arm reads.

    An arm caught in that interval against a settled one then had a differing composer and a
    differing scaffold with NO run-state difference to account for it -- and that is precisely the
    shape the comparison layer is entitled to call a rendering regression, now that the scaffold
    suppression requires independent run-state evidence. It is queue timing, so it must refuse.
    """
    dispatched = _capture_html(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = _STOP_QUEUED_BUTTON, queue_stack = False)),
    )
    settled = _capture_html(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = _SEND_BUTTON, queue_stack = False)),
    )
    # The composer really does differ, which is what makes this the interesting pair.
    assert dispatched["composer_control"] != settled["composer_control"]
    # ...and the run state now says why, off the run state rather than off the composer token.
    assert bool(dispatched["queued_idle"]) is True, dispatched
    assert bool(settled["queued_idle"]) is False, settled
    assert P._run_state_disagrees(dispatched, settled) is True

    assert P.compare(dispatched, settled)["verdict"] == P.NOT_COMPARABLE
    assert P.compare(settled, dispatched)["verdict"] == P.NOT_COMPARABLE


def test_the_queued_idle_arm_against_a_settled_one_is_still_refused(page):
    """The other legitimate suppression: `isRunning()` cannot separate queued-idle from settled, so
    `queued_idle` is what carries this pair. Without it the Queue button would read as a
    regression."""
    base = _capture_html(page, _page(tail = "same", **QUEUED_IDLE))
    treat = _capture_html(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = _SEND_BUTTON, queue_stack = False)),
    )
    assert base["composer_control"] != treat["composer_control"]
    assert bool(base["queued_idle"]) != bool(treat["queued_idle"]), (
        base["queued_idle"],
        treat["queued_idle"],
    )
    assert P.compare(base, treat)["verdict"] == P.NOT_COMPARABLE


# ── the style probe walks the run-state control too ──────────────────
#
# `report` now collects the style verdict BEFORE it buckets a structural refusal, because the
# computed-style probe is an independent reading and the refusal is not about it. That makes the
# probe's own reading of the composer swap visible for the first time, and the probe walks
# `button[aria-label="Send message"]` and `button[aria-label="Stop generating"]` as SEPARATE
# selectors whose names go into its signature. So the pairs the refusals above exist to withhold
# arrived at the advisory line instead.


def _capture_html_raw(page, html: str) -> dict:
    """`_capture_html`, keeping `styles.sig` so a test can say WHY the digest moved."""
    page.set_content(html)
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    got = page.evaluate("() => window.__sb.parity.capture({ raw: true })")
    assert got.get("parity_attempted") is True, got
    return got


def _style_values(cap: dict) -> list[str]:
    """The probe's readings with the selector NAMES dropped: just the three properties, in order."""
    return [entry.split(":", 1)[1] for entry in cap["styles"]["sig"].split(";") if ":" in entry]


def test_the_style_probe_does_not_report_a_control_swap_as_a_css_regression(page):
    """THE ADVISORY THAT WAS NOT ABOUT CSS.

    One arm still generating, the other settled: the structural pair is refused as run-state
    timing, and the style probe walks `Stop generating` on one side and `Send message` on the
    other. Its signature carries the selector that matched, so the digest moves while `display`,
    `visibility` and `pointer-events` are IDENTICAL on every element -- asserted here rather than
    asserted about, off the raw signature.
    """
    settled = _capture_html_raw(page, _page(tail = "same", **_finished()))
    writing = _capture_html_raw(page, _page(tail = "same", **_writing()))
    # Same number of elements, same three properties on all of them, different digest.
    assert settled["styles"]["elements"] == writing["styles"]["elements"]
    assert _style_values(settled) == _style_values(writing), (
        _style_values(settled),
        _style_values(writing),
    )
    assert settled["styles"]["digest"] != writing["styles"]["digest"]
    # ...and the run state says why, off the run state rather than off the composer.
    assert settled["streaming"] is False and writing["streaming"] is True
    verdict, reason = P.compare_styles(settled, writing)
    assert verdict == P.NOT_COMPARABLE, (verdict, reason)
    assert "run-state controls" in reason, reason
    assert P.compare(settled, writing)["style_verdict"] == P.NOT_COMPARABLE


@pytest.mark.parametrize("control", [_QUEUE_BUTTON_IDLE, _STOP_QUEUED_BUTTON])
def test_a_queue_control_missing_from_the_selector_list_is_not_a_css_regression(page, control):
    """The other flavour, and it does not even reach the digest.

    Neither queue control is in `STYLE_SELECTORS`, so a queued arm matches one element FEWER than
    a settled one and the probe reports "a different number of elements" -- over a page whose CSS
    is byte-identical. Both the undispatched wait and the dispatched one.
    """
    settled = _capture_html_raw(page, _page(tail = "same", **_finished()))
    queued = _capture_html_raw(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = control, queue_stack = False)),
    )
    assert settled["styles"]["elements"] != queued["styles"]["elements"]
    assert bool(settled["queued_idle"]) != bool(queued["queued_idle"]) or (
        settled["streaming"] != queued["streaming"]
    )
    assert P.compare_styles(settled, queued)[0] == P.NOT_COMPARABLE
    assert P.compare(settled, queued)["style_verdict"] == P.NOT_COMPARABLE


def test_a_real_style_regression_between_two_arms_in_one_run_state_is_still_reported(page):
    """THE COVERAGE THIS MUST NOT COST. Same control on both arms, and the treatment hides the
    viewport from CSS alone -- no structural trace whatever, which is the only thing this probe
    exists to see."""
    base = _capture_html_raw(page, _page(tail = "same", **_finished()))
    treat = _capture_html_raw(
        page,
        "<style>.aui-thread-viewport { visibility: hidden }</style>"
        + _page(tail = "same", **_finished()),
    )
    assert base["digest"] == treat["digest"], "the difference must be CSS only"
    assert _style_values(base) != _style_values(treat)
    verdict, reason = P.compare_styles(base, treat)
    assert verdict == P.DIFFER, (verdict, reason)


def test_a_composer_that_lost_its_control_is_still_a_style_finding(page):
    """AND THE SUPPRESSION IS NOT KEYED ON THE COMPOSER TOKEN ALONE. The treatment simply has no
    Send button: the token differs, the probe matches one element fewer, and the run state agrees
    on both independent readings -- so this is a rendering regression and it is reported."""
    base = _capture_html_raw(page, _page(tail = "same", **_finished()))
    treat = _capture_html_raw(
        page,
        _page(tail = "same", **dict(_finished(), control = _NO_CONTROL)),
    )
    assert base["composer_control"] == "Send message" and treat["composer_control"] == ""
    assert P._run_state_disagrees(base, treat) is False
    verdict, reason = P.compare_styles(base, treat)
    assert verdict == P.DIFFER, (verdict, reason)
    assert "different number of elements" in reason


def test_what_the_style_elision_gives_up(page):
    """Pinned rather than left for a later reader: on a pair that DOES straddle the control swap,
    a genuine CSS regression elsewhere goes with it. The probe reads ONE aggregate digest over
    every matched element, so the swap cannot be separated from anything else inside it. The
    verdict is a refusal and not a MATCH, so nothing here reads as a pass."""
    settled = _capture_html_raw(page, _page(tail = "same", **_finished()))
    writing_and_broken = _capture_html_raw(
        page,
        "<style>.aui-thread-viewport { visibility: hidden }</style>"
        + _page(tail = "same", **_writing()),
    )
    assert P.compare_styles(settled, writing_and_broken)[0] == P.NOT_COMPARABLE
