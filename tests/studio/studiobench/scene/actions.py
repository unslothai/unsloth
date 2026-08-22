# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fifteen actions, each with an `expect` that proves it actually happened.

THE RULE THIS FILE EXISTS TO ENFORCE: an action that did not happen is `ran = False`. It is NEVER
a fast timing. That is not a style preference, it is the failure mode that wasted a day of
measurement. A menu whose trigger opens on `pointerdown` does not open when you call `.click()`,
and the column then reads a tidy small number that looks like a fast menu. A jump scroll from the
bottom is read by Studio's intent-aware autoscroll as programmatic and snapped straight back, so
the viewport lands where it started and the timing is real, precise and about nothing. Every
action below therefore asserts a POSITIVE observable consequence -- the scroll travelled at least
90% of what was commanded, the menu opened AND closed with a non-zero item count, the delete
dropped the `[data-role]` count -- and reports `ran = False` with a reason when it cannot.

The JS is ported from `tests/studio/playwright_heavy_thread.py` (KEYSTROKE_JS, SCROLL_JS, JUMP_JS,
MENU_JS, DELETE_JS, REOPEN_JS, PAINT_FLOOR_JS) with `window.__heavyThread` replaced by
`window.__sb.dom`, which is the same API backed by the shipping app's own selectors.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..runtime.types import ActionContext, ActionResult, not_run
from . import register_action

SETTLE_TIMEOUT_MS = 8000

#: How long an action that needs the message ACTION BAR waits for it before reporting NOT RUN.
#:
#: The bar is mounted `hideWhenRunning`, so Copy, Delete and More do not exist while the thread is
#: generating. Every film opens these slots after a `send_turn`, spaced by the NOMINAL drain of the
#: follow-up turn -- FOLLOW_UP_CHARS over the field cadence, 4.59 s. The nominal drain is a floor
#: rather than an estimate: it assumes the pacer is the binding constraint, and at the 100K rung it
#: is not. Measured over six 100K cells here, the follow-up streamed for 4.4 to 4.7 s AFTER the
#: send window closed, so the reply settles within a few hundred milliseconds of the slot on a
#: quiet machine and a little after it on a loaded one. The CI payload of the run that failed the
#: liveness gate shows exactly that: one more SSE chunk arrived INSIDE the `message_menu` window
#: and the reply stopped growing inside it too.
#:
#: 1,500 ms, and deliberately NOT `ctx.budget_ms`. The remaining budget at entry is whatever the
#: previous window's teardown left over -- measured at 51 ms on one of those cells -- so a wait
#: capped by it is a wait that is not taken in exactly the conditions that need it. It is bounded
#: so a control that is genuinely absent still reports NOT RUN rather than eating the film: an
#: action that overruns pushes nothing (every slot has an absolute start) and the overrun is
#: recorded as `over_budget_ms` on the row.
ACTION_BAR_WAIT_MS = 1500


def _ev(
    ctx: ActionContext,
    script: str,
    arg: Any = None,
) -> Any:
    """Evaluate, and turn a thrown page error into `ran = False` rather than a lost cell."""
    try:
        return ctx.page.evaluate(script, arg) if arg is not None else ctx.page.evaluate(script)
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"    action js failed: {type(exc).__name__}: {exc}")
        return {"__error": f"{type(exc).__name__}: {exc}"}


def _failed(raw: Any) -> str | None:
    if raw is None:
        return "the page returned nothing"
    if isinstance(raw, dict) and raw.get("__error"):
        return raw["__error"]
    return None


# ── the paint floor ─────────────────────────────────────────────────
#
# Two rAFs resolve no sooner than two vsync intervals, so ANY timing clocked across a double rAF
# has a ~33ms floor under it on a 60Hz display. An action that never happened still reports that
# floor, which reads as a plausible measurement rather than as a failure. Measured per cell and
# recorded so a reader can subtract it.

PAINT_FLOOR_JS = """
async (samples) => {
  const values = [];
  for (let i = 0; i < samples; i += 1) {
    await window.__sbNextPaint();
    const started = performance.now();
    await window.__sbNextPaint();
    values.push(performance.now() - started);
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}
"""


def paint_floor_ms(page, samples: int = 9) -> float | None:
    try:
        return round(page.evaluate(PAINT_FLOOR_JS, samples), 2)
    except Exception:  # noqa: BLE001
        return None


# ── 1. keystroke to paint ───────────────────────────────────────────


@register_action(name = "keystroke", default_budget_ms = 6000)
def keystroke(ctx: ActionContext) -> ActionResult:
    """Type with REAL key events and measure keystroke-to-paint from the page side.

    `page.keyboard.type` goes in through CDP as a real input event: it is hit-tested, routed and
    queued like a user's keypress, and it is the only kind that carries `latencyInfo` in a trace.
    The salvaged KEYSTROKE_JS instead called the native value setter and dispatched a synthetic
    `input` event, which reaches React but enters the pipeline after the queue -- so it cannot
    show input queueing delay at all, and reads cleanest exactly when a user is waiting longest.
    """
    count = int(ctx.args.get("count", 12))
    selector = 'textarea[aria-label="Message input"]'
    if ctx.page.query_selector(selector) is None:
        return not_run("no composer on the page")

    inst = ctx.args.get("_input_instrument")
    armed = (
        inst.arm(selector)
        if inst is not None
        else {"armed": False, "reason": "input instrument not loaded"}
    )
    if not armed.get("armed"):
        return not_run(f"could not arm the input instrument: {armed.get('reason')}")

    ctx.page.click(selector)
    started = time.monotonic()
    # A real delay between characters. Typing with delay=0 sends the whole burst in one CDP
    # message, which the renderer coalesces into a single input event and a single paint, so the
    # measurement collapses to one sample no matter how many characters were sent.
    ctx.page.keyboard.type("a" * count, delay = 60)
    ctx.page.wait_for_timeout(200)
    got = inst.collect(count)
    elapsed_ms = (time.monotonic() - started) * 1000

    grew = got.get("grew_by")
    if not got.get("samples"):
        return not_run(f"no keystroke reached the composer ({got.get('reason', 'no samples')})")
    expect = {
        "commanded_chars": count,
        "measured_keystrokes": got.get("samples"),
        "coalesced": got.get("coalesced"),
        "composer_grew_by": grew,
        "composer_text_length": got.get("text_length"),
    }
    # The composer's VALUE grew, which proves the characters reached the controlled component and
    # not merely the DOM node.
    ok = grew is not None and grew >= count
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = expect,
        timings = {
            "p50_ms": got.get("p50_ms"),
            "p95_ms": got.get("p95_ms"),
            "max_ms": got.get("max_ms"),
            "first_ms": got.get("first_ms"),
            "total_ms": round(elapsed_ms, 1),
        },
        reason = None if ok else f"typed {count} characters but the composer value grew by {grew}",
    )


# ── 2, 3. scrolling ─────────────────────────────────────────────────

SCROLL_JS = """
async ([steps, stepPx, settleMs]) => {
  const D = window.__sb.dom;
  const viewport = D.viewport();
  if (!viewport) return { ran: false, reason: "no thread viewport" };
  // The viewport carries `scroll-smooth`, so each scrollTop write starts an animation and the
  // next read lands mid-flight. Stepping from a TRACKED target with an explicit instant
  // behaviour is what a wheel gesture actually does, and the only way the gesture covers the
  // distance it asks for.
  const bottom = viewport.scrollHeight - viewport.clientHeight;
  if (bottom <= 0) return { ran: false, reason: "the thread is shorter than the viewport" };
  viewport.scrollTo({ top: bottom, behavior: "instant" });
  await window.__sbNextPaint();
  let target = viewport.scrollTop;
  // Reverse at either end rather than stopping. A small thread runs out of travel long before a
  // large one, and a gesture that covers 2,600px at one rung and 8,000px at another is not the
  // same gesture, so the columns would not be comparable.
  let direction = -1;
  let travelled = 0;
  let commanded = 0;
  const started = performance.now();
  for (let i = 0; i < steps; i += 1) {
    if (direction < 0 && target <= 0) direction = 1;
    else if (direction > 0 && target >= bottom) direction = -1;
    const next = Math.min(bottom, Math.max(0, target + direction * stepPx));
    commanded += Math.abs(next - target);
    // The wheel event is what the app's own scroll listeners key off; the scrollTo is what moves
    // the viewport in a headless run with no compositor input. Both, or the intent-aware
    // autoscroll reads the move as programmatic and snaps it back.
    viewport.dispatchEvent(
      new WheelEvent("wheel", { deltaY: direction * stepPx, bubbles: true, cancelable: true }),
    );
    viewport.scrollTo({ top: next, behavior: "instant" });
    await window.__sbNextPaint();
    travelled += Math.abs(viewport.scrollTop - target);
    target = viewport.scrollTop;
  }
  const gestureMs = performance.now() - started;
  const settleStart = performance.now();
  while (performance.now() - settleStart < settleMs) await window.__sbNextPaint();
  return {
    ran: true,
    commandedPx: Math.round(commanded),
    travelledPx: Math.round(travelled),
    gestureMs: Math.round(gestureMs * 10) / 10,
    steps,
    landedAt: Math.round(viewport.scrollTop),
    bottom: Math.round(bottom),
  };
}
"""


def _scroll(ctx: ActionContext, label: str) -> ActionResult:
    steps = int(ctx.args.get("steps", 14))
    step_px = int(ctx.args.get("step_px", 420))
    settle_ms = int(ctx.args.get("settle_ms", 200))
    raw = _ev(ctx, SCROLL_JS, [steps, step_px, settle_ms])
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the scroll did not run"))
    commanded = raw["commandedPx"]
    travelled = raw["travelledPx"]
    # 90% of commanded. Studio replaces assistant-ui's autoscroll with an intent-aware one that
    # snaps a move it reads as programmatic straight back to the bottom: measured on an earlier
    # harness, the gesture landed where it started and the column timed a scroll that did not
    # happen. Travel is the only thing that separates the two.
    ok = commanded > 0 and travelled >= 0.9 * commanded
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "commanded_px": commanded,
            "travelled_px": travelled,
            "travel_fraction": round(travelled / commanded, 3) if commanded else None,
            "landed_at": raw["landedAt"],
            "bottom": raw["bottom"],
            "phase": label,
        },
        timings = {
            "gesture_ms": raw["gestureMs"],
            "per_step_ms": round(raw["gestureMs"] / max(1, steps), 2),
        },
        reason = None
        if ok
        else f"the gesture commanded {commanded}px and the viewport moved {travelled}px, so the "
        f"autoscroll snapped it back and nothing was scrolled",
    )


@register_action(name = "scroll_during_generation", default_budget_ms = 8000)
def scroll_during_generation(ctx: ActionContext) -> ActionResult:
    if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
        return not_run("nothing was generating, so this is not a scroll during generation")
    return _scroll(ctx, "during_generation")


@register_action(name = "scroll_after", default_budget_ms = 8000)
def scroll_after(ctx: ActionContext) -> ActionResult:
    return _scroll(ctx, "after")


# ── 4. reasoning expand / collapse ──────────────────────────────────

REASONING_JS = """
async (timeoutMs) => {
  const D = window.__sb.dom;
  const triggers = D.reasoningTriggers();
  if (triggers.length === 0) return { ran: false, reason: "no reasoning pane in the thread" };
  const before = D.reasoningOpenCount();
  const settle = async (want) => {
    const started = performance.now();
    while (performance.now() - started < timeoutMs) {
      if (D.reasoningOpenCount() === want) return performance.now() - started;
      await window.__sbNextPaint();
    }
    return null;
  };
  // Toggle EVERY pane, not one: the mechanism under investigation scales with how much content
  // is mounted, and opening a single pane in a thread of forty is a constant-size action being
  // reported on an axis of thread length.
  const openStart = performance.now();
  for (const t of triggers) t.click();
  const openedIn = await settle(triggers.length);
  const openMs = openedIn === null ? null : performance.now() - openStart;
  const openCount = D.reasoningOpenCount();
  const spansOpen = document.querySelectorAll("pre span").length;
  const closeStart = performance.now();
  for (const t of D.reasoningTriggers()) t.click();
  const closedIn = await settle(0);
  const closeMs = closedIn === null ? null : performance.now() - closeStart;
  return {
    ran: true,
    panes: triggers.length,
    before,
    openCount,
    afterClose: D.reasoningOpenCount(),
    spansOpen,
    openMs: openMs === null ? null : Math.round(openMs * 10) / 10,
    closeMs: closeMs === null ? null : Math.round(closeMs * 10) / 10,
  };
}
"""


@register_action(name = "reasoning_toggle", default_budget_ms = 12000)
def reasoning_toggle(ctx: ActionContext) -> ActionResult:
    raw = _ev(ctx, REASONING_JS, SETTLE_TIMEOUT_MS)
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the reasoning toggle did not run"))
    # Both directions, and the pane count is read from `data-state` on the Collapsible ROOT.
    # Radix keeps collapsed content mounted for its animation, so a presence check on the content
    # element reports every pane as open and the assertion can never fail.
    ok = (
        raw["openMs"] is not None
        and raw["closeMs"] is not None
        and raw["openCount"] == raw["panes"]
        and raw["afterClose"] == 0
    )
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "panes": raw["panes"],
            "open_after_expand": raw["openCount"],
            "open_after_collapse": raw["afterClose"],
            "highlight_spans_while_open": raw["spansOpen"],
        },
        timings = {"open_ms": raw["openMs"], "close_ms": raw["closeMs"]},
        reason = None
        if ok
        else f"{raw['openCount']} of {raw['panes']} panes opened and "
        f"{raw['afterClose']} were still open after collapsing",
    )


# ── 5. stop generation ──────────────────────────────────────────────

#: What `stop_generation`'s throwaway turn costs in FIXED waits once it starts, on any machine: 80
#: ms for the composer to take the text, 600 ms to let the new turn get going so stop is measured
#: against a live stream rather than a starting one, 400 ms for the chunks already in flight to
#: land, and 200 ms for the cleanup delete to settle. Polling for the turn to start and for the
#: stream to stop is on top of it, so this is a floor.
#:
#: RESERVED OUT OF THE DRAIN WAIT rather than spent past the deadline. The drain wait below is
#: bounded by the slot's remaining budget, so a reply that finishes at the last moment of the slot
#: used to leave nothing at all for the 1.28 s that follows. `stop_generation` closes 300 ms before
#: `scroll_after` opens on the fast film and 500 ms before it on the quick one, and the runner is
#: sequential -- every slot has an absolute start, but nothing enters it until the previous action
#: returns. So the overrun came out of `scroll_after`'s own 1,200 ms window on the fast film and
#: recorded `slot_missed` there, whose reason blames the machine for reaching the slot late when
#: what happened is that this action was still cleaning up its scaffolding.
#:
#: SPLIT, because the two halves are reserved at different moments. The 80 ms that settles the fill
#: is spent BEFORE the turn is sent, so it is already gone by the time the turn-start wait is
#: bounded and a bound that subtracts the whole figure charges it twice.
OWN_TURN_FIXED_AFTER_SEND_MS = 600 + 400 + 200
OWN_TURN_FIXED_MS = 80 + OWN_TURN_FIXED_AFTER_SEND_MS

#: What the SAME throwaway turn costs BEYOND those sleeps: two polls and the driver round trips
#: underneath them, neither of which is free.
#:
#: MEASURED, because the first version of this reserve was not. It was sized from a page shim whose
#: `evaluate` is a Python call and whose clock only moves when `wait_for_timeout` is called, so the
#: waits it recorded were `[100 x29, 80, 600, 400, 200]` and the two polls cost nothing. Driving
#: the same shipped action against real chromium and a real clock, with a page standing in for the
#: five calls it makes, the stretch after the drain costs:
#:
#:   a page that answers instantly            1,394 - 1,451 ms   (13 - 17% over the fixed 1,280)
#:   120 ms to start, 90 ms to stop, 60 ms
#:   to delete -- an unremarkable local app     1,723 - 1,938 ms   (35 - 51% over)
#:
#: The gap is nine to thirteen CDP round trips (43 - 130 ms of pure driver time), the 50 ms
#: granularity of both poll loops, and the app's own latency in answering them. Against a 1,280 ms
#: reserve that put the action 506 - 516 ms past the end of its slot on the fast and quick films,
#: whose gaps before `scroll_after` opens are 300 ms and 500 ms, so `scroll_after` recorded
#: `slot_missed` and its reason blamed the machine.
#:
#: 700 ms covers the measured 443 - 658 ms with margin and still leaves every film a real drain
#: wait: the smallest stop slot is 3,000 ms, which keeps 1,020 ms of waiting after the reserve,
#: and the worst-case drain overhang the fast film has to absorb is 400 ms.
#:
#: SPLIT AT THE MOMENT THE TURN STARTS, because that is where the wait below is bounded and a total
#: is not a bound. The stop-settle poll, the cleanup delete's own round trips and the driver calls
#: between them are ALL still ahead at that point, and a bound that reserves only the fixed sleeps
#: leaves the action exactly `OWN_TURN_FIXED_MS` for a stretch that costs more than that. Measured
#: against real chromium on the same page as the totals above, with the turn starting on the last
#: millisecond the bound allows and the fast film's 3,000 ms stop slot:
#:
#:   a page that answers instantly            60 ms after the sleeps    slot + 0.0 s
#:   90 ms to stop, 60 ms to delete          227 ms after the sleeps    slot + 0.15 s
#:   200 ms to stop, 150 ms to delete        424 ms after the sleeps    slot + 0.34 s
#:
#: The fast film leaves 300 ms between this slot's deadline and `scroll_after`, so the last row
#: overran into `scroll_after`'s own window and was recorded there as `slot_missed` -- the same
#: symptom, one action later, that the reserve above was written to remove. 500 ms covers the
#: measured 60 - 424 ms; the 200 ms left over is the start poll's own share, which is what the
#: turn-start wait is allowed to spend.
OWN_TURN_STOP_POLL_MS = 500
OWN_TURN_START_POLL_MS = 200
OWN_TURN_POLL_MS = OWN_TURN_START_POLL_MS + OWN_TURN_STOP_POLL_MS

#: What the throwaway turn needs in the slot, in total.
#:
#: RESERVED OUT OF THE DRAIN WAIT rather than spent past the deadline. The drain wait below is
#: bounded by the slot's remaining budget, so a reply that finishes at the last moment of the slot
#: used to leave nothing at all for the ~1.7 s that follows. `stop_generation` closes 300 ms before
#: `scroll_after` opens on the fast film and 500 ms before it on the quick one, and the runner is
#: sequential -- every slot has an absolute start, but nothing enters it until the previous action
#: returns. So the overrun came out of `scroll_after`'s own 1,200 ms window on the fast film and
#: recorded `slot_missed` there, whose reason blames the machine for reaching the slot late when
#: what happened is that this action was still cleaning up its scaffolding.
#:
#: A RESERVE IS NOT ENOUGH ON ITS OWN, which is why `stop_generation` also re-reads the clock
#: before it commits. The drain loop tests its deadline at the TOP, so one iteration -- a 100 ms
#: wait plus a round trip -- lands past it, and no constant chosen here can pay for time that has
#: already been spent. The reserve decides how much of the slot the drain may have; the check
#: decides whether what is actually left is enough to start a turn that cannot be abandoned once
#: it is running.
OWN_TURN_RESERVE_MS = OWN_TURN_FIXED_MS + OWN_TURN_POLL_MS

#: How long to wait for the throwaway turn to start, in total. The wait for a turn this action will
#: still MEASURE is bounded by the slot on top of this -- 8 s is 2.7x the whole budget of the fast
#: film's stop slot, so a turn that was slow to start used to overrun by seconds whatever had been
#: reserved for it -- but the turn does not stop existing when the slot runs out.
#:
#: ENTER IS PRESSED BEFORE THIS WAIT, which is what an earlier version of this comment had wrong
#: when it called cutting the wait short free. Something IS committed by then: the send is
#: accepted, the user turn is in the thread, and the reply starts whenever the relay gets round to
#: it. Returning at the slot bound left exactly the scaffolding the paragraph below refuses to
#: leave -- measured in chromium on a page that took 1,800 ms to start against the fast film's
#: 3,000 ms slot, the action returned `not_run` after 1,759 ms and the thread it handed on had two
#: extra messages in it and a live stream running through the next action's window.
#:
#: So the slot bounds how long the turn is WORTH MEASURING, and this bounds how long it is worth
#: waiting for so it can be taken back: `_reclaim_pending_turn` polls on to this deadline, stops
#: the turn if it ever becomes stoppable and deletes it, then reports `not_run`. The total spent
#: waiting for a start is therefore what it was before the slot bound existed, and the slot bound
#: still does its job on every run where the turn arrives.
#:
#: The stop-settle poll and `STOP_CLEANUP_JS` are deliberately NOT bounded by the slot at all: once
#: Enter is pressed the turn has to be stopped and removed or the thread every later action
#: measures keeps the scaffolding, and a settle poll cut short would report a stop that worked as
#: `still_running`.
TURN_START_TIMEOUT_MS = 8000

#: The composer text the throwaway turn is sent with. Read back as well as written: a composer that
#: still holds it is a send the app REFUSED, and that is the one case on the timeout path where
#: nothing was committed and there is nothing to take back.
OWN_TURN_TEXT = "one more"

#: Remove the throwaway turn `stop_generation` created, so the thread it leaves behind is the
#: thread it found. Deletes the assistant turn and then the user turn that prompted it, in that
#: order, because deleting the user message first can take the reply with it and leave the count
#: ambiguous. Reports what it managed rather than asserting: a cleanup that half-worked must be
#: visible in the row, not swallowed.
STOP_CLEANUP_JS = """
async (timeoutMs) => {
  const D = window.__sb.dom;
  const before = D.messageCount();
  const removeLast = async () => {
    const target = D.lastAssistantMessage();
    if (!target) return false;
    target.dispatchEvent(new PointerEvent("pointerover", { bubbles: true, pointerType: "mouse" }));
    const button = D.actionButton("Delete message");
    if (!button) return false;
    const started = performance.now();
    button.click();
    while (performance.now() - started < timeoutMs) {
      if (!target.isConnected) return true;
      await window.__sbNextPaint();
    }
    return false;
  };
  const dropped = await removeLast();
  const after = D.messageCount();
  return {
    removed: dropped && after < before,
    before, after,
    reason: dropped ? null : "no Delete control on the throwaway turn",
  };
}
"""


def _own_turn_was_accepted(ctx: ActionContext, messages_before: Any) -> bool:
    """Did Enter actually SEND the throwaway turn?

    Two signals, either of which is enough, because they become true at different moments and the
    question is asked at whichever one the timeout landed on. The composer clears on send, so text
    still in it is a refusal -- `queueDisabled` turns Send into Queue whenever something is already
    running, and a Queue press leaves the box alone. The thread grows on send too, and it can grow
    before the composer's own re-render lands.

    ASKED IN THE CONSERVATIVE DIRECTION: anything that is not clearly a refusal counts as accepted,
    including a page call that threw, because the cost of treating a refusal as a send is a wait
    this action was going to spend anyway and the cost of the reverse is a live turn left in the
    thread.
    """
    if _ev(ctx, "() => window.__sb.dom.composerText()") != OWN_TURN_TEXT:
        return True
    after = _ev(ctx, "() => window.__sb.dom.messageCount()")
    return not (
        isinstance(messages_before, int) and isinstance(after, int) and after <= messages_before
    )


def _reclaim_pending_turn(
    ctx: ActionContext, messages_before: Any, reason: str, deadline: float
) -> ActionResult:
    """Take back the throwaway turn Enter has ALREADY submitted, then report `not_run`.

    THE SLOT RAN OUT, THE TURN DID NOT. `stop_generation` presses Enter and then waits for the
    reply to start, and that wait is bounded by the slot -- but returning at the bound abandons a
    turn the app has accepted. It starts a second or two later, streams through the windows the
    next actions are being timed in, and leaves a user message and an assistant message in the
    thread that the rest of the film, the final census and the seeded-versus-streamed comparison
    all then measure. That is the same scaffolding `STOP_CLEANUP_JS` exists to remove, arrived at
    by giving up rather than by finishing.

    SO THE WAIT IS SPLIT RATHER THAN SHORTENED. Up to the slot's bound the turn is still worth
    measuring; past it, up to `deadline`, it is only worth catching, and this polls on for it,
    stops it and deletes it. Nothing here is bounded by the slot, for the reason the stop-settle
    poll is not either: an overrun costs the next action's window, and a turn left generating costs
    every window after that plus the census.

    DELETES ONLY WHAT IT ADDED. `STOP_CLEANUP_JS` removes the LAST assistant message, so running it
    when the send was refused or when the app never rendered the turn would take a seeded reply out
    of the thread and every count after this point would be wrong in the other direction. The
    message count is read before Enter and again here, and the delete runs only if the thread grew.
    """
    if not _own_turn_was_accepted(ctx, messages_before):
        return not_run(f"{reason}, and the composer still held it, so nothing was sent")

    running = _ev(ctx, "() => window.__sb.dom.isRunning()")
    while running is not True and time.monotonic() < deadline:
        ctx.page.wait_for_timeout(50)
        running = _ev(ctx, "() => window.__sb.dom.isRunning()")

    stopped = False
    if running is True:
        button = ctx.page.query_selector('button[aria-label="Stop generating"]')
        if button is not None:
            button.click()
            settle_deadline = time.monotonic() + SETTLE_TIMEOUT_MS / 1000.0
            while time.monotonic() < settle_deadline:
                if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
                    stopped = True
                    break
                ctx.page.wait_for_timeout(50)
            # The chunks already in flight, as on the measured path.
            ctx.page.wait_for_timeout(400)

    messages_after = _ev(ctx, "() => window.__sb.dom.messageCount()")
    grew = (
        isinstance(messages_before, int)
        and isinstance(messages_after, int)
        and messages_after > messages_before
    )
    removed = _ev(ctx, STOP_CLEANUP_JS, SETTLE_TIMEOUT_MS) if grew else None
    if removed is not None:
        ctx.page.wait_for_timeout(200)

    # BOTH HALVES, REPORTED SEPARATELY. A turn that was deleted but never stopped and one that was
    # stopped but not deleted leave the film in different states, and a row that says only "cleaned
    # up" cannot be read in the light of either.
    if running is not True:
        state = f"never started within the {TURN_START_TIMEOUT_MS}ms it was worth waiting for"
    elif stopped:
        state = "was stopped"
    else:
        state = f"could not be stopped within {SETTLE_TIMEOUT_MS}ms"
    if removed is None:
        cleaned = "left nothing in the thread to remove"
    elif removed.get("removed"):
        cleaned = "was deleted from the thread"
    else:
        cleaned = f"is still in the thread ({removed.get('reason')})"
    return not_run(
        f"{reason}. Nothing was measured; the turn it had already sent {state} and {cleaned}"
    )


@register_action(name = "stop_generation", default_budget_ms = 8000)
def stop_generation(ctx: ActionContext) -> ActionResult:
    """Press stop mid-stream and time until the run is really over.

    THE COMPOSER MUST BE EMPTY. `queueDisabled` in thread.tsx depends on
    `composerText.trim().length > 0`, so with text in the box the Stop control is replaced by a
    Queue control at the same position with the same class. Pressing it queues a message and the
    stream carries on, and the action reports a fast, precise, entirely wrong number.
    """
    # THE SLOT, ON THE CLOCK THE ACTION ITSELF RUNS ON. Everything below that is bounded is
    # bounded against this rather than against a constant, because a budget spent is not a budget
    # available: `ctx.budget_ms` is what the slot HAD when the runner entered it.
    slot_deadline = time.monotonic() + ctx.budget_ms / 1000.0

    def remaining_ms() -> float:
        return (slot_deadline - time.monotonic()) * 1000.0

    text = _ev(ctx, "() => window.__sb.dom.composerText()")
    if text:
        ctx.page.fill('textarea[aria-label="Message input"]', "")
        ctx.page.wait_for_timeout(120)

    # STOP GETS ITS OWN GENERATION. It used to stop whatever the cell was streaming, which
    # permanently truncated the measured reply: at the 10K rung the pacer got 5,304 of 17,737
    # characters away before the socket broke, so the ten actions that follow -- all of them
    # labelled "after the reply is complete" -- and the final census ran against a thread barely
    # a third of the size the rung claims. The seeded-vs-streamed equivalence check then compared
    # that truncated reply against a complete seeded one and reported 20% drift, which read as a
    # finding about seeding and was a finding about this action.
    #
    # Sending its own short turn keeps the measured reply intact and means the action has
    # something to stop at EVERY rung, instead of reporting `not_run` at the small ones where the
    # main stream is already over by the time the slot opens.
    #
    # THE GUARD BELOW IS THE OTHER HALF OF THAT, and without it the paragraph above only held for
    # the default fixture. The own-turn path was entered only when `isRunning()` was FALSE, so the
    # moment anything WAS running this action fell straight through and clicked Stop on it -- the
    # exact behaviour the paragraph above says was removed. The supported way to make that happen
    # is `--stream-tail-chars`, whose entire purpose is a long opening reply: at 96,000 characters
    # the reply streams for 291 s at field cadence against a 243 s standard film, so this slot
    # opens at 28 s with the opening turn still in flight and kills it at about 9,200 characters.
    # The reply-length axis the flag exists to provide is then never exercised, every later action
    # still runs against a settled thread, `ran` is still true, and `--assert-liveness` -- which
    # the flag's own help text sends the caller to -- still passes. A silently truncated fixture
    # reported as a clean run is the most expensive failure shape this harness has.
    #
    # So the reply is given the rest of this slot's budget LESS what the throwaway turn then costs
    # (`OWN_TURN_RESERVE_MS`) to finish on its own, which is all a marginally slow drain needs (the
    # fast film opens this slot 0.4 s after the worst-case drain on the ladder, and 1.0 s of the
    # fast film's 3 s budget is still left over for it). If it has not finished by then, NOTHING IS
    # STOPPED and the row says why. An honest `not_run` costs a column; stopping the reply costs
    # the whole cell, and spending the rest of the slot draining and THEN starting a turn of our
    # own costs the next slot. With the default tail nothing is ever running when this slot opens
    # -- the packing test in fixture/selftest holds every film to that -- so this path is
    # unreachable on an unmodified run and the behaviour there is exactly what it was.
    if _ev(ctx, "() => window.__sb.dom.isRunning()"):
        drain_ms = max(0.0, ctx.budget_ms - OWN_TURN_RESERVE_MS)
        settle_deadline = time.monotonic() + drain_ms / 1000.0
        running: Any = True
        while time.monotonic() < settle_deadline:
            ctx.page.wait_for_timeout(100)
            running = _ev(ctx, "() => window.__sb.dom.isRunning()")
            if not running:
                break
        if running:
            return not_run(
                "the cell's own reply was still streaming when this slot opened and had not "
                f"drained {drain_ms:.0f}ms later -- this slot's remaining {ctx.budget_ms}ms less "
                f"the {OWN_TURN_RESERVE_MS}ms it then takes to start, stop and remove a turn of "
                "our own. Stopping it would permanently truncate the reply the rest of the film "
                "and the final census measure, so nothing was stopped. Lower --stream-tail-chars "
                "or move this slot past the drain"
            )
        # AND THE RESERVE IS CHECKED AGAINST THE CLOCK, not assumed off the budget. The loop above
        # tests `settle_deadline` at the top, so the iteration that finds the reply drained has
        # already spent a 100 ms wait and a round trip past it; a drain that lands in the last
        # iteration therefore leaves LESS than the reserve however the reserve is sized. Starting
        # the turn anyway is the same overrun by a smaller amount, and the turn cannot be
        # abandoned halfway -- once Enter is pressed the thread has to be stopped and cleaned up
        # or every later action measures the scaffolding.
        if remaining_ms() < OWN_TURN_RESERVE_MS:
            return not_run(
                "the cell's own reply drained with only "
                f"{max(0.0, remaining_ms()):.0f}ms left of this slot's {ctx.budget_ms}ms, and "
                f"starting, stopping and removing a turn of our own takes about "
                f"{OWN_TURN_RESERVE_MS}ms. Running it here would finish inside the next slot and "
                "record a missed slot against it, so nothing was stopped. Lower "
                "--stream-tail-chars or move this slot past the drain"
            )

    # READ BEFORE ENTER, because it is the only way to tell the turn this action added from the
    # thread it was handed. `_reclaim_pending_turn` deletes only if this number grew.
    messages_before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    ctx.page.fill('textarea[aria-label="Message input"]', OWN_TURN_TEXT)
    ctx.page.wait_for_timeout(80)
    ctx.page.keyboard.press("Enter")
    own_generation = True
    sent_at = time.monotonic()
    # HOW LONG THE TURN IS WORTH MEASURING. Bounded by the SLOT as well as by
    # `TURN_START_TIMEOUT_MS`: 8 s is 2.7x the whole stop slot on the fast and quick films, so a
    # turn slow to start overran by seconds no matter what had been reserved for it.
    #
    # WHAT IS RESERVED OUT OF IT is the rest of the turn, not only its sleeps. A turn starting on
    # the last millisecond this allows still has the stop-settle poll, the delete and the driver
    # calls between them ahead of it, and reserving `OWN_TURN_FIXED_MS` left 424 ms of that unpaid
    # against real chromium -- 344 ms past a 3,000 ms slot with 300 ms before `scroll_after` opens,
    # recorded there as a missed slot. `OWN_TURN_FIXED_MS` also counts the 80 ms above, which is
    # already spent. See `OWN_TURN_STOP_POLL_MS`.
    start_wait_ms = max(
        0.0,
        min(
            float(TURN_START_TIMEOUT_MS),
            remaining_ms() - OWN_TURN_FIXED_AFTER_SEND_MS - OWN_TURN_STOP_POLL_MS,
        ),
    )
    deadline = time.monotonic() + start_wait_ms / 1000.0
    # HOW LONG IT IS WORTH WAITING FOR, which is a different question and is NOT bounded by the
    # slot. See `_reclaim_pending_turn`.
    reclaim_deadline = sent_at + TURN_START_TIMEOUT_MS / 1000.0
    started_late = (
        f"nothing was generating and a new turn did not start within {start_wait_ms:.0f}ms"
    )
    while time.monotonic() < deadline:
        if _ev(ctx, "() => window.__sb.dom.isRunning()"):
            break
        ctx.page.wait_for_timeout(50)
    else:
        return _reclaim_pending_turn(ctx, messages_before, started_late, reclaim_deadline)
    if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
        return _reclaim_pending_turn(ctx, messages_before, started_late, reclaim_deadline)
    # Let it get going, so stop is measured against a live stream rather than a starting one.
    ctx.page.wait_for_timeout(600)
    button = ctx.page.query_selector('button[aria-label="Stop generating"]')
    if button is None:
        queue = ctx.page.query_selector('button[aria-label="Queue message"]')
        return not_run(
            "the stop button is not present"
            + (" -- the composer still has text, so it is a Queue button" if queue else "")
        )
    chars_before = _ev(ctx, "() => window.__sb.dom.assistantChars()")
    started = time.monotonic()
    button.click()
    stopped_ms = None
    deadline = started + SETTLE_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
            stopped_ms = (time.monotonic() - started) * 1000
            break
        ctx.page.wait_for_timeout(50)
    ctx.page.wait_for_timeout(400)
    chars_after = _ev(ctx, "() => window.__sb.dom.assistantChars()")
    ok = stopped_ms is not None

    # LEAVE THE THREAD AS WE FOUND IT. The throwaway turn is scaffolding, not content: left in
    # place it adds an assistant message and a reasoning pane that the rest of the film, the final
    # census and the seeded-versus-streamed comparison all then measure. That showed up
    # immediately as "streamed 5 assistant messages vs seeded 4", a drift introduced entirely by
    # this action while it was busy fixing a different one.
    removed = None
    if own_generation:
        removed = _ev(ctx, STOP_CLEANUP_JS, SETTLE_TIMEOUT_MS)
        ctx.page.wait_for_timeout(200)
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "chars_before": chars_before,
            "chars_after": chars_after,
            "still_running": not ok,
            # Which reply was stopped. A reader comparing `chars_added_after_stop` across
            # rungs needs to know whether this was a throwaway turn or the cell's own.
            "own_generation": own_generation,
            # Whether the scaffolding was removed again. Reported rather than asserted: a
            # cleanup that failed leaves an extra turn in the thread, and every census
            # after this point needs to be readable in that light.
            "scaffold_removed": (None if removed is None else bool(removed.get("removed"))),
            "scaffold_note": (None if removed is None else removed.get("reason")),
            # A stop that worked leaves the text where it was, give or take the chunks
            # already in flight. A large jump means the stream ran on.
            "chars_added_after_stop": (
                None if chars_after is None or chars_before is None else chars_after - chars_before
            ),
        },
        timings = {"stop_ms": None if stopped_ms is None else round(stopped_ms, 1)},
        reason = None
        if ok
        else f"the run was still going {SETTLE_TIMEOUT_MS}ms after stop was pressed",
    )


# ── 6. settings ─────────────────────────────────────────────────────


@register_action(name = "settings", default_budget_ms = 12000)
def settings(ctx: ActionContext) -> ActionResult:
    """Open the Settings dialog, scroll its body, close it.

    Opened by navigating to `/settings`, which the app's own route does by calling
    `openDialog()`. Deterministic in a way that clicking a sidebar cog is not: the sidebar can be
    collapsed, and the same aria-label appears on more than one control.
    """
    trigger = ctx.page.query_selector('button[aria-label="Settings"]')
    started = time.monotonic()
    if trigger is not None:
        trigger.click()
    else:
        ctx.page.evaluate("() => window.history.pushState({}, '', '/settings')")
        ctx.page.goto(ctx.args.get("base_url", "") + "/settings", wait_until = "domcontentloaded")
    opened_ms = None
    deadline = started + SETTLE_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        if _ev(ctx, "() => Boolean(window.__sb.dom.settingsDialog())"):
            opened_ms = (time.monotonic() - started) * 1000
            break
        ctx.page.wait_for_timeout(50)
    if opened_ms is None:
        return not_run("the settings dialog never appeared")

    scrolled = _ev(
        ctx,
        """
      async () => {
        const D = window.__sb.dom;
        const el = D.settingsScroller();
        if (!el) return { ran: false, reason: "no scrollable settings body" };
        const bottom = el.scrollHeight - el.clientHeight;
        const started = performance.now();
        el.scrollTo({ top: bottom, behavior: "instant" });
        await window.__sbNextPaint();
        const landed = el.scrollTop;
        el.scrollTo({ top: 0, behavior: "instant" });
        await window.__sbNextPaint();
        return { ran: true, commandedPx: Math.round(bottom), landedAt: Math.round(landed),
                 ms: Math.round((performance.now() - started) * 10) / 10 };
      }
    """,
    )

    close_started = time.monotonic()
    ctx.page.keyboard.press("Escape")
    closed_ms = None
    deadline = close_started + SETTLE_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        if not _ev(ctx, "() => Boolean(window.__sb.dom.settingsDialog())"):
            closed_ms = (time.monotonic() - close_started) * 1000
            break
        ctx.page.wait_for_timeout(50)

    scroll_ok = bool(
        scrolled
        and scrolled.get("ran")
        and (
            scrolled.get("commandedPx", 0) == 0
            or scrolled.get("landedAt", 0) >= 0.9 * scrolled.get("commandedPx", 1)
        )
    )
    ok = closed_ms is not None and scroll_ok
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "opened": True,
            "closed": closed_ms is not None,
            "scroll": scrolled if isinstance(scrolled, dict) else None,
        },
        timings = {
            "open_ms": round(opened_ms, 1),
            "close_ms": None if closed_ms is None else round(closed_ms, 1),
            "scroll_ms": (scrolled or {}).get("ms"),
        },
        reason = None if ok else "the dialog did not close, or its body did not scroll",
    )


# ── 7. model change ─────────────────────────────────────────────────


@register_action(name = "model_change", default_budget_ms = 10000)
def model_change(ctx: ActionContext) -> ActionResult:
    """Open the model picker and select a row.

    THE WEAKEST SELECTOR IN THE SUITE, and it is recorded as such rather than hidden. The picker's
    option rows are plain `<button>` elements with utility classes: no `role="option"`, no
    `data-model-id`, no CommandItem, nothing stable anywhere in features/model-picker. So the row
    is found by position among the menu's buttons and the assertion is on the trigger's LABEL
    changing, which is an observable consequence rather than a selector.
    """
    trigger = ctx.page.query_selector("button.unsloth-model-selector-trigger")
    if trigger is None:
        return not_run("no model selector trigger on the page")
    before = _ev(ctx, "() => window.__sb.dom.currentModelLabel()")
    started = time.monotonic()
    # Click the LABEL, not the trigger's right edge: a `span[data-eject-hit]` sits there and
    # clicking it ejects the model instead of opening the picker.
    trigger.click(position = {"x": 8, "y": 8})
    opened_ms = None
    deadline = started + SETTLE_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        if _ev(ctx, "() => window.__sb.dom.modelOptions().length > 0"):
            opened_ms = (time.monotonic() - started) * 1000
            break
        ctx.page.wait_for_timeout(50)
    if opened_ms is None:
        return not_run("the model picker never opened")
    options = _ev(ctx, "() => window.__sb.dom.modelOptions().map(b => (b.textContent||'').trim())")
    select_started = time.monotonic()
    picked = _ev(
        ctx,
        """
      (name) => {
        const opts = window.__sb.dom.modelOptions();
        const target = opts.find(b => (b.textContent || '').trim() === name) || opts[0];
        if (!target) return null;
        target.click();
        return (target.textContent || '').trim();
      }
    """,
        ctx.args.get("model_name", ""),
    )
    ctx.page.wait_for_timeout(300)
    select_ms = (time.monotonic() - select_started) * 1000
    after = _ev(ctx, "() => window.__sb.dom.currentModelLabel()")
    closed = not _ev(ctx, "() => Boolean(window.__sb.dom.modelMenu())")
    ok = picked is not None and closed
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "options_offered": len(options or []),
            "picked": picked,
            "label_before": before,
            "label_after": after,
            "menu_closed": closed,
            "selector_confidence": "low: the option rows carry no stable attribute",
        },
        timings = {"open_ms": round(opened_ms, 1), "select_ms": round(select_ms, 1)},
        reason = None if ok else "no option could be selected, or the menu stayed open",
    )


# ── 8. composer lengths, then send ──────────────────────────────────


@register_action(name = "composer_fill", default_budget_ms = 10000)
def composer_fill(ctx: ActionContext) -> ActionResult:
    """Short, medium and very long text into the composer, timing each paint.

    A very long paste is its own axis: the composer is an autosizing textarea capped at 12 rows,
    so a 20,000-character paste re-lays-out the whole composer AND the thread beneath it, and the
    app's paste path scans it for attachment-worthy content.
    """
    lengths = ctx.args.get("lengths") or [40, 2000, 20000]
    selector = 'textarea[aria-label="Message input"]'
    if ctx.page.query_selector(selector) is None:
        return not_run("no composer on the page")
    timings: dict = {}
    observed: dict = {}
    for n in lengths:
        text = ("studiobench composer probe " * ((n // 27) + 1))[:n]
        got = _ev(
            ctx,
            """
          async ([sel, text]) => {
            const el = document.querySelector(sel);
            if (!el) return null;
            const setValue = Object.getOwnPropertyDescriptor(
              HTMLTextAreaElement.prototype, "value").set;
            el.focus();
            await window.__sbNextPaint();
            const started = performance.now();
            setValue.call(el, text);
            el.dispatchEvent(new Event("input", { bubbles: true }));
            await window.__sbNextPaint();
            const ms = performance.now() - started;
            return { ms: Math.round(ms * 10) / 10, length: el.value.length,
                     runtimeLength: (window.__sb.dom.composerText() || "").length,
                     rows: Math.round(el.getBoundingClientRect().height) };
          }
        """,
            [selector, text],
        )
        if not isinstance(got, dict) or got.get("ms") is None:
            return not_run(f"the composer did not accept {n} characters")
        timings[f"fill_{n}_ms"] = got["ms"]
        observed[f"length_{n}"] = got["length"]
        observed[f"height_{n}_px"] = got["rows"]
    # The LAST fill is sent, so the send path is exercised with the heaviest composer state.
    send = ctx.page.query_selector('button[aria-label="Send message"]')
    sent = False
    if send is not None and ctx.args.get("send", False):
        send.click()
        ctx.page.wait_for_timeout(300)
        sent = True
    else:
        ctx.page.fill(selector, "")
    ok = all(observed.get(f"length_{n}") == n for n in lengths)
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {**observed, "sent": sent, "lengths_requested": list(lengths)},
        timings = timings,
        reason = None if ok else "the composer did not hold every length it was given",
    )


# ── 9. copy markdown ────────────────────────────────────────────────


@register_action(name = "copy_markdown", default_budget_ms = 6000)
def copy_markdown(ctx: ActionContext) -> ActionResult:
    """The message action bar's Copy, which copies `getCopyText()` for the whole message.

    NOT the More menu: Copy is a top-level icon button and "Export as markdown" in the More menu
    triggers a file download rather than a clipboard write. The action bar has `autohide`, so it
    is hovered first or the button is not in the tree to click.
    """
    # Hover AND WAIT, on the same reasoning as `message_menu`: this bar is `hideWhenRunning`, so a
    # slot that opens with the follow-up's last chunks still arriving finds no Copy button at all.
    # The fixed 150 ms this replaced was a hover settle, not a wait for the reply, and it returned
    # the same "no Copy button" whether the control was missing or merely not mounted yet.
    found = _ev(
        ctx,
        """
      async (waitMs) => {
        const found = await window.__sb.dom.waitForActionButton("Copy", waitMs);
        return { found: Boolean(found.el), waitedMs: found.waitedMs, running: found.running };
      }
    """,
        ACTION_BAR_WAIT_MS,
    )
    if not isinstance(found, dict) or not found.get("found"):
        waited = (found or {}).get("waitedMs") if isinstance(found, dict) else None
        running = (found or {}).get("running") if isinstance(found, dict) else None
        return not_run(
            f"no Copy button on the last assistant message after waiting {waited}ms"
            + (", and the thread was still generating" if running else "")
        )
    started = time.monotonic()
    ok_click = _ev(
        ctx,
        """
      () => {
        // Re-hovered: the wait above ended with a hover, and a re-render between the two
        // evaluates can unmount the bar again.
        window.__sb.dom.hoverLastAssistantMessage();
        const b = window.__sb.dom.actionButton("Copy");
        if (!b) return false;
        b.click();
        return true;
      }
    """,
    )
    if not ok_click:
        return not_run("no Copy button on the last assistant message")
    ctx.page.wait_for_timeout(200)
    elapsed = (time.monotonic() - started) * 1000
    # Read back from the clipboard. Headless Chromium grants clipboard-read only with the
    # permission, which the browser factory requests; if it is not granted the action still RAN,
    # and the assertion says it could not be proved rather than that it failed.
    clip = None
    reason = None
    try:
        clip = ctx.page.evaluate("async () => await navigator.clipboard.readText()")
    except Exception as exc:  # noqa: BLE001
        reason = f"the clipboard could not be read back: {type(exc).__name__}"
    chars = len(clip) if isinstance(clip, str) else None
    ok = chars is not None and chars > 0
    return ActionResult(
        ran = True,
        expect_ok = ok if reason is None else False,
        expect = {"clipboard_chars": chars, "clipboard_readable": reason is None},
        timings = {"copy_ms": round(elapsed, 1)},
        reason = reason or (None if ok else "the clipboard was empty after Copy"),
    )


# ── 10, 11. selection ───────────────────────────────────────────────


@register_action(name = "select_text", default_budget_ms = 6000)
def select_text(ctx: ActionContext) -> ActionResult:
    """Select a range inside the last assistant message. Selection over a large thread forces the
    engine to walk and paint selection geometry across whatever is mounted."""
    raw = _ev(
        ctx,
        """
      async () => {
        const m = window.__sb.dom.lastAssistantMessage();
        if (!m) return { ran: false, reason: "no assistant message" };
        const started = performance.now();
        const range = document.createRange();
        range.selectNodeContents(m);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        await window.__sbNextPaint();
        const text = sel.toString();
        const ms = performance.now() - started;
        sel.removeAllRanges();
        // innerText, NOT textContent, as the denominator. textContent counts text inside
        // collapsed elements -- and a finished reply's reasoning pane is collapsed by default, so
        // most of a reasoning turn's characters are in the tree and not on the screen. A
        // selection can only cover what is rendered, so comparing against textContent failed this
        // assertion on every reasoning-heavy cell while the selection was working perfectly.
        return { ran: true, ms: Math.round(ms * 10) / 10, chars: text.length,
                 visibleChars: (m.innerText || "").length,
                 messageChars: (m.textContent || "").length };
      }
    """,
    )
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "selection did not run"))
    visible = raw.get("visibleChars") or 0
    # NON-EMPTY, and the coverage fraction reported as evidence rather than gated on.
    #
    # "Half the visible characters" looked like a reasonable bar and is not one: innerText
    # collapses runs of whitespace and skips nested scrollers, and Selection.toString applies its
    # own normalisation, so the two counts are not the same quantity even when the selection
    # covers the whole message. Measured on a real reply: 22 selected against 80 "visible", with
    # the selection plainly spanning the entire message. The action's claim is that text in this
    # message got selected; the fraction belongs in the evidence, not in the verdict.
    ok = raw["chars"] > 0
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "selected_chars": raw["chars"],
            "visible_chars": visible,
            "coverage": round(raw["chars"] / visible, 3) if visible else None,
            "message_chars_including_collapsed": raw["messageChars"],
        },
        timings = {"select_ms": raw["ms"]},
        reason = None if ok else "the selection was empty",
    )


@register_action(name = "select_all_copy", default_budget_ms = 10000)
def select_all_copy(ctx: ActionContext) -> ActionResult:
    """Ctrl+A then Ctrl+C over the WHOLE thread. The heaviest selection there is, and a real
    keyboard path so the app's own key handlers run."""
    ctx.page.evaluate("""() => {
        const v = window.__sb.dom.viewport();
        if (v) v.focus({ preventScroll: true });
    }""")
    started = time.monotonic()
    raw = _ev(
        ctx,
        """
      async () => {
        const started = performance.now();
        const sel = window.getSelection();
        sel.removeAllRanges();
        const range = document.createRange();
        const root = window.__sb.dom.viewport() || document.body;
        range.selectNodeContents(root);
        sel.addRange(range);
        await window.__sbNextPaint();
        const chars = sel.toString().length;
        return { selectMs: Math.round((performance.now() - started) * 10) / 10, chars };
      }
    """,
    )
    err = _failed(raw)
    if err:
        return not_run(err)
    copy_started = time.monotonic()
    ctx.page.keyboard.press("Control+C")
    ctx.page.wait_for_timeout(250)
    copy_ms = (time.monotonic() - copy_started) * 1000
    ctx.page.evaluate("() => window.getSelection().removeAllRanges()")
    ok = raw["chars"] > 0
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {"selected_chars": raw["chars"]},
        timings = {
            "select_all_ms": raw["selectMs"],
            "copy_ms": round(copy_ms, 1),
            "total_ms": round((time.monotonic() - started) * 1000, 1),
        },
        # `expect_ok` stays `chars > 0` deliberately. Within ONE run there is nothing to compare
        # against: the selection is taken over the viewport's DOM, so if the DOM is windowed the
        # selection and every DOM-derived reference shrink together and agree with each other.
        # An absolute floor would need calibrating per rung and per platform, and would still be a
        # guess. The count below is the real check, and it is paired against the other arm.
        counts = {"selected_chars": raw["chars"]},
        reason = None if ok else "select-all selected nothing",
    )


# ── 12. image upload ────────────────────────────────────────────────


@register_action(name = "send_turn", default_budget_ms = 10000)
def send_turn(ctx: ActionContext) -> ActionResult:
    """Send another prompt mid-film and let the next reply stream in.

    ONE STREAMED TURN IS NOT A CONVERSATION. Everything above this action measured a thread that
    was seeded in bulk and then streamed exactly once, at the end. A real session streams
    repeatedly into a thread that is already large, and the interesting cost -- what a chunk
    costs given what is already on screen -- is only sampled once per cell that way.

    The extra turns come out of the SAME fixed streaming budget as the first, split between them,
    so adding turns costs no wall clock. The corpus alternates reasoning-heavy and code-heavy
    units, so consecutive sends exercise the `<think>` re-parse and the Streamdown/Shiki path
    rather than repeating one of them.
    """
    pacer = ctx.args.get("_pacer")
    queue = ctx.args.get("_stream_queue") or []
    # A SHARED MUTABLE cursor, not a scalar in `args`. The scene runner builds each action's args
    # as `{**base_args, **slot.args}`, so a scalar written back here lands in a dict that is
    # discarded when the action returns: the second send re-sent the first turn, reported
    # `turn_index: 1` twice, and failed because that message was already in the thread.
    cursor = ctx.args.get("_stream_cursor")
    if not isinstance(cursor, dict):
        return not_run("no shared stream cursor was passed to the action")
    index = int(cursor.get("i", 0))
    if pacer is None:
        return not_run("no pacer was passed to the action")
    if index >= len(queue):
        return not_run(f"the stream queue is exhausted ({len(queue)} turns planned)")
    if _ev(ctx, "() => window.__sb.dom.isRunning()"):
        # Sending while a reply is in flight queues the message instead of starting a stream, and
        # the action would report a fast, precise number about a message that is merely parked.
        return not_run("a reply was still streaming, so this send would have been queued")

    unit = queue[index]
    cursor["i"] = index + 1
    pacer.reset()
    pacer.load(
        unit["reasoning"],
        unit["content"],
        cadence = ctx.args.get("cadence", "field"),
        tag = f"{ctx.args.get('cell_id', 'cell')}#turn{index + 1}",
    )

    selector = 'textarea[aria-label="Message input"]'
    if ctx.page.query_selector(selector) is None:
        return not_run("no composer on the page")
    ctx.page.fill(selector, f"studiobench follow-up {index + 1}")
    ctx.page.wait_for_timeout(80)
    before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    started = time.monotonic()
    ctx.page.keyboard.press("Enter")

    first_ms = None
    deadline = started + SETTLE_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        if _ev(ctx, "() => window.__sb.dom.isRunning()"):
            first_ms = (time.monotonic() - started) * 1000
            break
        ctx.page.wait_for_timeout(50)
    after = _ev(ctx, "() => window.__sb.dom.messageCount()")
    # A POSITIVE consequence: the turn actually started streaming AND the thread grew. A send that
    # silently failed leaves both unchanged and would otherwise read as an instant send.
    ok = (
        first_ms is not None
        and isinstance(after, int)
        and isinstance(before, int)
        and after > before
    )
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "messages_before": before,
            "messages_after": after,
            "turn_index": index + 1,
            "queued_turns": len(queue),
            "streamed_chars": len(unit["reasoning"]) + len(unit["content"]),
            "unit_kind": unit.get("kind"),
        },
        timings = {"to_first_token_ms": None if first_ms is None else round(first_ms, 1)},
        reason = None if ok else "the send did not start a new streaming reply",
    )


#: What the page can tell us when the attachments button cannot be found. Reads geometry, style,
#: hit-testing and the surrounding chrome, so the three explanations that look identical from a
#: `not_run` string look different here.
IMAGE_BUTTON_DIAGNOSTIC = """() => {
    const sel = 'button[aria-label="Tools and attachments"]';
    const all = [...document.querySelectorAll(sel)];
    return {
        in_dom: all.length,
        boxes: all.map((b) => {
            const r = b.getBoundingClientRect();
            const cs = getComputedStyle(b);
            const top = r.width && r.height
                ? document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2) : null;
            return {
                w: Math.round(r.width), h: Math.round(r.height),
                x: Math.round(r.left), y: Math.round(r.top),
                in_viewport: r.top >= 0 && r.bottom <= window.innerHeight,
                display: cs.display, visibility: cs.visibility, opacity: cs.opacity,
                pointer_events: cs.pointerEvents,
                mounted: b.offsetParent !== null,
                hit_self: top === b || (top !== null && b.contains(top)),
            };
        }),
        composers: document.querySelectorAll('textarea[aria-label="Message input"]').length,
        open_dialogs: document.querySelectorAll('[role="dialog"]').length,
        popper_layers: document.querySelectorAll('[data-radix-popper-content-wrapper]').length,
        body_pointer_events: getComputedStyle(document.body).pointerEvents,
        model_loaded_hint: (document.body.innerText || "").includes("Load a model"),
        viewport: {w: window.innerWidth, h: window.innerHeight},
    };
}"""


@register_action(name = "image_upload", default_budget_ms = 12000)
def image_upload(ctx: ActionContext) -> ActionResult:
    """Attach an image through the composer's file chooser.

    THERE IS NO PERSISTENT `<input type=file>` in the composer. thread.tsx creates one
    imperatively on click, appends it to `<body>` hidden, clicks it and removes it, and its
    `accept` is derived at runtime from the selected model's vision capability. So the only
    reliable path is Playwright's file chooser event around the menu item, and any hard-coded
    `accept` assertion would be unsafe.
    """
    png = ctx.args.get("image_path")
    if not png:
        return not_run("no image path was supplied to the action")
    # `:visible`, and the FIRST visible match.
    #
    # Two reasons, both measured. `document.querySelector` returns the first match in DOCUMENT
    # ORDER and the composer exists more than once: the welcome-screen instance stays in the tree
    # behind the docked one, and compare mode has two threads. The hidden duplicate comes first,
    # so a plain query handed back a button that can never be clicked and the action reported "not
    # visible" about a control that a direct probe found at 36x36 pixels, fully opaque, on screen.
    #
    # And Playwright's actionability wait does not give up quickly on an unclickable element: at
    # its 30s default it blocked a 9s slot for more than three times the slot's budget, to report
    # that nothing happened. Everything here is bounded by what is left of the slot.
    locator = ctx.page.locator('button[aria-label="Tools and attachments"]:visible').first
    try:
        plus = locator.element_handle(timeout = 2000)
    except Exception:  # noqa: BLE001
        plus = None
    if plus is None:
        # WHY, not just THAT. A bare "not visible" is the opaque zero of the action layer: it
        # conflates a button that is absent, a button that is present and covered, and a locator
        # that disagrees with the page, and it has already cost three wrong hypotheses. A direct
        # probe found the control at 36x36, fully opaque and hit-testable, on a fresh chat and
        # after a settings round trip and under a 20,000-character composer fill, so whatever
        # this is, it is none of those. Carrying the state into the row means the NEXT run
        # answers it instead of the next investigation.
        return not_run(
            "no visible attachments button on the composer: "
            + json.dumps(_ev(ctx, IMAGE_BUTTON_DIAGNOSTIC) or {})
        )
    before = _ev(
        ctx,
        "() => document.querySelectorAll('.aui-composer-attachment, "
        '[data-slot="composer-attachment"]\').length',
    )
    started = time.monotonic()
    # Bounded by what is left of the slot, never by Playwright's 30s default.
    try:
        plus.click(timeout = max(500, min(ctx.budget_ms // 3, 5000)))
    except Exception as exc:  # noqa: BLE001
        return not_run(f"the attachments button could not be clicked: {type(exc).__name__}")
    ctx.page.wait_for_timeout(200)
    try:
        with ctx.page.expect_file_chooser(timeout = 6000) as fc:
            ctx.page.evaluate("""() => {
                const item = window.__sb.dom.menuItemByText("Add photos");
                if (item) item.click();
            }""")
        fc.value.set_files(png)
    except Exception as exc:  # noqa: BLE001
        ctx.page.keyboard.press("Escape")
        return not_run(f"the file chooser never opened: {type(exc).__name__}: {exc}")
    ctx.page.wait_for_timeout(800)
    after = _ev(
        ctx,
        "() => document.querySelectorAll('.aui-composer-attachment, "
        '[data-slot="composer-attachment"]\').length',
    )
    elapsed = (time.monotonic() - started) * 1000
    ok = after is not None and before is not None and after > before
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {"attachments_before": before, "attachments_after": after},
        timings = {"upload_ms": round(elapsed, 1)},
        reason = None if ok else "no attachment appeared in the composer after the file was set",
    )


# ── 13. thread reopen ───────────────────────────────────────────────


@register_action(name = "thread_reopen", default_budget_ms = 30000)
def thread_reopen(ctx: ActionContext) -> ActionResult:
    """Leave the thread and come back.

    The runtime keeps the messages; the Thread subtree is torn down and rebuilt, so every markdown
    block, every Shiki fence and every action bar is mounted again from nothing. This is the
    action users describe as "it hangs when I click back into the conversation" and it is the one
    with no incremental path at all.
    """
    thread_id = ctx.args.get("thread_id")
    if not thread_id:
        return not_run("no thread id was supplied to the action")
    before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    if not before:
        return not_run("the thread has no messages to rebuild")
    started = time.monotonic()
    # Click if the control is really clickable, otherwise NAVIGATE. The sidebar's sticky group
    # label overlays the New chat button and intercepts the pointer, so the click retries until it
    # times out on a button that is visible, enabled and stable -- and the action then reports a
    # rebuild that never happened. `/chat?new=` is what the button itself does.
    if not _click_or_navigate(
        ctx, 'button[aria-label="New chat"]', f"{ctx.args['base_url']}/chat?new=studiobench"
    ):
        return not_run("the thread could not be left, by click or by navigation")
    # Unmount FIRST, or "already back" is indistinguishable from "never left".
    closed_ms = None
    deadline = started + 15
    while time.monotonic() < deadline:
        if _ev(ctx, "() => window.__sb.dom.messageCount()") == 0:
            closed_ms = (time.monotonic() - started) * 1000
            break
        ctx.page.wait_for_timeout(50)
    if closed_ms is None:
        return not_run("the thread never unmounted, so the rebuild could not be timed")

    reopen_started = time.monotonic()
    if not _click_or_navigate(
        ctx, f'[data-thread-id="{thread_id}"]', f"{ctx.args['base_url']}/chat?thread={thread_id}"
    ):
        return not_run("the thread could not be reopened, by click or by navigation")
    reopen_ms = None
    deadline = reopen_started + 60
    while time.monotonic() < deadline:
        if (_ev(ctx, "() => window.__sb.dom.messageCount()") or 0) >= before:
            reopen_ms = (time.monotonic() - reopen_started) * 1000
            break
        ctx.page.wait_for_timeout(100)
    after = _ev(ctx, "() => window.__sb.dom.messageCount()")
    spans = _ev(ctx, "() => document.querySelectorAll('pre span').length")
    ok = reopen_ms is not None and after == before
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {"messages_before": before, "messages_after": after, "highlight_spans_after": spans},
        timings = {
            "close_ms": round(closed_ms, 1),
            "reopen_ms": None if reopen_ms is None else round(reopen_ms, 1),
        },
        reason = None if ok else f"the thread came back with {after} of {before} messages",
    )


def _click_or_navigate(ctx: ActionContext, selector: str, url: str) -> bool:
    """Click the control a user would click; fall back to the URL it would produce.

    A click is preferred because it exercises the app's own handler. But Playwright's
    actionability retries do not give up on an element that is visible, enabled and stable and
    merely covered by something else, which is the sidebar's sticky group label over the New chat
    button -- so the click burns its whole timeout and the action reports a rebuild that never
    happened. The navigation is the same state transition by the route the app's own control uses.
    """
    handle = ctx.page.query_selector(selector)
    if handle is not None:
        try:
            handle.click(timeout = 2000)
            return True
        except Exception:  # noqa: BLE001
            ctx.log(f"    {selector} was not clickable; navigating instead")
    try:
        ctx.page.goto(url, wait_until = "domcontentloaded", timeout = 60_000)
        return True
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"    navigation to {url} failed: {type(exc).__name__}: {exc}")
        return False


# ── 14. message menu ────────────────────────────────────────────────

MENU_JS = """
async (opts) => {
  const D = window.__sb.dom;
  const timeoutMs = opts.timeoutMs;
  // WAITED FOR, not sampled for. The bar carrying this trigger is `hideWhenRunning`, so the
  // control is absent -- not hidden -- for as long as the follow-up turn the preceding `send_turn`
  // started is still arriving. See `dom.js waitForActionButton` for the measurement that says how
  // late that can be, and why a single sample turns a third of a second of stream into a report of
  // a missing control. The clock below starts AFTER this returns, so nothing that is waited for
  // here lands in the latency this action exists to measure.
  const found = await D.waitForActionButton("More", opts.waitForButtonMs);
  const trigger = found.el;
  const waitedMs = found.waitedMs;
  if (!trigger) {
    return {
      ran: false,
      waitedMs,
      running: found.running,
      reason:
        "no More button on the last assistant message after waiting " + waitedMs + "ms" +
        (found.running
          ? ": the thread was STILL GENERATING, which unmounts the action bar, so this slot "
            + "opened before the reply settled rather than the control being missing"
          : ""),
    };
  }
  // A MutationObserver flag, NOT a querySelector per frame. The menu content is portaled to the
  // end of document.body, so polling for it walks the whole message list and finds nothing for
  // the entire open latency: a harness cost that grows exactly like the signal being measured.
  let open = Boolean(D.openMenu());
  const watcher = new MutationObserver(() => { open = Boolean(D.openMenu()); });
  watcher.observe(document.body, { childList: true, subtree: false });
  const settle = async (want) => {
    const started = performance.now();
    while (performance.now() - started < timeoutMs) {
      if (open === want) return performance.now() - started;
      await window.__sbNextPaint();
    }
    return null;
  };
  const pointer = { bubbles: true, cancelable: true, composed: true, button: 0,
                    pointerId: 1, pointerType: "mouse", isPrimary: true };
  const openStarted = performance.now();
  // pointerdown/up, not click(). The Radix trigger opens on pointerdown, so an element.click()
  // leaves the menu shut and the whole measurement silently reads zero.
  trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const opened = await settle(true);
  const openMs = opened === null ? null : performance.now() - openStarted;
  const bodyPointerEvents = getComputedStyle(document.body).pointerEvents;
  const items = D.openMenuItemCount();
  // The clock starts BEFORE the dispatch: Radix dismisses synchronously inside it -- layer
  // teardown, focus restore, the body coming off the modal layer and the re-render after -- which
  // is the fan-out being measured. Starting it after excluded exactly the part worth timing.
  const closeStarted = performance.now();
  document.dispatchEvent(new KeyboardEvent("keydown",
    { key: "Escape", bubbles: true, cancelable: true }));
  const closed = await settle(false);
  const closeMs = closed === null ? null : performance.now() - closeStarted;
  watcher.disconnect();
  return { ran: true, waitedMs,
           openMs: openMs === null ? null : Math.round(openMs * 10) / 10,
           closeMs: closeMs === null ? null : Math.round(closeMs * 10) / 10,
           items, bodyPointerEvents,
           bodyPointerEventsAfterClose: getComputedStyle(document.body).pointerEvents };
}
"""


@register_action(name = "message_menu", default_budget_ms = 12000)
def message_menu(ctx: ActionContext) -> ActionResult:
    raw = _ev(
        ctx,
        MENU_JS,
        {"timeoutMs": SETTLE_TIMEOUT_MS, "waitForButtonMs": ACTION_BAR_WAIT_MS},
    )
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the menu action did not run"))
    waited_ms = raw.get("waitedMs")
    if waited_ms:
        # Logged rather than swallowed. A cell that had to wait says so, because a wait means the
        # film opened this slot before the follow-up had drained, and that is a fact about the
        # PACKING which is worth seeing even though the action ran.
        ctx.log(f"    message_menu waited {waited_ms}ms for the action bar to be mounted")
    # Opened AND closed AND a non-zero item count. Any one of the three alone can be satisfied by
    # a menu that never rendered its items.
    ok = raw["openMs"] is not None and raw["closeMs"] is not None and raw["items"] > 0
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "items_while_open": raw["items"],
            # Radix puts the body on the modal layer while the menu is up, which is the
            # fan-out under suspicion. This proves the open really took that path.
            "body_pointer_events_open": raw["bodyPointerEvents"],
            "body_pointer_events_closed": raw["bodyPointerEventsAfterClose"],
            # In the payload, not only in the log: a run whose cells all waited is a run whose
            # film opens this slot too early, and that is not visible from a row that only says
            # `ran`.
            "action_bar_wait_ms": waited_ms,
        },
        timings = {
            "open_ms": raw["openMs"],
            "close_ms": raw["closeMs"],
            "open_close_ms": (
                None
                if raw["openMs"] is None or raw["closeMs"] is None
                else round(raw["openMs"] + raw["closeMs"], 1)
            ),
        },
        reason = None
        if ok
        else f"opened={raw['openMs'] is not None} closed={raw['closeMs'] is not None} "
        f"items={raw['items']}",
    )


# ── 15. delete ──────────────────────────────────────────────────────

DELETE_JS = """
async (opts) => {
  const D = window.__sb.dom;
  const timeoutMs = opts.timeoutMs;
  // "a running message hides it" was already the diagnosis in the line this replaced, and the
  // action still reported NOT RUN on the first sample rather than waiting for the running message
  // to stop running. Same wait, same bound and same reporting as `message_menu`.
  const found = await D.waitForActionButton("Delete message", opts.waitForButtonMs);
  const button = found.el;
  const waitedMs = found.waitedMs;
  if (!button) {
    return {
      ran: false,
      waitedMs,
      running: found.running,
      reason:
        "no Delete button after waiting " + waitedMs + "ms" +
        (found.running ? ": the thread was still generating, which unmounts the action bar" : ""),
    };
  }
  const target = D.lastAssistantMessage();
  const before = D.messageCount();
  const started = performance.now();
  button.click();
  let ms = null;
  // isConnected on the captured node is O(1). Re-counting [data-role] every frame would put an
  // O(messages) query INSIDE the window being timed, growing like the signal.
  while (performance.now() - started < timeoutMs) {
    if (target === null || !target.isConnected) { ms = performance.now() - started; break; }
    await window.__sbNextPaint();
  }
  return { ran: true, waitedMs, ms: ms === null ? null : Math.round(ms * 10) / 10,
           before, after: D.messageCount() };
}
"""


@register_action(name = "delete_message", default_budget_ms = 15000)
def delete_message(ctx: ActionContext) -> ActionResult:
    raw = _ev(
        ctx,
        DELETE_JS,
        {"timeoutMs": SETTLE_TIMEOUT_MS, "waitForButtonMs": ACTION_BAR_WAIT_MS},
    )
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the delete action did not run"))
    # The [data-role] count DROPPED. A delete that detached the node but left the count is a
    # different bug and must not read as a successful delete.
    ok = raw["ms"] is not None and raw["after"] < raw["before"]
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "messages_before": raw["before"],
            "messages_after": raw["after"],
            "dropped": raw["before"] - raw["after"],
            "action_bar_wait_ms": raw.get("waitedMs"),
        },
        timings = {"delete_ms": raw["ms"]},
        reason = None if ok else f"the message count went {raw['before']} -> {raw['after']}",
    )
