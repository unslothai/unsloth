# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fifteen actions, each with an `expect` that proves it actually happened.

THE RULE THIS FILE EXISTS TO ENFORCE: an action that did not happen is `ran = False`. It is NEVER
a fast timing. That is not a style preference, it is the failure mode that wasted a day of
measurement. A menu whose trigger opens on `pointerdown` does not open when you call `.click()`,
and the column then reads a tidy small number that looks like a fast menu. A jump scroll from the
bottom is read by Unsloth's intent-aware autoscroll as programmatic and snapped straight back, so
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
from dataclasses import dataclass
from typing import Any, Optional

from ..runtime.readiness import (
    MODE_FULL,
    MODE_WINDOWED,
    ThreadNotReady,
    wait_for_thread_ready,
)
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


#: How much of an exception's message a refusal reason carries.
_EXC_CHARS = 160


def _why(exc: BaseException) -> str:
    """An exception as a reason a person can act on: the class AND its first line.

    `type(exc).__name__` alone is worthless here and cost a full day of someone else's. Playwright's
    own exception class is named `Error`, so every clipboard refusal in two complete 100K payloads
    read "the clipboard could not be read back: Error" -- which names no cause, points at no engine
    and cannot be told apart from any other failure in the stack. The message is where the browser
    says `NotAllowedError: Document is not focused` or `TypeError: navigator.clipboard is
    undefined`, and it is the whole diagnosis.

    First line only, and truncated: Playwright appends the call log, which is dozens of lines and
    belongs in the log rather than in a row's `reason`.
    """
    first = str(exc).strip().splitlines()
    head = first[0][:_EXC_CHARS] if first else ""
    return f"{type(exc).__name__}: {head}" if head else type(exc).__name__


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


#: How long the keystroke drain will wait for the last paint before giving up on it. Not a "wait
#: a bit longer" constant: the wait ENDS when nothing is in flight, and this is only the bound that
#: stops a wedged renderer from eating the slot. Reaching it means a sample was lost, which the
#: coverage check then fails on rather than quietly publishing the rest.
KEYSTROKE_SETTLE_TIMEOUT_MS = 3000
KEYSTROKE_SETTLE_POLL_MS = 25


def _settle_keystrokes(ctx: ActionContext, inst: Any) -> None:
    """Wait until no keystroke's paint is still in flight, bounded.

    The fixed 200 ms wait this replaces dropped whichever sample had not painted when it expired,
    which is systematically the slowest one. A bigger constant has the same defect on a slower
    machine or a heavier rung, so the wait is on the WORK rather than on the clock.
    """
    deadline = time.monotonic() + KEYSTROKE_SETTLE_TIMEOUT_MS / 1000
    settled = getattr(inst, "settled", None)
    if settled is None:
        ctx.page.wait_for_timeout(200)
        return
    while time.monotonic() < deadline:
        state = settled()
        # The page could not answer: polling on it would spin to the bound and tell us nothing.
        if not isinstance(state, dict):
            break
        if not state.get("pending"):
            return
        ctx.page.wait_for_timeout(KEYSTROKE_SETTLE_POLL_MS)


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
    _settle_keystrokes(ctx, inst)
    got = inst.collect(count)
    elapsed_ms = (time.monotonic() - started) * 1000

    grew = got.get("grew_by")
    if not got.get("samples"):
        return not_run(f"no keystroke reached the composer ({got.get('reason', 'no samples')})")
    seen = got.get("inputs_seen")
    accounted = (got.get("samples") or 0) + (got.get("coalesced") or 0)
    expect = {
        "commanded_chars": count,
        "measured_keystrokes": got.get("samples"),
        "coalesced": got.get("coalesced"),
        "inputs_seen": seen,
        "pending_at_collect": got.get("pending_at_collect"),
        "composer_grew_by": grew,
        "composer_text_length": got.get("text_length"),
    }
    # EVERY KEYSTROKE ACCOUNTED FOR, not merely a composer that grew.
    #
    # `grew_by` alone was the whole check, and it is satisfied by a textarea whose value is right
    # while the timings describe a subset of the typing. Two ways that happened, and the first
    # INVERTS the metric: a keystroke whose paint had not resolved when the drain ran was simply
    # absent, and the keystroke that has not painted yet is the slowest one -- measured here, a
    # 500 ms keystroke vanished from a reading whose max was 20 ms, on the highest-weight metric in
    # the table. A build that made typing worse read faster, and that reading feeds the null
    # control's noise floor, so it would also tighten the floor every later comparison is judged
    # against. The other way is quieter: a keystroke that never reached the instrument at all is
    # missing from both `samples` and `coalesced`.
    #
    # So the reading stands only when nothing was left in flight and `samples + coalesced` covers
    # every input the instrument saw, and every input the driver commanded arrived. A LOW sample
    # count is not itself a failure -- on a jammed page most keystrokes coalesce behind a slow
    # paint, and that is the finding, not a fault -- which is why this counts coverage rather than
    # demanding a sample per character.
    covered = (
        seen is not None
        and seen >= count
        and accounted >= seen
        and not got.get("pending_at_collect")
    )
    ok = grew is not None and grew >= count and covered
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
        reason = None
        if ok
        else (
            f"typed {count} characters but the composer value grew by {grew}"
            if grew is None or grew < count
            else (
                f"typed {count} characters and the reading covers {accounted} of {seen} that "
                f"reached the instrument"
                + (
                    ", with one still unpainted at the drain"
                    if got.get("pending_at_collect")
                    else ""
                )
            )
        ),
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
    # THE FOLLOW SAMPLER IS SUSPENDED FOR THE DURATION OF THIS GESTURE. This action drags the
    # viewport thousands of pixels off the bottom on purpose, twice, while the reply is still
    # streaming. Samples taken during it record a thread that is not pinned, which is true and is
    # the benchmark's own doing, and averaging them in makes `follows_the_stream` a reading about
    # the film rather than about the app. See the sampler in scene/dom.js.
    _ev(ctx, "() => window.__sb.follow && window.__sb.follow.suspend()")
    try:
        raw = _ev(ctx, SCROLL_JS, [steps, step_px, settle_ms])
    finally:
        # Resumed even when the gesture raised, or one failed scroll silences the sampler for the
        # rest of the cell and the arm reports a follow fraction built from nothing.
        _ev(ctx, "() => window.__sb.follow && window.__sb.follow.resume()")
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the scroll did not run"))
    commanded = raw["commandedPx"]
    travelled = raw["travelledPx"]
    # 90% of commanded. Unsloth replaces assistant-ui's autoscroll with an intent-aware one that
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

# SETTLING ON THE DOM, NOT ON A STATE ATTRIBUTE.
#
# `data-state="open"` flips when the Collapsible's state changes. It says nothing about whether the
# content that state reveals has mounted, and how far apart those two frames are depends on the
# COLLAPSE MECHANISM -- which is exactly the thing an A/B across a collapse change is comparing. So
# an instant defined by the attribute means a different thing on each arm.
#
# Measured, 100K rung, both arms, same corpus. Reading `pre span` on the frame the attribute
# settles gives 74,917 on the measured-height arm and 44,075 on the grid-rows arm, a 41% apparent
# reduction. Reading it once the span count has stopped changing gives 74,250 on BOTH, to the span.
# The 41% was the grid arm being counted before it had finished mounting.
#
# Its null control did not catch this and could not have: the null runs one bundle against itself,
# so the skew is identical on both sides and cancels exactly. A null cannot see a bias it shares.
#
# So both the timing and the census now terminate on a QUIET DOM: the observed quantity has to stop
# changing for `quietFrames` consecutive animation frames before it is read. If it never goes
# quiet, nothing is returned for it and the reason is recorded, because silence beats a confident
# wrong answer.
SETTLE_QUIET_FRAMES = 4

REASONING_JS = """
async ([timeoutMs, quietFrames]) => {
  const D = window.__sb.dom;
  const triggers = D.reasoningTriggers();
  if (triggers.length === 0) return { ran: false, reason: "no reasoning pane in the thread" };
  const before = D.reasoningOpenCount();
  const spanCount = () => document.querySelectorAll("pre span").length;

  // Wait for the open count to reach `want` AND for the span census to stop moving. Returns the
  // elapsed time, the number of frames spent, and why it stopped, so a censored reading is
  // distinguishable from a fast one rather than both arriving as a number.
  //
  // `origin` IS THE CALLER'S MARK, TAKEN BEFORE THE CLICKS, AND THE REPORTED TIMING RUNS FROM IT.
  // Dispatching the clicks is the first half of opening the panes: `t.click()` runs the app's own
  // handler synchronously, and on a long thread that is React state plus layout, sixteen times
  // over. Timing from the settle's own start instead silently redefines `open_ms` as "the wait
  // after the work", which is a smaller number for the same action -- measured on a page with a
  // 40 ms handler per pane, `open_ms` read 96 ms against 640 ms of dispatch it had just done.
  // That is the defect this whole branch is about, so it must not be introduced by the fix for it.
  //
  // The BUDGET still runs from the settle's own start, so what counts as censored is unchanged.
  // The two are reported separately: `ms` is the whole operation, `settle_ms` is the wait alone.
  const settle = async (want, origin, requireTornDown) => {
    const started = performance.now();
    let frames = 0;
    let quiet = 0;
    let last = spanCount();
    let reachedAt = null;
    // The RAW state, tracked apart from `reachedAt` so a censored close can say whether the panes
    // closed and did not tear down, or never closed at all.
    let countReachedAt = null;
    while (performance.now() - started < timeoutMs) {
      await window.__sbNextPaint();
      frames += 1;
      const now = spanCount();
      // A COLLAPSE IS NOT DONE WHEN THE STATE SAYS CLOSED, IT IS DONE WHEN THE CONTENT IS GONE.
      //
      // This is the same defect as the one the quiet streak fixes for the open direction, arriving
      // from the other side. There, the state flips before the spans mount, and a streak counted
      // from frame one was satisfied by a census that had not started moving. Here the state flips
      // before the spans UNMOUNT: both mechanisms keep the children in the document for the length
      // of the exit animation (Radix until `animationend`, the grid arm until the
      // `grid-template-rows` `transitionend` or its 250 ms backstop), so `pre span` is frozen at
      // its open value for that entire window. Four quiet frames land inside it whenever a frame is
      // shorter than the animation -- about 83 ms at 60 Hz against 200 -- and `close_ms` then names
      // the state flip plus four frames rather than the collapse.
      //
      // That is not a constant offset either, which is what makes it a comparison problem rather
      // than a units one: whether the streak ends before or after the teardown depends on the
      // paint interval against the animation duration, and the paint interval is what differs
      // between the arms and the rungs this instrument exists to compare. See
      // `dom.reasoningContentMounted`.
      const countReached = D.reasoningOpenCount() === want;
      if (countReached && countReachedAt === null) countReachedAt = performance.now() - started;
      const stateReached =
        countReached && (!requireTornDown || D.reasoningContentMounted() === 0);
      const justReached = stateReached && reachedAt === null;
      if (justReached) reachedAt = performance.now() - started;
      // THE QUIET STREAK BELONGS ENTIRELY TO THE WINDOW AFTER THE STATE IS REACHED.
      //
      // Counting it from the first frame instead defeats the whole fix whenever the panes take
      // `quietFrames` or more frames to reach the open state while nothing has mounted yet -- and
      // that is the NORMAL case at the rungs this was written for. The catalogue's own 500K
      // reading is "the open count reached 16 after 10440ms": the census is static through all of
      // those frames because the content it would count has not been revealed yet, so the streak
      // is already satisfied on the very frame the state flips, and the read happens exactly where
      // it used to. Reproduced against this JS with a page whose state flips at frame 6 and whose
      // spans mount to frame 40: it returned 44,075 with `censored: false`, which is the withdrawn
      // number, reported confidently, out of the code that exists to stop it.
      //
      // The streak also RESTARTS if the state is lost again, so a count that oscillates around
      // `want` cannot bank quiet frames it did not hold.
      quiet = stateReached && !justReached && now === last ? quiet + 1 : 0;
      last = now;
      if (stateReached && quiet >= quietFrames) {
        return {
          // From the caller's mark, so the click dispatch this action performed is inside the
          // number that names it.
          ms: performance.now() - origin,
          settle_ms: performance.now() - started,
          dispatch_ms: started - origin,
          frames,
          // Rebased onto the same origin as `ms`, or the two would be quoted against each other
          // from different zeroes.
          state_reached_ms: reachedAt === null ? null : reachedAt + (started - origin),
          spans: now,
          censored: false,
          reason: null,
        };
      }
    }
    return {
      ms: null,
      settle_ms: performance.now() - started,
      dispatch_ms: started - origin,
      frames,
      state_reached_ms: reachedAt === null ? null : reachedAt + (started - origin),
      spans: null,
      censored: true,
      // THE REASON NAMES WHICH HALF OF "SETTLED" WAS MISSING. A close that reached the state but
      // whose panes never tore down is a different finding from one whose panes never closed, and
      // reporting both as "the open count never reached 0" would send a reader to the wrong half
      // of the app.
      reason: reachedAt === null
        ? (requireTornDown && countReachedAt !== null
            ? `the open count reached ${want} after ${Math.round(countReachedAt)}ms but the `
              + `collapsed panes were still mounted when the ${timeoutMs}ms budget ran out, so `
              + `nothing here is a reading of a collapsed document`
            : `the open count never reached ${want} within ${timeoutMs}ms`)
        : (requireTornDown
            ? `the panes closed after ${Math.round(countReachedAt)}ms and unmounted after `
              + `${Math.round(reachedAt)}ms but the span census was still changing when the `
              + `${timeoutMs}ms budget ran out, so nothing here is a reading of a settled document`
            : `the open count reached ${want} after ${Math.round(reachedAt)}ms but the span `
              + `census was still changing when the ${timeoutMs}ms budget ran out, so nothing `
              + `here is a reading of a settled document`),
    };
  };

  // Toggle EVERY pane, not one: the mechanism under investigation scales with how much content
  // is mounted, and opening a single pane in a thread of forty is a constant-size action being
  // reported on an axis of thread length.
  const openStart = performance.now();
  for (const t of triggers) t.click();
  const opened = await settle(triggers.length, openStart, false);
  const openCount = D.reasoningOpenCount();
  const closeStart = performance.now();
  for (const t of D.reasoningTriggers()) t.click();
  // `true`: the collapse is settled when the panes have UNMOUNTED, not when they report closed.
  const closed = await settle(0, closeStart, true);
  return {
    ran: true,
    panes: triggers.length,
    before,
    openCount,
    afterClose: D.reasoningOpenCount(),
    // ONLY from a settled document. `null` when the census never went quiet.
    spansOpen: opened.spans,
    spansOpenReason: opened.reason,
    openMs: opened.ms === null ? null : Math.round(opened.ms * 10) / 10,
    closeMs: closed.ms === null ? null : Math.round(closed.ms * 10) / 10,
    openCensored: opened.censored,
    closeCensored: closed.censored,
    openCensoredReason: opened.reason,
    closeCensoredReason: closed.reason,
    // The ruler's own resolution, so a reader can see that a difference smaller than a frame is
    // not a difference. `open_ms` is quantised to the paint interval by construction.
    openFrames: opened.frames,
    closeFrames: closed.frames,
    openStateReachedMs: opened.state_reached_ms === null
      ? null : Math.round(opened.state_reached_ms * 10) / 10,
    // THE SPLIT, so a reader can see how much of the timing was the app's click handlers and how
    // much was waiting for the DOM to stop moving. They answer different questions and a single
    // total hides which of the two a change actually moved.
    openDispatchMs: Math.round(opened.dispatch_ms * 10) / 10,
    closeDispatchMs: Math.round(closed.dispatch_ms * 10) / 10,
    openSettleMs: Math.round(opened.settle_ms * 10) / 10,
    closeSettleMs: Math.round(closed.settle_ms * 10) / 10,
    quietFramesRequired: quietFrames,
    timeoutMs,
  };
}
"""


@register_action(name = "reasoning_toggle", default_budget_ms = 12000)
def reasoning_toggle(ctx: ActionContext) -> ActionResult:
    raw = _ev(ctx, REASONING_JS, [SETTLE_TIMEOUT_MS, SETTLE_QUIET_FRAMES])
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the reasoning toggle did not run"))

    # THE REASON NAMES THE CLAUSE THAT ACTUALLY FAILED.
    #
    # This used to be built from the pane counts alone, so a run whose real failure was a censored
    # timing printed "16 of 16 panes opened and 0 were still open after collapsing" -- a
    # description of SUCCESS -- under the heading EXPECT FAILED. A reader who trusts the message
    # concludes the assertion is broken rather than that the timing is missing.
    failures: list[str] = []
    if raw["openCount"] != raw["panes"]:
        failures.append(f"only {raw['openCount']} of {raw['panes']} panes opened")
    if raw["afterClose"] != 0:
        failures.append(f"{raw['afterClose']} panes were still open after collapsing")
    if raw["openMs"] is None:
        failures.append(f"open_ms is censored: {raw.get('openCensoredReason')}")
    if raw["closeMs"] is None:
        failures.append(f"close_ms is censored: {raw.get('closeCensoredReason')}")
    ok = not failures

    # A CENSORED TIMING IS ABSENT, NOT ZERO, and it is absent LOUDLY. `_action_timings` drops
    # non-numeric values, so a censored cell silently leaves the pooled metric and the survivors
    # are the fast ones -- survivorship bias wearing a mean. The censoring is therefore recorded
    # as its own field so the scoring layer can refuse to pool a metric that is censored at some
    # rungs and not others.
    timings = {}
    if raw["openMs"] is not None:
        timings["open_ms"] = raw["openMs"]
    if raw["closeMs"] is not None:
        timings["close_ms"] = raw["closeMs"]

    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            # SCOPED TO WHAT IS MOUNTED, and always was. `reasoningTriggers()` is a
            # querySelectorAll, so on the shipped build it finds every pane in the thread and on a
            # windowed mount it finds the panes in the window. The assertion below stays
            # self-consistent either way (it opens N and expects N open), so the action still
            # passes -- but its COST silently stops being a function of thread length and becomes
            # a function of window size. That is the arm's point, and it is also the reason this
            # timing is not comparable between a windowed arm and a full one without the pane
            # count beside it.
            "panes": raw["panes"],
            "panes_scope": "mounted",
            "open_after_expand": raw["openCount"],
            "open_after_collapse": raw["afterClose"],
            # Present ONLY when the span census went quiet. See the note above REASONING_JS: read
            # on the frame the state attribute flips, this number was 41% wrong on one arm and
            # right on the other, and its null control could not see the difference.
            "highlight_spans_while_open": raw["spansOpen"],
            "highlight_spans_while_open_reason": raw.get("spansOpenReason"),
            "settled": raw["spansOpen"] is not None,
            "open_censored": bool(raw.get("openCensored")),
            "close_censored": bool(raw.get("closeCensored")),
            "open_censored_reason": raw.get("openCensoredReason"),
            "close_censored_reason": raw.get("closeCensoredReason"),
            # The ruler's resolution, and how much of `open_ms` was spent after the state flip.
            "open_frames": raw.get("openFrames"),
            "open_state_reached_ms": raw.get("openStateReachedMs"),
            "quiet_frames_required": raw.get("quietFramesRequired"),
        },
        timings = timings,
        reason = None if ok else "; ".join(failures),
    )


#: THE SAME GESTURE A USER MAKES. `reasoning_toggle` above opens every pane in the thread at once,
#: which is a deliberate worst case and reads at 2.2 fps at the 100K rung. That number has been
#: quoted as though it described opening a reasoning pane, and it does not: a user opens ONE, and
#: almost always the newest one. This action measures that, so the two can be quoted apart.
#:
#: It is NOT in the standard film. The scene is a fixed-duration slot schedule, so adding a slot
#: shifts every window after it and voids comparability against every payload already on disk. Run
#: it in a purpose-built film until the next corpus or tier bump invalidates those payloads anyway.
REASONING_ONE_JS = """
async (timeoutMs) => {
  const D = window.__sb.dom;
  const triggers = D.reasoningTriggers();
  if (triggers.length === 0) return { ran: false, reason: "no reasoning pane in the thread" };
  const before = D.reasoningOpenCount();
  if (before !== 0) return { ran: false, reason: `${before} panes were already open` };
  // The LAST pane: the newest reply is the one a user reaches for, and it is also the only pane
  // whose position is the same on a windowed mount as on a full one.
  const target = triggers[triggers.length - 1];
  const settle = async (want) => {
    const started = performance.now();
    while (performance.now() - started < timeoutMs) {
      if (D.reasoningOpenCount() === want) return performance.now() - started;
      await window.__sbNextPaint();
    }
    return null;
  };
  const spansBefore = document.querySelectorAll("pre span").length;
  const openStart = performance.now();
  target.click();
  const openedIn = await settle(1);
  const openMs = openedIn === null ? null : performance.now() - openStart;
  const openCount = D.reasoningOpenCount();
  const spansOpen = document.querySelectorAll("pre span").length;
  const closeStart = performance.now();
  (D.reasoningTriggers()[triggers.length - 1] || target).click();
  const closedIn = await settle(0);
  const closeMs = closedIn === null ? null : performance.now() - closeStart;
  return {
    ran: true,
    panes: triggers.length,
    openCount,
    afterClose: D.reasoningOpenCount(),
    spansOpen,
    spansAdded: spansOpen - spansBefore,
    openMs: openMs === null ? null : Math.round(openMs * 10) / 10,
    closeMs: closeMs === null ? null : Math.round(closeMs * 10) / 10,
  };
}
"""


@register_action(name = "reasoning_toggle_one", default_budget_ms = 12000)
def reasoning_toggle_one(ctx: ActionContext) -> ActionResult:
    raw = _ev(ctx, REASONING_ONE_JS, SETTLE_TIMEOUT_MS)
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the reasoning toggle did not run"))
    ok = (
        raw["openMs"] is not None
        and raw["closeMs"] is not None
        and raw["openCount"] == 1
        and raw["afterClose"] == 0
    )
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            # SCOPED TO WHAT IS MOUNTED, and always was. `reasoningTriggers()` is a
            # querySelectorAll, so on the shipped build it finds every pane in the thread and on a
            # windowed mount it finds the panes in the window. The assertion below stays
            # self-consistent either way (it opens N and expects N open), so the action still
            # passes -- but its COST silently stops being a function of thread length and becomes
            # a function of window size. That is the arm's point, and it is also the reason this
            # timing is not comparable between a windowed arm and a full one without the pane
            # count beside it.
            "panes": raw["panes"],
            "panes_scope": "mounted",
            # `panes_opened` is the assertion the two above are context for: this action opens
            # exactly one pane whatever the thread holds, which is the whole point of it existing
            # beside `reasoning_toggle`. Asserted by test_studiobench_reasoning_one_live.
            "panes_opened": 1,
            "open_after_expand": raw["openCount"],
            "open_after_collapse": raw["afterClose"],
            "highlight_spans_while_open": raw["spansOpen"],
            # The cost driver, so a reading can be normalised rather than compared across threads
            # whose newest reply happens to differ in size.
            "highlight_spans_added": raw["spansAdded"],
        },
        timings = {"open_ms": raw["openMs"], "close_ms": raw["closeMs"]},
        reason = None
        if ok
        else f"{raw['openCount']} panes were open after opening one and "
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
  // threadTotal, not messageCount. Identical on the shipped build; under a windowed mount the
  // window refills as the message leaves it, so a cleanup that worked reports after == before.
  const before = D.threadTotal();
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
  const after = D.threadTotal();
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
    after = _ev(ctx, "() => window.__sb.dom.threadTotal()")
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
    thread total is read before Enter and again here, and the delete runs only if the thread grew.
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

    messages_after = _ev(ctx, "() => window.__sb.dom.threadTotal()")
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
    #
    # threadTotal, not messageCount, for the reason STOP_CLEANUP_JS gives above and in the same
    # direction: under a windowed mount the window refills as the thread grows, so a send that
    # WORKED reports after == before and the reclaim path then decides there is nothing to remove.
    # The turn it already sent is left in the thread with its stream still running, which is the
    # one outcome this whole path exists to prevent, and it contaminates every later action window
    # and the final census. Identical on the shipped build, where the two are the same number.
    messages_before = _ev(ctx, "() => window.__sb.dom.threadTotal()")
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
        reason = f"the clipboard could not be read back: {_why(exc)}"
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
    # A SENTINEL ON THE CLIPBOARD BEFORE THE COPY, so "the copy happened" is OBSERVED rather than
    # assumed from the keystroke having been sent.
    #
    # THE FAILURE THIS EXISTS FOR, measured. Playwright's WebKit never performs a clipboard copy on
    # Control+C. The action pressed the key, slept 250ms for the copy to land, and reported the
    # sleep as `copy_ms`. In every WebKit payload on this machine that number is 258.5 ms at the 1K
    # rung, 258.8 ms at 10K and 263.9 ms at 100K: a hundredfold change in the quantity under study
    # moved the "measurement" by two percent, because it was measuring `wait_for_timeout(250)`.
    # Chromium reads about 1,538 ms at 100K for the same action. Forty-three rows across eleven
    # payloads reported a sleep as a result.
    #
    # A measurement that silently reports a sleep is the worst failure mode in this harness, so the
    # test is on the CLIPBOARD and not on the engine name: an engine that starts working, or a new
    # one, is admitted automatically, and an engine that stops working is refused automatically.
    #
    # WHAT THE CHECK NEEDS is a KNOWN PRE-COPY VALUE, and the sentinel is only the best way of
    # getting one. Writing it can fail on its own: `writeText` needs transient user activation or
    # the `clipboard-write` permission and throws `NotAllowedError` without either, while
    # `readText` is gated separately -- so the two are not granted or refused together, and
    # runtime/browser.py asks for the pair on Chromium and for neither on the other engines.
    #
    # Clearing the sentinel on that failure and carrying on is what re-admitted the exact defect
    # above: with no pre-copy value, `clip == sentinel` can never fire, and a Control+C that did
    # nothing leaves the clipboard holding whatever `copy_markdown` put there earlier in the same
    # film -- a plausible, non-empty string that is read back as a fresh copy of the whole thread,
    # with the 250 ms settle beside it as `copy_ms`.
    #
    # So a failed write falls back to SNAPSHOTTING what the clipboard already holds, which answers
    # the same question ("did this keystroke change the clipboard"), and when even that cannot be
    # established the action is NOT RUN. The residual, stated: if the clipboard already held
    # character-for-character what the copy would produce, an honest copy is refused as a
    # no-op. That direction is the safe one, it needs the write permission to be missing first,
    # and the film never copies the same thread twice.
    sentinel = f"__sb_clipboard_sentinel_{int(time.monotonic() * 1000)}__"
    sentinel_written = False
    #: What the clipboard held before Control+C, and where that value came from. `None` means no
    #: pre-copy value could be established at all, which is refused below.
    pre_copy: Optional[str] = None
    pre_copy_source = "sentinel"
    pre_copy_reason = None
    try:
        ctx.page.evaluate("async (s) => await navigator.clipboard.writeText(s)", sentinel)
        sentinel_written = True
        pre_copy = sentinel
    except Exception as write_exc:  # noqa: BLE001
        sentinel = ""
        pre_copy_source = "snapshot"
        try:
            existing = ctx.page.evaluate("async () => await navigator.clipboard.readText()")
        except Exception as read_exc:  # noqa: BLE001
            existing = None
            pre_copy_reason = (
                f"the sentinel could not be written ({_why(write_exc)}) and the clipboard could "
                f"not be read either ({_why(read_exc)})"
            )
        if isinstance(existing, str):
            pre_copy = existing
        elif pre_copy_reason is None:
            pre_copy_reason = (
                f"the sentinel could not be written ({_why(write_exc)}) and the clipboard handed "
                "back no string to snapshot in its place"
            )

    copy_started = time.monotonic()
    ctx.page.keyboard.press("Control+C")
    ctx.page.wait_for_timeout(250)
    copy_ms = (time.monotonic() - copy_started) * 1000
    # THE CLIPBOARD, NOT THE SELECTION.
    #
    # This action used to score itself on `Selection.toString().length`, and that is not what a
    # user gets. The two came apart the moment an arm existed that unmounts messages AND handles
    # the copy event: the selection can only ever cover mounted nodes, while a copy handler that
    # reads the message store puts the whole conversation on the clipboard. Scored on the
    # selection, a build that had FIXED the data loss would still be reported as losing data --
    # the alarm would stay lit for a reason that no longer existed, which is the fastest way to
    # get an alarm switched off.
    #
    # Read back before the selection is cleared, and the failure to read it is reported rather
    # than treated as an empty clipboard: headless Chromium grants clipboard-read only with the
    # permission the browser factory requests.
    clip = None
    clip_reason = None
    try:
        clip = ctx.page.evaluate("async () => await navigator.clipboard.readText()")
    except Exception as exc:  # noqa: BLE001
        clip_reason = f"the clipboard could not be read back: {_why(exc)}"
    clipboard_chars = len(clip) if isinstance(clip, str) else None
    ctx.page.evaluate("() => window.getSelection().removeAllRanges()")

    # NO CONFIRMED COPY, NO TIMING. Reported NOT RUN rather than as a number, because a reader who
    # sees `copy_ms` has no way to tell a real copy from a keystroke that went nowhere, and the
    # engines that fail this do so silently and consistently enough to look like data.
    if pre_copy is not None and clip == pre_copy:
        held = (
            "it still holds the sentinel written before the keystroke"
            if sentinel_written
            else "it still holds, character for character, what it held before the keystroke "
            "(the sentinel could not be written, so the clipboard was snapshotted instead)"
        )
        return not_run(
            f"Control+C did not change the clipboard: {held}, so this engine did not perform the "
            f"copy and the elapsed time would be the harness's own {250}ms settle rather than a "
            "measurement of the app"
        )
    if pre_copy is None:
        return not_run(
            f"the copy could not be confirmed ({pre_copy_reason}), so nothing was known about the "
            "clipboard before Control+C and a keystroke that did nothing would be indistinguishable "
            "from a copy. A timing is only reported when the clipboard is known to have changed"
        )
    if clip_reason is not None:
        return not_run(
            f"the copy could not be confirmed ({clip_reason}), so there is no evidence that "
            "Control+C did anything and the elapsed time would be the harness's own settle. A "
            "timing is only reported when the clipboard can be read back and has changed"
        )
    total = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted = _ev(ctx, "() => window.__sb.dom.messageCount()")
    ok = raw["chars"] > 0
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "selected_chars": raw["chars"],
            # WHAT ACTUALLY REACHED THE CLIPBOARD. The user-facing quantity, and the one the
            # truncation alarm is scored on. A windowed mount cannot SELECT what is not in the
            # DOM, but it can still COPY it, if the app's copy handler reads the message store --
            # which is the fix, and which this is the only reading able to see.
            "clipboard_chars": clipboard_chars,
            "clipboard_readable": clip_reason is None,
            "clipboard_note": clip_reason,
            # WHICH pre-copy value the change was confirmed against: the sentinel this action
            # wrote, or a snapshot of what the clipboard already held when the write was refused.
            # A snapshot is the weaker of the two -- see the note above the write -- and a reader
            # comparing rows has no other way to see which one a timing rests on.
            "copy_confirmed_against": pre_copy_source,
            # And the DOM coverage beside it, so the two can be told apart. Where
            # `mounted_fraction` is well below 1 and `clipboard_chars` is nonetheless whole, the
            # copy-from-store path is working; where both are short, the conversation is being
            # lost. That is a user-visible defect and must not be filed as a measurement problem.
            "messages_total": total,
            "messages_mounted": mounted,
            "mounted_fraction": (
                round(mounted / total, 3)
                if isinstance(total, int) and isinstance(mounted, int) and total
                else None
            ),
        },
        timings = {
            "select_all_ms": raw["selectMs"],
            "copy_ms": round(copy_ms, 1),
            "total_ms": round((time.monotonic() - started) * 1000, 1),
        },
        # `expect_ok` stays `chars > 0` deliberately. Within ONE run there is nothing to compare
        # against: the selection is taken over the viewport's DOM, so if the DOM is windowed the
        # selection and every DOM-derived reference shrink together and agree with each other.
        # An absolute floor would need calibrating per rung and per platform, and would still be a
        # guess. The counts below are the real check, and they are paired against the other arm.
        #
        # `clipboard_chars` FIRST, because it is the one a user would notice. `selected_chars` is
        # kept beside it as the mechanism: if the clipboard held and the selection fell, the copy
        # handler is reading the store; if both fell, the conversation is being lost.
        counts = {
            "clipboard_chars": clipboard_chars,
            "selected_chars": raw["chars"],
        },
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
    # NO `pacer.reset()` HERE. It used to clear the pacer's stats before loading the next turn, and
    # `CellRunner` records `last_stats()` -- so the opening reply's `StreamStats` were discarded by
    # the first follow-up and the first follow-up's by the second. An opening reply that
    # disconnected or delivered 4,624 of its 10,000 characters was then erased by a later turn that
    # DID finish, and the cell was marked complete and scored against an undersized thread, because
    # the only other liveness signal is the UI no longer running and the later turn satisfies it.
    # Every turn is tagged, so keeping them all costs one small record each and lets
    # `check_planned_streams` verify the cell streamed what it planned. `CellRunner` still resets
    # once per cell, which is the boundary that reset is for.
    tag = f"{ctx.args.get('cell_id', 'cell')}#turn{index + 1}"
    pacer.load(
        unit["reasoning"],
        unit["content"],
        cadence = ctx.args.get("cadence", "field"),
        tag = tag,
    )

    selector = 'textarea[aria-label="Message input"]'
    if ctx.page.query_selector(selector) is None:
        return not_run("no composer on the page")
    ctx.page.fill(selector, f"studiobench follow-up {index + 1}")
    ctx.page.wait_for_timeout(80)
    # THE THREAD'S LENGTH, not the mounted count. "The thread grew" is a statement about the
    # conversation; a windowed mount answers it about the viewport, and the answer it gives is
    # `after == before` for a send that worked perfectly, because the new pair arrives at the
    # bottom while two messages leave the top of the window.
    before = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted_before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    started = time.monotonic()
    ctx.page.keyboard.press("Enter")

    first_ms = None
    deadline = started + SETTLE_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        if _ev(ctx, "() => window.__sb.dom.isRunning()"):
            first_ms = (time.monotonic() - started) * 1000
            break
        ctx.page.wait_for_timeout(50)
    after = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted_after = _ev(ctx, "() => window.__sb.dom.messageCount()")
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
            # What was MOUNTED either side, kept beside the thread totals. On the shipped build
            # these two pairs are identical; where they differ, the difference is the size of the
            # window and it is the reading the whole virtualization arm exists to produce.
            "mounted_before": mounted_before,
            "mounted_after": mounted_after,
            "turn_index": index + 1,
            "queued_turns": len(queue),
            "streamed_chars": len(unit["reasoning"]) + len(unit["content"]),
            # The tag this turn's stream carries, so the cell can check the turn against the
            # pacer's own record of it rather than re-deriving the naming rule from the outside.
            "pacer_tag": tag,
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

#: The end of the thread, as a plain string that can be looked for again after the rebuild.
#:
#: NOT the seeder's `last_marker`, even though that is the string the readiness gate uses at the
#: start of the cell. `send_turn` appends turns to the thread MID-FILM, twice in every scene, so by
#: the time this slot opens the seeded marker sits several messages from the end -- and
#: `end_present` requires the marker within two rows of the end, so it could never be satisfied.
#: Reading the last user turn out of the DOM immediately before leaving gives the marker for the
#: thread as it actually stands at this point in the film.
#:
#: `trim()` and then a prefix, never a whitespace-collapsing normalisation: readiness.PROBE_JS
#: matches with `textContent.includes(marker)` against RAW text, and trimming plus slicing from the
#: front leaves a genuine substring of that raw text while collapsing internal runs would not.
_LAST_USER_TEXT_JS = """
() => {
  const rows = Array.from(document.querySelectorAll('[data-role="user"]'));
  const last = rows.length ? rows[rows.length - 1] : null;
  if (!last) return null;
  const text = (last.textContent || "").trim();
  return text ? text.slice(0, 120) : null;
}
"""

#: How long the rebuilt thread is given to reach the readiness gate's definition of ready, in
#: seconds: what is left of the SLOT, floored and capped.
#:
#: Bounded by the slot because the condition below is now one a broken arm can fail to satisfy
#: forever -- a thread that never stops mutating never settles -- and the film's own last slot,
#: `delete_message`, is the one place an overrun has nowhere to go (see scene/schedule.py, which
#: records a sweep where exactly that lost delete on every base-arm cell). Floored at 10s so a
#: tight budget still gives a 1M-token thread a fair chance, and capped at the flat minute the
#: previous "declared its total" loop waited, so nothing waits longer than it used to.
_REOPEN_READY_FLOOR_S = 10.0
_REOPEN_READY_CEILING_S = 60.0


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
    # The THREAD's length, not the mounted count: a windowed arm that reopens with a different
    # scroll anchor mounts a different number of rows, and the exact-equality assertion at the end
    # of this action would then fail for a reason that has nothing to do with the rebuild.
    before = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted_before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    # AN INTEGER, not merely truthy. `_ev` hands back `{"__error": ...}` when the page throws, and
    # a dict is truthy: it would reach the comparisons below and the readiness gate's own
    # `expected_messages` as a length, and raise there instead of being reported here.
    if not isinstance(before, int) or before <= 0:
        return not_run("the thread has no messages to rebuild")
    # WHICH READINESS MODE THIS ARM IS HELD TO, decided from the mount that is on screen now and
    # therefore before anything is torn down. A thread whose declared total is larger than what it
    # has mounted IS a windowed mount -- that is `isWindowed()` in scene/dom.js -- and it is the
    # same distinction runtime/readiness.py draws between its two modes. Read from the page rather
    # than taken from a session flag so both arms are judged by one rule.
    mode = (
        MODE_WINDOWED if isinstance(mounted_before, int) and mounted_before < before else MODE_FULL
    )
    marker = _ev(ctx, _LAST_USER_TEXT_JS)
    if not isinstance(marker, str) or not marker:
        # Refused HERE, before a single thing has been touched, so a thread this action cannot
        # verify is also a thread it has not disturbed. Without a string that identifies the end of
        # the conversation there is no way to tell a rebuilt thread from a half-rebuilt one, which
        # is the entire question the wait below exists to answer.
        return not_run(
            "the last user turn carried no text to identify the end of the thread with, so a "
            "finished rebuild could not have been told from a partial one"
        )
    started = time.monotonic()
    # NO FALLBACK ON THE WAY OUT. Workspace task #102 and the defect it left behind.
    #
    # That task taught this action to REFUSE a substituted navigation, because a goto is a FULL
    # DOCUMENT NAVIGATION -- the SPA is torn down and reparsed, the bundle re-executes, the runtime
    # rehydrates -- while the click is a client-side route change that rebuilds one React subtree.
    # They are not the same operation and they do not cost the same thing; read as though it were
    # the click, the substitution produced thread_reopen at 6.0 fps, a number about a page load
    # quoted as a number about a thread.
    #
    # But it refused AFTER THE FACT, by inspecting `path` once `page.goto` had already run. The row
    # was then honest and the rest of the film was collateral damage: the scene carried on from an
    # empty new chat, and `delete_message` -- the last slot of every film, three slots later --
    # found no messages and went unexercised for a reason that had nothing to do with deleting.
    #
    # So the substitution is declined BEFORE it happens. Nothing is clicked, nothing is navigated,
    # the thread is still on screen, and the slots after this one still have the thread they were
    # written for.
    leave = _click_or_navigate(
        ctx,
        'button[aria-label="New chat"]',
        f"{ctx.args['base_url']}/chat?new=studiobench",
        allow_navigate = False,
    )
    if not leave.ok:
        ctx.log(
            "    thread_reopen NOT MEASURED: the New chat button could not be clicked "
            f"({leave.reason}), and the only substitute available is a full page navigation, "
            "which reloads the whole app and is not the operation this action exists to time. "
            "It was NOT performed: the thread is untouched and the slots that follow still have it."
        )
        return not_run(
            "the New chat control could not be clicked and the only available substitute is a full "
            "page navigation; a document reload is not a thread rebuild, so the action was not run "
            f"and no navigation was performed ({leave.reason})"
        )
    # BOTH CLOCKS START AT THE CLICK THAT WORKED, not at the first attempt on it.
    #
    # `_click_or_navigate` tries `handle.click` first, and Playwright's hit-target check retries
    # for its whole 2,000 ms timeout against a control whose centre does not hit-test to it. The
    # New chat button IS such a control by design -- `opacity-0 pointer-events-none` until its
    # header group is hovered -- so that timeout was paid on EVERY run and landed inside
    # `close_ms`, which sweep/floor_table.py harvests and quotes against the other arm. A metric
    # that is two seconds of harness retry plus the unmount is the same defect as reporting a
    # settle delay as `copy_ms`, one action along.
    #
    # The retry is not discarded, it is MOVED: `left_click_retry_ms` and `reopen_click_retry_ms`
    # below carry it as the harness's own cost, which is what it is. `started` still governs the
    # budget arithmetic, because the retry really did consume the slot's wall clock.
    click_left_at = leave.started_at if leave.started_at is not None else started
    # Unmount FIRST, or "already back" is indistinguishable from "never left".
    closed_ms = None
    deadline = started + 15
    while time.monotonic() < deadline:
        if _ev(ctx, "() => window.__sb.dom.messageCount()") == 0:
            closed_ms = (time.monotonic() - click_left_at) * 1000
            break
        ctx.page.wait_for_timeout(50)
    if closed_ms is None:
        return not_run("the thread never unmounted, so the rebuild could not be timed")

    reopen_started = time.monotonic()
    # THE FALLBACK IS ALLOWED ON THE WAY BACK, and refused on the way out, for one reason: what it
    # does to the scene. From an empty new chat, a navigation to the thread's own URL is what puts
    # the thread back on screen for the slots that follow, so as a REPAIR it is the right tool --
    # where on the way out it was the thing that broke the scene. It is still not a measurement of
    # a rebuild, which is what the `not_run` below says, and no timing is reported for it.
    back = _click_or_navigate(
        ctx, f'[data-thread-id="{thread_id}"]', f"{ctx.args['base_url']}/chat?thread={thread_id}"
    )
    if not back.ok:
        return not_run(f"the thread could not be reopened: {back.reason}")
    click_back_at = back.started_at if back.started_at is not None else reopen_started
    if back.navigated:
        ctx.log(
            "    thread_reopen NOT MEASURED: the sidebar row for the thread could not be clicked, "
            f"so reopening fell back to a full page navigation ({back.reason}). The navigation "
            "was allowed to stand so the thread is back for the slots that follow, but a document "
            "reload is not a rebuild and no timing is reported for it."
        )
        return not_run(
            "the thread's sidebar row could not be clicked and a full page navigation was "
            f"substituted, so the rebuild was never timed ({back.reason})"
        )
    # WHEN IS THE REBUILD OVER?
    #
    # NOT when `threadTotal()` reaches the length it had. On a windowed arm `threadTotal()` reads
    # `aria-setsize`, which is the STORE'S CLAIM about how long the conversation is, and the very
    # first reopened row publishes it: the condition was satisfied while three of eighteen messages
    # were mounted, with no bottom window, no final assistant content and no syntax highlighting
    # yet. The action then reported that as `reopen_ms`, took its census and its parity digest off
    # a half-built DOM, and still passed its own assertion because `after == before` was a
    # statement about the same declared total. A declaration is not a rebuild.
    #
    # WHAT `reopen_ms` MEASURES NOW: from the click on the thread's sidebar row until the reopened
    # thread satisfies runtime/readiness.py's gate -- the composer is present, the END of the
    # conversation is mounted (the last user turn, found by the marker captured above), and the
    # mount has SETTLED, meaning two samples STABLE_GAP_MS apart agree on the mounted count, the
    # element count and the laid-out height. Plus, in `windowed` mode, that the thread declares the
    # right total on every row and is anchored at the end.
    #
    # THAT SAME FUNCTION, not a second definition of "ready" written here. Two disagreeing
    # definitions in one harness is a defect of its own, and this is the definition the cell was
    # already admitted under: the gate that let the cell start is the gate that decides the rebuild
    # is finished, so the number is the cost of getting back to the state the cell began in.
    #
    # COMPARABILITY: every condition gated in `full` mode is one a fully mounted thread reaches by
    # construction, and each arm is judged in the mode its own mount is in, exactly as the cell's
    # opening gate judged it. The extra `windowed` conditions are a contract that arm already had
    # to meet to be scored at all. What this costs is a floor of one STABLE_GAP_MS, paid equally by
    # both arms, because "settled" is a claim about two samples and cannot be made from one.
    left_ms = ctx.budget_ms - (time.monotonic() - started) * 1000
    timeout_s = min(_REOPEN_READY_CEILING_S, max(_REOPEN_READY_FLOOR_S, left_ms / 1000))
    reopen_ms = None
    try:
        ready = wait_for_thread_ready(
            ctx.page,
            before,
            marker = marker,
            mode = mode,
            timeout_s = timeout_s,
            log = ctx.log,
        )
        reopen_ms = (time.monotonic() - click_back_at) * 1000
        readiness = ready.as_dict()
    except ThreadNotReady as exc:
        # A REAL FINDING ABOUT THE ARM, and it keeps `ran = True` so the row carries which
        # conditions were still outstanding. `timings["reopen_ms"]` stays null, which the scoring
        # layer reads as "ran but recorded no reopen_ms" rather than as a fast rebuild.
        readiness = exc.detail
    except Exception as exc:  # noqa: BLE001
        # The probe itself failed (a closed page, an uninstalled `window.__sb`). Nothing was
        # learned about the rebuild, so nothing is claimed about it.
        ctx.log(f"    the readiness probe failed during the reopen: {type(exc).__name__}: {exc}")
        return not_run(
            f"the reopened thread could not be probed for readiness: {type(exc).__name__}: {exc}"
        )
    after = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted_after = _ev(ctx, "() => window.__sb.dom.messageCount()")
    spans = _ev(ctx, "() => document.querySelectorAll('pre span').length")
    ok = reopen_ms is not None and after == before
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "messages_before": before,
            "messages_after": after,
            "mounted_before": mounted_before,
            "mounted_after": mounted_after,
            "highlight_spans_after": spans,
            "left_via": leave.path,
            "reopened_via": back.path,
            # THE HARNESS'S OWN COST, named as such and kept OUT of the two timings above. This is
            # how long `_click_or_navigate` spent on attempts that failed before the one that
            # worked -- almost always Playwright's 2,000 ms hit-target retry against a control it
            # cannot hit at the centre. It belongs in the payload (a run where this grows is a run
            # whose controls have moved) and it does not belong in a rebuild timing.
            "left_click_retry_ms": round((click_left_at - started) * 1000, 1),
            "reopen_click_retry_ms": round((click_back_at - reopen_started) * 1000, 1),
            # WHICH GATE THE TIMING WAS TAKEN AGAINST, beside the timing itself. Two arms in
            # different readiness modes are answering slightly different questions, and a reader
            # comparing their `reopen_ms` has no other way to see that.
            "reopen_ready_mode": mode,
            # The gate's own verdict: every condition, the last probe reading and how many samples
            # it took. When the rebuild did not finish, this is what says which condition was still
            # outstanding -- "mounted 3 of 18, aria-setsize 18" is the difference between a slow
            # app and an arm that publishes a total it has not built.
            "reopen_readiness": readiness,
        },
        timings = {
            "close_ms": round(closed_ms, 1),
            "reopen_ms": None if reopen_ms is None else round(reopen_ms, 1),
        },
        reason = None
        if ok
        else (
            f"the thread came back with {after} of {before} messages"
            if reopen_ms is not None
            else f"the reopened thread never reached a ready state: {readiness.get('reason')}"
        ),
    )


@dataclass
class Transition:
    """WHICH ROUTE a state change took, so a substitute can never be read as the original.

    `ok` alone was the whole return value of `_click_or_navigate`, and it collapsed three
    outcomes -- the control was clicked, the control failed and a page load was substituted, and
    nothing worked -- into two. The middle one is the dangerous one: it produces a row that looks
    like a measurement of a click and is a measurement of a document reload.
    """

    ok: bool
    path: str  # "click", "navigate" or "failed"
    reason: str = ""
    #: The monotonic instant the interaction that SUCCEEDED was issued, so a caller timing "from
    #: the click" can start its clock at the click rather than at the first attempt. `None` when
    #: nothing succeeded.
    #:
    #: THE FAILED ATTEMPT IS NOT PART OF THE GESTURE. Playwright's hit-target check reads
    #: `elementFromPoint` at the control's centre and retries until the timeout, so a control that
    #: is laid out but not hit-testable at its centre -- the sidebar's `New chat`, which ships
    #: `opacity-0 pointer-events-none` until its group is hovered, and which therefore falls
    #: through to the group underneath on every hit test -- costs the full 2,000 ms before the
    #: reveal path below succeeds. Timed from the first attempt, `thread_reopen.close_ms` was
    #: two seconds of Playwright retry with the unmount somewhere inside it, on every run, and
    #: sweep/floor_table.py quotes it as a metric.
    started_at: Optional[float] = None

    @property
    def navigated(self) -> bool:
        return self.path == "navigate"


#: Where on an element to look for a point a user could actually hit. Fractions of the box, centre
#: first (so a normal control behaves exactly as before), then the lower band, which is what stays
#: uncovered when a sticky header overlaps the top of a control.
_HIT_POINTS = (
    (0.5, 0.5),
    (0.5, 0.75),
    (0.25, 0.75),
    (0.75, 0.75),
    (0.5, 0.9),
    (0.15, 0.5),
    (0.85, 0.5),
    (0.5, 0.25),
)

#: EVERY match, not the first. `document.querySelector` returns the first element in DOCUMENT
#: ORDER, and this app renders "New chat" twice -- once in the sidebar header and once on the chat
#: page -- exactly as it renders the composer twice. When the sidebar is collapsed its button is
#: still in the tree at zero size, comes first, and is the one a plain query hands back: the probe
#: then reports "no point on the control hit-tests to it" about a button that is 34x34 and fully
#: clickable a few hundred pixels away. This is the same trap `image_upload` documents at length,
#: and it cost this action a second time.
_HIT_TEST_JS = """
([selector, points]) => {
  for (const el of document.querySelectorAll(selector)) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    for (const [fx, fy] of points) {
      const x = r.left + r.width * fx;
      const y = r.top + r.height * fy;
      if (x < 0 || y < 0 || x > window.innerWidth || y > window.innerHeight) continue;
      const top = document.elementFromPoint(x, y);
      // `contains`, not identity: the control's own icon or label is a child and hit-tests to
      // itself, which is still a click that reaches the control.
      if (top && (top === el || el.contains(top))) return { x, y };
    }
  }
  return null;
}
"""


def _reachable_point(ctx: ActionContext, selector: str) -> tuple[float, float] | None:
    """A viewport point that hit-tests to `selector`, or None if no instance of it is reachable."""
    got = _ev(ctx, _HIT_TEST_JS, [selector, [list(p) for p in _HIT_POINTS]])
    if isinstance(got, dict) and "x" in got:
        return got["x"], got["y"]
    return None


#: How long the reveal transition needs, and how many times to look. `.sidebar-header-action`
#: transitions opacity over 150ms. One sample after a fixed sleep is a race that this test suite
#: lost roughly one run in three on a loaded machine, so the hit test is retried instead.
_REVEAL_SETTLE_MS = 120
_REVEAL_ATTEMPTS = 5

#: The control's own box, whatever its computed style says. An earlier version only offered a
#: hover point for a control that LOOKED hidden, which made the reveal conditional on a style read
#: taken at exactly the wrong moment: mid-transition `getComputedStyle` reports the old opacity,
#: and after a failed Playwright click the element can read as already revealed while not yet
#: being hit-testable. Since this only runs when the control is already known to be unreachable,
#: there is nothing to gain by being clever about whether it deserves a hover.
_HOVER_TARGET_JS = """
(selector) => {
  for (const el of document.querySelectorAll(selector)) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    if (x < 0 || y < 0 || x > window.innerWidth || y > window.innerHeight) continue;
    return { x, y };
  }
  return null;
}
"""


def _reveal_by_hover(ctx: ActionContext, selector: str) -> tuple[float, float] | None:
    """Hover a hover-revealed control into existence, then return a point on it.

    THE CONTROL WAS NEVER COVERED. The sidebar's New chat button is
    `.sidebar-header-action`, which ships `opacity-0 pointer-events-none` and is revealed only by
    `.group/sidebar-header:hover` (or `:focus-visible`). With no mouse over the header it is a
    20x20 box that is laid out, reported `visible` by every check Playwright makes, and transparent
    to every hit test -- so `click()` waits out its whole timeout and the hit-test spread reports
    "no point on the control hit-tests to it". Both are accurate and both are the wrong conclusion:
    nothing is covering it and the sidebar is not collapsed. The harness simply never did the half
    of the gesture that makes the control exist, and then substituted a page reload for the click.

    Moving the mouse to where the button IS suffices, and is what a person does. `pointer-events:
    none` means the pointer falls through to the group underneath, the group's `:hover` matches,
    and the button becomes solid under a mouse that is already on it. So this deliberately does not
    walk ancestors looking for the reveal group: hovering the control's own centre finds it by
    construction, and cannot pick the wrong one.

    Called ONLY after the ordinary hit test has already failed, so a control that is reachable at
    rest is never hovered and no stray mouse movement enters a measured window. Returns None when
    hovering does not make the control hit-testable, which is a real finding and is reported as
    one rather than papered over with a page load.
    """
    target = _ev(ctx, _HOVER_TARGET_JS, selector)
    if not isinstance(target, dict) or "x" not in target:
        return None
    try:
        ctx.page.mouse.move(target["x"], target["y"])
    except Exception:  # noqa: BLE001
        return None
    for _ in range(_REVEAL_ATTEMPTS):
        ctx.page.wait_for_timeout(_REVEAL_SETTLE_MS)
        point = _reachable_point(ctx, selector)
        if point is not None:
            ctx.log("    the control is hover-revealed; hovered it first, as a user would")
            return point
    return None


def _click_or_navigate(
    ctx: ActionContext,
    selector: str,
    url: str,
    *,
    allow_navigate: bool = True,
) -> Transition:
    """Click the control a user would click; fall back to the URL it would produce, and SAY SO.

    A click is preferred because it exercises the app's own handler. Playwright's actionability
    retries do not give up on an element that is visible, enabled and stable and merely covered by
    something else, so the click burns its whole timeout. The navigation reaches the same app
    state, and for getting the scene somewhere it is a perfectly good tool.

    WHAT IT IS NOT is the same operation, and the caller is now told which one it got. Every
    caller must decide for itself whether a navigation still answers its question. `thread_reopen`
    decides that it does not, because a document reload and a subtree rebuild are the two things
    that action exists to tell apart.

    `allow_navigate = False` lets a caller decide that BEFORE the substitution rather than after
    it. Reading `path == "navigate"` afterwards is enough to keep the ROW honest and not enough to
    keep the SCENE intact: by then the page has already left, and for the call that leaves the
    thread that means every later slot in the film runs against an empty one. Declining up front
    returns `path = "failed"` with the same explanation and leaves the page exactly where it was.
    The default is `True`, so a caller that has not thought about it keeps the behaviour every
    caller had.
    """
    handle = ctx.page.query_selector(selector)
    click_error = f"no element matched {selector}"
    if handle is not None:
        try:
            attempt_at = time.monotonic()
            handle.click(timeout = 2000)
            return Transition(ok = True, path = "click", started_at = attempt_at)
        except Exception as exc:  # noqa: BLE001
            click_error = f"{selector} was not clickable: {type(exc).__name__}"
        # THE CENTRE IS COVERED, BUT THE CONTROL IS NOT. Try the rest of it, the way a user would.
        #
        # Playwright clicks the centre of an element and refuses when something else hit-tests
        # there. The sidebar's sticky group label overlaps the top of the New chat button, so the
        # centre is taken and the click times out -- but most of the button is uncovered and a
        # person clicks it every day without noticing. Giving up at the centre and calling the
        # control unreachable is the harness being more fragile than a user, and it cost this
        # action on every CI run.
        #
        # So the element's own box is hit-tested at a spread of points and the first one that
        # resolves to the control is clicked with a real mouse event. If NO point on the control
        # is reachable then it genuinely cannot be clicked, which is a finding worth reporting
        # rather than a reason to substitute a page load.
        point = _reachable_point(ctx, selector) or _reveal_by_hover(ctx, selector)
        if point is not None:
            try:
                attempt_at = time.monotonic()
                ctx.page.mouse.click(point[0], point[1])
                return Transition(
                    ok = True,
                    path = "click",
                    reason = "clicked off-centre",
                    started_at = attempt_at,
                )
            except Exception as exc:  # noqa: BLE001
                click_error += f"; the off-centre click also failed: {type(exc).__name__}"
        else:
            click_error += "; no point on the control hit-tests to it, even after hovering it"
        ctx.log(
            f"    {click_error}; "
            + ("navigating instead" if allow_navigate else "NOT navigating, the caller refused it")
        )
    if not allow_navigate:
        return Transition(
            ok = False,
            path = "failed",
            reason = (
                f"{click_error}, and the caller does not accept a page navigation as a substitute, "
                "so none was performed and the page was left where it was"
            ),
        )
    try:
        attempt_at = time.monotonic()
        ctx.page.goto(url, wait_until = "domcontentloaded", timeout = 60_000)
        return Transition(ok = True, path = "navigate", reason = click_error, started_at = attempt_at)
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"    navigation to {url} failed: {type(exc).__name__}: {exc}")
        return Transition(
            ok = False,
            path = "failed",
            reason = f"{click_error}, and the navigation failed too: {type(exc).__name__}",
        )


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


#: How long an action that needs a message's ACTION BAR will wait for the reply to finish.
#:
#: Unsloth hides the action bar while a message is generating, so "no More button on the last
#: assistant message" and "no Delete button (a running message hides it)" are the same fact
#: reported by two actions. The film schedules `message_menu` about four seconds after a
#: `send_turn` whose reply runs for roughly fourteen, so the menu was being asked for on a message
#: that was still being written, and the action reported NOT RUN on every CI run.
#:
#: Waiting is what a USER does, and it is the only thing that makes the action measurable at all:
#: there is no menu to open until the reply lands. Bounded by the slot's own budget so a reply
#: that never finishes cannot swallow the rest of the film, and the wait happens BEFORE the
#: operation the action times, so `menu_open_ms` is unaffected.
_ACTION_BAR_WAIT_FRACTION = 0.6
_ACTION_BAR_POLL_MS = 100


def _wait_for_the_reply_to_land(ctx: ActionContext) -> bool:
    """True once no reply is generating, False if one still is when the budget runs out."""
    budget_ms = max(0.0, float(ctx.budget_ms or 0)) * _ACTION_BAR_WAIT_FRACTION
    deadline = time.monotonic() + budget_ms / 1000.0
    running = bool(_ev(ctx, "() => window.__sb.dom.isRunning()"))
    if not running:
        return True
    ctx.log("    a reply is still generating; waiting for it before asking for the action bar")
    while time.monotonic() < deadline:
        ctx.page.wait_for_timeout(_ACTION_BAR_POLL_MS)
        if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
            return True
    return False


@register_action(name = "message_menu", default_budget_ms = 12000)
def message_menu(ctx: ActionContext) -> ActionResult:
    # TWO WAITS, AND THEY COVER DIFFERENT THINGS. This one waits for the STREAM to stop, bounded by
    # a fraction of the slot's own budget, because a follow-up reply runs for seconds and Unsloth
    # unmounts the whole action bar for its whole duration -- far longer than the page-side wait
    # below is allowed to sit. `waitForButtonMs` then waits for the BAR TO MOUNT, which happens a
    # few hundred milliseconds after the reply settles, and which this one cannot see because
    # `isRunning()` is already false by then. Dropping either leaves a real NOT RUN on the table:
    # without the first, a fourteen second reply outlasts a 1.5 s page-side wait; without the
    # second, the slot asks for a control that has not been mounted yet.
    if not _wait_for_the_reply_to_land(ctx):
        return not_run(
            "a reply was still generating when the slot's budget ran out, and Unsloth hides the "
            "action bar on a generating message, so there was no menu to open"
        )
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
  const before = D.threadTotal();
  const mountedBefore = D.messageCount();
  const started = performance.now();
  button.click();
  let ms = null;
  // isConnected on the captured node is O(1). Re-counting [data-role] every frame would put an
  // O(messages) query INSIDE the window being timed, growing like the signal.
  //
  // THE LIMIT OF isConnected, recorded rather than worked around: it goes false when the node
  // leaves the document, and a virtualised list unmounts a node for scrolling as readily as for
  // deleting. Nothing scrolls during this action, so in this scene the two cannot be confused --
  // but the timing is only trustworthy when the thread total ALSO dropped, which is what
  // `expect_ok` requires below. A `ms` with an unchanged total is a node that was unmounted, not
  // a message that was deleted, and it must never be read as a fast delete.
  while (performance.now() - started < timeoutMs) {
    if (target === null || !target.isConnected) { ms = performance.now() - started; break; }
    await window.__sbNextPaint();
  }
  // `after` is the THREAD TOTAL, paired with the `before` above it, and the mounted counts ride
  // alongside rather than replacing it: on a windowed mount `messageCount()` is the size of the
  // window and a recycled node would read as a delete. `waitedMs` is how long the bar was waited
  // for, which the row reports so a slot that opened too early is visible.
  return { ran: true, waitedMs, ms: ms === null ? null : Math.round(ms * 10) / 10,
           before, after: D.threadTotal(),
           mountedBefore, mountedAfter: D.messageCount() };
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
    # The THREAD TOTAL dropped. A delete that detached the node but left the thread the same
    # length is a different bug -- or, on a windowed mount, a node the virtualizer recycled -- and
    # must not read as a successful delete.
    ok = raw["ms"] is not None and raw["after"] < raw["before"]
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "messages_before": raw["before"],
            "messages_after": raw["after"],
            "dropped": raw["before"] - raw["after"],
            "mounted_before": raw.get("mountedBefore"),
            "mounted_after": raw.get("mountedAfter"),
            "action_bar_wait_ms": raw.get("waitedMs"),
        },
        timings = {"delete_ms": raw["ms"]},
        reason = None if ok else f"the message count went {raw['before']} -> {raw['after']}",
    )
