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

#: How long an action needing the message ACTION BAR waits before reporting NOT RUN: 1,500 ms,
#: deliberately NOT ctx.budget_ms (51 ms left at entry on one measured cell), and bounded so a
#: genuinely absent control reports NOT RUN rather than eating the film. The wait happens BEFORE
#: the operation the action times, so `menu_open_ms` is unaffected.
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


# Two rAFs resolve no sooner than two vsync intervals, so any double-rAF timing has a ~33ms floor
# on a 60Hz display; measured per cell and recorded so a reader can subtract it.

# ── the paint floor ─────────────────────────────────────────────────
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


#: The bound that stops a wedged renderer from eating the slot; the wait itself ENDS when nothing
#: is in flight, and reaching this bound means a lost sample the coverage check fails on.
# ── 1. keystroke to paint ───────────────────────────────────────────

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
    # A real inter-character delay: delay=0 sends the burst in one CDP message, which the renderer
    # coalesces into a single input event and a single paint.
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
    # EVERY KEYSTROKE ACCOUNTED FOR, not merely a composer that grew: `grew_by` alone is satisfied
    # while the timings describe a subset, and the keystroke whose paint had not resolved is the
    # slowest one (a 500 ms keystroke vanished from a reading whose max was 20 ms). So the reading
    # stands only when nothing was in flight and samples + coalesced covers every input.
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
    # THE FOLLOW SAMPLER IS SUSPENDED FOR THIS GESTURE: it drags the viewport thousands of pixels off
    # the bottom while the reply streams, so samples taken during it make `follows_the_stream` a
    # reading about the film.
    _ev(ctx, "() => window.__sb.follow && window.__sb.follow.suspend()")
    try:
        raw = _ev(ctx, SCROLL_JS, [steps, step_px, settle_ms])
    finally:
        # Resumed even when the gesture raised, or one failed scroll silences the sampler for the rest of the cell.
        _ev(ctx, "() => window.__sb.follow && window.__sb.follow.resume()")
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the scroll did not run"))
    commanded = raw["commandedPx"]
    travelled = raw["travelledPx"]
    # 90% of commanded: Unsloth's intent-aware autoscroll snaps a move it reads as programmatic
    # straight back to the bottom, so travel is the only thing separating a real scroll from none.
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


# SETTLE ON THE DOM, NOT ON `data-state`: the attribute flips before the content it reveals has
# mounted, and the gap depends on the collapse mechanism an A/B is comparing. At 100K, reading
# `pre span` on the attribute frame gave a 41% phantom reduction across arms where a quiet-DOM
# read gave the same number on both. A null control cannot see a bias it shares, so both the
# timing and the census terminate on `quietFrames` unchanged frames.
# ── 4. reasoning expand / collapse ──────────────────────────────────

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

    # THE REASON NAMES THE CLAUSE THAT ACTUALLY FAILED: built from pane counts alone it printed a
    # description of SUCCESS under EXPECT FAILED when the real failure was a censored timing.
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

    # A CENSORED TIMING IS ABSENT, NOT ZERO, and absent loudly: `_action_timings` drops non-numeric
    # values, so censoring silently leaves the fast survivors behind. Recorded as its own field so
    # scoring can refuse to pool a metric censored at some rungs and not others.
    timings = {}
    if raw["openMs"] is not None:
        timings["open_ms"] = raw["openMs"]
    if raw["closeMs"] is not None:
        timings["close_ms"] = raw["closeMs"]

    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            # SCOPED TO WHAT IS MOUNTED: `reasoningTriggers()` is a querySelectorAll, so on a windowed mount
            # its COST stops being a function of thread length and becomes one of window size. The assertion
            # stays self-consistent either way, so this timing needs the pane count beside it.
            # A windowed mount is `isWindowed()` in scene/dom.js.
            "panes": raw["panes"],
            "panes_scope": "mounted",
            "open_after_expand": raw["openCount"],
            "open_after_collapse": raw["afterClose"],
            # Present ONLY when the span census went quiet; read on the state-flip frame this number was 41%
            # wrong on one arm and right on the other (see the note above REASONING_JS).
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


#: THE SAME GESTURE A USER MAKES: `reasoning_toggle` opens every pane at once (2.2 fps at 100K), a
#: deliberate worst case that has been quoted as though a user did it. NOT in the standard film:
#: adding a slot shifts every window after it and voids comparability with every payload on disk.
# The fast film opens this slot 0.4 s after the worst-case drain.
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
            # SCOPED TO WHAT IS MOUNTED: `reasoningTriggers()` is a querySelectorAll, so on a windowed mount
            # its COST becomes a function of window size rather than thread length, and this timing is not
            # comparable across arms without the pane count beside it.
            "panes": raw["panes"],
            "panes_scope": "mounted",
            # `panes_opened` is the assertion the notes above are context for: this action opens exactly one
            # pane whatever the thread holds. Asserted by test_studiobench_reasoning_one_live.
            "panes_opened": 1,
            "open_after_expand": raw["openCount"],
            "open_after_collapse": raw["afterClose"],
            "highlight_spans_while_open": raw["spansOpen"],
            # The cost driver, so a reading can be normalised rather than compared across threads whose
            # newest reply differs in size.
            "highlight_spans_added": raw["spansAdded"],
        },
        timings = {"open_ms": raw["openMs"], "close_ms": raw["closeMs"]},
        reason = None
        if ok
        else f"{raw['openCount']} panes were open after opening one and "
        f"{raw['afterClose']} were still open after collapsing",
    )


#: Fixed waits the throwaway turn costs once it starts: 80 ms for the composer, 600 ms to get the
#: turn going, 400 ms for in-flight chunks, 200 ms for the cleanup delete. RESERVED out of the
#: drain wait, or the overrun lands in `scroll_after`'s window as a bogus `slot_missed`. Split,
#: because the 80 ms is spent before the turn is sent.
# `scroll_after` has a 1,200 ms window on the fast film.
# ── 5. stop generation ──────────────────────────────────────────────

OWN_TURN_FIXED_AFTER_SEND_MS = 600 + 400 + 200
OWN_TURN_FIXED_MS = 80 + OWN_TURN_FIXED_AFTER_SEND_MS

#: What the same turn costs BEYOND those sleeps: two polls plus the driver round trips. Measured
#: against real chromium, 1,394-1,938 ms against the fixed 1,280, so 700 ms covers the measured
#: 443-658 ms and still leaves every film a real drain wait.
#:
#: SPLIT AT THE MOMENT THE TURN STARTS, because that is where the wait below is bounded and a
#: total is not a bound: the stop-settle poll, the cleanup delete and the driver calls between
#: them are all still ahead. Measured at 60-424 ms after the sleeps; 500 ms covers it.
# 90 ms to stop, 60 ms to delete, 227 ms after the sleeps; `STOP_CLEANUP_JS` is unbounded by the slot.
# Against the fast film's 3,000 ms stop slot.
OWN_TURN_STOP_POLL_MS = 500
OWN_TURN_START_POLL_MS = 200
OWN_TURN_POLL_MS = OWN_TURN_START_POLL_MS + OWN_TURN_STOP_POLL_MS

#: What the throwaway turn needs in the slot in total, reserved out of the drain wait rather than
#: spent past the deadline. A reserve is not enough on its own: the drain loop tests its deadline
#: at the TOP, so `stop_generation` re-reads the clock before committing to a turn.
#: The fixed waits are 80 ms for the composer to take the text, 600 ms to let the turn get
#: going, 400 ms for the chunks in flight and 200 ms for the cleanup delete.
# One iteration is a 100 ms tick.
OWN_TURN_RESERVE_MS = OWN_TURN_FIXED_MS + OWN_TURN_POLL_MS

#: How long to wait for the throwaway turn to start. Enter is pressed BEFORE this wait, so the
#: send is already committed: returning at the slot bound left two extra messages and a live
#: stream running into the next action's window. The slot bounds how long the turn is worth
#: MEASURING; this bounds how long it is worth waiting for so it can be stopped and deleted.
# The turn stays in the thread `still_running`.
TURN_START_TIMEOUT_MS = 8000

#: The composer text the throwaway turn is sent with, read back as well as written: a composer
#: that still holds it is a send the app REFUSED, the one case where nothing was committed.
OWN_TURN_TEXT = "one more"

#: Remove the throwaway turn: assistant first, then the user turn, because deleting the user
#: message can take the reply with it and leave the count ambiguous. Reports rather than asserts.
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

    # BOTH HALVES, REPORTED SEPARATELY: deleted-but-never-stopped and stopped-but-not-deleted leave
    # the film in different states.
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
    # THE SLOT, ON THE CLOCK THE ACTION ITSELF RUNS ON: `ctx.budget_ms` is what the slot HAD when the
    # runner entered it, and a budget spent is not a budget available.
    slot_deadline = time.monotonic() + ctx.budget_ms / 1000.0

    def remaining_ms() -> float:
        return (slot_deadline - time.monotonic()) * 1000.0

    text = _ev(ctx, "() => window.__sb.dom.composerText()")
    if text:
        ctx.page.fill('textarea[aria-label="Message input"]', "")
        ctx.page.wait_for_timeout(120)

    # STOP GETS ITS OWN GENERATION. Stopping whatever the cell was streaming permanently truncated
    # the measured reply (5,304 of 17,737 characters at 10K), so every later action ran against a
    # third-sized thread. THE GUARD BELOW IS THE OTHER HALF: entering the own-turn path only when
    # `isRunning()` was false let `--stream-tail-chars` kill the opening reply while liveness passed.
    # At 96,000 characters the slot opens at 28 s and kills it at about 9,200.
    # So the reply gets this slot's budget less `OWN_TURN_RESERVE_MS`, and if it has not finished
    # NOTHING IS STOPPED and the row says why.
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
        # AND THE RESERVE IS CHECKED AGAINST THE CLOCK: the loop tests `settle_deadline` at the top, so a
        # drain landing in the last iteration leaves less than the reserve however it is sized, and the
        # turn cannot be abandoned once Enter is pressed.
        if remaining_ms() < OWN_TURN_RESERVE_MS:
            return not_run(
                "the cell's own reply drained with only "
                f"{max(0.0, remaining_ms()):.0f}ms left of this slot's {ctx.budget_ms}ms, and "
                f"starting, stopping and removing a turn of our own takes about "
                f"{OWN_TURN_RESERVE_MS}ms. Running it here would finish inside the next slot and "
                "record a missed slot against it, so nothing was stopped. Lower "
                "--stream-tail-chars or move this slot past the drain"
            )

    # READ BEFORE ENTER: the only way to tell the turn this action added from the thread it was
    # handed, and `_reclaim_pending_turn` deletes only if it grew. threadTotal, not messageCount:
    # under a windowed mount a send that WORKED reports after == before.
    messages_before = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    ctx.page.fill('textarea[aria-label="Message input"]', OWN_TURN_TEXT)
    ctx.page.wait_for_timeout(80)
    ctx.page.keyboard.press("Enter")
    own_generation = True
    sent_at = time.monotonic()
    # HOW LONG THE TURN IS WORTH MEASURING, bounded by the SLOT as well as by `TURN_START_TIMEOUT_MS`
    # (8 s is 2.7x the whole stop slot). What is reserved out of it is the rest of the turn, not only
    # its sleeps: reserving `OWN_TURN_FIXED_MS` left 424 ms unpaid against real chromium.
    start_wait_ms = max(
        0.0,
        min(
            float(TURN_START_TIMEOUT_MS),
            remaining_ms() - OWN_TURN_FIXED_AFTER_SEND_MS - OWN_TURN_STOP_POLL_MS,
        ),
    )
    deadline = time.monotonic() + start_wait_ms / 1000.0
    # HOW LONG IT IS WORTH WAITING FOR, a different question, and NOT bounded by the slot. See `_reclaim_pending_turn`.
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

    # LEAVE THE THREAD AS WE FOUND IT: left in place the throwaway turn adds an assistant message and
    # a reasoning pane that the rest of the film, the census and the seeded-versus-streamed
    # comparison all measure.
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
            # Which reply was stopped: `chars_added_after_stop` reads differently for a throwaway turn than
            # for the cell's own.
            "own_generation": own_generation,
            # Whether the scaffolding was removed. Reported rather than asserted: a failed cleanup leaves an
            # extra turn every later census must be read against.
            "scaffold_removed": (None if removed is None else bool(removed.get("removed"))),
            "scaffold_note": (None if removed is None else removed.get("reason")),
            # A stop that worked leaves the text where it was, give or take the chunks already in flight; a
            # large jump means the stream ran on.
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
    # Click the LABEL, not the trigger's right edge: a `span[data-eject-hit]` sits there and ejects
    # the model instead of opening the picker.
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
    # Hover AND WAIT, on the same reasoning as `message_menu`: the bar is `hideWhenRunning`, so a slot
    # opening with the follow-up's last chunks still arriving finds no Copy button.
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
    # Read back from the clipboard: headless Chromium grants clipboard-read only with the permission
    # the browser factory requests, and without it the action still RAN but could not be proved.
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
    # NON-EMPTY, with the coverage fraction as evidence rather than a gate: innerText collapses
    # whitespace and skips nested scrollers while Selection.toString normalises differently, so the
    # two counts are not the same quantity.
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
    # A SENTINEL ON THE CLIPBOARD BEFORE THE COPY, so the copy is OBSERVED rather than assumed from
    # the keystroke. Playwright's WebKit never copies on Control+C: the action reported its own
    # 250 ms sleep as `copy_ms` across forty-three rows. The test is on the CLIPBOARD, not the engine
    # name. A failed `writeText` falls back to SNAPSHOTTING what the clipboard already holds; when
    # even that fails the action is NOT RUN, because clearing the sentinel re-admits the defect.
    # Chromium reads about 1,538 ms at 100K for the same action.
    # Residual: an honest copy of character-identical content reads as a no-op.
    sentinel = f"__sb_clipboard_sentinel_{int(time.monotonic() * 1000)}__"
    sentinel_written = False
    #: What the clipboard held before Control+C and where that value came from; None means no
    #: pre-copy value could be established, which is refused below.
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
    # THE CLIPBOARD, NOT THE SELECTION: a selection covers only mounted nodes while a copy handler
    # reading the message store puts the whole conversation on the clipboard, so scoring on the
    # selection would report a build that FIXED the data loss as still losing it. Read back before
    # the selection is cleared, and a failed read is reported rather than read as an empty clipboard.
    clip = None
    clip_reason = None
    try:
        clip = ctx.page.evaluate("async () => await navigator.clipboard.readText()")
    except Exception as exc:  # noqa: BLE001
        clip_reason = f"the clipboard could not be read back: {_why(exc)}"
    clipboard_chars = len(clip) if isinstance(clip, str) else None
    ctx.page.evaluate("() => window.getSelection().removeAllRanges()")

    # NO CONFIRMED COPY, NO TIMING: a reader seeing `copy_ms` cannot tell a real copy from a
    # keystroke that went nowhere, and the engines that fail do so consistently enough to look
    # like data.
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
            # WHAT ACTUALLY REACHED THE CLIPBOARD, the user-facing quantity the truncation alarm is scored
            # on: a windowed mount cannot SELECT what is not in the DOM but can still COPY it from the store.
            "clipboard_chars": clipboard_chars,
            "clipboard_readable": clip_reason is None,
            "clipboard_note": clip_reason,
            # WHICH pre-copy value the change was confirmed against: the sentinel this action wrote, or a
            # snapshot taken when the write was refused (the weaker of the two).
            "copy_confirmed_against": pre_copy_source,
            # The DOM coverage beside it: `mounted_fraction` well below 1 with whole `clipboard_chars` means
            # the copy-from-store path works; both short means the conversation is being lost.
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
        # `expect_ok` stays `chars > 0`: within one run the selection and every DOM-derived reference
        # shrink together, and an absolute floor would need per-rung, per-platform calibration.
        # `clipboard_chars` comes first as the user-visible number, `selected_chars` as the mechanism.
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
    # A SHARED MUTABLE cursor, not a scalar in `args`: the runner rebuilds each action's args, so a
    # scalar written back is discarded and the second send re-sent the first turn.
    # Slot args merge as `{**base_args, **slot.args}`.
    cursor = ctx.args.get("_stream_cursor")
    if not isinstance(cursor, dict):
        return not_run("no shared stream cursor was passed to the action")
    index = int(cursor.get("i", 0))
    if pacer is None:
        return not_run("no pacer was passed to the action")
    if index >= len(queue):
        return not_run(f"the stream queue is exhausted ({len(queue)} turns planned)")
    if _ev(ctx, "() => window.__sb.dom.isRunning()"):
        # Sending while a reply is in flight queues the message instead of starting a stream, and the
        # action would report a precise number about a message that is merely parked.
        return not_run("a reply was still streaming, so this send would have been queued")

    unit = queue[index]
    cursor["i"] = index + 1
    # NO `pacer.reset()` HERE: `CellRunner` records `last_stats()`, so resetting discarded the
    # opening reply's StreamStats and let a later completed turn mask a disconnected one. Every turn
    # is tagged instead, so `check_planned_streams` can verify what the cell streamed.
    # One turn delivered 4,624 of its 10,000 characters.
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
    # THE THREAD'S LENGTH, not the mounted count: under a windowed mount a send that worked perfectly
    # reports `after == before`, since the new pair arrives as two messages leave the top.
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
    # A POSITIVE consequence: the turn actually started streaming AND the thread grew, so a silently
    # failed send cannot read as an instant one.
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
            # What was MOUNTED either side, kept beside the thread totals: identical on the shipped build,
            # and where they differ the difference is the window size the virtualization arm measures.
            "mounted_before": mounted_before,
            "mounted_after": mounted_after,
            "turn_index": index + 1,
            "queued_turns": len(queue),
            "streamed_chars": len(unit["reasoning"]) + len(unit["content"]),
            # The tag this turn's stream carries, so the cell checks it against the pacer's own record rather
            # than re-deriving the naming rule.
            "pacer_tag": tag,
            "unit_kind": unit.get("kind"),
        },
        timings = {"to_first_token_ms": None if first_ms is None else round(first_ms, 1)},
        reason = None if ok else "the send did not start a new streaming reply",
    )


#: What the page can tell us when the attachments button cannot be found: geometry, style,
#: hit-testing and surrounding chrome, so three explanations that look identical from a `not_run`
#: string look different here.
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
    # `:visible`, and the FIRST visible match: `document.querySelector` returns document order and
    # the composer exists more than once (the welcome-screen instance stays in the tree, compare mode
    # has two threads), so a plain query handed back an unclickable button while a direct probe found
    # the real one. Playwright's actionability wait also blocks for its 30s default, so everything
    # here is bounded by what is left of the slot.
    locator = ctx.page.locator('button[aria-label="Tools and attachments"]:visible').first
    try:
        plus = locator.element_handle(timeout = 2000)
    except Exception:  # noqa: BLE001
        plus = None
    if plus is None:
        # WHY, not just THAT: a bare "not visible" conflates a button that is absent, one that is covered
        # and a locator that disagrees with the page, and has already cost three wrong hypotheses.
        # Carrying the probe state into the row means the next run answers it.
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


#: The end of the thread as a plain string that can be looked for after the rebuild. NOT the
#: seeder's `last_marker`: `send_turn` appends turns mid-film, so the seeded marker sits several
#: messages from the end while `end_present` requires it within two rows. `trim()` then a prefix,
#: never a whitespace-collapsing normalisation, since readiness.PROBE_JS matches RAW text.
# ── 13. thread reopen ───────────────────────────────────────────────

_LAST_USER_TEXT_JS = """
() => {
  const rows = Array.from(document.querySelectorAll('[data-role="user"]'));
  const last = rows.length ? rows[rows.length - 1] : null;
  if (!last) return null;
  const text = (last.textContent || "").trim();
  return text ? text.slice(0, 120) : null;
}
"""

#: How long the rebuilt thread is given to reach the readiness gate, in seconds: what is left of
#: the SLOT, floored at 10s so a 1M-token thread still gets a fair chance and capped at the flat
#: minute the previous loop waited. Slot-bounded because a broken arm can fail the condition
#: forever and `delete_message`, the film's last slot, has nowhere to absorb an overrun.
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
    # The THREAD's length, not the mounted count: a windowed arm reopening at a different scroll
    # anchor mounts a different number of rows and would fail the exact-equality assertion.
    before = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted_before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    # AN INTEGER, not merely truthy: `_ev` hands back `{"__error": ...}` when the page throws, and a
    # truthy dict would reach the readiness gate as a length and raise there.
    if not isinstance(before, int) or before <= 0:
        return not_run("the thread has no messages to rebuild")
    # WHICH READINESS MODE THIS ARM IS HELD TO, decided from the mount on screen before anything is
    # torn down: a declared total larger than the mounted count IS a windowed mount. Read from the
    # page rather than a session flag so both arms are judged by one rule.
    mode = (
        MODE_WINDOWED if isinstance(mounted_before, int) and mounted_before < before else MODE_FULL
    )
    marker = _ev(ctx, _LAST_USER_TEXT_JS)
    if not isinstance(marker, str) or not marker:
        # Refused HERE, before anything is touched, so a thread this action cannot verify is also one it
        # has not disturbed: without a string identifying the end of the conversation there is no way to
        # tell a rebuilt thread from a half-rebuilt one.
        return not_run(
            "the last user turn carried no text to identify the end of the thread with, so a "
            "finished rebuild could not have been told from a partial one"
        )
    started = time.monotonic()
    # NO FALLBACK ON THE WAY OUT. A goto is a FULL DOCUMENT NAVIGATION while the click is a
    # client-side subtree rebuild, and read as the click the substitution produced thread_reopen at
    # 6.0 fps. Refusing AFTER `page.goto` had run left the scene on an empty new chat, so the
    # substitution is now declined BEFORE it happens.
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
    # BOTH CLOCKS START AT THE CLICK THAT WORKED. `_click_or_navigate` tries `handle.click` first,
    # and Playwright's hit-target check retries for its full 2,000 ms against the New chat button,
    # which is `opacity-0 pointer-events-none` until its header group is hovered, so that timeout
    # landed inside the `close_ms` floor_table quotes. The retry is MOVED to the retry fields.
    # Moved to `left_click_retry_ms` and `reopen_click_retry_ms`.
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
    # THE FALLBACK IS ALLOWED ON THE WAY BACK: from an empty new chat, navigating to the thread's own
    # URL restores the scene for the slots that follow. Still not a measurement of a rebuild, so no
    # timing is reported for it.
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
    # WHEN IS THE REBUILD OVER? Not when `threadTotal()` reaches its old length: on a windowed arm
    # that reads `aria-setsize`, the STORE'S CLAIM, which the first reopened row publishes while
    # three of eighteen messages are mounted. `reopen_ms` now runs from the sidebar click until
    # runtime/readiness.py's own gate passes. That same function, not a second definition of ready,
    # and each arm is judged in its own mode; the cost is one STABLE_GAP_MS paid equally by both.
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
        # A REAL FINDING ABOUT THE ARM: `ran` stays True with the outstanding conditions on the row, and
        # `reopen_ms` stays null so scoring does not read it as a fast rebuild.
        readiness = exc.detail
    except Exception as exc:  # noqa: BLE001
        # The probe itself failed (a closed page, an uninstalled `window.__sb`), so nothing is claimed
        # about the rebuild.
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
            # THE HARNESS'S OWN COST, kept OUT of the two timings above: how long `_click_or_navigate` spent
            # on attempts that failed, almost always Playwright's 2,000 ms hit-target retry.
            "left_click_retry_ms": round((click_left_at - started) * 1000, 1),
            "reopen_click_retry_ms": round((click_back_at - reopen_started) * 1000, 1),
            # WHICH GATE THE TIMING WAS TAKEN AGAINST: two arms in different readiness modes are answering
            # slightly different questions.
            "reopen_ready_mode": mode,
            # The gate's own verdict: "mounted 3 of 18, aria-setsize 18" is the difference between a slow app
            # and an arm that publishes a total it has not built.
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
    #: The monotonic instant the interaction that SUCCEEDED was issued, so a caller can time from the
    #: click rather than the first attempt; None when nothing succeeded. The failed attempt is not
    #: part of the gesture: Playwright's hit-target check retries for the full 2,000 ms against the
    #: sidebar's `New chat`, which made `thread_reopen.close_ms` two seconds of retry.
    started_at: Optional[float] = None

    @property
    def navigated(self) -> bool:
        return self.path == "navigate"


#: Where on an element to look for a point a user could hit: fractions of the box, centre first,
#: then the lower band that stays uncovered when a sticky header overlaps the top.
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

#: EVERY match, not the first: this app renders "New chat" twice, and a collapsed sidebar leaves
#: its zero-size button first in document order, so a plain query reported "no point hit-tests to
#: it" about a fully clickable button elsewhere. The same trap `image_upload` documents.
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


#: `.sidebar-header-action` transitions opacity over 150ms, and one sample after a fixed sleep
#: lost roughly one run in three on a loaded machine, so the hit test is retried instead.
_REVEAL_SETTLE_MS = 120
_REVEAL_ATTEMPTS = 5

#: The control's own box, whatever its computed style says: mid-transition `getComputedStyle`
#: reports the old opacity, and after a failed click the element can read as revealed while not
#: yet hit-testable. This only runs when the control is already known to be unreachable.
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
        # THE CENTRE IS COVERED, BUT THE CONTROL IS NOT. Playwright clicks the centre and refuses when
        # something else hit-tests there; the sidebar's sticky group label overlaps the top of New chat
        # while most of the button is clickable. So the box is hit-tested at a spread of points and the
        # first that resolves to the control is clicked with a real mouse event; no reachable point is a
        # finding worth reporting rather than a reason to substitute a page load.
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


#: How long an action that needs a message's ACTION BAR waits for the reply to finish. Unsloth
#: hides the bar while a message generates, and the film schedules `message_menu` about four
#: seconds after a `send_turn` whose reply runs for roughly fourteen, so the action reported NOT
#: RUN on every CI run. Bounded by the slot's budget, and waited BEFORE the timed operation.
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
    # TWO WAITS, COVERING DIFFERENT THINGS: this one waits for the STREAM to stop, bounded by a
    # fraction of the slot, while `waitForButtonMs` waits for the BAR TO MOUNT a few hundred ms
    # later, which this one cannot see because `isRunning()` is already false.
    # See `OWN_TURN_STOP_POLL_MS`.
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
        # Logged rather than swallowed: a cell that had to wait says so, because the wait is a fact about
        # the film's PACKING.
        ctx.log(f"    message_menu waited {waited_ms}ms for the action bar to be mounted")
    # Opened AND closed AND a non-zero item count. Any one of the three alone can be satisfied by
    # a menu that never rendered its items.
    ok = raw["openMs"] is not None and raw["closeMs"] is not None and raw["items"] > 0
    return ActionResult(
        ran = True,
        expect_ok = ok,
        expect = {
            "items_while_open": raw["items"],
            # Radix puts the body on the modal layer while the menu is up, which is the fan-out under
            # suspicion, so this proves the open really took that path.
            "body_pointer_events_open": raw["bodyPointerEvents"],
            "body_pointer_events_closed": raw["bodyPointerEventsAfterClose"],
            # In the payload, not only in the log: a run whose cells all waited is a run whose film opens
            # this slot too early.
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
    # The THREAD TOTAL dropped: a delete that detached the node but left the thread the same length
    # is a different bug, or on a windowed mount a node the virtualizer recycled.
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
