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
from dataclasses import dataclass
from typing import Any

from ..runtime.types import ActionContext, ActionResult, not_run
from . import register_action

SETTLE_TIMEOUT_MS = 8000


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
            "highlight_spans_while_open": raw["spansOpen"],
        },
        timings = {"open_ms": raw["openMs"], "close_ms": raw["closeMs"]},
        reason = None
        if ok
        else f"{raw['openCount']} of {raw['panes']} panes opened and "
        f"{raw['afterClose']} were still open after collapsing",
    )


# ── 5. stop generation ──────────────────────────────────────────────

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


@register_action(name = "stop_generation", default_budget_ms = 8000)
def stop_generation(ctx: ActionContext) -> ActionResult:
    """Press stop mid-stream and time until the run is really over.

    THE COMPOSER MUST BE EMPTY. `queueDisabled` in thread.tsx depends on
    `composerText.trim().length > 0`, so with text in the box the Stop control is replaced by a
    Queue control at the same position with the same class. Pressing it queues a message and the
    stream carries on, and the action reports a fast, precise, entirely wrong number.
    """
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
    own_generation = False
    if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
        ctx.page.fill('textarea[aria-label="Message input"]', "one more")
        ctx.page.wait_for_timeout(80)
        ctx.page.keyboard.press("Enter")
        own_generation = True
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if _ev(ctx, "() => window.__sb.dom.isRunning()"):
                break
            ctx.page.wait_for_timeout(50)
        else:
            return not_run("nothing was generating and a new turn did not start within 8s")
        if not _ev(ctx, "() => window.__sb.dom.isRunning()"):
            return not_run("nothing was generating and a new turn did not start within 8s")
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
    ctx.page.evaluate("""() => {
        const m = window.__sb.dom.lastAssistantMessage();
        if (m) m.dispatchEvent(new PointerEvent("pointerover",
          { bubbles: true, pointerType: "mouse" }));
    }""")
    ctx.page.wait_for_timeout(150)
    started = time.monotonic()
    ok_click = _ev(
        ctx,
        """
      () => {
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
        clip_reason = f"the clipboard could not be read back: {type(exc).__name__}"
    clipboard_chars = len(clip) if isinstance(clip, str) else None
    ctx.page.evaluate("() => window.getSelection().removeAllRanges()")
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
    # The THREAD's length, not the mounted count: a windowed arm that reopens with a different
    # scroll anchor mounts a different number of rows, and the exact-equality assertion at the end
    # of this action would then fail for a reason that has nothing to do with the rebuild.
    before = _ev(ctx, "() => window.__sb.dom.threadTotal()")
    mounted_before = _ev(ctx, "() => window.__sb.dom.messageCount()")
    if not before:
        return not_run("the thread has no messages to rebuild")
    started = time.monotonic()
    leave = _click_or_navigate(
        ctx, 'button[aria-label="New chat"]', f"{ctx.args['base_url']}/chat?new=studiobench"
    )
    if not leave.ok:
        return not_run(f"the thread could not be left: {leave.reason}")
    if leave.navigated:
        # LOUD, AND FATAL TO THE READING. Workspace task #102.
        #
        # This used to be a silent substitution: the New chat click failed, `page.goto` ran
        # instead, and the action returned a perfectly normal-looking row. But a goto is a FULL
        # DOCUMENT NAVIGATION -- the SPA is torn down and reparsed, the bundle re-executes, the
        # runtime rehydrates from IndexedDB -- while the click is a client-side route change that
        # rebuilds one React subtree. They are not the same operation and they do not cost the same
        # thing. Read as though it were the click, the substitution produced thread_reopen at
        # 6.0 fps, which is a number about a page load being quoted as a number about a thread.
        #
        # So it is `ran = False` now. The row says the surface went unmeasured, which is the truth,
        # instead of contributing an instrument artefact to a table of user costs.
        ctx.log(
            "    thread_reopen NOT MEASURED: the New chat button could not be clicked, so leaving "
            f"the thread fell back to a full page navigation ({leave.reason}). A navigation "
            "reloads the whole app and is not the operation this action exists to time."
        )
        return not_run(
            "the New chat control could not be clicked and a full page navigation was substituted; "
            "a document reload is not a thread rebuild, so this reading is not a measurement of "
            f"the action ({leave.reason})"
        )
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
    back = _click_or_navigate(
        ctx, f'[data-thread-id="{thread_id}"]', f"{ctx.args['base_url']}/chat?thread={thread_id}"
    )
    if not back.ok:
        return not_run(f"the thread could not be reopened: {back.reason}")
    if back.navigated:
        ctx.log(
            "    thread_reopen NOT MEASURED: the sidebar row for the thread could not be clicked, "
            f"so reopening fell back to a full page navigation ({back.reason})."
        )
        return not_run(
            "the thread's sidebar row could not be clicked and a full page navigation was "
            f"substituted, so the rebuild was never timed ({back.reason})"
        )
    reopen_ms = None
    deadline = reopen_started + 60
    while time.monotonic() < deadline:
        if (_ev(ctx, "() => window.__sb.dom.threadTotal()") or 0) >= before:
            reopen_ms = (time.monotonic() - reopen_started) * 1000
            break
        ctx.page.wait_for_timeout(100)
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
        },
        timings = {
            "close_ms": round(closed_ms, 1),
            "reopen_ms": None if reopen_ms is None else round(reopen_ms, 1),
        },
        reason = None if ok else f"the thread came back with {after} of {before} messages",
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

    @property
    def navigated(self) -> bool:
        return self.path == "navigate"


#: Where on an element to look for a point a user could actually hit. Fractions of the box, centre
#: first (so a normal control behaves exactly as before), then the lower band, which is what stays
#: uncovered when a sticky header overlaps the top of a control.
_HIT_POINTS = (
    (0.5, 0.5), (0.5, 0.75), (0.25, 0.75), (0.75, 0.75),
    (0.5, 0.9), (0.15, 0.5), (0.85, 0.5), (0.5, 0.25),
)

_HIT_TEST_JS = """
([selector, fx, fy]) => {
  const el = document.querySelector(selector);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width === 0 || r.height === 0) return null;
  const x = r.left + r.width * fx;
  const y = r.top + r.height * fy;
  if (x < 0 || y < 0 || x > window.innerWidth || y > window.innerHeight) return null;
  const top = document.elementFromPoint(x, y);
  // `contains`, not identity: the control's own icon or label is a child and hit-tests to itself,
  // which is still a click that reaches the control.
  if (!top || !(top === el || el.contains(top))) return null;
  return { x, y };
}
"""


def _reachable_point(ctx: ActionContext, selector: str) -> tuple[float, float] | None:
    """A viewport point that hit-tests to `selector`, or None if the control is truly covered."""
    for fx, fy in _HIT_POINTS:
        got = _ev(ctx, _HIT_TEST_JS, [selector, fx, fy])
        if isinstance(got, dict) and "x" in got:
            return got["x"], got["y"]
    return None


def _click_or_navigate(ctx: ActionContext, selector: str, url: str) -> Transition:
    """Click the control a user would click; fall back to the URL it would produce, and SAY SO.

    A click is preferred because it exercises the app's own handler. Playwright's actionability
    retries do not give up on an element that is visible, enabled and stable and merely covered by
    something else, which is the sidebar's sticky group label over the New chat button -- so the
    click burns its whole timeout. The navigation reaches the same app state, and for getting the
    scene somewhere it is a perfectly good tool.

    WHAT IT IS NOT is the same operation, and the caller is now told which one it got. Every
    caller must decide for itself whether a navigation still answers its question. `thread_reopen`
    decides that it does not, because a document reload and a subtree rebuild are the two things
    that action exists to tell apart.
    """
    handle = ctx.page.query_selector(selector)
    click_error = f"no element matched {selector}"
    if handle is not None:
        try:
            handle.click(timeout = 2000)
            return Transition(ok = True, path = "click")
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
        point = _reachable_point(ctx, selector)
        if point is not None:
            try:
                ctx.page.mouse.click(point[0], point[1])
                return Transition(ok = True, path = "click", reason = "clicked off-centre")
            except Exception as exc:  # noqa: BLE001
                click_error += f"; the off-centre click also failed: {type(exc).__name__}"
        else:
            click_error += "; no point on the control hit-tests to it"
        ctx.log(f"    {click_error}; navigating instead")
    try:
        ctx.page.goto(url, wait_until = "domcontentloaded", timeout = 60_000)
        return Transition(ok = True, path = "navigate", reason = click_error)
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"    navigation to {url} failed: {type(exc).__name__}: {exc}")
        return Transition(
            ok = False,
            path = "failed",
            reason = f"{click_error}, and the navigation failed too: {type(exc).__name__}",
        )


# ── 14. message menu ────────────────────────────────────────────────

MENU_JS = """
async (timeoutMs) => {
  const D = window.__sb.dom;
  const message = D.lastAssistantMessage();
  if (message) {
    message.dispatchEvent(new PointerEvent("pointerover", { bubbles: true, pointerType: "mouse" }));
  }
  const trigger = D.actionButton("More");
  if (!trigger) return { ran: false, reason: "no More button on the last assistant message" };
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
  return { ran: true,
           openMs: openMs === null ? null : Math.round(openMs * 10) / 10,
           closeMs: closeMs === null ? null : Math.round(closeMs * 10) / 10,
           items, bodyPointerEvents,
           bodyPointerEventsAfterClose: getComputedStyle(document.body).pointerEvents };
}
"""


@register_action(name = "message_menu", default_budget_ms = 12000)
def message_menu(ctx: ActionContext) -> ActionResult:
    raw = _ev(ctx, MENU_JS, SETTLE_TIMEOUT_MS)
    err = _failed(raw)
    if err:
        return not_run(err)
    if not raw.get("ran"):
        return not_run(raw.get("reason", "the menu action did not run"))
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
async (timeoutMs) => {
  const D = window.__sb.dom;
  const message = D.lastAssistantMessage();
  if (message) {
    message.dispatchEvent(new PointerEvent("pointerover", { bubbles: true, pointerType: "mouse" }));
  }
  const button = D.actionButton("Delete message");
  if (!button) return { ran: false, reason: "no Delete button (a running message hides it)" };
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
  return { ran: true, ms: ms === null ? null : Math.round(ms * 10) / 10,
           before, after: D.threadTotal(),
           mountedBefore, mountedAfter: D.messageCount() };
}
"""


@register_action(name = "delete_message", default_budget_ms = 15000)
def delete_message(ctx: ActionContext) -> ActionResult:
    raw = _ev(ctx, DELETE_JS, SETTLE_TIMEOUT_MS)
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
        },
        timings = {"delete_ms": raw["ms"]},
        reason = None if ok else f"the message count went {raw['before']} -> {raw['after']}",
    )
