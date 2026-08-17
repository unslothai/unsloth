// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

/**
 * Keep the hovered-message state still while the thread is scrolling.
 *
 * THE PROBLEM, MEASURED
 *
 * assistant-ui's `MessagePrimitive.Root` puts a `mouseenter` / `mouseleave` pair on every message
 * and calls `message.setIsHovering()` from them. When a person wheel-scrolls with the cursor
 * resting on the conversation, which is how a long chat is read, the cursor does not move but the
 * content moves under it, so the browser re-hit-tests on every step and the hovered message
 * changes on every step.
 *
 * On chromium at 300K characters of thread (43,422 DOM nodes, 220 messages), a 20-step gesture
 * with the cursor on a message produces exactly 20 pointerover events against 20 distinct
 * targets, and costs 1380.9ms with 11 long tasks totalling 1000ms. The same gesture with the
 * cursor over the scroller gutter, where the element underneath never changes, costs 664.4ms with
 * no long tasks at all, which is the gesture's own 20-step floor. So each boundary crossing costs
 * about 90ms of main-thread work on a thread that size.
 *
 * WHY THE COST IS NOT LOCAL, AND WHY THE OBVIOUS FIX MAKES IT WORSE
 *
 * The write is per-message but the notification is not: its cost scales with how many subscribed
 * components are mounted in the whole thread, not with the one message whose hover changed.
 * Measured as a natural experiment. Revealing the action bar with a CSS `group-hover` rule instead
 * of assistant-ui's `autohide` keeps all 110 user action bars mounted rather than mounting and
 * unmounting one at a time, which adds roughly 440 subscribed buttons and changes nothing else.
 * Same 11 boundary events, and the cost per event went from 91ms to 293ms:
 *
 *     control (autohide, bars unmounted)          1380.9ms, 11 long tasks, 1000ms
 *     CSS group-hover, all bars mounted           3635.1ms, 11 long tasks, 3223ms
 *     the same without the two :has() rules       3952.2ms, 11 long tasks, 3544ms
 *
 * That is the defect, and it is upstream: one boundary event commits across the list rather than
 * at the one message whose hover changed. Studio is pinned to @assistant-ui/react 0.12.28 and
 * cannot fix the notification scoping from here.
 *
 * WHAT THIS DOES INSTEAD
 *
 * It stops the churn rather than making it cheaper. While the viewport is scrolling, the
 * `mouseenter` and `mouseleave` that a message would otherwise receive are swallowed in the
 * CAPTURE phase, before they reach the listener assistant-ui installed on the message. When
 * scrolling stops, the message actually under the cursor is resolved once and a single synthetic
 * pair is delivered, so the state ends up exactly where it would have without this.
 *
 * Capture-phase interception is what makes this possible without patching upstream: `mouseenter`
 * and `mouseleave` do not bubble, but they do have a capture phase, so a listener on `document`
 * sees them first and `stopPropagation` there prevents the target's own listener from running.
 *
 * The filter is deliberately narrow. Only events whose target is a `[data-message-id]` element --
 * which is exactly what `MessagePrimitive.Root` renders and attaches to -- are swallowed, so
 * nothing else in the app loses hover events. CSS `:hover` is an engine-level state rather than
 * an event, so it is not affected at all and the buttons inside the bar keep their hover styling.
 *
 * THIS IS A CHROMIUM PROBLEM, AND THAT MATTERS FOR WHICH SURFACES IT FIXES
 *
 * Whether the engine re-hit-tests hover while the content moves under a stationary cursor is an
 * engine decision, and the three do not agree. Counting boundary events over the same 20-step
 * gesture, control against this hook, medians of 3:
 *
 *     engine     pointerover per gesture   before                     after
 *     chromium   20, 20, 20                1562.2ms, 11 frames >33ms  666.8ms, 0
 *     firefox    3, 1, 1                    850.0ms,  1 frame  >33ms  667.0ms, 0
 *     webkit     2, 0, 0                    644.0ms,  0 frames        643.0ms, 0
 *
 * Chromium re-hit-tests on every scroll step. WebKit essentially does not, so there is no
 * regression there to fix and the two columns are the same number. Firefox is in between: few
 * events, but the one it does fire is expensive, worst frame 201ms against 18ms after.
 *
 * So this is load-bearing for Chromium surfaces, which includes Unsloth Desktop on WINDOWS via
 * WebView2, and it is a no-op rather than a fix for Desktop on macOS and Linux. Note that
 * Playwright's WebKit is a proxy for WKWebView and WebKitGTK rather than the webview Desktop
 * actually embeds, so "no regression on the proxy" is good evidence and not a guarantee.
 *
 * The counter above watches `pointerover`, which is the engine's hit-test churn, while this hook
 * swallows `mouseenter` / `mouseleave`, which is what assistant-ui listens to. That is why the
 * chromium count stays at 20 in both columns: the hook does not stop the engine hit-testing, it
 * stops the result reaching React. A count that dropped would mean this was doing something other
 * than what it claims.
 *
 * THE BEHAVIOUR CHANGE, STATED
 *
 * During an active scroll the action bar does not follow the cursor from message to message; it
 * settles onto the correct message once scrolling stops. A bar that flickers between messages
 * during a fast scroll is not something anyone aims for, but it IS a change, and it is the only
 * one: at rest, on hover, and after the pointer leaves, the behaviour is identical.
 */

/** How long after the last scroll event the thread counts as still. */
const QUIET_MS = 150;

/**
 * How recently a real pointer move counts as "the user is moving the cursor".
 *
 * The regression this hook exists for is content moving under a STATIONARY cursor. A cursor the
 * user is actively moving produces boundary events they asked for, and swallowing those is what
 * makes the action bar unreachable for the whole of a streaming reply, whose auto-scroll keeps
 * `scrolling` true continuously.
 */
const MOVING_MS = 120;

/**
 * How long a wheel, touch drag or scrolling key counts as "the user is scrolling this".
 *
 * Momentum keeps the scroll events coming after the last wheel tick, so this outlasts the input
 * rather than tracking it exactly. It only needs to be shorter than the gap between a gesture
 * ending and a stream's auto-scroll being mistaken for one.
 */
const USER_SCROLL_MS = 600;

/** What `MessagePrimitive.Root` renders, and the only thing this will interfere with. */
const MESSAGE_SELECTOR = "[data-message-id]";

/**
 * Takes the viewport ELEMENT rather than a getter, because Thread already keeps it in state for
 * exactly this reason: the keyed provider remounts the viewport on a thread switch, and an effect
 * that resolved the node itself would keep listening to the old one.
 */
export function useHoverQuietDuringScroll(viewport: HTMLElement | null): void {
  useEffect(() => {
    if (!viewport) return;
    const doc = viewport.ownerDocument;
    if (!doc) return;

    let scrolling = false;
    let quietTimer: number | undefined;
    // The message assistant-ui currently believes is hovered, which is the last one this let a
    // `mouseenter` through to. Tracked here rather than read back, because assistant-ui exposes
    // no way to ask.
    let active: HTMLElement | null = null;
    let lastMouseMoveAt = Number.NEGATIVE_INFINITY;
    let lastUserScrollInputAt = Number.NEGATIVE_INFINITY;

    /**
     * The message the ENGINE says the cursor is over.
     *
     * Deliberately not a remembered pointer position. A position is only ever updated by a
     * pointermove ON the viewport, so nothing invalidates it when the cursor leaves for the
     * composer or the sidebar, and a later scroll -- a streaming reply auto-scrolls the thread --
     * would resolve whatever message had drifted under that stale point and reveal its action bar
     * with the cursor nowhere near it. `:hover` is engine state that this hook does not disturb,
     * it is empty when the cursor is outside the thread, and it is empty for touch, which
     * produces no hover at all.
     */
    const hoveredMessage = (): HTMLElement | null =>
      viewport.querySelector<HTMLElement>(`${MESSAGE_SELECTOR}:hover`);

    const send = (el: HTMLElement, type: "mouseenter" | "mouseleave"): void => {
      // Non-bubbling, because that is what the real event is: assistant-ui's listener sits on
      // this element, so a bubbling stand-in would also reach ancestors that never get one today.
      // No coordinates: the only listener this can reach is assistant-ui's, which reads none, and
      // carrying a position here is what made the stale-pointer bug possible.
      el.dispatchEvent(
        new MouseEvent(type, { bubbles: false, cancelable: false }),
      );
    };

    const settle = (): void => {
      scrolling = false;
      const next = hoveredMessage();
      if (next === active) return;
      if (active?.isConnected) send(active, "mouseleave");
      if (next) send(next, "mouseenter");
      active = next;
    };

    const onScroll = (event: Event): void => {
      // Only a scroll the USER drove suppresses hover.
      //
      // The regression this hook exists for is a wheel gesture with the cursor resting on the
      // conversation. A thread that scrolls ITSELF is a different thing, and the difference is
      // not academic: while a reply streams, the viewport auto-scrolls on every token, so
      // `scrolling` never goes quiet and every boundary event is swallowed for the whole
      // response. Measured on chromium, 3 of 3 repetitions, sampling the visible action bar 14
      // times across a stream with the cursor resting on a message: merge base 1 bar throughout,
      // without this guard 0 bars throughout. The bar came back only once the stream ended.
      // Losing the action bar for the length of a model response is not the behaviour change
      // that was signed off, which was that it settles rather than follows DURING a gesture.
      if (event.timeStamp - lastUserScrollInputAt > USER_SCROLL_MS) return;
      scrolling = true;
      if (quietTimer !== undefined) window.clearTimeout(quietTimer);
      quietTimer = window.setTimeout(settle, QUIET_MS);
    };

    /**
     * The inputs by which a person scrolls this viewport themselves. Tracked rather than inferred
     * from the scroll event, because a scroll event is identical whoever caused it.
     */
    const onUserScrollInput = (event: Event): void => {
      lastUserScrollInputAt = event.timeStamp;
    };

    const onPointerMove = (event: PointerEvent): void => {
      // Mouse only. A touch drag emits pointermove before the browser claims the gesture, and
      // touch scrolling produces no hover, so treating a finger as a cursor is how a bar appears
      // on a device that has none.
      if (event.pointerType !== "mouse") return;
      lastMouseMoveAt = event.timeStamp;
    };

    const onBoundary = (event: Event): void => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (!target.matches(MESSAGE_SELECTOR)) return;
      if (!viewport.contains(target)) return;
      if (!scrolling || event.timeStamp - lastMouseMoveAt < MOVING_MS) {
        // Not scrolling, or the user is moving the cursor rather than the content moving under
        // it: let it through, and keep our idea of the hovered message in step with the one
        // assistant-ui is about to form.
        if (event.type === "mouseenter") active = target;
        else if (active === target) active = null;
        return;
      }
      // `stopImmediatePropagation` as well as `stopPropagation`: another capture listener on the
      // same node would otherwise still run, and the point is that nothing downstream sees this.
      event.stopPropagation();
      event.stopImmediatePropagation();
    };

    viewport.addEventListener("scroll", onScroll, { passive: true });
    viewport.addEventListener("pointermove", onPointerMove, { passive: true });
    viewport.addEventListener("wheel", onUserScrollInput, { passive: true });
    viewport.addEventListener("touchmove", onUserScrollInput, {
      passive: true,
    });
    viewport.addEventListener("keydown", onUserScrollInput);
    doc.addEventListener("mouseenter", onBoundary, true);
    doc.addEventListener("mouseleave", onBoundary, true);
    return () => {
      if (quietTimer !== undefined) window.clearTimeout(quietTimer);
      viewport.removeEventListener("scroll", onScroll);
      viewport.removeEventListener("pointermove", onPointerMove);
      viewport.removeEventListener("wheel", onUserScrollInput);
      viewport.removeEventListener("touchmove", onUserScrollInput);
      viewport.removeEventListener("keydown", onUserScrollInput);
      doc.removeEventListener("mouseenter", onBoundary, true);
      doc.removeEventListener("mouseleave", onBoundary, true);
    };
  }, [viewport]);
}
