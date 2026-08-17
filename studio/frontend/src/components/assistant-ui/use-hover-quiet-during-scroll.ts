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
 * THE BEHAVIOUR CHANGE, STATED
 *
 * During an active scroll the action bar does not follow the cursor from message to message; it
 * settles onto the correct message once scrolling stops. A bar that flickers between messages
 * during a fast scroll is not something anyone aims for, but it IS a change, and it is the only
 * one: at rest, on hover, and after the pointer leaves, the behaviour is identical.
 */

/** How long after the last scroll event the thread counts as still. */
const QUIET_MS = 150;

/** What `MessagePrimitive.Root` renders, and the only thing this will interfere with. */
const MESSAGE_SELECTOR = "[data-message-id]";

/**
 * Takes the viewport ELEMENT rather than a getter, because Thread already keeps it in state for
 * exactly this reason: the keyed provider remounts the viewport on a thread switch, and an effect
 * that resolved the node itself would keep listening to the old one.
 */
export function useHoverQuietDuringScroll(
  viewport: HTMLElement | null,
): void {
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
    let pointerX = -1;
    let pointerY = -1;
    let pointerSeen = false;

    const messageAt = (x: number, y: number): HTMLElement | null => {
      if (x < 0 || y < 0) return null;
      const el = doc.elementFromPoint(x, y);
      if (!(el instanceof Element)) return null;
      const message = el.closest(MESSAGE_SELECTOR);
      return message instanceof HTMLElement && viewport.contains(message)
        ? message
        : null;
    };

    const send = (el: HTMLElement, type: "mouseenter" | "mouseleave"): void => {
      // Non-bubbling, because that is what the real event is: assistant-ui's listener sits on
      // this element, so a bubbling stand-in would also reach ancestors that never get one today.
      el.dispatchEvent(
        new MouseEvent(type, {
          bubbles: false,
          cancelable: false,
          clientX: pointerX,
          clientY: pointerY,
        }),
      );
    };

    const settle = (): void => {
      scrolling = false;
      // Without a pointer position there is nothing to resolve. Leaving `active` alone is right:
      // it is whatever the last real mouseenter set, and no scroll has invalidated it.
      if (!pointerSeen) return;
      const next = messageAt(pointerX, pointerY);
      if (next === active) return;
      if (active?.isConnected) send(active, "mouseleave");
      if (next) send(next, "mouseenter");
      active = next;
    };

    const onScroll = (): void => {
      scrolling = true;
      if (quietTimer !== undefined) window.clearTimeout(quietTimer);
      quietTimer = window.setTimeout(settle, QUIET_MS);
    };

    const onPointerMove = (event: PointerEvent): void => {
      pointerX = event.clientX;
      pointerY = event.clientY;
      pointerSeen = true;
    };

    const onBoundary = (event: Event): void => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (!target.matches(MESSAGE_SELECTOR)) return;
      if (!viewport.contains(target)) return;
      if (!scrolling) {
        // Not scrolling: let it through, and keep our idea of the hovered message in step with
        // the one assistant-ui is about to form.
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
    doc.addEventListener("mouseenter", onBoundary, true);
    doc.addEventListener("mouseleave", onBoundary, true);
    return () => {
      if (quietTimer !== undefined) window.clearTimeout(quietTimer);
      viewport.removeEventListener("scroll", onScroll);
      viewport.removeEventListener("pointermove", onPointerMove);
      doc.removeEventListener("mouseenter", onBoundary, true);
      doc.removeEventListener("mouseleave", onBoundary, true);
    };
  }, [viewport]);
}
