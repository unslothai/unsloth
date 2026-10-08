// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Wheeling a long thread slides messages under a still pointer, and every message boundary that
 * passes fires mouseenter/mouseleave on assistant-ui's MessagePrimitive.Root, which writes
 * `isHovering` to the store and re-runs every selector in the thread (#12025). While a wheel
 * scroll is moving the viewport, those events are stopped in the capture phase; once it settles,
 * one leave/enter pair moves the hover to the message now under the pointer.
 *
 * Not `pointer-events: none`: it pushes Chromium's scroll hit-testing onto the main thread, which
 * made heavy threads far slower. Only wheel-driven scrolls, so follow-to-bottom while streaming
 * keeps action bars responsive; not with a button held, so selection drags keep their events.
 */

// Chromium keeps animating a smooth wheel scroll for about 200ms after the last wheel event.
const WHEEL_TAIL_MS = 300;
const SETTLE_MS = 150;
const MESSAGE = "[data-message-id]";

export function attachWheelHoverSuppression(viewport: HTMLElement): () => void {
  let wheelAt = Number.NEGATIVE_INFINITY;
  let timer: ReturnType<typeof setTimeout> | undefined;
  // The message root whose mouseenter assistant-ui last received.
  let hovered: Element | null = null;
  const settle = () => {
    timer = undefined;
    const now = viewport.querySelector(`${MESSAGE}:hover`);
    if (now === hovered) return;
    hovered?.dispatchEvent(new MouseEvent("mouseleave"));
    now?.dispatchEvent(new MouseEvent("mouseenter"));
  };
  const onBoundary = (event: Event) => {
    const target = event.target as Element | null;
    if (!target?.matches?.(MESSAGE)) return;
    if (timer !== undefined) {
      event.stopPropagation();
    } else if (event.type === "mouseenter") {
      hovered = target;
    } else if (target === hovered) {
      hovered = null;
    }
  };
  const onWheel = (event: WheelEvent) => {
    wheelAt = event.buttons ? Number.NEGATIVE_INFINITY : event.timeStamp;
  };
  const onScroll = (event: Event) => {
    if (event.timeStamp - wheelAt > WHEEL_TAIL_MS) return;
    clearTimeout(timer);
    timer = setTimeout(settle, SETTLE_MS);
  };
  viewport.addEventListener("mouseenter", onBoundary, true);
  viewport.addEventListener("mouseleave", onBoundary, true);
  viewport.addEventListener("wheel", onWheel, { passive: true });
  viewport.addEventListener("scroll", onScroll, { passive: true });
  return () => {
    viewport.removeEventListener("mouseenter", onBoundary, true);
    viewport.removeEventListener("mouseleave", onBoundary, true);
    viewport.removeEventListener("wheel", onWheel);
    viewport.removeEventListener("scroll", onScroll);
    if (timer !== undefined) {
      clearTimeout(timer);
      settle();
    }
  };
}
