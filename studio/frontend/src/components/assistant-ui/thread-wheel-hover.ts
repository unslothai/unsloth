// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** suppresses wheel-scroll hover churn while preserving Chromium hit-testing and drags (#12025). */

// Chromium smooth-scroll animation can outlast the final wheel event by about 200ms.
const WHEEL_TAIL_MS = 300;
const SETTLE_MS = 150;
const MESSAGE = "[data-message-id]";

export function attachWheelHoverSuppression(viewport: HTMLElement): () => void {
  let wheelAt = Number.NEGATIVE_INFINITY;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let hovered: Element | null = null;
  const settle = () => {
    timer = undefined;
    const now = viewport.querySelector(`${MESSAGE}:hover`);
    if (now === hovered) return;
    hovered?.dispatchEvent(new MouseEvent("mouseleave"));
    now?.dispatchEvent(new MouseEvent("mouseenter"));
  };
  const onBoundary = (event: MouseEvent) => {
    const target = event.target as Element | null;
    if (!target?.matches?.(MESSAGE)) return;
    if (timer !== undefined && event.buttons === 0) {
      event.stopPropagation();
    } else if (event.type === "mouseenter") {
      hovered = target;
    } else if (target === hovered) {
      hovered = null;
    }
  };
  const onWheel = (event: WheelEvent) => {
    wheelAt = event.buttons ? Number.NEGATIVE_INFINITY : event.timeStamp;
    // mounting under a still pointer sets `:hover` without mouseenter; wheel fires before scroll.
    if (timer === undefined) {
      hovered = viewport.querySelector(`${MESSAGE}:hover`);
    }
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
