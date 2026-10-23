// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const USER_SCROLL_EVENTS = ["wheel", "touchstart"] as const;
export const USER_DISCRETE_EVENTS = ["pointerdown", "keydown"] as const;

export type GestureTarget = {
  addEventListener(
    type: string,
    listener: () => void,
    options?: AddEventListenerOptions,
  ): void;
  removeEventListener(
    type: string,
    listener: () => void,
    options?: EventListenerOptions,
  ): void;
};

export type CollapseScrollContainer = GestureTarget & {
  readonly scrollTop: number;
  scrollTo(options: { top: number; behavior: "instant" }): void;
};

export function lockCollapseScroll(
  container: CollapseScrollContainer,
  durationMs: number,
  discrete: GestureTarget = container,
): () => void {
  const scrollPosition = container.scrollTop;
  let released = false;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  const resetPosition = () => {
    if (container.scrollTop === scrollPosition) return;
    container.scrollTo({ top: scrollPosition, behavior: "instant" });
  };
  const release = () => {
    if (released) return;
    released = true;
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    container.removeEventListener("scroll", resetPosition);
    for (const type of USER_SCROLL_EVENTS) {
      container.removeEventListener(type, release, { capture: true });
    }
    for (const type of USER_DISCRETE_EVENTS) {
      discrete.removeEventListener(type, release, { capture: true });
    }
  };
  container.addEventListener("scroll", resetPosition);
  for (const type of USER_SCROLL_EVENTS) {
    container.addEventListener(type, release, { capture: true, passive: true });
  }
  for (const type of USER_DISCRETE_EVENTS) {
    discrete.addEventListener(type, release, { capture: true, passive: true });
  }
  timeoutId = setTimeout(release, durationMs);
  return release;
}
