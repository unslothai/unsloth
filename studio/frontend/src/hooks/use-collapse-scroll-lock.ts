// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useCallback, useEffect, useRef } from "react";

/**
 * Locks the nearest scrollable ancestor's scrollTop for the duration of a
 * collapsible animation so the page doesn't jump when content height changes.
 *
 * Unlike @assistant-ui/react's `useScrollLock`, this hook does NOT toggle
 * `scrollbar-width: none` on the container. Hiding the scrollbar mid-animation
 * caused a visible disappear/reappear flicker on tool-call collapsibles.
 */
export function useCollapseScrollLock(
  animatedElementRef: RefObject<HTMLElement | null>,
  animationDurationMs: number,
): () => void {
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      cleanupRef.current?.();
    };
  }, []);

  return useCallback(() => {
    cleanupRef.current?.();

    const animatedElement = animatedElementRef.current;
    if (!animatedElement) return;

    let scrollContainer: HTMLElement | null = animatedElement;
    while (scrollContainer) {
      const { overflowY } = getComputedStyle(scrollContainer);
      if (overflowY === "scroll" || overflowY === "auto") {
        break;
      }
      scrollContainer = scrollContainer.parentElement;
    }
    if (!scrollContainer) return;

    const container = scrollContainer;
    const scrollPosition = container.scrollTop;
    const resetPosition = () => {
      container.scrollTop = scrollPosition;
    };

    container.addEventListener("scroll", resetPosition);
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const cleanup = () => {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      container.removeEventListener("scroll", resetPosition);
      if (cleanupRef.current === cleanup) {
        cleanupRef.current = null;
      }
    };
    timeoutId = setTimeout(cleanup, animationDurationMs);
    cleanupRef.current = cleanup;
  }, [animatedElementRef, animationDurationMs]);
}
