// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useCallback, useEffect, useRef } from "react";

import { lockCollapseScroll } from "./collapse-scroll-lock";

function findScrollContainer(element: HTMLElement): HTMLElement | null {
  let node: HTMLElement | null = element;
  while (node) {
    const { overflowY } = getComputedStyle(node);
    if (overflowY === "scroll" || overflowY === "auto") {
      return node;
    }
    node = node.parentElement;
  }
  return null;
}

/**
 * Locks the nearest scrollable ancestor's scrollTop during a collapsible
 * animation so the page doesn't jump when content height changes.
 *
 * Unlike @assistant-ui/react's `useScrollLock`, this does NOT toggle
 * `scrollbar-width: none` on the container; hiding the scrollbar mid-animation
 * caused a visible flicker on tool-call collapsibles.
 */
export function useCollapseScrollLock(
  animatedElementRef: RefObject<HTMLElement | null>,
  animationDurationMs: number,
): () => void {
  const releaseRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      releaseRef.current?.();
      releaseRef.current = null;
    };
  }, []);

  return useCallback(() => {
    releaseRef.current?.();
    releaseRef.current = null;

    const animatedElement = animatedElementRef.current;
    if (!animatedElement) return;
    const container = findScrollContainer(animatedElement);
    if (!container) return;

    releaseRef.current = lockCollapseScroll(
      container,
      animationDurationMs,
      container.ownerDocument,
    );
  }, [animatedElementRef, animationDurationMs]);
}
