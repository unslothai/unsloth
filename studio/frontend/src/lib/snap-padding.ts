// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";
import { useCallback } from "react";

/**
 * Rounds a surface's left and right margin and padding to whole CSS pixels.
 *
 * A menu's padding scales with the UI (8px is 7.47px at font size 14), and so does the small
 * margin that aligns it to its trigger. Chromium draws a row's rounded hover pill on whole CSS
 * pixels while the menu around it keeps its fraction, so a fractional padding leaves the pill a
 * device pixel nearer one edge of its menu than the other. Whole pixels give the same gap on both
 * sides in every engine. Read once, as the surface mounts: a style read, no layout.
 */
export function snapInlinePadding(element: HTMLElement | null): void {
  if (!element || typeof window === "undefined") return;
  const style = getComputedStyle(element);
  for (const side of ["paddingLeft", "paddingRight", "marginLeft", "marginRight"] as const) {
    const px = Number.parseFloat(style[side]);
    if (!Number.isFinite(px)) continue;
    const snapped = Math.round(px);
    if (Math.abs(snapped - px) > 0.001) element.style[side] = `${snapped}px`;
  }
}

/** A callback ref that snaps the surface's padding as it mounts and passes the element on. */
export function useSnappedPaddingRef<T extends HTMLElement>(
  ref: React.Ref<T> | undefined,
): (element: T | null) => void {
  return useCallback(
    (element: T | null) => {
      if (typeof ref === "function") ref(element);
      else if (ref) ref.current = element;
      snapInlinePadding(element);
    },
    [ref],
  );
}
