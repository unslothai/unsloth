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

/** What a surface's hover pills are drawn on: its menu rows, list options and plain buttons. */
const ROW_SELECTOR =
  '[role="menuitem"],[role="menuitemcheckbox"],[role="menuitemradio"],[role="option"],[cmdk-item],button';

/** How long a surface waits for rows that render after it opens (a list still loading). */
const ROW_WAIT_MS = 3000;

/**
 * Nudges the surface's side padding so its first full-width row sits a whole number of CSS
 * pixels from each edge. A picker's rows often sit in lists inside the surface, each layer with a
 * scaled inset of its own; rounding the layers one by one could round the two sides apart, so
 * this measures the row's actual inset on each side and corrects both at the surface. Rows are
 * never touched, so every row keeps the same inset. Returns whether a row was there to measure.
 */
function balanceRowInsets(surface: HTMLElement): boolean {
  const width = surface.offsetWidth;
  if (!width) return false;
  let row: HTMLElement | undefined;
  for (const candidate of surface.querySelectorAll<HTMLElement>(ROW_SELECTOR)) {
    if (candidate.offsetWidth >= width * 0.7) {
      row = candidate;
      break;
    }
  }
  if (!row) return false;
  const box = surface.getBoundingClientRect();
  const rect = row.getBoundingClientRect();
  // The open animation scales the surface; offsets scale with it.
  const scale = box.width / width || 1;
  const style = getComputedStyle(surface);
  const left = (rect.left - box.left) / scale;
  const right = (box.right - rect.right) / scale;
  // Insets within a pixel of each other are meant to match (the surface's own rounding can
  // split them), so both land on the same whole pixel; others each round to their nearest.
  const even = Math.abs(left - right) < 1;
  const padLeft = Number.parseFloat(style.paddingLeft);
  const padRight = Number.parseFloat(style.paddingRight);
  let targetLeft = even ? Math.round((left + right) / 2) : Math.round(left);
  let targetRight = even ? targetLeft : Math.round(right);
  // A surface padded less than the correction (a list padded inside a bare surface) can only
  // widen its inset, so it rounds up instead.
  if (padLeft + targetLeft - left < 0 || padRight + targetRight - right < 0) {
    targetLeft = even ? Math.ceil(Math.max(left, right)) : Math.ceil(left);
    targetRight = even ? targetLeft : Math.ceil(right);
  }
  const insets = [
    ["paddingLeft", padLeft + targetLeft - left],
    ["paddingRight", padRight + targetRight - right],
  ] as const;
  for (const [side, padded] of insets) {
    if (Math.abs(padded - Number.parseFloat(style[side])) > 0.01) {
      surface.style[side] = `${padded}px`;
    }
  }
  return true;
}

/**
 * Rounds a surface's own padding and margin, then balances its rows' insets, now or once its
 * rows render.
 */
export function snapRowInsets(surface: HTMLElement | null): void {
  if (!surface || typeof window === "undefined") return;
  snapInlinePadding(surface);
  if (balanceRowInsets(surface) || typeof MutationObserver === "undefined") return;
  const observer = new MutationObserver(() => {
    if (!surface.isConnected || balanceRowInsets(surface)) observer.disconnect();
  });
  observer.observe(surface, { childList: true, subtree: true });
  window.setTimeout(() => observer.disconnect(), ROW_WAIT_MS);
}

/** A callback ref that snaps the surface's padding as it mounts and passes the element on. */
export function useSnappedPaddingRef<T extends HTMLElement>(
  ref: React.Ref<T> | undefined,
): (element: T | null) => void {
  return useCallback(
    (element: T | null) => {
      if (typeof ref === "function") ref(element);
      else if (ref) ref.current = element;
      snapRowInsets(element);
    },
    [ref],
  );
}
