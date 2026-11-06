// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";
import { useCallback } from "react";

/**
 * Chromium draws hover pills on whole CSS pixels, so fractional scaled padding makes the gap
 * uneven. Rounds once at mount: a style read, no layout.
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

const ROW_SELECTOR =
  '[role="menuitem"],[role="menuitemcheckbox"],[role="menuitemradio"],[role="option"],[cmdk-item],button';

const ROW_WAIT_MS = 3000;

/** Measures the first row's real inset on each side and corrects both at the surface, not per layer. */
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
  // Insets within a pixel are meant to match, so both land on the same whole pixel.
  const even = Math.abs(left - right) < 1;
  const padLeft = Number.parseFloat(style.paddingLeft);
  const padRight = Number.parseFloat(style.paddingRight);
  let targetLeft = even ? Math.round((left + right) / 2) : Math.round(left);
  let targetRight = even ? targetLeft : Math.round(right);
  // A surface padded less than the correction can only widen its inset, so round up.
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

const snappedSurfaces = new WeakSet<HTMLElement>();

export function snapRowInsets(surface: HTMLElement | null): void {
  if (!surface || typeof window === "undefined" || snappedSurfaces.has(surface)) return;
  snappedSurfaces.add(surface);
  snapInlinePadding(surface);
  if (balanceRowInsets(surface) || typeof MutationObserver === "undefined") return;
  const observer = new MutationObserver(() => {
    if (!surface.isConnected || balanceRowInsets(surface)) observer.disconnect();
  });
  observer.observe(surface, { childList: true, subtree: true });
  window.setTimeout(() => observer.disconnect(), ROW_WAIT_MS);
}

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
