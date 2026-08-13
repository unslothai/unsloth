// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How close to an edge starts the scroll, and the per-frame cap in px.
export const AUTOSCROLL_EDGE = 48;
export const AUTOSCROLL_MAX_STEP = 14;

/**
 * Per-frame scroll step for a drag held near the edge of a scrolling pane.
 *
 * Negative scrolls up, positive scrolls down, zero leaves the pane alone. The
 * step ramps with how far into the edge band the pointer sits, so easing off
 * slows the scroll instead of stopping it dead.
 */
export function autoscrollDelta(
  pointerY: number,
  paneTop: number,
  paneBottom: number,
  edge: number = AUTOSCROLL_EDGE,
  maxStep: number = AUTOSCROLL_MAX_STEP,
): number {
  // A pane shorter than two edge bands would otherwise scroll in both
  // directions at once; bias to the half the pointer is actually in.
  const band = Math.min(edge, (paneBottom - paneTop) / 2);
  if (band <= 0) return 0;

  if (pointerY < paneTop + band) {
    const depth = Math.min(band, paneTop + band - pointerY);
    return -(depth / band) * maxStep;
  }
  if (pointerY > paneBottom - band) {
    const depth = Math.min(band, pointerY - (paneBottom - band));
    return (depth / band) * maxStep;
  }
  return 0;
}
