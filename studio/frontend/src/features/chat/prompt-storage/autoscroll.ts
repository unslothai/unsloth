// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How close to an edge starts the scroll, and the per-frame cap in px.
export const AUTOSCROLL_EDGE = 48;
export const AUTOSCROLL_MAX_STEP = 14;

/**
 * Per-frame scroll step for a drag held near a pane edge. Negative scrolls up,
 * positive down, zero holds. Ramps with depth into the edge band.
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

export interface VerticalSpan {
  top: number;
  bottom: number;
}

/**
 * A pane's own rect intersected with everything clipping it. A nested scroller
 * reports its full unclipped height, so its edge can sit past the ancestor that
 * hides it and past the viewport, where the pointer can never reach it.
 * Null once nothing is left visible.
 */
export function clipSpan(
  span: VerticalSpan,
  clips: readonly VerticalSpan[],
): VerticalSpan | null {
  let { top, bottom } = span;
  for (const clip of clips) {
    top = Math.max(top, clip.top);
    bottom = Math.min(bottom, clip.bottom);
  }
  return bottom > top ? { top, bottom } : null;
}
