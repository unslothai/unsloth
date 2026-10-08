// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const AUTOSCROLL_EDGE = 48;
export const AUTOSCROLL_MAX_STEP = 14;

/** Per-frame scroll step near a pane edge: negative up, positive down, 0 holds. */
export function autoscrollDelta(
  pointerY: number,
  paneTop: number,
  paneBottom: number,
  edge: number = AUTOSCROLL_EDGE,
  maxStep: number = AUTOSCROLL_MAX_STEP,
): number {
  // Bands overlap on short panes: use the half the pointer is in.
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

/** Pane rect clipped by ancestors and viewport (nested scrollers report full height); null if hidden. */
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
