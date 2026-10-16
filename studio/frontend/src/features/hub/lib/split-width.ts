// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SPLIT_LIST_MIN_PX = 340;
export const SPLIT_DETAIL_MIN_PX = 480;
export const SPLIT_RESIZE_STEP_PX = 16;

export function splitListMax(totalPx: number): number {
  return Math.max(SPLIT_LIST_MIN_PX, totalPx - SPLIT_DETAIL_MIN_PX);
}

export function clampSplitListWidth(px: number, totalPx: number): number {
  return Math.min(Math.max(px, SPLIT_LIST_MIN_PX), splitListMax(totalPx));
}

/**
 * Width after an arrow key, stepped from the RENDERED width: a stored width
 * wider than the window allows is shown clamped, and stepping from it would
 * clamp straight back to the same edge.
 */
export function keyboardSplitWidth(
  renderedPx: number,
  key: string,
  totalPx: number,
): number | null {
  if (key !== "ArrowLeft" && key !== "ArrowRight") return null;
  const step =
    key === "ArrowRight" ? SPLIT_RESIZE_STEP_PX : -SPLIT_RESIZE_STEP_PX;
  return clampSplitListWidth(renderedPx + step, totalPx);
}
