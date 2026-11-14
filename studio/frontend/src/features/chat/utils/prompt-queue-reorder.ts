// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function canReorderPromptQueueRange(
  fromIndex: number,
  toIndex: number,
  activeIndex: number,
  itemCount: number,
): boolean {
  if (fromIndex === toIndex) return false;
  if (!Number.isInteger(fromIndex) || !Number.isInteger(toIndex)) return false;
  if (fromIndex < 0 || toIndex < 0) return false;
  if (fromIndex >= itemCount || toIndex >= itemCount) return false;
  return fromIndex >= activeIndex && toIndex >= activeIndex;
}

/** `toIndex` is read off the original array, matching how a dragged row reads on screen. */
export function reorderPromptQueueItems<T>(
  items: readonly T[],
  fromIndex: number,
  toIndex: number,
  activeIndex = 0,
): T[] | null {
  if (
    !canReorderPromptQueueRange(fromIndex, toIndex, activeIndex, items.length)
  ) {
    return null;
  }
  const next = [...items];
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  return next;
}

export function promptQueueActiveItemChanged<T>(
  before: readonly T[],
  after: readonly T[],
  activeIndex: number,
): boolean {
  if (activeIndex < 0) return false;
  return before[activeIndex] !== after[activeIndex];
}
