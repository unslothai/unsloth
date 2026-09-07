// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type PromptQueueUserStopItem = {
  dispatched: boolean;
};

export type UserPromptQueueStopPlan = {
  cancelActiveItem: boolean;
  retainedItemIndexes: number[];
  pause: boolean;
};

/** Stop generation without discarding prompts that have not been sent yet. */
export function planUserPromptQueueStop(
  items: readonly PromptQueueUserStopItem[],
  runIndex: number,
): UserPromptQueueStopPlan {
  if (runIndex < 0) {
    return {
      cancelActiveItem: false,
      retainedItemIndexes: items.map((_, index) => index),
      pause: items.length > 0,
    };
  }
  const activeIndex = runIndex;
  const activeItem = items[activeIndex];
  const dropActive = Boolean(activeItem?.dispatched);
  const retainedItemIndexes = items.flatMap((_, index) =>
    dropActive && index === activeIndex ? [] : [index],
  );
  return {
    cancelActiveItem: dropActive,
    retainedItemIndexes,
    pause: retainedItemIndexes.some((index) => !items[index]?.dispatched),
  };
}
