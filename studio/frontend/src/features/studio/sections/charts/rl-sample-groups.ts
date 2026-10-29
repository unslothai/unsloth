// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const KEEP_GROUPS = 12;

export function keepNewest<T>(prev: T[], fresh: T[]): T[] {
  return [...prev, ...fresh].slice(-KEEP_GROUPS);
}

// The selection is a group's seq, not its position, so trimming old groups never moves it to
// another sample. null follows the newest group; a trimmed selection falls to the oldest kept.
export function shownIndex(
  groups: readonly { seq: number }[],
  selectedSeq: number | null,
): number {
  if (selectedSeq === null) {
    return groups.length - 1;
  }
  return Math.max(
    0,
    groups.findIndex((g) => g.seq === selectedSeq),
  );
}
