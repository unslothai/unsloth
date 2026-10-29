// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const KEEP_GROUPS = 12;

// The newest groups, plus the one being browsed even when it is older than those: a trim must
// never move the card to another sample while the user reads it.
export function keepNewest<T extends { seq: number }>(
  prev: T[],
  fresh: T[],
  selectedSeq: number | null = null,
): T[] {
  const all = [...prev, ...fresh];
  const kept = all.slice(-KEEP_GROUPS);
  if (selectedSeq === null || kept.some((g) => g.seq === selectedSeq)) {
    return kept;
  }
  const selected = all.find((g) => g.seq === selectedSeq);
  return selected ? [selected, ...kept] : kept;
}

// The selection is a group's seq, not its position. null follows the newest group; a selection
// that is gone (a new run) falls to the oldest kept.
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
