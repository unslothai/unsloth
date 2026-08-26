// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RowBox {
  top: number;
  height: number;
}

/**
 * Where the dragged row belongs for a pointer at `localY`, as an index into the
 * current order.
 *
 * Counts the rows the pointer has passed rather than hit-testing the row under
 * it, and skips the dragged row itself. Testing "which row contains the
 * pointer" oscillates across uneven heights: dropping a short row onto a tall
 * one leaves the tall row still under the pointer, which sends it straight
 * back. Midpoints of the others are stable, because after the move the dragged
 * row occupies the space that shifted them.
 */
export function insertionIndex(
  rows: readonly (RowBox | undefined)[],
  from: number,
  localY: number,
): number {
  let to = 0;
  for (let i = 0; i < rows.length; i++) {
    if (i === from) continue;
    const row = rows[i];
    if (!row) continue;
    if (row.top + row.height / 2 < localY) to++;
  }
  return to;
}
