// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RowBox {
  top: number;
  height: number;
}

/** Counts passed midpoints rather than hit-testing, which oscillates across uneven heights. */
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

/** No active pointer is not a match, or a held button reorders after blur ended the drag. */
export function ownsDrag(
  activePointerId: number | null,
  eventPointerId: number,
): boolean {
  return activePointerId !== null && activePointerId === eventPointerId;
}

/** `first` must be read when the reorder is requested; row heights change independently. */
export function flipShifts(
  first: ReadonlyMap<string, number>,
  last: ReadonlyMap<string, number>,
): Map<string, number> {
  const shifts = new Map<string, number>();
  last.forEach((offset, uid) => {
    const from = first.get(uid);
    if (from === undefined) return;
    const dy = from - offset;
    if (Math.abs(dy) >= 1) shifts.set(uid, dy);
  });
  return shifts;
}
