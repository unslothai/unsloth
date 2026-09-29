// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RowBox {
  top: number;
  height: number;
}

/** Where the dragged row belongs for a pointer at `localY`, as an index into the current order.
 *  Counts the rows the pointer has passed rather than hit-testing the row under it, and skips the
 *  dragged row. Testing "which row contains the pointer" oscillates across uneven heights:
 *  dropping a short row onto a tall one leaves the tall row still under the pointer, sending it
 *  straight back. Midpoints of the others are stable. */
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

/** Whether a window-level pointer event belongs to the drag `activePointerId` started. The grip
 *  deliberately does not capture the pointer, so the listeners are on `window` and see every
 *  pointer on the page. Ending a drag clears the id before React can unsubscribe them, so
 *  treating "no active pointer" as a match lets a still-held button reorder the list after a
 *  window blur already ended the drag. */
export function ownsDrag(
  activePointerId: number | null,
  eventPointerId: number,
): boolean {
  return activePointerId !== null && activePointerId === eventPointerId;
}

/** The translateY each row needs so that a reorder appears to start from where the rows were
 *  before it. `first` must be read at the moment the reorder is requested: rows change height
 *  without the order changing, from the preview toggle and a textarea regrowing, so a baseline
 *  recorded at an earlier commit describes a layout that no longer exists. Sub-pixel moves are
 *  dropped rather than animated. */
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
