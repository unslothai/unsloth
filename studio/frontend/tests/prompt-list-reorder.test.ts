// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type RowBox,
  insertionIndex,
} from "../src/features/chat/prompt-storage/reorder.ts";

const GAP = 2;

// Lay rows out the way the flex column does, so a reorder can be replayed
// against the geometry it actually produces.
function layout(heights: number[]): RowBox[] {
  const rows: RowBox[] = [];
  let top = 0;
  for (const height of heights) {
    rows.push({ top, height });
    top += height + GAP;
  }
  return rows;
}

function move<T>(arr: T[], from: number, to: number): T[] {
  const next = [...arr];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

// One drag frame: hit-test, then re-lay-out at the resulting order.
function step(
  heights: number[],
  from: number,
  localY: number,
): { order: number[]; heights: number[]; to: number } {
  const to = insertionIndex(layout(heights), from, localY);
  const order = move(
    heights.map((_, i) => i),
    from,
    to,
  );
  return { order, heights: move(heights, from, to), to };
}

test("a pointer above every midpoint targets the first slot", () => {
  assert.equal(insertionIndex(layout([40, 40, 40]), 2, 0), 0);
});

test("a pointer past every midpoint targets the last slot", () => {
  assert.equal(insertionIndex(layout([40, 40, 40]), 0, 1000), 2);
});

test("the dragged row's own box does not count towards its target", () => {
  const rows = layout([40, 40, 40]);
  // Pointer inside row 0, which is also the row being dragged.
  assert.equal(insertionIndex(rows, 0, 20), 0);
  // The same pointer with row 1 dragged instead: row 0's midpoint is above it.
  assert.equal(insertionIndex(rows, 1, 20), 0);
});

test("a short row dragged onto a tall one settles instead of oscillating", () => {
  // The case bottom-edge hit-testing gets wrong: after the swap the tall row is
  // still under the pointer, which sends the short row straight back.
  let heights = [40, 200];
  const pointer = 150;

  const first = step(heights, 0, pointer);
  assert.deepEqual(first.order, [1, 0]);
  heights = first.heights;

  // Same pointer, next frame: the dragged row is now at index 1 and stays.
  const second = step(heights, 1, pointer);
  assert.equal(second.to, 1);
  assert.deepEqual(second.order, [0, 1]);
});

test("dragging back up over the tall row settles too", () => {
  // Continue from the settled [tall, short] order above.
  let heights = [200, 40];
  const pointer = 90;

  const first = step(heights, 1, pointer);
  assert.deepEqual(first.order, [1, 0]);
  heights = first.heights;

  const second = step(heights, 0, pointer);
  assert.equal(second.to, 0);
});

test("a drag across uneven rows reaches a fixed point at every pointer height", () => {
  const heights = [40, 200, 60, 24, 120, 80];
  for (let from = 0; from < heights.length; from++) {
    for (let pointer = -40; pointer <= 640; pointer += 4) {
      const first = step(heights, from, pointer);
      const settledAt = first.to;
      // Replaying the same pointer against the new layout must not move it
      // again, or the drag flip-flops for as long as the pointer is held.
      const second = step(first.heights, settledAt, pointer);
      assert.equal(
        second.to,
        settledAt,
        `row ${from} at y=${pointer} moved to ${settledAt} then ${second.to}`,
      );
    }
  }
});

test("rows without a measured box are skipped rather than counted", () => {
  const rows = layout([40, 40, 40]);
  assert.equal(insertionIndex([rows[0], undefined, rows[2]], 0, 1000), 1);
});
