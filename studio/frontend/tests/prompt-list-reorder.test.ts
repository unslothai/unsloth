// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type RowBox,
  flipShifts,
  insertionIndex,
  ownsDrag,
} from "../src/features/chat/prompt-storage/reorder.ts";

import { readSrcAsync } from "./helpers/kit.ts";

const GAP = 2;

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
  assert.equal(insertionIndex(rows, 0, 20), 0);
  assert.equal(insertionIndex(rows, 1, 20), 0);
});

test("a short row dragged onto a tall one settles instead of oscillating", () => {
  // After the swap the tall row is still under the pointer; bottom-edge hit-testing flips back.
  let heights = [40, 200];
  const pointer = 150;

  const first = step(heights, 0, pointer);
  assert.deepEqual(first.order, [1, 0]);
  heights = first.heights;

  const second = step(heights, 1, pointer);
  assert.equal(second.to, 1);
  assert.deepEqual(second.order, [0, 1]);
});

test("dragging back up over the tall row settles too", () => {
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

// The grip does not capture the pointer, so window listeners see every pointer.
test("only the pointer that started the drag drives it", () => {
  assert.equal(ownsDrag(3, 3), true);
  assert.equal(ownsDrag(3, 7), false, "a second finger reordered with its own y");
});

// React unsubscribes listeners a commit after the drag ends.
test("an ended drag owns no pointer at all", () => {
  assert.equal(ownsDrag(null, 3), false, "a blur-ended drag kept reordering");
  assert.equal(ownsDrag(null, 0), false, "pointer id 0 is a real id, not absent");
});

function offsets(height: number): Map<string, number> {
  const map = new Map<string, number>();
  ["i1", "i2", "i3"].forEach((uid, i) => map.set(uid, i * (height + GAP)));
  return map;
}

test("only the rows a reorder moved are animated", () => {
  const before = offsets(100);
  const after = new Map([
    ["i2", 0],
    ["i1", 102],
    ["i3", 204],
  ]);
  assert.deepEqual(
    [...flipShifts(before, after)].sort(),
    [
      ["i1", -102],
      ["i2", 102],
    ].sort(),
    "a row that did not move must not be transformed",
  );
});

// Rows change height without reordering, so a stale baseline shifts unmoved rows.
test("a baseline from the wrong heights moves rows that stayed put", () => {
  const stale = offsets(40);
  const after = new Map([
    ["i2", 0],
    ["i1", 102],
    ["i3", 204],
  ]);
  const shifts = flipShifts(stale, after);
  assert.equal(shifts.get("i3"), -120, "the untouched last row jumps 120px");
  assert.equal(shifts.get("i2"), 42, "the swap animates from the wrong distance");
});

test("a row with no baseline is left alone rather than animated from zero", () => {
  const shifts = flipShifts(new Map([["i1", 0]]), new Map([["i9", 300]]));
  assert.equal(shifts.has("i9"), false);
});

test("sub-pixel settling is not worth a transform", () => {
  const shifts = flipShifts(new Map([["i1", 10]]), new Map([["i1", 10.4]]));
  assert.equal(shifts.size, 0);
});

test("the baseline is captured when the reorder is requested", async () => {
  const source = await readSrcAsync("features/chat/prompt-storage/sortable-prompt-items.tsx");
  const captures = source.split("prevOffsets.current = measureOffsets();").length - 1;
  assert.equal(captures, 1, "the FLIP baseline is captured somewhere else too");
  const [beforeApply] = source.split("const applyOrder");
  assert.doesNotMatch(
    beforeApply,
    /prevOffsets\.current =/,
    "the baseline is recorded from a commit again, which the heights outrun",
  );
});
