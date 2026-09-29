// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  canReorderPromptQueueRange,
  promptQueueActiveItemChanged,
  reorderPromptQueueItems,
} from "../src/features/chat/utils/prompt-queue-reorder.ts";

const QUEUE = ["a", "b", "c", "d"];

test("a downward drag lands after the target", () => {
  assert.deepEqual(reorderPromptQueueItems(QUEUE, 0, 2), ["b", "c", "a", "d"]);
});

test("an upward drag lands before the target", () => {
  assert.deepEqual(reorderPromptQueueItems(QUEUE, 3, 1), ["a", "d", "b", "c"]);
});

test("adjacent rows swap", () => {
  assert.deepEqual(reorderPromptQueueItems(QUEUE, 1, 2), ["a", "c", "b", "d"]);
  assert.deepEqual(reorderPromptQueueItems(QUEUE, 2, 1), ["a", "c", "b", "d"]);
});

test("the queue keeps every item, and the source array is untouched", () => {
  const next = reorderPromptQueueItems(QUEUE, 0, 3);
  assert.ok(next);
  assert.deepEqual([...next].sort(), [...QUEUE].sort());
  assert.deepEqual(QUEUE, ["a", "b", "c", "d"]);
});

test("a move onto itself is refused", () => {
  assert.equal(reorderPromptQueueItems(QUEUE, 2, 2), null);
});

test("nothing crosses the item about to dispatch", () => {
  // activeIndex 1: "a" is spent, so it is neither a source nor a destination.
  assert.equal(reorderPromptQueueItems(QUEUE, 0, 2, 1), null);
  assert.equal(reorderPromptQueueItems(QUEUE, 2, 0, 1), null);
  // Moves at or past the active slot still go through.
  assert.deepEqual(reorderPromptQueueItems(QUEUE, 1, 3, 1), [
    "a",
    "c",
    "d",
    "b",
  ]);
});

test("out-of-range and non-integer indices are refused", () => {
  assert.equal(reorderPromptQueueItems(QUEUE, -1, 2), null);
  assert.equal(reorderPromptQueueItems(QUEUE, 0, 4), null);
  assert.equal(reorderPromptQueueItems(QUEUE, 1.5, 2), null);
  assert.equal(reorderPromptQueueItems([], 0, 0), null);
});

test("the range check agrees with the reorder it guards", () => {
  for (let from = -1; from <= QUEUE.length; from += 1) {
    for (let to = -1; to <= QUEUE.length; to += 1) {
      assert.equal(
        canReorderPromptQueueRange(from, to, 1, QUEUE.length),
        reorderPromptQueueItems(QUEUE, from, to, 1) !== null,
        `from ${from} to ${to}`,
      );
    }
  }
});

test("a move into the active slot reports the dispatch target changed", () => {
  const before = QUEUE;
  const after = reorderPromptQueueItems(before, 2, 1, 1);
  assert.ok(after);
  assert.equal(promptQueueActiveItemChanged(before, after, 1), true);
});

test("a move below the active slot leaves the dispatch target alone", () => {
  const before = QUEUE;
  const after = reorderPromptQueueItems(before, 2, 3, 1);
  assert.ok(after);
  assert.equal(promptQueueActiveItemChanged(before, after, 1), false);
});

test("a run with no active item reports no change", () => {
  // run.index is -1 before the first dispatch, so there is nothing to retarget.
  assert.equal(
    promptQueueActiveItemChanged(QUEUE, ["b", "a", "c", "d"], -1),
    false,
  );
});
