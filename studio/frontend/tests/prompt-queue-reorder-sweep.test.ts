// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * An exhaustive sweep of the reorder math, on top of the named cases in
 * `prompt-queue-reorder.test.ts`.
 *
 * The named cases say what a drag should do; this file says what a drag may
 * never do, over every queue length, every from/to pair and every active slot
 * the UI can present. The invariants are the ones a queue engine depends on:
 * nothing is lost or duplicated, nothing at or before the dispatching slot
 * moves, the caller's array is never mutated, and the reported "the next
 * dispatch changed" flag agrees with what actually changed.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  canReorderPromptQueueRange,
  promptQueueActiveItemChanged,
  reorderPromptQueueItems,
} from "../src/features/chat/utils/prompt-queue-reorder.ts";

const MAX_LEN = 7;

function queue(length: number): string[] {
  return Array.from({ length }, (_, index) => `item-${index}`);
}

test("every accepted move is a permutation that loses nothing", () => {
  for (let length = 0; length <= MAX_LEN; length++) {
    const items = queue(length);
    for (let active = 0; active <= length; active++) {
      for (let from = 0; from < length; from++) {
        for (let to = 0; to < length; to++) {
          const next = reorderPromptQueueItems(items, from, to, active);
          if (!next) continue;
          assert.equal(next.length, items.length);
          assert.deepEqual([...next].sort(), [...items].sort());
          assert.equal(new Set(next).size, next.length);
        }
      }
    }
  }
});

test("the range check and the reorder never disagree", () => {
  // The check guards a move the UI is about to make, so a move the check
  // allows must produce an array and one it refuses must produce null. If the
  // two drift apart, a row either refuses a legal drag or slips past the
  // dispatch boundary.
  for (let length = 0; length <= MAX_LEN; length++) {
    const items = queue(length);
    for (let active = 0; active <= length; active++) {
      for (let from = -2; from <= length + 1; from++) {
        for (let to = -2; to <= length + 1; to++) {
          const allowed = canReorderPromptQueueRange(from, to, active, length);
          const next = reorderPromptQueueItems(items, from, to, active);
          assert.equal(
            allowed,
            next !== null,
            `length=${length} active=${active} from=${from} to=${to}`,
          );
        }
      }
    }
  }
});

test("the dragged item lands exactly where the drop said", () => {
  for (let length = 1; length <= MAX_LEN; length++) {
    const items = queue(length);
    for (let from = 0; from < length; from++) {
      for (let to = 0; to < length; to++) {
        const next = reorderPromptQueueItems(items, from, to, 0);
        if (!next) continue;
        assert.equal(
          next.indexOf(items[from]),
          to,
          `dragging ${from} onto ${to} in a queue of ${length}`,
        );
      }
    }
  }
});

test("nothing at or before the dispatching slot ever moves", () => {
  // The item in the active slot is the one about to be sent, and everything
  // before it has already gone. A move that reached them would either send a
  // prompt the user had moved out of the way or re-send a spent one.
  for (let length = 1; length <= MAX_LEN; length++) {
    const items = queue(length);
    for (let active = 0; active < length; active++) {
      for (let from = 0; from < length; from++) {
        for (let to = 0; to < length; to++) {
          const next = reorderPromptQueueItems(items, from, to, active);
          if (!next) continue;
          assert.deepEqual(
            next.slice(0, active),
            items.slice(0, active),
            `the spent head moved: active=${active} from=${from} to=${to}`,
          );
        }
      }
    }
  }
});

test("a move and its reverse restore the queue", () => {
  for (let length = 2; length <= MAX_LEN; length++) {
    const items = queue(length);
    for (let from = 0; from < length; from++) {
      for (let to = 0; to < length; to++) {
        const moved = reorderPromptQueueItems(items, from, to, 0);
        if (!moved) continue;
        assert.deepEqual(reorderPromptQueueItems(moved, to, from, 0), items);
      }
    }
  }
});

test("the caller's array is never mutated", () => {
  for (let length = 1; length <= MAX_LEN; length++) {
    const items = queue(length);
    const snapshot = [...items];
    for (let from = 0; from < length; from++) {
      for (let to = 0; to < length; to++) {
        reorderPromptQueueItems(items, from, to, 0);
        assert.deepEqual(items, snapshot);
      }
    }
  }
});

test("a frozen queue can be reordered", () => {
  // run.items is handed straight to this helper, and a store that freezes its
  // state in development would throw on any in-place splice.
  const frozen = Object.freeze(queue(4));
  assert.deepEqual(reorderPromptQueueItems(frozen, 0, 2, 0), [
    "item-1",
    "item-2",
    "item-0",
    "item-3",
  ]);
});

test("indices that are not real positions are refused", () => {
  const items = queue(4);
  for (const bad of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    1.5,
    -1,
    4,
    99,
    -0.5,
  ]) {
    assert.equal(reorderPromptQueueItems(items, bad, 1, 0), null, `from=${bad}`);
    assert.equal(reorderPromptQueueItems(items, 1, bad, 0), null, `to=${bad}`);
  }
  // -0 is an integer and index 0, so it is a real position and must be allowed
  // rather than falling into the refusals above.
  assert.deepEqual(reorderPromptQueueItems(items, -0, 2, 0), [
    "item-1",
    "item-2",
    "item-0",
    "item-3",
  ]);
});

test("an empty queue and a single row have nothing to reorder", () => {
  assert.equal(reorderPromptQueueItems([], 0, 0, 0), null);
  assert.equal(reorderPromptQueueItems(["only"], 0, 0, 0), null);
});

test("an active slot past the end refuses everything", () => {
  // The run has dispatched every item it holds; there is nothing left to move.
  const items = queue(4);
  for (let from = 0; from < 4; from++) {
    for (let to = 0; to < 4; to++) {
      assert.equal(reorderPromptQueueItems(items, from, to, 4), null);
    }
  }
});

test("the dispatch-changed flag agrees with the slot it reports on", () => {
  for (let length = 1; length <= MAX_LEN; length++) {
    const items = queue(length);
    for (let active = 0; active < length; active++) {
      for (let from = 0; from < length; from++) {
        for (let to = 0; to < length; to++) {
          const next = reorderPromptQueueItems(items, from, to, active);
          if (!next) continue;
          assert.equal(
            promptQueueActiveItemChanged(items, next, active),
            items[active] !== next[active],
            `active=${active} from=${from} to=${to}`,
          );
        }
      }
    }
  }
});

test("a run that has not dispatched yet reports no dispatch change", () => {
  // run.index is -1 before the first send, and there is no pending dispatch to
  // retarget, so the caller must not be told to reschedule one.
  const items = queue(3);
  const next = reorderPromptQueueItems(items, 0, 2, 0);
  assert.ok(next);
  assert.equal(promptQueueActiveItemChanged(items, next, -1), false);
});

test("the flag compares identity, not contents", () => {
  // Queue items are objects, so two prompts with the same text are still two
  // items. Comparing by value would miss a real change between them.
  const a = { prompt: "same" };
  const b = { prompt: "same" };
  assert.equal(promptQueueActiveItemChanged([a, b], [b, a], 0), true);
  assert.equal(promptQueueActiveItemChanged([a, b], [a, b], 0), false);
});
