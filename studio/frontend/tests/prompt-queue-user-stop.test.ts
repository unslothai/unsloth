import assert from "node:assert/strict";
import test from "node:test";

import { planUserPromptQueueStop } from "../src/features/chat/utils/prompt-queue-user-stop.ts";

test("a dispatched stop keeps pending follow-ups and pauses", () => {
  assert.deepEqual(
    planUserPromptQueueStop(
      [
        { dispatched: true },
        { dispatched: false },
        { dispatched: false },
      ],
      0,
    ),
    {
      cancelActiveItem: true,
      retainedItemIndexes: [1, 2],
      pause: true,
    },
  );
});

test("a dispatched stop with no follow-ups leaves nothing to resume", () => {
  assert.deepEqual(
    planUserPromptQueueStop([{ dispatched: true }], 0),
    {
      cancelActiveItem: true,
      retainedItemIndexes: [],
      pause: false,
    },
  );
});

test("an undispatched stop keeps the whole queue and pauses", () => {
  assert.deepEqual(
    planUserPromptQueueStop(
      [{ dispatched: false }, { dispatched: false }],
      0,
    ),
    {
      cancelActiveItem: false,
      retainedItemIndexes: [0, 1],
      pause: true,
    },
  );
});

test("completed history is kept when a later dispatched item is stopped", () => {
  assert.deepEqual(
    planUserPromptQueueStop(
      [
        { dispatched: true },
        { dispatched: true },
        { dispatched: false },
      ],
      1,
    ),
    {
      cancelActiveItem: true,
      retainedItemIndexes: [0, 2],
      pause: true,
    },
  );
});

test("a queue still waiting to start is paused without cancelling", () => {
  assert.deepEqual(
    planUserPromptQueueStop(
      [{ dispatched: false }, { dispatched: false }],
      -1,
    ),
    {
      cancelActiveItem: false,
      retainedItemIndexes: [0, 1],
      pause: true,
    },
  );
});
