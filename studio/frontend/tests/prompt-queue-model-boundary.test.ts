import assert from "node:assert/strict";
import test from "node:test";

import {
  localPromptQueueModelBoundary,
  planLocalPromptQueueStop,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
} from "../src/features/chat/utils/prompt-queue-model-boundary.ts";

test("a local model stop preserves an active external item and holds local follow-ups", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: false, dispatched: true },
        { usesLocalModel: true, dispatched: false },
        { usesLocalModel: false, dispatched: false },
      ],
      0,
    ),
    {
      cancelActiveItem: false,
      retainedItemIndexes: [0, 1, 2],
      heldItemIndexes: [1],
    },
  );
});

test("a dispatched local item is cancelled without losing external follow-ups", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: true },
        { usesLocalModel: false, dispatched: false },
      ],
      0,
    ),
    {
      cancelActiveItem: true,
      retainedItemIndexes: [1],
      heldItemIndexes: [],
    },
  );
});

test("a reload mid-reply drops only the reply in flight and holds the queued prompts (#10428)", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: true },
        { usesLocalModel: true, dispatched: false },
        { usesLocalModel: true, dispatched: false },
      ],
      0,
    ),
    {
      cancelActiveItem: true,
      retainedItemIndexes: [1, 2],
      heldItemIndexes: [1, 2],
    },
  );
});

test("a paused queue keeps every prompt across a reload (#10428)", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: false },
        { usesLocalModel: true, dispatched: false },
      ],
      0,
    ),
    {
      cancelActiveItem: false,
      retainedItemIndexes: [0, 1],
      heldItemIndexes: [0, 1],
    },
  );
});

test("a queue waiting on a direct send is held, not dropped", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: false },
        { usesLocalModel: false, dispatched: false },
      ],
      -1,
    ),
    {
      cancelActiveItem: false,
      retainedItemIndexes: [0, 1],
      heldItemIndexes: [0],
    },
  );
});

test("completed queue history is preserved and only pending local work is held", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: true },
        { usesLocalModel: false, dispatched: true },
        { usesLocalModel: true, dispatched: false },
        { usesLocalModel: false, dispatched: false },
      ],
      1,
    ),
    {
      cancelActiveItem: false,
      retainedItemIndexes: [0, 1, 2, 3],
      heldItemIndexes: [2],
    },
  );
});

test("a local model boundary invalidates only pending local factories", () => {
  const capturedGeneration = localPromptQueueModelBoundary.capture();
  localPromptQueueModelBoundary.advance();

  assert.equal(
    shouldAbortPendingQueueForModelBoundary({
      capturedGeneration,
      usesLocalModel: true,
    }),
    true,
  );
  assert.equal(
    shouldAbortPendingQueueForModelBoundary({
      capturedGeneration,
      usesLocalModel: false,
    }),
    false,
  );
});

test("a new local factory is accepted within the current model boundary", () => {
  assert.equal(
    shouldAbortPendingQueueForModelBoundary({
      capturedGeneration: localPromptQueueModelBoundary.capture(),
      usesLocalModel: true,
    }),
    false,
  );
});

test("a hydrated queue aborts when accepted settings or temporary mode changed", () => {
  assert.equal(
    shouldAbortPendingQueueForSettingsChange({
      capturedEpoch: 4,
      currentEpoch: 4,
      capturedTemporary: false,
      currentTemporary: false,
    }),
    false,
  );
  assert.equal(
    shouldAbortPendingQueueForSettingsChange({
      capturedEpoch: 4,
      currentEpoch: 5,
      capturedTemporary: false,
      currentTemporary: false,
    }),
    true,
  );
  assert.equal(
    shouldAbortPendingQueueForSettingsChange({
      capturedEpoch: 4,
      currentEpoch: 4,
      capturedTemporary: false,
      currentTemporary: true,
    }),
    true,
  );
});
