import assert from "node:assert/strict";
import test from "node:test";

import {
  localPromptQueueModelBoundary,
  planLocalPromptQueueStop,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
} from "../src/features/chat/utils/prompt-queue-model-boundary.ts";

test("a local model stop preserves an active external item", () => {
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
      stopEntireRun: false,
      activeItemRemoved: false,
      retainedItemIndexes: [0, 2],
    },
  );
});

test("a dispatched local item stops its sequential run", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: true },
        { usesLocalModel: false, dispatched: false },
      ],
      0,
    ),
    {
      stopEntireRun: true,
      activeItemRemoved: true,
      retainedItemIndexes: [],
    },
  );
});

test("an undispatched local item is dropped without losing external follow-ups", () => {
  assert.deepEqual(
    planLocalPromptQueueStop(
      [
        { usesLocalModel: true, dispatched: false },
        { usesLocalModel: false, dispatched: false },
      ],
      -1,
    ),
    {
      stopEntireRun: false,
      activeItemRemoved: true,
      retainedItemIndexes: [1],
    },
  );
});

test("completed queue history is preserved when pending local work is dropped", () => {
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
      stopEntireRun: false,
      activeItemRemoved: false,
      retainedItemIndexes: [0, 1, 3],
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
      modelLoading: false,
    }),
    true,
  );
  assert.equal(
    shouldAbortPendingQueueForModelBoundary({
      capturedGeneration,
      usesLocalModel: false,
      modelLoading: false,
    }),
    false,
  );
});

test("a pending local factory cannot materialize during a model load", () => {
  assert.equal(
    shouldAbortPendingQueueForModelBoundary({
      capturedGeneration: localPromptQueueModelBoundary.capture(),
      usesLocalModel: true,
      modelLoading: true,
    }),
    true,
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
