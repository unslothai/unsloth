import assert from "node:assert/strict";
import test from "node:test";

import {
  localPromptQueueModelBoundary,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
} from "../src/features/chat/utils/prompt-queue-model-boundary.ts";

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
