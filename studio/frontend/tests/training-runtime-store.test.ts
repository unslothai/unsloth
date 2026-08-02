// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../src/features/training/stores/training-runtime-store.ts";

test("an unconfirmed start remains pending without inventing a job id", () => {
  const runtime = useTrainingRuntimeStore.getState();
  runtime.resetRuntime();
  assert.equal(runtime.tryBeginStarting(), true);

  useTrainingRuntimeStore
    .getState()
    .setStartPending(null, "Checking training status");

  const pending = useTrainingRuntimeStore.getState();
  assert.equal(pending.jobId, null);
  assert.equal(pending.phase, "configuring");
  assert.equal(pending.isStarting, false);
  assert.equal(isTrainingStartPending(pending), true);
});

test("start-pending protection spans start synchronization and active phases", () => {
  assert.equal(
    isTrainingStartPending({
      phase: "idle",
      isStarting: false,
      isTrainingRunning: false,
    }),
    false,
  );
  assert.equal(
    isTrainingStartPending({
      phase: "idle",
      isStarting: true,
      isTrainingRunning: false,
    }),
    true,
  );
  assert.equal(
    isTrainingStartPending({
      phase: "configuring",
      isStarting: false,
      isTrainingRunning: false,
    }),
    true,
  );
  assert.equal(
    isTrainingStartPending({
      phase: "training",
      isStarting: false,
      isTrainingRunning: false,
    }),
    true,
  );
  assert.equal(
    isTrainingStartPending({
      phase: "error",
      isStarting: false,
      isTrainingRunning: false,
    }),
    false,
  );
});
