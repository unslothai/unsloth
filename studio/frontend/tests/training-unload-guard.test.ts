// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { subscribeToTrainingActivity } = await import(
  "../src/features/training/lib/training-activity.ts"
);
const { useTrainingRuntimeStore } = await import(
  "../src/features/training/stores/training-runtime-store.ts"
);

test("desktop activity mirrors pending starts through active phases", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const published: boolean[] = [];
  const unsubscribe = subscribeToTrainingActivity((active) => {
    published.push(active);
  });

  try {
    assert.deepEqual(published, [false]);
    assert.equal(
      useTrainingRuntimeStore.getState().tryBeginStarting("start-unload"),
      true,
    );
    assert.deepEqual(published, [false, true]);

    useTrainingRuntimeStore
      .getState()
      .setStartPending(null, "Checking training status");
    assert.deepEqual(published, [false, true]);

    useTrainingRuntimeStore.getState().setRuntimeError("Start failed");
    assert.deepEqual(published, [false, true, false]);
  } finally {
    unsubscribe();
    useTrainingRuntimeStore.getState().resetRuntime();
  }
});

test("desktop activity remains active until a stop request is reconciled", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const published: boolean[] = [];
  const unsubscribe = subscribeToTrainingActivity((active) => {
    published.push(active);
  });

  try {
    useTrainingRuntimeStore.getState().setStopRequested(true);
    assert.deepEqual(published, [false, true]);

    useTrainingRuntimeStore.getState().setRuntimeError("Status unavailable");
    assert.deepEqual(published, [false, true]);

    useTrainingRuntimeStore.getState().applyStatus({
      job_id: "",
      phase: "stopped",
      is_training_running: false,
      eval_enabled: false,
      message: "Stopped",
      error: null,
    });
    assert.deepEqual(published, [false, true, false]);
  } finally {
    unsubscribe();
    useTrainingRuntimeStore.getState().resetRuntime();
  }
});
