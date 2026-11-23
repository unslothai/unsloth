// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useTrainingRuntimeStore } =
  await import("../src/features/training/stores/training-runtime-store.ts");

function status(rl: Record<string, number>[]) {
  return {
    job_id: "job-1",
    phase: "training",
    is_training_running: true,
    message: "",
    error: null,
    details: null,
    metric_history: { rl },
  } as never;
}

test("a status poll that trails SSE keeps the reward points already shown", () => {
  useTrainingRuntimeStore.setState(
    useTrainingRuntimeStore.getInitialState?.() ?? {},
    true,
  );
  useTrainingRuntimeStore.setState({
    jobId: "job-1",
    rlMetricHistory: [1, 2, 3].map((step) => ({
      step,
      values: { reward: step },
    })),
  } as never);

  useTrainingRuntimeStore
    .getState()
    .applyStatus(status([{ step: 1, reward: 1 }]));
  assert.deepEqual(
    useTrainingRuntimeStore.getState().rlMetricHistory.map((p) => p.step),
    [1, 2, 3],
  );

  useTrainingRuntimeStore.getState().applyStatus(
    status([
      { step: 1, reward: 1 },
      { step: 4, reward: 4 },
    ]),
  );
  assert.deepEqual(
    useTrainingRuntimeStore.getState().rlMetricHistory.map((p) => p.step),
    [1, 2, 3, 4],
  );
});
