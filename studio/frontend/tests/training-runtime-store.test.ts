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
  assert.equal(runtime.tryBeginStarting("start-unconfirmed"), true);

  useTrainingRuntimeStore
    .getState()
    .setStartPending(null, "Checking training status", "start-unconfirmed");

  const pending = useTrainingRuntimeStore.getState();
  assert.equal(pending.jobId, null);
  assert.equal(pending.phase, "configuring");
  assert.equal(pending.isStarting, false);
  assert.equal(pending.startRequestId, "start-unconfirmed");
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

test("requesting a stop invalidates an in-flight start lease", () => {
  const runtime = useTrainingRuntimeStore.getState();
  runtime.resetRuntime();
  assert.equal(runtime.tryBeginStarting("start-stop"), true);
  const resetGeneration = useTrainingRuntimeStore.getState().resetGeneration;

  useTrainingRuntimeStore.getState().setStopRequested(true);

  const stopped = useTrainingRuntimeStore.getState();
  assert.equal(stopped.isStarting, false);
  assert.equal(stopped.stopRequested, true);
  assert.equal(stopped.resetGeneration, resetGeneration + 1);

  stopped.setStopRequested(false);
});

test("a terminal stream error invalidates earlier runtime requests", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const runtime = useTrainingRuntimeStore.getState();
  const resetGeneration = runtime.resetGeneration;

  runtime.setRuntimeError("Stream failed");

  const failed = useTrainingRuntimeStore.getState();
  assert.equal(failed.phase, "error");
  assert.equal(failed.resetGeneration, resetGeneration + 1);
});

test("training warnings survive subsequent status updates and reset for a new run", () => {
  const runtime = useTrainingRuntimeStore.getState();
  runtime.resetRuntime();

  runtime.applyStatus({
    job_id: "job-1",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Training",
    error: null,
    warnings: [" Evaluation was disabled. ", "Evaluation was disabled.", ""],
  });
  runtime.applyStatus({
    job_id: "job-1",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Step 2",
    error: null,
  });

  assert.deepEqual(useTrainingRuntimeStore.getState().warnings, [
    "Evaluation was disabled.",
  ]);

  useTrainingRuntimeStore.getState().setStartPending("job-2", "Starting");
  assert.deepEqual(useTrainingRuntimeStore.getState().warnings, []);
});

function progressPayload(jobId: string, step: number) {
  return {
    job_id: jobId,
    step,
    total_steps: 10,
    loss: 1.25,
    learning_rate: 0.0001,
    progress_percent: step * 10,
    epoch: 0.5,
    elapsed_seconds: 5,
    eta_seconds: 5,
    grad_norm: 0.75,
    num_tokens: 100,
    eval_loss: null,
  };
}

test("progress cannot establish a run identity", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const before = useTrainingRuntimeStore.getState();

  before.applyProgress(progressPayload("job-unscoped", 3), 3);

  assert.strictEqual(useTrainingRuntimeStore.getState(), before);
});

test("stale and unidentified progress cannot mutate the current run", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-current", "Starting");
  const before = useTrainingRuntimeStore.getState();

  before.applyProgress(progressPayload("job-stale", 7), 7);
  assert.strictEqual(useTrainingRuntimeStore.getState(), before);

  before.applyProgress(progressPayload("", 8), 8);
  assert.strictEqual(useTrainingRuntimeStore.getState(), before);
});

test("matching progress updates metrics without replacing the run identity", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-current", "Starting");

  useTrainingRuntimeStore
    .getState()
    .applyProgress(progressPayload("job-current", 4), 4);

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.jobId, "job-current");
  assert.equal(current.currentStep, 4);
  assert.equal(current.lastEventId, 4);
  assert.deepEqual(current.lossHistory, [{ step: 4, value: 1.25 }]);
});

test("accepting a job already adopted from status preserves live progress", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const runtime = useTrainingRuntimeStore.getState();
  assert.equal(runtime.tryBeginStarting("start-current"), true);

  runtime.applyStatus({
    job_id: "job-current",
    start_request_id: "start-current",
    start_request_state: "accepted",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Training",
    error: null,
  });
  useTrainingRuntimeStore
    .getState()
    .applyProgress(progressPayload("job-current", 4), 4);
  const generation = useTrainingRuntimeStore.getState().resetGeneration;

  useTrainingRuntimeStore
    .getState()
    .setStartPending("job-current", "Training started");

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.isStarting, false);
  assert.equal(current.startRequestId, null);
  assert.equal(current.message, "Training");
  assert.equal(current.currentStep, 4);
  assert.equal(current.lastEventId, 4);
  assert.deepEqual(current.lossHistory, [{ step: 4, value: 1.25 }]);
  assert.equal(current.resetGeneration, generation);
});

test("same-job updates cannot roll live progress backward", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-current", "Starting");
  const currentProgress = {
    ...progressPayload("job-current", 10),
    loss: 0.5,
    learning_rate: 0.00005,
    progress_percent: 50,
  };
  useTrainingRuntimeStore.getState().applyProgress(currentProgress, 10);
  const liveState = useTrainingRuntimeStore.getState();

  liveState.applyProgress(progressPayload("job-current", 0), 0);
  assert.strictEqual(useTrainingRuntimeStore.getState(), liveState);

  useTrainingRuntimeStore.getState().applyStatus({
    job_id: "job-current",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Older status",
    error: null,
    details: {
      step: 9,
      total_steps: 10,
      loss: 2,
      learning_rate: 0.001,
      epoch: 0.4,
    },
    metric_history: {
      steps: [8, 10],
      loss: [2.5, 9],
      lr: [0.002, 0.009],
    },
  });
  useTrainingRuntimeStore.getState().applyMetrics({
    job_id: "job-current",
    loss_history: [3, 2],
    lr_history: [0.003, 0.002],
    step_history: [7, 9],
    grad_norm_history: [1],
    grad_norm_step_history: [9],
    current_loss: 2,
    current_lr: 0.002,
    current_step: 9,
  });

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.currentStep, 10);
  assert.equal(current.currentLoss, 0.5);
  assert.equal(current.currentLearningRate, 0.00005);
  assert.equal(current.progressPercent, 50);
  assert.equal(current.lastEventId, 10);
  assert.deepEqual(current.lossHistory, [
    { step: 7, value: 3 },
    { step: 8, value: 2.5 },
    { step: 9, value: 2 },
    { step: 10, value: 0.5 },
  ]);
  assert.deepEqual(current.lrHistory, [
    { step: 7, value: 0.003 },
    { step: 8, value: 0.002 },
    { step: 9, value: 0.002 },
    { step: 10, value: 0.00005 },
  ]);
});

test("a graceful stop keeps accepting progress for the same run", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-current", "Starting");
  useTrainingRuntimeStore.getState().setStopRequested(true);

  useTrainingRuntimeStore
    .getState()
    .applyProgress(progressPayload("job-current", 5), 5);

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.jobId, "job-current");
  assert.equal(current.currentStep, 5);
  assert.equal(current.lastEventId, 5);
});

test("status adoption resets data and invalidates requests from the prior run", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-old", "Starting");
  useTrainingRuntimeStore
    .getState()
    .applyProgress(progressPayload("job-old", 6), 6);
  const priorGeneration = useTrainingRuntimeStore.getState().resetGeneration;

  useTrainingRuntimeStore.getState().applyStatus({
    job_id: "job-new",
    phase: "configuring",
    is_training_running: true,
    eval_enabled: false,
    message: "Preparing",
    error: null,
  });

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.jobId, "job-new");
  assert.equal(current.currentStep, 0);
  assert.equal(current.lastEventId, null);
  assert.deepEqual(current.lossHistory, []);
  assert.equal(current.resetGeneration, priorGeneration + 1);
});

test("local start status adoption preserves the active request lease", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const runtime = useTrainingRuntimeStore.getState();
  assert.equal(runtime.tryBeginStarting("start-local"), true);
  runtime.setStartResources(
    "model-local",
    "dataset-local",
    true,
    "project-local",
  );
  const priorGeneration = useTrainingRuntimeStore.getState().resetGeneration;

  useTrainingRuntimeStore.getState().applyStatus({
    job_id: "job-local",
    start_request_id: "start-local",
    start_request_state: "pending",
    phase: "configuring",
    is_training_running: true,
    eval_enabled: false,
    message: "Preparing",
    error: null,
  });

  let current = useTrainingRuntimeStore.getState();
  assert.equal(current.isStarting, true);
  assert.equal(current.startRequestId, "start-local");
  assert.equal(current.resetGeneration, priorGeneration + 1);
  assert.equal(current.startModelName, "model-local");
  assert.equal(current.startDatasetName, "dataset-local");
  assert.equal(current.startProjectName, "project-local");
  assert.equal(current.startFromResume, true);

  current.applyStatus({
    job_id: "job-local",
    start_request_id: "start-local",
    start_request_state: "accepted",
    phase: "configuring",
    is_training_running: true,
    eval_enabled: false,
    message: "Starting",
    error: null,
  });

  current = useTrainingRuntimeStore.getState();
  assert.equal(current.isStarting, true);
  assert.equal(current.startRequestId, "start-local");
});

test("external job adoption clears labels from the prior run", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const runtime = useTrainingRuntimeStore.getState();
  assert.equal(runtime.tryBeginStarting("start-local"), true);
  runtime.setStartResources(
    "model-local",
    "dataset-local",
    true,
    "project-local",
  );

  runtime.applyStatus({
    job_id: "job-external",
    start_request_id: "start-external",
    start_request_state: "accepted",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Training",
    error: null,
  });

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.isStarting, true);
  assert.equal(current.startRequestId, "start-local");
  assert.equal(current.startModelName, null);
  assert.equal(current.startDatasetName, null);
  assert.equal(current.startProjectName, null);
  assert.equal(current.startFromResume, false);
});

test("late adoption of an unconfirmed request preserves its run labels", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const runtime = useTrainingRuntimeStore.getState();
  assert.equal(runtime.tryBeginStarting("start-late"), true);
  runtime.setStartResources("model-late", "dataset-late", true, "project-late");
  runtime.setStartPending(null, "Checking status", "start-late");

  useTrainingRuntimeStore.getState().applyStatus({
    job_id: "job-late",
    start_request_id: "start-late",
    start_request_state: "accepted",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Training",
    error: null,
  });

  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.startRequestId, null);
  assert.equal(current.startModelName, "model-late");
  assert.equal(current.startDatasetName, "dataset-late");
  assert.equal(current.startProjectName, "project-late");
  assert.equal(current.startFromResume, true);
});

test("external status adoption does not erase a start failure", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  const runtime = useTrainingRuntimeStore.getState();
  runtime.setStartError("Start rejected");

  runtime.applyStatus({
    job_id: "job-external",
    start_request_id: "start-external",
    start_request_state: "accepted",
    phase: "training",
    is_training_running: true,
    eval_enabled: false,
    message: "Training",
    error: null,
  });

  assert.equal(useTrainingRuntimeStore.getState().startError, "Start rejected");
});

test("metrics are scoped to the current run", () => {
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-current", "Starting");
  const runtime = useTrainingRuntimeStore.getState();
  const staleMetrics = {
    job_id: "job-stale",
    loss_history: [2],
    lr_history: [0.001],
    step_history: [8],
    grad_norm_history: [1],
    grad_norm_step_history: [8],
    current_loss: 2,
    current_lr: 0.001,
    current_step: 8,
  };

  runtime.applyMetrics(staleMetrics);
  assert.strictEqual(useTrainingRuntimeStore.getState(), runtime);

  runtime.applyMetrics({ ...staleMetrics, job_id: "job-current" });
  const current = useTrainingRuntimeStore.getState();
  assert.equal(current.currentStep, 8);
  assert.deepEqual(current.lossHistory, [{ step: 8, value: 2 }]);
});
