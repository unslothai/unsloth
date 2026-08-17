// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
* Why this file exists (unslothai/unsloth#7897): the training bar sat at 100% with no
* completion. `applyStatus` never touches `progressPercent`, so once the SSE has reported step
* N/N the bar stays at 100 whatever phase the status poll reports. Reaching 100% means the
* optimizer loop ended, NOT that the save succeeded, so completion must come from the phase.
*/

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useTrainingRuntimeStore, shouldShowTrainingView } = await import(
  "../src/features/training/stores/training-runtime-store.ts"
);

function reset() {
  useTrainingRuntimeStore.setState(
    useTrainingRuntimeStore.getInitialState?.() ?? {},
    true,
  );
  // applyProgress ignores payloads whose job_id does not match, so a run has to be adopted first.
  useTrainingRuntimeStore.setState({ jobId: "job-1" } as never);
}

function status(partial: Record<string, unknown>) {
  return {
    job_id: "job-1",
    phase: "idle",
    is_training_running: false,
    message: "",
    error: null,
    details: null,
    metric_history: null,
    ...partial,
  } as never;
}

function progress(partial: Record<string, unknown>) {
  return {
    job_id: "job-1",
    step: 0,
    total_steps: 126,
    loss: 0.5,
    learning_rate: 1e-4,
    progress_percent: 0,
    epoch: 1,
    elapsed_seconds: 1,
    eta_seconds: null,
    grad_norm: null,
    num_tokens: null,
    eval_loss: null,
    ...partial,
  } as never;
}

test("100% does not imply completion - the bar stays pinned while phase goes idle", () => {
  reset();
  const store = useTrainingRuntimeStore.getState();
  store.applyProgress(progress({ step: 126, progress_percent: 100 }), 126);

  assert.equal(useTrainingRuntimeStore.getState().progressPercent, 100);
  assert.equal(useTrainingRuntimeStore.getState().currentStep, 126);

  // The status poll settles the run without a `completed` phase.
  useTrainingRuntimeStore
    .getState()
    .applyStatus(status({ phase: "idle", is_training_running: false }));

  const after = useTrainingRuntimeStore.getState();
  assert.equal(after.phase, "idle");
  // This is the reported symptom: a bar reading 100% with nothing terminal.
  assert.equal(after.progressPercent, 100);
  assert.equal(after.currentStep, 126);
  // ...and the view stays mounted because currentStep > 0, so the user sees it.
  assert.equal(shouldShowTrainingView(after), true);
});

test("a real completion is carried by the phase, not the percentage", () => {
  reset();
  const store = useTrainingRuntimeStore.getState();
  store.applyProgress(progress({ step: 126, progress_percent: 100 }), 126);
  store.applyStatus(status({ phase: "completed", is_training_running: false }));

  const after = useTrainingRuntimeStore.getState();
  assert.equal(after.phase, "completed");
  assert.equal(after.isTrainingRunning, false);
});

test("the post-training save is visible as its own phase, not silent 'training'", () => {
  reset();
  const store = useTrainingRuntimeStore.getState();
  store.applyProgress(progress({ step: 126, progress_percent: 100 }), 126);
  store.applyStatus(
    status({
      phase: "finalizing",
      is_training_running: true,
      message: "Saving model...",
    }),
  );

  const after = useTrainingRuntimeStore.getState();
  assert.equal(after.phase, "finalizing");
  // Still running, so live sync/SSE must stay on.
  assert.equal(after.isTrainingRunning, true);
  assert.equal(after.progressPercent, 100);
});

test("applyStatus clears stopRequested once the run is no longer running", () => {
  reset();
  useTrainingRuntimeStore.setState({ stopRequested: true } as never);
  useTrainingRuntimeStore
    .getState()
    .applyStatus(status({ phase: "training", is_training_running: true }));
  assert.equal(useTrainingRuntimeStore.getState().stopRequested, true);

  useTrainingRuntimeStore
    .getState()
    .applyStatus(status({ phase: "completed", is_training_running: false }));
  assert.equal(useTrainingRuntimeStore.getState().stopRequested, false);
});

test("a non-finite loss at a NEW step clears the display instead of going stale", () => {
  reset();
  const store = useTrainingRuntimeStore.getState();
  store.applyProgress(progress({ step: 10, loss: 0.42 }), 10);
  assert.equal(useTrainingRuntimeStore.getState().currentLoss, 0.42);

  // Backend reports a non-finite loss as null at a later step.
  useTrainingRuntimeStore
    .getState()
    .applyProgress(progress({ step: 11, loss: null }), 11);
  assert.equal(useTrainingRuntimeStore.getState().currentLoss, null);
});

test("a null loss at the SAME step keeps the last good value", () => {
  reset();
  const store = useTrainingRuntimeStore.getState();
  store.applyProgress(progress({ step: 10, loss: 0.42 }), 10);
  useTrainingRuntimeStore
    .getState()
    .applyProgress(progress({ step: 10, loss: null }), 10);
  assert.equal(useTrainingRuntimeStore.getState().currentLoss, 0.42);
});
