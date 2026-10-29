// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { diffusionTrainingProgress } from "../src/features/images/train/diffusion-train-progress.ts";

const running = {
  active: true,
  status: "running",
  step: 0,
  total_steps: 100,
  in_model_load: true,
  message: "Loading base model...",
};

test("image preparation advances its own bar before any training step", () => {
  const progress = diffusionTrainingProgress({
    ...running,
    preparation: { stage: "cache_latents", done: 48, total: 60 },
  });
  assert.equal(progress.title, "Preparing training data");
  assert.equal(progress.percent, 80);
  assert.equal(progress.count, "48 / 60 prepared");
  assert.equal(progress.preparing, true);
});

test("60/60 means data is prepared, not that training is complete", () => {
  const progress = diffusionTrainingProgress({
    ...running,
    preparation: { stage: "cache_latents", done: 60, total: 60 },
  });
  assert.equal(progress.title, "Training data prepared");
  assert.equal(progress.percent, 100);
  assert.match(progress.detail!, /before training starts/);
  assert.equal(progress.preparing, true);
});

test("unknown model-loading and first-step durations remain indeterminate", () => {
  assert.equal(diffusionTrainingProgress(running).percent, null);
  const firstStep = diffusionTrainingProgress({
    ...running,
    in_model_load: false,
  });
  assert.equal(firstStep.title, "Starting training");
  assert.equal(firstStep.percent, null);
  assert.match(firstStep.detail!, /Compilation/);
});

test("the first training step switches back to optimization progress", () => {
  const progress = diffusionTrainingProgress({
    ...running,
    step: 1,
    in_model_load: false,
  });
  assert.equal(progress.title, "Training");
  assert.equal(progress.percent, 1);
  assert.equal(progress.count, "1 / 100 steps");
  assert.equal(progress.preparing, false);
});

test("older backends' preparation messages still move the bar", () => {
  const progress = diffusionTrainingProgress({
    ...running,
    message: "Preparing (cache latents 48/60)...",
  });
  assert.equal(progress.percent, 80);
  assert.equal(progress.count, "48 / 60 prepared");
});

test("unreadable preparation counts never fabricate progress", () => {
  for (const preparation of [
    { done: 0, total: 0 },
    { done: -1, total: 60 },
    { done: NaN, total: 60 },
    { done: 4, total: Infinity },
  ]) {
    assert.equal(
      diffusionTrainingProgress({ ...running, preparation }).percent,
      null,
    );
  }
});

test("stopped and failed runs do not keep an animated preparation bar", () => {
  for (const status of ["stopped", "error"]) {
    const progress = diffusionTrainingProgress({
      ...running,
      active: false,
      status,
    });
    assert.equal(progress.preparing, false);
    assert.equal(progress.percent, 0);
    assert.equal(progress.title, status === "stopped" ? "Stopped" : "Error");
  }
});

test("resuming a positive-step run still shows its preparation stage", () => {
  const progress = diffusionTrainingProgress({
    ...running,
    step: 20,
    preparation: { stage: "cache_latents", done: 30, total: 60 },
  });
  assert.equal(progress.preparing, true);
  assert.equal(progress.percent, 50);
});

test("warnings and stop requests stay visible while preparing", () => {
  const stopping =
    "Cancel requested; finishing the current step (no adapter will be saved)...";
  const prep = diffusionTrainingProgress({
    ...running,
    message: stopping,
    preparation: { stage: "cache_latents", done: 12, total: 60 },
  });
  assert.equal(prep.detail, stopping);
  assert.equal(prep.percent, 20);
  const warning = "torch.compile disabled (eager fallback): boom";
  const firstStep = diffusionTrainingProgress({
    ...running,
    in_model_load: false,
    message: warning,
  });
  assert.equal(firstStep.detail, warning);
  assert.match(
    diffusionTrainingProgress({
      ...running,
      message: "Preparing (cache latents 12/60)...",
    }).detail!,
    /Preparing images and captions/,
  );
});

test("a run stopped mid-preparation does not read as starting while its worker exits", () => {
  const progress = diffusionTrainingProgress({
    active: true,
    status: "stopped",
    step: 0,
    total_steps: 0,
    in_model_load: false,
    message: "Stopped (no adapter saved).",
    preparation: null,
  });
  assert.equal(progress.preparing, false);
  assert.equal(progress.title, "Stopped");
  assert.equal(progress.percent, 0);
  assert.equal(progress.detail, "Stopped (no adapter saved).");
});

test("encoded captions alone do not read as prepared training data", () => {
  const progress = diffusionTrainingProgress({
    ...running,
    preparation: { stage: "encode_prompts", done: 12, total: 12 },
  });
  assert.equal(progress.title, "Preparing training data");
  assert.equal(progress.percent, 100);
});


test("download bytes belong to the global manager, not the inline preparation bar", () => {
  const progress = diffusionTrainingProgress({ ...running, download_progress: {
    bytes_done: 2 * 1024 ** 3, bytes_total: 4 * 1024 ** 3, label: "model.safetensors",
  } });
  assert.equal(progress.title, "Loading training model");
  assert.equal(progress.percent, null);
  assert.equal(progress.count, null);
});

test("download progress yields to preparation and actual training", () => {
  const download_progress = { bytes_done: 1, bytes_total: 10 };
  const prep = diffusionTrainingProgress({ ...running, download_progress,
    preparation: { stage: "cache_latents", done: 8, total: 10 } });
  assert.equal(prep.title, "Preparing training data");
  assert.equal(prep.percent, 80);
  const train = diffusionTrainingProgress({ ...running, in_model_load: false, step: 20, download_progress });
  assert.equal(train.title, "Training");
  assert.equal(train.percent, 20);
});

test("unknown download totals never fabricate a percentage", () => {
  for (const total of [0, NaN, Infinity]) {
    assert.equal(diffusionTrainingProgress({ ...running,
      download_progress: { bytes_done: 1, bytes_total: total } }).percent, null);
  }
});
