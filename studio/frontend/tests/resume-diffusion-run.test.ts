// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionTrainingRunDetail } from "../src/features/images/api.ts";

const { buildDiffusionResumePayload, resumeActionLabel } = await import(
  "../src/features/images/train/resume-diffusion-run.ts"
);

// A finished run's persisted record, as GET /api/train/diffusion/runs/{id} returns it. `config`
// is the scrubbed start request the run was launched with, which a resume replays verbatim.
function stoppedRun(
  overrides: Partial<DiffusionTrainingRunDetail> = {},
): DiffusionTrainingRunDetail {
  return {
    job_id: "a".repeat(32),
    status: "stopped",
    step: 11,
    total_steps: 500,
    saved: true,
    can_resume: true,
    checkpoint_step: 11,
    checkpoint_path: "/studio/outputs/my-lora/checkpoint-11",
    output_dir: "/studio/outputs/my-lora",
    config: {
      base_model: "stabilityai/sdxl-turbo",
      data_dir: "/studio/datasets/my-images",
      output_dir: "/studio/outputs/my-lora",
      train_steps: 500,
      lora_rank: 16,
      seed: 42,
      // Left over from the run that produced this record; both must be replaced, not inherited.
      resume_from_checkpoint: "/studio/outputs/my-lora/checkpoint-3",
      resumed_from_job_id: "b".repeat(32),
    },
    ...overrides,
  };
}

test("replays the run's own config and points it at the run's output directory", () => {
  const payload = buildDiffusionResumePayload(stoppedRun(), { hfToken: "hf_x" });
  // train_steps is the TARGET TOTAL: the backend continues at 12 and stops at 500.
  assert.equal(payload.train_steps, 500);
  assert.equal(payload.lora_rank, 16);
  assert.equal(payload.seed, 42);
  assert.equal(payload.base_model, "stabilityai/sdxl-turbo");
  // The EXACT bundle the backend named, not just the folder: two runs can share an output
  // directory, so "newest in that folder" is not necessarily the step the UI is showing.
  assert.equal(
    payload.resume_from_checkpoint,
    "/studio/outputs/my-lora/checkpoint-11",
  );
  assert.equal(payload.resumed_from_job_id, "a".repeat(32));
  assert.equal(payload.hf_token, "hf_x");
});

test("falls back to the run folder when the backend names no bundle", () => {
  const payload = buildDiffusionResumePayload(
    stoppedRun({ checkpoint_path: null }),
  );
  assert.equal(payload.resume_from_checkpoint, "/studio/outputs/my-lora");
});

test("refuses a run the backend says cannot resume, quoting its reason", () => {
  assert.throws(
    () =>
      buildDiffusionResumePayload(
        stoppedRun({
          can_resume: false,
          resume_blocked_reason: "The training images have changed since this checkpoint.",
        }),
      ),
    /training images have changed/,
  );
});

test("refuses a record with no output directory rather than guessing one", () => {
  assert.throws(
    () => buildDiffusionResumePayload(stoppedRun({ output_dir: null })),
    /no checkpoint to continue from/,
  );
});

test("refuses a record whose stored settings are incomplete", () => {
  const run = stoppedRun();
  delete (run.config as Record<string, unknown>).base_model;
  assert.throws(() => buildDiffusionResumePayload(run), /settings are incomplete/);
});

test("the action label names the step it would continue from", () => {
  assert.equal(resumeActionLabel({ checkpoint_step: 11 }), "Resume from step 11");
  assert.equal(resumeActionLabel({ checkpoint_step: null }), "Resume training");
});
