// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { TrainingStartRequest } from "../types/api.ts";
import {
  getResumeStepConflict,
  withImportedResumeCheckpoint,
} from "./resume-start-payload.ts";

test("an inspected checkpoint replaces resume sources restored from older configs", () => {
  const payload = {
    model_name: "org/model",
    resume_from_checkpoint: "/old/output",
    resume_checkpoint_path: "/old/checkpoint",
    in_place_continuation: true,
  } as TrainingStartRequest;

  const result = withImportedResumeCheckpoint(payload, "/chosen/checkpoint-20");

  assert.equal(result.resume_from_checkpoint, null);
  assert.equal(result.resume_checkpoint_path, null);
  assert.equal(result.imported_resume_checkpoint, "/chosen/checkpoint-20");
  assert.equal(result.in_place_continuation, false);
  assert.equal(result.model_name, "org/model");
});

test("reports when a checkpoint has no remaining configured steps", () => {
  assert.match(
    getResumeStepConflict(31, 30) ?? "",
    /already reached Max Steps/,
  );
  assert.equal(getResumeStepConflict(20, 30), null);
  assert.equal(getResumeStepConflict(30, 0), null);
});
