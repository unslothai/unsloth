// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createTrainingStreamScope,
  isTrainingProgressForScope,
  isTrainingStreamScopeCurrent,
} from "../src/features/training/lib/training-stream-scope.ts";

test("a stream scope requires a nonempty control-plane job identity", () => {
  assert.equal(createTrainingStreamScope({ jobId: null }), null);
  assert.equal(createTrainingStreamScope({ jobId: "" }), null);
  assert.deepEqual(createTrainingStreamScope({ jobId: "job-1" }), {
    jobId: "job-1",
  });
});

test("a stream scope expires when the runtime changes runs", () => {
  const scope = createTrainingStreamScope({ jobId: "job-1" });
  assert.ok(scope);
  assert.equal(isTrainingStreamScopeCurrent(scope, { jobId: "job-1" }), true);
  assert.equal(isTrainingStreamScopeCurrent(scope, { jobId: "job-2" }), false);
  assert.equal(isTrainingStreamScopeCurrent(scope, { jobId: null }), false);
});

test("stream progress must declare the exact scoped job", () => {
  const scope = createTrainingStreamScope({ jobId: "job-1" });
  assert.ok(scope);
  assert.equal(isTrainingProgressForScope(scope, { job_id: "job-1" }), true);
  assert.equal(isTrainingProgressForScope(scope, { job_id: "job-2" }), false);
  assert.equal(isTrainingProgressForScope(scope, { job_id: "" }), false);
});
