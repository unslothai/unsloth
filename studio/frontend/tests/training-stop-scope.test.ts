// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { trainingStopScope } from "../src/features/training/lib/training-stop-scope.ts";

test("a resolved job is always stopped by its job id", () => {
  assert.deepEqual(
    trainingStopScope({
      jobId: "job-current",
      startRequestId: null,
    }),
    { kind: "job", jobId: "job-current" },
  );
});

test("an unconfirmed start is cancelled by its request id", () => {
  assert.deepEqual(
    trainingStopScope({
      jobId: null,
      startRequestId: "start-pending",
    }),
    { kind: "start", startRequestId: "start-pending" },
  );
});

test("a pending request wins over its provisional job id", () => {
  assert.deepEqual(
    trainingStopScope({
      jobId: "job-reserved",
      startRequestId: "start-pending",
    }),
    { kind: "start", startRequestId: "start-pending" },
  );
});

test("missing runtime identity never produces an unscoped stop", () => {
  assert.equal(trainingStopScope({ jobId: null, startRequestId: null }), null);
  assert.equal(trainingStopScope({ jobId: " ", startRequestId: " " }), null);
});
