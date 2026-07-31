// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  resolveTrainingStartRequestOutcome,
  statusConfirmsActiveTrainingStart,
} from "../src/features/training/lib/training-start-reconciliation.ts";

test("start request reconciliation distinguishes pending and rejected outcomes", () => {
  assert.deepEqual(
    resolveTrainingStartRequestOutcome(
      {
        start_request_id: "request_123",
        job_id: "job_123",
        state: "pending",
        message: "Validating",
        error: null,
      },
      "request_123",
    ),
    { kind: "pending", jobId: "job_123", message: "Validating" },
  );
  assert.deepEqual(
    resolveTrainingStartRequestOutcome(
      {
        start_request_id: "request_123",
        job_id: "job_123",
        state: "accepted",
        message: "Queued",
        error: null,
      },
      "request_123",
    ),
    { kind: "accepted", jobId: "job_123", message: "Queued" },
  );
  assert.deepEqual(
    resolveTrainingStartRequestOutcome(
      {
        start_request_id: "request_123",
        job_id: "job_123",
        state: "rejected",
        message: "Rejected",
        error: "Model unavailable",
      },
      "request_123",
    ),
    { kind: "rejected", error: "Model unavailable" },
  );
});

test("transport reconciliation requires an active backend job", () => {
  assert.equal(
    statusConfirmsActiveTrainingStart(
      {
        job_id: "job_123",
        is_training_running: true,
        start_request_id: "request_123",
      },
      "request_123",
    ),
    true,
  );
  assert.equal(
    statusConfirmsActiveTrainingStart(
      {
        job_id: "",
        is_training_running: true,
        start_request_id: "request_123",
      },
      "request_123",
    ),
    false,
  );
  assert.equal(
    statusConfirmsActiveTrainingStart(
      {
        job_id: "job_123",
        is_training_running: false,
        start_request_id: "request_123",
      },
      "request_123",
    ),
    false,
  );
  assert.equal(
    statusConfirmsActiveTrainingStart(
      {
        job_id: "job_other",
        is_training_running: true,
        start_request_id: "request_other",
      },
      "request_123",
    ),
    false,
  );
});
