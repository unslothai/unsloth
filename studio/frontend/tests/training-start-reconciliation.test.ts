// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { statusConfirmsActiveTrainingStart } from "../src/features/training/lib/training-start-reconciliation.ts";

test("transport reconciliation requires an active backend job", () => {
  assert.equal(
    statusConfirmsActiveTrainingStart({
      job_id: "job_123",
      is_training_running: true,
    }),
    true,
  );
  assert.equal(
    statusConfirmsActiveTrainingStart({
      job_id: "",
      is_training_running: true,
    }),
    false,
  );
  assert.equal(
    statusConfirmsActiveTrainingStart({
      job_id: "job_123",
      is_training_running: false,
    }),
    false,
  );
});
