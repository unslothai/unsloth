// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { terminalJobStatus } from "../src/features/rag/types/rag.ts";

test("completed, failed, and cancelled jobs are terminal", () => {
  assert.equal(terminalJobStatus("completed"), "completed");
  assert.equal(terminalJobStatus("failed"), "failed");
  assert.equal(terminalJobStatus("cancelled"), "cancelled");
});

test("pending and running jobs are not terminal", () => {
  assert.equal(terminalJobStatus("pending"), null);
  assert.equal(terminalJobStatus("running"), null);
});
