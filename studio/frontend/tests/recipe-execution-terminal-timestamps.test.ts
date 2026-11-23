// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Re-polled terminal snapshots must not move finishedAt; server timestamps are seconds.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { JobStatusResponse } from "../src/features/recipe-studio/api/index.ts";

register("./bundler-resolver.mjs", import.meta.url);

const { applyExecutionStatusSnapshot, createBaseExecutionRecord } =
  await import("../src/features/recipe-studio/executions/runtime.ts");

const BROWSER_START_MS = 1_763_000_000_000;
const SERVER_STARTED_AT = 1_763_003_610.3045754;
const SERVER_FINISHED_AT = 1_763_003_652.00544;
const SERVER_DURATION_MS = 41_701;

function baseExecution() {
  return createBaseExecutionRecord({
    recipeId: "recipe-1",
    kind: "preview",
    rows: 5,
    currentSignature: "sig",
  });
}

test("reapplying a terminal payload keeps the server duration on the browser clock", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  t.mock.timers.setTime(BROWSER_START_MS);
  const execution = baseExecution();
  const payload: JobStatusResponse = {
    job_id: "job-1",
    status: "completed",
    started_at: SERVER_STARTED_AT,
    finished_at: SERVER_FINISHED_AT,
  };

  t.mock.timers.setTime(BROWSER_START_MS + 60_000);
  const first = applyExecutionStatusSnapshot(execution, payload);
  t.mock.timers.setTime(BROWSER_START_MS + 900_000);
  const second = applyExecutionStatusSnapshot(first, payload);

  assert.equal(first.createdAt, BROWSER_START_MS);
  assert.equal(first.finishedAt, BROWSER_START_MS + SERVER_DURATION_MS);
  assert.equal(second.createdAt, first.createdAt);
  assert.equal(second.finishedAt, first.finishedAt);
  assert.ok(Number.isInteger(second.finishedAt));
});

test("a terminal payload without server timestamps keeps the first observed finish", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  t.mock.timers.setTime(BROWSER_START_MS);
  const execution = baseExecution();
  const payload = { job_id: "job-1", status: "completed" };

  t.mock.timers.setTime(BROWSER_START_MS + 42_000);
  const first = applyExecutionStatusSnapshot(execution, payload);
  assert.equal(first.finishedAt, BROWSER_START_MS + 42_000);

  t.mock.timers.setTime(BROWSER_START_MS + 900_000);
  const second = applyExecutionStatusSnapshot(first, payload);
  assert.equal(second.finishedAt, first.finishedAt);
});

test("an active snapshot leaves the run open and its start untouched", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  t.mock.timers.setTime(BROWSER_START_MS);
  const running = applyExecutionStatusSnapshot(baseExecution(), {
    job_id: "job-1",
    status: "active",
    started_at: SERVER_STARTED_AT,
    finished_at: null,
  });
  assert.equal(running.createdAt, BROWSER_START_MS);
  assert.equal(running.finishedAt, null);

  const failed = applyExecutionStatusSnapshot(running, {
    job_id: "job-1",
    status: "error",
    started_at: SERVER_STARTED_AT,
    finished_at: SERVER_FINISHED_AT,
  });
  assert.equal(failed.finishedAt, BROWSER_START_MS + SERVER_DURATION_MS);
});
