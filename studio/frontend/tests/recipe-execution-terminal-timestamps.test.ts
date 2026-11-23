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

const STARTED_AT_SEC = 1_763_000_010;
const STARTED_AT_SEC_FLOAT = 1_763_000_010.3045754;
const FINISHED_AT_SEC_FLOAT = 1_763_000_052.00544;
const FINISHED_AT_SEC = 1_763_000_052;

function baseExecution() {
  return createBaseExecutionRecord({
    recipeId: "recipe-1",
    kind: "preview",
    rows: 5,
    currentSignature: "sig",
  });
}

test("reapplying an identical terminal payload keeps the same finish and duration", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  t.mock.timers.setTime(STARTED_AT_SEC * 1000);
  const execution = baseExecution();
  const payload: JobStatusResponse = {
    job_id: "job-1",
    status: "completed",
    started_at: STARTED_AT_SEC,
    finished_at: FINISHED_AT_SEC,
  };

  t.mock.timers.setTime(FINISHED_AT_SEC * 1000);
  const first = applyExecutionStatusSnapshot(execution, payload);
  const firstDuration = (first.finishedAt ?? 0) - first.createdAt;

  t.mock.timers.setTime(FINISHED_AT_SEC * 1000 + 900_000);
  const second = applyExecutionStatusSnapshot(first, payload);

  assert.equal(second.finishedAt, first.finishedAt);
  assert.equal(second.finishedAt, FINISHED_AT_SEC * 1000);
  assert.equal(second.createdAt, STARTED_AT_SEC * 1000);
  assert.equal((second.finishedAt ?? 0) - second.createdAt, firstDuration);
  assert.equal(firstDuration, (FINISHED_AT_SEC - STARTED_AT_SEC) * 1000);
});

test("a terminal payload without server timestamps keeps the first observed finish", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  const createdAt = 1_763_000_000_000;
  t.mock.timers.setTime(createdAt);
  const execution = baseExecution();
  const payload = { job_id: "job-1", status: "completed" };

  t.mock.timers.setTime(createdAt + 42_000);
  const first = applyExecutionStatusSnapshot(execution, payload);
  assert.equal(first.finishedAt, createdAt + 42_000);
  assert.equal(first.createdAt, createdAt);

  t.mock.timers.setTime(createdAt + 900_000);
  const second = applyExecutionStatusSnapshot(first, payload);
  assert.equal(second.finishedAt, first.finishedAt);
  assert.equal(
    (second.finishedAt ?? 0) - second.createdAt,
    (first.finishedAt ?? 0) - first.createdAt,
  );
});

test("server timestamps are seconds, not milliseconds", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  t.mock.timers.setTime(STARTED_AT_SEC * 1000);
  const execution = baseExecution();

  const running = applyExecutionStatusSnapshot(execution, {
    job_id: "job-1",
    status: "active",
    started_at: STARTED_AT_SEC,
    finished_at: null,
  });
  assert.equal(running.createdAt, STARTED_AT_SEC * 1000);
  assert.equal(running.finishedAt, null);

  const failed = applyExecutionStatusSnapshot(running, {
    job_id: "job-1",
    status: "error",
    started_at: STARTED_AT_SEC,
    finished_at: FINISHED_AT_SEC,
  });
  assert.equal(failed.createdAt, STARTED_AT_SEC * 1000);
  assert.equal(failed.finishedAt, FINISHED_AT_SEC * 1000);
  assert.equal(
    (failed.finishedAt ?? 0) - failed.createdAt,
    (FINISHED_AT_SEC - STARTED_AT_SEC) * 1000,
  );
});

test("fractional server seconds land as integer milliseconds", (t) => {
  t.mock.timers.enable({ apis: ["Date"] });
  t.mock.timers.setTime(STARTED_AT_SEC * 1000);
  const done = applyExecutionStatusSnapshot(baseExecution(), {
    job_id: "job-1",
    status: "completed",
    started_at: STARTED_AT_SEC_FLOAT,
    finished_at: FINISHED_AT_SEC_FLOAT,
  });
  assert.equal(done.createdAt, 1_763_000_010_305);
  assert.equal(done.finishedAt, 1_763_000_052_005);
});
