// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// Saving an edited source only starts the re-index, so the modal has to wait for
// the job before it can call the save a success. These cover the waiting: that a
// terminal status ends it, that a failure comes back rather than throwing, and
// that a job which never settles fails instead of hanging the dialog open.

const { pollJobUntilTerminal } = await import(
  "../src/features/rag/lib/poll-job.ts"
);

/** Answer with each status in turn, repeating the last. */
function jobReturning(statuses: string[]) {
  let calls = 0;
  const fetchJob = async (jobId: string) => {
    const status = statuses[Math.min(calls, statuses.length - 1)];
    calls += 1;
    return { id: jobId, documentId: "doc-1", status } as never;
  };
  return { fetchJob, calls: () => calls };
}

const fast = { pollMs: 0 };

test("a job that is already finished returns on the first read", async () => {
  const job = jobReturning(["completed"]);
  const result = await pollJobUntilTerminal(job.fetchJob, "job-1", fast);
  assert.equal(result.status, "completed");
  assert.equal(job.calls(), 1, "a settled job must not be polled twice");
});

test("polling continues until the job leaves the running state", async () => {
  const job = jobReturning(["pending", "running", "completed"]);
  const result = await pollJobUntilTerminal(job.fetchJob, "job-1", fast);
  assert.equal(result.status, "completed");
  assert.equal(job.calls(), 3);
});

test("a failed re-index is returned, not thrown", async () => {
  // The caller distinguishes "the job finished and failed" from "the wait
  // broke", so a failure has to come back as a value it can report.
  const job = jobReturning(["running", "failed"]);
  const result = await pollJobUntilTerminal(job.fetchJob, "job-1", fast);
  assert.equal(result.status, "failed");
});

test("a cancelled job also ends the wait", async () => {
  const job = jobReturning(["cancelled"]);
  assert.equal(
    (await pollJobUntilTerminal(job.fetchJob, "job-1", fast)).status,
    "cancelled",
  );
});

test("a job that never settles times out instead of hanging", async () => {
  // A worker can die without ever writing a terminal status. The dialog would
  // otherwise sit on a spinner with Save disabled and the edit unrecoverable.
  const job = jobReturning(["running"]);
  await assert.rejects(
    () => pollJobUntilTerminal(job.fetchJob, "job-1", { ...fast, timeoutMs: 0 }),
    /Timed out waiting for indexing/,
  );
});

test("a transport failure propagates rather than looping forever", async () => {
  const fetchJob = async () => {
    throw new Error("network down");
  };
  await assert.rejects(
    () => pollJobUntilTerminal(fetchJob, "job-1", fast),
    /network down/,
  );
});
