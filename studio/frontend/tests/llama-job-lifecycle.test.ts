// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  LLAMA_JOB_POLL_MISS_LIMIT,
  llamaJobPollTick,
  llamaUpdateAdoptsRunningJob,
  llamaUpdatePresentation,
  ownedLlamaSwitchOutcome,
} from "../src/lib/llama-job-lifecycle.ts";

const SWITCH_STARTED_AT = "2026-08-12T15:00:00Z";

test("an owned switch recognizes only its explicit running and terminal states", () => {
  for (const state of ["running", "success", "error"] as const) {
    assert.equal(
      ownedLlamaSwitchOutcome(
        { state, operation: "switch", startedAt: SWITCH_STARTED_AT },
        SWITCH_STARTED_AT,
      ),
      state,
    );
  }
});

test("a lost or replaced switch is interrupted rather than successful", () => {
  for (const job of [
    { state: "idle" as const, operation: null, startedAt: null },
    {
      state: "success" as const,
      operation: "update" as const,
      startedAt: SWITCH_STARTED_AT,
    },
    {
      state: "running" as const,
      operation: "switch" as const,
      startedAt: "a-different-job",
    },
    {
      state: "success" as const,
      operation: "switch" as const,
      startedAt: null,
    },
  ]) {
    assert.equal(
      ownedLlamaSwitchOutcome(job, SWITCH_STARTED_AT),
      "interrupted",
    );
  }
});

test("a running switch hides the update banner without showing update progress", () => {
  assert.deepEqual(
    llamaUpdatePresentation(true, {
      state: "running",
      operation: "switch",
    }),
    { applying: false, visible: false, running: true },
  );
});

test("every terminal switch status restores a pending update", () => {
  for (const state of ["success", "error", "idle"] as const) {
    assert.deepEqual(
      llamaUpdatePresentation(true, { state, operation: "switch" }),
      { applying: false, visible: true, running: false },
    );
  }
});

test("a completed update stays hidden when no update remains", () => {
  assert.deepEqual(
    llamaUpdatePresentation(false, {
      state: "success",
      operation: "update",
    }),
    { applying: false, visible: false, running: false },
  );
});

test("an apply adopts an already-running update but never a switch", () => {
  // Both share one job. Following a switch here would resolve the update action
  // as applied while the release it offered is still not installed.
  assert.equal(
    llamaUpdateAdoptsRunningJob("already_running", {
      state: "running",
      operation: "update",
    }),
    true,
  );
  assert.equal(
    llamaUpdateAdoptsRunningJob("already_running", {
      state: "running",
      operation: "switch",
    }),
    false,
  );
  assert.equal(
    llamaUpdateAdoptsRunningJob("up_to_date", {
      state: "success",
      operation: "update",
    }),
    false,
  );
});

test("a fetch that answers keeps the poll alive, whatever the job says", () => {
  assert.deepEqual(
    llamaJobPollTick({ update_available: true, job: { state: "running", operation: "update" } }, 0),
    { kind: "polling" },
  );
  // A successful fetch also clears an earlier failure streak.
  assert.deepEqual(
    llamaJobPollTick({ update_available: false, job: { state: "running", operation: "switch" } }, 9),
    { kind: "polling" },
  );
});

test("a terminal job state finishes the poll", () => {
  for (const state of ["success", "error", "idle"] as const) {
    assert.deepEqual(
      llamaJobPollTick({ update_available: false, job: { state, operation: "update" } }, 0),
      { kind: "finished", state },
    );
  }
});

test("a sustained fetch-failure streak stalls instead of pinning the toast", () => {
  // #9196: the poll owns the applying flag, so a run whose every fetch fails
  // kept the update toast pinned forever; one miss short of the limit is still
  // polling, at the limit it is stalled.
  assert.deepEqual(llamaJobPollTick(null, LLAMA_JOB_POLL_MISS_LIMIT - 1), { kind: "polling" });
  assert.deepEqual(llamaJobPollTick(null, LLAMA_JOB_POLL_MISS_LIMIT), { kind: "stalled" });
  assert.deepEqual(llamaJobPollTick(null, LLAMA_JOB_POLL_MISS_LIMIT + 5), { kind: "stalled" });
});
