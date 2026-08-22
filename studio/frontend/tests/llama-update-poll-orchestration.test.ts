// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  llamaStatusRequestIsStale,
  llamaUpdatePresentation,
} from "../src/lib/llama-job-lifecycle.ts";

const source = await readFile(
  new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
  "utf8",
);
const startMarker = "pollTimer.current = setInterval(async () => {";
const endMarker = "}, JOB_POLL_INTERVAL_MS);";
const start = source.indexOf(startMarker);
const end = source.indexOf(endMarker, start + startMarker.length);
assert.ok(start >= 0 && end > start, "the llama job interval moved");
const body = source.slice(start + startMarker.length, end);

test("active polling uses the lightweight job endpoint before full reconciliation", () => {
  assert.match(body, /requestJob\(\)/);
  assert.match(source, /\/api\/llama\/update-job-status/);
  assert.match(body, /const reconciled = await requestStatus\(\)/);
  assert.match(source, /requestStatus\(true\)\.then/);
});

const AsyncFunction = Object.getPrototypeOf(async () => undefined)
  .constructor as new (
  ...args: string[]
) => (...args: unknown[]) => Promise<void>;
const runTick = new AsyncFunction(
  "pollInFlightGeneration",
  "generation",
  "requestJob",
  "requestStatus",
  "pollGeneration",
  "terminalRecheckJob",
  "llamaJobMarker",
  "llamaStatusRequestIsStale",
  "latestAppliedStatusRequest",
  "setStatus",
  "llamaUpdatePresentation",
  "setApplying",
  "setVisible",
  "clearPollTimer",
  "refreshHardwareInfo",
  "notifyReloadIfNeeded",
  "onDone",
  "surfaceIfAvailableRef",
  body,
);

type JobState = "running" | "success" | "error" | "idle";

function status(state: JobState, updateAvailable: boolean, jobId = "job-1") {
  return {
    update_available: updateAvailable,
    job: {
      job_id: jobId,
      state,
      operation: "update" as const,
      started_at: "2026-08-18T13:02:21Z",
      to_tag: "b10472-mix-4b653db",
      reload_required: false,
      error: state === "error" ? "failed" : null,
    },
  };
}

type Snapshot = {
  requestId: number;
  status: ReturnType<typeof status> | null;
};

function harness(responses: Snapshot[], latestApplied = 0) {
  let responseIndex = 0;
  const state = {
    applying: true,
    visible: true,
    timerActive: true,
    adopted: [] as Snapshot[],
    done: [] as unknown[],
    hardwareRefreshes: 0,
    reloadNotifications: 0,
  };
  const pollInFlightGeneration = { current: null as number | null };
  const pollGeneration = { current: 1 };
  const terminalRecheckJob = { current: null as string | null };
  const latestAppliedStatusRequest = { current: latestApplied };

  const adopt = (next: Snapshot) => {
    if (
      !next.status ||
      llamaStatusRequestIsStale(
        latestAppliedStatusRequest.current,
        next.requestId,
      )
    ) {
      return;
    }
    latestAppliedStatusRequest.current = next.requestId;
    state.adopted.push(next);
    const presentation = llamaUpdatePresentation(
      next.status.update_available,
      next.status.job,
    );
    state.applying = presentation.applying;
    state.visible = presentation.visible;
  };

  const args = [
    pollInFlightGeneration,
    1,
    () => {
      assert.ok(responseIndex < responses.length, "status queue exhausted");
      const next = responses[responseIndex++];
      return Promise.resolve({
        requestId: next.requestId,
        job: next.status?.job,
      });
    },
    () => {
      assert.ok(responseIndex < responses.length, "status queue exhausted");
      return Promise.resolve(responses[responseIndex++]);
    },
    pollGeneration,
    terminalRecheckJob,
    (job: ReturnType<typeof status>["job"]) => job.job_id,
    llamaStatusRequestIsStale,
    latestAppliedStatusRequest,
    () => undefined,
    llamaUpdatePresentation,
    (next: boolean) => {
      state.applying = next;
    },
    (next: boolean) => {
      state.visible = next;
    },
    () => {
      state.timerActive = false;
    },
    () => {
      state.hardwareRefreshes += 1;
    },
    () => {
      state.reloadNotifications += 1;
    },
    (next: unknown) => state.done.push(next),
    { current: adopt },
  ];

  return {
    tick: async () => {
      await runTick(...args);
      await new Promise((resolve) => setImmediate(resolve));
    },
    snapshot: () => ({
      applying: state.applying,
      visible: state.visible,
      timerActive: state.timerActive,
      requests: responseIndex,
      latestApplied: latestAppliedStatusRequest.current,
      done: state.done,
    }),
    effectCounts: () => ({
      hardwareRefreshes: state.hardwareRefreshes,
      reloadNotifications: state.reloadNotifications,
    }),
  };
}

test("a terminal poll rechecks and clears stale pre-install availability", async () => {
  const poll = harness([
    { requestId: 1, status: status("success", true) },
    { requestId: 2, status: status("success", false) },
  ]);
  await poll.tick();
  assert.deepEqual(poll.snapshot(), {
    applying: false,
    visible: false,
    timerActive: false,
    requests: 2,
    latestApplied: 2,
    done: [
      {
        ok: true,
        tag: "b10472-mix-4b653db",
        reloadRequired: false,
      },
    ],
  });
});

test("a failed terminal reconciliation keeps polling until a valid status arrives", async () => {
  const poll = harness([
    { requestId: 1, status: status("success", true) },
    { requestId: 2, status: null },
    { requestId: 3, status: status("success", true) },
    { requestId: 4, status: status("success", false) },
  ]);
  await poll.tick();
  assert.deepEqual(poll.snapshot(), {
    applying: false,
    visible: true,
    timerActive: true,
    requests: 2,
    latestApplied: 1,
    done: [
      {
        ok: true,
        tag: "b10472-mix-4b653db",
        reloadRequired: false,
      },
    ],
  });

  await poll.tick();
  assert.deepEqual(poll.snapshot(), {
    applying: false,
    visible: false,
    timerActive: false,
    requests: 4,
    latestApplied: 4,
    done: [
      {
        ok: true,
        tag: "b10472-mix-4b653db",
        reloadRequired: false,
      },
    ],
  });
});

test("a distinct terminal job runs its completion effects during reconciliation", async () => {
  const poll = harness([
    { requestId: 1, status: status("success", true, "job-1") },
    { requestId: 2, status: null },
    { requestId: 3, status: status("success", true, "job-2") },
    { requestId: 4, status: status("success", false, "job-2") },
  ]);
  await poll.tick();
  assert.deepEqual(poll.effectCounts(), {
    hardwareRefreshes: 1,
    reloadNotifications: 1,
  });

  await poll.tick();
  assert.deepEqual(poll.snapshot(), {
    applying: false,
    visible: false,
    timerActive: false,
    requests: 4,
    latestApplied: 4,
    done: [
      {
        ok: true,
        tag: "b10472-mix-4b653db",
        reloadRequired: false,
      },
      {
        ok: true,
        tag: "b10472-mix-4b653db",
        reloadRequired: false,
      },
    ],
  });
  assert.deepEqual(poll.effectCounts(), {
    hardwareRefreshes: 2,
    reloadNotifications: 2,
  });
});

test("an older response cannot overwrite a newer adopted status", async () => {
  const poll = harness(
    [
      { requestId: 1, status: status("running", true) },
      { requestId: 3, status: status("success", false) },
      { requestId: 4, status: status("success", false) },
    ],
    2,
  );
  await poll.tick();
  assert.deepEqual(poll.snapshot(), {
    applying: true,
    visible: true,
    timerActive: true,
    requests: 1,
    latestApplied: 2,
    done: [],
  });

  await poll.tick();
  assert.deepEqual(poll.snapshot(), {
    applying: false,
    visible: false,
    timerActive: false,
    requests: 3,
    latestApplied: 4,
    done: [
      {
        ok: true,
        tag: "b10472-mix-4b653db",
        reloadRequired: false,
      },
    ],
  });
});
