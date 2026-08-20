// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Execute the job interval from the hook with deterministic status responses.
// Importing the hook pulls in the React shell and auth runtime, so this follows
// the same source-lifting pattern as verdict-poll-stall-guard.test.ts.

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

const AsyncFunction = Object.getPrototypeOf(async () => undefined)
  .constructor as new (
  ...args: string[]
) => (...args: unknown[]) => Promise<void>;
const runTick = new AsyncFunction(
  "pollInFlightGeneration",
  "generation",
  "requestStatus",
  "pollGeneration",
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

function status(state: JobState, updateAvailable: boolean) {
  return {
    update_available: updateAvailable,
    job: {
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
  };
  const pollInFlightGeneration = { current: null as number | null };
  const pollGeneration = { current: 1 };
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
      return Promise.resolve(responses[responseIndex++]);
    },
    pollGeneration,
    llamaStatusRequestIsStale,
    latestAppliedStatusRequest,
    (next: ReturnType<typeof status>) => {
      state.adopted.push({
        requestId: latestAppliedStatusRequest.current,
        status: next,
      });
    },
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
    () => undefined,
    () => undefined,
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
