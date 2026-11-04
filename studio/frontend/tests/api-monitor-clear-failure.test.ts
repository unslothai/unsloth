// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// use-api-monitor.ts imports React, so the Clear log sequence is tested via its plain module.
import {
  CLEAR_MONITOR_FAILED,
  type ClearMonitorDeps,
  clearMonitor,
} from "../src/features/api-monitor/clear-monitor.ts";

type Trace = {
  deps: ClearMonitorDeps;
  errors: string[];
  detailsReset: number;
  reloads: number;
};

function trace(remote: () => Promise<void>): Trace {
  const state: Trace = {
    errors: [],
    detailsReset: 0,
    reloads: 0,
    deps: {
      clearRemote: remote,
      resetDetails: () => {
        state.detailsReset += 1;
      },
      reload: () => {
        state.reloads += 1;
        return Promise.resolve();
      },
      onError: (message) => {
        state.errors.push(message);
      },
    },
  };
  return state;
}

test("a refused clear reports through the monitor's error state", async () => {
  const state = trace(() => Promise.reject(new Error("Monitor is read-only")));

  // The click handler discards the promise, so a rejection would be unhandled and silent.
  await assert.doesNotReject(() => clearMonitor(state.deps));

  assert.deepEqual(state.errors, ["Monitor is read-only"]);
  assert.equal(state.detailsReset, 0);
  assert.equal(state.reloads, 0);
});

test("a rejection that is not an Error still reports", async () => {
  const state = trace(() => Promise.reject("offline"));

  await assert.doesNotReject(() => clearMonitor(state.deps));

  assert.deepEqual(state.errors, [CLEAR_MONITOR_FAILED]);
});

test("a clear that lands drops the details and refetches", async () => {
  const state = trace(() => Promise.resolve());

  await clearMonitor(state.deps);

  assert.deepEqual(state.errors, []);
  assert.equal(state.detailsReset, 1);
  assert.equal(state.reloads, 1);
});
