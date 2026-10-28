// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  STREAM_COOLDOWN_MS,
  createStreamActivity,
} from "../src/components/assistant-ui/stream-activity.ts";

type Running = { runningByThreadId: Record<string, boolean> };

const stubStore = () => {
  let state: Running = { runningByThreadId: {} };
  const listeners = new Set<(next: Running, previous: Running) => void>();
  return {
    getState: () => state,
    subscribe: (listener: (next: Running, previous: Running) => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    set: (running: Record<string, boolean>) => {
      const previous = state;
      state = { runningByThreadId: running };
      for (const listener of listeners) listener(state, previous);
    },
  };
};

test("a stream is active while any thread runs and for the cooldown after the last run ends", () => {
  assert.equal(STREAM_COOLDOWN_MS, 2_000);
  let now = 10_000;
  const store = stubStore();
  const activity = createStreamActivity(store, () => now);
  assert.equal(activity.active(), false);
  store.set({ a: true });
  assert.equal(activity.active(), true);
  store.set({ a: true, b: false });
  assert.equal(activity.active(), true);
  store.set({ a: false, b: false });
  assert.equal(activity.active(), true);
  now += STREAM_COOLDOWN_MS - 1;
  assert.equal(activity.active(), true);
  now += 1;
  assert.equal(activity.active(), false);
});

test("a run that never started leaves no cooldown behind", () => {
  let now = 0;
  const store = stubStore();
  const activity = createStreamActivity(store, () => now);
  store.set({ a: false });
  now = 1;
  assert.equal(activity.active(), false);
});
