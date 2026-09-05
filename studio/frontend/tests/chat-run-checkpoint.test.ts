// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  RUN_CHECKPOINT_INTERVAL_MS,
  type RunCheckpointTimers,
  createRunCheckpointScheduler,
} from "../src/features/chat/utils/run-checkpoint-scheduler.ts";

const INTERVAL = 1000;

test("uses an eight-second production checkpoint interval", () => {
  assert.equal(RUN_CHECKPOINT_INTERVAL_MS, 8_000);
});

/** The scheduler reschedules from a promise continuation, so let those run. */
async function flushMicrotasks(): Promise<void> {
  for (let i = 0; i < 8; i += 1) {
    await Promise.resolve();
  }
}

function createFakeTimers() {
  let now = 0;
  let nextHandle = 1;
  const scheduled = new Map<number, { at: number; callback: () => void }>();

  const timers: RunCheckpointTimers = {
    setTimeout: (callback, ms) => {
      const handle = nextHandle;
      nextHandle += 1;
      scheduled.set(handle, { at: now + ms, callback });
      return handle;
    },
    clearTimeout: (handle) => {
      scheduled.delete(handle);
    },
  };

  return {
    timers,
    pending: () => scheduled.size,
    async advance(ms: number): Promise<void> {
      now += ms;
      const due = [...scheduled.entries()]
        .filter(([, timer]) => timer.at <= now)
        .sort(([, a], [, b]) => a.at - b.at);
      for (const [handle, timer] of due) {
        if (!scheduled.delete(handle)) continue;
        timer.callback();
        await flushMicrotasks();
      }
    },
  };
}

test("keeps checkpointing a running thread until it is stopped", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  assert.deepEqual(saved, [], "nothing is saved before the first interval");

  await clock.advance(INTERVAL);
  assert.deepEqual(saved, ["thread-a"]);

  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  assert.deepEqual(saved, ["thread-a", "thread-a", "thread-a"]);

  scheduler.stop("thread-a");
  await clock.advance(INTERVAL * 5);
  assert.deepEqual(
    saved,
    ["thread-a", "thread-a", "thread-a"],
    "runEnd stops the schedule",
  );
  assert.equal(clock.pending(), 0, "the pending timer is dropped on stop");
});

test("waits for a slow checkpoint instead of stacking the next one behind it", async () => {
  const clock = createFakeTimers();
  let releaseSave = (): void => {};
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    () => {
      saves += 1;
      return new Promise<void>((resolve) => {
        releaseSave = resolve;
      });
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1);

  // No timer is armed while the save is in flight.
  await clock.advance(INTERVAL * 10);
  assert.equal(saves, 1);
  assert.equal(clock.pending(), 0);

  releaseSave();
  await flushMicrotasks();
  assert.equal(
    clock.pending(),
    1,
    "the next checkpoint is armed once the save lands",
  );

  await clock.advance(INTERVAL);
  assert.equal(saves, 2);
});

test("keeps checkpointing after a checkpoint fails to write", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      attempts += 1;
      throw new Error("write failed");
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  assert.equal(
    attempts,
    2,
    "one transient write error must not leave the rest of the run unprotected",
  );
});

test("a stop during an in-flight checkpoint ends the schedule", async () => {
  const clock = createFakeTimers();
  let releaseSave = (): void => {};
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    () => {
      saves += 1;
      return new Promise<void>((resolve) => {
        releaseSave = resolve;
      });
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1);

  scheduler.stop("thread-a");
  releaseSave();
  await flushMicrotasks();
  assert.equal(
    clock.pending(),
    0,
    "the settled save must not rearm after stop",
  );

  await clock.advance(INTERVAL * 5);
  assert.equal(saves, 1);
});

test("starting a thread twice keeps one schedule, and stopAll ends every thread", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-a");
  scheduler.start("thread-b");
  assert.equal(clock.pending(), 2, "one timer per thread, not per start");

  await clock.advance(INTERVAL);
  assert.deepEqual(saved.sort(), ["thread-a", "thread-b"]);

  scheduler.stopAll();
  assert.equal(clock.pending(), 0);
  await clock.advance(INTERVAL * 5);
  assert.equal(saved.length, 2);
});

test("the chat autosave drives the scheduler from runStart to runEnd", async () => {
  const src = await readFile(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /createRunCheckpointScheduler\(/,
    "ThreadBackendAutosave no longer builds a checkpoint scheduler",
  );
  const runStart = src.indexOf('useAuiEvent("thread.runStart"');
  const runEnd = src.indexOf('useAuiEvent("thread.runEnd"');
  assert.ok(runStart > 0 && runEnd > 0, "the autosave run events moved");
  assert.match(
    src.slice(runStart, runStart + 200),
    /checkpoints\(\)\.start\(threadId\)/,
    "runStart no longer starts checkpointing",
  );
  assert.match(
    src.slice(runEnd, runEnd + 200),
    /checkpoints\(\)\.stop\(threadId\)/,
    "runEnd no longer stops checkpointing, so checkpoints outlive the run",
  );
});
