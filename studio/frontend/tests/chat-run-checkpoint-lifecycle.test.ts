// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  RUN_CHECKPOINT_INTERVAL_MS,
  RUN_CHECKPOINT_MAX_DURATION_MS,
  type RunCheckpointTimers,
  createRunCheckpointScheduler,
} from "../src/features/chat/utils/run-checkpoint-scheduler.ts";

const INTERVAL = 1000;

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
    // The staleness bound reads a clock, and this harness already keeps one. Without
    // this the bound would be measured against the real wall clock, so a test that
    // advances an hour of fake time would take none of it and never reach the cap.
    now: () => now,
  };

  const fireDue = async (): Promise<number> => {
    const due = [...scheduled.entries()]
      .filter(([, timer]) => timer.at <= now)
      .sort(([, a], [, b]) => a.at - b.at);
    let fired = 0;
    for (const [handle, timer] of due) {
      if (!scheduled.delete(handle)) continue;
      timer.callback();
      fired += 1;
      await flushMicrotasks();
    }
    return fired;
  };

  return {
    timers,
    pending: () => scheduled.size,
    /** Due timers are snapshotted once, exactly like the harness the PR ships. */
    async advance(ms: number): Promise<void> {
      now += ms;
      await fireDue();
    },
    /**
     * Advance and then keep firing anything that came due during the pass, so a timer
     * armed mid-advance still runs. Used where a flush or a settle rearms in the middle.
     */
    async advanceUntilQuiet(ms: number): Promise<void> {
      now += ms;
      for (let i = 0; i < 50; i += 1) {
        if ((await fireDue()) === 0) return;
      }
      throw new Error("timers never went quiet");
    },
  };
}

/** A save whose settling the test controls. */
function createGatedSave() {
  const releases: Array<() => void> = [];
  const calls: string[] = [];
  return {
    calls,
    save: (threadId: string) => {
      calls.push(threadId);
      return new Promise<void>((resolve) => {
        releases.push(resolve);
      });
    },
    releaseAll(): void {
      const pending = releases.splice(0, releases.length);
      for (const release of pending) release();
    },
  };
}

// A. backwards compatibility of the new options

// This file used to assert that omitting isActive checkpoints INDEFINITELY, and called
// that intentional. It is no longer true, deliberately. A run that never reaches a
// terminal status never fires runEnd, and `isActive` reports the runtime's own
// `isRunning`, which that same stuck run holds true, so the two agreed forever and the
// schedule outlived the page: a real user log showed 160 four-request cycles at 8-9s
// against one thread, unbroken by two full app reloads. An absent or always-true liveness
// probe still must not END a live run, which is what the first twelve intervals below
// pin; what it may no longer do is run without any bound at all.

test("omitting isActive keeps checkpointing until the staleness bound, not forever", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      maxDurationMs: 20 * INTERVAL,
    },
  );

  scheduler.start("thread-a");
  for (let i = 0; i < 12; i += 1) {
    await clock.advance(INTERVAL);
  }
  assert.equal(
    saved.length,
    12,
    "an absent liveness probe must not end the run early",
  );
  assert.equal(clock.pending(), 1, "the schedule is still armed");

  for (let i = 0; i < 12; i += 1) {
    await clock.advance(INTERVAL);
  }
  assert.equal(
    saved.length,
    20,
    "the cap takes a final checkpoint on the way out and then writes no more",
  );
  assert.equal(clock.pending(), 0, "the schedule must not rearm past the cap");
  scheduler.stop("thread-a");
});

test("an isActive that always returns true behaves like no isActive at all", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: () => true,
      maxDurationMs: 20 * INTERVAL,
    },
  );

  scheduler.start("thread-a");
  for (let i = 0; i < 12; i += 1) {
    await clock.advance(INTERVAL);
  }
  assert.equal(saved.length, 12);
  assert.equal(clock.pending(), 1);
  scheduler.stop("thread-a");
});

test("the staleness bound is generous enough for a long legitimate run", () => {
  // Thirty minutes, held at the follow deadline. Tripping it costs only the periodic
  // partial saves, never the run's own writes, so the bound is set to outlast any answer
  // a user waits through, including a prefill the backend still allows 1200s for.
  assert.equal(RUN_CHECKPOINT_MAX_DURATION_MS, 30 * 60_000);
  assert.ok(
    RUN_CHECKPOINT_MAX_DURATION_MS / RUN_CHECKPOINT_INTERVAL_MS >= 100,
    "the cap must leave room for a hundred checkpoints before it fires",
  );
});

test("a thread restarted after the bound gets a fresh window", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers, maxDurationMs: 3 * INTERVAL },
  );

  scheduler.start("thread-a");
  for (let i = 0; i < 6; i += 1) {
    await clock.advance(INTERVAL);
  }
  const afterFirstRun = saves;
  assert.equal(clock.pending(), 0, "the first window closed");

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, afterFirstRun + 1, "the next run checkpoints again");
  scheduler.stop("thread-a");
});

test("an empty options object still uses the production interval", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(RUN_CHECKPOINT_INTERVAL_MS - 1);
  assert.equal(saves, 0, "a default-interval checkpoint fired early");
  await clock.advance(1);
  assert.equal(saves, 1);
  scheduler.stop("thread-a");
});

test("the original no-options call shape arms a window timer at the production interval", () => {
  const armed: Array<{ ms: number; callback: () => void }> = [];
  const cleared: number[] = [];
  const globals = globalThis as unknown as { window?: unknown };
  const original = globals.window;
  globals.window = {
    setTimeout: (callback: () => void, ms: number) => {
      armed.push({ ms, callback });
      return armed.length;
    },
    clearTimeout: (handle: number) => {
      cleared.push(handle);
    },
  };
  try {
    const scheduler = createRunCheckpointScheduler(async () => {});
    scheduler.start("thread-a");
    assert.equal(
      armed.length,
      1,
      "createRunCheckpointScheduler(save) must still work",
    );
    assert.equal(armed[0]?.ms, RUN_CHECKPOINT_INTERVAL_MS);
    scheduler.stop("thread-a");
    assert.deepEqual(cleared, [1], "stop must clear the window timer");
  } finally {
    globals.window = original;
  }
});

// B. the liveness guard

test("a thread that is already inactive at the first tick takes exactly one final save", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers, isActive: () => false },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.deepEqual(
    saved,
    ["thread-a"],
    "the lost runEnd must still get its final save",
  );
  assert.equal(clock.pending(), 0, "an inactive thread must not stay armed");

  await clock.advance(INTERVAL * 20);
  assert.deepEqual(saved, ["thread-a"], "the schedule must be over");
});

test("checkpoints continue while active and end with one final save when the run goes inactive", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  let active = true;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers, isActive: () => active },
  );

  scheduler.start("thread-a");
  for (let i = 0; i < 4; i += 1) {
    await clock.advance(INTERVAL);
  }
  assert.equal(saves, 4, "four periodic checkpoints while the run is live");

  active = false;
  await clock.advance(INTERVAL);
  assert.equal(saves, 5, "four periodic saves plus one final save");
  assert.equal(clock.pending(), 0);

  await clock.advance(INTERVAL * 10);
  assert.equal(saves, 5, "nothing may be scheduled after the final save");
});

test("an isActive that throws is treated as not running", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: () => {
        throw new Error("thread record is gone");
      },
    },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1, "a throwing probe must still yield the final save");
  assert.equal(clock.pending(), 0, "a throwing probe must end the schedule");
  await clock.advance(INTERVAL * 10);
  assert.equal(saves, 1);
});

test("a thread can be started again after it self-terminated", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  let active = false;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers, isActive: () => active },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1);
  assert.equal(clock.pending(), 0);

  // The dead-record hazard: a stale Map entry would make this start a silent no-op.
  active = true;
  scheduler.start("thread-a");
  assert.equal(
    clock.pending(),
    1,
    "the self-terminated thread left a stranded map entry",
  );
  await clock.advance(INTERVAL);
  assert.equal(saves, 2, "the revived thread must checkpoint again");
  scheduler.stop("thread-a");
});

test("isActive is not consulted before the first interval elapses", async () => {
  const clock = createFakeTimers();
  let probes = 0;
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: () => {
        probes += 1;
        return true;
      },
    },
  );

  scheduler.start("thread-a");
  assert.equal(probes, 0, "start must not probe liveness");
  assert.equal(saves, 0);
  await clock.advance(INTERVAL - 1);
  assert.equal(probes, 0, "no probe before the interval elapses");
  assert.equal(saves, 0);
  await clock.advance(1);
  assert.equal(probes, 1);
  assert.equal(saves, 1);
  scheduler.stop("thread-a");
});

test("liveness is probed once per tick", async () => {
  const clock = createFakeTimers();
  let probes = 0;
  const scheduler = createRunCheckpointScheduler(async () => {}, {
    intervalMs: INTERVAL,
    timers: clock.timers,
    isActive: () => {
      probes += 1;
      return true;
    },
  });

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(probes, 1);
  await clock.advance(INTERVAL);
  assert.equal(probes, 2, "one probe per checkpoint, not per continuation");
  scheduler.stop("thread-a");
});

test("the final save is still attempted when the save itself rejects", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      attempts += 1;
      throw new Error("write failed");
    },
    { intervalMs: INTERVAL, timers: clock.timers, isActive: () => false },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(
    attempts,
    1,
    "the final save must be attempted even if it will fail",
  );
  assert.equal(
    clock.pending(),
    0,
    "a failed final save must not resurrect the schedule",
  );
  await clock.advance(INTERVAL * 10);
  assert.equal(attempts, 1);
});

test("a stop arriving while the final save is in flight does not rearm", async () => {
  const clock = createFakeTimers();
  const gated = createGatedSave();
  const scheduler = createRunCheckpointScheduler(gated.save, {
    intervalMs: INTERVAL,
    timers: clock.timers,
    isActive: () => false,
  });

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.deepEqual(gated.calls, ["thread-a"], "the final save is in flight");

  scheduler.stop("thread-a");
  gated.releaseAll();
  await flushMicrotasks();
  assert.equal(clock.pending(), 0, "the settled final save must not rearm");
  await clock.advance(INTERVAL * 10);
  assert.deepEqual(gated.calls, ["thread-a"]);
});

test("the final save is given the thread id that went inactive", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: (threadId) => threadId !== "thread-b",
    },
  );

  scheduler.start("thread-b");
  await clock.advance(INTERVAL);
  assert.deepEqual(saved, ["thread-b"]);
  scheduler.stopAll();
});

test("one thread going inactive does not stop its sibling", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: (threadId) => threadId === "thread-a",
    },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  await clock.advance(INTERVAL);
  assert.deepEqual([...saved].sort(), ["thread-a", "thread-b"]);
  assert.equal(clock.pending(), 1, "only the live thread stays armed");

  await clock.advance(INTERVAL);
  assert.deepEqual(
    saved.slice(2),
    ["thread-a"],
    "the dead thread must not save again",
  );
  scheduler.stopAll();
});

test("the liveness probe receives the thread id under checkpoint", async () => {
  const clock = createFakeTimers();
  const probed: string[] = [];
  const scheduler = createRunCheckpointScheduler(async () => {}, {
    intervalMs: INTERVAL,
    timers: clock.timers,
    isActive: (threadId) => {
      probed.push(threadId);
      return true;
    },
  });

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  await clock.advance(INTERVAL);
  assert.deepEqual([...probed].sort(), ["thread-a", "thread-b"]);
  scheduler.stopAll();
});

// C. sync-throw and non-thenable hardening

test("a save that throws synchronously does not stop the schedule", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    (() => {
      attempts += 1;
      throw new Error("serialiser blew up");
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(attempts, 1);
  assert.equal(clock.pending(), 1, "a synchronous throw must still rearm");
  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  assert.equal(
    attempts,
    3,
    "a synchronous throw must not disable later checkpoints",
  );
  scheduler.stop("thread-a");
});

test("a synchronous throw does not escape the timer callback", async () => {
  const clock = createFakeTimers();
  const scheduler = createRunCheckpointScheduler(
    (() => {
      throw new Error("boom");
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  // An escaping throw would reject this advance and fail the test.
  await clock.advance(INTERVAL);
  assert.equal(clock.pending(), 1);
  scheduler.stop("thread-a");
});

test("a save returning undefined reschedules like a resolved promise", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    (() => {
      attempts += 1;
      return undefined;
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  assert.equal(attempts, 3, "a void return must not strand the schedule");
  scheduler.stop("thread-a");
});

test("a save returning a plain non-thenable object reschedules", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    (() => {
      attempts += 1;
      return { ok: true };
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  assert.equal(attempts, 2);
  assert.equal(clock.pending(), 1);
  scheduler.stop("thread-a");
});

test("a save returning a rejected promise reschedules", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    () => {
      attempts += 1;
      return Promise.reject(new Error("write failed"));
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  await clock.advance(INTERVAL);
  assert.equal(attempts, 2);
  assert.equal(clock.pending(), 1);
  scheduler.stop("thread-a");
});

test("a synchronous throw leaves no stranded map entry", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    (() => {
      attempts += 1;
      throw new Error("boom");
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(attempts, 1);

  scheduler.stop("thread-a");
  assert.equal(
    clock.pending(),
    0,
    "stop after a synchronous throw must disarm",
  );
  scheduler.start("thread-a");
  assert.equal(clock.pending(), 1, "a thrown save stranded the map entry");
  await clock.advance(INTERVAL);
  assert.equal(attempts, 2);
  scheduler.stop("thread-a");
});

test("a synchronously throwing save can still be stopped mid-schedule", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    (() => {
      attempts += 1;
      throw new Error("boom");
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  scheduler.stop("thread-a");
  await clock.advance(INTERVAL * 10);
  assert.equal(attempts, 1, "stop must beat the retry loop");
  assert.equal(clock.pending(), 0);
});

test("a throwing probe and a throwing save still terminate cleanly", async () => {
  const clock = createFakeTimers();
  let attempts = 0;
  const scheduler = createRunCheckpointScheduler(
    (() => {
      attempts += 1;
      throw new Error("save boom");
    }) as unknown as (threadId: string) => Promise<unknown>,
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: () => {
        throw new Error("probe boom");
      },
    },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(attempts, 1, "the final save is attempted even when both throw");
  assert.equal(clock.pending(), 0);
  await clock.advance(INTERVAL * 10);
  assert.equal(attempts, 1);
});

// D. flushAll

test("flushAll checkpoints every started thread", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  scheduler.flushAll();
  await flushMicrotasks();
  assert.deepEqual([...saved].sort(), ["thread-a", "thread-b"]);
  scheduler.stopAll();
});

test("flushAll does not checkpoint a stopped thread", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  scheduler.stop("thread-b");
  scheduler.flushAll();
  await flushMicrotasks();
  assert.deepEqual(saved, ["thread-a"], "a stopped thread must not be flushed");
  scheduler.stopAll();
});

test("flushAll leaves the pending timer armed and on its original schedule", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1);
  const armedBefore = clock.pending();
  assert.equal(armedBefore, 1);

  scheduler.flushAll();
  await flushMicrotasks();
  assert.equal(saves, 2, "the flush is an extra checkpoint");
  assert.equal(
    clock.pending(),
    armedBefore,
    "a flush must not clear or stack timers",
  );

  await clock.advance(INTERVAL - 1);
  assert.equal(
    saves,
    2,
    "the flush must not have pulled the next checkpoint forward",
  );
  await clock.advance(1);
  assert.equal(
    saves,
    3,
    "the next checkpoint must still land on its original deadline",
  );
  scheduler.stop("thread-a");
});

test("flushAll with no started threads is a no-op", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.flushAll();
  await flushMicrotasks();
  assert.equal(saves, 0);
  assert.equal(clock.pending(), 0);
});

test("flushAll during an in-flight checkpoint still issues the extra save", async () => {
  const clock = createFakeTimers();
  const gated = createGatedSave();
  const scheduler = createRunCheckpointScheduler(gated.save, {
    intervalMs: INTERVAL,
    timers: clock.timers,
  });

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(gated.calls.length, 1, "the periodic checkpoint is in flight");
  assert.equal(clock.pending(), 0, "no timer while a save is in flight");

  scheduler.flushAll();
  await flushMicrotasks();
  assert.equal(
    gated.calls.length,
    2,
    "a page-hide flush must not be swallowed",
  );

  gated.releaseAll();
  await flushMicrotasks();
  assert.equal(
    clock.pending(),
    1,
    "the schedule rearms once the periodic save settles",
  );
  scheduler.stop("thread-a");
});

test("flushAll after stopAll saves nothing", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  scheduler.stopAll();
  scheduler.flushAll();
  await flushMicrotasks();
  assert.equal(saves, 0, "unmount must not be followed by a flush write");
});

test("a synchronous throw inside flushAll does not break the scheduler", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    ((threadId: string) => {
      saved.push(threadId);
      if (threadId === "thread-a") throw new Error("boom");
      return Promise.resolve();
    }) as unknown as (threadId: string) => Promise<unknown>,
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  scheduler.flushAll();
  await flushMicrotasks();
  assert.deepEqual(
    [...saved].sort(),
    ["thread-a", "thread-b"],
    "one throwing thread must not abort the rest of the flush",
  );
  assert.equal(clock.pending(), 2, "the flush must leave both schedules armed");

  await clock.advance(INTERVAL);
  assert.equal(
    saved.length,
    4,
    "both threads keep checkpointing after a throwing flush",
  );
  scheduler.stopAll();
});

test("flushAll does not consult isActive", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  let probes = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    {
      intervalMs: INTERVAL,
      timers: clock.timers,
      isActive: () => {
        probes += 1;
        return false;
      },
    },
  );

  scheduler.start("thread-a");
  scheduler.flushAll();
  await flushMicrotasks();
  assert.equal(
    saves,
    1,
    "a page-hide flush must persist regardless of run liveness",
  );
  assert.equal(probes, 0, "flushAll must not probe liveness");
  assert.equal(
    clock.pending(),
    1,
    "the flush must not have ended the schedule",
  );
  scheduler.stopAll();
});

test("repeated flushAll calls each write once per thread", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.flushAll();
  scheduler.flushAll();
  scheduler.flushAll();
  await flushMicrotasks();
  assert.deepEqual(saved, ["thread-a", "thread-a", "thread-a"]);
  assert.equal(clock.pending(), 1);
  scheduler.stop("thread-a");
});

test("flushAll ignores a thread that already self-terminated", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers, isActive: () => false },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1, "the final save");

  scheduler.flushAll();
  await flushMicrotasks();
  assert.equal(saves, 1, "a thread already released must not be flushed again");
});

test("flushAll writes a thread that was restarted after stopAll", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.stopAll();
  scheduler.start("thread-a");
  scheduler.flushAll();
  await flushMicrotasks();
  assert.deepEqual(saved, ["thread-a"]);
  scheduler.stopAll();
});

// E. pre-existing behaviour that must not regress

test("the checkpoint interval constant is still eight seconds", () => {
  assert.equal(RUN_CHECKPOINT_INTERVAL_MS, 8_000);
});

test("quiet time is measured after the checkpoint settles, not on a fixed cadence", async () => {
  const clock = createFakeTimers();
  const gated = createGatedSave();
  const scheduler = createRunCheckpointScheduler(gated.save, {
    intervalMs: INTERVAL,
    timers: clock.timers,
  });

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(gated.calls.length, 1, "checkpoint 1 at t = interval");

  // The save takes five intervals to land.
  await clock.advance(INTERVAL * 5);
  assert.equal(gated.calls.length, 1);
  gated.releaseAll();
  await flushMicrotasks();

  // Checkpoint 2 must land at save_duration + interval, not at a fixed 2 x interval.
  await clock.advance(INTERVAL - 1);
  assert.equal(
    gated.calls.length,
    1,
    "the next checkpoint ignored the settle time",
  );
  await clock.advance(1);
  assert.equal(
    gated.calls.length,
    2,
    "checkpoint 2 lands one quiet interval after settle",
  );
  scheduler.stop("thread-a");
});

test("no timer is armed while a checkpoint is in flight", async () => {
  const clock = createFakeTimers();
  const gated = createGatedSave();
  const scheduler = createRunCheckpointScheduler(gated.save, {
    intervalMs: INTERVAL,
    timers: clock.timers,
  });

  scheduler.start("thread-a");
  assert.equal(clock.pending(), 1);
  await clock.advance(INTERVAL);
  assert.equal(
    clock.pending(),
    0,
    "checkpoints must not stack behind a slow save",
  );
  await clock.advance(INTERVAL * 10);
  assert.equal(gated.calls.length, 1);
  gated.releaseAll();
  await flushMicrotasks();
  assert.equal(clock.pending(), 1);
  scheduler.stop("thread-a");
});

test("a duplicate start does not stack timers", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-a");
  scheduler.start("thread-a");
  assert.equal(clock.pending(), 1, "one timer per thread, not per start");
  await clock.advanceUntilQuiet(INTERVAL);
  assert.equal(
    saves,
    1,
    "a repeated runStart must not double the checkpoint rate",
  );
  scheduler.stop("thread-a");
});

test("stopping an unknown thread is a no-op and does not disturb a live thread", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.stop("thread-never-started");
  assert.equal(clock.pending(), 1, "an unknown stop cleared a live timer");
  await clock.advance(INTERVAL);
  assert.deepEqual(saved, ["thread-a"]);
  scheduler.stop("thread-a");
});

test("stopping a thread twice is a no-op the second time", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.stop("thread-a");
  scheduler.stop("thread-a");
  assert.equal(clock.pending(), 0);
  await clock.advance(INTERVAL * 5);
  assert.equal(saves, 0, "a double stop must not resurrect anything");
});

test("start after stop restarts the schedule cleanly", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(saves, 1);
  scheduler.stop("thread-a");
  await clock.advance(INTERVAL * 3);
  assert.equal(saves, 1);

  scheduler.start("thread-a");
  assert.equal(clock.pending(), 1);
  await clock.advance(INTERVAL);
  assert.equal(saves, 2, "a restarted thread must checkpoint again");
  scheduler.stop("thread-a");
});

test("stopAll is idempotent and threads can restart after it", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  scheduler.stopAll();
  scheduler.stopAll();
  assert.equal(clock.pending(), 0);
  await clock.advance(INTERVAL * 3);
  assert.deepEqual(saved, []);

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.deepEqual(saved, ["thread-a"], "unmount must not poison later runs");
  scheduler.stopAll();
});

test("threads are checkpointed independently", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  scheduler.start("thread-b");
  await clock.advance(INTERVAL);
  assert.deepEqual([...saved].sort(), ["thread-a", "thread-b"]);

  scheduler.stop("thread-a");
  await clock.advance(INTERVAL);
  assert.deepEqual(
    saved.slice(2),
    ["thread-b"],
    "stopping one thread stopped the other",
  );
  scheduler.stop("thread-b");
});

test("a run shorter than one interval produces no checkpoints", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(INTERVAL - 1);
  scheduler.stop("thread-a");
  await clock.advance(INTERVAL * 10);
  assert.equal(saves, 0, "a short run must not write a checkpoint at all");
  assert.equal(clock.pending(), 0);
});

test("each thread's save receives its own thread id", async () => {
  const clock = createFakeTimers();
  const saved: string[] = [];
  const scheduler = createRunCheckpointScheduler(
    async (threadId) => {
      saved.push(threadId);
    },
    { intervalMs: INTERVAL, timers: clock.timers },
  );

  scheduler.start("alpha");
  scheduler.start("beta");
  scheduler.start("gamma");
  await clock.advance(INTERVAL);
  assert.deepEqual([...saved].sort(), ["alpha", "beta", "gamma"]);
  scheduler.stopAll();
});

test("a custom interval is honoured for every thread", async () => {
  const clock = createFakeTimers();
  let saves = 0;
  const scheduler = createRunCheckpointScheduler(
    async () => {
      saves += 1;
    },
    { intervalMs: 250, timers: clock.timers },
  );

  scheduler.start("thread-a");
  await clock.advance(249);
  assert.equal(saves, 0);
  await clock.advance(1);
  assert.equal(saves, 1);
  await clock.advance(250);
  assert.equal(saves, 2);
  scheduler.stop("thread-a");
});

test("stop during an in-flight checkpoint ends the schedule", async () => {
  const clock = createFakeTimers();
  const gated = createGatedSave();
  const scheduler = createRunCheckpointScheduler(gated.save, {
    intervalMs: INTERVAL,
    timers: clock.timers,
  });

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(gated.calls.length, 1);

  scheduler.stop("thread-a");
  gated.releaseAll();
  await flushMicrotasks();
  assert.equal(
    clock.pending(),
    0,
    "the settled save must not rearm after stop",
  );
  await clock.advance(INTERVAL * 5);
  assert.equal(gated.calls.length, 1);
});

test("a thread restarted while its old save is in flight keeps exactly one schedule", async () => {
  const clock = createFakeTimers();
  const gated = createGatedSave();
  const scheduler = createRunCheckpointScheduler(gated.save, {
    intervalMs: INTERVAL,
    timers: clock.timers,
  });

  scheduler.start("thread-a");
  await clock.advance(INTERVAL);
  assert.equal(gated.calls.length, 1);

  scheduler.stop("thread-a");
  scheduler.start("thread-a");
  assert.equal(clock.pending(), 1, "the restart arms its own timer");

  // The abandoned save settles; it must not arm a second timer for the new run.
  gated.releaseAll();
  await flushMicrotasks();
  assert.equal(
    clock.pending(),
    1,
    "an abandoned save rearmed on top of the new schedule",
  );
  scheduler.stop("thread-a");
});
