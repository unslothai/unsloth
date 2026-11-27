// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  startVisibilityPoll,
  type VisibilityPollTargets,
} from "../src/features/hub/download-manager/visibility-poll.ts";
import { HIDDEN_POLL_INTERVAL_MS } from "../src/features/hub/download-manager/download-manager-config.ts";

function world() {
  const listeners = new Set<() => void>();
  const timers = new Map<number, { callback: () => void; ms: number }>();
  let next = 1;
  let hidden = false;
  const targets: VisibilityPollTargets = {
    document: {
      get hidden() {
        return hidden;
      },
      addEventListener: (type: string, listener: unknown) => {
        assert.equal(type, "visibilitychange");
        listeners.add(listener as () => void);
      },
      removeEventListener: (_type: string, listener: unknown) => {
        listeners.delete(listener as () => void);
      },
    },
    setTimeout: (callback, ms) => {
      const id = next++;
      timers.set(id, { callback, ms });
      return id;
    },
    clearTimeout: (handle) => {
      timers.delete(handle as number);
    },
  };
  return {
    targets,
    timers,
    listeners,
    setHidden(value: boolean) {
      hidden = value;
      for (const listener of [...listeners]) listener();
    },
    fire() {
      assert.equal(timers.size, 1, "exactly one armed timer");
      const [[id, { callback }]] = [...timers];
      timers.delete(id);
      callback();
    },
    delays: () => [...timers.values()].map((timer) => timer.ms),
  };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

function deferredRuns() {
  const pending: Array<(again: boolean) => void> = [];
  return {
    pending,
    run: () =>
      new Promise<boolean>((resolve) => {
        pending.push(resolve);
      }),
  };
}

const trainingDelay = (hidden: boolean) =>
  hidden ? HIDDEN_POLL_INTERVAL_MS : 1_500;

test("training discovery polls at the hidden pace while hidden and at once when shown", async () => {
  const w = world();
  let runs = 0;
  const poll = startVisibilityPoll(
    async () => {
      runs += 1;
      return true;
    },
    trainingDelay,
    { targets: w.targets },
  );
  poll.restart();
  await settle();
  assert.equal(runs, 1);
  assert.deepEqual(w.delays(), [1_500]);
  w.setHidden(true);
  w.fire();
  await settle();
  assert.equal(runs, 2);
  assert.deepEqual(w.delays(), [HIDDEN_POLL_INTERVAL_MS]);
  // A job that starts while hidden is still adopted on the next hidden tick.
  w.fire();
  await settle();
  assert.equal(runs, 3);
  w.setHidden(false);
  assert.equal(runs, 4, "shown tab polls immediately");
  await settle();
  assert.deepEqual(w.delays(), [1_500]);
  poll.stop();
  assert.equal(w.timers.size, 0);
  assert.equal(w.listeners.size, 0);
});

test("the hidden interval matches the regular download manager", () => {
  assert.equal(HIDDEN_POLL_INTERVAL_MS, 10_000);
  const hook = readFileSync(
    new URL(
      "../src/features/hub/download-manager/use-training-downloads.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /hidden \? HIDDEN_POLL_INTERVAL_MS : VISIBLE_DISCOVERY_INTERVAL_MS/);
  assert.doesNotMatch(hook, /setTimeout\(/);
});

test("showing the tab during an in-flight poll never arms a second timer", async () => {
  const w = world();
  const runs = deferredRuns();
  const poll = startVisibilityPoll(runs.run, trainingDelay, {
    targets: w.targets,
  });
  poll.restart();
  w.setHidden(true);
  w.setHidden(false);
  assert.equal(runs.pending.length, 1, "coalesced into the in-flight run");
  runs.pending[0](true);
  await settle();
  assert.deepEqual(w.delays(), [1_500]);
  poll.stop();
});

test("an auth reset supersedes the in-flight poll so only one chain survives", async () => {
  const w = world();
  const runs = deferredRuns();
  const poll = startVisibilityPoll(runs.run, trainingDelay, {
    targets: w.targets,
  });
  poll.restart();
  poll.restart();
  assert.equal(runs.pending.length, 2);
  runs.pending[0](true);
  await settle();
  assert.equal(w.timers.size, 0, "superseded run does not re-arm");
  runs.pending[1](true);
  await settle();
  assert.equal(w.timers.size, 1);
  poll.stop();
  assert.equal(w.timers.size, 0);
});

test("a poll that reports a stale session ends the chain until the next reset", async () => {
  const w = world();
  let again = false;
  let runs = 0;
  const poll = startVisibilityPoll(
    async () => {
      runs += 1;
      return again;
    },
    trainingDelay,
    { targets: w.targets },
  );
  poll.restart();
  await settle();
  assert.equal(w.timers.size, 0);
  w.setHidden(true);
  w.setHidden(false);
  assert.equal(runs, 1, "visibility does not revive an ended chain");
  again = true;
  poll.restart();
  await settle();
  assert.equal(runs, 2);
  assert.equal(w.timers.size, 1);
  poll.stop();
});

test("disposing during an in-flight poll leaves nothing armed", async () => {
  const w = world();
  const runs = deferredRuns();
  const poll = startVisibilityPoll(runs.run, trainingDelay, {
    targets: w.targets,
  });
  poll.restart();
  poll.stop();
  runs.pending[0](true);
  await settle();
  assert.equal(w.timers.size, 0);
  w.setHidden(false);
  assert.equal(runs.pending.length, 1);
});

test("the background engines poll parks while hidden and refreshes when shown", async () => {
  const { enginePollDelayMs } = await import(
    "../src/features/model-picker/hooks/engine-poll-delay.ts"
  );
  const w = world();
  let runs = 0;
  let installing = false;
  const poll = startVisibilityPoll(
    async () => {
      runs += 1;
      return true;
    },
    (hidden) => enginePollDelayMs(hidden, installing, true),
    { targets: w.targets },
  );
  poll.restart();
  await settle();
  assert.deepEqual(w.delays(), [30_000]);
  w.setHidden(true);
  w.fire();
  await settle();
  assert.equal(runs, 2);
  assert.equal(w.timers.size, 0, "parked while hidden");
  installing = true;
  w.setHidden(false);
  assert.equal(runs, 3);
  await settle();
  assert.deepEqual(w.delays(), [1_000]);
  poll.stop();
});

test("foreground engine pickers keep their cadence, hidden or visible", async () => {
  const { enginePollDelayMs } = await import(
    "../src/features/model-picker/hooks/engine-poll-delay.ts"
  );
  assert.equal(enginePollDelayMs(false, false, false), 3_000);
  assert.equal(enginePollDelayMs(false, true, false), 1_000);
  assert.equal(enginePollDelayMs(true, true, false), 30_000);
  assert.equal(enginePollDelayMs(true, false, false), 30_000);
  assert.equal(enginePollDelayMs(false, false, true), 30_000);
  assert.equal(enginePollDelayMs(false, true, true), 1_000);
  assert.equal(enginePollDelayMs(true, true, true), null);
  const hook = readFileSync(
    new URL("../src/features/model-picker/hooks/use-engines.ts", import.meta.url),
    "utf8",
  );
  assert.match(hook, /\{ wakeOnVisible: background \}/);
});
