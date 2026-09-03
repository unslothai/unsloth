// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The recovery poll's no-stacking guard in app-sidebar.tsx, run rather than pattern-matched.
// The effect lives in a .tsx that pulls in the whole shell, so lift the interval out of the
// source and drive it with a fake clock, a fake setInterval and reads the test settles by hand.
//
// The race: a /api/health read that outlives the stall window is given up on and the next tick
// starts a replacement, which takes the guard over. The abandoned read still settles eventually,
// and its `finally` used to zero the shared marker while the replacement was in flight, so every
// following tick saw a free guard and fired another forced read. On the backend this exists for,
// one still importing torch, that is a read every three seconds piled onto the process the poll
// is waiting for.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const src = await readFile(
  new URL("../src/components/app-sidebar.tsx", import.meta.url),
  "utf8",
);

// The interval, verbatim, from the marker it opens with to the cleanup it returns.
const START = "let pollingSince = 0;";
const END = "return () => window.clearInterval(id);";
const from = src.indexOf(START);
const to = src.indexOf(END, from);
assert.ok(from > 0 && to > from, "the recovery poll's interval moved out of app-sidebar.tsx");
const body = src.slice(from, to + END.length);
assert.ok(
  body.includes("void fetchDeviceType({ force: true })"),
  "the lifted block is not the one that re-reads the verdict",
);

function constant(name: string): number {
  const found = new RegExp(`const ${name} = (\\d+);`).exec(src);
  assert.ok(found, `${name} is no longer declared in app-sidebar.tsx`);
  return Number(found[1]);
}
const STALL_MS = constant("VERDICT_POLL_STALL_MS");
const POLL_MS = constant("VERDICT_UNKNOWN_POLL_MS");

const startPoll = new Function(
  "window",
  "Date",
  "fetchDeviceType",
  "capabilitiesUnknown",
  "VERDICT_POLL_STALL_MS",
  "VERDICT_UNKNOWN_POLL_MS",
  "SELF_HEAL_POLL_MS",
  body,
) as (
  window: { setInterval: (fn: () => void, ms: number) => number; clearInterval: (id: number) => void },
  date: { now: () => number },
  fetchDeviceType: () => Promise<void>,
  capabilitiesUnknown: boolean,
  stallMs: number,
  unknownPollMs: number,
  selfHealPollMs: number,
) => () => void;

/** The interval, wired to a clock and a queue of reads the test settles when it chooses. */
function harness() {
  let clock = 1_700_000_000_000; // any non-zero start: the guard reads its marker as truthy
  let tick: (() => void) | undefined;
  let cadence = 0;
  const pending: Array<{ resolve: () => void; reject: () => void }> = [];
  let cleared = 0;

  const stop = startPoll(
    {
      setInterval: (fn, ms) => {
        tick = fn;
        cadence = ms;
        return 1;
      },
      clearInterval: () => {
        cleared += 1;
      },
    },
    { now: () => clock },
    () =>
      new Promise<void>((resolve, reject) => {
        pending.push({ resolve: () => resolve(), reject: () => reject(new Error("offline")) });
      }),
    true,
    STALL_MS,
    POLL_MS,
    // The self-heal cadence is the other branch; this scenario is the unknown verdict.
    15000,
  );

  assert.ok(tick, "the poll never scheduled an interval");
  return {
    cadence,
    reads: () => pending.length,
    advance: (ms: number) => {
      clock += ms;
    },
    tick: () => tick?.(),
    // Node runs the promise callbacks the interval attached before the next macrotask.
    settle: async (index: number, how: "resolve" | "reject" = "resolve") => {
      pending[index][how]();
      await new Promise((r) => setImmediate(r));
    },
    stop,
    cleared: () => cleared,
  };
}

test("a read that is still outstanding holds the poll off", async () => {
  const poll = harness();
  assert.equal(poll.cadence, POLL_MS, "an unknown verdict polls at the wrong cadence");
  poll.tick();
  assert.equal(poll.reads(), 1);
  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 1, "a second read was stacked on the first");
  poll.stop();
});

test("a read that settles hands the guard back", async () => {
  const poll = harness();
  poll.tick();
  await poll.settle(0);
  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 2, "the guard latched the poll off after a completed read");
  poll.stop();
});

test("a stalled read cannot clear the guard its replacement now owns", async () => {
  const poll = harness();
  poll.tick();
  assert.equal(poll.reads(), 1);

  // The first read outlives the stall window, so the poll gives up on it and replaces it.
  poll.advance(STALL_MS + POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 2, "the stall window did not release the guard");

  // The abandoned read finally answers, long after ownership moved on.
  await poll.settle(0);

  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(
    poll.reads(),
    2,
    "the abandoned read cleared a guard it no longer held, so the poll stacked another " +
      "forced /api/health onto the backend it is waiting for",
  );
  // And it keeps holding: the replacement is still in flight, tick after tick.
  poll.advance(POLL_MS);
  poll.tick();
  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 2, "the guard leaked one tick later instead");

  // The replacement still releases it normally when it answers.
  await poll.settle(1);
  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 3, "the owning read never handed the guard back");
  poll.stop();
});

test("a stalled read that fails cannot clear it either", async () => {
  const poll = harness();
  poll.tick();
  poll.advance(STALL_MS + POLL_MS);
  poll.tick();
  await poll.settle(0, "reject");
  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(
    poll.reads(),
    2,
    "a rejected read takes the same finally, so it frees the live guard too",
  );
  poll.stop();
});

test("the interval is torn down with the effect", () => {
  const poll = harness();
  poll.stop();
  assert.equal(poll.cleared(), 1, "the poll outlives the component that started it");
});
