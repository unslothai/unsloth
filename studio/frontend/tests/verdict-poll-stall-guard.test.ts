// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Lifted from app-sidebar.tsx and run with a fake clock. An abandoned stalled read must not
// release the guard while its replacement is in flight, or every tick fires a forced read.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const src = await readSrcAsync("components/app-sidebar.tsx");

const START = "let pollingSince = 0;";
// The follow-up cancel appears twice, so it cannot delimit the block.
const END = "return () => stopPolling();";
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
const FOLLOW_UP_MS = constant("INVENTORY_FOLLOW_UP_MS");

const startPoll = new Function(
  "window",
  "Date",
  "fetchDeviceType",
  "capabilitiesUnknown",
  "VERDICT_POLL_STALL_MS",
  "VERDICT_UNKNOWN_POLL_MS",
  "SELF_HEAL_POLL_MS",
  // Unused in this scenario but needed so the lifted block evaluates.
  "selfHealSettled",
  "INVENTORY_FOLLOW_UP_MS",
  body,
) as (
  window: {
    setInterval: (fn: () => void, ms: number) => number;
    clearInterval: (id: number) => void;
    setTimeout: (fn: () => void, ms: number) => number;
    clearTimeout: (id: number) => void;
  },
  date: { now: () => number },
  fetchDeviceType: () => Promise<void>,
  capabilitiesUnknown: boolean,
  stallMs: number,
  unknownPollMs: number,
  selfHealPollMs: number,
  selfHealSettled: boolean,
  followUpMs: number,
) => () => void;

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
      setTimeout: () => 0,
      clearTimeout: () => undefined,
    },
    { now: () => clock },
    () =>
      new Promise<void>((resolve, reject) => {
        pending.push({ resolve: () => resolve(), reject: () => reject(new Error("offline")) });
      }),
    true,
    STALL_MS,
    POLL_MS,
    15000,
    false,
    FOLLOW_UP_MS,
  );

  assert.ok(tick, "the poll never scheduled an interval");
  return {
    cadence,
    reads: () => pending.length,
    advance: (ms: number) => {
      clock += ms;
    },
    tick: () => tick?.(),
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

  poll.advance(STALL_MS + POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 2, "the stall window did not release the guard");

  await poll.settle(0);

  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(
    poll.reads(),
    2,
    "the abandoned read cleared a guard it no longer held, so the poll stacked another " +
      "forced /api/health onto the backend it is waiting for",
  );
  poll.advance(POLL_MS);
  poll.tick();
  poll.advance(POLL_MS);
  poll.tick();
  assert.equal(poll.reads(), 2, "the guard leaked one tick later instead");

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
