// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { type PageWait, waitForPage } from "../src/features/browser/agent/page-wait.ts";

const QUIET = 350;

/** A page on a fake clock: requests open from `start` to `end`, DOM changes at `mutations`. The status
 *  call answers as page-agent.ts does, counting open requests started at or after `since`. */
function page(requests: Array<{ start: number; end: number }>, mutations: number[] = []) {
  let now = 0;
  const wait = (since: number, maxMs: number, changingMaxMs: number): PageWait => ({
    maxMs, changingMaxMs, quietMs: QUIET,
    now: () => now,
    sleep: async (ms) => {
      now += ms;
    },
    status: async () => {
      // The command takes a moment to reach the page.
      now += 5;
      const pending = requests.filter((r) => r.start >= since && r.start <= now && now < r.end).length;
      const last = Math.max(-Infinity, ...mutations.filter((t) => t <= now));
      return { requests: pending, quietMs: now - last };
    },
    moved: () => false,
  });
  return { wait, at: (t: number) => (now = t), now: () => now };
}

test("a request the action set off at once is waited for, though it started as the action went out", async () => {
  // The click handler fetches synchronously: the request starts before the click's answer comes back.
  const p = page([{ start: 1003, end: 2500 }], [0]);
  p.at(1120);
  assert.equal(await waitForPage(p.wait(1000, 3000, 3000)), "settled");
  assert.ok(p.now() >= 2500, `returned at ${p.now()} with the request still open`);
});

test("a request open since the page loaded still counts at the deadline, and the wait says so", async () => {
  const p = page([{ start: 200, end: 60_000 }], [0]);
  p.at(400);
  assert.equal(await waitForPage(p.wait(0, 3000, 1000)), "loading");
  const waited = p.now() - 400;
  assert.ok(waited >= 3000 && waited < 3300, `waited ${waited} ms`);
});

test("a long poll the page had open before the action doesn't hold it", async () => {
  const p = page([{ start: -5000, end: 60_000 }], [0]);
  p.at(1000);
  assert.equal(await waitForPage(p.wait(1000, 3000, 3000)), "settled");
  assert.ok(p.now() < 1100);
});

test("a DOM that never rests is waited on only for changingMaxMs once no request is open", async () => {
  const ticks = Array.from({ length: 200 }, (_, i) => i * 100);
  const p = page([], ticks);
  assert.equal(await waitForPage(p.wait(0, 3000, 1000)), "settled");
  assert.ok(p.now() >= 1000 && p.now() < 1300, `waited ${p.now()} ms`);
});

test("a quiet page with nothing open is read at once", async () => {
  const p = page([], [0]);
  p.at(5000);
  assert.equal(await waitForPage(p.wait(5000, 3000, 3000)), "settled");
  assert.ok(p.now() < 5100);
});

test("a page that moves away hands back to the caller", async () => {
  const p = page([{ start: 0, end: 60_000 }]);
  const wait = { ...p.wait(0, 3000, 3000), moved: () => p.now() > 300 };
  assert.equal(await waitForPage(wait), "moved");
});

test("the DOM a slow response sets off still gets its quiet time after the request closes", async () => {
  // The list's request is open past changingMaxMs; drawing it takes 100 ms after it lands.
  const drawing = Array.from({ length: 6 }, (_, i) => 1500 + i * 20);
  const p = page([{ start: 0, end: 1500 }], [0, ...drawing]);
  assert.equal(await waitForPage(p.wait(0, 3000, 1000)), "settled");
  assert.ok(p.now() >= 1600 + QUIET, `read at ${p.now()}, while the list was still being drawn`);
});

test("a DOM quiet while a request was open is not read as settled the moment it closes", async () => {
  // Nothing changes while the list loads; it is drawn 200 ms after its response lands.
  const p = page([{ start: 0, end: 800 }], [0, 1000]);
  assert.equal(await waitForPage(p.wait(0, 3000, 1000)), "settled");
  assert.ok(p.now() >= 1000 + QUIET, `read at ${p.now()}, before the list was drawn`);
});

test("a loaded page that opened no request is read after the short idle quiet", async () => {
  const p = page([], [0]);
  assert.equal(await waitForPage({ ...p.wait(0, 3000, 1000), idleQuietMs: 120 }), "settled");
  assert.ok(p.now() >= 120 && p.now() < QUIET, `read at ${p.now()}`);
});

test("a page whose requests nothing counts (a native view) still gets the full quiet", async () => {
  const p = page([], [0]);
  const wait = p.wait(0, 3000, 1000);
  const counted = wait.status;
  wait.status = async () => {
    const status = await counted();
    return status && { quietMs: status.quietMs };
  };
  assert.equal(await waitForPage({ ...wait, idleQuietMs: 120 }), "settled");
  assert.ok(p.now() >= QUIET, `read at ${p.now()}`);
});

test("a page that had a request open gets the full quiet after it closes, short idle or not", async () => {
  const p = page([{ start: 0, end: 300 }], [0, 300]);
  assert.equal(await waitForPage({ ...p.wait(0, 3000, 1000), idleQuietMs: 120 }), "settled");
  assert.ok(p.now() >= 300 + QUIET, `read at ${p.now()}`);
});
