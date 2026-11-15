// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MIC_SILENT_MS,
  createReacquireBackoff,
  createSilenceWatch,
  createSingleFlight,
} from "../src/features/chat/voice/mic-health.ts";

const FRAME = 85; // 4096 samples at 48 kHz

test("a mic delivering only zeros for 1.5 s is re-acquired, once", () => {
  const w = createSilenceWatch(0);
  let t = 0;
  for (; t < MIC_SILENT_MS; t += FRAME) {
    w.frame(t, false);
    assert.equal(w.due(t), false);
  }
  w.frame(t, false);
  assert.equal(w.due(t), true);
  assert.equal(w.due(t + 5000), false, "one re-acquire per session");
});

test("a mic that sends no frames at all is caught the same way", () => {
  const w = createSilenceWatch(1000);
  assert.equal(w.due(2000), false);
  assert.equal(w.due(1000 + MIC_SILENT_MS), true);
});

test("a quiet room is not a dead mic: any non-zero sample keeps it alive", () => {
  const w = createSilenceWatch(0);
  for (let t = 0; t < 10000; t += FRAME) {
    w.frame(t, t % (FRAME * 10) === 0);
    assert.equal(w.due(t), false, `at ${t}`);
  }
});

test("a user mute resets the clock", () => {
  const w = createSilenceWatch(0);
  w.reset(1400);
  assert.equal(w.due(2000), false);
  assert.equal(w.due(1400 + MIC_SILENT_MS), true);
});

test("re-acquiring backs off after three tries that heard nothing, and recovers on sound", () => {
  const b = createReacquireBackoff(3, 10000);
  for (let i = 0; i < 3; i++) {
    assert.ok(b.allow(i * 2000));
    b.tried(i * 2000);
  }
  assert.equal(b.allow(5000), false);
  assert.ok(b.allow(4000 + 10000));
  b.heard();
  assert.ok(b.allow(4100));
});

test("mic opens run one at a time, never overlapping", async () => {
  const run = createSingleFlight();
  let open = 0;
  let most = 0;
  const order: string[] = [];
  const opener = (name: string, ms: number) => async () => {
    open++;
    most = Math.max(most, open);
    order.push(`${name}+`);
    await new Promise((r) => setTimeout(r, ms));
    order.push(`${name}-`);
    open--;
    return name;
  };
  const results = await Promise.all([run(opener("a", 20)), run(opener("b", 1)), run(opener("c", 5))]);
  assert.deepEqual(results, ["a", "b", "c"]);
  assert.equal(most, 1);
  assert.deepEqual(order, ["a+", "a-", "b+", "b-", "c+", "c-"]);
});

test("a failed open does not block the next one", async () => {
  const run = createSingleFlight();
  const failed = run(async () => {
    throw new Error("NotAllowedError");
  });
  await assert.rejects(failed);
  assert.equal(await run(async () => "ok"), "ok");
});
