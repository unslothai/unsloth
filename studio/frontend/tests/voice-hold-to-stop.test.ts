// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  HOLD_TO_STOP_MS,
  TAP_MAX_MS,
  createHoldToStop,
  type HoldPhase,
  type HoldTimers,
} from "../src/features/chat/voice/hold-to-stop.ts";

function clock() {
  let t = 0;
  let nextId = 1;
  const timers = new Map<number, { at: number; fn: () => void }>();
  const api: HoldTimers = {
    now: () => t,
    setTimer: (fn, ms) => {
      const id = nextId++;
      timers.set(id, { at: t + ms, fn });
      return id;
    },
    clearTimer: (h) => {
      timers.delete(h as number);
    },
  };
  const advance = (ms: number) => {
    const until = t + ms;
    for (;;) {
      const due = [...timers.entries()]
        .filter(([, e]) => e.at <= until)
        .sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      timers.delete(due[0]);
      t = due[1].at;
      due[1].fn();
    }
    t = until;
  };
  return { api, advance, pending: () => timers.size };
}

function setup() {
  const c = clock();
  const phases: HoldPhase[] = [];
  const hold = createHoldToStop((p) => phases.push(p), c.api);
  return { c, phases, hold };
}

test("a quick press and release is a tap", () => {
  const { c, phases, hold } = setup();
  hold.press();
  c.advance(120);
  assert.equal(hold.release(), "tap");
  assert.deepEqual(phases, ["holding", "idle"]);
  assert.equal(c.pending(), 0);
});

test("a press held past the arm time stops on release", () => {
  const { c, phases, hold } = setup();
  hold.press();
  c.advance(HOLD_TO_STOP_MS);
  assert.equal(hold.phase(), "armed");
  assert.equal(hold.release(), "stop");
  assert.deepEqual(phases, ["holding", "armed", "idle"]);
});

test("a slow click that never armed does nothing", () => {
  const { c, hold } = setup();
  hold.press();
  c.advance(TAP_MAX_MS + 1);
  assert.equal(hold.release(), "none");
  assert.equal(c.pending(), 0);
});

test("the boundary release is still a tap", () => {
  const { c, hold } = setup();
  hold.press();
  c.advance(TAP_MAX_MS);
  assert.equal(hold.release(), "tap");
});

test("a lost press fires nothing, even once armed", () => {
  const { c, phases, hold } = setup();
  hold.press();
  c.advance(HOLD_TO_STOP_MS + 50);
  hold.cancel();
  assert.equal(hold.phase(), "idle");
  assert.equal(hold.release(), "none");
  assert.deepEqual(phases, ["holding", "armed", "idle"]);
});

test("a repeated press while held does not restart the clock", () => {
  const { c, hold } = setup();
  hold.press();
  c.advance(400);
  hold.press();
  c.advance(250);
  assert.equal(hold.phase(), "armed");
  assert.equal(hold.release(), "stop");
});

test("release with no press is nothing", () => {
  const { hold } = setup();
  assert.equal(hold.release(), "none");
});
