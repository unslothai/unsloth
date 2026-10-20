// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Barge-in fires on a held burst over the reply's own leak, never on the leak.

import assert from "node:assert/strict";
import test from "node:test";

import { createBargeGate, stepBargeGate } from "../src/features/chat/voice/barge-gate.ts";

const FRAME_MS = 85; // 4096 samples at 48 kHz

function feed(gate: ReturnType<typeof createBargeGate>, rms: number, frames: number) {
  let fired = false;
  for (let i = 0; i < frames; i++) fired = stepBargeGate(gate, rms, FRAME_MS) || fired;
  return fired;
}

test("the reply leaking into the mic never cuts it off", () => {
  const gate = createBargeGate();
  assert.equal(feed(gate, 0.03, 200), false);
});

test("a held burst cuts in, a short one does not", () => {
  const short = createBargeGate();
  feed(short, 0.005, 20);
  assert.equal(feed(short, 0.2, 5), false);
  const held = createBargeGate();
  feed(held, 0.005, 20);
  assert.equal(feed(held, 0.2, 6), true);
});

test("a loud leak raises the bar above the fixed minimum", () => {
  const gate = createBargeGate();
  feed(gate, 0.04, 100);
  // Over the 0.045 floor, but not twice the measured leak.
  assert.equal(feed(gate, 0.07, 10), false);
  assert.equal(feed(gate, 0.2, 6), true);
});

test("a cough-length burst does not cut in", () => {
  const gate = createBargeGate();
  feed(gate, 0.005, 20);
  // ~340 ms loud then quiet: the shape of a cough, not of someone talking.
  assert.equal(feed(gate, 0.3, 4), false);
  assert.equal(feed(gate, 0.005, 5), false);
});

test("a shorter hold applies when a detector confirms the burst afterwards", () => {
  const gate = createBargeGate(250);
  let fired = false;
  for (let i = 0; i < 3; i++) fired = stepBargeGate(gate, 0.2, 93);
  assert.equal(fired, true);
});
