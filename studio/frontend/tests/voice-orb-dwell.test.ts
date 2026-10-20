// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ORB_DWELL_MS,
  createOrbDwell,
  type OrbDwellTimers,
} from "../src/features/chat/voice/orb-dwell.ts";
import type { VoiceOrbState } from "../src/features/chat/voice/orb-state.ts";

/** A fake clock whose timers fire only when the test advances it. */
function clock() {
  let t = 0;
  let nextId = 1;
  const timers = new Map<number, { at: number; fn: () => void }>();
  const api: OrbDwellTimers = {
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
    const target = t + ms;
    for (;;) {
      const due = [...timers.entries()]
        .filter(([, x]) => x.at <= target)
        .sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      timers.delete(due[0]);
      t = due[1].at;
      due[1].fn();
    }
    t = target;
  };
  return { api, advance, pending: () => timers.size };
}

function setup() {
  const c = clock();
  const emitted: Array<VoiceOrbState | null> = [];
  const dwell = createOrbDwell((s) => emitted.push(s), c.api);
  return { ...c, emitted, dwell };
}

test("the first state after a blank orb lands at once", () => {
  const { dwell, emitted } = setup();
  dwell.push("listening");
  assert.deepEqual(emitted, ["listening"]);
});

test("a state that would flash is held until the one before it has had its dwell", () => {
  const { dwell, emitted, advance } = setup();
  dwell.push("listening");
  advance(1000);
  dwell.push("transcribing");
  advance(60);
  // The transcript came back in 60 ms; the pencil must not blink.
  dwell.push("listening");
  assert.deepEqual(emitted, ["listening", "transcribing"]);
  advance(ORB_DWELL_MS - 60 - 1);
  assert.deepEqual(emitted, ["listening", "transcribing"]);
  advance(1);
  assert.deepEqual(emitted, ["listening", "transcribing", "listening"]);
});

test("only the newest raw state is pending; the ones it overtook never show", () => {
  const { dwell, emitted, advance, pending } = setup();
  dwell.push("listening");
  advance(1000);
  dwell.push("transcribing");
  advance(30);
  dwell.push("thinking");
  advance(30);
  dwell.push("generating");
  assert.equal(pending(), 1);
  advance(ORB_DWELL_MS);
  assert.deepEqual(emitted, ["listening", "transcribing", "generating"]);
});

test("hearing and speaking always land at once, even mid-dwell", () => {
  for (const instant of ["hearing", "speaking"] as const) {
    const { dwell, emitted, advance } = setup();
    dwell.push("listening");
    advance(1000);
    dwell.push("transcribing");
    advance(20);
    dwell.push(instant);
    assert.deepEqual(emitted, ["listening", "transcribing", instant]);
  }
});

test("an instant state still protects itself from the state after it", () => {
  const { dwell, emitted, advance } = setup();
  dwell.push("listening");
  advance(1000);
  dwell.push("speaking");
  advance(50);
  dwell.push("synthesizing");
  assert.deepEqual(emitted, ["listening", "speaking"]);
  advance(ORB_DWELL_MS);
  assert.deepEqual(emitted, ["listening", "speaking", "synthesizing"]);
});

test("returning to the displayed state cancels the pending switch", () => {
  const { dwell, emitted, advance, pending } = setup();
  dwell.push("listening");
  advance(1000);
  dwell.push("speaking");
  advance(50);
  // A brief gap between two clips: synthesizing was queued, then playback resumed.
  dwell.push("synthesizing");
  dwell.push("speaking");
  assert.equal(pending(), 0);
  advance(ORB_DWELL_MS);
  assert.deepEqual(emitted, ["listening", "speaking"]);
});

test("the loop stopping clears the orb at once and drops anything pending", () => {
  const { dwell, emitted, advance, pending } = setup();
  dwell.push("listening");
  advance(1000);
  dwell.push("transcribing");
  advance(20);
  dwell.push("generating");
  dwell.push(null);
  assert.equal(pending(), 0);
  assert.deepEqual(emitted, ["listening", "transcribing", null]);
  // A fresh loop starts clean: no dwell carried over from before the stop.
  advance(10);
  dwell.push("loading");
  assert.deepEqual(emitted, ["listening", "transcribing", null, "loading"]);
});

test("after the dwell has passed a change shows immediately", () => {
  const { dwell, emitted, advance } = setup();
  dwell.push("listening");
  advance(ORB_DWELL_MS);
  dwell.push("thinking");
  assert.deepEqual(emitted, ["listening", "thinking"]);
  assert.equal(dwell.shown(), "thinking");
});

test("dispose drops the pending switch without touching what is shown", () => {
  const { dwell, emitted, advance, pending } = setup();
  dwell.push("listening");
  advance(1000);
  dwell.push("transcribing");
  advance(20);
  dwell.push("generating");
  dwell.dispose();
  assert.equal(pending(), 0);
  advance(ORB_DWELL_MS);
  assert.deepEqual(emitted, ["listening", "transcribing"]);
  assert.equal(dwell.shown(), "transcribing");
});
