// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isVoiceRequested,
  registerVoiceThreadReset,
  registerVoiceToggle,
  requestVoiceThreadReset,
  requestVoiceToggle,
  subscribeVoiceRequested,
  voiceRunToStop,
} from "../src/features/chat/voice/voice-loop-bridge.ts";

// Runs first: the requested flag is module state that only ever turns on.
test("the loop is not requested until Voice is clicked, then stays requested", () => {
  assert.equal(isVoiceRequested(), false);
  let notified = 0;
  const unsubscribe = subscribeVoiceRequested(() => notified++);
  requestVoiceToggle();
  requestVoiceToggle();
  assert.equal(isVoiceRequested(), true);
  assert.equal(notified, 1);
  unsubscribe();
  registerVoiceToggle(() => {});
  registerVoiceToggle(null);
});

test("clicks before the loop registers replay by parity", () => {
  for (const clicks of [1, 2, 3, 4]) {
    let toggled = 0;
    for (let i = 0; i < clicks; i++) requestVoiceToggle();
    registerVoiceToggle(() => toggled++);
    assert.equal(toggled, clicks % 2, `${clicks} early clicks`);
    registerVoiceToggle(null);
  }
});

test("a click with the loop registered toggles at once and is not replayed later", () => {
  let toggled = 0;
  registerVoiceToggle(() => toggled++);
  requestVoiceToggle();
  assert.equal(toggled, 1);
  registerVoiceToggle(null);
  registerVoiceToggle(() => toggled++);
  assert.equal(toggled, 1);
  registerVoiceToggle(null);
});

test("a thread switch stops only the run the loop was voicing, and only while it runs", () => {
  assert.equal(voiceRunToStop("a", "a", { a: true }), "a");
  assert.equal(voiceRunToStop("a", "a", { a: false }), null);
  assert.equal(voiceRunToStop("a", "b", { a: true }), null);
  assert.equal(voiceRunToStop(null, null, {}), null);
  assert.equal(voiceRunToStop("a", null, { a: true }), null);
});

test("the thread reset hands the loop the thread it is leaving", () => {
  const seen: Array<string | null> = [];
  registerVoiceThreadReset((prev) => seen.push(prev));
  requestVoiceThreadReset("old");
  requestVoiceThreadReset(null);
  registerVoiceThreadReset(null);
  requestVoiceThreadReset("ignored");
  assert.deepEqual(seen, ["old", null]);
});
