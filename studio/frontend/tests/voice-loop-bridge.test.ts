// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  isVoiceRequested,
  registerVoiceToggle,
  requestVoiceToggle,
  subscribeVoiceRequested,
} from "../src/features/chat/voice/voice-loop-bridge.ts";

const SRC = new URL("../src/", import.meta.url);
const read = (relative: string) => readFileSync(new URL(relative, SRC), "utf8");

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

test("the voice chunk is rendered only once Voice is requested", () => {
  const thread = read("components/assistant-ui/thread.tsx");
  for (const name of ["VoiceOrb", "VoiceEngine", "VoiceMuteButton", "VoiceControlButton"]) {
    const at = thread.indexOf(`<${name} />`);
    assert.ok(at > 0, name);
    const gate = thread.lastIndexOf("{voiceRequested && (", at);
    assert.ok(gate > 0 && at - gate < 160, `${name} is gated on voiceRequested`);
  }
});

test("the loop's mic plays nothing back and starts its context", () => {
  const adapter = read("features/chat/adapters/studio-voice-loop-adapter.ts");
  assert.doesNotMatch(adapter, /processor\.connect\(audioCtx\.destination\)/);
  assert.match(adapter, /sink\.gain\.value = 0;/);
  assert.match(adapter, /processor\.connect\(sink\);\s*sink\.connect\(audioCtx\.destination\);/);
  assert.ok(
    adapter.indexOf("await audioCtx.resume();") < adapter.indexOf("createScriptProcessor("),
  );
});
