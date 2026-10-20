// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice mode unloads only the voice it loaded, and never keeps one it stopped wanting mid-load.

import assert from "node:assert/strict";
import test from "node:test";

import {
  holdForServing,
  settleVoiceLoad,
  voiceToUnload,
} from "../src/features/chat/voice/voice-ownership.ts";

test("a voice the Audio page already serves is adopted, not owned", () => {
  const hold = holdForServing(null, "unsloth/voice-a");
  assert.deepEqual(hold, { id: "unsloth/voice-a", owned: false });
  assert.equal(voiceToUnload(hold), null);
});

test("a voice this page loaded is unloaded when voice mode lets go", () => {
  const { hold, unload } = settleVoiceLoad("unsloth/voice-a", 3, 3);
  assert.deepEqual(hold, { id: "unsloth/voice-a", owned: true });
  assert.equal(unload, null);
  assert.equal(voiceToUnload(hold), "unsloth/voice-a");
  assert.deepEqual(holdForServing(hold, "unsloth/voice-a"), hold);
});

test("a load that settles after voice mode turned off is unloaded, not kept", () => {
  const { hold, unload } = settleVoiceLoad("unsloth/voice-a", 3, 4);
  assert.equal(hold, null);
  assert.equal(unload, "unsloth/voice-a");
});
