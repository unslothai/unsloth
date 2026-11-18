// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A reply still finishing on the server is written back by its run follow from a metadata copy taken
// before the user cut in. That write replaced the saved heard cut, in the thread and in the database,
// so the model heard the whole reply again from the turn after next.

import assert from "node:assert/strict";
import test from "node:test";

import { keepVoiceHeard } from "../src/features/chat/voice/heard-cut.ts";
import { noteWrittenHeardCut, writtenHeardCustom } from "../src/features/chat/voice/voice-loop-bridge.ts";

const cut = { offset: 148, at: 1 };

test("an older copy keeps the cut the live reply carries", () => {
  const next = { stopReason: "stop", generationChunkCount: 597 };
  assert.deepEqual(keepVoiceHeard(next, { voiceHeard: cut, stopReason: "stop" }), { ...next, voiceHeard: cut });
});

test("no live cut (none taken, or the full reply restored) adds nothing", () => {
  const next = { stopReason: "stop" };
  assert.equal(keepVoiceHeard(next, {}), next);
  assert.equal(keepVoiceHeard(next, undefined), next);
});

test("the open reply's cut wins over the one an older copy carries", () => {
  const stale = { stopReason: "stop", voiceHeard: { offset: 300, at: 2 } };
  assert.deepEqual(keepVoiceHeard(stale, { voiceHeard: cut }), { stopReason: "stop", voiceHeard: cut });
  assert.equal(keepVoiceHeard({ voiceHeard: cut }, { voiceHeard: cut }).voiceHeard, cut);
});

test("a cut the user cleared with Use full reply is not written back from an older copy", () => {
  const stale = { stopReason: "stop", voiceHeard: cut };
  assert.deepEqual(keepVoiceHeard(stale, {}), { stopReason: "stop" });
  // No open copy to ask: the writer's own stands.
  assert.equal(keepVoiceHeard(stale, undefined), stale);
});

test("with the thread off screen, the cut this session last wrote stands in for the open copy", () => {
  assert.equal(writtenHeardCustom("r-offscreen"), undefined);
  noteWrittenHeardCut("r-offscreen", cut);
  assert.deepEqual(keepVoiceHeard({ stopReason: "stop" }, writtenHeardCustom("r-offscreen")), { stopReason: "stop", voiceHeard: cut });
  noteWrittenHeardCut("r-offscreen", null);
  assert.deepEqual(keepVoiceHeard({ voiceHeard: cut }, writtenHeardCustom("r-offscreen")), {});
});
