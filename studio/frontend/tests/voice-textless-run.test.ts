// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const engine = readFileSync(
  new URL("../src/features/chat/voice/voice-engine.tsx", import.meta.url),
  "utf8",
);

const end = engine.slice(engine.indexOf("const streamBegun = streamBegunRef.current;"));
const start = end.indexOf("if (!text) {");
const branch = end.slice(start, end.indexOf("const endMsgs", start));

test("a run that ends with no answer text closes the playback stream it opened", () => {
  // Left open, the player polls forever: isSpeaking stays true and queued turns never send.
  assert.match(branch, /if \(!streamBegun\) \{\s*endTurns\("noTts", true\);\s*resumeListen\(\);\s*return;\s*\}/);
  assert.match(branch, /endStreamRef\.current\(late\);/);
});

test("the answer is read again before a textless stream is closed", () => {
  // A short answer can land whole after isRunning flipped: closed on the first read, it was never spoken.
  assert.match(branch, /setTimeout\(\(\) => \{[\s\S]*const late = latestAssistantText\(\);[\s\S]*\}, 150\);/);
  // A run that began meanwhile owns the open stream: the late read must not close that one.
  assert.match(branch, /if \(streamBegunRef\.current\) return;\s*const late/);
});
