// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const engine = readFileSync(
  new URL("../src/features/chat/voice/voice-engine.tsx", import.meta.url),
  "utf8",
);

test("a run that ends with no answer text closes the playback stream it opened", () => {
  const end = engine.slice(engine.indexOf("const streamBegun = streamBegunRef.current;"));
  const start = end.indexOf("if (!text) {");
  const branch = end.slice(start, end.indexOf("const endMsgs", start));
  // Left open, the player polls forever: isSpeaking stays true and queued turns never send.
  assert.match(branch, /if \(streamBegun\) endStreamRef\.current\(""\);\s*else resumeListen\(\);/);
});
