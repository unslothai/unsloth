// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { voiceRunToStop } from "../src/features/chat/voice/voice-loop-bridge.ts";

const engine = readFileSync(
  new URL("../src/features/chat/voice/voice-engine.tsx", import.meta.url),
  "utf8",
);

test("the voiced thread outlives a VoiceEngine remount (New Chat)", () => {
  // A per-instance ref reads null on the remounted engine, so the old run kept generating.
  assert.doesNotMatch(engine, /voicedThreadRef\s*=\s*useRef/);
  assert.match(engine, /^let _voicedThreadId: string \| null = null;$/m);
  // The reset reads the module value and clears it once used.
  assert.match(engine, /voiceRunToStop\(\s*prevThreadId,\s*_voicedThreadId,/);
  // Turning voice off forgets it; a run ending only forgets it once that thread is idle,
  // because the remounted engine's first effect sees a run "end" that is still going.
  assert.match(engine, /const turnOff = useCallback\(\(\) => \{\s*_voicedThreadId = null;/);
  assert.match(
    engine,
    /if \(_voicedThreadId && !useChatRuntimeStore\.getState\(\)\.runningByThreadId\[_voicedThreadId\]\)/,
  );
});

test("with the voiced id kept, the reset stops the old thread's run", () => {
  const running = { "thread-A": true };
  assert.equal(voiceRunToStop("thread-A", "thread-A", running), "thread-A");
  assert.equal(voiceRunToStop("thread-A", null, running), null);
  assert.equal(voiceRunToStop("thread-A", "thread-A", { "thread-A": false }), null);
});
