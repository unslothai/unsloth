// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How many pending sentences go into one synth request: one for a fast start,
// then merged up to a budget, with a bounded wait for more while nothing plays.

import assert from "node:assert/strict";
import test from "node:test";

import {
  planTtsChunk,
  TTS_CHUNK_MAX_CHARS,
  TTS_CHUNK_TARGET_CHARS,
  TTS_CHUNK_WAIT_MS,
} from "../src/features/chat/voice/tts-chunk-plan.ts";

const s = (n: number) => "x".repeat(n - 1) + ".";
const plan = (over: Partial<Parameters<typeof planTtsChunk>[0]>) =>
  planTtsChunk({ pending: [], first: false, final: false, playing: false, waitedMs: 0, ...over });

test("nothing pending is a wait", () => {
  assert.equal(plan({ pending: [] }), 0);
  assert.equal(plan({ pending: [], final: true }), 0);
});

test("the first chunk is the first sentence alone", () => {
  assert.equal(plan({ pending: [s(10), s(10), s(10)], first: true }), 1);
  assert.equal(plan({ pending: [s(400)], first: true, final: true }), 1);
});

test("later sentences merge up to the target", () => {
  // 100 + 1 + 100 = 201 < target, the third one crosses it and is still taken.
  assert.equal(plan({ pending: [s(100), s(100), s(100), s(100)] }), 3);
  // Once over the target the next sentence waits for its own chunk.
  assert.equal(plan({ pending: [s(TTS_CHUNK_TARGET_CHARS), s(10), s(10)] }), 1);
});

test("the hard cap closes a chunk and never splits a sentence", () => {
  // 200 + 1 + 150 would pass the cap, so the second sentence starts the next chunk.
  assert.equal(plan({ pending: [s(200), s(150), s(10)] }), 1);
  // A sentence longer than the cap goes alone, whole.
  assert.equal(plan({ pending: [s(TTS_CHUNK_MAX_CHARS + 50), s(10)] }), 1);
  assert.equal(plan({ pending: [s(TTS_CHUNK_MAX_CHARS + 50)], final: true }), 1);
});

test("a short tail is sent once the stream is final", () => {
  assert.equal(plan({ pending: [s(20), s(20)], final: true, playing: true }), 2);
  assert.equal(plan({ pending: [s(20)], final: true }), 1);
});

test("a short chunk waits for more while a clip is playing", () => {
  assert.equal(plan({ pending: [s(20), s(20)], playing: true }), 0);
  assert.equal(plan({ pending: [s(20)], playing: true, waitedMs: 10_000 }), 0);
});

test("an idle player waits at most TTS_CHUNK_WAIT_MS for a bigger chunk", () => {
  assert.equal(plan({ pending: [s(20)], waitedMs: 0 }), 0);
  assert.equal(plan({ pending: [s(20)], waitedMs: TTS_CHUNK_WAIT_MS - 1 }), 0);
  assert.equal(plan({ pending: [s(20), s(20)], waitedMs: TTS_CHUNK_WAIT_MS }), 2);
});

test("sentences stripped to nothing count but add no text", () => {
  assert.equal(plan({ pending: ["", s(20), ""], final: true }), 3);
});
