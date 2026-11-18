// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  createClipSpeculation,
  utteranceSpan,
} from "../src/features/chat/voice/clip-speculation.ts";

const frames = (pattern: string) => [...pattern].map((c) => c === "v");
const quiet = (pattern: string) => [...pattern].map((c) => c !== "_");

test("the clip stops changing once the pad of silence after the last voiced frame is in", () => {
  const p = "__qvvv_v";
  const at = utteranceSpan(frames(p + "__"), quiet(p + "__"), 2);
  assert.deepEqual(at, { start: 0, end: 9 });
  assert.deepEqual(utteranceSpan(frames(p + "______"), quiet(p + "______"), 2), at);
  assert.notDeepEqual(utteranceSpan(frames(p + "__v__"), quiet(p + "__v__"), 2), at);
  assert.equal(utteranceSpan(frames("____"), quiet("____"), 2), null);
});

test("a quiet onset before the first voiced frame stays in the clip", () => {
  assert.deepEqual(utteranceSpan(frames("____qqv____"), quiet("____qqv____"), 1), { start: 3, end: 7 });
});

test("the same clip is analysed once; another clip cancels the stale analysis", () => {
  const made: string[] = [];
  const cancelled: string[] = [];
  const spec = createClipSpeculation<{ key: string; cancel(): void }>();
  const analyse = (key: string) => () => {
    made.push(key);
    return { key, cancel: () => cancelled.push(key) };
  };
  spec.start("a", analyse("a"));
  spec.start("a", analyse("a"));
  assert.deepEqual(made, ["a"]);
  spec.start("b", analyse("b"));
  assert.deepEqual(cancelled, ["a"]);
  assert.equal(spec.take("c"), null);
  assert.deepEqual(cancelled, ["a", "b"]);
  spec.start("d", analyse("d"));
  assert.equal(spec.take("d")?.key, "d");
  assert.equal(spec.take("d"), null);
  spec.start("e", analyse("e"));
  spec.clear();
  assert.deepEqual(cancelled, ["a", "b", "e"]);
});
