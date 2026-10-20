// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_FADE_LAYERS,
  pushGradient,
  settleGradient,
  type FadeLayer,
} from "../src/features/chat/voice/orb-crossfade.ts";

const start: FadeLayer[] = [{ g: "listening", id: 0, fading: false }];

test("a new state fades in over the one showing", () => {
  const layers = pushGradient(start, "hearing");
  assert.deepEqual(layers, [
    { g: "listening", id: 0, fading: false },
    { g: "hearing", id: 1, fading: true },
  ]);
  assert.deepEqual(settleGradient(layers, 1), [{ g: "hearing", id: 1, fading: false }]);
});

test("the same state again changes nothing", () => {
  assert.equal(pushGradient(start, "listening"), start);
});

test("hearing arriving mid-fade keeps the earlier fade going instead of snapping it", () => {
  // transcribing fades in over listening, then hearing lands before it has finished.
  const mid = pushGradient(pushGradient(start, "transcribing"), "hearing");
  assert.deepEqual(
    mid.map((l) => [l.g, l.fading]),
    [
      ["listening", false],
      ["transcribing", true],
      ["hearing", true],
    ],
  );
  // The earlier fade ending clears only what is under it; hearing is still fading.
  const after = settleGradient(mid, 1);
  assert.deepEqual(
    after.map((l) => [l.g, l.fading]),
    [
      ["transcribing", false],
      ["hearing", true],
    ],
  );
  assert.deepEqual(settleGradient(after, 2), [{ g: "hearing", id: 2, fading: false }]);
});

test("a settled layer that is gone already is ignored", () => {
  const layers = settleGradient(pushGradient(start, "hearing"), 1);
  assert.equal(settleGradient(layers, 0), layers);
});

test("a flurry of changes keeps the stack bounded", () => {
  let layers = start;
  for (let i = 0; i < 20; i++) layers = pushGradient(layers, i % 2 ? "hearing" : "speaking");
  assert.equal(layers.length, MAX_FADE_LAYERS);
  assert.equal(layers[layers.length - 1].g, "hearing");
});
