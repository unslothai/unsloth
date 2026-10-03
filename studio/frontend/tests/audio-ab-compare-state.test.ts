// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { INITIAL_AB_STATE, clampPosition, resumeAfterLoad, switchSide } =
  await import("../src/features/audio/components/ab-compare-state.ts");

test("a result opens on Edited, stopped at the start", () => {
  assert.deepEqual(INITIAL_AB_STATE, {
    side: "edited",
    position: 0,
    playing: false,
  });
});

test("switching keeps the moment and the playing flag", () => {
  const playing = { side: "edited" as const, position: 2.5, playing: true };
  assert.deepEqual(switchSide(playing, "original", 4.7), {
    side: "original",
    position: 2.5,
    playing: true,
  });
  const paused = { ...playing, playing: false };
  assert.equal(switchSide(paused, "original", 4.7).playing, false);
  // The same side is a no-op.
  assert.equal(switchSide(playing, "edited", 1), playing);
});

test("the position is clamped to the other clip's length", () => {
  const late = { side: "original" as const, position: 4.6, playing: true };
  assert.equal(switchSide(late, "edited", 3.2).position, 3.2);
  // Unknown length keeps the position; the element clamps it once loaded.
  assert.equal(switchSide(late, "edited", null).position, 4.6);
  assert.equal(clampPosition(-1, 3), 0);
  assert.equal(clampPosition(Number.NaN, 3), 0);
});

test("after loading, the clip resumes at the kept moment, or the top when past its end", () => {
  const state = { side: "edited" as const, position: 2, playing: true };
  assert.deepEqual(resumeAfterLoad(state, 3.2), { time: 2, play: true });
  assert.deepEqual(resumeAfterLoad({ ...state, position: 3.2 }, 3.2), {
    time: 0,
    play: true,
  });
  assert.deepEqual(resumeAfterLoad({ ...state, playing: false }, null), {
    time: 2,
    play: false,
  });
});

test("the A/B player is one keyboard-operable slider over a single audio element", () => {
  const source = readSrc("features/audio/components/ab-compare.tsx");
  assert.match(source, /role="slider"/);
  for (const key of ['" "', '"ArrowRight"', '"ArrowLeft"', '"Home"']) {
    assert.ok(source.includes(`event.key === ${key}`), key);
  }
  assert.equal(source.match(/<audio\b/g)?.length, 1);
});
