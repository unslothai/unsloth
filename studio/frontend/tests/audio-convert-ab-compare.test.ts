// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { nextPlayback } = await import(
  "../src/features/audio/components/convert-ab-compare-state.ts"
);

test("switching sides keeps the moment and play state, clamped to the new side", () => {
  const cases: [number, boolean, number, number, boolean][] = [
    // time, playing, next duration -> seek, resume
    [4.2, true, 10, 4.2, true],
    [4.2, false, 12, 4.2, false],
    [3, true, 6, 3, true],
    [2, true, Number.NaN, 2, true],
    [Number.NaN, false, 5, 0, false],
    [-1, true, 5, 0, true],
  ];
  for (const [time, playing, next, seek, resume] of cases) {
    assert.deepEqual(nextPlayback({ time, playing, duration: 10 }, next), {
      seek,
      resume,
    });
  }
  const past = nextPlayback({ time: 9, playing: true, duration: 10 }, 6);
  assert.ok(past.seek < 6 && past.seek > 5.9, String(past.seek));
  assert.equal(past.resume, false);
});

test("one player, swapped by a Tab-reachable toggle", () => {
  const jsx = readSrc("features/audio/components/convert-ab-compare.tsx");
  assert.equal(jsx.match(/<audio\b/g)?.length, 1);
  assert.match(jsx, /<PillTabs\b/);
  assert.doesNotMatch(jsx, /autoPlay/);
  const tabs = readSrc(
    "features/model-picker/components/model-selector/pill-tabs.tsx",
  );
  assert.match(tabs, /tabIndex=\{value === tab\.value \? 0 : -1\}/);
  assert.match(tabs, /disabled=\{disabled \|\| tab\.disabled\}/);
});
