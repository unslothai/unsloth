// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { nextPlayback } = await import(
  "../src/features/audio/components/ab-compare-state.ts"
);

test("switching sides keeps the moment and whether it was playing", () => {
  assert.deepEqual(
    nextPlayback({ time: 4.2, playing: true, duration: 10 }, 10),
    {
      seek: 4.2,
      resume: true,
    },
  );
  assert.deepEqual(
    nextPlayback({ time: 4.2, playing: false, duration: 10 }, 12),
    {
      seek: 4.2,
      resume: false,
    },
  );
});

test("a shorter side clamps to its end and does not resume past it", () => {
  const next = nextPlayback({ time: 9, playing: true, duration: 10 }, 6);
  assert.ok(next.seek < 6 && next.seek > 5.9, String(next.seek));
  assert.equal(next.resume, false);
  // Inside the shorter side it plays on.
  assert.deepEqual(nextPlayback({ time: 3, playing: true, duration: 10 }, 6), {
    seek: 3,
    resume: true,
  });
});

test("unknown lengths and bad times fall back safely", () => {
  assert.deepEqual(
    nextPlayback({ time: 2, playing: true, duration: Number.NaN }, Number.NaN),
    {
      seek: 2,
      resume: true,
    },
  );
  assert.deepEqual(
    nextPlayback({ time: Number.NaN, playing: false, duration: 5 }, 5),
    {
      seek: 0,
      resume: false,
    },
  );
  assert.deepEqual(nextPlayback({ time: -1, playing: true, duration: 5 }, 5), {
    seek: 0,
    resume: true,
  });
});

test("one player, swapped by a Tab-reachable toggle", () => {
  const jsx = readSrc("features/audio/components/ab-compare.tsx");
  assert.equal(jsx.match(/<audio\b/g)?.length, 1);
  assert.match(jsx, /<PillTabs\b/);
  assert.match(jsx, /onLoadedMetadata=\{handleLoadedMetadata\}/);
  // An unavailable side disables its tab and says why.
  assert.match(jsx, /disabled: Boolean\(a\.unavailable\)/);
  assert.match(jsx, /disabled: Boolean\(b\.unavailable\)/);
  // No autoplay and no motion of its own.
  assert.doesNotMatch(jsx, /autoPlay|animate-|transition-/);
  const tabs = readSrc(
    "features/model-picker/components/model-selector/pill-tabs.tsx",
  );
  // Roving tabindex: the selected tab is always in the Tab order.
  assert.match(tabs, /tabIndex=\{value === tab\.value \? 0 : -1\}/);
  assert.match(tabs, /disabled=\{disabled \|\| tab\.disabled\}/);
});
