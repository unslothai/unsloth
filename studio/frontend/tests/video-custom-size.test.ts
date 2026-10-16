// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import {
  fitVideoSize,
  isValidVideoSize,
  videoDimOptions,
  videoSizeLimits,
} from "../src/features/video/video-size.ts";

const LTX2_PRESETS: Array<[number, number]> = [
  [768, 512],
  [1216, 704],
  [704, 1216],
  [512, 768],
];

function ltx2Defaults(extra: Partial<VideoGenerationDefaults> = {}): VideoGenerationDefaults {
  return {
    steps: 8,
    guidance: 1,
    num_frames: 121,
    fps: 24,
    frame_step: 8,
    frame_offset: 1,
    duration_presets: [1, 2, 3, 5],
    resolution_multiple: 32,
    resolution_presets: LTX2_PRESETS,
    custom_size_min_pixels: 768 * 512,
    custom_size_max_pixels: 1216 * 704,
    custom_size_max_aspect: 4,
    ...extra,
  };
}

test("custom sizes on the grid inside the backend band are kept as typed", () => {
  const limits = videoSizeLimits(ltx2Defaults(), LTX2_PRESETS);
  assert.deepEqual(fitVideoSize(1024, 576, limits), [1024, 576]);
  assert.deepEqual(fitVideoSize(768, 512, limits), [768, 512]);
  assert.equal(isValidVideoSize(1000, 576, limits), false);
});

test("sizes outside the band are rescaled into it at the same ratio", () => {
  const limits = videoSizeLimits(ltx2Defaults(), LTX2_PRESETS);
  const [w, h] = fitVideoSize(1920, 1080, limits);
  assert.ok(isValidVideoSize(w, h, limits));
  assert.ok(Math.abs(w / h - 16 / 9) < 0.1, `${w}x${h} drifted from 16:9`);
  const [sw, sh] = fitVideoSize(256, 256, limits);
  assert.ok(isValidVideoSize(sw, sh, limits) && sw === sh, `${sw}x${sh}`);
});

test("a backend without custom-size bounds keeps the presets only", () => {
  const limits = videoSizeLimits(
    ltx2Defaults({
      custom_size_min_pixels: undefined,
      custom_size_max_pixels: undefined,
      custom_size_max_aspect: undefined,
    }),
    LTX2_PRESETS,
  );
  assert.equal(limits.presetsOnly, true);
  assert.deepEqual(fitVideoSize(1920, 1080, limits), [1216, 704]);
  assert.deepEqual(videoDimOptions(limits), [512, 704, 768, 1216]);
});

test("before a model loads the fallback presets bound the size like the backend would", () => {
  const limits = videoSizeLimits(null, LTX2_PRESETS.slice(0, 3));
  assert.equal(limits.presetsOnly, false);
  assert.equal(limits.minPixels, 768 * 512);
  assert.equal(limits.maxPixels, 1216 * 704);
  assert.ok(videoDimOptions(limits).every((n) => n % 32 === 0));
});
