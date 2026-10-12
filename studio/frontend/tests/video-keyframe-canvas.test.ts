// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import {
  keyframeSupport,
  matchedCanvas,
  nearestPresetCanvas,
} from "../src/features/video/keyframe-canvas.ts";

const h3: VideoGenerationDefaults = {
  steps: 30,
  guidance: 1,
  num_frames: 124,
  fps: 24,
  frame_step: 17,
  frame_offset: 5,
  duration_presets: [5, 10, 14.4],
  resolution_multiple: 32,
  resolution_presets: [[1344, 768]],
  canvas_short_edge: 768,
  canvas_max_pixels: 768 * 1344,
};

test("previews the same canvas the backend derives from a keyframe", () => {
  // The pairs the backend's own test pins.
  assert.deepEqual(matchedCanvas(1920, 1080, h3), [1344, 768]);
  assert.deepEqual(matchedCanvas(1080, 1920, h3), [768, 1344]);
  assert.deepEqual(matchedCanvas(1000, 1000, h3), [768, 768]);
  assert.deepEqual(matchedCanvas(1024, 768, h3), [1024, 768]);
});

test("reads the ratio only, so the source's own scale never matters", () => {
  assert.deepEqual(matchedCanvas(160, 90, h3), matchedCanvas(3840, 2160, h3));
});

test("previews nothing where the backend would refuse or has no rule", () => {
  // Outside the trained 1:4 - 4:1 band the backend raises.
  assert.equal(matchedCanvas(2000, 400, h3), null);
  assert.equal(matchedCanvas(400, 2000, h3), null);
  assert.equal(matchedCanvas(0, 100, h3), null);
  assert.equal(
    matchedCanvas(1920, 1080, { ...h3, canvas_short_edge: null, canvas_max_pixels: null }),
    null,
  );
  assert.equal(matchedCanvas(1920, 1080, null), null);
});

// What /video/status reports for an image-to-video pipeline family (Wan2.2 TI2V 5B, LTX-2): no
// aspect rule, the preset nearest the start frame instead.
const ti2v: VideoGenerationDefaults = {
  ...h3,
  resolution_presets: [
    [1280, 704],
    [704, 1280],
  ],
  canvas_short_edge: null,
  canvas_max_pixels: null,
  canvas_from_presets: true,
};
const ltx2: VideoGenerationDefaults = {
  ...ti2v,
  resolution_presets: [
    [768, 512],
    [1216, 704],
    [704, 1216],
    [512, 768],
  ],
};

test("an image-to-video family previews the preset nearest the start frame", () => {
  assert.deepEqual(matchedCanvas(1920, 1080, ti2v), [1280, 704]);
  assert.deepEqual(matchedCanvas(832, 1104, ti2v), [704, 1280]);
  assert.deepEqual(matchedCanvas(512, 512, ti2v), [1280, 704]);
  assert.deepEqual(matchedCanvas(1920, 1080, ltx2), [1216, 704]);
  assert.deepEqual(matchedCanvas(3, 2, ltx2), [768, 512]);
  assert.deepEqual(matchedCanvas(2, 3, ltx2), [512, 768]);
  assert.deepEqual(matchedCanvas(9, 16, ltx2), [704, 1216]);
  // Extreme ratios still land on a preset: the picture is cropped, never refused.
  assert.deepEqual(matchedCanvas(2000, 400, ti2v), [1280, 704]);
  assert.equal(matchedCanvas(0, 100, ti2v), null);
});

test("nearestPresetCanvas ignores an empty list", () => {
  assert.equal(nearestPresetCanvas(16, 9, []), null);
});

test("keyframe support: per-frame fields win, the legacy flag covers old backends", () => {
  // Wan2.2 TI2V 5B / LTX-2 on a new backend: the start frame only.
  assert.deepEqual(
    keyframeSupport({
      supports_keyframes: false,
      supports_first_frame: true,
      supports_last_frame: false,
    }),
    { first: true, last: false },
  );
  assert.deepEqual(
    keyframeSupport({ supports_keyframes: true, supports_first_frame: true, supports_last_frame: true }),
    { first: true, last: true },
  );
  // An older backend only reports supports_keyframes (MiniMax-H3's pair).
  assert.deepEqual(keyframeSupport({ supports_keyframes: true }), { first: true, last: true });
  assert.deepEqual(keyframeSupport({ supports_keyframes: false }), { first: false, last: false });
  assert.deepEqual(
    keyframeSupport({ supports_keyframes: false, supports_first_frame: false, supports_last_frame: false }),
    { first: false, last: false },
  );
  assert.deepEqual(keyframeSupport(null), { first: false, last: false });
});
