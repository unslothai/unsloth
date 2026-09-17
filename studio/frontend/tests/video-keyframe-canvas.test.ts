// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import { matchedCanvas } from "../src/features/video/keyframe-canvas.ts";

// What /video/status reports for MiniMax-H3: the released checkpoint's canvas rule.
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
  // The pairs the backend's own test pins, so a divergence in either direction is caught.
  assert.deepEqual(matchedCanvas(1920, 1080, h3), [1344, 768]);
  assert.deepEqual(matchedCanvas(1080, 1920, h3), [768, 1344]);
  assert.deepEqual(matchedCanvas(1000, 1000, h3), [768, 768]);
  assert.deepEqual(matchedCanvas(1024, 768, h3), [1024, 768]);
});

test("reads the ratio only, so the source's own scale never matters", () => {
  assert.deepEqual(matchedCanvas(160, 90, h3), matchedCanvas(3840, 2160, h3));
});

test("previews nothing where the backend would refuse or has no rule", () => {
  // Outside the trained 1:4 - 4:1 band the backend raises, so offering a size would be a lie.
  assert.equal(matchedCanvas(2000, 400, h3), null);
  assert.equal(matchedCanvas(400, 2000, h3), null);
  assert.equal(matchedCanvas(0, 100, h3), null);
  // A family that declares no canvas rule takes no keyframes either.
  assert.equal(
    matchedCanvas(1920, 1080, { ...h3, canvas_short_edge: null, canvas_max_pixels: null }),
    null,
  );
  assert.equal(matchedCanvas(1920, 1080, null), null);
});
