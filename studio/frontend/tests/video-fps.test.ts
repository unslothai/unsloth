// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cosmos3 declares the frame rates NVIDIA lists (Edge 12-30, Nano 10/16/24/30); every other family keeps a
// fixed rate, so its page and requests must not change.

import assert from "node:assert/strict";
import test from "node:test";

import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import {
  keptNumFrames,
  pickedFpsFor,
  resolveVideoFps,
  restoredClipDuration,
  videoDurationOptions,
  videoFpsOptions,
} from "../src/features/video/video-fps.ts";
import { isValidVideoSize, videoSizeLimits } from "../src/features/video/video-size.ts";

const P256: Array<[number, number]> = [
  [320, 192],
  [192, 320],
  [256, 256],
  [320, 256],
  [256, 320],
];

function defaults(extra: Partial<VideoGenerationDefaults> = {}): VideoGenerationDefaults {
  return {
    steps: 20,
    guidance: 6,
    num_frames: 121,
    fps: 24,
    frame_step: 4,
    frame_offset: 1,
    duration_presets: [3, 5, 6],
    resolution_multiple: 16,
    resolution_presets: [[832, 480], [480, 832], [640, 640], [736, 544], [544, 736], ...P256],
    ...extra,
  };
}

// What the Video page computed before the picker; a family without a window must still get exactly this.
function previousDurationOptions(targets: number[], fps: number, step: number, offset: number) {
  const seen = new Set<number>();
  const out: Array<{ frames: number; seconds: number }> = [];
  for (const t of targets) {
    const k = Math.max(1, Math.round((t * fps - offset) / step));
    const frames = k * step + offset;
    if (seen.has(frames)) continue;
    seen.add(frames);
    out.push({ frames, seconds: frames / fps });
  }
  return out;
}

test("a family with no fps options keeps its fixed rate whatever was picked", () => {
  const fixed = defaults({ fps: 25, fps_range: null, fps_choices: [] });
  assert.equal(videoFpsOptions(fixed), null);
  assert.equal(resolveVideoFps(fixed, null), 25);
  assert.equal(resolveVideoFps(fixed, 12), 25);
  // An older backend sends neither field.
  assert.equal(videoFpsOptions(defaults()), null);
  assert.equal(resolveVideoFps(null, 30), 24);
});

test("Edge offers every rate from 12 to 30 and Nano NVIDIA's four", () => {
  const edge = defaults({ fps_range: [12, 30], fps_choices: [] });
  assert.deepEqual(videoFpsOptions(edge), Array.from({ length: 19 }, (_, i) => 12 + i));
  assert.equal(resolveVideoFps(edge, null), 24);
  assert.equal(resolveVideoFps(edge, 12), 12);
  assert.equal(resolveVideoFps(edge, 30), 30);
  assert.equal(resolveVideoFps(edge, 31), 24);
  const nano = defaults({ fps_range: null, fps_choices: [30, 10, 24, 16] });
  assert.deepEqual(videoFpsOptions(nano), [10, 16, 24, 30]);
  assert.equal(resolveVideoFps(nano, 16), 16);
  assert.equal(resolveVideoFps(nano, 12), 24);
});

test("durations without a frame window are what the page computed before", () => {
  for (const [targets, fps, step, offset] of [
    [[1, 2, 3, 5], 24, 8, 1],
    [[1, 2, 3, 5], 25, 8, 1],
    [[5, 10, 15], 25, 17, 5],
    [[2, 5, 7.875], 24, 4, 1],
  ] as Array<[number[], number, number, number]>) {
    assert.deepEqual(
      videoDurationOptions(targets, fps, step, offset),
      previousDurationOptions(targets, fps, step, offset),
    );
    // A window the presets already sit inside changes nothing either.
    assert.deepEqual(
      videoDurationOptions(targets, fps, step, offset, offset, 2000),
      previousDurationOptions(targets, fps, step, offset),
    );
  }
});

test("Edge durations follow the picked rate and stay inside 53-149 frames", () => {
  const at = (fps: number) => videoDurationOptions([3, 5, 6], fps, 4, 1, 53, 149);
  assert.deepEqual(
    at(24).map((o) => o.frames),
    [73, 121, 145],
  );
  assert.deepEqual(
    at(12).map((o) => o.frames),
    [53, 61, 73],
  );
  assert.deepEqual(
    at(30).map((o) => o.frames),
    [89, 149],
  );
  for (const fps of [12, 30]) {
    for (const { frames, seconds } of at(fps)) {
      assert.equal((frames - 1) % 4, 0);
      assert.equal(seconds, frames / fps);
    }
  }
});

test("the 256p presets and NVIDIA's 448x256 Jetson size pass the size check", () => {
  const edge = defaults({
    custom_size_min_pixels: 192 * 320,
    custom_size_max_pixels: 640 * 640,
    custom_size_max_aspect: 4,
  });
  const limits = videoSizeLimits(edge, edge.resolution_presets);
  for (const [w, h] of [...P256, [448, 256], [256, 448]] as Array<[number, number]>) {
    assert.ok(isValidVideoSize(w, h, limits), `${w}x${h}`);
  }
  assert.ok(!isValidVideoSize(176, 176, limits));
});

test("a restored clip takes its frame count only when the loaded family's window offers it", () => {
  const edgeAt24 = videoDurationOptions([3, 5, 6], 24, 4, 1, 53, 149);
  // An Edge clip restores as it was.
  assert.deepEqual(restoredClipDuration(edgeAt24, 121, 24), { frames: 121, seconds: 121 / 24 });
  // A 189-frame Nano clip at 24 fps maps to Edge's nearest option, never 189.
  assert.deepEqual(restoredClipDuration(edgeAt24, 189, 24), { frames: 145, seconds: 145 / 24 });
  const edgeAt12 = videoDurationOptions([3, 5, 6], 12, 4, 1, 53, 149);
  assert.deepEqual(restoredClipDuration(edgeAt12, 25, 12), { frames: 53, seconds: 53 / 12 });
  assert.equal(restoredClipDuration([], 121, 24), null);
});

test("a new frame rate keeps the picked length, not a frame count that is another length at that rate", () => {
  // Nano at 24 fps, 2 s picked (49 frames). 49 is also an option at 10 fps, where it lasts 4.9 s.
  const nanoAt10 = videoDurationOptions([2, 5, 7.875], 10, 4, 1, 5, 397);
  assert.ok(nanoAt10.some((o) => o.frames === 49));
  assert.equal(keptNumFrames(nanoAt10, 49, 49 / 24, true), 21);
  // Edge at 24 fps, 3 s (73 frames); at 12 fps 73 is 6.1 s, so the nearest to 3 s (53, 4.4 s) wins.
  const edgeAt12 = videoDurationOptions([3, 5, 6], 12, 4, 1, 53, 149);
  assert.equal(keptNumFrames(edgeAt12, 73, 73 / 24, true), 53);
  // Same rate: a count still on offer stays, one that is not snaps to the picked length.
  const edgeAt24 = videoDurationOptions([3, 5, 6], 24, 4, 1, 53, 149);
  assert.equal(keptNumFrames(edgeAt24, 145, 145 / 24, false), 145);
  assert.equal(keptNumFrames(edgeAt24, 61, 5, false), 121);
  assert.equal(keptNumFrames([], 61, 5, true), 61);
});

test("a picked rate applies only to the family it was picked on", () => {
  const pick = { family: "cosmos3-edge", fps: 30 };
  assert.equal(pickedFpsFor(pick, "cosmos3-edge"), 30);
  // Edge at 30 fps to Nano: Nano's defaults are seeded at its own 24 fps, not at Edge's pick.
  assert.equal(pickedFpsFor(pick, "cosmos3-nano"), null);
  const edge = defaults({ fps_range: [12, 30] });
  assert.equal(resolveVideoFps(edge, pickedFpsFor({ family: "cosmos3-nano", fps: 16 }, "cosmos3-edge")), 24);
  assert.equal(pickedFpsFor(null, "cosmos3-edge"), null);
});
