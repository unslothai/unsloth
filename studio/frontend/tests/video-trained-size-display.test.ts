// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import {
  lockedVideoSize,
  trainedSizeValue,
  videoSizeAspect,
  videoSizeLimits,
} from "../src/features/video/video-size.ts";

// Same table as components/media-size-controls.tsx ASPECT_RATIOS.
const RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};

// Cosmos3 Edge's presets (480p tier and part of the 256p tier).
const presets: Array<[number, number]> = [
  [832, 480],
  [480, 832],
  [640, 640],
  [736, 544],
  [544, 736],
  [320, 192],
];
const defaults = {
  resolution_multiple: 16,
  resolution_presets: presets,
  custom_size_min_pixels: 61440,
  custom_size_max_pixels: 399360,
  custom_size_max_aspect: 4,
} as unknown as VideoGenerationDefaults;
const limits = videoSizeLimits(defaults, presets);

test("a trained size shows the ratio it was cut to, not Custom", () => {
  assert.deepEqual(videoSizeAspect(832, 480, limits, RATIOS), { key: "16:9", portrait: false });
  assert.deepEqual(videoSizeAspect(480, 832, limits, RATIOS), { key: "16:9", portrait: true });
  assert.deepEqual(videoSizeAspect(736, 544, limits, RATIOS), { key: "4:3", portrait: false });
  assert.deepEqual(videoSizeAspect(640, 640, limits, RATIOS), { key: "1:1", portrait: false });
  // 5:3 is more than 3% from every named ratio
  assert.equal(videoSizeAspect(320, 192, limits, RATIOS).key, "custom");
});

test("an untrained size keeps the exact ratio rule", () => {
  // 816x480 is a valid custom size 1.7 (4% off 16:9): Custom, as before
  assert.equal(videoSizeAspect(816, 480, limits, RATIOS).key, "custom");
  assert.equal(videoSizeAspect(768, 432, limits, RATIOS).key, "16:9");
  assert.equal(videoSizeAspect(704, 544, limits, RATIOS).key, "custom");
});

test("the Trained sizes select shows the trained size in use", () => {
  assert.equal(trainedSizeValue([832, 480], limits), "832x480");
  assert.equal(trainedSizeValue([720, 544], limits), "");
  // Match source sizes the canvas from the keyframe, so no preset is in use
  assert.equal(trainedSizeValue([832, 480], limits, true), "");
});

test("the page wires both controls to the shown size", async () => {
  const page = await readFile(new URL("../src/features/video/video-page.tsx", import.meta.url), "utf8");
  assert.match(page, /value=\{trainedSizeValue\(videoSize, sizeLimits, sourceSized\)\}/);
  assert.doesNotMatch(page, /<Select\s+value=""\s+onValueChange=\{\(v\) => \{\s+const preset = resolutionPresets/);
  // every size the page sets picks its ratio through the trained-size rule
  assert.equal((page.match(/matchAspect\(size\[0\], size\[1\]\)/g) ?? []).length, 0);
  assert.equal((page.match(/videoSizeAspect\([^)]*sizeLimits, ASPECT_RATIOS\)/g) ?? []).length, 4);
});

test("a locked ratio keeps a trained size when one side is edited", () => {
  const wide: [number, number] = [16, 9];
  // Re-picking the held side of 832x480 stays on it, not on 832x464 / 848x480.
  assert.deepEqual(lockedVideoSize("width", 832, wide, false, limits), [832, 480]);
  assert.deepEqual(lockedVideoSize("height", 480, wide, false, limits), [832, 480]);
  assert.deepEqual(lockedVideoSize("height", 832, wide, true, limits), [480, 832]);
  // Wan2.2 TI2V-5B: 1280x704 is its 16:9 preset.
  const ti2v: Array<[number, number]> = [
    [1280, 704],
    [704, 1280],
  ];
  const ti2vLimits = videoSizeLimits(
    { resolution_multiple: 32, resolution_presets: ti2v } as unknown as VideoGenerationDefaults,
    ti2v,
  );
  assert.deepEqual(lockedVideoSize("width", 1280, wide, false, ti2vLimits), [1280, 704]);
  // No preset with that side: the exact ratio, as before.
  assert.deepEqual(lockedVideoSize("width", 960, wide, false, limits), [960, 544]);
});

test("the page's width and height edits go through the trained-size lock", async () => {
  const page = await readFile(new URL("../src/features/video/video-page.tsx", import.meta.url), "utf8");
  assert.match(page, /lockedVideoSize\("width", v, lockedRatio, portrait, sizeLimits\)/);
  assert.match(page, /lockedVideoSize\("height", v, lockedRatio, portrait, sizeLimits\)/);
});
