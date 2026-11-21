// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import {
  defaultVideoResolution,
  familyFirstRenderResolution,
  keyframeSelectsMatchSource,
  keyframeSupport,
  matchedCanvas,
  nearestPresetCanvas,
  requestCanvas,
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

// LTX-2.5's rule: 704 px short edge, 1280x704 cap, 64 px units.
const ltx25: VideoGenerationDefaults = {
  ...h3,
  resolution_multiple: 64,
  resolution_presets: [[1280, 704]],
  canvas_short_edge: 704,
  canvas_max_pixels: 1280 * 704,
};

test("snapping never leaves the pixel cap, and matches the backend at its edges", () => {
  // Rounding both sides up used to give 576x1664 here, 6% past LTX-2.5's cap.
  assert.deepEqual(matchedCanvas(0.32841, 1, ltx25), [512, 1664]);
  assert.deepEqual(matchedCanvas(0.32841, 1, h3), [576, 1760]);
  assert.deepEqual(matchedCanvas(0.25, 1, ltx25), [448, 1920]);
  assert.deepEqual(matchedCanvas(3.045, 1, ltx25), [1664, 512]);
  for (const defaults of [ltx25, h3]) {
    const multiple = defaults.resolution_multiple ?? 1;
    for (let i = 2500; i <= 40000; i += 7) {
      const canvas = matchedCanvas(i / 10000, 1, defaults);
      assert.ok(canvas);
      const [width, height] = canvas;
      assert.ok(width * height <= (defaults.canvas_max_pixels ?? 0), `${i}: ${width}x${height}`);
      assert.equal(width % multiple, 0);
      assert.equal(height % multiple, 0);
    }
  }
});

test("a family whose keyframes keep the canvas does not switch to Match source on its own", () => {
  assert.equal(keyframeSelectsMatchSource({ ...h3, keyframe_keeps_canvas: true }), false);
  assert.equal(keyframeSelectsMatchSource({ ...h3, keyframe_keeps_canvas: false }), true);
  // older backends never sent the flag: the old behaviour
  assert.equal(keyframeSelectsMatchSource(h3), true);
  assert.equal(keyframeSelectsMatchSource(null), true);
});

test("Match source sends the matched size where an omitted size means the default canvas", () => {
  assert.deepEqual(requestCanvas(false, [1024, 768], [864, 480], true), { width: 864, height: 480 });
  assert.deepEqual(requestCanvas(true, [1024, 768], [864, 480], true), { width: 1024, height: 768 });
  assert.deepEqual(requestCanvas(true, [1024, 768], [864, 480], false), {
    width: undefined,
    height: undefined,
  });
  assert.deepEqual(requestCanvas(false, null, [1344, 768], false), { width: 1344, height: 768 });
});

test("the Default preset takes the family's default size only when it is a preset", () => {
  const presets: Array<[number, number]> = [
    [1344, 768],
    [864, 480],
  ];
  assert.deepEqual(defaultVideoResolution({ default_resolution: [864, 480] }, presets), [864, 480]);
  assert.deepEqual(defaultVideoResolution({ default_resolution: [640, 640] }, presets), [1344, 768]);
  assert.deepEqual(defaultVideoResolution({}, presets), [1344, 768]);
  assert.deepEqual(defaultVideoResolution(null, []), null);
});

test("a family's declared first-render size is used only when declared and a preset", () => {
  const presets: Array<[number, number]> = [
    [1344, 768],
    [864, 480],
  ];
  assert.deepEqual(familyFirstRenderResolution({ default_resolution: [864, 480] }, presets), [864, 480]);
  assert.equal(familyFirstRenderResolution({ default_resolution: [640, 640] }, presets), null);
  // a family that declares none keeps the size the page holds (no fallback to presets[0])
  assert.equal(familyFirstRenderResolution({}, presets), null);
  assert.equal(familyFirstRenderResolution(null, presets), null);
});

test("a family that declares no first size starts at its first preset when the held size is untrained", () => {
  // Wan2.1 1.3B's presets: the page's 768x512 fallback would otherwise stay (or be fitted to an untrained size).
  const presets: Array<[number, number]> = [
    [832, 480],
    [480, 832],
  ];
  assert.deepEqual(familyFirstRenderResolution({}, presets, [768, 512]), [832, 480]);
  assert.deepEqual(familyFirstRenderResolution(null, presets, [864, 480]), [832, 480]);
  // a held trained size stays
  assert.equal(familyFirstRenderResolution({}, presets, [480, 832]), null);
  // a declared size still wins over the held one
  assert.deepEqual(
    familyFirstRenderResolution({ default_resolution: [480, 832] }, presets, [768, 512]),
    [480, 832],
  );
  assert.equal(familyFirstRenderResolution({}, [], [768, 512]), null);
});

test("the page applies the first-render size when the family loads, behind the model-defaults gate", async () => {
  const page = await readFile(new URL("../src/features/video/video-page.tsx", import.meta.url), "utf8");
  const start = page.indexOf("const familyChanged = loadedFamily !== prevFamilyRef.current;");
  assert.ok(start >= 0);
  const effect = page.slice(start, page.indexOf("keptNumFrames(", start));
  const gate = effect.indexOf(
    "if (familyChanged && loadedFamily && applyFamilyDefault && !resolutionChosen.current) {",
  );
  assert.ok(gate >= 0, "the clip-length gate, and never over a size the user chose");
  const apply = effect.slice(gate);
  assert.match(
    apply,
    /familyFirstRenderResolution\(\s*status\?\.defaults,\s*resolutionPresets,\s*resolutionIntentRef\.current,\s*\)/,
  );
  assert.match(apply, /setResolutionIntent\(firstSize\)/);
  // before the clip-length branch, which returns early
  assert.ok(gate < effect.indexOf("familyDefaultFrames && applyFamilyDefault"));
  // a picked size and a restored clip's size both count as chosen, so a reload or family switch keeps them
  const applySize = page.slice(page.indexOf("const applySize = ("), page.indexOf("const applySize = (") + 400);
  assert.match(applySize, /resolutionChosen\.current = true;/);
  const restore = page.slice(page.indexOf("const restoreSettings = useCallback("));
  assert.match(restore.slice(0, restore.indexOf("setResolutionIntent(size)")), /resolutionChosen\.current = true;/);
});
