// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  familyPresetsCopy,
  h3ImageSourceSize,
  matchSizeChoosable,
  resolveEditSize,
} from "../src/features/images/edit-conditioning.ts";
import { defaultSizeFor } from "../src/features/images/image-generation-defaults.ts";
import { sizeLimitsFrom } from "../src/features/images/image-size.ts";

import { readSrc } from "./helpers/kit.ts";

const H3_LIMITS = sizeLimitsFrom({
  dimension_multiple: 32,
  max_output_side: 6272,
  max_output_pixels: 20 * 1024 * 1024,
} as never);

test("MiniMax-H3-Image picks start text-to-image at 2048x2048; other families keep the current size", () => {
  assert.deepEqual(defaultSizeFor("MiniMaxAI/MiniMax-H3-Image"), {
    width: 2048,
    height: 2048,
  });
  assert.deepEqual(defaultSizeFor("minimax-h3-image"), {
    width: 2048,
    height: 2048,
  });
  assert.equal(defaultSizeFor("black-forest-labs/FLUX.1-dev"), null);
  assert.equal(defaultSizeFor("MiniMaxAI/MiniMax-H3"), null);
});

test("the H3-Image edit size is the reference edit workflow's: Image 1 at 4 MP, sides cut to the 32 grid", () => {
  // the backend's reference_edit_size (Python round, halves to even; then each side floored to 32)
  assert.deepEqual(h3ImageSourceSize(1475, 981), { width: 2496, height: 1664 });
  assert.deepEqual(h3ImageSourceSize(600, 400), { width: 2496, height: 1664 });
  assert.deepEqual(h3ImageSourceSize(427, 640), { width: 1664, height: 2496 });
  assert.deepEqual(h3ImageSourceSize(1203, 1203), {
    width: 2048,
    height: 2048,
  });
  // already 4 MP on the grid: unchanged
  assert.deepEqual(h3ImageSourceSize(2048, 2048), {
    width: 2048,
    height: 2048,
  });
  // 2367.59 rounds up onto the grid (2368), where flooring first would cut it to 2336
  assert.deepEqual(h3ImageSourceSize(1001, 749), {
    width: 2368,
    height: 1760,
  });
  assert.deepEqual(h3ImageSourceSize(5000, 3000), {
    width: 2624,
    height: 1568,
  });
  const src = { width: 1475, height: 981 };
  const custom = { width: 1024, height: 1024 };
  assert.deepEqual(
    resolveEditSize("source", src, 1024, custom, H3_LIMITS, "minimax-h3-image"),
    {
      width: 2496,
      height: 1664,
    },
  );
  // other families keep the matched-area rule; a custom size is untouched
  assert.notDeepEqual(
    resolveEditSize("source", src, 1024, custom, H3_LIMITS, "flux.1-kontext"),
    {
      width: 2496,
      height: 1664,
    },
  );
  assert.deepEqual(
    resolveEditSize("custom", src, 1024, custom, H3_LIMITS, "minimax-h3-image"),
    custom,
  );
});

test("a panoramic H3-Image source past the side limit keeps its aspect ratio", () => {
  const custom = { width: 1024, height: 1024 };
  // 8000x1000 is 5792x704 at 4 MP: inside the 6272 side limit, still about 8:1
  const size = resolveEditSize(
    "source",
    { width: 8000, height: 1000 },
    1024,
    custom,
    H3_LIMITS,
    "minimax-h3-image",
  );
  assert.ok(size.width <= H3_LIMITS.maxSide && size.width % 32 === 0 && size.height % 32 === 0);
  // within one 32 px grid step of 8:1
  assert.ok(Math.abs(size.height - size.width / 8) <= 32, `${size.width}x${size.height}`);
  assert.ok(size.width / size.height > 7.5, `${size.width}x${size.height}`);
});

test("a very thin H3-Image source keeps the minimum output side", () => {
  const size = resolveEditSize(
    "source",
    { width: 8000, height: 250 },
    1024,
    { width: 1024, height: 1024 },
    H3_LIMITS,
    "minimax-h3-image",
  );
  assert.ok(Math.min(size.width, size.height) >= 256, `${size.width}x${size.height}`);
  assert.ok(size.width <= H3_LIMITS.maxSide && size.width * size.height <= H3_LIMITS.maxPixels);
});

test("Match Image 1 shows no size picker for H3-Image, whose match size ignores it", () => {
  // the picker's value never reaches an H3-Image request: a 2048 Image 1 renders 2048x2048 at any pick
  for (const pick of [512, 1024, 2048]) {
    assert.deepEqual(
      resolveEditSize(
        "source",
        { width: 2048, height: 2048 },
        pick,
        { width: 1024, height: 1024 },
        H3_LIMITS,
        "minimax-h3-image",
      ),
      { width: 2048, height: 2048 },
    );
  }
  assert.equal(matchSizeChoosable("minimax-h3-image"), false);
  assert.equal(matchSizeChoosable("flux.1-kontext"), true);
  assert.equal(matchSizeChoosable(null), true);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /editSizing === "source" && matchSizeChoosable\(status\?\.family\) && \(\s*<Select\s+value=\{String\(matchResolution\)\}/,
  );
});

test("the H3-Image size hint does not claim a 4 MP floor that is opt-in since the reference defaults", () => {
  const { hint } = familyPresetsCopy("minimax-h3-image");
  assert.doesNotMatch(hint, /always renders at least 4 MP/);
  assert.match(hint, /16 MP/);
});
