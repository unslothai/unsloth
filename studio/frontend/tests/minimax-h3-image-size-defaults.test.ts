// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  h3ImageSourceSize,
  resolveEditSize,
} from "../src/features/images/edit-conditioning.ts";
import { defaultSizeFor } from "../src/features/images/image-generation-defaults.ts";
import { sizeLimitsFrom } from "../src/features/images/image-size.ts";

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
