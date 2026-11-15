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

test("the H3-Image edit size is Image 1's reference size: short edge clamped to 1024..2048, sides on the 32 grid", () => {
  // the backend's size_short_edge (Python round, halves to even)
  assert.deepEqual(h3ImageSourceSize(1475, 981), { width: 1536, height: 1024 });
  assert.deepEqual(h3ImageSourceSize(1500, 1200), {
    width: 1504,
    height: 1216,
  });
  assert.deepEqual(h3ImageSourceSize(5000, 3000), {
    width: 3424,
    height: 2048,
  });
  assert.deepEqual(h3ImageSourceSize(1203, 1203), {
    width: 1216,
    height: 1216,
  });
  assert.deepEqual(h3ImageSourceSize(4097, 1023), {
    width: 4096,
    height: 1024,
  });
  const src = { width: 1475, height: 981 };
  const custom = { width: 1024, height: 1024 };
  assert.deepEqual(
    resolveEditSize("source", src, 1024, custom, H3_LIMITS, "minimax-h3-image"),
    {
      width: 1536,
      height: 1024,
    },
  );
  // other families keep the matched-area rule; a custom size is untouched
  assert.notDeepEqual(
    resolveEditSize("source", src, 1024, custom, H3_LIMITS, "flux.1-kontext"),
    {
      width: 1536,
      height: 1024,
    },
  );
  assert.deepEqual(
    resolveEditSize("custom", src, 1024, custom, H3_LIMITS, "minimax-h3-image"),
    custom,
  );
});

test("a panoramic H3-Image source past the side limit keeps its aspect ratio", () => {
  const custom = { width: 1024, height: 1024 };
  // 8000x1000 is 8192x1024 natively, over the 6272 side limit: scaled as a pair, still 8:1
  const size = resolveEditSize(
    "source",
    { width: 8000, height: 1000 },
    1024,
    custom,
    H3_LIMITS,
    "minimax-h3-image",
  );
  assert.ok(size.width <= H3_LIMITS.maxSide && size.width % 32 === 0 && size.height % 32 === 0);
  // within one 32 px grid step of 8:1 (the old per-side clamp gave 6272x1024, 6.125:1)
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
