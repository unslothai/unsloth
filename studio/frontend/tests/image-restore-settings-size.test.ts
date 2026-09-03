// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Restore settings copied a gallery record's raw size into the Create form. Image-conditioned
// workflows derive that size from the upload, so it could fall outside the 256..2048 that
// ImageGenerationPresetParams forbids -- 422ing every debounced preset PUT for the rest of the session.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MAX_DIM,
  MIN_DIM,
  restorableSize,
} from "../src/features/images/image-size.ts";

const withinSchema = ({ width, height }: { width: number; height: number }) =>
  width >= MIN_DIM &&
  width <= MAX_DIM &&
  height >= MIN_DIM &&
  height <= MAX_DIM &&
  width % 16 === 0 &&
  height % 16 === 0;

const ratio = ({ width, height }: { width: number; height: number }) =>
  width / height;

test("a size the gallery can record always restores inside the preset schema", () => {
  const recorded = [
    [4032, 3024], // Edit on a phone photo: no clamp, only _snap_to_multiple
    [1024, 208], // Transform a 1920x400 source with the sliders at 1024
    [4096, 4096], // decode_b64_image's ceiling
    [2048, 256], // already legal
    [1024, 1024],
  ];
  for (const [width, height] of recorded) {
    const restored = restorableSize(width, height);
    assert.ok(
      withinSchema(restored),
      `${width}x${height} restored to ${restored.width}x${restored.height}`,
    );
  }
  assert.ok(!withinSchema({ width: 4032, height: 3024 }));
  assert.ok(!withinSchema({ width: 1024, height: 208 }));
});

test("a legal size is restored unchanged", () => {
  assert.deepEqual(restorableSize(2048, 256), { width: 2048, height: 256 });
  assert.deepEqual(restorableSize(1024, 1024), { width: 1024, height: 1024 });
});

test("the recipe's aspect ratio survives the scale into range", () => {
  for (const [width, height] of [
    [4032, 3024],
    [1024, 208],
    [4096, 2048],
  ]) {
    const restored = restorableSize(width, height);
    assert.ok(
      Math.abs(ratio(restored) - width / height) < 0.05,
      `${width}x${height} became ${restored.width}x${restored.height}`,
    );
  }
});

test("a degenerate record still produces a usable size", () => {
  for (const [width, height] of [
    [0, 0],
    [Number.NaN, 512],
    [-1024, 512],
  ]) {
    assert.ok(withinSchema(restorableSize(width, height)));
  }
});

test("restoreSettings puts the record through restorableSize", () => {
  const source = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("const restoreSettings = useCallback(");
  assert.ok(start > 0, "restoreSettings not found");
  const body = source.slice(start, source.indexOf("}, [", start));
  assert.match(body, /restorableSize\(image\.width, image\.height\)/);
  assert.doesNotMatch(
    body,
    /setWidth\(image\.width\)|setHeight\(image\.height\)/,
    "the raw recorded size must not reach the form",
  );
});
