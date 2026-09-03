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
  snapDim,
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
  assert.match(body, /restorableSize\(image\.width, image\.height, image\.workflow\)/);
  assert.doesNotMatch(
    body,
    /setWidth\(image\.width\)|setHeight\(image\.height\)/,
    "the raw recorded size must not reach the form",
  );
});

// Transform bounds the upload by the requested size instead of taking it literally, so the
// restored recipe only reproduces the record when the in-range side is left where it was.

/** _fit_within + _snap_to_multiple from studio/backend/core/inference/diffusion.py. */
const transform = (
  source: [number, number],
  reqW: number,
  reqH: number,
): [number, number] => {
  const [w, h] = source;
  const bw = Math.min(2048, reqW);
  const bh = Math.min(2048, reqH);
  const [fw, fh] =
    w <= bw && h <= bh
      ? [w, h]
      : (() => {
          const s = Math.min(bw / w, bh / h);
          return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
        })();
  return [
    Math.max(16, Math.round(fw / 16) * 16),
    Math.max(16, Math.round(fh / 16) * 16),
  ];
};

test("restoring a Transform record reproduces it exactly as well as main did", () => {
  // Not an absolute round-trip assertion: Transform is already not perfectly self-reproducing on
  // main, because _snap_to_multiple rounds a side and the tightened box changes the next run
  // (3000x500 at 2048 records 2048x336 and re-runs to 2016x336, with or without this change).
  // What this change owes is that it never reproduces WORSE than main, while making the recipe
  // savable -- so main is the baseline, not an ideal.
  for (const source of [
    [1920, 400],
    [1920, 320],
    [3000, 500],
    [4000, 3000],
  ] as Array<[number, number]>) {
    for (const requested of [512, 768, 1024, 2048]) {
      const recorded = transform(source, requested, requested);
      // main puts the raw record in the form; Generate then snaps each side on its own.
      const onMain = transform(
        source,
        snapDim(recorded[0]),
        snapDim(recorded[1]),
      );
      const restored = restorableSize(recorded[0], recorded[1], "img2img");
      const onThisBranch = transform(
        source,
        snapDim(restored.width),
        snapDim(restored.height),
      );
      assert.ok(
        withinSchema(restored),
        `${recorded[0]}x${recorded[1]} restored to ${restored.width}x${restored.height}`,
      );
      assert.deepEqual(
        onThisBranch,
        onMain,
        `source ${source[0]}x${source[1]} at ${requested}: recorded ${recorded[0]}x${recorded[1]}, ` +
          `restored ${restored.width}x${restored.height}, re-ran to ${onThisBranch[0]}x${onThisBranch[1]} ` +
          `where main re-ran to ${onMain[0]}x${onMain[1]}`,
      );
    }
  }
});

test("scaling a Transform record as a pair would NOT reproduce it", () => {
  // Guards the reason the img2img branch exists: the shared scale is right for every other
  // workflow and wrong for this one, so a later simplification that drops it has to fail here.
  const recorded = transform([1920, 400], 1024, 1024);
  assert.deepEqual(recorded, [1024, 208]);
  const asTransform = restorableSize(1024, 208, "img2img");
  const asPair = restorableSize(1024, 208, "edit");
  assert.deepEqual(asTransform, { width: 1024, height: 256 });
  assert.deepEqual(asPair, { width: 1264, height: 256 });
  assert.deepEqual(transform([1920, 400], asTransform.width, asTransform.height), recorded);
  assert.notDeepEqual(transform([1920, 400], asPair.width, asPair.height), recorded);
});

test("every other workflow still keeps the recipe's shape", () => {
  // The headline case: an Edit of a phone photo must not come back square.
  for (const workflow of [null, undefined, "edit", "inpaint", "upscale", "reference"]) {
    const restored = restorableSize(4032, 3024, workflow);
    assert.deepEqual(
      restored,
      { width: 2048, height: 1536 },
      `workflow ${String(workflow)} restored 4032x3024 to ${restored.width}x${restored.height}`,
    );
  }
  // Per-side clamping, which img2img wants, would square this one up.
  assert.deepEqual(restorableSize(4032, 3024, "img2img"), { width: 2048, height: 2048 });
});

test("a Transform record already inside the schema is untouched", () => {
  assert.deepEqual(restorableSize(1024, 512, "img2img"), { width: 1024, height: 512 });
  assert.deepEqual(restorableSize(2048, 256, "img2img"), { width: 2048, height: 256 });
});
