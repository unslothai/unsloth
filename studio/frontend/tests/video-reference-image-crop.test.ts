// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type CropRasterCanvas,
  MAX_REFERENCE_IMAGE_DATA_URL_LENGTH,
  applyReferenceImageCrop,
  clampCropRect,
  createCropImageLoadGate,
  createReferenceImageEditorActions,
  cropRectFromPoints,
  displayPointToSource,
  moveCropRect,
  rasterizeReferenceImageCrop,
  referenceCropExportSize,
  referenceImageDataUrlError,
  referenceImageDataUrls,
  stageReferenceImage,
} from "../src/features/video/reference-image-crop.ts";

test("crop rectangles stay within natural image pixels in either drag direction", () => {
  assert.deepEqual(
    cropRectFromPoints(
      { x: 900, y: 700 },
      { x: -20, y: 100 },
      { width: 800, height: 600 },
    ),
    { x: 0, y: 100, width: 800, height: 500 },
  );
  assert.deepEqual(
    clampCropRect(
      { x: 790.4, y: 590.2, width: 100, height: 100 },
      { width: 800, height: 600 },
    ),
    { x: 790, y: 590, width: 10, height: 10 },
  );
});

test("landscape preview coordinates map to the same source rectangle", () => {
  const start = displayPointToSource(
    { x: 40, y: 20 },
    { width: 400, height: 200 },
    { width: 4000, height: 2000 },
  );
  const end = displayPointToSource(
    { x: 240, y: 120 },
    { width: 400, height: 200 },
    { width: 4000, height: 2000 },
  );
  assert.deepEqual(
    cropRectFromPoints(start, end, { width: 4000, height: 2000 }),
    { x: 400, y: 200, width: 2000, height: 1000 },
  );
});

test("portrait preview coordinates keep the decoded natural orientation", () => {
  const start = displayPointToSource(
    { x: 25, y: 100 },
    { width: 200, height: 500 },
    { width: 1200, height: 3000 },
  );
  const end = displayPointToSource(
    { x: 150, y: 400 },
    { width: 200, height: 500 },
    { width: 1200, height: 3000 },
  );
  assert.deepEqual(
    cropRectFromPoints(start, end, { width: 1200, height: 3000 }),
    { x: 150, y: 600, width: 750, height: 1800 },
  );
});

test("fractional display edges include every selected source pixel", () => {
  const start = displayPointToSource(
    { x: 1.2, y: 2.2 },
    { width: 10, height: 10 },
    { width: 100, height: 100 },
  );
  const end = displayPointToSource(
    { x: 4.3, y: 6.3 },
    { width: 10, height: 10 },
    { width: 100, height: 100 },
  );
  assert.deepEqual(
    cropRectFromPoints(start, end, { width: 100, height: 100 }),
    { x: 12, y: 22, width: 31, height: 41 },
  );
});

test("moving a crop preserves its size and keeps it inside the source", () => {
  assert.deepEqual(
    moveCropRect(
      { x: 200, y: 100, width: 400, height: 300 },
      { x: 500, y: -200 },
      { width: 800, height: 600 },
    ),
    { x: 400, y: 0, width: 400, height: 300 },
  );
  assert.deepEqual(
    moveCropRect(
      { x: 200, y: 100, width: 400, height: 300 },
      { x: -50.4, y: 25.6 },
      { width: 800, height: 600 },
    ),
    { x: 150, y: 126, width: 400, height: 300 },
  );
});

test("raster output uses the selected source pixels without upscaling", () => {
  const source = {} as CanvasImageSource;
  const calls: unknown[][] = [];
  const canvas: CropRasterCanvas = {
    width: 0,
    height: 0,
    getContext: () => ({
      drawImage: (...args: unknown[]) => calls.push(args),
    }),
    toDataURL: (type) => {
      assert.equal(type, "image/png");
      return "data:image/png;base64,CROP";
    },
  };

  assert.equal(
    rasterizeReferenceImageCrop(
      source,
      { x: 20, y: 30, width: 40, height: 50 },
      { width: 100, height: 100 },
      () => canvas,
    ),
    "data:image/png;base64,CROP",
  );
  assert.equal(canvas.width, 40);
  assert.equal(canvas.height, 50);
  assert.deepEqual(calls, [[source, 20, 30, 40, 50, 0, 0, 40, 50]]);
});

test("applying a crop retains the original data URL and picture ordering", () => {
  const staged = ["original-1", "original-2", "original-3"].map(
    stageReferenceImage,
  );
  const crop = { x: 10, y: 20, width: 300, height: 200 };
  const cropped = applyReferenceImageCrop(staged, 1, "crop-2", crop);

  assert.deepEqual(referenceImageDataUrls(cropped), [
    "original-1",
    "crop-2",
    "original-3",
  ]);
  assert.equal(cropped[1]?.originalDataUrl, "original-2");
  assert.deepEqual(cropped[1]?.crop, crop);
  assert.strictEqual(cropped[0], staged[0]);
  assert.strictEqual(cropped[2], staged[2]);

  const restored = applyReferenceImageCrop(
    cropped,
    1,
    cropped[1]?.originalDataUrl ?? "",
    null,
  );
  assert.equal(restored[1]?.dataUrl, "original-2");
  assert.equal(restored[1]?.crop, null);
});

test("request serialization keeps current rasters in picture order", () => {
  let images = ["original-1", "original-2", "original-3"].map(
    stageReferenceImage,
  );
  images = applyReferenceImageCrop(images, 0, "crop-1", {
    x: 1,
    y: 2,
    width: 3,
    height: 4,
  });
  images = applyReferenceImageCrop(images, 2, "crop-3", {
    x: 5,
    y: 6,
    width: 7,
    height: 8,
  });
  assert.deepEqual(referenceImageDataUrls(images), [
    "crop-1",
    "original-2",
    "crop-3",
  ]);
});

test("only the latest image decode claim can publish editor state", () => {
  const gate = createCropImageLoadGate();
  const oldLoad = gate.begin("old-picture");
  const currentLoad = gate.begin("current-picture");
  assert.equal(oldLoad.isCurrent(), false);
  assert.equal(currentLoad.isCurrent(), true);
  currentLoad.cancel();
  assert.equal(currentLoad.isCurrent(), false);
});

test("a crop export over the request field cap is rejected locally", () => {
  assert.equal(
    referenceImageDataUrlError("x".repeat(MAX_REFERENCE_IMAGE_DATA_URL_LENGTH)),
    null,
  );
  assert.match(
    referenceImageDataUrlError(
      "x".repeat(MAX_REFERENCE_IMAGE_DATA_URL_LENGTH + 1),
    ) ?? "",
    /too large/,
  );
});

test("Cancel closes without invoking Apply while Apply performs both actions", () => {
  const applied: Array<{ dataUrl: string; crop: unknown }> = [];
  const openChanges: boolean[] = [];
  const actions = createReferenceImageEditorActions({
    onApply: (dataUrl, crop) => applied.push({ dataUrl, crop }),
    onOpenChange: (open) => openChanges.push(open),
  });

  actions.cancel();
  assert.deepEqual(applied, []);
  assert.deepEqual(openChanges, [false]);

  const crop = { x: 1, y: 2, width: 3, height: 4 };
  actions.apply("crop", crop);
  assert.deepEqual(applied, [{ dataUrl: "crop", crop }]);
  assert.deepEqual(openChanges, [false, false]);
});

test("a crop larger than the model can use is exported at a size it can", () => {
  // Measured in real engines, an unbounded PNG export of a 12MP photo reaches ~52 MiB and a
  // 24MP one 36-105 MiB, all past the 32 MiB cap this module enforces.
  assert.deepEqual(referenceCropExportSize({ width: 5712, height: 4284 }), {
    width: 2730,
    height: 2048,
  });
  assert.deepEqual(referenceCropExportSize({ width: 4032, height: 3024 }), {
    width: 2730,
    height: 2048,
  });
  // The long edge is capped too, for aspects whose short edge already fits.
  assert.deepEqual(referenceCropExportSize({ width: 8192, height: 2000 }), {
    width: 4096,
    height: 1000,
  });
  // Anything already within the bounds is untouched, including both exact edges.
  assert.deepEqual(referenceCropExportSize({ width: 40, height: 50 }), {
    width: 40,
    height: 50,
  });
  assert.deepEqual(referenceCropExportSize({ width: 4096, height: 2048 }), {
    width: 4096,
    height: 2048,
  });
});

test("the raster is drawn at the reduced size, from the full selected source rect", () => {
  const source = {} as CanvasImageSource;
  const calls: unknown[][] = [];
  const canvas: CropRasterCanvas = {
    width: 0,
    height: 0,
    getContext: () => ({
      drawImage: (...args: unknown[]) => calls.push(args),
    }),
    toDataURL: () => "data:image/png;base64,BIG",
  };

  rasterizeReferenceImageCrop(
    source,
    { x: 100, y: 200, width: 5712, height: 4284 },
    { width: 6000, height: 4500 },
    () => canvas,
  );

  assert.equal(canvas.width, 2730);
  assert.equal(canvas.height, 2048);
  // Source rect unchanged, destination reduced: the downscale crops nothing away, and the
  // stored crop stays in source pixels for the editor to restore.
  assert.deepEqual(calls, [[source, 100, 200, 5712, 4284, 0, 0, 2730, 2048]]);
});
