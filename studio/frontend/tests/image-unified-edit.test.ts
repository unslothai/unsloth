// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Qwen-Image-2.1 editing on the Images page: the size the form shows is the size the backend
// renders, the request keeps image order and drops nothing silently, and a restored edit reopens
// Edit with its inputs requested again instead of replaying as text-to-image.

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionConditioning } from "../src/features/images/api.ts";
import {
  additionalImageNumber,
  conditionedRequestFields,
  maxAdditionalImages,
  presetsWithin,
  resolveEditSize,
  restoreInputsNote,
  seedReferenceResolution,
  withLocalizedHint,
  withTransparencyPrompt,
} from "../src/features/images/edit-conditioning.ts";
import { defaultsFor } from "../src/features/images/image-generation-defaults.ts";
import {
  DEFAULT_SIZE_LIMITS,
  fitSize,
  matchSourceSize,
  restorableSize,
  sizeLimitsFrom,
  snapDim,
} from "../src/features/images/image-size.ts";

import { readSrc } from "./helpers/kit.ts";

const QWEN21: DiffusionConditioning = {
  max_condition_images: 10,
  alpha: true,
  dimension_multiple: 32,
  max_output_side: 2752,
  max_output_pixels: 2400 * 1792,
  reference_resolutions: [512, 1024, 2048],
  unified_edit: true,
  localized_edit_modes: ["annotate", "paint", "mask"],
};
const LIMITS = sizeLimitsFrom(QWEN21);

test("match-source sizes agree with the backend's match_source_size", () => {
  // Same pairs the backend tests pin, so the form never shows a size the engine does not render.
  assert.deepEqual(matchSourceSize(300, 200, 1024, LIMITS), {
    width: 1248,
    height: 832,
  });
  assert.deepEqual(matchSourceSize(64, 32, 1024, LIMITS), {
    width: 1440,
    height: 736,
  });
  const big = matchSourceSize(4000, 1000, 2048, LIMITS);
  assert.ok(
    big.width <= 2752 && big.width * big.height <= 2400 * 1792,
    JSON.stringify(big),
  );
  assert.equal(big.width % 32, 0);
  assert.equal(big.height % 32, 0);
});

test("the grid and bounds follow the loaded model, and default to the historical ones", () => {
  assert.deepEqual(sizeLimitsFrom(null), DEFAULT_SIZE_LIMITS);
  assert.equal(snapDim(1040), 1040);
  assert.equal(snapDim(1040, LIMITS), 1056);
  assert.equal(snapDim(2752), 2048);
  assert.equal(snapDim(2752, LIMITS), 2752);
  const fitted = fitSize(2752, 2752, LIMITS);
  assert.ok(fitted.width * fitted.height <= LIMITS.maxPixels);
  assert.equal(fitted.width, fitted.height);
  // A 2K recipe restores whole on this model and scaled on another.
  assert.deepEqual(restorableSize(2752, 1536, "txt2img", LIMITS), {
    width: 2752,
    height: 1536,
  });
  const elsewhere = restorableSize(2752, 1536, "txt2img");
  assert.ok(elsewhere.width <= 2048);
});

test("official 2K presets are offered only where they fit", () => {
  assert.equal(presetsWithin(LIMITS).length, 7);
  assert.equal(presetsWithin(DEFAULT_SIZE_LIMITS).length, 1); // only 2048 x 2048
});

test("the unified edit sends the size it shows", () => {
  const source = { width: 1536, height: 1024 };
  assert.deepEqual(
    resolveEditSize(
      "source",
      source,
      1024,
      { width: 512, height: 512 },
      LIMITS,
    ),
    { width: 1248, height: 832 },
  );
  assert.deepEqual(
    resolveEditSize(
      "custom",
      source,
      1024,
      { width: 1040, height: 768 },
      LIMITS,
    ),
    fitSize(1040, 768, LIMITS),
  );
});

test("slots count the mask, and additional images are numbered after it", () => {
  assert.equal(maxAdditionalImages(QWEN21, null), 9);
  assert.equal(maxAdditionalImages(QWEN21, "mask"), 8);
  assert.equal(maxAdditionalImages(null, null), 3);
  assert.equal(additionalImageNumber(0, null), 2);
  assert.equal(additionalImageNumber(0, "mask"), 3);
});

test("reference detail seeds from the build tier and never picks 2048 on its own", () => {
  assert.equal(seedReferenceResolution([512, 1024, 2048], 512), 512);
  assert.equal(seedReferenceResolution([512, 1024, 2048], 1024), 1024);
  assert.equal(seedReferenceResolution([512, 1024, 2048], 768), 1024);
  assert.equal(seedReferenceResolution([], 1024), null);
});

test("the request keeps order, drops empty slots and only sends what the model takes", () => {
  const fields = conditionedRequestFields({
    workflow: "edit",
    initImage: "SRC",
    extras: ["A", "", "B"],
    referenceResolution: 2048,
    conditioning: QWEN21,
    localized: { mode: "mask", image: "M" },
  });
  assert.deepEqual(fields, {
    workflow: "edit",
    init_image: "SRC",
    reference_images: ["A", "B"],
    reference_resolution: 2048,
    localized_edit: { mode: "mask", image: "M" },
  });
  const flux = conditionedRequestFields({
    workflow: "reference",
    initImage: "SRC",
    extras: ["A"],
    referenceResolution: 1024,
    conditioning: { ...QWEN21, reference_resolutions: [], unified_edit: false },
    localized: { mode: "paint", image: "P" },
  });
  assert.equal(flux.reference_resolution, undefined);
  assert.equal(flux.localized_edit, undefined);
  assert.deepEqual(flux.reference_images, ["A"]);
});

test("prompt helpers edit the visible instruction idempotently", () => {
  const once = withTransparencyPrompt("A red apple.");
  assert.match(
    once,
    /^This is an RGBA image with transparency\. A red apple\. The image has alpha channel/,
  );
  assert.equal(withTransparencyPrompt(once), once);
  const marked = withLocalizedHint("Change the hair to black.", "annotate", [
    "red",
    "blue",
  ]);
  assert.match(
    marked,
    /Do not render the red and blue annotation lines in the image\.$/,
  );
  assert.equal(withLocalizedHint(marked, "annotate", ["red", "blue"]), marked);
  assert.match(
    withLocalizedHint("add a diver", "paint"),
    /^In the area marked with white paint, add a diver$/,
  );
});

test("Qwen-Image-2.1 samples at 40 steps without guidance under every artifact name", () => {
  for (const id of [
    "Qwen/Qwen-Image-2.1",
    "unsloth/Qwen-Image-2.1-GGUF",
    "unsloth/Qwen-Image-2.1-FP8",
  ]) {
    assert.deepEqual(defaultsFor(id), { steps: 40, guidance: 1 }, id);
  }
  assert.deepEqual(defaultsFor("Qwen/Qwen-Image-2512"), {
    steps: 20,
    guidance: 4,
  });
});

test("a restored edit names every input it needs again", () => {
  assert.equal(
    restoreInputsNote({
      workflow: "edit",
      reference_image_count: 2,
      localized_edit: "mask",
    }),
    "the source image, the mask and 2 additional images in the same order",
  );
  assert.equal(
    restoreInputsNote({ workflow: "reference" }),
    "the source image",
  );
  assert.equal(restoreInputsNote({ workflow: "img2img" }), null);
});

test("restoreSettings reopens Edit and Reference instead of Create", () => {
  // Structural: the page component is not renderable under node:test. The workflow decision and
  // the cleared uploads are what keep a restored edit from generating as text-to-image.
  const src = readSrc("features/images/images-page.tsx");
  const body = src.slice(src.indexOf("const restoreSettings = useCallback"));
  assert.match(
    body,
    /image\.workflow === "edit" \? "edit" : image\.workflow === "reference" \? "reference" : "create"/,
  );
  assert.match(body, /setInitImage\(null\)/);
  assert.match(body, /setReferenceImages\(/);
  // Generation stays blocked until the source and every restored slot are supplied again.
  assert.match(src, /usesInit && !initImage/);
  assert.match(src, /is empty\. Add it, or remove that slot\./);
});
