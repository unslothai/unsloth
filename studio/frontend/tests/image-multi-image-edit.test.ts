// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Multi-image editing for edit-only models (Qwen-Image-Edit-Plus): the Edit tab offers extra image
// slots up to the backend's limit, the request sends them after the source in order, and models
// that take one image (Kontext, Layered, an older backend) keep the single-source Edit request.

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionConditioning } from "../src/features/images/api.ts";
import {
  conditionedRequestFields,
  maxAdditionalImages,
  restoreInputsNote,
} from "../src/features/images/edit-conditioning.ts";

import { readSrc } from "./helpers/kit.ts";

const QWEN_EDIT: DiffusionConditioning = {
  max_condition_images: 3,
  alpha: false,
  dimension_multiple: 16,
  max_output_side: 2048,
  max_output_pixels: 2048 * 2048,
  reference_resolutions: [],
  unified_edit: false,
  multi_image_edit: true,
  localized_edit_modes: [],
};
const KONTEXT: DiffusionConditioning = {
  ...QWEN_EDIT,
  max_condition_images: 1,
  multi_image_edit: false,
};

test("Qwen-Image-Edit takes two images after the source, Kontext none", () => {
  assert.equal(maxAdditionalImages(QWEN_EDIT, null), 2);
  assert.equal(maxAdditionalImages(KONTEXT, null), 0);
});

test("a multi-image edit sends the extras after the source, empty slots dropped in order", () => {
  const fields = conditionedRequestFields({
    workflow: "edit",
    initImage: "SRC",
    extras: ["A", "", "B"],
    referenceResolution: 1024,
    conditioning: QWEN_EDIT,
    localized: { mode: "mask", image: "M" },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(fields)), {
    workflow: "edit",
    init_image: "SRC",
    reference_images: ["A", "B"],
  });
});

test("a single-image edit request is unchanged", () => {
  for (const conditioning of [QWEN_EDIT, KONTEXT, null]) {
    const fields = conditionedRequestFields({
      workflow: "edit",
      initImage: "SRC",
      extras: [],
      referenceResolution: null,
      conditioning,
      localized: null,
    });
    assert.deepEqual(JSON.parse(JSON.stringify(fields)), {
      workflow: "edit",
      init_image: "SRC",
    });
  }
  // A family that does not advertise it never sends extras on Edit, even when slots are filled.
  const older = { ...QWEN_EDIT, multi_image_edit: undefined };
  const stale = conditionedRequestFields({
    workflow: "edit",
    initImage: "SRC",
    extras: ["A"],
    referenceResolution: null,
    conditioning: older,
    localized: null,
  });
  assert.equal(stale.reference_images, undefined);
});

test("a restored multi-image edit asks for its extras again", () => {
  assert.equal(
    restoreInputsNote({ workflow: "edit", reference_image_count: 2 }),
    "the source image and 2 additional images in the same order",
  );
});

test("the Edit tab shows and validates the extra slots for a multi-image edit model", () => {
  // Structural: the page component is not renderable under node:test.
  const src = readSrc("features/images/images-page.tsx");
  assert.match(
    src,
    /const multiImageEdit = !unifiedEdit && Boolean\(conditioning\?\.multi_image_edit\)/,
  );
  const editTab = src.slice(src.indexOf('{workflow === "edit" && !unifiedEdit && ('));
  assert.match(
    editTab.slice(0, 1500),
    /\{multiImageEdit &&\s*renderAdditionalImages\(\s*\(i\) => i \+ 2,/,
  );
  assert.match(src, /const multiEditRun = isEdit && multiImageEdit;/);
  assert.match(
    src,
    /\(isReference \|\| unifiedEditRun \|\| multiEditRun\) \? referenceImages\.findIndex/,
  );
  assert.match(
    src,
    /\(isReference \|\| unifiedEditRun \|\| multiEditRun\) && referenceImages\.filter\(Boolean\)\.length > maxExtras/,
  );
  assert.match(
    src,
    /\} else if \(isReference \|\| unifiedEditRun \|\| multiEditRun\) \{\s*condFields = conditionedRequestFields\(/,
  );
});
