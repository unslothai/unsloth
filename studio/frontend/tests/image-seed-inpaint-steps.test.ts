// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  defaultsFor,
  STEPS_REQUEST_MAX,
  STEPS_SLIDER_MAX,
  stepsSliderMax,
} from "../src/features/images/image-generation-defaults.ts";
import {
  INPAINT_PADDING_DEFAULT,
  inpaintMaskRequestFields,
  MASK_BLUR_DEFAULT,
} from "../src/features/images/lib/inpaint-options.ts";
import {
  lastSeedOf,
  reusableSeed,
} from "../src/features/images/lib/seed-controls.ts";
import { readSrc } from "./helpers/kit.ts";

test("the Steps slider reaches every model's default: Krea-2-raw ships 52", () => {
  const krea2Raw = defaultsFor("krea/Krea-2-raw").steps;
  assert.equal(krea2Raw, 52);
  assert.ok(stepsSliderMax(krea2Raw, krea2Raw) >= krea2Raw);
  assert.equal(stepsSliderMax(krea2Raw, krea2Raw), STEPS_REQUEST_MAX);
  assert.equal(
    stepsSliderMax(defaultsFor("black-forest-labs/FLUX.1-schnell").steps, 4),
    STEPS_SLIDER_MAX,
  );
  assert.equal(stepsSliderMax(25, 25), 50);
  assert.equal(stepsSliderMax(30, 30), 60);
  assert.equal(stepsSliderMax(8, 80), 80);
  assert.equal(stepsSliderMax(8, 150), STEPS_REQUEST_MAX);
  assert.equal(stepsSliderMax(Number.NaN, 4), STEPS_SLIDER_MAX);
  for (const id of [
    "krea-2-raw",
    "lumina",
    "hunyuanimage",
    "hidream",
    "qwen-image-2512",
    "flux.3-klein-base",
  ]) {
    const d = defaultsFor(id).steps;
    assert.ok(stepsSliderMax(d, d) >= d, id);
  }
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /max=\{stepsSliderMax\(pendingModelDefaults\?\.steps \?\? residentSteps, steps\)\}/,
  );
  assert.doesNotMatch(page, /label="Steps"[\s\S]{0,300}max=\{50\}/);
});

test("Reuse last seed takes this session's run base seed, else the newest record's", () => {
  const run = [
    { seed: 101, batch_seed: 100, created_at: 5 },
    { seed: 102, batch_seed: 100, created_at: 5 },
  ];
  assert.equal(lastSeedOf(run), 100);
  assert.equal(lastSeedOf([{ seed: 7, created_at: 1 }]), 7);
  assert.equal(
    lastSeedOf([{ seed: 7, created_at: 1, model_kind: "connection" }]),
    null,
  );
  assert.equal(lastSeedOf([]), null);
  const gallery = [
    { seed: 1, created_at: 10 },
    { seed: 2, batch_seed: 20, created_at: 30 },
    { seed: 3, created_at: 40, model_kind: "connection" },
  ];
  assert.equal(reusableSeed(null, gallery), 20);
  assert.equal(reusableSeed(555, gallery), 555);
  assert.equal(reusableSeed(null, []), null);
  // Seed 0 is a real seed, not "none".
  assert.equal(reusableSeed(0, gallery), 0);
});

test("the Seed field has Random and Reuse last seed buttons", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /aria-label="Random seed"[\s\S]{0,200}onClick=\{\(\) => setSeed\(""\)\}/,
  );
  assert.match(
    page,
    /aria-label="Reuse last seed"[\s\S]{0,300}if \(lastSeed !== null\) setSeed\(String\(lastSeed\)\);/,
  );
  assert.match(
    page,
    /const runSeed = lastSeedOf\(res\.images\);\n\s+if \(runSeed !== null\) setSessionSeed\(runSeed\);/,
  );
});

test("inpaint mask options send nothing at their defaults", () => {
  const defaults = {
    maskBlur: MASK_BLUR_DEFAULT,
    onlyMasked: false,
    padding: INPAINT_PADDING_DEFAULT,
  };
  assert.deepEqual(inpaintMaskRequestFields(defaults, true), {});
  assert.deepEqual(inpaintMaskRequestFields(defaults, false), {});
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, maskBlur: 4 }, false),
    { mask_blur: 4 },
  );
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, maskBlur: 99.6 }, false),
    { mask_blur: 64 },
  );
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, onlyMasked: true }, true),
    {
      inpaint_only_masked: true,
      inpaint_padding: 32,
    },
  );
  // A model whose inpaint pipeline cannot crop never gets the field (the backend would refuse it).
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, onlyMasked: true }, false),
    {},
  );
  assert.deepEqual(
    inpaintMaskRequestFields(
      { maskBlur: 8, onlyMasked: true, padding: 900 },
      true,
    ),
    {
      mask_blur: 8,
      inpaint_only_masked: true,
      inpaint_padding: 256,
    },
  );
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /condStrength = strength;\n\s+condInpaint = inpaintMaskRequestFields\(/,
  );
  assert.match(page, /\.\.\.condInpaint,/);
  assert.match(page, /\{status\?\.supports_inpaint_only_masked && \(/);
});
