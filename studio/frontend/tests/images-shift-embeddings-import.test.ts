// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  APPLIED_RECIPE_KEYS,
  importedRecipe,
  importNotApplied,
  INFO_RECIPE_KEYS,
  parseA1111Parameters,
  PENDING_RECIPE_KEYS,
  recipeSettings,
  unknownRecipeKeys,
} from "../src/features/images/lib/image-metadata.ts";
import {
  shiftFromRecipe,
  shiftNotApplied,
  shiftRequestFields,
  splitInstalledEmbeddings,
} from "../src/features/images/lib/shift-choice.ts";
import { readSrc } from "./helpers/kit.ts";

const DEFAULTS = { width: 1024, height: 1024, steps: 30, guidance: 4 };
const SD3 = { node: "sd3" as const, shift: 3 };
const FLUX = { node: "flux" as const, base_shift: 0.5, max_shift: 1.15 };

test("a pasted Shift restores into the Shift box and is sent on a model with the sd3 node", () => {
  const r = parseA1111Parameters("a cat\nSteps: 20, CFG scale: 4, Seed: 1, Size: 1024x1024, Shift: 3.1");
  assert.ok(r);
  assert.equal(r.shift, 3.1);
  const settings = recipeSettings(r, DEFAULTS);
  const value = shiftFromRecipe(settings);
  assert.deepEqual(value, { shift: "3.1", baseShift: "", maxShift: "" });
  assert.deepEqual(shiftRequestFields(SD3, value), { shift: 3.1 });
  assert.deepEqual(shiftNotApplied(r, SD3), []);
  // No Shift control, or the ModelSamplingFlux node: named, not dropped.
  assert.deepEqual(shiftNotApplied(r, null), ["shift 3.1"]);
  assert.deepEqual(shiftNotApplied(r, FLUX), ["shift 3.1"]);
  assert.deepEqual(importNotApplied(r, "sd3"), []);
});

test("a Studio recipe's base / max shift restore on a flux-node model; out-of-range values are named", () => {
  const recipe = importedRecipe({
    recipe: JSON.stringify({
      prompt: "p",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: 5,
      base_shift: 0.6,
      max_shift: 1.4,
      embeddings: ["easynegative"],
      schema_version: 1,
    }),
  });
  assert.ok(recipe);
  // shift / base_shift / max_shift restore and embeddings are informational: none is an unknown key.
  assert.deepEqual(unknownRecipeKeys(recipe), []);
  const settings = recipeSettings(recipe, DEFAULTS);
  assert.deepEqual(shiftRequestFields(FLUX, shiftFromRecipe(settings)), { base_shift: 0.6, max_shift: 1.4 });
  assert.deepEqual(shiftNotApplied(recipe, FLUX), []);
  assert.deepEqual(shiftNotApplied(recipe, SD3), ["base shift 0.6", "max shift 1.4"]);
  assert.deepEqual(shiftNotApplied({ shift: 0 }, SD3), ["shift 0 (out of range)"]);
  assert.equal(recipeSettings({ prompt: "p", shift: 0 }, DEFAULTS).shift, undefined);
  assert.equal(recipeSettings({ prompt: "p", shift: 250 }, DEFAULTS).shift, undefined);
  assert.equal(recipeSettings({ prompt: "p", max_shift: 250 }, DEFAULTS).max_shift, 100);
  for (const key of ["shift", "base_shift", "max_shift"]) assert.ok((APPLIED_RECIPE_KEYS as readonly string[]).includes(key));
  assert.ok((INFO_RECIPE_KEYS as readonly string[]).includes("embeddings"));
  assert.ok((PENDING_RECIPE_KEYS as readonly string[]).includes("textual_inversions"));
});

test("TI hashes names stay reported unless an installed embedding matches", () => {
  const r = parseA1111Parameters(
    'a cat, EasyNegative\nSteps: 20, Seed: 1, TI hashes: "easynegative: c74b4e810b03, badhandv4: 5e40d722fc3d"',
  );
  assert.ok(r);
  assert.deepEqual(r.textual_inversions, ["easynegative", "badhandv4"]);
  // Without a model that takes embeddings (or before the lookup) every name is reported, as #14508 did.
  assert.deepEqual(importNotApplied(r, "sdxl"), [
    "textual inversion easynegative",
    "textual inversion badhandv4",
  ]);
  // The backend matches a prompt word to the file stem in any case.
  assert.deepEqual(splitInstalledEmbeddings(r.textual_inversions as string[], ["EasyNegative", "other"]), {
    installed: ["easynegative"],
    missing: ["badhandv4"],
  });
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /const resolved = new Map<string, string \| null>\(\[\n\s+\.\.\.installed\.map\(\(n\): \[string, null\] => \[textualInversionLabel\(n\), null\]\),/);
  assert.match(page, /`\$\{textualInversionLabel\(n\)\} \(not installed\)`/);
  // Superseded or model-switched imports drop the embedding match, like the LoRA lookup.
  assert.match(
    page,
    /if \(seq !== importSeq\.current\) return false;\n[^\n]*\n\s+if \(triggers && loadedModelKey\(statusRef\.current\) === modelBefore\)\n\s+tiMatch = splitInstalledEmbeddings\(tiNames, triggers\);/,
  );
  assert.match(page, /if \(tiNames\.length > 0 && embeddingsEnabled && status\?\.family\)/);
  assert.match(page, /\.\.\.shiftNotApplied\(recipe, shiftControl\),/);
  // A gallery restore and an import both fill the Shift boxes.
  assert.match(page, /setBatchSize\(image\.batch_size \?\? 1\);\n\s+setShiftValue\(shiftFromRecipe\(image\)\);/);
});
