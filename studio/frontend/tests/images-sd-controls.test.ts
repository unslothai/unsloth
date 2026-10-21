// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_SD_CONTROLS,
  parseVariationSeed,
  sdControlsFromRecipe,
  sdControlsRecipeRows,
  sdControlsRequestFields,
  varyChoice,
} from "../src/features/images/lib/sd-controls.ts";
import { readSrc } from "./helpers/kit.ts";

const SDXL = {
  prompt_weighting: true,
  clip_skip: { min: 1, max: 12, default: 1, same_as_default_through: 2 },
  cfg_rescale: { min: 0, max: 1, default: 0, recommended: null },
  variation_seed: true,
  vary_strength: 0.15,
  prediction_type: "epsilon",
};

test("defaults send nothing, so the request is unchanged", () => {
  assert.deepEqual(sdControlsRequestFields(SDXL, DEFAULT_SD_CONTROLS), {});
  assert.deepEqual(
    sdControlsRequestFields(null, { ...DEFAULT_SD_CONTROLS, clipSkip: 3 }),
    {},
  );
});

test("a model without the controls never gets them", () => {
  const choice = {
    promptWeighting: false,
    clipSkip: 3,
    cfgRescale: 0.7,
    variationSeed: "5",
    variationStrength: 0.2,
  };
  assert.deepEqual(sdControlsRequestFields(undefined, choice), {});
  assert.deepEqual(sdControlsRequestFields({}, choice), {});
});

test("non-default values map to request fields", () => {
  assert.deepEqual(
    sdControlsRequestFields(SDXL, {
      promptWeighting: false,
      clipSkip: 2,
      cfgRescale: 0.7,
      variationSeed: "42",
      variationStrength: 0.25,
    }),
    {
      prompt_weighting: false,
      clip_skip: 2,
      cfg_rescale: 0.7,
      variation_strength: 0.25,
      variation_seed: 42,
    },
  );
  // An explicit per-run variation seed wins over the field; an empty field leaves it to the server.
  assert.deepEqual(
    sdControlsRequestFields(
      SDXL,
      { ...DEFAULT_SD_CONTROLS, variationStrength: 0.1 },
      7,
    ),
    { variation_strength: 0.1, variation_seed: 7 },
  );
  assert.deepEqual(
    sdControlsRequestFields(SDXL, {
      ...DEFAULT_SD_CONTROLS,
      variationStrength: 0.1,
    }),
    {
      variation_strength: 0.1,
    },
  );
  assert.equal(
    sdControlsRequestFields(
      { ...SDXL, clip_skip: { min: 1, max: 5, default: 1 } },
      { ...DEFAULT_SD_CONTROLS, clipSkip: 9 },
    ).clip_skip,
    5,
  );
});

test("image-conditioned workflows never send the variation fields", () => {
  const choice = { ...DEFAULT_SD_CONTROLS, cfgRescale: 0.5, variationSeed: "7", variationStrength: 0.3 };
  assert.deepEqual(sdControlsRequestFields(SDXL, choice, 7, false), { cfg_rescale: 0.5 });
  assert.equal(sdControlsRequestFields(SDXL, choice, 7).variation_strength, 0.3);
  const src = readSrc("features/images/images-page.tsx");
  assert.match(
    src,
    /const varying = Boolean\(sdCaps\?\.variation_seed\) && workflow === "create" && runChoice\.variationStrength > 0;/,
  );
  assert.match(src, /% 2 \*\* 32 : null,\n\s+varying,\n\s+\),/);
  assert.match(src, /\{sdCaps\.variation_seed && workflow === "create" && \(/);
});

test("variation seeds parse as 32-bit integers", () => {
  assert.equal(parseVariationSeed(""), null);
  assert.equal(parseVariationSeed(" 12 "), 12);
  assert.equal(parseVariationSeed("-1"), null);
  assert.equal(parseVariationSeed("1.5"), null);
  assert.equal(parseVariationSeed(String(2 ** 32)), null);
});

test("Vary keeps the settings and uses the advertised small strength", () => {
  const v = varyChoice(SDXL, { ...DEFAULT_SD_CONTROLS, clipSkip: 3 }, 99);
  assert.equal(v.variationSeed, "99");
  assert.equal(v.variationStrength, 0.15);
  assert.equal(v.clipSkip, 3);
  assert.equal(varyChoice({}, DEFAULT_SD_CONTROLS, 1).variationStrength, 0.15);
});

test("recipe restore and rows", () => {
  assert.deepEqual(sdControlsFromRecipe({}), DEFAULT_SD_CONTROLS);
  const restored = sdControlsFromRecipe({
    clip_skip: 2,
    cfg_rescale: 0.7,
    variation_seed: 15,
    variation_strength: 0.2,
    batch_index: 3,
    batch_size: 4,
  });
  assert.equal(restored.variationSeed, "12");
  assert.equal(restored.clipSkip, 2);
  assert.equal(restored.promptWeighting, true);
  // A run with weighting off records false; restoring it must keep the prompt literal.
  assert.equal(sdControlsFromRecipe({ prompt_weighting: false }).promptWeighting, false);
  assert.equal(sdControlsFromRecipe({ prompt_weighting: null }).promptWeighting, true);
  assert.equal(sdControlsFromRecipe({ prompt_weighting: true }).promptWeighting, true);
  assert.deepEqual(sdControlsRecipeRows({}), []);
  assert.deepEqual(
    sdControlsRecipeRows({
      clip_skip: 2,
      variation_seed: 5,
      variation_strength: 0.1,
    }),
    [
      ["CLIP skip", "2"],
      ["Variation", "5 @ 0.1"],
    ],
  );
});

test("the page gates the controls and Vary on the status capability", () => {
  const src = readSrc("features/images/images-page.tsx");
  assert.match(
    src,
    /const sdCaps = \(!external && status\?\.loaded \? status\.sd_controls : null\) \?\? null;/,
  );
  assert.match(src, /\{sdCaps && \(/);
  assert.match(src, /\.\.\.sdControlsRequestFields\(/);
  assert.match(
    src,
    /Boolean\(status\?\.loaded && sdCaps\?\.variation_seed\) && workflow === "create"/,
  );
});

test("AUTOMATIC1111 parameters import the controls", async () => {
  const { parseA1111Parameters, recipeSettings, unknownRecipeKeys } =
    await import("../src/features/images/lib/image-metadata.ts");
  const parsed = parseA1111Parameters(
    "a (fox:1.2)\nSteps: 25, Sampler: Euler, CFG scale: 5, Seed: 11, Size: 832x1216, Model: m, Variation seed: 12, Variation seed strength: 0.15, Clip skip: 2, CFG rescale: 0.7",
  );
  assert.ok(parsed);
  assert.deepEqual(unknownRecipeKeys(parsed), []);
  const settings = recipeSettings(parsed, {
    width: 1024,
    height: 1024,
    steps: 9,
    guidance: 0,
  });
  assert.equal(settings.clip_skip, 2);
  assert.equal(settings.cfg_rescale, 0.7);
  assert.equal(settings.variation_seed, 12);
  assert.equal(settings.variation_strength, 0.15);
  assert.equal(settings.prompt_weighting, undefined);
  const literal = recipeSettings(
    { prompt: "(fox:1.5)", prompt_weighting: false },
    { width: 1024, height: 1024, steps: 9, guidance: 0 },
  );
  assert.equal(sdControlsFromRecipe(literal).promptWeighting, false);
  const clamped = recipeSettings(
    {
      prompt: "p",
      clip_skip: 40,
      cfg_rescale: 3,
      variation_seed: 2 ** 40,
      variation_strength: -1,
    },
    { width: 1024, height: 1024, steps: 9, guidance: 0 },
  );
  assert.equal(clamped.clip_skip, 12);
  assert.equal(clamped.cfg_rescale, 1);
  assert.equal(clamped.variation_seed, undefined);
  assert.equal(clamped.variation_strength, 0);
});

test("pasted A1111 Clip skip, CFG Rescale and Variation seed restore into the SDXL controls", async () => {
  const { parseA1111Parameters, importNotApplied, recipeSettings, variationApplies } = await import(
    "../src/features/images/lib/image-metadata.ts"
  );
  const r = parseA1111Parameters(
    "(a cat:1.2), \\(sign\\)\nSteps: 25, Sampler: Euler a, CFG scale: 5, Seed: 1, Size: 1024x1024, Clip skip: 3, CFG Rescale: 0.7, Variation seed: 99, Variation seed strength: 0.3",
  );
  assert.ok(r);
  const settings = recipeSettings(r, { width: 1024, height: 1024, steps: 30, guidance: 5 });
  assert.deepEqual(sdControlsFromRecipe(settings), {
    promptWeighting: true,
    clipSkip: 3,
    cfgRescale: 0.7,
    variationSeed: "99",
    variationStrength: 0.3,
  });
  assert.equal(variationApplies(r, SDXL), true);
  // SDXL's prompt weighting reads the emphasis and escapes, so nothing is reported.
  assert.deepEqual(importNotApplied(r, "sdxl", false, SDXL), []);
  assert.deepEqual(importNotApplied(r, "flux", true, null), [
    "Clip skip 3",
    "CFG rescale 0.7",
    "variation seed 99 (strength 0.3)",
  ]);
  // Past the model's CLIP skip range the restore clamps, and says so.
  assert.deepEqual(
    importNotApplied({ ...r, clip_skip: 14 }, "sdxl", false, { ...SDXL, clip_skip: { ...SDXL.clip_skip, max: 12 } }),
    ["Clip skip 14 (clamped to 12)"],
  );
  // The backend refuses a variation seed for img2img / inpaint, so it is named and not restored.
  const img2img = { ...r, workflow: "img2img" };
  assert.equal(variationApplies(img2img, SDXL), false);
  assert.deepEqual(importNotApplied(img2img, "sdxl", false, SDXL), ["variation seed 99 (strength 0.3)"]);
  // A structured Unsloth recipe records prompt_weighting: true, not the parser-only prompt_emphasis.
  const weighted = { prompt: "a (fox:1.5)", prompt_weighting: true };
  assert.deepEqual(importNotApplied(weighted, "flux", false, null), ["prompt weighting"]);
  assert.deepEqual(importNotApplied(weighted, "sdxl", true, null), []);
  assert.deepEqual(importNotApplied(weighted, "sdxl", false, SDXL), []);
  const src = readSrc("features/images/images-page.tsx");
  assert.match(src, /if \(!variationApplies\(merged, sdCaps\)\)/);
  assert.match(src, /isNativeEngineStatus\(status\)\),\n\s+sdCaps,/);
  // importNotApplied already names the controls a model lacks; the page must not add them again.
  assert.doesNotMatch(src, /notApplied\.unshift\(`CLIP skip|notApplied\.unshift\(`CFG rescale|notApplied\.unshift\("variation seed"\)/);
});

test("Vary runs the selected image's recipe, not the current form", async () => {
  const { canVaryImage } = await import("../src/features/images/lib/sd-controls.ts");
  assert.equal(canVaryImage({ model: "sdxl", workflow: "txt2img" }, "sdxl"), true);
  assert.equal(canVaryImage({ model: null, workflow: null }, "sdxl"), true);
  // Made with another model, or from a source image the gallery does not keep: its recipe cannot replay here.
  assert.equal(canVaryImage({ model: "other", workflow: "txt2img" }, "sdxl"), false);
  assert.equal(canVaryImage({ model: "sdxl", workflow: "img2img" }, "sdxl"), false);
  assert.equal(canVaryImage({ model: "sdxl", workflow: "controlnet" }, "sdxl"), false);
  const src = readSrc("features/images/images-page.tsx");
  const vary = src.slice(src.indexOf("const handleVary = useCallback("));
  assert.match(vary, /^[\s\S]*?restoreSettings\(selected, true, "vary"\);\n\s+setVaryQueued\(true\);/);
  assert.doesNotMatch(vary.slice(0, vary.indexOf("}, [busy")), /void handleGenerate\(\)/);
  assert.match(src, /if \(!varyQueued \|\| busy !== null\) return;\n\s+setVaryQueued\(false\);\n\s+void handleGenerate\(\);/);
  assert.match(src, /canVaryImage\(selected, status\?\.display_repo_id \?\? status\?\.repo_id \?\? null\)/);
  // The Vary restore is silent, and the import path reports what was not applied itself.
  assert.match(src, /if \(via === "vary"\) return;/);
  assert.match(src, /false,\n\s+"import",\n\s+\);/);
});

test("a literal-prompt recipe restored onto sd.cpp names prompt weighting off as not applied", async () => {
  const { importNotApplied, literalPromptNotApplied } = await import(
    "../src/features/images/lib/image-metadata.ts"
  );
  const literal = { prompt: "a (fox:1.5)", prompt_weighting: false };
  const note = "prompt weighting off (the native sd.cpp engine always reads (word:1.2) syntax)";
  assert.equal(literalPromptNotApplied(literal, true, null), note);
  // Diffusers SDXL turns weighting off; diffusers without the control never parsed the syntax anyway.
  assert.equal(literalPromptNotApplied(literal, false, SDXL), null);
  assert.equal(literalPromptNotApplied(literal, false, null), null);
  assert.equal(literalPromptNotApplied({ prompt_weighting: true }, true, null), null);
  assert.deepEqual(importNotApplied(literal, "sdxl", true, null), [note]);
  assert.deepEqual(importNotApplied(literal, "sdxl", false, SDXL), []);
  const src = readSrc("features/images/images-page.tsx");
  assert.match(
    src,
    /via === "restore"\n\s+\? literalPromptNotApplied\(image, Boolean\(status\?\.loaded && isNativeEngineStatus\(status\)\), sdCaps\)/,
  );
});

test("imported variation values out of the request bounds are named, not randomised or clamped", async () => {
  const { importNotApplied, variationApplies } = await import(
    "../src/features/images/lib/image-metadata.ts"
  );
  const ok = { prompt: "x", variation_seed: 99, variation_strength: 0.2 };
  assert.equal(variationApplies(ok, SDXL), true);
  assert.equal(variationApplies({ prompt: "x", variation_strength: 0.2 }, SDXL), true);
  assert.equal(variationApplies({ prompt: "x", variation_seed: 2 ** 32 - 1, variation_strength: 1 }, SDXL), true);
  for (const bad of [
    { prompt: "x", variation_seed: 2 ** 40, variation_strength: 0.2 },
    { prompt: "x", variation_seed: 2 ** 32, variation_strength: 0.2 },
    { prompt: "x", variation_seed: -1, variation_strength: 0.2 },
    { prompt: "x", variation_seed: 1.5, variation_strength: 0.2 },
    { prompt: "x", variation_seed: 99, variation_strength: 1.5 },
  ]) {
    assert.equal(variationApplies(bad, SDXL), false, JSON.stringify(bad));
    assert.deepEqual(importNotApplied(bad, "sdxl", false, SDXL), [
      `variation seed ${bad.variation_seed} (strength ${bad.variation_strength})`,
    ]);
  }
  assert.deepEqual(importNotApplied({ prompt: "x", variation_strength: 3 }, "sdxl", false, SDXL), ["variation strength 3"]);
});

test("pasted A1111 Emphasis: None restores prompt weighting off, and sd.cpp names it as not applied", async () => {
  const { importNotApplied, parseA1111Parameters, recipeSettings } = await import(
    "../src/features/images/lib/image-metadata.ts"
  );
  const r = parseA1111Parameters("a (fox:1.5), \\(x\\)\nSteps: 20, Seed: 1, Emphasis: None") ?? { prompt: "" };
  assert.equal(r.prompt_weighting, false);
  const settings = recipeSettings(r, { width: 1024, height: 1024, steps: 20, guidance: 7 });
  assert.equal(settings.prompt_weighting, false);
  assert.equal(sdControlsFromRecipe(settings).promptWeighting, false);
  // Diffusers SDXL turns weighting off; diffusers without the control reads it literally already.
  assert.deepEqual(importNotApplied(r, "sdxl", false, SDXL), []);
  assert.deepEqual(importNotApplied(r, "sdxl", false, null), []);
  assert.deepEqual(importNotApplied(r, "sdxl", true, null), [
    "prompt weighting off (the native sd.cpp engine always reads (word:1.2) syntax)",
  ]);
});
