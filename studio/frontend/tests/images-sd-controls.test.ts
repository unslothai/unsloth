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
  assert.match(src, /% 2 \*\* 32 : null,\n\s+varying,\n\s+!controlnetEngaged,\n\s+\),/);
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

test("Vary is disabled with a reason when the load cannot apply the selected recipe as recorded", async () => {
  const { varyBlockedReason } = await import("../src/features/images/lib/sd-controls.ts");
  const BUILD = { schema_version: 1, engine: "diffusers", model_kind: "pipeline" };
  const recipe = { ...BUILD, loras: ["style:0.8"], cfg_rescale: 0.7, clip_skip: 2, prompt_weighting: false };
  assert.equal(varyBlockedReason(recipe, true, SDXL, BUILD), null);
  assert.equal(varyBlockedReason({ ...BUILD, loras: [] }, false, SDXL, BUILD), null);
  // Speed=Auto compiled after the render: supports_lora is now false, so the adapters would be dropped.
  assert.match(varyBlockedReason({ ...BUILD, loras: ["style:0.8"] }, false, SDXL, BUILD) ?? "", /its LoRAs/);
  const noRescale = { ...SDXL, cfg_rescale: null, clip_skip: { ...SDXL.clip_skip, max: 2 } };
  assert.match(varyBlockedReason({ ...BUILD, cfg_rescale: 0.7 }, true, noRescale, BUILD) ?? "", /CFG rescale 0\.7/);
  assert.match(varyBlockedReason({ ...BUILD, clip_skip: 4 }, true, noRescale, BUILD) ?? "", /CLIP skip 4/);
  assert.match(varyBlockedReason({ ...BUILD, prompt_weighting: false }, true, { ...SDXL, prompt_weighting: false }, BUILD) ?? "", /weighting off/);
  const src = readSrc("features/images/images-page.tsx");
  assert.match(src, /const varyBlocked = canVary && selected \? varyBlockedReason\(selected, loraCapable, sdCaps, status \?\? \{\}\) : null;/);
  assert.match(src, /disabled=\{busy !== null \|\| varyBlocked !== null\}/);
  assert.match(src, /\{varyBlocked \?\? \(/);
});

test("Vary is disabled unless the selected image came off the exact loaded build", async () => {
  const { varyBlockedReason } = await import("../src/features/images/lib/sd-controls.ts");
  const loaded = {
    engine: "diffusers",
    model_kind: "gguf",
    gguf_filename: "sdxl-Q4_K_M.gguf",
    transformer_quant: null,
    text_encoder_quant: "fp8",
  };
  const image = { ...loaded, schema_version: 1, model: "sdxl", workflow: "txt2img" };
  // The exact same build keeps Vary on; null and absent quant fields read alike.
  assert.equal(varyBlockedReason(image, true, SDXL, loaded), null);
  assert.equal(varyBlockedReason({ ...image, transformer_quant: undefined }, true, SDXL, loaded), null);
  for (const [field, value, label] of [
    ["engine", "sd_cpp", /engine/],
    ["model_kind", "pipeline", /model kind/],
    ["gguf_filename", "sdxl-Q8_0.gguf", /gguf filename/],
    ["transformer_quant", "int8", /transformer quant/],
    ["text_encoder_quant", null, /text encoder quant/],
  ] as const) {
    assert.match(
      varyBlockedReason({ ...image, [field]: value }, true, SDXL, loaded) ?? "",
      new RegExp(`different build \\(.*${label.source}`),
      field,
    );
  }
  // Older records: no schema_version (build not stored) or no engine (stored before it was recorded) cannot be checked.
  const { schema_version: _drop, ...noSchema } = image;
  assert.match(varyBlockedReason(noSchema, true, SDXL, loaded) ?? "", /does not record which engine/);
  assert.match(varyBlockedReason({ ...image, engine: null }, true, SDXL, loaded) ?? "", /does not record which engine/);
  // The build check comes before the variation-seed one.
  assert.match(varyBlockedReason({ ...image, engine: "mlx", variation_strength: 0.2 }, true, SDXL, loaded) ?? "", /different build/);
});

test("Vary compares the load-time baked LoRAs, attention backend and step cache as part of the build", async () => {
  const { varyBlockedReason } = await import("../src/features/images/lib/sd-controls.ts");
  const loaded = { engine: "diffusers", model_kind: "pipeline", baked_loras: ["ink", "sloth"], attention_backend: "_native_cudnn", transformer_cache: "fbcache" };
  const image = { ...loaded, schema_version: 1, baked_loras: ["sloth", "ink"] };
  // The same adapter set in another order is the same build.
  assert.equal(varyBlockedReason(image, true, SDXL, loaded), null);
  // A baked record against a load without the bake, a different set, or a bake the record lacks.
  for (const other of [{ ...loaded, baked_loras: [] }, { ...loaded, baked_loras: undefined }, { ...loaded, baked_loras: ["sloth"] }])
    assert.match(varyBlockedReason(image, true, SDXL, other) ?? "", /different build \(baked loras\)/);
  assert.match(varyBlockedReason({ ...image, baked_loras: [] }, true, SDXL, loaded) ?? "", /baked loras/);
  // An empty and an absent set read alike on both sides.
  const plain = { engine: "diffusers", schema_version: 1 };
  assert.equal(varyBlockedReason({ ...plain, baked_loras: [] }, true, SDXL, { engine: "diffusers" }), null);
  assert.match(varyBlockedReason({ ...image, attention_backend: null }, true, SDXL, loaded) ?? "", /attention backend/);
  assert.match(varyBlockedReason({ ...image, transformer_cache: "static" }, true, SDXL, loaded) ?? "", /transformer cache/);
  // The status type declares the field the page passes in.
  assert.match(readSrc("features/images/api.ts"), /baked_loras\?: string\[\] \| null;/);
});

test("Create with an SDXL ControlNet engaged sends no CFG rescale, and the control says why", () => {
  const choice = { ...DEFAULT_SD_CONTROLS, cfgRescale: 0.7, variationStrength: 0.2, variationSeed: "5" };
  assert.deepEqual(sdControlsRequestFields(SDXL, choice, null, true, false), {
    variation_strength: 0.2,
    variation_seed: 5,
  });
  assert.equal(sdControlsRequestFields(SDXL, choice).cfg_rescale, 0.7);
  const src = readSrc("features/images/images-page.tsx");
  assert.match(
    src,
    /const controlnetEngaged =\n\s+controlnetCapable && Boolean\(controlnetId && controlImage\) && workflow === "create" && controlStrength > 0;/,
  );
  assert.match(src, /varying,\n\s+!controlnetEngaged,\n\s+\),/);
  assert.match(src, /disabled=\{controlnetEngaged\}/);
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

test("a gallery restore names the SD controls the current load drops or clamps", async () => {
  const { sdControlsNotRestored } = await import("../src/features/images/lib/sd-controls.ts");
  const recipe = { clip_skip: 3, cfg_rescale: 0.7, variation_seed: 42, variation_strength: 0.3 };
  assert.deepEqual(sdControlsNotRestored(recipe, SDXL), []);
  assert.deepEqual(sdControlsNotRestored({ clip_skip: 1, cfg_rescale: 0 }, null), []);
  // Same SDXL repo reloaded on the native engine: no sd_controls, so Generate sends none of them.
  assert.deepEqual(sdControlsNotRestored(recipe, null), [
    "CLIP skip 3",
    "CFG rescale 0.7",
    "variation seed 42 (strength 0.3)",
  ]);
  const narrow = { ...SDXL, clip_skip: { ...SDXL.clip_skip, max: 2 }, cfg_rescale: { ...SDXL.cfg_rescale, max: 0.5 } };
  assert.deepEqual(sdControlsNotRestored(recipe, narrow), ["CLIP skip 3 (clamped to 2)", "CFG rescale 0.7 (clamped to 0.5)"]);
  const src = readSrc("features/images/images-page.tsx");
  assert.match(src, /via === "restore" && status\?\.loaded \? sdControlsNotRestored\(image, sdCaps\) : \[\]/);
  assert.match(src, /description: `Not applied: \$\{notApplied\.join\(", "\)\}\./);
});

test("an imported CFG rescale outside the accepted range is named with its clamp", async () => {
  const { importNotApplied } = await import("../src/features/images/lib/image-metadata.ts");
  assert.deepEqual(importNotApplied({ prompt: "x", cfg_rescale: 0.7 }, "sdxl", false, SDXL), []);
  assert.deepEqual(importNotApplied({ prompt: "x", cfg_rescale: 1.5 }, "sdxl", false, SDXL), ["CFG rescale 1.5 (clamped to 1)"]);
  assert.deepEqual(importNotApplied({ prompt: "x", cfg_rescale: -0.2 }, "sdxl", false, SDXL), ["CFG rescale -0.2 (clamped to 0)"]);
  const narrow = { ...SDXL, cfg_rescale: { ...SDXL.cfg_rescale, max: 0.5 } };
  assert.deepEqual(importNotApplied({ prompt: "x", cfg_rescale: 0.7 }, "sdxl", false, narrow), ["CFG rescale 0.7 (clamped to 0.5)"]);
  assert.deepEqual(importNotApplied({ prompt: "x", cfg_rescale: 0.7 }, "sdxl", false, null), ["CFG rescale 0.7"]);
});

test("Vary is disabled for an image that already used a variation seed", async () => {
  const { varyBlockedReason } = await import("../src/features/images/lib/sd-controls.ts");
  const B = { schema_version: 1, engine: "diffusers" };
  assert.equal(varyBlockedReason({ ...B, variation_strength: 0 }, true, SDXL, B), null);
  assert.match(varyBlockedReason({ ...B, variation_strength: 1 }, true, SDXL, B) ?? "", /already used a variation seed/);
  assert.match(varyBlockedReason({ ...B, variation_strength: 0.15 }, true, SDXL, B) ?? "", /already used a variation seed/);
});
