// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  a1111Fields,
  extractExtraNetworks,
  importNotApplied,
  importedRecipe,
  isGenerationParametersPaste,
  matchLoraTags,
  loraTagNames,
  matchLoraTagsForImport,
  MAX_LORAS,
  lorasOverLimit,
  parseA1111Parameters,
  pastedParameters,
  recipeSettings,
  unknownRecipeKeys,
} from "../src/features/images/lib/image-metadata.ts";
import { readSrc } from "./helpers/kit.ts";

const DEFAULTS = { width: 1024, height: 1024, steps: 8, guidance: 3.5 };

// What Civitai's "Copy generation data" gives for an SDXL Pony image (shape of a real one).
const CIVITAI = [
  "score_9, score_8_up, 1girl, solo, <lora:add-detail-xl:0.8>, (masterpiece:1.2), <lora:missing_style:1>, smile",
  "Negative prompt: score_4, lowres, bad hands",
  'Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 3812461337, Size: 832x1216, Clip skip: 2, Created Date: 2024-05-01T10:00:00.000Z, Civitai resources: [{"type":"checkpoint","modelVersionId":290640,"modelName":"Pony Diffusion V6 XL, final"},{"type":"lora","weight":0.8,"modelVersionId":1}], Civitai metadata: {}, Model hash: 67ab2fd8ec, Model: ponyDiffusionV6XL, Lora hashes: "add-detail-xl: 9c783c8ce46c, missing_style: abc"',
].join("\n");

test("a Civitai block parses: legacy sampler name, JSON values with commas, quoted hashes", () => {
  const r = parseA1111Parameters(CIVITAI);
  assert.ok(r);
  assert.equal(
    r.prompt,
    "score_9, score_8_up, 1girl, solo, (masterpiece:1.2), smile",
  );
  assert.equal(r.negative_prompt, "score_4, lowres, bad hands");
  assert.equal(r.steps, 30);
  assert.equal(r.sampler, "dpmpp_2m");
  assert.equal(r.schedule_type, "karras");
  assert.equal(r.guidance, 7);
  assert.equal(r.seed, 3812461337);
  assert.equal(r.width, 832);
  assert.equal(r.height, 1216);
  assert.equal(r.model, "ponyDiffusionV6XL");
  assert.equal(r.clip_skip, 2);
  assert.deepEqual(r.lora_tags, [
    { name: "add-detail-xl", weight: 0.8 },
    { name: "missing_style", weight: 1 },
  ]);
  assert.equal(r.parameters_not_applied, undefined);
  // The (masterpiece:1.2) weight stays in the prompt; only sd.cpp parses it.
  const emphasis = "prompt emphasis ((masterpiece:1.2)) is passed as plain text";
  assert.equal(r.prompt.includes("(masterpiece:1.2)"), true);
  // Clip skip 2 is what SDXL already renders, so nothing is reported for it there.
  assert.deepEqual(importNotApplied(r, "sdxl"), [emphasis]);
  assert.deepEqual(importNotApplied(r, "sdxl", true), []);
  // Any other family (or none loaded) applies no clip skip, so it is named.
  assert.deepEqual(importNotApplied(r), [emphasis, "Clip skip 2"]);
  assert.deepEqual(importNotApplied(r, "flux", true), ["Clip skip 2"]);
  assert.deepEqual(importNotApplied({ prompt: "p", clip_skip: 1 }, "flux"), []);
  assert.deepEqual(unknownRecipeKeys(r), []);
});

test("the fields reader unquotes, keeps JSON whole and skips non-fields like re_param", () => {
  assert.deepEqual(
    a1111Fields(
      'Steps: 20, Lora hashes: "a: 1, b: \\"2\\"", Hashes: {"model": "x, y"}, junk, Seed: 5',
    ),
    [
      ["Steps", "20"],
      ["Lora hashes", 'a: 1, b: "2"'],
      ["Hashes", '{"model": "x, y"}'],
      ["Seed", "5"],
    ],
  );
  // An unclosed JSON value falls back to the next comma rather than eating the line.
  assert.deepEqual(a1111Fields("Resources: [oops, Seed: 5"), [
    ["Resources", "[oops"],
    ["Seed", "5"],
  ]);
});

test("Clip skip, CFG rescale, variation seed and hires keys are read and named as not applied", () => {
  const r = parseA1111Parameters(
    "a cat\nSteps: 25, Sampler: Euler a, CFG scale: 5, Seed: 1, Size: 1024x1024, Clip skip: 3, CFG Rescale: 0.7, Variation seed: 99, Variation seed strength: 0.3, Hires upscale: 1.5, Hires steps: 10, Hires upscaler: R-ESRGAN 4x+, Denoising strength: 0.4, VAE: sdxl_vae.safetensors, ADetailer model: face_yolov8n.pt",
  );
  assert.ok(r);
  assert.equal(r.clip_skip, 3);
  assert.equal(r.cfg_rescale, 0.7);
  assert.equal(r.variation_seed, 99);
  assert.equal(r.variation_strength, 0.3);
  assert.deepEqual(r.hires, {
    "Hires upscale": "1.5",
    "Hires steps": "10",
    "Hires upscaler": "R-ESRGAN 4x+",
  });
  assert.equal(r.strength, undefined);
  assert.equal(r.workflow, undefined);
  assert.deepEqual(importNotApplied(r), [
    "Clip skip 3",
    "CFG rescale 0.7",
    "variation seed 99 (strength 0.3)",
    "hires fix (Hires upscale: 1.5, Hires steps: 10, Hires upscaler: R-ESRGAN 4x+)",
    "VAE: sdxl_vae.safetensors",
    "ADetailer model: face_yolov8n.pt",
  ]);
  assert.deepEqual(unknownRecipeKeys(r), []);
  for (const label of [
    "RescaleCFG: 0.5",
    "CFG rescale phi: 0.5",
    "latent_modifier_rescale_cfg_phi: 0.5",
  ]) {
    assert.equal(
      parseA1111Parameters(`a\nSteps: 20, Seed: 1, ${label}`)?.cfg_rescale,
      0.5,
      label,
    );
  }
  // A1111 omits Clip skip at 1 and writes no variation strength 0 (modules/processing.py:785-790).
  assert.deepEqual(
    importNotApplied(
      parseA1111Parameters("a\nSteps: 20, Clip skip: 1, Seed: 1") ?? {
        prompt: "",
      },
    ),
    [],
  );
  assert.deepEqual(
    importNotApplied({ prompt: "p", variation_seed: 4, variation_strength: 0 }),
    [],
  );
});

test("inpaint mask blur, only-masked area and padding restore within the request bounds", () => {
  const r = parseA1111Parameters(
    "a cat\nSteps: 20, CFG scale: 7, Seed: 1, Denoising strength: 0.75, Mask blur: 4, Inpaint area: Only masked, Masked area padding: 48",
  );
  assert.ok(r);
  assert.equal(r.mask_blur, 4);
  assert.equal(r.inpaint_only_masked, true);
  assert.equal(r.inpaint_padding, 48);
  assert.equal(r.workflow, "inpaint");
  assert.equal(
    parseA1111Parameters("a\nSteps: 20, Seed: 1, Denoising strength: 0.5, Mask blur: 4")
      ?.workflow,
    "inpaint",
  );
  const settings = recipeSettings(r, DEFAULTS);
  assert.equal(settings.mask_blur, 4);
  assert.equal(settings.inpaint_only_masked, true);
  assert.equal(settings.inpaint_padding, 48);
  assert.equal(
    parseA1111Parameters("a\nSteps: 20, Seed: 1, Inpaint area: Whole picture")
      ?.inpaint_only_masked,
    false,
  );
  const wild = recipeSettings(
    {
      prompt: "p",
      mask_blur: 500,
      inpaint_padding: -3,
      inpaint_only_masked: "yes",
    },
    DEFAULTS,
  );
  assert.equal(wild.mask_blur, 64);
  assert.equal(wild.inpaint_padding, 0);
  assert.equal(wild.inpaint_only_masked, undefined);
  const old = recipeSettings({ prompt: "p" }, DEFAULTS);
  assert.equal(old.mask_blur, undefined);
  assert.equal(old.inpaint_only_masked, undefined);
});

test("LoRA tags: positional and named weights, other networks stripped and reported", () => {
  const r = extractExtraNetworks(
    "<lora:a:0.5>, cat, <lora:b>, <LORA:c:0.4:0.9>, <lora:d:te=0.3:unet=0.6>, <hypernet:h:1>, dog,",
  );
  assert.equal(r.prompt, "cat, dog");
  assert.deepEqual(r.loras, [
    { name: "a", weight: 0.5 },
    { name: "b", weight: 1 },
    { name: "c", weight: 0.4, unet_weight: 0.9 },
    { name: "d", weight: 0.3, unet_weight: 0.6 },
  ]);
  assert.deepEqual(r.other, ["<hypernet:h:1>"]);
  const plain = "a  cat ,, on a mat";
  assert.equal(extractExtraNetworks(plain).prompt, plain);
  const parsed = parseA1111Parameters("cat <hypernet:h:1>\nSteps: 20, Seed: 1");
  assert.deepEqual(importNotApplied(parsed ?? { prompt: "" }), [
    "<hypernet:h:1>",
  ]);
});

test("LoRA tags select installed LoRAs by file name; the rest are named", () => {
  const available = [
    { id: "add-detail-xl", display_name: "add-detail-xl", source: "local" },
    {
      id: "styles/Pixel_Art.safetensors",
      display_name: "Pixel_Art",
      source: "local",
    },
    {
      id: "unsloth/krea2-retroanime",
      display_name: "Retro anime",
      source: "hub",
    },
  ];
  const m = matchLoraTags(
    [
      { name: "add-detail-xl", weight: 0.8 },
      { name: "pixel_art", weight: 3 },
      { name: "krea2-retroanime", weight: 1 },
      { name: "missing", weight: 1 },
      { name: "two", weight: 1, unet_weight: 0.5 },
    ],
    available,
  );
  assert.deepEqual(m.loras, [
    "add-detail-xl:0.8",
    "styles/Pixel_Art.safetensors:2",
  ]);
  assert.deepEqual(m.unmatched, ["krea2-retroanime", "missing", "two"]);
  assert.deepEqual(m.notes, ["LoRA pixel_art weight 3 (clamped to 2)"]);
  assert.deepEqual(matchLoraTags(undefined, available), {
    loras: [],
    unmatched: [],
    notes: [],
  });
  const settings = recipeSettings({ prompt: "p", loras: m.loras }, DEFAULTS);
  assert.deepEqual(settings.loras, [
    "add-detail-xl:0.8",
    "styles/Pixel_Art.safetensors:2",
  ]);
});

test("pasted text needs a settings line; a bare prompt is not parameters", () => {
  assert.equal(pastedParameters("just a cat on a mat"), null);
  assert.equal(pastedParameters("style: noir, mood: dark, light: low"), null);
  assert.equal(pastedParameters("a cat\nSteps: 20")?.steps, 20);
  assert.equal(
    pastedParameters("a cat\nSampler: Euler, CFG scale: 7, Seed: 3")?.seed,
    3,
  );
  const crlf = pastedParameters(
    "a cat\r\nNegative prompt: dog\r\nSteps: 12, Seed: 9",
  );
  assert.equal(crlf?.negative_prompt, "dog");
  assert.equal(crlf?.steps, 12);
  assert.equal(
    pastedParameters(JSON.stringify({ prompt: "x", steps: 4 }))?.steps,
    4,
  );
  assert.equal(pastedParameters("{not json"), null);
  assert.deepEqual(
    pastedParameters(CIVITAI),
    importedRecipe({ parameters: CIVITAI }),
  );
});

test("only a whole parameters block pasted into the prompt box is taken over", () => {
  assert.equal(isGenerationParametersPaste(CIVITAI), true);
  assert.equal(isGenerationParametersPaste("a cat\nSteps: 20, Seed: 1"), true);
  assert.equal(
    isGenerationParametersPaste("a cat, Steps: 20 of a staircase"),
    false,
  );
  assert.equal(isGenerationParametersPaste("a cat\nSteps: 20"), false);
  assert.equal(isGenerationParametersPaste("masterpiece, (cat:1.2)"), false);
});

test("the Images page wires paste into the same restore path as a drop", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /const applyImportedRecipe = useCallback\(/);
  assert.match(
    page,
    /const importSettingsFromText = useCallback\(\n\s+async \(text: string\): Promise<boolean> => \{\n\s+const recipe = pastedParameters\(text\);[\s\S]*?return applyImportedRecipe\(recipe\);\n\s+\},\n\s+\[applyImportedRecipe\],/,
  );
  assert.match(page, /const recipe = pastedParameters\(text\);/);
  assert.match(page, /onPasteParameters=\{\(\) => setPasteOpen\(true\)\}/);
  assert.match(
    page,
    /if \(isGenerationParametersPaste\(text\)\) \{\n\s+e\.preventDefault\(\);\n\s+void importSettingsFromText\(text\);/,
  );
  assert.match(
    page,
    /\.\.\.importNotApplied\(\n\s+recipe,\n\s+status\?\.loaded \? status\.family : null,\n\s+Boolean\(status\?\.loaded && isNativeEngineStatus\(status\)\),\n\s+sdCaps,\n\s+\),/,
  );
  assert.match(
    page,
    /tagged\.unmatched\.map\(\(name\) => `LoRA \$\{name\} \(not installed\)`\)/,
  );
  const control = readSrc(
    "features/generation-presets/media-generation-preset-control.tsx",
  );
  assert.match(control, /Paste generation parameters/);
});

test("LoRA tags pasted before a LoRA-capable model is resident match the whole local catalog", async () => {
  const local = [{ id: "add-detail-xl", display_name: "Add detail", source: "local" }];
  const tags = [
    { name: "add-detail-xl", weight: 0.8 },
    { name: "missing", weight: 1 },
  ];
  let fetched = 0;
  const fetchAll = async () => {
    fetched += 1;
    return local;
  };
  // No catalog on the page yet: the installed adapter is found, not reported missing.
  assert.deepEqual(await matchLoraTagsForImport(tags, false, [], fetchAll), {
    loras: ["add-detail-xl:0.8"],
    unmatched: ["missing"],
    notes: [],
  });
  assert.equal(fetched, 1);
  // A resident catalog is used as is, with no extra request.
  assert.deepEqual(
    await matchLoraTagsForImport(tags, true, local, fetchAll),
    matchLoraTags(tags, local),
  );
  assert.deepEqual(await matchLoraTagsForImport(undefined, false, [], fetchAll), {
    loras: [],
    unmatched: [],
    notes: [],
  });
  assert.equal(fetched, 1);
  // The list cannot be read: nothing is called missing.
  const failed = await matchLoraTagsForImport(tags, false, [], async () => {
    throw new Error("offline");
  });
  assert.deepEqual(failed.unmatched, []);
  assert.deepEqual(failed.notes, [
    "LoRA add-detail-xl (the installed LoRAs could not be listed)",
    "LoRA missing (the installed LoRAs could not be listed)",
  ]);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /let tagged = await matchLoraTagsForImport\(\n\s+recipe\.lora_tags,\n\s+\/\/[^\n]*\n\s+loraCapable && loraCatalogFamily === \(status\?\.family \?\? null\),\n\s+availableLoras,\n\s+\/\/[^\n]*\n\s+\(\) => listDiffusionLoras\(status\?\.loaded \? \(status\.family \?\? undefined\) : undefined\),\n\s+\);/,
  );
  // The resident catalog counts only once the fetch for the current family has landed.
  assert.match(
    page,
    /setLoraCatalogFamily\(undefined\);\n\s+if \(!loraCapable\) \{/,
  );
  assert.match(
    page,
    /if \(cancelled\) return;\n\s+setAvailableLoras\(list\);\n\s+setLoraCatalogFamily\(fam\);/,
  );
});

test("LoRAs past the request's adapter limit are named as not applied", () => {
  const loras = Array.from({ length: MAX_LORAS + 2 }, (_, i) => `lora${i}:1`);
  const settings = recipeSettings({ prompt: "p", loras }, DEFAULTS);
  assert.equal(settings.loras.length, MAX_LORAS);
  assert.deepEqual(lorasOverLimit({ prompt: "p", loras }), [
    `lora${MAX_LORAS}`,
    `lora${MAX_LORAS + 1}`,
  ]);
  // Duplicates and invalid rows are not counted against the limit.
  assert.deepEqual(
    lorasOverLimit({ prompt: "p", loras: [...loras.slice(0, MAX_LORAS), "lora0:0.5", "bad"] }),
    [],
  );
  assert.deepEqual(lorasOverLimit({ prompt: "p" }), []);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /const settings = recipeSettings\(merged, \{ width, height, steps, guidance \}\);/);
  assert.match(
    page,
    /\.\.\.lorasOverLimit\(merged\)\.map\(\(id\) => `LoRA \$\{id\} \(over the \$\{MAX_LORAS\} adapter limit\)`\),/,
  );
});

test("a dynamic-prompts prompt starting with { still parses as A1111 parameters", () => {
  const text = "{red|blue} car\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 5, Size: 512x512";
  const r = pastedParameters(text);
  assert.ok(r);
  assert.equal(r.prompt, "{red|blue} car");
  assert.equal(r.seed, 5);
  assert.equal(isGenerationParametersPaste(text), true);
  assert.equal(pastedParameters("{not json"), null);
});

test("a DyLoRA dimension is named as not applied rather than dropped", () => {
  const positional = extractExtraNetworks("a cat <lora:dy:0.8:0.8:16>");
  assert.deepEqual(positional.loras, [{ name: "dy", weight: 0.8, dyn_dim: 16 }]);
  const named = extractExtraNetworks("a cat <lora:dy:te=0.5:unet=0.7:dyn=8>");
  assert.deepEqual(named.loras, [{ name: "dy", weight: 0.5, unet_weight: 0.7, dyn_dim: 8 }]);
  assert.deepEqual(extractExtraNetworks("<lora:dy:1>").loras, [{ name: "dy", weight: 1 }]);
  const m = matchLoraTags(positional.loras, [{ id: "dy", display_name: "dy" }]);
  assert.deepEqual(m.loras, ["dy:0.8"]);
  assert.deepEqual(m.notes, ["LoRA dy dynamic dim 16"]);
});

test("a non-default A1111 Emphasis mode is named as not applied; Original stays silent", () => {
  const parse = (emphasis: string) =>
    importNotApplied(
      parseA1111Parameters(`a cat\nSteps: 20, Seed: 1, Emphasis: ${emphasis}`) ?? { prompt: "" },
      "sdxl",
    );
  assert.deepEqual(parse("No norm"), ["Emphasis: No norm"]);
  assert.deepEqual(parse("Ignore"), ["Emphasis: Ignore"]);
  assert.deepEqual(parse("Original"), []);
});

test("a paste is reported applied only once its settings land, and a later import wins", () => {
  const page = readSrc("features/images/images-page.tsx");
  // A sequence token taken before the LoRA lookup: an earlier import finishing late changes nothing.
  assert.match(
    page,
    /async \(recipe: ImportedRecipe\): Promise<boolean> => \{\n\s+const seq = \+\+importSeq\.current;\n\s+const modelBefore = loadedModelKey\(status\);\n\s+let tagged = await matchLoraTagsForImport\(/,
  );
  assert.match(
    page,
    /\(\) => listDiffusionLoras\([^\n]*\),\n\s+\);\n\s+if \(seq !== importSeq\.current\) return false;/,
  );
  assert.match(page, /\n\s+return true;\n\s+\},\n\s+\[restoreSettings, width, height,/);
  assert.match(page, /await applyImportedRecipe\(recipe\);\n\s+\},\n\s+\[applyImportedRecipe\],/);
  // The dialog clears and closes only after the awaited apply succeeded.
  const dialog = readSrc("features/images/paste-parameters-dialog.tsx");
  assert.match(dialog, /onApply: \(text: string\) => Promise<boolean>;/);
  assert.match(
    dialog,
    /const apply = async \(\) => \{\n\s+if \(text\.trim\(\) && \(await onApply\(text\)\)\) \{\n\s+onTextChange\(""\);\n\s+onOpenChange\(false\);/,
  );
});

test("LoRA matches from before a model switch during the lookup are not restored", () => {
  assert.deepEqual(loraTagNames([{ name: "a", weight: 1 }, { weight: 1 }, null, { name: "b", weight: 0.5 }]), [
    "a",
    "b",
  ]);
  assert.deepEqual(loraTagNames(undefined), []);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /const statusRef = useRef\(status\);\n\s+statusRef\.current = status;/);
  assert.match(
    page,
    /const seq = \+\+importSeq\.current;\n\s+const modelBefore = loadedModelKey\(status\);\n\s+let tagged = await matchLoraTagsForImport\(/,
  );
  assert.match(
    page,
    /if \(seq !== importSeq\.current\) return false;\n[^\n]*\n\s+if \(loadedModelKey\(statusRef\.current\) !== modelBefore\) \{\n\s+tagged = \{\n\s+loras: \[\],\n\s+unmatched: \[\],\n\s+notes: loraTagNames\(recipe\.lora_tags\)\.map\(\(n\) => `LoRA \$\{n\} \(model changed during import\)`\),/,
  );
});

test("steps past the request limit are reported as clamped", () => {
  const r = parseA1111Parameters("a cat\nSteps: 150, Seed: 1");
  assert.ok(r);
  assert.equal(recipeSettings(r, DEFAULTS).steps, 100);
  assert.deepEqual(importNotApplied(r, "sdxl"), ["Steps 150 (clamped to 100)"]);
  assert.deepEqual(importNotApplied({ prompt: "p", steps: 0 }, "sdxl"), ["Steps 0 (clamped to 1)"]);
  assert.deepEqual(importNotApplied({ prompt: "p", steps: 100 }, "sdxl"), []);
});

test("a LoRA tag takes the exact path first and never guesses between same-named adapters", () => {
  const available = [
    { id: "sdxl/detail.safetensors", display_name: "detail", source: "local" },
    { id: "flux/detail.safetensors", display_name: "detail", source: "local" },
    { id: "styles/Pixel_Art.safetensors", display_name: "Pixel_Art", source: "local" },
  ];
  const m = matchLoraTags(
    [
      { name: "flux\\detail", weight: 1 },
      { name: "detail", weight: 0.5 },
      { name: "pixel_art", weight: 1 },
    ],
    available,
  );
  assert.deepEqual(m.loras, ["flux/detail.safetensors:1", "styles/Pixel_Art.safetensors:1"]);
  assert.deepEqual(m.unmatched, []);
  assert.deepEqual(m.notes, ["LoRA detail (ambiguous: 2 adapters named detail)"]);
});

test("each textual-inversion embedding in TI hashes is named as not applied", () => {
  const r = parseA1111Parameters(
    'a cat, easynegative\nSteps: 20, Seed: 1, TI hashes: "easynegative: c74b4e810b03, badhandv4: 5e40d722fc3d"',
  );
  assert.ok(r);
  assert.deepEqual(importNotApplied(r, "sdxl"), [
    "textual inversion easynegative",
    "textual inversion badhandv4",
  ]);
});

test("the model identity an import checks covers the engine and build, not only repo and family", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /const loadedModelKey = \(s: DiffusionStatus \| null\): string \| null =>\n\s+s\?\.loaded\n\s+\? JSON\.stringify\(\[\n\s+s\.repo_id,\n\s+s\.base_repo,\n\s+s\.family,\n\s+s\.engine,\n\s+s\.gguf_filename,\n\s+s\.dtype,\n\s+s\.transformer_quant,\n\s+s\.text_encoder_quant,\n\s+\]\)\n\s+: null;/,
  );
});
