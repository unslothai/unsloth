// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DETAILER_DEFAULTS,
  detailerAvailable,
  detailerEngagesNegative,
  detailerErrors,
  detailerFromInfotext,
  detailerModelCapable,
  detailerNotApplied,
  detailerRecordOf,
  detailerRequestField,
  detailerSummary,
  readDetailerOptions,
  restoredDetailer,
} from "../src/features/images/lib/detailer.ts";
import {
  APPLIED_RECIPE_KEYS,
  importNotApplied,
  parseA1111Parameters,
  recipeSettings,
} from "../src/features/images/lib/image-metadata.ts";

const inpaintModel = { loaded: true, workflows: ["txt2img", "img2img", "inpaint", "outpaint"] };
const ready = { supported: true, downloaded: true };

test("the Detailer shows only for a model with an inpaint path and a downloaded SAM 3", () => {
  assert.equal(detailerAvailable(inpaintModel, false, ready), true);
  // No inpaint pipeline (FLUX.2 dev, edit models, sd.cpp): hidden.
  assert.equal(detailerAvailable({ loaded: true, workflows: ["txt2img", "reference"] }, false, ready), false);
  assert.equal(detailerAvailable({ loaded: true, workflows: ["edit"] }, false, ready), false);
  // Nothing loaded, or a Connection (external provider): hidden.
  assert.equal(detailerAvailable({ loaded: false, workflows: ["inpaint"] }, false, ready), false);
  assert.equal(detailerAvailable(inpaintModel, true, ready), false);
  assert.equal(detailerAvailable(null, false, ready), false);
  // SAM 3 unsupported (transformers 4) or not downloaded, or its status not read yet: hidden.
  assert.equal(detailerAvailable(inpaintModel, false, { supported: false, downloaded: false }), false);
  assert.equal(detailerAvailable(inpaintModel, false, { supported: true, downloaded: false }), false);
  assert.equal(detailerAvailable(inpaintModel, false, null), false);
  // The segmenter status is only fetched for a capable model.
  assert.equal(detailerModelCapable(inpaintModel, false), true);
  assert.equal(detailerModelCapable({ loaded: true, workflows: ["txt2img"] }, false), false);
});

test("off by default, and nothing is sent unless on and available", () => {
  assert.deepEqual(detailerRequestField(false, true, DETAILER_DEFAULTS), {});
  assert.deepEqual(detailerRequestField(true, false, DETAILER_DEFAULTS), {});
  // Defaults send only the targets: the backend holds the same FaceDetailer defaults.
  assert.deepEqual(detailerRequestField(true, true, DETAILER_DEFAULTS), { detailer: { targets: "face" } });
  // Node has no localStorage: the stored options read as the defaults.
  assert.deepEqual(readDetailerOptions(), DETAILER_DEFAULTS);
});

test("changed options are sent, bounded to the request schema", () => {
  const field = detailerRequestField(true, true, {
    ...DETAILER_DEFAULTS,
    targets: "  ",
    prompt: " detailed face ",
    denoise: 0.35,
    cropFactor: 99,
    guideSize: 768,
    feather: 5,
    steps: 12.4,
  });
  assert.deepEqual(field, {
    detailer: { targets: "face", prompt: "detailed face", denoise: 0.35, crop_factor: 10, guide_size: 768, steps: 12 },
  });
});

test("a recipe's non-default guide_size_for and drop_size survive restore and are sent again", () => {
  const restored = restoredDetailer({ detailer: { ...record, guide_size_for: "crop_region", drop_size: 32 } });
  assert.equal(restored.options.guideSizeFor, "crop_region");
  assert.equal(restored.options.dropSize, 32);
  const field = detailerRequestField(true, true, { ...DETAILER_DEFAULTS, ...restored.options });
  assert.equal(field.detailer?.guide_size_for, "crop_region");
  assert.equal(field.detailer?.drop_size, 32);
  // A record without them restores FaceDetailer's defaults, never a stale stored value.
  const plain = restoredDetailer({ detailer: { targets: "face" } });
  assert.deepEqual([plain.options.guideSizeFor, plain.options.dropSize], ["bbox", 10]);
});

test("the frontend defaults are the backend's", () => {
  const source = readFileSync(
    new URL("../../backend/core/inference/diffusion_detailer.py", import.meta.url),
    "utf8",
  );
  const constant = (name: string) => Number(new RegExp(`^${name} = ([\\d.]+)`, "m").exec(source)?.[1]);
  assert.equal(constant("DEFAULT_DENOISE"), DETAILER_DEFAULTS.denoise);
  assert.equal(constant("DEFAULT_CONFIDENCE"), DETAILER_DEFAULTS.confidence);
  assert.equal(constant("DEFAULT_CROP_FACTOR"), DETAILER_DEFAULTS.cropFactor);
  assert.equal(constant("DEFAULT_GUIDE_SIZE"), DETAILER_DEFAULTS.guideSize);
  assert.equal(constant("DEFAULT_MAX_SIZE"), DETAILER_DEFAULTS.maxSize);
  assert.equal(constant("DEFAULT_DILATION"), DETAILER_DEFAULTS.dilation);
  assert.equal(constant("DEFAULT_FEATHER"), DETAILER_DEFAULTS.feather);
  assert.equal(constant("DEFAULT_DROP_SIZE"), DETAILER_DEFAULTS.dropSize);
  assert.match(source, new RegExp(`^DEFAULT_GUIDE_SIZE_FOR = "${DETAILER_DEFAULTS.guideSizeFor}"`, "m"));
});

const record = {
  targets: "face, hand:2",
  prompt: "sharp eyes",
  confidence: 0.5,
  denoise: 0.4,
  crop_factor: 2.5,
  guide_size: 512,
  guide_size_for: "bbox",
  max_size: 1024,
  dilation: 10,
  feather: 5,
  drop_size: 10,
  steps: 12,
  model: "sam3.1",
  detections: [{ target: "face", score: 0.91, box: [1, 2, 3, 4], seed: 7 }],
  found: 1,
};

test("a gallery record restores the pass and turns the switch on; one without turns it off", () => {
  const on = restoredDetailer({ detailer: record });
  assert.equal(on.enabled, true);
  assert.deepEqual(on.options, {
    targets: "face, hand:2",
    prompt: "sharp eyes",
    denoise: 0.4,
    confidence: 0.5,
    cropFactor: 2.5,
    guideSize: 512,
    maxSize: 1024,
    dilation: 10,
    feather: 5,
    dropSize: 10,
    guideSizeFor: "bbox",
    steps: 12,
    negativePrompt: "",
    guidance: null,
  });
  assert.deepEqual(restoredDetailer({}), { enabled: false, options: {} });
  assert.deepEqual(restoredDetailer({ detailer: "nonsense" }), { enabled: false, options: {} });
  assert.deepEqual(detailerRecordOf({ denoise: "0.4", targets: 3, steps: 9 }), { steps: 9 });
});

test("the Unsloth recipe carries the record through import", () => {
  const settings = recipeSettings(
    { prompt: "p", detailer: record },
    { width: 1024, height: 1024, steps: 9, guidance: 0 },
  );
  assert.equal(settings.detailer?.denoise, 0.4);
  assert.ok((APPLIED_RECIPE_KEYS as readonly string[]).includes("detailer"));
  assert.equal(recipeSettings({ prompt: "p" }, { width: 1024, height: 1024, steps: 9, guidance: 0 }).detailer, null);
});

// What core/inference/image_gallery._params_text writes with a pass (diffusion_detailer.infotext_fields).
const STUDIO_INFOTEXT = [
  "a portrait of two friends",
  "Steps: 9, CFG scale: 0.0, Seed: 42, Size: 1024x1024, Model: Tongyi-MAI/Z-Image-Turbo, ADetailer model: sam3.1, " +
    'ADetailer model classes: face, ADetailer prompt: "detailed face, smile", ADetailer confidence: 0.5, ' +
    "ADetailer dilate erode: 10, ADetailer mask blur: 5, ADetailer denoising strength: 0.4, " +
    "ADetailer inpaint only masked: True, ADetailer use separate steps: True, ADetailer steps: 12, Version: Unsloth",
].join("\n");

test("Studio's own ADetailer infotext pastes back as the same pass", () => {
  const parsed = parseA1111Parameters(STUDIO_INFOTEXT);
  assert.ok(parsed);
  assert.deepEqual(parsed.detailer, {
    model: "sam3.1",
    targets: "face",
    prompt: "detailed face, smile",
    confidence: 0.5,
    dilation: 10,
    feather: 5,
    denoise: 0.4,
    steps: 12,
  });
  // Not reported as unapplied, and the main image stays text-to-image (ADetailer's strength is not img2img's).
  assert.equal(parsed.parameters_not_applied, undefined);
  assert.equal(parsed.workflow, undefined);
  assert.equal(parsed.strength, undefined);
  assert.deepEqual(importNotApplied(parsed, "z-image", false), []);
  const restored = restoredDetailer(recipeSettings(parsed, { width: 1024, height: 1024, steps: 9, guidance: 0 }));
  assert.equal(restored.enabled, true);
  assert.equal(restored.options.denoise, 0.4);
  assert.equal(restored.options.steps, 12);
});

test("an AUTOMATIC1111 ADetailer infotext maps its YOLO model to a target", () => {
  const parsed = parseA1111Parameters(
    "girl\nSteps: 28, Sampler: DPM++ 2M, CFG scale: 7, Seed: 1, Size: 512x768, ADetailer model: face_yolov8n.pt, " +
      "ADetailer confidence: 0.3, ADetailer dilate erode: 4, ADetailer mask blur: 4, ADetailer denoising strength: 0.4, " +
      "ADetailer inpaint only masked: True, ADetailer inpaint padding: 32, ADetailer model 2nd: hand_yolov8n.pt, " +
      "ADetailer version: 24.9.0",
  );
  assert.ok(parsed);
  assert.deepEqual(parsed.detailer, {
    model: "face_yolov8n.pt",
    targets: "face",
    confidence: 0.3,
    dilation: 4,
    feather: 4,
    denoise: 0.4,
  });
  // Only the first tab restores: a later tab and the options Studio has no control for are named, not dropped.
  assert.deepEqual(parsed.parameters_not_applied, [
    "ADetailer inpaint padding: 32",
    "ADetailer model 2nd: hand_yolov8n.pt",
  ]);
  assert.deepEqual(
    parseA1111Parameters("a\nSteps: 20, Seed: 1, ADetailer model: None, ADetailer prompt: eyes")?.parameters_not_applied,
    ["ADetailer prompt: eyes"],
  );
  assert.equal(detailerFromInfotext({ "adetailer model": "hand_yolov8s.pt" })?.targets, "hand");
  assert.equal(detailerFromInfotext({ "adetailer model": "None" }), null);
  assert.equal(detailerFromInfotext({}), null);
});

test("a restored or imported pass this load cannot run is named as not applied", () => {
  assert.deepEqual(detailerNotApplied(true, false), ["Detailer (face fix)"]);
  assert.deepEqual(detailerNotApplied(true, true), []);
  assert.deepEqual(detailerNotApplied(false, false), []);
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  // The gallery restore names it beside the SD controls, as an import does.
  assert.match(page, /\.\.\.\(via === "restore" \? detailerNotApplied\(detail\.enabled, detailerReady\) : \[\]\)/);
  assert.match(page, /notApplied\.unshift\(\.\.\.detailerNotApplied\(Boolean\(settings\.detailer\), detailerReady\)\)/);
});

test("errors are surfaced, a cancel is not", () => {
  assert.deepEqual(
    detailerErrors([{ detailer: { error: "cancelled" } }, { detailer: { error: "CUDA out of memory" } }, {}]),
    ["CUDA out of memory"],
  );
  assert.equal(detailerSummary(record), "face, hand:2: 1 fixed, denoise 0.4");
  assert.equal(detailerSummary({ error: "the model changed" }), "not applied (the model changed)");
  assert.equal(detailerSummary(null), null);
});

test("the page wires the control behind the gate and keeps it out of the startup bundle", () => {
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  assert.match(page, /<DetailerField\s+available=\{detailerReady\}/);
  assert.match(page, /detailerRequestField\(detailerOn, detailerReady, detailerOptions\)/);
  assert.match(page, /useState\(false\);\n\s*const \[detailerOptions/);
  // The field is a separate module of the lazily loaded Images page; api.ts only takes its types.
  const api = readFileSync(new URL("../src/features/images/api.ts", import.meta.url), "utf8");
  assert.match(api, /^import type \{ DetailerRecord, DetailerRequest \} from "\.\/lib\/detailer";$/m);
  const root = readFileSync(new URL("../src/app/routes/__root.tsx", import.meta.url), "utf8");
  assert.match(root, /import\("@\/features\/images"\)/);
});

test("a recorded Detailer negative prompt and CFG restore and are sent again (ADetailer's separate ones)", () => {
  const restored = restoredDetailer({ detailer: { ...record, negative_prompt: "blurry eyes", guidance: 4.5 } });
  assert.equal(restored.options.negativePrompt, "blurry eyes");
  assert.equal(restored.options.guidance, 4.5);
  const field = detailerRequestField(true, true, { ...DETAILER_DEFAULTS, ...restored.options });
  assert.equal(field.detailer?.negative_prompt, "blurry eyes");
  assert.equal(field.detailer?.guidance, 4.5);
  // From an AUTOMATIC1111 infotext: separate negative prompt and CFG scale.
  const parsed = parseA1111Parameters(
    "girl\nSteps: 28, CFG scale: 7, Seed: 1, Size: 512x768, ADetailer model: face_yolov8n.pt, " +
      "ADetailer negative prompt: blurry, ADetailer denoising strength: 0.4, " +
      "ADetailer use separate CFG scale: True, ADetailer CFG scale: 5.5",
  );
  assert.ok(parsed);
  const fromPaste = restoredDetailer(recipeSettings(parsed, { width: 512, height: 768, steps: 28, guidance: 7 }));
  const sent = detailerRequestField(true, true, { ...DETAILER_DEFAULTS, ...fromPaste.options }).detailer;
  assert.equal(sent?.negative_prompt, "blurry");
  assert.equal(sent?.guidance, 5.5);
  // A record without them clears an override left from an earlier restore.
  const plain = restoredDetailer({ detailer: record });
  assert.equal(plain.options.negativePrompt, "");
  assert.equal(plain.options.guidance, null);
  assert.equal(detailerRequestField(true, true, DETAILER_DEFAULTS).detailer?.guidance, undefined);
});

test("the Detailer control learns about a finished SAM 3 download without a reload", () => {
  const api = readFileSync(new URL("../src/features/images/mask-segment-api.ts", import.meta.url), "utf8");
  // Every status read (the Mask from text panel polls while it downloads) is broadcast...
  assert.match(api, /dispatchEvent\(new CustomEvent\(SEGMENTER_STATUS_EVENT, \{ detail: status \}\)\)/);
  const field = readFileSync(new URL("../src/features/images/detailer-field.tsx", import.meta.url), "utf8");
  // ...the Detailer's hook takes it, and polls on its own while a download runs.
  assert.match(field, /addEventListener\(SEGMENTER_STATUS_EVENT, onStatus\)/);
  assert.match(field, /status\?\.download\.downloading === true/);
  assert.match(field, /setInterval\(read, 2000\)/);
});

test("an imported default ADetailer run does not keep geometry stored from an earlier run", () => {
  // The page merges a restore over the stored options; an infotext carries no crop factor, guide size or max size.
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  assert.match(page, /setDetailerOptions\(\{ \.\.\.readDetailerOptions\(\), \.\.\.detail\.options \}\)/);
  const stored = { ...DETAILER_DEFAULTS, cropFactor: 6, guideSize: 1536, maxSize: 2048, guideSizeFor: "crop_region" as const };
  const parsed = parseA1111Parameters(STUDIO_INFOTEXT);
  assert.ok(parsed);
  const restored = restoredDetailer(recipeSettings(parsed, { width: 1024, height: 1024, steps: 9, guidance: 0 }));
  const sent = detailerRequestField(true, true, { ...stored, ...restored.options }).detailer;
  assert.ok(sent);
  assert.equal(sent.crop_factor, undefined);
  assert.equal(sent.guide_size, undefined);
  assert.equal(sent.max_size, undefined);
  assert.equal(sent.guide_size_for, undefined);
  assert.equal(sent.denoise, 0.4);
  assert.equal(sent.steps, 12);
  const custom = restoredDetailer({ detailer: { targets: "face", crop_factor: 4, guide_size: 768, max_size: 1536 } });
  const resent = detailerRequestField(true, true, { ...DETAILER_DEFAULTS, ...custom.options }).detailer;
  assert.equal(resent?.crop_factor, 4);
  assert.equal(resent?.guide_size, 768);
  assert.equal(resent?.max_size, 1536);
});

test("the Detailer's own CFG sends the main negative prompt it inherits", () => {
  // Base CFG 0 leaves the negative prompt unused; a Detailer CFG of 5 with no separate negative reads it.
  const cfg = { ...DETAILER_DEFAULTS, guidance: 5 };
  assert.equal(detailerEngagesNegative(true, true, cfg, 1), true);
  assert.equal(detailerEngagesNegative(true, true, cfg, null), true);
  // Not when off, unavailable, using the image's CFG, at or under the threshold, or with its own negative prompt.
  assert.equal(detailerEngagesNegative(false, true, cfg, 1), false);
  assert.equal(detailerEngagesNegative(true, false, cfg, 1), false);
  assert.equal(detailerEngagesNegative(true, true, DETAILER_DEFAULTS, 1), false);
  assert.equal(detailerEngagesNegative(true, true, { ...cfg, guidance: 1 }, 1), false);
  assert.equal(detailerEngagesNegative(true, true, { ...cfg, negativePrompt: "blurry" }, 1), false);
  // A separate negative template that reads the image's own ([PROMPT], an empty [SEP] part) still needs it sent.
  assert.equal(detailerEngagesNegative(true, true, { ...cfg, negativePrompt: "[PROMPT], bad teeth" }, 1), true);
  assert.equal(detailerEngagesNegative(true, true, { ...cfg, negativePrompt: " [SEP] bad teeth" }, 1), true);
  assert.equal(detailerEngagesNegative(true, true, { ...cfg, negativePrompt: "bad teeth [SEP] [SKIP]" }, 1), false);
  // The page's gate reads it, after the Detailer state it depends on.
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  const gate = page.indexOf("const negativeEngaged =");
  assert.ok(gate > page.indexOf("const detailerReady ="));
  assert.match(
    page.slice(gate, gate + 600),
    /detailerEngagesNegative\(detailerOn, detailerReady, detailerOptions, negativeMinGuidance\)/,
  );
  assert.match(page, /negative_prompt: negativeEngaged \? negativePrompt\.trim\(\) \|\| undefined : undefined/);
});
