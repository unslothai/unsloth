// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  A1111_INFO_KEYS,
  importNotApplied,
  parseA1111Parameters,
  promptEmphasisGroups,
} from "../src/features/images/lib/image-metadata.ts";

type Handling = "applied" | "reported" | "ignored";

// Every key AUTOMATIC1111 (create_infotext, extra_generation_params, the `infotext=` options) and Forge
// write, with a value that changes the image where one exists. "ignored" is only for keys that set
// nothing: hashes, the writer, the version. Anything not handled falls through to "reported".
const KEYS: Array<[string, string, Handling]> = [
  // create_infotext
  ["Steps", "30", "applied"],
  ["Sampler", "Euler a", "applied"],
  ["Schedule type", "Karras", "applied"],
  ["CFG scale", "6", "applied"],
  ["Image CFG scale", "1.5", "reported"],
  ["Seed", "7", "applied"],
  ["Face restoration", "CodeFormer", "reported"],
  ["Size", "832x1216", "applied"],
  ["Model hash", "31e35c80fc", "ignored"],
  ["Model", "ponyDiffusionV6XL", "applied"],
  ["FP8 weight", "Enable", "reported"],
  ["Cache FP16 weight for LoRA", "True", "reported"],
  ["VAE hash", "235745af8d", "ignored"],
  ["VAE", "sdxl_vae.safetensors", "reported"],
  ["Variation seed", "99", "reported"],
  ["Variation seed strength", "0.3", "reported"],
  ["Seed resize from", "512x512", "reported"],
  ["Denoising strength", "0.5", "applied"],
  ["Conditional mask weight", "0.5", "reported"],
  ["Clip skip", "3", "reported"],
  ["ENSD", "31337", "reported"],
  ["Tiling", "True", "reported"],
  ["Token merging ratio", "0.3", "reported"],
  ["Token merging ratio hr", "0.3", "reported"],
  ["Init image hash", "abc", "ignored"],
  ["RNG", "CPU", "reported"],
  ["Version", "v1.10.1", "ignored"],
  ["User", "someone", "ignored"],
  // extra_generation_params (modules, scripts, extensions-builtin)
  ["Hires upscale", "2", "reported"],
  ["Hires steps", "10", "reported"],
  ["Hires upscaler", "Latent", "reported"],
  ["Hires resize", "1024x1024", "reported"],
  ["Hires checkpoint", "other.safetensors", "reported"],
  ["Hires sampler", "Euler", "reported"],
  ["Hires schedule type", "Karras", "reported"],
  ["Hires prompt", "x", "reported"],
  ["Hires negative prompt", "y", "reported"],
  ["Hires CFG Scale", "5", "reported"],
  ["Hires Distilled CFG Scale", "3.5", "reported"],
  ["Hires Module 1", "Built-in", "reported"],
  ["Hires refiner", "r.safetensors", "reported"],
  ["Refiner", "sd_xl_refiner_1.0", "reported"],
  ["Refiner switch at", "0.8", "reported"],
  ["Refiner switch by sampling steps", "True", "reported"],
  ["Mask blur", "4", "applied"],
  ["Inpaint area", "Only masked", "applied"],
  ["Masked area padding", "48", "applied"],
  ["Masked content", "fill", "reported"],
  ["Mask mode", "Inpaint not masked", "reported"],
  ["Mask rounding", "False", "reported"],
  ["Emphasis", "No norm", "reported"],
  ["Pad conds", "True", "reported"],
  ["Pad conds v0", "True", "reported"],
  ["Downcast alphas_cumprod", "True", "reported"],
  ["Old prompt editing timelines", "True", "reported"],
  ["Discard penultimate sigma", "True", "reported"],
  ["Noise multiplier", "1.05", "reported"],
  ["Extra noise", "0.1", "reported"],
  ["SGM noise multiplier", "True", "reported"],
  ["NGMS", "0.5", "reported"],
  ["NGMS all steps", "True", "reported"],
  ["Skip Early CFG", "0.2", "reported"],
  ["Noise Schedule", "Zero Terminal SNR", "reported"],
  ["Schedule max sigma", "14", "reported"],
  ["Schedule min sigma", "0.03", "reported"],
  ["Schedule rho", "7", "reported"],
  ["Beta schedule alpha", "0.6", "reported"],
  ["Beta schedule beta", "0.6", "reported"],
  ["Sigma churn", "0.1", "reported"],
  ["Sigma tmin", "0.1", "reported"],
  ["Sigma tmax", "10", "reported"],
  ["Sigma noise", "1.01", "reported"],
  ["Eta", "0.67", "reported"],
  ["Eta DDIM", "0.1", "reported"],
  ["UniPC variant", "bh1", "reported"],
  ["UniPC skip type", "time_uniform", "reported"],
  ["UniPC order", "2", "reported"],
  ["UniPC lower order final", "False", "reported"],
  ["VAE Encoder", "TAESD", "reported"],
  ["VAE Decoder", "TAESD", "reported"],
  ["Hypertile U-Net", "True", "reported"],
  ["Hypertile VAE", "True", "reported"],
  ["SD upscale overlap", "64", "reported"],
  ["SD upscale upscaler", "R-ESRGAN 4x+", "reported"],
  ["Script", "X/Y/Z plot", "reported"],
  ["Lora hashes", '"add-detail-xl: 9c783c8ce46c"', "ignored"],
  ["TI hashes", '"easynegative: c74b4e810b03"', "reported"],
  // Forge
  ["Distilled CFG Scale", "3.5", "reported"],
  ["Module 1", "ae", "reported"],
  ["Diffusion in Low Bits", "bnb-nf4", "reported"],
  ["Original Size", "1024x1024", "reported"],
  // Civitai and other writers
  ["Hashes", '{"model": "31e35c80fc"}', "ignored"],
  ["Civitai resources", '[{"type": "checkpoint"}]', "ignored"],
  ["Civitai metadata", "{}", "ignored"],
  ["Created Date", "2024-05-01T00:00:00.000Z", "ignored"],
  // A key nobody lists yet still falls through to "reported", never silently.
  ["Some future setting", "on", "reported"],
];

const BASE = "a cat\nSteps: 20, Seed: 1";

function handling(key: string, value: string): Handling {
  const base = parseA1111Parameters(BASE);
  const r = parseA1111Parameters(`${BASE}, ${key}: ${value}`);
  assert.ok(base && r, key);
  const reported = importNotApplied(r, "sdxl").filter(
    (item) => !importNotApplied(base, "sdxl").includes(item),
  );
  if (reported.length > 0) return "reported";
  const { parameters_not_applied: _a, ...rest } = r;
  const { parameters_not_applied: _b, ...baseRest } = base;
  return JSON.stringify(rest) === JSON.stringify(baseRest) ? "ignored" : "applied";
}

for (const [key, value, expected] of KEYS) {
  test(`A1111 infotext ${key} is ${expected}`, () => {
    assert.equal(handling(key, value), expected);
  });
}

test("only the keys the table calls ignored are skipped silently", () => {
  const ignored = KEYS.filter(([, , h]) => h === "ignored").map(([k]) => k.toLowerCase());
  assert.deepEqual([...A1111_INFO_KEYS].sort(), [...ignored].sort());
  assert.equal(new Set(KEYS.map(([k]) => k.toLowerCase())).size, KEYS.length);
});

test("numbers past the request bounds are reported as clamped", () => {
  const r = parseA1111Parameters(
    "a cat\nSteps: 150, CFG scale: 30, Seed: 1, Denoising strength: 0.5, Mask blur: 99, Masked area padding: 300",
  );
  assert.ok(r);
  assert.deepEqual(importNotApplied(r, "sdxl"), [
    "Steps 150 (clamped to 100)",
    "CFG scale 30 (clamped to 20)",
    "Mask blur 99 (clamped to 64)",
    "masked area padding 300 (clamped to 256)",
  ]);
});

// Prompt content, not a key: A1111 attention syntax and its escaped literal brackets are kept in the
// prompt as written and named unless the engine (sd.cpp) parses A1111 syntax itself.
const PROMPT_CONTENT: Array<[string, string[], string[]]> = [
  ["a (red:1.2) car", ["(red:1.2)"], []],
  ["a ((red)) car, [blurry]", ["((red))", "[blurry]"], []],
  ["a \\(literal\\) car \\[x\\]", [], ["\\(", "\\)", "\\[", "\\]"]],
  ["a (red car", [], []],
  ["plain text", [], []],
];

for (const [prompt, groups, escaped] of PROMPT_CONTENT) {
  test(`prompt content in ${JSON.stringify(prompt)}`, () => {
    assert.deepEqual(promptEmphasisGroups(prompt), groups);
    const r = parseA1111Parameters(`${prompt}\nNegative prompt: (worst quality:1.4)\n${BASE.split("\n")[1]}`);
    assert.ok(r);
    assert.equal(r.prompt, prompt);
    const all = [...groups, "(worst quality:1.4)"];
    const expected = [
      `prompt emphasis (${all.slice(0, 3).join(", ")}${all.length > 3 ? ", ..." : ""}) is passed as plain text`,
    ];
    if (escaped.length > 0)
      expected.push(`escaped brackets (${escaped.join(" ")}) are passed with their backslashes`);
    assert.deepEqual(importNotApplied(r, "sdxl"), expected);
    // The native sd.cpp engine reads A1111 syntax itself.
    assert.deepEqual(importNotApplied(r, "sdxl", true), []);
  });
}
