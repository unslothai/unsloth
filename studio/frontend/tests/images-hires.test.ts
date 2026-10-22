// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  HIRES_OFF,
  hiresBaseSize,
  hiresFromRecipe,
  hiresNotApplied,
  hiresRequest,
  hiresSummary,
  hiresTargetSize,
  hiresVisible,
  matchHiresUpscaler,
  resolveRestoredHires,
  upscaleModelRequest,
  type HiresCapability,
} from "../src/features/images/lib/hires-choice.ts";
import {
  importNotApplied,
  parseA1111Parameters,
  recipeSettings,
  unknownRecipeKeys,
} from "../src/features/images/lib/image-metadata.ts";

const SDXL: HiresCapability = {
  upscalers: [
    { id: "latent", label: "Latent (nearest-exact)", kind: "latent" },
    { id: "lanczos", label: "Lanczos", kind: "pixel" },
    { id: "realesrgan_x4plus", label: "R-ESRGAN 4x+", kind: "model" },
    { id: "realesrgan_x4plus_anime_6b", label: "R-ESRGAN 4x+ Anime6B", kind: "model" },
  ],
  default_upscaler: "latent",
  defaults: { scale: 1.5, denoise: 0.5 },
  max_side: 2048,
};
const FLUX: HiresCapability = {
  ...SDXL,
  upscalers: SDXL.upscalers.filter((u) => u.kind !== "latent"),
  default_upscaler: "lanczos",
};
const DEFAULTS = { width: 1024, height: 1024, steps: 20, guidance: 5 };

test("the control shows in Create only, and only with a capability", () => {
  assert.equal(hiresVisible(SDXL, "create"), true);
  assert.equal(hiresVisible(SDXL, "img2img"), false);
  assert.equal(hiresVisible(SDXL, "upscale"), false);
  assert.equal(hiresVisible(null, "create"), false);
  assert.equal(hiresVisible(undefined, "create"), false);
});

test("off (the default) or hidden sends nothing", () => {
  assert.equal(hiresRequest(HIRES_OFF, SDXL, "create"), undefined);
  const on = { ...HIRES_OFF, enabled: true };
  assert.equal(hiresRequest(on, null, "create"), undefined);
  assert.equal(hiresRequest(on, SDXL, "edit"), undefined);
  assert.deepEqual(hiresRequest(on, SDXL, "create"), {
    scale: 1.5,
    denoise: 0.5,
    steps: null,
    upscaler: "latent",
    guidance: null,
  });
});

test("latent is offered only where listed: a pick the model lacks falls back to its default", () => {
  const choice = { ...HIRES_OFF, enabled: true, upscaler: "latent" };
  assert.equal(hiresRequest(choice, FLUX, "create")?.upscaler, "lanczos");
  assert.equal(matchHiresUpscaler("Latent", FLUX), null);
  assert.equal(matchHiresUpscaler("Latent (bicubic antialiased)", SDXL), "latent");
  assert.equal(matchHiresUpscaler("r-esrgan 4x+", FLUX), "realesrgan_x4plus");
  assert.equal(matchHiresUpscaler("4x_foolhardy_Remacri", SDXL), null);
});

test("the size preview matches the backend rounding and cap", () => {
  assert.deepEqual(hiresTargetSize(832, 1216, 1.5), { width: 1248, height: 1824 });
  assert.deepEqual(hiresTargetSize(768, 768, 1.5), { width: 1152, height: 1152 });
  // 1216 * 2 = 2432 > 2048: both sides scale by 2048 / 2432, then round to 16.
  assert.deepEqual(hiresTargetSize(832, 1216, 2), { width: 1408, height: 2048 });
});

test("a gallery record restores the choice and the base size", () => {
  const record = {
    width: 1248,
    height: 1824,
    hires_scale: 1.5,
    hires_denoise: 0.5,
    hires_steps: 14,
    hires_upscaler: "realesrgan_x4plus",
    hires_guidance: null,
    hires_base_width: 832,
    hires_base_height: 1216,
  };
  assert.deepEqual(hiresFromRecipe(record), {
    enabled: true,
    scale: 1.5,
    denoise: 0.5,
    steps: 14,
    upscaler: "realesrgan_x4plus",
    guidance: null,
  });
  assert.deepEqual(hiresBaseSize(record), { width: 832, height: 1216 });
  assert.equal(hiresFromRecipe({ width: 1024, height: 1024 }), HIRES_OFF);
  assert.equal(hiresBaseSize({ width: 1024, height: 1024 }), null);
  assert.equal(hiresSummary(hiresFromRecipe(record)), "1.5x, denoise 0.5, 14 steps, R-ESRGAN 4x+");
});

test("pasted AUTOMATIC1111 hires infotext round-trips into the Hires fix settings", () => {
  const r = parseA1111Parameters(
    "a castle\nSteps: 28, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 832x1216, Model: x, Denoising strength: 0.5, Hires upscale: 1.5, Hires steps: 14, Hires upscaler: Latent (nearest-exact)",
  );
  assert.ok(r);
  assert.equal(r.strength, undefined);
  assert.equal(r.workflow, undefined);
  assert.deepEqual(unknownRecipeKeys(r), []);
  assert.deepEqual(importNotApplied(r), []);
  const settings = recipeSettings(r, DEFAULTS);
  assert.equal(settings.width, 832);
  assert.equal(settings.height, 1216);
  const choice = resolveRestoredHires(hiresFromRecipe(settings), SDXL);
  assert.deepEqual(choice, {
    enabled: true,
    scale: 1.5,
    denoise: 0.5,
    steps: 14,
    upscaler: "latent",
    guidance: null,
  });
  assert.deepEqual(hiresBaseSize(settings), { width: 832, height: 1216 });
  assert.deepEqual(hiresNotApplied(choice, SDXL), []);
  assert.deepEqual(hiresRequest(choice, SDXL, "create"), {
    scale: 1.5,
    denoise: 0.5,
    steps: 14,
    upscaler: "latent",
    guidance: null,
  });
});

test("an R-ESRGAN line maps to the model upscaler; Hires CFG Scale to the hires guidance", () => {
  const r = parseA1111Parameters(
    "x\nSteps: 20, Seed: 2, Size: 1024x1024, Denoising strength: 0.4, Hires upscale: 2, Hires upscaler: R-ESRGAN 4x+, Hires CFG Scale: 4",
  );
  assert.ok(r);
  const choice = resolveRestoredHires(hiresFromRecipe(recipeSettings(r, DEFAULTS)), FLUX);
  assert.equal(choice.upscaler, "realesrgan_x4plus");
  assert.equal(choice.denoise, 0.4);
  assert.equal(choice.scale, 2);
  assert.equal(choice.steps, null);
  assert.equal(choice.guidance, 4);
});

test("an unknown upscaler or a model without Hires fix is reported as not applied", () => {
  const r = parseA1111Parameters(
    "x\nSteps: 20, Seed: 2, Size: 1024x1024, Denoising strength: 0.4, Hires upscale: 2, Hires upscaler: 4x_foolhardy_Remacri",
  );
  assert.ok(r);
  const restored = hiresFromRecipe(recipeSettings(r, DEFAULTS));
  assert.deepEqual(hiresNotApplied(restored, SDXL), [
    "Hires upscaler 4x_foolhardy_Remacri (not available; using Latent (nearest-exact))",
  ]);
  assert.equal(resolveRestoredHires(restored, SDXL).upscaler, null);
  assert.deepEqual(hiresNotApplied(restored, null), [
    "Hires fix (2x, denoise 0.4, base steps, 4x_foolhardy_Remacri)",
  ]);
  const resize = parseA1111Parameters("x\nSteps: 20, Seed: 2, Hires resize: 1024x1024, Hires sampler: Euler");
  assert.ok(resize);
  assert.deepEqual(importNotApplied(resize), ["hires fix (Hires resize: 1024x1024, Hires sampler: Euler)"]);
});

test("a Studio recipe's Hires fields and upscale model pass the import type checks", () => {
  const settings = recipeSettings(
    {
      prompt: "p",
      width: 1248,
      height: 1824,
      hires_scale: 9,
      hires_denoise: 0.5,
      hires_steps: 14,
      hires_upscaler: "lanczos",
      hires_base_width: 832,
      hires_base_height: 1216,
      upscale_model: "realesrgan_x4plus",
    },
    DEFAULTS,
  );
  assert.equal(settings.hires_scale, 4);
  assert.equal(settings.hires_upscaler, "lanczos");
  assert.equal(settings.upscale_model, "realesrgan_x4plus");
  assert.deepEqual(hiresBaseSize(settings), { width: 832, height: 1216 });
});

test("the native capability drops Hires CFG and upscale-only from the request", () => {
  const NATIVE: HiresCapability = { ...FLUX, supports_guidance: false, min_denoise: 0.01 };
  const choice = { ...HIRES_OFF, enabled: true, denoise: 0, guidance: 4 };
  assert.deepEqual(hiresRequest(choice, NATIVE, "create"), {
    scale: 1.5,
    denoise: 0.01,
    steps: null,
    upscaler: "lanczos",
    guidance: null,
  });
  assert.deepEqual(hiresNotApplied(choice, NATIVE), [
    "Hires CFG Scale 4 (not available on this engine; using the base guidance)",
    "Hires denoise 0 (this engine needs at least 0.01)",
  ]);
  // The diffusers capability keeps both, and a typed value snaps to the 0.01 grid.
  assert.equal(hiresRequest(choice, FLUX, "create")?.guidance, 4);
  assert.equal(hiresRequest(choice, FLUX, "create")?.denoise, 0);
  assert.equal(hiresRequest({ ...choice, denoise: 0.004 }, FLUX, "create")?.denoise, 0);
  assert.equal(hiresRequest({ ...choice, denoise: 0.356 }, FLUX, "create")?.denoise, 0.36);
});

test("a model upscaler is sent only while the status still lists it", () => {
  const models = [{ id: "realesrgan_x4plus", label: "R-ESRGAN 4x+", scale: 4 }];
  assert.equal(upscaleModelRequest("realesrgan_x4plus", models), "realesrgan_x4plus");
  assert.equal(upscaleModelRequest("local:gone", models), undefined);
  assert.equal(upscaleModelRequest("realesrgan_x4plus", []), undefined);
  assert.equal(upscaleModelRequest("realesrgan_x4plus", undefined), undefined);
  assert.equal(upscaleModelRequest(null, models), undefined);
});
