// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  defaultsFor,
  defaultsKeyFor,
  loadedRecipeFor,
  residentDefaultsKey,
  residentRecipeFor,
  resolutionFor,
} from "../src/features/images/image-generation-defaults.ts";

import { readSrc } from "./helpers/kit.ts";

test("distinguishes Klein base checkpoints from distilled checkpoints", () => {
  for (const size of ["4B", "9B"]) {
    assert.deepEqual(defaultsFor(`unsloth/FLUX.2-klein-base-${size}`), {
      steps: 20,
      guidance: 5,
    });
    assert.deepEqual(defaultsFor(`unsloth/FLUX.2-klein-${size}`), {
      steps: 4,
      guidance: 1,
    });
  }
});

test("keeps the existing family defaults and fallback", () => {
  assert.deepEqual(defaultsFor("krea/Krea-2-Raw"), {
    steps: 52,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("black-forest-labs/FLUX.1-dev"), {
    steps: 20,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("local/unknown-image-model"), {
    steps: 9,
    guidance: 0,
  });
});

test("an explicit family keys defaults only for an opaque path, never flattening a named variant", () => {
  const opaque = "/models/my-private-finetune";
  const schnell = "black-forest-labs/FLUX.1-schnell";
  const explicit = (value: string) => ({ value, source: "explicit" as const });
  for (const [key, want] of [
    [defaultsKeyFor(opaque, "qwen-image"), "qwen-image"],
    [defaultsKeyFor(opaque, "auto"), opaque],
    [defaultsKeyFor(schnell, "flux.1"), schnell],
    [defaultsKeyFor("Tongyi-MAI/Z-Image-Turbo", "z-image"), "Tongyi-MAI/Z-Image-Turbo"],
    [residentDefaultsKey(opaque, opaque, explicit("qwen-image")), "qwen-image"],
    [residentDefaultsKey(opaque, null, { value: "qwen-image", source: "auto" }), opaque],
    [residentDefaultsKey(schnell, schnell, explicit("flux.1")), schnell],
  ]) {
    assert.equal(key, want);
  }
  assert.deepEqual(defaultsFor("qwen-image"), { steps: 20, guidance: 4 });
  assert.deepEqual(defaultsFor(schnell), { steps: 4, guidance: 0 });
});

test("routed image picks apply and transactionally roll back model defaults", () => {
  const source = readSrc("features/images/images-page.tsx");
  const routeStart = source.indexOf(
    "const pick = diffusionRoutePick(",
    source.indexOf("const handledRouteModel"),
  );
  const routeEnd = source.indexOf(
    "// Reload the current model with the current advanced options.",
    routeStart,
  );
  assert.ok(routeStart >= 0 && routeEnd > routeStart);
  const routeBlock = source.slice(routeStart, routeEnd);
  assert.match(routeBlock, /imagePresets\.hydrated/);
  assert.match(routeBlock, /quantRevert\.current = revert/);
  assert.match(routeBlock, /applyImageModelDefaults\(wanted, "auto"\)/);
  assert.match(routeBlock, /!started[\s\S]*revertPick\(revert\)/);
});

test("routed video picks apply and transactionally roll back model defaults", () => {
  const source = readSrc("features/video/video-page.tsx");
  const routeStart = source.indexOf(
    "const pick = diffusionRoutePick(",
    source.indexOf("const handledRouteModel"),
  );
  const routeEnd = source.indexOf(
    "// The task dialog defers the load",
    routeStart,
  );
  assert.ok(routeStart >= 0 && routeEnd > routeStart);
  const routeBlock = source.slice(routeStart, routeEnd);
  assert.match(routeBlock, /videoPresets\.hydrated/);
  assert.match(routeBlock, /quantRevert\.current = revert/);
  assert.match(routeBlock, /applyVideoModelDefaults\(/);
  assert.match(routeBlock, /!started[\s\S]*revertPick\(revert\)/);
});

test("failed image and video picks release their recipe hydration claims", () => {
  for (const path of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, /const claim = claim\w+Recipe\(\)/);
    assert.match(source, /commitRecipeClaim = claim\.commit/);
    assert.match(source, /releaseRecipeClaim = claim\.release/);
    assert.match(source, /quantRevert\.current\?\.commitRecipeClaim\?\.\(\)/);
    assert.match(source, /revertPick[\s\S]*r\.releaseRecipeClaim\?\.\(\)/);
  }

  const hook = readSrc(
    "features/generation-presets/use-media-generation-presets.ts",
  );
  assert.match(
    hook,
    /deferredSavedSettingsRef\.current = committed \? null : settings/,
  );
  assert.match(hook, /formClaim\.current = previousClaim/);
  assert.match(hook, /hydrateSavedSettings\(deferred\)/);
  assert.match(hook, /source === "claiming"\s*\? "claimed" : source/);
});

test("an auto-engaged Qwen-Image-2.1 quant keeps the 1024 canvas; a picked quant still shrinks it", () => {
  const repo = "Qwen/Qwen-Image-2.1";
  assert.deepEqual(
    resolutionFor(repo, { modelKind: "pipeline", transformerQuant: "int8", transformerQuantSource: "auto" }),
    { width: 1024, height: 1024 },
  );
  assert.deepEqual(
    resolutionFor(repo, { modelKind: "pipeline", transformerQuant: "fp8", transformerQuantSource: "auto" }),
    { width: 1024, height: 1024 },
  );
  assert.deepEqual(
    resolutionFor(repo, { modelKind: "pipeline", transformerQuant: "int8", transformerQuantSource: "explicit" }),
    { width: 512, height: 512 },
  );
  assert.deepEqual(
    resolutionFor(repo, { modelKind: "pipeline", transformerQuant: "int8" }),
    { width: 512, height: 512 },
  );
  assert.deepEqual(
    resolutionFor(repo, { modelKind: "gguf", transformerQuant: "int8", transformerQuantSource: "explicit" }),
    { width: 1024, height: 1024 },
  );
  assert.deepEqual(resolutionFor(repo, { modelKind: "pipeline", transformerQuant: null }), {
    width: 1024,
    height: 1024,
  });
});

test("every images-page canvas seed passes the quant provenance", () => {
  const source = readSrc("features/images/images-page.tsx");
  const calls = source.split("resolutionFor(").slice(1);
  assert.equal(calls.length, 3);
  for (const call of calls) {
    assert.match(call.slice(0, 400), /transformerQuantSource: status\??\.resolved\?\.transformer_quant\?\.source/);
  }
});

test("defaults follow ComfyUI's official templates for the same model", () => {
  for (const [id, want] of [
    ["black-forest-labs/FLUX.1-dev", { steps: 20, guidance: 3.5 }],
    ["black-forest-labs/FLUX.1-Krea-dev", { steps: 20, guidance: 3.5 }],
    ["black-forest-labs/FLUX.1-Kontext-dev", { steps: 20, guidance: 2.5 }],
    ["black-forest-labs/FLUX.2-dev", { steps: 20, guidance: 4 }],
    ["Qwen/Qwen-Image-Edit-2511", { steps: 40, guidance: 4 }],
    ["Qwen/Qwen-Image-Edit-2509", { steps: 20, guidance: 4 }],
    ["Qwen/Qwen-Image-2512", { steps: 50, guidance: 4 }],
    ["Qwen/Qwen-Image", { steps: 20, guidance: 4 }],
    ["Tongyi-MAI/Z-Image-Turbo", { steps: 8, guidance: 0 }],
    // diffusers Z-Image guidance g equals ComfyUI cfg g + 1.
    ["Tongyi-MAI/Z-Image", { steps: 25, guidance: 3 }],
    ["stabilityai/stable-diffusion-xl-base-1.0", { steps: 25, guidance: 7 }],
    ["ideogram-ai/ideogram-4-fp8", { steps: 20, guidance: 7 }],
  ] as const) {
    assert.deepEqual(defaultsFor(id), want, id);
  }
});

test("every Qwen-Image-Layered spelling the backend accepts gets its 20 / 2.5 recipe", () => {
  for (const id of [
    "unsloth/Qwen-Image-Layered-GGUF",
    "local/qwen_image_layered",
    "local/qwenimagelayered-q4",
  ]) {
    assert.deepEqual(defaultsFor(id), { steps: 20, guidance: 2.5 }, id);
  }
});

test("every Qwen-Image-2.1 spelling gets Turbo's 8 / 1 recipe when it names Turbo, and the base keeps 25", () => {
  for (const id of [
    "Qwen/Qwen-Image-2.1-Turbo",
    "/models/qwen-image-21-turbo",
    "/models/qwen_image_21_turbo",
    "/models/qwen-image-21_turbo",
    "/models/qwenimage21-turbo",
    "/models/qwenimage21_turbo",
    "/models/qwenimage21turbo",
    "/models/qwenimage21turbo-Q4_K_M.gguf",
    "/models/qwen_image_2.1_turbo_Q4_K_M.gguf",
  ]) {
    assert.deepEqual(defaultsFor(id), { steps: 8, guidance: 1 }, id);
  }
  for (const id of ["Qwen/Qwen-Image-2.1", "/models/qwen_image_21", "/models/qwenimage21", "/models/qwen_image_2.1.gguf"]) {
    assert.deepEqual(defaultsFor(id), { steps: 25, guidance: 1 }, id);
  }
});

test("a community single file picked by path takes its family recipe once loaded (#11391)", () => {
  const pick = defaultsFor(defaultsKeyFor("/models/checkpoints/RealVisXL_V4.0.safetensors", null));
  const resident = residentDefaultsKey(
    "/models/checkpoints",
    "stabilityai/stable-diffusion-xl-base-1.0",
    { value: "sdxl", source: "auto" },
  );
  assert.deepEqual(loadedRecipeFor(pick, resident), { steps: 25, guidance: 7 });
  // A pick whose name already chose a recipe keeps it (Schnell is not re-seeded as dev).
  assert.equal(
    loadedRecipeFor(defaultsFor("black-forest-labs/FLUX.1-schnell"), "black-forest-labs/FLUX.1-dev"),
    null,
  );
  // An unrecognised resident leaves the form alone.
  assert.equal(loadedRecipeFor(pick, "/models/unknown"), null);
  assert.equal(loadedRecipeFor(null, resident), null);
  // The backend's header-read recipe wins: a renamed FLUX.1-dev keeps 20 steps, not its schnell base's 4.
  const schnellBase = residentDefaultsKey("/models/my_flux.safetensors", "black-forest-labs/FLUX.1-schnell", null);
  assert.deepEqual(loadedRecipeFor(pick, schnellBase, { steps: 20, guidance: 3.5 }), { steps: 20, guidance: 3.5 });
  assert.equal(loadedRecipeFor(pick, schnellBase, { steps: 9, guidance: 0 }), null);
});

test("the Images page applies the loaded family recipe only to an untouched fallback form", () => {
  const src = readSrc("features/images/images-page.tsx");
  const ready = src.slice(src.indexOf('if (p.phase === "ready")'), src.indexOf('if (p.phase === "error")'));
  assert.match(ready, /loadedRecipeFor\(\s*pickDefaults\.current/);
  assert.match(ready, /!pickRecipeSuperseded\.current\?\.\(\)/);
  assert.match(ready, /cur === DEFAULT_GEN\.steps \? loadedRecipe\.steps : cur/);
});

test("the resident recipe prefers the backend-reported one", () => {
  const schnellBase = residentDefaultsKey("/models/my_flux.safetensors", "black-forest-labs/FLUX.1-schnell", null);
  assert.deepEqual(residentRecipeFor(schnellBase, { steps: 20, guidance: 3.5 }), { steps: 20, guidance: 3.5 });
  // An older backend (or the native engine) reports nothing: the base-repo key decides, as before.
  assert.deepEqual(residentRecipeFor(schnellBase, null), defaultsFor("black-forest-labs/FLUX.1-schnell"));
  assert.deepEqual(residentRecipeFor(schnellBase, {}), defaultsFor("black-forest-labs/FLUX.1-schnell"));
});

test("the Images page seeds Default from the resident recipe and drops the pick token on revert", () => {
  const src = readSrc("features/images/images-page.tsx");
  assert.match(src, /residentRecipeFor\(\s*residentDefaults,\s*status\?\.generation_defaults/);
  assert.doesNotMatch(src, /defaultsFor\(residentDefaults\)/);
  const revert = src.slice(src.indexOf("const revertPick = useCallback"), src.indexOf("}, []);", src.indexOf("const revertPick = useCallback")));
  assert.match(revert, /pickDefaults\.current = null/);
});
