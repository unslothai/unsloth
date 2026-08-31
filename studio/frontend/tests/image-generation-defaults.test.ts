// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  defaultsFor,
  residentImageDefaultsIdentity,
  residentImageDefaultsSeedKey,
} from "../src/features/images/image-generation-defaults.ts";
import {
  isH3PipelinePick,
  videoDefaultsFor,
} from "../src/features/video/video-generation-defaults.ts";

test("an explicit video family is authoritative for the H3 task dialog", () => {
  assert.equal(
    isH3PipelinePick("C:/models/MiniMax-H3", "pipeline", "ltx-2"),
    false,
    "a misleading directory name must not override explicit LTX architecture evidence",
  );
  assert.equal(
    isH3PipelinePick("C:/models/opaque", "pipeline", "minimax-h3"),
    true,
  );
  assert.equal(isH3PipelinePick("C:/models/MiniMax-H3", "pipeline"), true);
  assert.equal(
    isH3PipelinePick("C:/models/MiniMax-H3", "single_file", "minimax-h3"),
    false,
  );
});

test("resident image default identity includes the resolved family", () => {
  const zImage = residentImageDefaultsSeedKey({
    repoId: "C:/models/opaque",
    baseRepo: null,
    family: "z-image",
    modelKind: "pipeline",
  });
  const flux = residentImageDefaultsSeedKey({
    repoId: "C:/models/opaque",
    baseRepo: null,
    family: "flux.1",
    modelKind: "pipeline",
  });
  assert.notEqual(zImage, flux);
  assert.equal(
    flux,
    residentImageDefaultsSeedKey({
      repoId: "C:/models/opaque",
      baseRepo: null,
      family: "flux.1",
      modelKind: "pipeline",
    }),
  );
});

test("resident defaults retain a matching page-owned checkpoint filename", () => {
  assert.equal(
    residentImageDefaultsIdentity({
      repoId: "C:\\models\\opaque",
      baseRepo: "black-forest-labs/FLUX.1-dev",
      modelKind: "single_file",
      lastLoad: {
        repoId: "C:/models/opaque",
        kind: "single_file",
        filename: "FLUX_1_schnell.safetensors",
      },
    }),
    "C:/models/opaque/FLUX_1_schnell.safetensors",
  );
  assert.equal(
    residentImageDefaultsIdentity({
      repoId: "C:/models/replaced",
      baseRepo: "black-forest-labs/FLUX.1-dev",
      modelKind: "single_file",
      lastLoad: {
        repoId: "C:/models/old",
        kind: "single_file",
        filename: "FLUX_1_schnell.safetensors",
      },
    }),
    "C:/models/replaced",
    "a stale page-owned target must not describe another client's resident model",
  );
});

test("first-load confirmation reconciles optimistic defaults to the resolved family", () => {
  assert.deepEqual(defaultsFor("local/FLUX.2_klein_9B", "flux.2-klein"), {
    steps: 4,
    guidance: 1,
  });
  const source = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  const residentStart = source.indexOf("const residentSeeded = useRef(false)");
  const residentEnd = source.indexOf("// Reseed the Advanced selects", residentStart);
  const residentEffect = source.slice(residentStart, residentEnd);
  assert.match(residentEffect, /pendingModelDefaults/);
  assert.match(residentEffect, /pendingModelDefaults\.loadSeq === loadSeq\.current/);
  assert.match(residentEffect, /lastLoad\.current\?\.repoId === repoId/);
  assert.match(residentEffect, /lastLoad\.current\?\.family/);
  assert.match(
    residentEffect,
    /defaultsFor\(\s*pendingModelDefaults\.repoId,\s*status\?\.family/,
    "confirmation must retain a direct checkpoint filename for variant defaults",
  );
  assert.match(residentEffect, /pickRecipeSuperseded\.current\?\.\(\)/);
  assert.match(residentEffect, /setSteps\(d\.steps\)/);
  assert.match(residentEffect, /setGuidance\(d\.guidance\)/);
  const readyStart = source.indexOf('if (p.phase === "ready")');
  const readyEnd = source.indexOf('if (p.phase === "error")', readyStart);
  assert.doesNotMatch(
    source.slice(readyStart, readyEnd),
    /setPendingModelDefaults\(null\)/,
    "ready status must stay available for resolved-family reconciliation",
  );
});

test("the Default preset keeps resident defaults ahead of staged load defaults", () => {
  for (const page of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const source = readFileSync(new URL(page, import.meta.url), "utf8");
    const recipeStart = source.indexOf(
      page.includes("images")
        ? "const imageDefaultRecipe = useMemo"
        : "const videoDefaultRecipe = useMemo",
    );
    const recipeEnd = source.indexOf(
      page.includes("images")
        ? "const applyImagePresetParams"
        : "const applyVideoPresetParams",
      recipeStart,
    );
    assert.ok(recipeStart >= 0 && recipeEnd > recipeStart);
    assert.match(
      source.slice(recipeStart, recipeEnd),
      /status\?\.loaded\s*\?[\s\S]*:\s*pendingModelDefaults\s*\?\?/,
      `${page}: resident status must outrank an unapplied Family selection`,
    );
  }
});

test("distinguishes Klein base checkpoints from distilled checkpoints", () => {
  for (const size of ["4B", "9B"]) {
    assert.deepEqual(defaultsFor(`unsloth/FLUX.2-klein-base-${size}`), {
      steps: 50,
      guidance: 4,
    });
    assert.deepEqual(defaultsFor(`unsloth/FLUX.2-klein-${size}`), {
      steps: 4,
      guidance: 1,
    });
  }
});

test("variant defaults accept the same flexible delimiters as family detection", () => {
  assert.deepEqual(
    defaultsFor("C:\\models\\FLUX_1_schnell.safetensors", "flux.1"),
    { steps: 4, guidance: 0 },
  );
  assert.deepEqual(
    defaultsFor("/models/FLUX.2_klein.base-4B.safetensors", "flux.2-klein"),
    { steps: 50, guidance: 4 },
  );
});

test("keeps the existing family defaults and fallback", () => {
  assert.deepEqual(defaultsFor("krea/Krea-2-Raw"), {
    steps: 52,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("black-forest-labs/FLUX.1-dev"), {
    steps: 28,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("local/unknown-image-model"), {
    steps: 9,
    guidance: 0,
  });
});

test("matches family overrides to family defaults without selecting more-specific variants", () => {
  // krea-2 must hit Krea 2 Turbo (8/0), not krea-2-raw (52/3.5)
  assert.deepEqual(defaultsFor("", "krea-2"), {
    steps: 8,
    guidance: 0,
  });
  // flux.2-klein must hit distilled Klein (4/1), not flux.2-klein-base (50/4)
  assert.deepEqual(defaultsFor("", "flux.2-klein"), {
    steps: 4,
    guidance: 1,
  });
  // specific variant override still works if explicitly passed
  assert.deepEqual(defaultsFor("", "krea-2-raw"), {
    steps: 52,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("", "flux.2-klein-base"), {
    steps: 50,
    guidance: 4,
  });
  assert.deepEqual(defaultsFor("", "flux.1-kontext"), {
    steps: 28,
    guidance: 2.5,
  });
  assert.deepEqual(defaultsFor("", "flux.1"), {
    steps: 28,
    guidance: 3.5,
  });
  // An explicit family is the backend's authority even when an opaque merge has a misleading
  // family token in its filename.
  assert.deepEqual(defaultsFor("local/sdxl-merge", "flux.1"), {
    steps: 28,
    guidance: 3.5,
  });
});

test("a matching canonical family keeps variant-specific repository defaults", () => {
  for (const [repoId, family, expected] of [
    ["stabilityai/sdxl-turbo", "sdxl", { steps: 3, guidance: 0 }],
    ["black-forest-labs/FLUX.1-schnell", "flux.1", { steps: 4, guidance: 0 }],
    ["Tongyi-MAI/Z-Image-Turbo", "z-image", { steps: 9, guidance: 0 }],
    ["krea/Krea-2-Raw", "krea-2", { steps: 52, guidance: 3.5 }],
    ["unsloth/FLUX.2-klein-base-4B", "flux.2-klein", { steps: 50, guidance: 4 }],
  ] as const) {
    assert.deepEqual(defaultsFor(repoId, family), expected, repoId);
  }
});

test("video defaults preserve compatible variants and reject conflicting repo hints", () => {
  assert.deepEqual(
    videoDefaultsFor("unsloth/LTX-2.3-distilled-GGUF/model.safetensors", "ltx-2"),
    { steps: 8, guidance: 1 },
  );
  assert.deepEqual(videoDefaultsFor("local/ltx-distilled-merge", "minimax-h3"), {
    steps: 30,
    guidance: 1,
  });
});

test("the image Default preset keeps the resolved family after load completion", () => {
  const source = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  const recipeStart = source.indexOf("const imageDefaultRecipe = useMemo");
  const recipeEnd = source.indexOf("const applyImagePresetParams", recipeStart);
  assert.ok(recipeStart >= 0 && recipeEnd > recipeStart);
  const recipe = source.slice(recipeStart, recipeEnd);
  assert.match(
    recipe,
    /defaultsFor\(\s*residentImageDefaultsIdentity\([\s\S]*lastLoad: lastLoad\.current[\s\S]*status\?\.family/,
  );
  assert.match(recipe, /status\?\.family/);
});

test("loaded-model slider seeding also keeps the selected repository variant", () => {
  const source = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /defaultsFor\(status\?\.repo_id \?\? status\?\.base_repo \?\? repoId, status\?\.family\)/,
  );
});

test("routed image picks apply and transactionally roll back model defaults", () => {
  const source = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
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
  assert.match(routeBlock, /applyImageModelDefaults\(wanted\)/);
  assert.match(routeBlock, /!started[\s\S]*revertPick\(revert\)/);
});

test("routed video picks apply and transactionally roll back model defaults", () => {
  const source = readFileSync(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
    "utf8",
  );
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

  const hook = readFileSync(
    new URL(
      "../src/features/generation-presets/use-media-generation-presets.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /deferredSavedSettingsRef\.current = committed \? null : settings/,
  );
  assert.match(hook, /formClaim\.current = previousClaim/);
  assert.match(hook, /hydrateSavedSettings\(deferred\)/);
  assert.match(hook, /source === "claiming"\s*\? "claimed" : source/);
});
