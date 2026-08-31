// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { defaultsFor } from "../src/features/images/image-generation-defaults.ts";

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
