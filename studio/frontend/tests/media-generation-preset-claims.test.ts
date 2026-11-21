// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A pick's recipe claim must settle on every exit, including cancel and eject. Source-level.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const IMAGES = readText("../src/features/images/images-page.tsx");
const VIDEO = readText("../src/features/video/video-page.tsx");
const HOOK = readText(
  "../src/features/generation-presets/use-media-generation-presets.ts",
);

function dropResidentState(source: string) {
  const start = source.indexOf("const dropResidentState = useCallback(");
  assert.ok(start > 0, "dropResidentState must exist");
  return source.slice(start, source.indexOf("]);", start));
}

for (const [page, source] of [
  ["images", IMAGES],
  ["video", VIDEO],
] as const) {
  test(`${page}: cancelling or ejecting a load hands the pick back`, () => {
    const body = dropResidentState(source);
    assert.match(body, /revertPick\(quantRevert\.current\);/);
    assert.match(body, /quantRevert\.current = null;/);
  });

  test(`${page}: every pick baselines supersession on itself`, () => {
    // A second pick reuses the first's rollback object, so the counter is read outside the claim.
    const apply = source.slice(
      source.indexOf("ModelDefaults = useCallback("),
      source.indexOf("const recommended = defaultsFor(repoId);"),
    );
    assert.match(apply, /const claimedAt = \w+FormClaimId\(\);/);
    assert.match(
      apply,
      /pickRecipeSuperseded\.current = \(\) => \w+FormClaimId\(\) !== claimedAt;/,
    );
    const claimBlock = apply.slice(
      apply.indexOf("if (revert && !revert.releaseRecipeClaim) {"),
      apply.indexOf("const claimedAt"),
    );
    assert.doesNotMatch(claimBlock, /claimedAt/, "the baseline must survive a reused rollback");
  });

  test(`${page}: a superseded pick does not roll the recipe back`, () => {
    const revert = source.slice(
      source.indexOf("const revertPick = useCallback("),
      source.indexOf("r.releaseRecipeClaim?.();"),
    );
    const guard = revert.indexOf("if (!pickRecipeSuperseded.current?.()) {");
    assert.ok(guard > 0, "the value restore must sit behind the supersession guard");
    assert.ok(guard < revert.indexOf("cur === r.appliedSteps"));
    assert.ok(guard < revert.indexOf("cur === r.appliedGuidance"));
  });

  test(`${page}: a preset's negative prompt is revealed, not applied behind a closed field`, () => {
    const apply = source.slice(source.indexOf("PresetParams = useCallback("));
    const setter = apply.indexOf("setNegativePrompt(params.negativePrompt);");
    assert.ok(setter > 0);
    assert.match(
      apply.slice(setter, setter + 400),
      /if \(params\.negativePrompt\) setNegativeOpen\(true\);/,
    );
  });
}

test("the form claim counter is readable, so a pick can tell it was superseded", () => {
  assert.match(HOOK, /const formClaimId = useCallback\(\(\) => formClaim\.current, \[\]\);/);
  assert.match(HOOK, /^\s+formClaimId,$/m, "and returned from the hook");
});

test("a preset write that failed gives its form claim back", () => {
  for (const name of ["const savePreset = useCallback(", "const deletePreset = useCallback("]) {
    const body = HOOK.slice(HOOK.indexOf(name));
    const failure = body.slice(body.indexOf("} catch (error) {"));
    assert.match(
      failure.slice(0, failure.indexOf("toast.error")),
      /if \(formClaim\.current === claim\) formClaim\.current = previousClaim;/,
      `${name} must restore the claim before reporting the failure`,
    );
  }
});

test("state writes go out one at a time, newest last", () => {
  // Concurrent PUTs land in arbitrary order and the store keeps the last.
  assert.match(
    HOOK,
    /inflightWriteRef\.current = inflightWriteRef\.current\s*\n?\s*\.catch\(\(\) => undefined\)\s*\n?\s*\.then\(write\);/,
  );
  for (const site of [
    "saveMediaGenerationPresetSettings(kind, settings)",
    "saveMediaGenerationPresetSettings(kind, latest, true)",
  ]) {
    const at = HOOK.indexOf(site);
    assert.ok(at > 0, site);
    assert.match(
      HOOK.slice(at - 120, at),
      /queueWrite\(\(\) =>\s*$/,
      `${site} must go through the queue`,
    );
  }
});

test("a delete clears the selection even when a pick took the form meanwhile", () => {
  const del = HOOK.slice(
    HOOK.indexOf("const deletePreset = useCallback("),
    HOOK.indexOf("const activeDefinition ="),
  );
  assert.match(
    del,
    /restoreDefaultAfterDelete\(paramsBeforeDelete, formClaim\.current === claim\);/,
  );
  const restore = HOOK.slice(
    HOOK.indexOf("const restoreDefaultAfterDelete = useCallback("),
    HOOK.indexOf("const deletePreset = useCallback("),
  );
  assert.match(restore, /ownsForm &&/);
  assert.match(restore, /setActivePreset\(DEFAULT_PRESET_NAME\);/);
});

test("a store with presets but no recipe still hydrates the library", () => {
  // saved:false means recipe defaults, not that named presets are gone.
  assert.match(
    HOOK,
    /hydrateLocalSettings\("fresh", settings\.customPresets \?\? \[\]\)/,
  );
  const hydrate = HOOK.slice(HOOK.indexOf("const hydrateLocalSettings = useCallback("));
  assert.match(hydrate.slice(0, hydrate.indexOf("setActivePreset")), /setCustomPresets\(custom\);/);
});

test("video defers both status seeds to a preset picked during the load", () => {
  // Duration and steps/guidance seed independently, so both must ask.
  const asks = VIDEO.match(/pickRecipeSuperseded\.current\?\.\(\) \?\? false,/g) ?? [];
  assert.equal(asks.length, 2);
  const seed = VIDEO.slice(VIDEO.indexOf("const applyDefaults = shouldApplyModelDefaults("));
  assert.ok(
    seed.indexOf("pickRecipeSuperseded.current = null;") <
      seed.indexOf("modelSeeded.current = true;"),
    "the confirmed pick's question is answered once, in the later of the two effects",
  );
});
