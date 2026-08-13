// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A model pick claims the generation recipe before the settings GET has necessarily answered, so
// the claim has to be settled on every way out of that load. The progress poll covers the endings
// it sees; a cancel or an eject tears the poll down first, so the pages hand the pick back
// themselves. Without that, hydration parked behind the pick never resolves (the stored recipe is
// never applied and the preset controls stay disabled), and the rollback left behind is the one a
// later pick inherits in place of its own.
//
// The wiring lives inside two ~3000-line page components with no renderer in this suite, so these
// assert on the source, like the other page-wiring tests here.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string) {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

const IMAGES = read("../src/features/images/images-page.tsx");
const VIDEO = read("../src/features/video/video-page.tsx");
const HOOK = read(
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
    // The same two lines the poll's cancelled/evicted branch runs, which this tears down first.
    assert.match(body, /revertPick\(quantRevert\.current\);/);
    assert.match(body, /quantRevert\.current = null;/);
  });

  test(`${page}: every pick baselines supersession on itself`, () => {
    // A second pick reuses the first one's rollback object, so the claim block is skipped. Reading
    // the counter outside it is what stops the second pick from inheriting the first's answer.
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
  // Otherwise a rejected save reads as "something newer owns the form" for the rest of the
  // session, and a pending pick's model defaults are skipped over a change that never happened.
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
  // Two PUTs in flight land in whatever order the backend sees them, and the store keeps the last
  // one, so an older snapshot could outlive the newest recipe.
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
  // The preset is gone either way, and a selection naming it leaves the control on a definition
  // that no longer exists (and persists that name on the next debounced write).
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
  // Only the form VALUES are conditional; the selection reset is not.
  assert.match(restore, /ownsForm &&/);
  assert.match(restore, /setActivePreset\(DEFAULT_PRESET_NAME\);/);
});

test("a store with presets but no recipe still hydrates the library", () => {
  // saved:false means the recipe falls back to the model's defaults, not that the user's named
  // presets are gone; dropping them would hide a library the response is carrying.
  assert.match(
    HOOK,
    /hydrateLocalSettings\("fresh", settings\.customPresets \?\? \[\]\)/,
  );
  const hydrate = HOOK.slice(HOOK.indexOf("const hydrateLocalSettings = useCallback("));
  assert.match(hydrate.slice(0, hydrate.indexOf("setActivePreset")), /setCustomPresets\(custom\);/);
});

test("video defers both status seeds to a preset picked during the load", () => {
  // The duration branch and the steps/guidance branch seed independently, so both have to ask.
  const asks = VIDEO.match(/pickRecipeSuperseded\.current\?\.\(\) \?\? false,/g) ?? [];
  assert.equal(asks.length, 2);
  const seed = VIDEO.slice(VIDEO.indexOf("const applyDefaults = shouldApplyModelDefaults("));
  assert.ok(
    seed.indexOf("pickRecipeSuperseded.current = null;") <
      seed.indexOf("modelSeeded.current = true;"),
    "the confirmed pick's question is answered once, in the later of the two effects",
  );
});
