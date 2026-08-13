// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A model pick claims the generation recipe before the settings GET has necessarily answered, so
// the claim has to be settled on every way out of that load. The progress poll covers the ones it
// sees; a cancel or an eject tears the poll down first, so the pages settle it themselves. Without
// that, hydration parked behind the pick never resolves: the stored recipe is never applied and the
// preset controls stay disabled for the rest of the session.
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
  return source.slice(start, source.indexOf("}, [dismissLoadToast, pickGuard]);", start));
}

for (const [page, source] of [
  ["images", IMAGES],
  ["video", VIDEO],
] as const) {
  test(`${page}: cancelling or ejecting a load releases the pick's recipe claim`, () => {
    const body = dropResidentState(source);
    assert.match(body, /quantRevert\.current\.releaseRecipeClaim\?\.\(\)/);
    assert.match(body, /quantRevert\.current\.releaseRecipeClaim = undefined;/);
  });

  test(`${page}: it also drops the Default recipe the abandoned pick claimed`, () => {
    // Otherwise Default keeps describing a model that never became resident.
    assert.match(dropResidentState(source), /setPendingModelDefaults\(null\);/);
  });
}

test("a claim can report that a newer form action took the recipe", () => {
  const claim = HOOK.slice(
    HOOK.indexOf("const claimRecipe = useCallback("),
    HOOK.indexOf("const settings = useMemo"),
  );
  assert.match(claim, /superseded: \(\) => formClaim\.current !== claim,/);
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
