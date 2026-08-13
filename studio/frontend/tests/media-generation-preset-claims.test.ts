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
