// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { hydratedParams } from "../src/features/generation-presets/preset-policy.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const panel = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");
const images = await readSrcAsync("features/images/images-page.tsx");
const video = await readSrcAsync("features/video/video-page.tsx");
const hook = await readSrcAsync(
  "features/generation-presets/use-media-generation-presets.ts",
);

test("image and video generation still restore the working recipe", () => {
  const working = { steps: 30 };
  assert.equal(hydratedParams(working, { name: "Default", params: { steps: 9 } }, true), working);
  assert.equal(hydratedParams(working, { name: "Mine", params: { steps: 20 } }, true), working);
  assert.match(hook, /restoreWorkingParams = true,/);
});

test("training loads the active named preset, and Default leaves a fresh form", () => {
  const preset = { steps: 1200 };
  assert.equal(hydratedParams({ steps: 77 }, { name: "Mine", params: preset }, false), preset);
  assert.equal(hydratedParams({ steps: 77 }, { name: "Default", params: { steps: 500 } }, false), null);
});

test("the training panel uses its own preset kind without working-recipe restore", () => {
  const call = panel.slice(panel.indexOf("useMediaGenerationPresets({"));
  assert.match(call, /kind: "training",/);
  assert.match(call, /restoreWorkingParams: false,/);
  assert.match(panel, /<MediaGenerationPresetControl\s+kind="training"/);
});

test("only the Images Train panel owns the saved presets", () => {
  // Video keeps its own Train panel mounted; a second hook would race the saved selection.
  assert.match(panel, /savedPresets = false,/);
  assert.match(panel, /\{savedPresets \? \(\s+<TrainingPresetControl/);
  assert.match(images, /savedPresets=\{true\}/);
  assert.doesNotMatch(video, /savedPresets/);
});

test("a named preset survives the family re-seed and the loaded-model seed", () => {
  const apply = panel.slice(
    panel.indexOf("const applyPresetParams = useCallback("),
    panel.indexOf("const poll = useCallback("),
  );
  assert.match(apply, /const owned = params !== defaultPresetParams;/);
  for (const ref of ["settingsDirty", "lrScheduleDirty", "precisionDirty"]) {
    assert.match(apply, new RegExp(`${ref}\\.current = owned;`));
  }
  const ownedBlock = apply.slice(apply.indexOf("if (owned) {"));
  assert.match(ownedBlock, /seededFromLoaded\.current = true;/);
  assert.match(ownedBlock, /seededBaseFamily\.current = params\.familyName;/);
  assert.match(ownedBlock, /baseDirty\.current = true;/);
  assert.ok(
    apply.indexOf("if (owned) {") < apply.indexOf("setFamilyName(params.familyName);"),
    "the refs are set before the family changes, so the re-seed effect sees them",
  );
});

test("edits made before the stored preset lands are kept", () => {
  const apply = panel.slice(
    panel.indexOf("const applyPresetParams = useCallback("),
    panel.indexOf("const poll = useCallback("),
  );
  const guard = apply.indexOf("if (!presetsReadyRef.current && userEdited) {");
  assert.ok(guard > 0, "hydration must not overwrite a form the user already edited");
  assert.ok(guard < apply.indexOf("setFamilyName(params.familyName);"));
  assert.match(apply, /editedBeforePresets\.current \|\|/, "family and toggle edits count too");
  assert.match(panel, /root\.addEventListener\("pointerdown", noteFormEdit, true\);/);
  assert.match(panel, /root\.addEventListener\("keydown", noteFormEdit, true\);/);
  assert.match(panel, /<div ref=\{trainRailRef\} className=/);
  assert.match(panel, /readyRef\.current = presets\.presetsReady;/);
  assert.match(panel, /readyRef=\{presetsReadyRef\}/);
});

test("a preset family only /info reports is not seeded from the fallback family", () => {
  assert.match(
    panel,
    /const familyPending =\s+family\?\.name !== familyName && \(infoLoadState === "idle" \|\| infoLoadState === "loading"\);/,
  );
  const seed = panel.slice(panel.indexOf("const familyPending ="));
  assert.match(seed, /if \(!family \|\| familyPending\) return;/);
  assert.match(seed, /\}, \[family, familyPending, loadedBaseRepo, reportedFamily\]\);/);
});

test("every saved field is applied back to the form", () => {
  const current = panel.slice(
    panel.indexOf("const currentPresetParams = useMemo<TrainingPresetParams>("),
    panel.indexOf("const applyPresetParams = useCallback("),
  );
  const fields = [...current.slice(0, current.indexOf("}),")).matchAll(/^\s+(\w+)(?::|,)/gm)].map(
    (m) => m[1],
  );
  assert.ok(fields.length >= 25, `expected the full recipe, got ${fields.join(",")}`);
  const apply = panel.slice(panel.indexOf("const applyPresetParams = useCallback("));
  for (const field of fields) {
    assert.match(apply, new RegExp(`\\(params\\.${field}\\)`), `${field} is saved but never applied`);
  }
});
