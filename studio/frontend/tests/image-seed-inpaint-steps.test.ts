// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  defaultsFor,
  STEPS_REQUEST_MAX,
  STEPS_SLIDER_MAX,
  stepsSliderMax,
} from "../src/features/images/image-generation-defaults.ts";
import {
  INPAINT_PADDING_DEFAULT,
  inpaintMaskRequestFields,
  MASK_BLUR_DEFAULT,
  inpaintOptionsNotApplied,
  restoredInpaintMaskOptions,
} from "../src/features/images/lib/inpaint-options.ts";
import {
  lastSeedOf,
  reusableSeed,
} from "../src/features/images/lib/seed-controls.ts";
import { readSrc } from "./helpers/kit.ts";

test("the Steps slider reaches every model's default: Krea-2-raw ships 52", () => {
  const krea2Raw = defaultsFor("krea/Krea-2-raw").steps;
  assert.equal(krea2Raw, 52);
  assert.ok(stepsSliderMax(krea2Raw, krea2Raw) >= krea2Raw);
  assert.equal(stepsSliderMax(krea2Raw, krea2Raw), STEPS_REQUEST_MAX);
  assert.equal(
    stepsSliderMax(defaultsFor("black-forest-labs/FLUX.1-schnell").steps, 4),
    STEPS_SLIDER_MAX,
  );
  assert.equal(stepsSliderMax(25, 25), 50);
  assert.equal(stepsSliderMax(30, 30), 60);
  assert.equal(stepsSliderMax(8, 80), 80);
  assert.equal(stepsSliderMax(8, 150), STEPS_REQUEST_MAX);
  assert.equal(stepsSliderMax(Number.NaN, 4), STEPS_SLIDER_MAX);
  for (const id of [
    "krea-2-raw",
    "lumina",
    "hunyuanimage",
    "hidream",
    "qwen-image-2512",
    "flux.3-klein-base",
  ]) {
    const d = defaultsFor(id).steps;
    assert.ok(stepsSliderMax(d, d) >= d, id);
  }
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /max=\{stepsSliderMax\(pendingModelDefaults\?\.steps \?\? residentSteps, steps\)\}/,
  );
  assert.doesNotMatch(page, /label="Steps"[\s\S]{0,300}max=\{50\}/);
});

test("Reuse last seed takes this session's run base seed, else the newest record's", () => {
  const run = [
    { seed: 101, batch_seed: 100, created_at: 5 },
    { seed: 102, batch_seed: 100, created_at: 5 },
  ];
  assert.equal(lastSeedOf(run), 100);
  assert.equal(lastSeedOf([{ seed: 7, created_at: 1 }]), 7);
  assert.equal(
    lastSeedOf([{ seed: 7, created_at: 1, model_kind: "connection" }]),
    null,
  );
  assert.equal(lastSeedOf([]), null);
  const gallery = [
    { seed: 1, created_at: 10 },
    { seed: 2, batch_seed: 20, created_at: 30 },
    { seed: 3, created_at: 40, model_kind: "connection" },
  ];
  assert.equal(reusableSeed(null, gallery), 20);
  assert.equal(reusableSeed(555, gallery), 555);
  assert.equal(reusableSeed(null, []), null);
  // Seed 0 is a real seed, not "none".
  assert.equal(reusableSeed(0, gallery), 0);
});

test("the Seed field has Random and Reuse last seed buttons", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /aria-label="Random seed"[\s\S]{0,200}onClick=\{\(\) => setSeed\(""\)\}/,
  );
  assert.match(
    page,
    /aria-label="Reuse last seed"[\s\S]{0,300}if \(lastSeed !== null\) setSeed\(String\(lastSeed\)\);/,
  );
  assert.match(
    page,
    /const runSeed = lastSeedOf\(res\.images\);\n\s+if \(runSeed !== null\) setSessionSeed\(runSeed\);/,
  );
  // A HugeIcons glyph only this lazy page uses still lands in the eager icon chunk via the barrel
  // and pushed the startup budget over, so both seed buttons draw lucide glyphs.
  assert.match(page, /<Shuffle className="size-4" strokeWidth=\{1\.5\} \/>/);
  assert.match(page, /<HistoryIcon className="size-4" strokeWidth=\{1\.5\} \/>/);
  assert.doesNotMatch(page, /Recycle01Icon/);
});

test("inpaint mask options send nothing at their defaults", () => {
  const defaults = {
    maskBlur: MASK_BLUR_DEFAULT,
    onlyMasked: false,
    padding: INPAINT_PADDING_DEFAULT,
  };
  assert.deepEqual(inpaintMaskRequestFields(defaults, true), {});
  assert.deepEqual(inpaintMaskRequestFields(defaults, false), {});
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, maskBlur: 4 }, false),
    { mask_blur: 4 },
  );
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, maskBlur: 99.6 }, false),
    { mask_blur: 64 },
  );
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, onlyMasked: true }, true),
    {
      inpaint_only_masked: true,
      inpaint_padding: 32,
    },
  );
  // A model whose inpaint pipeline cannot crop never gets the field (the backend would refuse it).
  assert.deepEqual(
    inpaintMaskRequestFields({ ...defaults, onlyMasked: true }, false),
    {},
  );
  assert.deepEqual(
    inpaintMaskRequestFields(
      { maskBlur: 8, onlyMasked: true, padding: 900 },
      true,
    ),
    {
      mask_blur: 8,
      inpaint_only_masked: true,
      inpaint_padding: 256,
    },
  );
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /condStrength = strength;\n\s+condInpaint = inpaintMaskRequestFields\(/,
  );
  assert.match(page, /\.\.\.condInpaint,/);
  assert.match(page, /\{status\?\.supports_inpaint_only_masked && \(/);
});

test("restoring an inpaint record resets mask options it omitted at their defaults", () => {
  assert.deepEqual(restoredInpaintMaskOptions({ workflow: "inpaint" }), {
    maskBlur: MASK_BLUR_DEFAULT,
    onlyMasked: false,
    padding: INPAINT_PADDING_DEFAULT,
  });
  assert.deepEqual(
    restoredInpaintMaskOptions({
      workflow: "inpaint",
      mask_blur: 6,
      inpaint_only_masked: true,
      inpaint_padding: 48,
    }),
    { maskBlur: 6, onlyMasked: true, padding: 48 },
  );
  // Other workflows carry no mask options, so they leave the controls alone.
  assert.deepEqual(restoredInpaintMaskOptions({ workflow: "img2img" }), {});
  assert.deepEqual(restoredInpaintMaskOptions({ workflow: null }), {});
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /const maskOptions = restoredInpaintMaskOptions\(image\);\n\s+if \(maskOptions\.maskBlur !== undefined\) setMaskBlur\(maskOptions\.maskBlur\);/,
  );
});

test("a run whose response was lost still becomes the seed Reuse last seed picks", () => {
  const page = readSrc("features/images/images-page.tsx");
  // A variation run keeps one seed across runs (the variation seed advances instead).
  assert.match(page, /const runBaseSeed = varying \? baseSeed : baseSeed \+ i \* batchSize;/);
  assert.match(page, /seed: runBaseSeed,/);
  assert.match(
    page,
    /if \(!\(err instanceof GenerateResponseLostError\)\) throw err;[\s\S]*?if \(!external\) setSessionSeed\((?:compare\?\.sentSeed\(i\) \?\? )?runBaseSeed\);\n\s+setGenDone\(i \+ 1\);\n\s+continue;/,
  );
});

test("with more gallery pages unloaded, only this session's seed is offered", () => {
  const pinned = Array.from({ length: 50 }, (_, i) => ({
    seed: i,
    created_at: i,
    pinned: true,
  }));
  assert.equal(reusableSeed(null, pinned, true), null);
  // The whole gallery is loaded: the newest record by creation time is the newest.
  assert.equal(reusableSeed(null, pinned, false), 49);
  assert.equal(
    reusableSeed(null, [...pinned, { seed: 7, created_at: 100 }], false),
    7,
  );
  // Unpinned rows page in manual (drag) order, so a loaded unpinned row does not prove the newest is loaded.
  assert.equal(
    reusableSeed(null, [...pinned, { seed: 7, created_at: 100 }], true),
    null,
  );
  assert.equal(reusableSeed(null, [{ seed: 7, created_at: 100 }], true), null);
  assert.equal(reusableSeed(3, pinned, true), 3);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /reusableSeed\(sessionSeed, images, hasMore \|\| archivedEmpty !== true\)/);
});

test("an imported Only masked the loaded pipeline cannot crop is reported as not applied", () => {
  assert.deepEqual(
    inpaintOptionsNotApplied({ inpaint_only_masked: true, inpaint_padding: 48 }, false),
    ["inpaint area only masked (padding 48)"],
  );
  assert.deepEqual(
    inpaintOptionsNotApplied({ inpaint_only_masked: true }, false),
    [`inpaint area only masked (padding ${INPAINT_PADDING_DEFAULT})`],
  );
  assert.deepEqual(
    inpaintOptionsNotApplied({ inpaint_only_masked: true, inpaint_padding: 48 }, true),
    [],
  );
  assert.deepEqual(inpaintOptionsNotApplied({ inpaint_only_masked: false }, false), []);
  assert.deepEqual(inpaintOptionsNotApplied({}, false), []);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /const missed = inpaintOptionsNotApplied\(settings, status\.supports_inpaint_only_masked === true\);\n\s+notApplied\.unshift\(\.\.\.missed\);/,
  );
});

test("Only masked on with a model that cannot crop is reported once per model, however it got there", () => {
  const page = readSrc("features/images/images-page.tsx");
  // One effect over the status and the switch: an import, a restore, a toggle or a model switch.
  assert.match(
    page,
    /const key =\n\s+status\?\.loaded && inpaintOnlyMasked && status\.supports_inpaint_only_masked !== true\n\s+\? onlyMaskedKeyOf\(status\)\n\s+: null;\n\s+if \(key === null\) \{\n\s+onlyMaskedNotified\.current = null;\n\s+return;\n\s+\}\n\s+if \(onlyMaskedNotified\.current === key\) return;\n\s+onlyMaskedNotified\.current = key;/,
  );
  assert.match(page, /\}, \[status, inpaintOnlyMasked, inpaintPadding\]\);/);
  // The import names it in its own toast, so the effect does not repeat it.
  assert.match(
    page,
    /if \(missed\.length > 0\) onlyMaskedNotified\.current = onlyMaskedKeyOf\(status\);/,
  );
  assert.doesNotMatch(page, /onlyMaskedRecheck/);
});

test("the gallery seed fallback waits until the archive is known to be empty", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /if \(sessionSeed !== null \|\| hasMore\) return;\n\s+let cancelled = false;\n\s+getGallery\(0, 1, true\)\n\s+\.then\(\(page\) => \{\n\s+if \(!cancelled\) setArchivedEmpty\(page\.images\.length === 0\);/,
  );
  assert.match(page, /if \(!cancelled\) setArchivedEmpty\(false\);/);
  assert.match(page, /\}, \[sessionSeed, hasMore, images\.length, archiveChangeKey\]\);/);
  // An archive change made elsewhere (Settings) re-runs the check even when the strip is unchanged.
  assert.match(
    page,
    /subscribeArchiveChanged\("images", \(\) => setArchiveChangeKey\(\(k\) => k \+ 1\)\)/,
  );
  const dialog = readSrc("features/settings/components/archived-media-dialog.tsx");
  assert.match(dialog, /if \(!succeeded\) break;\n\s+changed = true;/);
  assert.match(dialog, /if \(changed\) notifyArchiveChanged\(kind\);/);
  // The gallery event (which refetches the active strip) still fires only for a restore.
  assert.match(dialog, /if \(restored\) notifyGalleryChanged\(kind\);/);
  // Unknown or non-empty archive: no gallery seed, so an archived newest image is never skipped.
  assert.equal(reusableSeed(null, [{ seed: 7, created_at: 1 }], true), null);
});
