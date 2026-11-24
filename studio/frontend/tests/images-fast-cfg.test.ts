// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { DiffusionGenerateRequest } from "../src/features/images/api.ts";
import { availableAxes, planCompare, type CompareContext } from "../src/features/images/compare/compare-axes.ts";
import { createCompareRun, gridParameters } from "../src/features/images/compare/compare-run.ts";
import {
  FAST_CFG_INTERVAL_END,
  fastCfgAvailable,
  fastCfgEndFromRecipe,
  fastCfgFromRecipe,
  fastCfgLabel,
  fastCfgNotApplied,
  fastCfgRequestFields,
} from "../src/features/images/lib/fast-cfg.ts";
import {
  APPLIED_RECIPE_KEYS,
  importedRecipe,
  importNotApplied,
  parseA1111Parameters,
  recipeSettings,
  unknownRecipeKeys,
} from "../src/features/images/lib/image-metadata.ts";
import { readSrc } from "./helpers/kit.ts";

const DEFAULTS = { width: 1024, height: 1024, steps: 50, guidance: 3.5 };
const LOADED = { loaded: true, supports_cfg_interval: true };

test("the switch shows only on a model that takes it, with CFG on, and sends the 0.8 preset", () => {
  assert.equal(FAST_CFG_INTERVAL_END, 0.8);
  assert.equal(fastCfgAvailable(LOADED, 3.5), true);
  assert.equal(fastCfgAvailable(LOADED, 1), false);
  assert.equal(fastCfgAvailable({ loaded: true, supports_cfg_interval: false }, 3.5), false);
  assert.equal(fastCfgAvailable({ loaded: true }, 3.5), false);
  assert.equal(fastCfgAvailable({ loaded: false, supports_cfg_interval: true }, 3.5), false);
  assert.equal(fastCfgAvailable(null, 3.5), false);
  assert.equal(fastCfgAvailable(LOADED, 3.5, true), false);
  assert.deepEqual(fastCfgRequestFields(true, true), { cfg_interval_end: 0.8 });
  assert.deepEqual(fastCfgRequestFields(true, false), {});
  assert.deepEqual(fastCfgRequestFields(false, true), {});
});

test("a Studio recipe restores the switch; full CFG and other models are handled", () => {
  const recipe = importedRecipe({
    recipe: JSON.stringify({ prompt: "p", steps: 50, guidance: 3.5, seed: 1, cfg_interval_end: 0.8, schema_version: 1 }),
  });
  assert.ok(recipe);
  assert.deepEqual(unknownRecipeKeys(recipe), []);
  assert.ok((APPLIED_RECIPE_KEYS as readonly string[]).includes("cfg_interval_end"));
  const settings = recipeSettings(recipe, DEFAULTS);
  assert.equal(settings.cfg_interval_end, 0.8);
  assert.equal(fastCfgFromRecipe(settings), true);
  assert.deepEqual(fastCfgNotApplied(recipe.cfg_interval_end, true), []);
  assert.deepEqual(fastCfgNotApplied(recipe.cfg_interval_end, false), ["Fast CFG 0.8 (not available for this model)"]);
  assert.deepEqual(importNotApplied(recipe), []);
  assert.equal(fastCfgFromRecipe(recipeSettings({ prompt: "p" }, DEFAULTS)), false);
  assert.equal(recipeSettings({ prompt: "p", cfg_interval_end: 1 }, DEFAULTS).cfg_interval_end, undefined);
  assert.equal(recipeSettings({ prompt: "p", cfg_interval_end: 1.5 }, DEFAULTS).cfg_interval_end, undefined);
  assert.equal(recipeSettings({ prompt: "p", cfg_interval_end: 0 }, DEFAULTS).cfg_interval_end, undefined);
  assert.deepEqual(fastCfgNotApplied(0.7, true), []);
  assert.equal(fastCfgLabel(0.8), "On (CFG for the first 80% of steps)");
  assert.equal(fastCfgLabel(null), null);
  assert.equal(fastCfgLabel(1), null);
});

test("the backend's infotext line parses back", () => {
  const gallery = readFileSync(new URL("../../backend/core/inference/image_gallery.py", import.meta.url), "utf8");
  assert.match(gallery, /fields \+= f", Fast CFG: \{meta\['cfg_interval_end'\]\}"/);
  const r = parseA1111Parameters("a fox\nSteps: 50, CFG scale: 3.5, Seed: 1, Size: 1024x1024, Fast CFG: 0.8");
  assert.ok(r);
  assert.equal(r.cfg_interval_end, 0.8);
  assert.equal(fastCfgFromRecipe(recipeSettings(r, DEFAULTS)), true);
});

function ctx(over: Partial<CompareContext> = {}): CompareContext {
  return {
    prompt: "a fox",
    negativePrompt: "",
    negativeCapable: true,
    negativeMinGuidance: 1,
    guidanceFixed: false,
    guidance: 3.5,
    samplers: [],
    samplerChoice: { sampler: "default", scheduleType: "automatic" },
    shift: null,
    loraCapable: false,
    loras: [],
    sdControls: null,
    controlnetEngaged: false,
    ...over,
  };
}

function base(over: Partial<DiffusionGenerateRequest> = {}): DiffusionGenerateRequest {
  return { prompt: "a fox", width: 1024, height: 1024, steps: 50, guidance: 3.5, seed: 9, batch_size: 1, ...over };
}

test("a Fast CFG compare axis runs full vs fast cells and the grid names a shared value", () => {
  const ids = (c: CompareContext) => availableAxes(c).map((a) => a.id);
  assert.ok(!ids(ctx()).includes("cfg_interval_end"));
  assert.ok(ids(ctx({ cfgInterval: true })).includes("cfg_interval_end"));
  assert.ok(!ids(ctx({ cfgInterval: true, guidanceFixed: true })).includes("cfg_interval_end"));
  const c = ctx({ cfgInterval: true });
  const r = planCompare([{ axisId: "cfg_interval_end", text: "1, 0.8" }], c, { keepRandomSeeds: false, rng: () => 0.5 });
  assert.ok(r.ok, r.ok ? "" : r.error);
  const run = createCompareRun(r.plan, c, { keepRandomSeeds: false, seedFieldEmpty: false, rng: () => 0.5 });
  const sent = r.plan.order.map((_, i) => run.request(i, base(), 7) as unknown as Record<string, unknown>);
  assert.deepEqual(sent.map((q) => q.cfg_interval_end).sort(), [0.8, 1]);
  assert.equal(planCompare([{ axisId: "cfg_interval_end", text: "0" }], c).ok, false);
  const steps = planCompare([{ axisId: "steps", text: "20, 30" }], c, { keepRandomSeeds: false, rng: () => 0.5 });
  assert.ok(steps.ok);
  const cells = [base({ steps: 20, cfg_interval_end: 0.8 }), base({ steps: 30, cfg_interval_end: 0.8 })];
  const info = gridParameters(steps.plan, c, cells, { steps: 20, guidance: 3.5, width: 1024, height: 1024, model: "m", keepRandomSeeds: false });
  assert.match(info, /Fast CFG: 0\.8/);
  assert.equal(parseA1111Parameters(info)?.cfg_interval_end, 0.8);
});

test("the page wires the switch, the request field, the restore and the backend's field names", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /usePersistedToggle\(FAST_CFG_STORAGE_KEY\)/);
  assert.match(page, /<FastCfgField available=\{fastCfgShown\} enabled=\{fastCfgOn\} onChange=\{changeFastCfg\} \/>/);
  assert.match(page, /\.\.\.fastCfgRequestFields\(fastCfgShown, fastCfgOn, fastCfgEnd\),/);
  assert.match(page, /const restoredFastCfgEnd = fastCfgEndFromRecipe\(image\);\s+setFastCfgOn\(restoredFastCfgEnd !== null\);\s+setFastCfgEnd\(restoredFastCfgEnd\);/);
  assert.match(page, /via === "restore" \? fastCfgNotApplied\(image\.cfg_interval_end, cfgIntervalSupported\) : \[\]/);
  assert.match(page, /setFastCfgOn\(next\);\s+setFastCfgEnd\(null\);/);
  const backend = readFileSync(new URL("../../backend/models/inference.py", import.meta.url), "utf8");
  assert.match(backend, /^\s+cfg_interval_end: Optional\[float\] = Field\(/m);
  assert.match(backend, /^\s+supports_cfg_interval: bool = Field\(/m);
});

test("gallery restore of a 0.7 compare cell sends 0.7 again, and warns on a model without Fast CFG", () => {
  const c = ctx({ cfgInterval: true });
  const r = planCompare([{ axisId: "cfg_interval_end", text: "1, 0.7" }], c, { keepRandomSeeds: false, rng: () => 0.5 });
  assert.ok(r.ok, r.ok ? "" : r.error);
  const run = createCompareRun(r.plan, c, { keepRandomSeeds: false, seedFieldEmpty: false, rng: () => 0.5 });
  const cell = r.plan.order
    .map((_, i) => run.request(i, base(), 7) as unknown as Record<string, unknown>)
    .find((q) => q.cfg_interval_end === 0.7);
  assert.ok(cell);
  const end = fastCfgEndFromRecipe(cell);
  assert.equal(end, 0.7);
  assert.deepEqual(fastCfgRequestFields(true, end !== null, end), { cfg_interval_end: 0.7 });
  assert.deepEqual(fastCfgRequestFields(true, true, null), { cfg_interval_end: 0.8 });
  assert.equal(fastCfgEndFromRecipe({ cfg_interval_end: 1 }), null);
  assert.equal(fastCfgEndFromRecipe({}), null);
  assert.deepEqual(fastCfgNotApplied(cell.cfg_interval_end, false), ["Fast CFG 0.7 (not available for this model)"]);
});
