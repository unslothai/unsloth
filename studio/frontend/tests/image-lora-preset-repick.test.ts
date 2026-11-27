// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionLoraInfo } from "../src/features/images/api.ts";
import { defaultsFor } from "../src/features/images/image-generation-defaults.ts";
import { loraPresetRecipe, pickRecipe } from "../src/features/images/lib/lora-preset.ts";
import { readSrcAsync } from "./helpers/kit.ts";

function lora(id: string, extra: Partial<DiffusionLoraInfo> = {}): DiffusionLoraInfo {
  return {
    id,
    display_name: id,
    source: "hub",
    format: "safetensors",
    families: ["qwen-image"],
    size_bytes: 1,
    weight_default: 1,
    ...extra,
  };
}

const LIGHTNING = lora("lightx2v/Qwen-Image-Lightning:8steps", { recommended_steps: 8, recommended_guidance: 1 });
const STYLE = lora("someone/style-lora");
const CATALOG = [STYLE, LIGHTNING];

test("a re-pick in the same family keeps a selected Lightning LoRA's steps and CFG", () => {
  const model = defaultsFor("Qwen/Qwen-Image-2512");
  assert.notDeepEqual(model, { steps: 8, guidance: 1 });
  assert.deepEqual(pickRecipe(model, [{ id: STYLE.id }, { id: LIGHTNING.id }], CATALOG), { steps: 8, guidance: 1 });
});

test("a re-pick with no distilled LoRA selected takes the model recipe", () => {
  const model = defaultsFor("Qwen/Qwen-Image-2512");
  assert.deepEqual(pickRecipe(model, [], CATALOG), model);
  assert.deepEqual(pickRecipe(model, [{ id: STYLE.id }], CATALOG), model);
  assert.deepEqual(pickRecipe(model, [{ id: "not/in-catalog" }], CATALOG), model);
});

test("loraPresetRecipe trims ids and defaults guidance to 1", () => {
  const noCfg = lora("x/four-step", { recommended_steps: 4 });
  assert.deepEqual(loraPresetRecipe([{ id: " x/four-step " }], [noCfg]), { steps: 4, guidance: 1 });
  assert.equal(loraPresetRecipe([{ id: STYLE.id }], CATALOG), null);
});

test("applyImageModelDefaults applies the LoRA-aware recipe, and a family change restores the model one", async () => {
  const page = await readSrcAsync("features/images/images-page.tsx");
  const start = page.indexOf("const applyImageModelDefaults = useCallback(");
  const body = page.slice(start, page.indexOf("const [loadToast]", start));
  assert.match(body, /const applied = pickRecipe\(recommended, lorasRef\.current, sortedLorasRef\.current\);/);
  assert.match(body, /setSteps\(applied\.steps\);\s*setGuidance\(applied\.guidance\);/);
  assert.doesNotMatch(body, /setSteps\(recommended\.steps\)/);
  assert.match(body, /revert\.appliedSteps = applied\.steps;/);
  assert.match(page, /setSteps\(\(cur\) => \(cur === held\.preset\.steps \? held\.model\.steps : cur\)\);/);
});
