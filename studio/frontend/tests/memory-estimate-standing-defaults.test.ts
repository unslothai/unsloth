// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// GPU memory mode and VRAM budget are usually absent and must resolve as the launch does.

import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_VRAM_FRACTION } from "../src/hooks/gpu-vram.ts";

import { readSrc, readText } from "./helpers/kit.ts";

const CONFIG_PAGE = readSrc("features/model-picker/components/model-config-page.tsx");
const APPLY_CONFIG = readSrc("features/model-picker/model-config/apply-per-model-config.ts");
const NORMALIZE_CONFIG = readSrc("features/model-picker/model-config/per-model-config.ts");
const BUDGET_SETTINGS = readText("../../backend/utils/vram_budget_settings.py");

test("only Manual is persisted per model, so an absent mode is not Auto", () => {
  // Only manual is persisted per model.
  assert.match(NORMALIZE_CONFIG, /if \(partial\.gpuMemoryMode === "manual"\)/);
});

test("the load path resolves an absent mode from the standing preference", () => {
  assert.match(
    APPLY_CONFIG,
    /gpuMemoryMode:[\s\S]{0,200}config\.gpuMemoryMode \?\? readPersistedGpuMemoryMode\(\)/,
  );
});

test("the panel resolves the same absence the same way", () => {
  assert.match(
    CONFIG_PAGE,
    /const runtimeGpuMemoryMode =\s*\n?\s*runtimeConfig\.gpuMemoryMode \?\? gpuMemoryModeFallback;/,
  );
  assert.match(
    CONFIG_PAGE,
    /const \[gpuMemoryModeFallback\] = useState\(readPersistedGpuMemoryMode\);/,
  );
});

test("the estimate request sends the resolved mode, not the raw record", () => {
  assert.match(CONFIG_PAGE, /gpuMemoryMode: runtimeGpuMemoryMode,/);
  assert.doesNotMatch(
    CONFIG_PAGE,
    /gpuMemoryMode: runtimeConfig\.gpuMemoryMode \?\? null,/,
  );
});

test("the two verdicts drawn from the mode read the resolved one too", () => {
  assert.match(CONFIG_PAGE, /runtimeGpuMemoryMode !== "manual" \|\|/);
  assert.match(CONFIG_PAGE, /runtimeGpuMemoryMode === "manual" &&/);
  assert.doesNotMatch(CONFIG_PAGE, /\(runtimeConfig\.gpuMemoryMode \?\? "auto"\)/);
});

test("the budget fraction starts at the loader's default, not a full card", () => {
  assert.match(
    CONFIG_PAGE,
    /const \[memoryVramBudgetFraction, setMemoryVramBudgetFraction\] =\s*\n?\s*useState\(DEFAULT_VRAM_FRACTION\);/,
  );
});

test("that default is the fraction the backend actually applies", () => {
  const declared = BUDGET_SETTINGS.match(/^VRAM_FRACTION_DEFAULT = ([0-9.]+)$/m);
  assert.ok(declared, "backend no longer declares VRAM_FRACTION_DEFAULT");
  assert.equal(Number(declared[1]), DEFAULT_VRAM_FRACTION);
});
