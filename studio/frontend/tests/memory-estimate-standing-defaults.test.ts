// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two values the Estimated Memory Usage row has to resolve the way the launch does,
// because both are absent from the per-model record in the ordinary case rather than
// in an edge one: the GPU memory mode (only "manual" is ever persisted per model) and
// the VRAM budget fraction (read asynchronously, and null on an older backend). Read
// as "unset means the default" they price a different launch than the one the Load
// button starts. Asserted against the source, the idiom tensor-parallel-row-gating
// already uses for logic that lives inside the component.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { DEFAULT_VRAM_FRACTION } from "../src/hooks/gpu-vram.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const CONFIG_PAGE = read(
  "src/features/model-picker/components/model-config-page.tsx",
);
const APPLY_CONFIG = read(
  "src/features/model-picker/model-config/apply-per-model-config.ts",
);
const NORMALIZE_CONFIG = read(
  "src/features/model-picker/model-config/per-model-config.ts",
);
const BUDGET_SETTINGS = read("../backend/utils/vram_budget_settings.py");

test("only Manual is persisted per model, so an absent mode is not Auto", () => {
  // The premise of the whole fix. If this stopped holding, reading the absence as
  // Auto would be right and the resolution below would be the wrong thing to keep.
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
  // A Manual placement with fixed layers is launched verbatim, so the VRAM budget
  // must not be applied to it, and its context comes from the resident load rather
  // than the control. Both branches turn on the mode.
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
