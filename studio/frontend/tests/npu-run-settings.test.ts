// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const source = (path: string) =>
  readFileSync(fileURLToPath(new URL(`../src/${path}`, import.meta.url)), "utf8");

const runtime = source("features/chat/hooks/use-chat-model-runtime.ts");
const npuLoad = runtime.slice(
  runtime.indexOf("const loadNpuModel = useCallback("),
  runtime.indexOf("const ejectModel = useCallback("),
);
const pickers = source("features/model-picker/components/model-selector/pickers.tsx");
const npuRow = pickers.slice(
  pickers.indexOf("const renderNpuRow = "),
  pickers.indexOf("const npuBrowseForcedOpen = "),
);
const configPage = source("features/model-picker/components/model-config-page.tsx");

test("an NPU load reads a context pinned in either field", () => {
  assert.match(npuLoad, /const contextLength = savedContextPin\(/);
});

test("a downloaded NPU row opens run settings with the model's context limit", () => {
  assert.match(npuRow, /onConfigure && model\.downloaded && !downloading/);
  assert.match(npuRow, /<ModelLoadSettingsAction/);
  assert.match(npuRow, /contextLength: model\.max_context_length/);
});

test("NPU run settings pin customContextLength and offer nothing FastFlowLM ignores", () => {
  assert.match(configPage, /const targetIsNpu = isNpuModelId\(target\.id\);/);
  assert.match(configPage, /const pinsContextLength = targetIsMlx \|\| targetIsNpu;/);
  assert.match(configPage, /contextPinPatch\(value, pinsContextLength\)/);
  const writers = configPage.match(/contextPinPatch\(/g) ?? [];
  const npuAware = configPage.match(/contextPinPatch\(\w+, pinsContextLength\)/g) ?? [];
  assert.ok(writers.length > 0);
  assert.equal(npuAware.length, writers.length);
  assert.match(configPage, /!target\.isGguf && !targetIsMlx && !targetIsNpu && /);
  assert.match(configPage, /target\.isGguf \|\| pinsContextLength\s+\? effectiveRuntimeConfig/);
});
