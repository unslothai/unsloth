// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const hook = readSrc("features/api-monitor/use-api-model-reload.ts");
const page = readSrc("features/api-monitor/api-monitor-page.tsx");

test("a served row reloads its checkpoint, not the public alias a kept slot is listed under", () => {
  assert.match(hook, /id: model\.checkpoint === model\.name \? id : model\.checkpoint,/);
  assert.doesNotMatch(hook, /id: standalone \? model\.checkpoint : id,/);
  assert.match(page, /checkpoint: model\.checkpoint \?\? model\.id,/);
});

test("the Context fact finds requests logged as <id>:<quant>, and shows chat's pinnable chip", () => {
  assert.match(page, /modelIdsMatch\(monitorSelection\(entry\.model\)\.id, id\) &&/);
  assert.doesNotMatch(page, /modelIdsMatch\(entry\.model, id\)/);
  assert.match(page, /<ContextUsageChip\s+pinOnClick=\{true\}/);
});

test("an MLX precision folder reloads as a model with its folder, not as a GGUF quant", () => {
  assert.match(hook, /const variant = model\.quant \?\? ggufVariant;/);
  assert.match(hook, /kind: variant != null && !model\.isMlx \? "gguf" : "model",/);
  assert.match(page, /const loadedIsMlx = useChatRuntimeStore\(\(s\) => s\.loadedIsMlx\);/);
  assert.match(page, /isMlx: index === activeIndex && loadedIsMlx === true,/);
});
