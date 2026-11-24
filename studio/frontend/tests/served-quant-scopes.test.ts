// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const LOADED = readSrc("features/loaded-models/loaded-models-api.ts");

test("deep research runs on the selected quant's copy", () => {
  const call = ADAPTER.slice(ADAPTER.indexOf("buildResearchInferenceRequest({"));
  assert.match(
    call.slice(0, 300),
    /checkpoint: servedCheckpoint\(\s*runtime\.loadedModels,\s*selectedCheckpoint,\s*runtime\.activeGgufVariant,\s*\)/,
  );
});

test("the loaded panel scopes the active row's stop check to its quant", () => {
  const eject = LOADED.slice(LOADED.indexOf("async function ejectChatRow("));
  assert.match(
    eject.slice(0, 400),
    /confirmStopRunningChatsIfNeeded\(\s*"Unloading this model",\s*"unloadOne",\s*entry\.label \?\? entry\.name,\s*\)/,
  );
});

const MONITOR = readSrc("features/api-monitor/api-monitor-page.tsx");
const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

test("the API monitor marks the primary's own quant active, not the first copy of its repo", () => {
  const served = MONITOR.slice(MONITOR.indexOf("function servedModels("));
  assert.match(served, /const \{ id: activeId, ggufVariant: activeQuant \} = monitorSelection\(/);
  assert.match(served, /ggufVariantsMatch\(model\.quant, activeQuant\)/);
});

test("a settings reload checks the chats of the quant it reloads", () => {
  assert.match(
    RUNTIME,
    /touchesOnlySelected\s*\?\s*servedCheckpoint\(loadedNow, modelId, ggufVariant\)/,
  );
});

const CONTROLS = readSrc("features/api-monitor/components/api-model-load-controls.tsx");

test("the API picker shares Chat's live settings only for the same quant", () => {
  const resident = CONTROLS.slice(CONTROLS.indexOf("const runtimeIsResident ="));
  assert.match(
    resident.slice(0, 400),
    /!\(runtimeVariant && activeGgufVariant\) \|\|\s*ggufVariantsMatch\(runtimeVariant, activeGgufVariant\)/,
  );
});
