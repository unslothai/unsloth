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
    /checkpoint: servedCheckpoint\(\s*useChatRuntimeStore\.getState\(\)\.loadedModels,\s*selectedCheckpoint,\s*runtime\.activeGgufVariant,\s*\)/,
  );
});

test("the loaded panel scopes the active row's stop check to its quant", () => {
  const eject = LOADED.slice(LOADED.indexOf("async function ejectChatRow("));
  assert.match(
    eject.slice(0, 400),
    /confirmStopRunningChatsIfNeeded\(\s*"Unloading this model",\s*"unloadOne",\s*entry\.label \?\? entry\.name,\s*\)/,
  );
});
