// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storage } = installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { applyActiveModelStatusToStore } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);

const NPU = "lemonade:qwen3.8-tk-4b-FLM";
const GPU = "unsloth/Qwen3-8B";

function loadWithTools(modelId: string) {
  storage.delete("unsloth_chat_tools_enabled");
  storage.delete("unsloth_chat_code_tools_enabled");
  useChatRuntimeStore.setState({
    modelLoading: false,
    toolsEnabled: true,
    codeToolsEnabled: true,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: modelId },
  });
  return {
    active_model: modelId,
    model_identifier: modelId,
    supports_tools: true,
  } as never;
}

test("an NPU model loads with Search and Code off when the user never chose", () => {
  applyActiveModelStatusToStore(loadWithTools(NPU), { previousCheckpoint: GPU });
  assert.equal(useChatRuntimeStore.getState().supportsTools, true);
  assert.equal(useChatRuntimeStore.getState().toolsEnabled, false);
  assert.equal(useChatRuntimeStore.getState().codeToolsEnabled, false);
});

test("a GPU model with tools still loads with Search and Code on", () => {
  applyActiveModelStatusToStore(loadWithTools(GPU), { previousCheckpoint: NPU });
  assert.equal(useChatRuntimeStore.getState().toolsEnabled, true);
  assert.equal(useChatRuntimeStore.getState().codeToolsEnabled, true);
});

test("an NPU model keeps a saved explicit choice to turn the pills on", () => {
  const status = loadWithTools(NPU);
  useChatRuntimeStore.getState().setToolsEnabled(true);
  useChatRuntimeStore.getState().setCodeToolsEnabled(true);
  applyActiveModelStatusToStore(status, { previousCheckpoint: GPU });
  assert.equal(useChatRuntimeStore.getState().toolsEnabled, true);
  assert.equal(useChatRuntimeStore.getState().codeToolsEnabled, true);
});
