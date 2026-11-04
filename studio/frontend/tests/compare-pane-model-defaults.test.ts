// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp, deepMerge } = await import(
  "./helpers/store-stubs/settings-http.ts"
);
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const {
  loadedContextForParams,
  mergeBackendRecommendedInference,
  replayMaxTokensCap,
} = await import("../src/features/chat/presets/preset-policy.ts");
const { loadedContextFields } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const QWEN = "unsloth/Qwen3-0.6B-GGUF";
const GEMMA = "unsloth/gemma-3-270m-it-GGUF";
const CONTEXT = 32768;
const RECOMMENDED: Record<string, { temperature: number; top_p: number; top_k: number }> = {
  [QWEN]: { temperature: 0.6, top_p: 0.95, top_k: 20 },
  [GEMMA]: { temperature: 1, top_p: 0.95, top_k: 64 },
};

const composer = readSrc("features/chat/shared-composer.tsx");
const loadAt = composer.indexOf("const resp = await loadModel(");
const start = composer.indexOf("upgradeUnloadedActive = false;", loadAt);
const end = composer.indexOf("store.setModelRequiresTrustRemoteCode(", start);
assert.ok(loadAt >= 0 && start > loadAt && end > start);
const commitPaneLoad = new Function(
  "useChatRuntimeStore",
  "resp",
  "sel",
  "targetIsGguf",
  "effectiveMaxSeqLength",
  "loadedContextFields",
  "replayMaxTokensCap",
  "mergeBackendRecommendedInference",
  "loadedContextForParams",
  `let upgradeUnloadedActive; ${composer.slice(start, end)}`,
);

function response(model: string) {
  return {
    status: "loaded",
    model,
    is_gguf: true,
    context_length: CONTEXT,
    inference: RECOMMENDED[model],
  };
}

function pickerLoad(model: string): void {
  const store = useChatRuntimeStore.getState();
  store.setParams(
    mergeBackendRecommendedInference({
      current: store.params,
      response: response(model),
      modelId: model,
      presetSource: store.activePresetSource,
      loadedContextLength: CONTEXT,
    }),
    { fromModelDefaults: true, maxTokensCap: CONTEXT },
  );
  store.setCheckpoint(model, "Q4_K_M");
}

function compareLoad(model: string): void {
  commitPaneLoad(
    useChatRuntimeStore,
    response(model),
    { id: model, ggufVariant: "Q4_K_M" },
    true,
    0,
    loadedContextFields,
    replayMaxTokensCap,
    mergeBackendRecommendedInference,
    loadedContextForParams,
  );
}

function sampling() {
  const { temperature, topP, topK } = useChatRuntimeStore.getState().params;
  return { temperature, topP, topK };
}

function stored(): Record<string, Record<string, unknown>> {
  const settings = settingsHttp.puts.reduce(
    (merged, put) => deepMerge(merged, put),
    settingsHttp.settings,
  );
  return (settings.inferenceParamsByModel ?? {}) as Record<string, Record<string, unknown>>;
}

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

test.before(async () => {
  settingsHttp.settings = {};
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().rememberParamsPerModel, true);
  assert.equal(useChatRuntimeStore.getState().activePresetSource, "builtin-default");
});

test.beforeEach(() => {
  useChatRuntimeStore.getState().clearCheckpoint();
  useChatRuntimeStore.setState({ paramsByModel: {} });
  settingsHttp.settings = {};
  settingsHttp.puts = [];
});

test("a compare pane model runs with its own recommended sampling", async () => {
  pickerLoad(QWEN);
  assert.deepEqual(sampling(), { temperature: 0.6, topP: 0.95, topK: 20 });

  compareLoad(GEMMA);
  assert.deepEqual(sampling(), { temperature: 1, topP: 0.95, topK: 64 });

  compareLoad(QWEN);
  assert.deepEqual(sampling(), { temperature: 0.6, topP: 0.95, topK: 20 });

  const byModel = useChatRuntimeStore.getState().paramsByModel;
  assert.equal(byModel[GEMMA]?.temperature, 1);
  assert.equal(byModel[GEMMA]?.topK, 64);
  await settled();
  assert.equal(stored()[GEMMA]?.temperature, 1);
  assert.equal(stored()[GEMMA]?.topK, 64);

  pickerLoad(GEMMA);
  assert.deepEqual(sampling(), { temperature: 1, topP: 0.95, topK: 64 });
});

test("a compare pane model keeps its remembered sampling over the recommendation", () => {
  useChatRuntimeStore.setState({
    paramsByModel: { [GEMMA]: { temperature: 0.4, topK: 30 } },
  });
  pickerLoad(QWEN);
  compareLoad(GEMMA);
  assert.equal(sampling().temperature, 0.4);
  assert.equal(sampling().topK, 30);
  compareLoad(QWEN);
  assert.deepEqual(sampling(), { temperature: 0.6, topP: 0.95, topK: 20 });
  assert.equal(useChatRuntimeStore.getState().paramsByModel[GEMMA]?.temperature, 0.4);
});
