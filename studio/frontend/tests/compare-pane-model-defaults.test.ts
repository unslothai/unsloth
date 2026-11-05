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
const { resolveQwenThinkingParams } = await import(
  "../src/features/chat/utils/qwen-sampling-table.ts"
);
const { loadedContextFields } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const QWEN = "unsloth/Qwen3-0.6B-GGUF";
const GEMMA = "unsloth/gemma-3-270m-it-GGUF";
const QWEN35 = "unsloth/Qwen3.5-27B-GGUF";
const CONTEXT = 32768;
const REQUESTED_CONTEXT = 4096;
const RECOMMENDED: Record<string, { temperature: number; top_p: number; top_k: number }> = {
  [QWEN]: { temperature: 0.6, top_p: 0.95, top_k: 20 },
  [GEMMA]: { temperature: 1, top_p: 0.95, top_k: 64 },
  [QWEN35]: { temperature: 0.7, top_p: 0.8, top_k: 20 },
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
  "resolveQwenThinkingParams",
  `let upgradeUnloadedActive; ${composer.slice(start, end)}`,
);

function response(model: string, gguf = true) {
  return {
    status: "loaded",
    model,
    supports_reasoning: model === QWEN35,
    is_gguf: gguf,
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

function compareLoad(model: string, gguf = true): void {
  commitPaneLoad(
    useChatRuntimeStore,
    response(model, gguf),
    { id: model, ggufVariant: gguf ? "Q4_K_M" : null },
    gguf,
    gguf ? 0 : REQUESTED_CONTEXT,
    loadedContextFields,
    replayMaxTokensCap,
    mergeBackendRecommendedInference,
    loadedContextForParams,
    resolveQwenThinkingParams,
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

test("a non-GGUF compare pane load takes its sampling and context without filing an entry", async () => {
  pickerLoad(QWEN);
  compareLoad(GEMMA, false);
  assert.deepEqual(sampling(), { temperature: 1, topP: 0.95, topK: 64 });
  assert.equal(useChatRuntimeStore.getState().params.maxSeqLength, REQUESTED_CONTEXT);
  assert.equal(useChatRuntimeStore.getState().paramsByModel[GEMMA], undefined);

  compareLoad(QWEN, false);
  assert.deepEqual(sampling(), { temperature: 0.6, topP: 0.95, topK: 20 });
  await settled();
  assert.equal(stored()[GEMMA]?.temperature, 1);
  assert.equal(stored()[GEMMA]?.topK, 64);
});

test("a compare pane Qwen model gets the sampling for the active thinking mode", () => {
  pickerLoad(GEMMA);
  useChatRuntimeStore.setState({ reasoningEnabled: true });
  compareLoad(QWEN35);
  assert.deepEqual(sampling(), { temperature: 0.6, topP: 0.95, topK: 20 });

  compareLoad(GEMMA);
  useChatRuntimeStore.setState({ reasoningEnabled: false });
  useChatRuntimeStore.setState({ paramsByModel: {} });
  compareLoad(QWEN35);
  assert.deepEqual(sampling(), { temperature: 0.7, topP: 0.8, topK: 20 });
});
