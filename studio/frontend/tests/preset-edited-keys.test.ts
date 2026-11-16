// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An edited Default keeps only the recommended keys the user moved. A system prompt edit used to
// mark the whole pick `modified`, so every model loaded afterwards ran the previous model's sampling.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

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
  BUILTIN_PRESETS,
  RECOMMENDED_SAMPLING_KEYS,
  applyPresetForProvider,
  getPresetSource,
  mergeBackendRecommendedInference,
} = await import("../src/features/chat/presets/preset-policy.ts");
const { applyQwenThinkingParams } = await import(
  "../src/features/chat/utils/qwen-params.ts"
);
const { sanitizeChatSettings } = await import(
  "../src/features/chat/utils/chat-settings-storage.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF";
const GEMMA = "unsloth/gemma-4-E4B-it-GGUF";
const LLAMA = "unsloth/Llama-3.2-1B-Instruct-GGUF";
const CONTEXT = 131072;
const RECOMMENDED: Record<string, Record<string, number>> = {
  [QWEN]: { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5 },
  [GEMMA]: { temperature: 1, top_p: 0.95, top_k: 64, min_p: 0, presence_penalty: 0 },
  [LLAMA]: { temperature: 0.6, top_p: 0.9, top_k: 40, min_p: 0.05, presence_penalty: 0 },
};
const OWN = {
  [QWEN]: { temperature: 0.7, topP: 0.8, topK: 20, presencePenalty: 1.5 },
  [GEMMA]: { temperature: 1, topP: 0.95, topK: 64, presencePenalty: 0 },
  [LLAMA]: { temperature: 0.6, topP: 0.9, topK: 40, presencePenalty: 0 },
};
const PIRATE = {
  name: "Pirate",
  params: { ...DEFAULT_INFERENCE_PARAMS, temperature: 0.3, topP: 0.7, topK: 10 },
};

function load(modelId: string): void {
  const store = useChatRuntimeStore.getState();
  store.setParams(
    mergeBackendRecommendedInference({
      current: store.params,
      response: { inference: RECOMMENDED[modelId], is_gguf: true, context_length: CONTEXT },
      modelId,
      presetSource: store.activePresetSource,
      presetEditedKeys: store.presetEditedKeys,
      loadedContextLength: CONTEXT,
    }),
    { fromModelDefaults: true, maxTokensCap: CONTEXT },
  );
}

/** chat-settings-sheet's applyPreset. */
function pickPreset(preset: { name: string; params: typeof DEFAULT_INFERENCE_PARAMS }): void {
  const store = useChatRuntimeStore.getState();
  store.setParams(applyPresetForProvider(store.params, preset, null), {
    minPChoiceEdited: true,
    fromPreset: true,
  } as never);
  store.setActivePreset(preset.name);
  store.setActivePresetSource(getPresetSource(preset.name));
}

/** chat-settings-sheet's saveSystemPromptEditor: no recommended key moves. */
function saveSystemPrompt(systemPrompt: string): void {
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, systemPrompt });
}

/** chat-settings-sheet's set(key) for a recommended key. */
function moveSlider(key: "temperature" | "topK", value: number): void {
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified", key);
  store.setParams({ ...useChatRuntimeStore.getState().params, [key]: value });
}

function sampling() {
  const { temperature, topP, topK, presencePenalty } = useChatRuntimeStore.getState().params;
  return { temperature, topP, topK, presencePenalty };
}

function stored(): Record<string, unknown> {
  return settingsHttp.puts.reduce(
    (settings, put) => deepMerge(settings, put),
    settingsHttp.settings,
  );
}

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

test.before(async () => {
  settingsHttp.settings = {};
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().rememberParamsPerModel, true);
});

test.beforeEach(() => {
  useChatRuntimeStore.getState().clearCheckpoint();
  useChatRuntimeStore.setState({ paramsByModel: {}, rememberParamsPerModel: true });
  pickPreset(BUILTIN_PRESETS[0]);
});

test("a system prompt edit leaves every later model on its own recommendation", () => {
  load(QWEN);
  saveSystemPrompt("Be terse.");
  assert.equal(useChatRuntimeStore.getState().activePresetSource, "modified");
  load(GEMMA);
  assert.deepEqual(sampling(), OWN[GEMMA]);
  assert.equal(useChatRuntimeStore.getState().params.systemPrompt, "Be terse.");
  load(QWEN);
  load(GEMMA);
  assert.deepEqual(sampling(), OWN[GEMMA], "Qwen's values are not filed under Gemma");
  const filed = useChatRuntimeStore.getState().paramsByModel[GEMMA];
  assert.equal(filed?.temperature, OWN[GEMMA].temperature);
  assert.equal(filed?.presencePenalty, OWN[GEMMA].presencePenalty);
});

test("a moved slider carries over while untouched keys follow the new model", () => {
  load(QWEN);
  moveSlider("temperature", 0.2);
  load(GEMMA);
  assert.deepEqual(sampling(), { ...OWN[GEMMA], temperature: 0.2 });
  moveSlider("topK", 5);
  load(LLAMA);
  assert.deepEqual(sampling(), { ...OWN[LLAMA], temperature: 0.2, topK: 5 });
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, ["temperature", "topK"]);
});

test("with per-model memory off, a system prompt edit still takes each model's recommendation", () => {
  useChatRuntimeStore.setState({ rememberParamsPerModel: false });
  load(QWEN);
  saveSystemPrompt("Be terse.");
  load(GEMMA);
  assert.deepEqual(sampling(), OWN[GEMMA]);
  moveSlider("temperature", 0.3);
  load(LLAMA);
  assert.deepEqual(sampling(), { ...OWN[LLAMA], temperature: 0.3 });
});

test("an edited custom preset still keeps every key", () => {
  load(QWEN);
  pickPreset(PIRATE);
  saveSystemPrompt("Arr.");
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, [...RECOMMENDED_SAMPLING_KEYS]);
  load(GEMMA);
  assert.deepEqual(sampling(), { temperature: 0.3, topP: 0.7, topK: 10, presencePenalty: 0 });
});

test("picking Default again drops the kept keys", () => {
  load(QWEN);
  moveSlider("temperature", 0.2);
  pickPreset(BUILTIN_PRESETS[0]);
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, []);
  load(GEMMA);
  assert.deepEqual(sampling(), OWN[GEMMA]);
});

test("the Qwen thinking switch skips only the keys the user moved", () => {
  load(GEMMA);
  saveSystemPrompt("Be terse.");
  load(QWEN);
  applyQwenThinkingParams(true);
  assert.deepEqual(sampling(), { temperature: 0.6, topP: 0.95, topK: 20, presencePenalty: 1.5 });
  moveSlider("temperature", 0.2);
  applyQwenThinkingParams(false);
  assert.deepEqual(sampling(), { temperature: 0.2, topP: 0.8, topK: 20, presencePenalty: 1.5 });
});

test("the kept keys are saved with the source", async () => {
  load(QWEN);
  settingsHttp.puts.length = 0;
  saveSystemPrompt("Be terse.");
  moveSlider("temperature", 0.2);
  await settled();
  assert.equal(stored().activePresetSource, "modified");
  assert.equal((stored().presetEditedKeys as Record<string, boolean>).temperature, true);
  pickPreset(BUILTIN_PRESETS[0]);
  await settled();
  assert.deepEqual(
    stored().presetEditedKeys,
    Object.fromEntries(RECOMMENDED_SAMPLING_KEYS.map((key) => [key, false])),
  );
});

test("unknown kept keys are dropped when settings are read", () => {
  assert.deepEqual(
    sanitizeChatSettings({ presetEditedKeys: { temperature: true, topK: false, systemPrompt: true, minP: 1 } })
      .presetEditedKeys,
    { temperature: true, topK: false },
  );
});

test("hydration restores the kept keys, and an older `modified` keeps every key", async () => {
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "modified",
    presetEditedKeys: { topK: true, temperature: false },
  };
  useChatRuntimeStore.setState({ settingsHydrated: false });
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, ["topK"]);

  settingsHttp.settings = { activePreset: "Default", activePresetSource: "modified" };
  useChatRuntimeStore.setState({ settingsHydrated: false });
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, [...RECOMMENDED_SAMPLING_KEYS]);
  settingsHttp.settings = {};
});

async function rehydrate(settings: Record<string, unknown>): Promise<void> {
  settingsHttp.settings = settings;
  useChatRuntimeStore.setState({ settingsHydrated: false });
  await useChatRuntimeStore.getState().hydratePersistedSettings();
}

test("a prompt edit made before hydration keeps a restored custom preset's values", async () => {
  load(QWEN);
  await settled();
  settingsHttp.settings = {
    activePreset: "Pirate",
    activePresetSource: "custom",
    inferenceParams: { temperature: 0.3, topP: 0.7, topK: 10, presencePenalty: 0 },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState({ settingsHydrated: false });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  saveSystemPrompt("Arr.");
  settingsHttp.release?.();
  await hydrating;
  assert.equal(useChatRuntimeStore.getState().activePreset, "Pirate");
  load(GEMMA);
  assert.deepEqual(sampling(), { temperature: 0.3, topP: 0.7, topK: 10, presencePenalty: 0 });
  await settled();
  await rehydrate(stored());
  load(LLAMA);
  assert.deepEqual(sampling(), { temperature: 0.3, topP: 0.7, topK: 10, presencePenalty: 0 });
  settingsHttp.settings = {};
});

test("a pre-hydration prompt edit keeps the keys an earlier session moved", async () => {
  load(QWEN);
  await settled();
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "modified",
    presetEditedKeys: { temperature: true },
    inferenceParams: { temperature: 0.2 },
  };
  useChatRuntimeStore.setState({ settingsHydrated: false });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  saveSystemPrompt("Be terse.");
  settingsHttp.release?.();
  await hydrating;
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, ["temperature"]);
  settingsHttp.settings = {};
});

test("a stale tab's prompt edit does not erase a key another tab moved", async () => {
  load(QWEN);
  await settled();
  // Another tab moved Temperature after this one hydrated on Default.
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "modified",
    presetEditedKeys: { temperature: true },
    inferenceParams: { temperature: 0.2 },
  };
  settingsHttp.puts.length = 0;
  saveSystemPrompt("Be terse.");
  await settled();
  // A fresh session: no model resident, so nothing replays over the saved temperature.
  useChatRuntimeStore.getState().clearCheckpoint();
  useChatRuntimeStore.setState({ paramsByModel: {} });
  await rehydrate(stored());
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, ["temperature"]);
  load(GEMMA);
  assert.equal(sampling().temperature, 0.2);
  settingsHttp.settings = {};
});

test("an older `modified` still keeps every key after a pre-hydration edit is saved", async () => {
  load(QWEN);
  await settled();
  settingsHttp.settings = { activePreset: "Default", activePresetSource: "modified" };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState({ settingsHydrated: false });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  saveSystemPrompt("Be terse.");
  settingsHttp.release?.();
  await hydrating;
  await settled();
  await rehydrate(stored());
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, [...RECOMMENDED_SAMPLING_KEYS]);
  moveSlider("topK", 5);
  await settled();
  await rehydrate(stored());
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, [...RECOMMENDED_SAMPLING_KEYS]);
  settingsHttp.settings = {};
});

test("a stale tab's edit does not bring back a key another tab cleared", async () => {
  load(QWEN);
  await settled();
  await rehydrate({
    activePreset: "Default",
    activePresetSource: "modified",
    presetEditedKeys: { temperature: true },
  });
  // Another tab picked Default, clearing every key.
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    presetEditedKeys: Object.fromEntries(RECOMMENDED_SAMPLING_KEYS.map((key) => [key, false])),
  };
  settingsHttp.puts.length = 0;
  moveSlider("topK", 5);
  await settled();
  await rehydrate(stored());
  assert.deepEqual(useChatRuntimeStore.getState().presetEditedKeys, ["topK"]);
  settingsHttp.settings = {};
});
