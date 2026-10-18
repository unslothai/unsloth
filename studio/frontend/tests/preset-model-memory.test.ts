// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Picking a preset is not a per-model edit (#14068). Default fell back to fixed sampling for every
// model, and a custom preset's values came back on the next load after switching to Default. These
// drive the real store the way the settings sheet, a model load and a status poll do.

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
  applyPresetForProvider,
  getPresetSource,
  mergeBackendRecommendedInference,
} = await import("../src/features/chat/presets/preset-policy.ts");
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

type Preset = (typeof BUILTIN_PRESETS)[number];

const GEMMA = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive";
const QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF";
const CONTEXT = 131072;
/** What the backend recommends for every model here. */
const RECOMMENDED = { temperature: 1, top_p: 0.9, top_k: 64 };
const DEFAULT = BUILTIN_PRESETS[0];
const CUSTOM: Preset = {
  name: "Pirate",
  params: {
    ...DEFAULT_INFERENCE_PARAMS,
    temperature: 0.3,
    topP: 0.7,
    topK: 10,
    systemPrompt: "Talk like a pirate.",
  },
};

/** A model load, or a status poll when the checkpoint is already the one loaded. */
function loadOrPoll(modelId: string): void {
  const store = useChatRuntimeStore.getState();
  store.setParams(
    mergeBackendRecommendedInference({
      current: store.params,
      response: { inference: RECOMMENDED, is_gguf: true, context_length: CONTEXT },
      modelId,
      presetSource: store.activePresetSource,
      loadedContextLength: CONTEXT,
    }),
    { fromModelDefaults: true, maxTokensCap: CONTEXT },
  );
}

/** chat-settings-sheet's applyPreset. */
function pickPreset(preset: Preset): void {
  const store = useChatRuntimeStore.getState();
  store.setParams(applyPresetForProvider(store.params, preset, null), {
    minPChoiceEdited: true,
    fromPreset: true,
  } as never);
  store.setActivePreset(preset.name);
  store.setActivePresetSource(getPresetSource(preset.name));
}

function sampling() {
  const { temperature, topP, topK, systemPrompt } =
    useChatRuntimeStore.getState().params;
  return { temperature, topP, topK, systemPrompt };
}

const AUTOMATIC = { temperature: 1, topP: 0.9, topK: 64, systemPrompt: "" };

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

test.before(async () => {
  settingsHttp.settings = {};
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().settingsHydrated, true);
  assert.equal(useChatRuntimeStore.getState().rememberParamsPerModel, true);
});

test.beforeEach(() => {
  useChatRuntimeStore.getState().clearCheckpoint();
  useChatRuntimeStore.setState({ paramsByModel: {} });
  pickPreset(DEFAULT);
});

/** What the server holds after every write so far, through the backend's merge. */
function stored(): Record<string, unknown> {
  return settingsHttp.puts.reduce(
    (settings, put) => deepMerge(settings, put),
    settingsHttp.settings,
  );
}

test("picking Default with a model loaded goes back to that model's recommendation", async () => {
  loadOrPoll(GEMMA);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.2 });
  await settled();
  assert.equal(
    (stored().inferenceParamsByModel as Record<string, { temperature?: number }>)[GEMMA]
      ?.temperature,
    0.2,
    "a slider edit is stored for the model",
  );
  pickPreset(CUSTOM);
  pickPreset(DEFAULT);
  loadOrPoll(GEMMA);
  assert.deepEqual(
    sampling(),
    AUTOMATIC,
    "Default's fixed sampling is not filed as the model's own",
  );
  await settled();
  const byModel = stored().inferenceParamsByModel as Record<string, unknown>;
  assert.equal(byModel[GEMMA], undefined, "nothing is left for the next launch");
  assert.equal(stored().forgetInferenceParamsByModel, undefined);
});

test("a custom preset does not come back after ejecting and picking Default", () => {
  loadOrPoll(GEMMA);
  pickPreset(CUSTOM);
  loadOrPoll(GEMMA);
  assert.deepEqual(sampling(), {
    temperature: 0.3,
    topP: 0.7,
    topK: 10,
    systemPrompt: "Talk like a pirate.",
  });
  useChatRuntimeStore.getState().clearCheckpoint();
  pickPreset(DEFAULT);
  loadOrPoll(GEMMA);
  assert.deepEqual(sampling(), AUTOMATIC);
});

test("a model first loaded under a custom preset does not keep it", () => {
  pickPreset(CUSTOM);
  loadOrPoll(QWEN);
  assert.equal(sampling().systemPrompt, "Talk like a pirate.");
  useChatRuntimeStore.getState().clearCheckpoint();
  pickPreset(DEFAULT);
  loadOrPoll(GEMMA);
  assert.deepEqual(sampling(), AUTOMATIC);
  loadOrPoll(QWEN);
  assert.deepEqual(sampling(), AUTOMATIC);
});

test("ejecting right after picking Default does not file its fixed sampling", () => {
  loadOrPoll(GEMMA);
  pickPreset(DEFAULT);
  assert.equal(sampling().temperature, DEFAULT_INFERENCE_PARAMS.temperature);
  useChatRuntimeStore.getState().clearCheckpoint();
  loadOrPoll(GEMMA);
  assert.deepEqual(sampling(), AUTOMATIC);
});

// Edits and picks share one debounced write, so the forget has to order against the edits queued
// beside it.
test("a pick and a slider edit in one write window land in the order they were made", async () => {
  loadOrPoll(GEMMA);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.2, systemPrompt: "Old" });
  pickPreset(DEFAULT);
  await settled();
  const byModel = () =>
    stored().inferenceParamsByModel as Record<string, Record<string, unknown>>;
  assert.equal(byModel()[GEMMA], undefined, "an edit queued before the pick is forgotten");

  loadOrPoll(GEMMA);
  pickPreset(CUSTOM);
  const live = useChatRuntimeStore.getState();
  live.setActivePresetSource("modified");
  live.setParams({ ...live.params, topK: 5 });
  await settled();
  assert.deepEqual(
    byModel()[GEMMA],
    { topK: 5 },
    "an edit after the pick starts a fresh entry, without the preset or the older one",
  );
});

// The memory itself still works: a slider moved on a model is that model's and outlasts a switch.
test("a slider edit is still remembered per model over its recommendation", () => {
  loadOrPoll(GEMMA);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.2 });
  loadOrPoll(QWEN);
  assert.equal(sampling().temperature, 0.2, "an unremembered model keeps what is on screen");
  useChatRuntimeStore.getState().setParams({
    ...useChatRuntimeStore.getState().params,
    temperature: 0.5,
  });
  loadOrPoll(GEMMA);
  assert.equal(sampling().temperature, 0.2);
  loadOrPoll(QWEN);
  assert.equal(sampling().temperature, 0.5);
});
