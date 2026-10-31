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
const { useChatRuntimeStore, beginThreadScopedPairing } = await import(
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

const NOT_PRESET_OWNED = ["checkpoint", "maxSeqLength", "trustRemoteCode", "fastMode"];
const PRESET_OWNED = new Set(
  Object.keys(DEFAULT.params).filter((key) => !NOT_PRESET_OWNED.includes(key)),
);

function presetOwnedKeys(entry: Record<string, unknown> | undefined): string[] {
  return Object.keys(entry ?? {}).filter((key) => PRESET_OWNED.has(key));
}

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
  const byModel = stored().inferenceParamsByModel as Record<string, Record<string, unknown>>;
  assert.deepEqual(
    presetOwnedKeys(byModel[GEMMA]),
    [],
    "no sampling is left for the next launch",
  );
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
  assert.deepEqual(
    presetOwnedKeys(byModel()[GEMMA]),
    [],
    "an edit queued before the pick is forgotten",
  );

  loadOrPoll(GEMMA);
  pickPreset(CUSTOM);
  const live = useChatRuntimeStore.getState();
  live.setActivePresetSource("modified");
  live.setParams({ ...live.params, topK: 5 });
  await settled();
  assert.deepEqual(
    presetOwnedKeys(byModel()[GEMMA]),
    ["topK"],
    "an edit after the pick starts a fresh entry, without the preset or the older one",
  );
  assert.equal(byModel()[GEMMA].topK, 5);
});

// Presets do not set these, so a pick keeps them for the model.
test("a preset pick keeps the model's remembered Fast mode", async () => {
  loadOrPoll(GEMMA);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, fastMode: true, temperature: 0.2 });
  pickPreset(CUSTOM);
  await settled();
  const local = useChatRuntimeStore.getState().paramsByModel[GEMMA];
  assert.equal(local?.fastMode, true);
  assert.deepEqual(presetOwnedKeys(local), []);
  const byModel = stored().inferenceParamsByModel as Record<string, Record<string, unknown>>;
  assert.equal(byModel[GEMMA].fastMode, true);
  assert.deepEqual(presetOwnedKeys(byModel[GEMMA]), []);
});

// The edit is filed as itself, not as a snapshot carrying the preset's sampling and prompt.
test("an edit after a preset pick does not file the preset for the model", () => {
  loadOrPoll(GEMMA);
  pickPreset(CUSTOM);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, topK: 5 });
  store.setParams({ ...useChatRuntimeStore.getState().params, fastMode: true });
  useChatRuntimeStore.getState().clearCheckpoint();
  pickPreset(DEFAULT);
  loadOrPoll(GEMMA);
  assert.deepEqual(sampling(), { ...AUTOMATIC, topK: 5 });
  assert.equal(useChatRuntimeStore.getState().params.fastMode, true);
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

// The model was not on screen when the preset was picked, so the pick could not mark it.
test("a model first loaded under a custom preset files later edits as themselves", () => {
  pickPreset(CUSTOM);
  loadOrPoll(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, fastMode: true });
  useChatRuntimeStore.getState().setActivePresetSource("modified");
  useChatRuntimeStore.getState().setParams({ ...useChatRuntimeStore.getState().params, topK: 5 });
  assert.deepEqual(presetOwnedKeys(useChatRuntimeStore.getState().paramsByModel[QWEN]), ["topK"]);
  useChatRuntimeStore.getState().clearCheckpoint();
  pickPreset(DEFAULT);
  loadOrPoll(QWEN);
  assert.deepEqual(sampling(), { ...AUTOMATIC, topK: 5 });
});

// Another tab may have set a key the pick does not own, so the pick names keys, not the row.
test("a preset pick asks the server to drop only the preset's keys", async () => {
  loadOrPoll(GEMMA);
  settingsHttp.puts.length = 0;
  pickPreset(DEFAULT);
  await settled();
  const forget = settingsHttp.puts.find((put) => put.forgetInferenceParamsByModel)
    ?.forgetInferenceParamsByModel as Record<string, string[]>;
  assert.deepEqual(new Set(forget[GEMMA]), PRESET_OWNED);
});

// An edit made while nothing was loaded is the user's, so reloading under it ends the preset mark.
test("a model reloaded under the user's own edits is remembered whole again", () => {
  loadOrPoll(GEMMA);
  pickPreset(DEFAULT);
  useChatRuntimeStore.getState().clearCheckpoint();
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.4 });
  loadOrPoll(GEMMA);
  const loaded = useChatRuntimeStore.getState();
  loaded.setParams({ ...loaded.params, topK: 7 });
  const entry = useChatRuntimeStore.getState().paramsByModel[GEMMA];
  assert.equal(entry?.topK, 7);
  assert.equal(entry?.temperature, 0.4, "the whole snapshot, not just the moved key");
});

// A status poll for the model already on screen replaces none of the preset's values.
test("a same-model status poll keeps later edits filed as themselves", () => {
  loadOrPoll(GEMMA);
  pickPreset(CUSTOM);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, topK: 5 });
  loadOrPoll(GEMMA);
  useChatRuntimeStore.getState().setParams({ ...useChatRuntimeStore.getState().params, topP: 0.5 });
  assert.deepEqual(
    presetOwnedKeys(useChatRuntimeStore.getState().paramsByModel[GEMMA]).sort(),
    ["topK", "topP"],
  );
});

// The model's row is forgotten, so a preset value that already matched the screen must still reach
// the defaults the model falls back to after a restart.
test("a preset pick writes the whole preset to the installation defaults", async () => {
  loadOrPoll(GEMMA);
  pickPreset(CUSTOM);
  await settled();
  settingsHttp.puts.length = 0;
  pickPreset(CUSTOM);
  await settled();
  const written = Object.assign(
    {},
    ...settingsHttp.puts.map((put) => (put.inferenceParams ?? {}) as Record<string, unknown>),
  );
  assert.equal(written.temperature, 0.3);
  assert.equal(written.topK, 10);
  assert.equal(written.systemPrompt, "Talk like a pirate.");
});

// Last: the open chat's pairing stays for the rest of the module.
// With a saved chat open the preset's sampling belongs to that chat, so the model's own stays.
test("a preset picked inside a chat leaves the model's remembered sampling alone", () => {
  loadOrPoll(GEMMA);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.2 });
  store.setActiveThreadId("chat-1" as never);
  beginThreadScopedPairing("chat-1");
  pickPreset(CUSTOM);
  assert.equal(useChatRuntimeStore.getState().paramsByModel[GEMMA]?.temperature, 0.2);
  // The chat kept the sampling, so the model is not marked: a later edit it owns files whole.
  useChatRuntimeStore.setState({ paramsByModel: {} });
  const inChat = useChatRuntimeStore.getState();
  const fastMode = !inChat.params.fastMode;
  inChat.setParams({ ...inChat.params, fastMode });
  const entry = useChatRuntimeStore.getState().paramsByModel[GEMMA] ?? {};
  assert.equal(entry.fastMode, fastMode);
  assert.ok("maxTokens" in entry, "a whole snapshot, not only the moved key");
});
