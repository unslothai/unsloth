// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A saved preset linked to a model goes on whenever that model is selected (#14900). These drive the
// real store the way the settings sheet, a model load and a status poll do.

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
  getLinkedPreset,
  getPresetSource,
  mergeBackendRecommendedInference,
} = await import("../src/features/chat/presets/preset-policy.ts");
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

type Preset = (typeof BUILTIN_PRESETS)[number];

const GEMMA = "unsloth/gemma-4-E4B-it-GGUF";
const QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF";
const LLAMA = "unsloth/Llama-3.3-8B-Instruct-GGUF";
const CONTEXT = 131072;
const RECOMMENDED = { temperature: 1, top_p: 0.95, top_k: 64 };
const DEFAULT = BUILTIN_PRESETS[0];

function preset(name: string, temperature: number, systemPrompt: string): Preset {
  return {
    name,
    params: { ...DEFAULT_INFERENCE_PARAMS, temperature, topP: 0.8, systemPrompt },
  };
}

const CREATIVE = preset("Creative", 1.3, "Be imaginative.");
const RESEARCH = preset("Research", 0.2, "Cite your sources.");

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

/** A load, then the microtask the link is applied in. */
async function switchTo(modelId: string): Promise<void> {
  loadOrPoll(modelId);
  await Promise.resolve();
}

/** chat-settings-sheet's applyPreset. */
function pickPreset(picked: Preset): void {
  const store = useChatRuntimeStore.getState();
  store.setParams(applyPresetForProvider(store.params, picked, null), {
    minPChoiceEdited: true,
    fromPreset: true,
  });
  store.setActivePreset(picked.name);
  store.setActivePresetSource(getPresetSource(picked.name));
}

function live() {
  const { activePreset, activePresetSource, params } = useChatRuntimeStore.getState();
  return {
    activePreset,
    activePresetSource,
    temperature: params.temperature,
    systemPrompt: params.systemPrompt,
  };
}

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

function stored(): Record<string, unknown> {
  return settingsHttp.puts.reduce(
    (settings, put) => deepMerge(settings, put),
    settingsHttp.settings,
  );
}

test.before(async () => {
  // A link the server already holds, beside a removed one it echoes back as null.
  settingsHttp.settings = {
    customPresets: [
      { name: CREATIVE.name, params: { temperature: 1.3, topP: 0.8, systemPrompt: "Be imaginative." } },
      { name: RESEARCH.name, params: { temperature: 0.2, topP: 0.8, systemPrompt: "Cite your sources." } },
    ],
    modelPresets: { [QWEN]: RESEARCH.name, [LLAMA]: null },
  };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().settingsHydrated, true);
});

test.beforeEach(async () => {
  useChatRuntimeStore.getState().clearCheckpoint();
  useChatRuntimeStore.setState({ paramsByModel: {} });
  pickPreset(DEFAULT);
  await Promise.resolve();
});

test("hydration keeps the server's links and drops the removed ones", () => {
  assert.deepEqual(useChatRuntimeStore.getState().modelPresets, {
    [QWEN]: RESEARCH.name,
  });
});

test("switching to a linked model applies its preset", async () => {
  await switchTo(QWEN);
  assert.deepEqual(live(), {
    activePreset: "Research",
    activePresetSource: "custom",
    temperature: 0.2,
    systemPrompt: "Cite your sources.",
  });
});

test("each linked model gets its own preset, and an unlinked one keeps the current pick", async () => {
  useChatRuntimeStore.getState().setModelPreset(GEMMA, CREATIVE.name);
  await switchTo(GEMMA);
  assert.equal(live().activePreset, "Creative");
  assert.equal(live().temperature, 1.3);

  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  assert.equal(live().temperature, 0.2);

  await switchTo(LLAMA);
  assert.equal(live().activePreset, "Research", "no link, so nothing is replaced");

  await switchTo(GEMMA);
  assert.equal(live().activePreset, "Creative");
  assert.equal(live().systemPrompt, "Be imaginative.");
  useChatRuntimeStore.getState().setModelPreset(GEMMA, null);
});

test("an unlinked model under Default still follows its recommendation", async () => {
  await switchTo(LLAMA);
  assert.deepEqual(live(), {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    temperature: RECOMMENDED.temperature,
    systemPrompt: "",
  });
});

test("a temporary override neither moves the link nor survives the next switch", async () => {
  await switchTo(QWEN);
  pickPreset(CREATIVE);
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.9 });
  loadOrPoll(QWEN);
  await Promise.resolve();
  assert.equal(live().temperature, 0.9, "a poll of the same model replaces nothing");
  assert.equal(useChatRuntimeStore.getState().modelPresets[QWEN], "Research");
  const research = useChatRuntimeStore
    .getState()
    .customPresets.find((saved) => saved.name === "Research");
  assert.equal(research?.params.temperature, 0.2, "the saved preset is untouched");

  await switchTo(LLAMA);
  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  assert.equal(live().temperature, 0.2);
});

test("a link to a preset that no longer exists is ignored", async () => {
  useChatRuntimeStore.getState().setModelPreset(GEMMA, "Deleted");
  await switchTo(GEMMA);
  assert.equal(live().activePreset, "Default");
  assert.equal(live().temperature, RECOMMENDED.temperature);
  useChatRuntimeStore.getState().setModelPreset(GEMMA, null);
});

test("linking and unlinking are saved per model", async () => {
  settingsHttp.puts = [];
  const store = useChatRuntimeStore.getState();
  store.setModelPreset(GEMMA, CREATIVE.name);
  store.setModelPreset(LLAMA, RESEARCH.name);
  await settled();
  assert.deepEqual(stored().modelPresets, {
    [QWEN]: RESEARCH.name,
    [LLAMA]: RESEARCH.name,
    [GEMMA]: CREATIVE.name,
  });
  store.setModelPreset(GEMMA, null);
  store.setModelPreset(LLAMA, null);
  await settled();
  const lastPut = settingsHttp.puts.at(-1) as { modelPresets?: unknown };
  assert.deepEqual(lastPut.modelPresets, { [GEMMA]: null, [LLAMA]: null });
  assert.deepEqual(useChatRuntimeStore.getState().modelPresets, {
    [QWEN]: RESEARCH.name,
  });
});

test("only a saved custom preset can be linked", () => {
  const customPresets = [CREATIVE];
  assert.equal(getLinkedPreset(GEMMA, { [GEMMA]: "Default" }, customPresets), null);
  assert.equal(getLinkedPreset(GEMMA, { [GEMMA]: "Missing" }, customPresets), null);
  assert.equal(getLinkedPreset("", { "": "Creative" }, customPresets), null);
  assert.equal(
    getLinkedPreset(GEMMA, { [GEMMA]: "Creative" }, customPresets)?.name,
    "Creative",
  );
});
