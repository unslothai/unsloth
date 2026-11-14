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
const {
  linkedPresetShown,
  presetSourceForModel,
  suspendLinkedPresetSync,
  useChatRuntimeStore,
} = await import(
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
const { useExternalProvidersStore } = await import(
  "../src/features/chat/stores/external-providers-store.ts"
);
const { REMEMBERED_MAX_TOKENS_MAX } = await import(
  "../src/features/chat/lib/per-model-params.ts"
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
      // The linked model recommends its own temperature, so a value carried from it shows.
      response: {
        inference: modelId === QWEN ? { ...RECOMMENDED, temperature: 0.55 } : RECOMMENDED,
        is_gguf: true,
        context_length: CONTEXT,
      },
      modelId,
      presetSource: presetSourceForModel(modelId),
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

// Runs first: no model has reached this page yet, as on a reload whose status lands after settings.
test("the first model adopted after a reload keeps a restored edit of its linked preset", async () => {
  pickPreset(RESEARCH);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.9 });
  store.setActivePresetSource("modified");
  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  assert.equal(live().temperature, 0.9);
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

test("each linked model gets its own preset, and an unlinked one gets the pick from before", async () => {
  useChatRuntimeStore.getState().setModelPreset(GEMMA, CREATIVE.name);
  await switchTo(GEMMA);
  assert.equal(live().activePreset, "Creative");
  assert.equal(live().temperature, 1.3);

  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  assert.equal(live().temperature, 0.2);

  await switchTo(LLAMA);
  assert.equal(live().activePreset, "Default", "no link, so the pick from before the links");

  await switchTo(GEMMA);
  assert.equal(live().activePreset, "Creative");
  assert.equal(live().systemPrompt, "Be imaginative.");
  useChatRuntimeStore.getState().setModelPreset(GEMMA, null);
});

test("a linked preset does not leak into the next unlinked model", async () => {
  await switchTo(QWEN);
  assert.equal(live().systemPrompt, "Cite your sources.");
  await switchTo(LLAMA);
  assert.deepEqual(
    { ...live(), temperature: undefined },
    { activePreset: "Default", activePresetSource: "builtin-default", temperature: undefined, systemPrompt: "" },
  );
  loadOrPoll(LLAMA);
  await Promise.resolve();
  assert.equal(live().temperature, RECOMMENDED.temperature, "the next status lays the recommendation on");
});

test("leaving a linked model keeps an unlinked model's own edits", async () => {
  await switchTo(LLAMA);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.77 });
  await switchTo(QWEN);
  assert.equal(live().temperature, 0.2);
  store.clearCheckpoint();
  await Promise.resolve();
  await switchTo(LLAMA);
  assert.equal(live().activePreset, "Default");
  assert.equal(live().temperature, 0.77);
});

test("a custom pick from before the link comes back, a pick made on the linked model stays", async () => {
  await switchTo(LLAMA);
  pickPreset(CREATIVE);
  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  await switchTo(GEMMA);
  assert.deepEqual(live(), {
    activePreset: "Creative",
    activePresetSource: "custom",
    temperature: 1.3,
    systemPrompt: "Be imaginative.",
  });

  await switchTo(QWEN);
  pickPreset(DEFAULT);
  pickPreset(CREATIVE);
  await switchTo(LLAMA);
  assert.equal(live().activePreset, "Creative", "picked by hand, so it is not the link's");
});

test("without per-model memory the pre-link values and provenance come back", async () => {
  useChatRuntimeStore.setState({ rememberParamsPerModel: false });
  try {
    await switchTo(LLAMA);
    const store = useChatRuntimeStore.getState();
    store.setParams({ ...store.params, temperature: 0.77, systemPrompt: "Be brief." });
    store.setActivePresetSource("modified");
    await switchTo(QWEN);
    assert.equal(live().activePreset, "Research");
    await switchTo(GEMMA);
    assert.deepEqual(live(), {
      activePreset: "Default",
      activePresetSource: "modified",
      temperature: 0.77,
      systemPrompt: "Be brief.",
    });
  } finally {
    useChatRuntimeStore.setState({ rememberParamsPerModel: true });
  }
});

test("a background load that puts the visible model back is no switch", async () => {
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.9 });
  store.setActivePresetSource("modified");
  const visible = { ...useChatRuntimeStore.getState().params };
  const resume = suspendLinkedPresetSync();
  loadOrPoll(LLAMA);
  await Promise.resolve();
  useChatRuntimeStore.setState({ params: visible });
  resume();
  await Promise.resolve();
  assert.equal(live().activePreset, "Research");
  assert.equal(live().temperature, 0.9, "the visible temporary edit is not replaced");
});

test("a switch the background load ended on still syncs on resume", async () => {
  await switchTo(QWEN);
  const resume = suspendLinkedPresetSync();
  loadOrPoll(LLAMA);
  await Promise.resolve();
  assert.equal(live().activePreset, "Research", "suspended");
  resume();
  await Promise.resolve();
  assert.equal(live().activePreset, "Default");
});

test("leaving a linked model gives an open chat back its own sampling", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(LLAMA);
  store.setActiveThreadId("chat-1");
  store.applyThreadScopedSettings("chat-1", { temperature: 0.77, systemPrompt: "Be brief." });
  assert.equal(live().temperature, 0.77);
  try {
    await switchTo(QWEN);
    assert.equal(live().temperature, 0.2);
    await switchTo(LLAMA);
    assert.equal(live().activePreset, "Default");
    assert.equal(live().temperature, 0.77);
    assert.equal(live().systemPrompt, "Be brief.");
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("another chat opened on the linked model keeps its own sampling on the way out", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(LLAMA);
  store.setActiveThreadId("chat-a");
  store.applyThreadScopedSettings("chat-a", { temperature: 0.77, systemPrompt: "Be brief." });
  try {
    await switchTo(QWEN);
    store.setActiveThreadId("chat-b");
    store.applyThreadScopedSettings("chat-b", { temperature: 0.4, systemPrompt: "Answer in French." });
    await switchTo(LLAMA);
    assert.equal(live().activePreset, "Default");
    assert.equal(live().temperature, 0.4);
    assert.equal(live().systemPrompt, "Answer in French.");
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("a pick deleted while on the linked model does not come back", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  const doomed = preset("Doomed", 1.7, "Speak like a pirate.");
  store.setCustomPresets([...saved, doomed]);
  try {
    await switchTo(LLAMA);
    pickPreset(doomed);
    await switchTo(QWEN);
    store.setCustomPresets(saved);
    await switchTo("unsloth/Unseen-GGUF");
    assert.equal(live().activePreset, "Default");
    assert.notEqual(live().systemPrompt, "Speak like a pirate.");
  } finally {
    store.setCustomPresets(saved);
  }
});

test("a link stays in this tab: the stored pick and installation defaults never take it", async () => {
  await switchTo(LLAMA);
  pickPreset(CREATIVE);
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setActiveThreadId("chat-x");
  store.applyThreadScopedSettings("chat-x", { temperature: 0.4 });
  try {
    await settled();
    assert.equal(live().activePreset, "Research");
    const saved = stored() as { activePreset?: string; inferenceParams?: { systemPrompt?: string } };
    assert.equal(saved.activePreset, "Creative", "a reload here finds the user's own pick");
    assert.notEqual(saved.inferenceParams?.systemPrompt, "Cite your sources.");
    await switchTo(GEMMA);
    await settled();
    assert.equal(live().activePreset, "Creative");
    assert.notEqual(
      (stored() as { inferenceParams?: { systemPrompt?: string } }).inferenceParams?.systemPrompt,
      "Cite your sources.",
    );
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("leaving a link under Default takes the new model's recommendation at once", async () => {
  await switchTo(GEMMA);
  await switchTo(QWEN);
  assert.equal(live().temperature, 0.2);
  await switchTo("unsloth/Adopted-GGUF");
  assert.equal(live().activePreset, "Default");
  assert.equal(live().temperature, RECOMMENDED.temperature, "no later refresh needed");
  assert.equal(live().systemPrompt, "");
});

test("a second chat switched between two linked models gets its own sampling back", async () => {
  const store = useChatRuntimeStore.getState();
  store.setModelPreset(GEMMA, CREATIVE.name);
  try {
    store.setActiveThreadId("chat-a");
    store.applyThreadScopedSettings("chat-a", { temperature: 0.77 });
    await switchTo(QWEN);
    store.setActiveThreadId("chat-b");
    store.applyThreadScopedSettings("chat-b", { temperature: 0.4, systemPrompt: "Answer in French." });
    await switchTo(GEMMA);
    assert.equal(live().activePreset, "Creative");
    await switchTo(LLAMA);
    assert.equal(live().temperature, 0.4);
    assert.equal(live().systemPrompt, "Answer in French.");
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
    store.setModelPreset(GEMMA, null);
  }
});

test("a status replay of the linked model keeps its preset over the model's memory", async () => {
  useChatRuntimeStore.setState({ paramsByModel: { [QWEN]: { temperature: 0.66 } } });
  await switchTo(QWEN);
  loadOrPoll(QWEN);
  await Promise.resolve();
  assert.equal(live().temperature, 0.2);
  assert.equal(live().systemPrompt, "Cite your sources.");
});

test("a status replay keeps the link over an open chat's pins, which come back on leaving", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(LLAMA);
  store.setActiveThreadId("chat-p");
  store.applyThreadScopedSettings("chat-p", { temperature: 0.4 });
  try {
    await switchTo(QWEN);
    loadOrPoll(QWEN);
    await Promise.resolve();
    assert.equal(live().temperature, 0.2);
    assert.equal(live().systemPrompt, "Cite your sources.");
    await switchTo(LLAMA);
    assert.equal(live().temperature, 0.4);
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("the link's overlay leaves the model's own memory in place", async () => {
  useChatRuntimeStore.setState({ paramsByModel: { [QWEN]: { temperature: 0.66 } } });
  await switchTo(QWEN);
  assert.equal(useChatRuntimeStore.getState().paramsByModel[QWEN]?.temperature, 0.66);
});

test("an edit over a link files only itself into the model's memory", async () => {
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, topK: 33 });
  const remembered = useChatRuntimeStore.getState().paramsByModel[QWEN] ?? {};
  assert.equal(remembered.topK, 33);
  assert.notEqual(remembered.systemPrompt, "Cite your sources.");
  assert.notEqual(remembered.temperature, 0.2);
});

test("a reset over a link survives the next status replay", async () => {
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.7, systemPrompt: "" });
  loadOrPoll(QWEN);
  await Promise.resolve();
  assert.equal(live().temperature, 0.7);
  assert.equal(live().systemPrompt, "");
});

test("an edit over a link keeps the link's untouched values through a replay", async () => {
  useChatRuntimeStore.setState({
    paramsByModel: { [QWEN]: { topP: 0.5, systemPrompt: "Old prompt." } },
  });
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.9 });
  store.setActivePresetSource("modified");
  loadOrPoll(QWEN);
  await Promise.resolve();
  assert.equal(live().temperature, 0.9);
  assert.equal(useChatRuntimeStore.getState().params.topP, 0.8);
  assert.equal(live().systemPrompt, "Cite your sources.");
});

test("an edit in a chat over a link saves only the edit to the chat", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(LLAMA);
  store.setActiveThreadId("chat-e");
  store.applyThreadScopedSettings("chat-e", { temperature: 0.4, systemPrompt: "Mine." });
  try {
    await switchTo(QWEN);
    const s = useChatRuntimeStore.getState();
    s.setParams({ ...s.params, topK: 33 });
    await settled();
    await switchTo(LLAMA);
    assert.equal(live().temperature, 0.4);
    assert.equal(live().systemPrompt, "Mine.");
    assert.equal(useChatRuntimeStore.getState().params.topK, 33);
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("a model reached from a link under an edited Default remembers what it showed", async () => {
  const B = "unsloth/Bee-GGUF";
  const C = "unsloth/Cee-GGUF";
  await switchTo(LLAMA);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.77 });
  store.setActivePresetSource("modified");
  await switchTo(QWEN);
  await switchTo(B);
  assert.equal(live().temperature, 0.77);
  await switchTo(C);
  const s = useChatRuntimeStore.getState();
  s.setParams({ ...s.params, temperature: 0.9 });
  await switchTo(B);
  assert.equal(live().temperature, 0.77);
});

test("unlinking the model on screen puts the earlier pick back for good", async () => {
  useChatRuntimeStore.setState({ paramsByModel: { [QWEN]: { temperature: 0.66 } } });
  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  useChatRuntimeStore.getState().setModelPreset(QWEN, null);
  await Promise.resolve();
  assert.equal(live().activePreset, "Default");
  loadOrPoll(QWEN);
  await Promise.resolve();
  assert.equal(live().activePreset, "Default");
  assert.notEqual(live().systemPrompt, "Cite your sources.");
  useChatRuntimeStore.getState().setModelPreset(QWEN, RESEARCH.name);
});

test("a switch that merges no recommendation gets the earlier values back", async () => {
  await switchTo(GEMMA);
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, checkpoint: "unsloth/Compared-GGUF" });
  await Promise.resolve();
  assert.equal(live().activePreset, "Default");
  assert.equal(live().temperature, 0.55, "not the link's 0.2");
});

test("a chat's cleared seed survives an edit over a link", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(LLAMA);
  store.setParams({ ...store.params, seed: 42 });
  store.setActiveThreadId("chat-s");
  store.applyThreadScopedSettings("chat-s", { seed: null });
  try {
    await switchTo(QWEN);
    const s = useChatRuntimeStore.getState();
    s.setParams({ ...s.params, topK: 21 });
    await settled();
    await switchTo(LLAMA);
    assert.equal(useChatRuntimeStore.getState().params.seed, null);
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("a linked preset's token budget stays within the loaded window", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  const long = { name: "Long", params: { ...DEFAULT_INFERENCE_PARAMS, maxTokens: 131072 } };
  store.setCustomPresets([...saved, long]);
  store.setModelPreset(GEMMA, "Long");
  try {
    await switchTo(GEMMA);
    const poll = () => {
      const s = useChatRuntimeStore.getState();
      s.setParams(
        mergeBackendRecommendedInference({
          current: s.params,
          response: { inference: RECOMMENDED, is_gguf: true, context_length: 8192 },
          modelId: GEMMA,
          presetSource: presetSourceForModel(GEMMA),
          loadedContextLength: 8192,
        }),
        { fromModelDefaults: true, maxTokensCap: 8192 },
      );
    };
    poll();
    poll();
    assert.ok(useChatRuntimeStore.getState().params.maxTokens <= 8192);
  } finally {
    store.setModelPreset(GEMMA, null);
    store.setCustomPresets(saved);
  }
});

test("an edit just before switching away keeps the chat's own values", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(LLAMA);
  store.setActiveThreadId("chat-q");
  store.applyThreadScopedSettings("chat-q", { temperature: 0.4, systemPrompt: "Mine." });
  try {
    await switchTo(QWEN);
    const s = useChatRuntimeStore.getState();
    s.setParams({ ...s.params, topK: 19 });
    s.setParams({ ...useChatRuntimeStore.getState().params, checkpoint: "external::p::m" });
    await Promise.resolve();
    await settled();
    await switchTo(LLAMA);
    assert.equal(live().temperature, 0.4);
    assert.equal(live().systemPrompt, "Mine.");
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("saving the linked preset after an edit keeps the saved values over the model's memory", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  useChatRuntimeStore.setState({
    paramsByModel: { [QWEN]: { topP: 0.5, systemPrompt: "Old prompt." } },
  });
  try {
    await switchTo(QWEN);
    let s = useChatRuntimeStore.getState();
    s.setParams({ ...s.params, temperature: 0.25 });
    s.setActivePresetSource("modified");
    s = useChatRuntimeStore.getState();
    s.setCustomPresets(
      saved.map((p) => (p.name === "Research" ? { name: "Research", params: { ...s.params } } : p)),
    );
    s.setActivePreset("Research");
    s.setActivePresetSource("custom");
    loadOrPoll(QWEN);
    await Promise.resolve();
    assert.equal(live().temperature, 0.25);
    assert.equal(useChatRuntimeStore.getState().params.topP, 0.8);
    assert.equal(live().systemPrompt, "Cite your sources.");
  } finally {
    store.setCustomPresets(saved);
  }
});

test("leaving a link replays the destination's window, not the earlier budget", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  const short = { name: "Short", params: { ...DEFAULT_INFERENCE_PARAMS, maxTokens: 512 } };
  store.setCustomPresets([...saved, short]);
  try {
    await switchTo(LLAMA);
    pickPreset(short);
    await switchTo(QWEN);
    useChatRuntimeStore.setState({
      paramsByModel: {
        ...useChatRuntimeStore.getState().paramsByModel,
        [GEMMA]: { maxTokens: REMEMBERED_MAX_TOKENS_MAX },
      },
    });
    await switchTo(GEMMA);
    assert.equal(live().activePreset, "Short");
    assert.equal(useChatRuntimeStore.getState().params.maxTokens, CONTEXT);
  } finally {
    store.setCustomPresets(saved);
  }
});

test("a chat edit just before switching onto a link is saved as typed", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  store.setCustomPresets(
    saved.map((p) => (p.name === "Research" ? { ...p, params: { ...p.params, topK: 20 } } : p)),
  );
  await switchTo(LLAMA);
  store.setActiveThreadId("chat-k");
  store.applyThreadScopedSettings("chat-k", { temperature: 0.4 });
  try {
    const s = useChatRuntimeStore.getState();
    s.setParams({ ...s.params, topK: 19 });
    await switchTo(QWEN);
    assert.equal(useChatRuntimeStore.getState().params.topK, 20);
    await settled();
    await switchTo(LLAMA);
    assert.equal(useChatRuntimeStore.getState().params.topK, 19);
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
    store.setCustomPresets(saved);
  }
});

test("saving the linked view under a new name keeps it over the model's memory", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  useChatRuntimeStore.setState({ paramsByModel: { [QWEN]: { temperature: 0.66 } } });
  try {
    await switchTo(QWEN);
    const s = useChatRuntimeStore.getState();
    s.setCustomPresets([...saved, { name: "Copy", params: { ...s.params } }]);
    s.setActivePreset("Copy");
    s.setActivePresetSource("custom");
    loadOrPoll(QWEN);
    await Promise.resolve();
    assert.equal(live().activePreset, "Copy");
    assert.equal(live().temperature, 0.2);
  } finally {
    store.setCustomPresets(saved);
  }
});

test("a linked external model's budget stays within its output cap", async () => {
  const EXT = "external::capped::gpt-x";
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  useExternalProvidersStore.setState({
    providers: [{
      id: "capped", providerType: "openai_compatible", name: "Capped",
      baseUrl: "http://localhost:1", models: ["gpt-x"], maxOutputTokens: 4096,
      createdAt: 0, updatedAt: 0,
    }],
  });
  const huge = { name: "Huge", params: { ...DEFAULT_INFERENCE_PARAMS, maxTokens: 100000 } };
  store.setCustomPresets([...saved, huge]);
  store.setModelPreset(EXT, "Huge");
  try {
    const s = useChatRuntimeStore.getState();
    s.setParams({ ...s.params, checkpoint: EXT });
    await Promise.resolve();
    assert.equal(live().activePreset, "Huge");
    assert.ok(useChatRuntimeStore.getState().params.maxTokens <= 4096);
  } finally {
    store.setModelPreset(EXT, null);
    store.setCustomPresets(saved);
    useExternalProvidersStore.setState({ providers: [] });
    await settled();
  }
});

test("a reload with a smaller window keeps the capped linked budget", async () => {
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  const big = { name: "Big", params: { ...DEFAULT_INFERENCE_PARAMS, maxTokens: 100000 } };
  store.setCustomPresets([...saved, big]);
  store.setModelPreset(GEMMA, "Big");
  useChatRuntimeStore.setState({ paramsByModel: { [GEMMA]: { maxTokens: 1024 } } });
  const poll = (window: number) => {
    const s = useChatRuntimeStore.getState();
    s.setParams(
      mergeBackendRecommendedInference({
        current: s.params,
        response: { inference: RECOMMENDED, is_gguf: true, context_length: window },
        modelId: GEMMA,
        presetSource: presetSourceForModel(GEMMA),
        loadedContextLength: window,
      }),
      { fromModelDefaults: true, maxTokensCap: window },
    );
  };
  try {
    await switchTo(GEMMA);
    poll(8192);
    assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
    poll(4096);
    poll(4096);
    assert.equal(useChatRuntimeStore.getState().params.maxTokens, 4096);
  } finally {
    store.setModelPreset(GEMMA, null);
    store.setCustomPresets(saved);
    await settled();
  }
});

test("a chat's settings landing after the link go under it", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(QWEN);
  store.setActiveThreadId("chat-r");
  store.applyThreadScopedSettings("chat-r", { temperature: 0.4, systemPrompt: "Mine." });
  try {
    assert.equal(live().temperature, 0.2);
    assert.equal(live().systemPrompt, "Cite your sources.");
    loadOrPoll(QWEN);
    await Promise.resolve();
    assert.equal(live().temperature, 0.2);
    await switchTo(LLAMA);
    assert.equal(live().temperature, 0.4);
    assert.equal(live().systemPrompt, "Mine.");
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("opening another chat on a linked model keeps a Max Tokens edit", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(QWEN);
  const s = useChatRuntimeStore.getState();
  s.setParams({ ...s.params, maxTokens: 4096 });
  await settled();
  store.setActiveThreadId("chat-m");
  store.applyThreadScopedSettings("chat-m", { temperature: 0.4 });
  try {
    assert.equal(useChatRuntimeStore.getState().params.maxTokens, 4096);
    assert.equal(live().temperature, 0.2);
    loadOrPoll(QWEN);
    await Promise.resolve();
    assert.equal(useChatRuntimeStore.getState().params.maxTokens, 4096);
  } finally {
    store.setActiveThreadId(null);
    store.applyThreadScopedSettings(null, null);
  }
});

test("an edit before a new chat's first send survives its pairing", async () => {
  const store = useChatRuntimeStore.getState();
  await switchTo(QWEN);
  const s = useChatRuntimeStore.getState();
  s.setParams({ ...s.params, temperature: 0.77 });
  await settled();
  store.applyThreadScopedSettings(null, null);
  assert.equal(live().temperature, 0.77);
});

test("leaving a link under Default for an external model puts the earlier values back", async () => {
  await switchTo(GEMMA);
  await switchTo(QWEN);
  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, checkpoint: "external::prov-1::gpt-x" });
  await Promise.resolve();
  assert.equal(live().activePreset, "Default");
  assert.equal(live().temperature, 0.55, "the values from before the link, not the link's 0.2");
  assert.equal(live().systemPrompt, "");
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

test("an edit over a link stays out of the stored pick and installation defaults", async () => {
  await switchTo(QWEN);
  // chat-settings-sheet's set(): the source, then the value.
  const store = useChatRuntimeStore.getState();
  store.setActivePresetSource("modified");
  store.setParams({ ...store.params, temperature: 0.9 });
  await settled();
  assert.equal(live().temperature, 0.9);
  const saved = stored() as {
    activePreset?: string;
    activePresetSource?: string;
    inferenceParams?: { temperature?: number };
    inferenceParamsByModel?: Record<string, { temperature?: number }>;
  };
  assert.equal(saved.activePreset, "Default");
  assert.equal(saved.activePresetSource, "builtin-default", "a reload here finds the pick as it was");
  assert.notEqual(saved.inferenceParams?.temperature, 0.9, "other models do not inherit the edit");
  assert.equal(saved.inferenceParamsByModel?.[QWEN]?.temperature, 0.9, "the model keeps its edit");
});

test("deleting the linked preset while its link shows puts the earlier pick back", async () => {
  await switchTo(LLAMA);
  pickPreset(CREATIVE);
  await switchTo(QWEN);
  assert.equal(live().activePreset, "Research");
  const store = useChatRuntimeStore.getState();
  const saved = store.customPresets;
  try {
    // chat-settings-sheet's deletePreset.
    const shownByLink = linkedPresetShown();
    assert.equal(shownByLink, true);
    store.setCustomPresets(saved.filter((kept) => kept.name !== RESEARCH.name));
    store.setModelPreset(QWEN, null);
    await Promise.resolve();
    assert.equal(linkedPresetShown(), false);
    assert.equal(live().activePreset, "Creative");
    assert.equal(live().systemPrompt, "Be imaginative.");
    await settled();
    assert.equal((stored() as { activePreset?: string }).activePreset, "Creative");
  } finally {
    store.setCustomPresets(saved);
    useChatRuntimeStore.getState().setModelPreset(QWEN, RESEARCH.name);
    await settled();
  }
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
