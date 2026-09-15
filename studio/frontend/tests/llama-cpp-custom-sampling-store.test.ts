// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
store.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);
const { markSamplingFields } = await import(
  "../src/features/model-picker/model-config/llama-cpp-config.ts"
);
const custom = {
  version: 1,
  mode: "custom",
  ini: "[*]\ntemp=0\n",
  section: null,
} as const;

function reset() {
  useChatRuntimeStore.getState().applyThreadScopedSettings(null, null);
  useChatRuntimeStore.setState({
    params: { ...DEFAULT_INFERENCE_PARAMS, checkpoint: "org/model" },
    llamaCppConfig: custom,
    loadedLlamaCppConfig: custom,
    activeThreadId: null,
    settingsHydrated: false,
    rememberParamsPerModel: false,
    paramsByModel: {},
  });
}

test("a custom same-model recommendation refresh preserves explicitly edited zero", () => {
  reset();
  let live = useChatRuntimeStore.getState();
  live.setParams({ ...live.params, temperature: 0 });
  live = useChatRuntimeStore.getState();
  assert.deepEqual(live.params.samplingFieldsExplicit, ["temperature"]);
  live.setParams(
    { ...live.params, temperature: 0.8, topP: 0.8 },
    { fromModelDefaults: true },
  );
  live = useChatRuntimeStore.getState();
  assert.equal(live.params.temperature, 0);
  assert.equal(live.params.topP, 0.8);
  assert.deepEqual(live.params.samplingFieldsExplicit, ["temperature"]);
});

test("an edit equal to the visible default remains explicit across default seeding", () => {
  reset();
  const live = useChatRuntimeStore.getState();
  live.setParams(markSamplingFields(live.params, "temperature"));
  live.setParams(
    { ...live.params, temperature: 0.8 },
    { fromModelDefaults: true },
  );
  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    DEFAULT_INFERENCE_PARAMS.temperature,
  );
  assert.deepEqual(
    useChatRuntimeStore.getState().params.samplingFieldsExplicit,
    ["temperature"],
  );
});

test("changing model resets automatic provenance and replays only the destination's memory", () => {
  reset();
  let live = useChatRuntimeStore.getState();
  live.setParams({ ...live.params, temperature: 0 });
  useChatRuntimeStore.setState({
    rememberParamsPerModel: true,
    paramsByModel: { "org/other": { topP: 0.25 } },
  });
  live = useChatRuntimeStore.getState();
  live.setParams(
    { ...DEFAULT_INFERENCE_PARAMS, checkpoint: "org/other", temperature: 0.8 },
    { fromModelDefaults: true },
  );
  live = useChatRuntimeStore.getState();
  assert.equal(live.params.temperature, 0.8);
  assert.equal(live.params.topP, 0.25);
  assert.deepEqual(live.params.samplingFieldsExplicit, ["top_p"]);
});

test("a legacy thread snapshot keeps explicit zero and false while new automatic snapshots stay empty", () => {
  reset();
  useChatRuntimeStore.getState().applyThreadScopedSettings("legacy", {
    temperature: 0,
    reasoningEnabled: false,
  });
  let live = useChatRuntimeStore.getState();
  assert.equal(live.params.temperature, 0);
  assert.equal(live.reasoningEnabled, false);
  assert.deepEqual(live.params.samplingFieldsExplicit, [
    "temperature",
    "enable_thinking",
  ]);
  live.applyThreadScopedSettings("automatic", {
    temperature: 0.6,
    samplingFieldsExplicit: [],
  });
  live = useChatRuntimeStore.getState();
  assert.deepEqual(live.params.samplingFieldsExplicit, []);
});

for (const [name, settings, expectedMask] of [
  ["non-sampling legacy row", { toolsEnabled: true }, ["temperature", "top_p"]],
  ["partial legacy row", { temperature: 0.2 }, ["temperature", "top_p"]],
  ["partial automatic row", { temperature: 0.2, samplingFieldsExplicit: [] }, ["top_p"]],
] as const) {
  test(`thread restore inherits global provenance for a ${name}`, () => {
    reset();
    const live = useChatRuntimeStore.getState();
    live.setParams({ ...live.params, temperature: 0.4, topP: 0.7 });
    useChatRuntimeStore.getState().applyThreadScopedSettings(
      "sparse",
      "samplingFieldsExplicit" in settings
        ? { ...settings, samplingFieldsExplicit: [...settings.samplingFieldsExplicit] }
        : settings,
    );
    const restored = useChatRuntimeStore.getState();
    assert.equal(restored.params.topP, 0.7);
    assert.deepEqual(restored.params.samplingFieldsExplicit, [...expectedMask]);
  });
}

test("explicit reasoning and preserve-thinking choices use their direct wire names", () => {
  reset();
  const live = useChatRuntimeStore.getState();
  live.setReasoningEnabled(false);
  live.setPreserveThinking(false);
  assert.deepEqual(
    useChatRuntimeStore.getState().params.samplingFieldsExplicit,
    ["enable_thinking", "preserve_thinking"],
  );
});
