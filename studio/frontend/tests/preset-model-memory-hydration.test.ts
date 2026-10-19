// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Startup order for #14068: the resident model's defaults can land before the settings response
// restores a custom preset, so that model loaded under the placeholder Default. Hydration has to
// treat it as loaded under the preset, or its first edit files the preset as the model's tuning.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { mergeBackendRecommendedInference } = await import(
  "../src/features/chat/presets/preset-policy.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF";
const PIRATE = {
  temperature: 0.3,
  topP: 0.7,
  topK: 10,
  systemPrompt: "Talk like a pirate.",
};

test("a model resident before hydration restores a custom preset as a preset load", async () => {
  settingsHttp.settings = {
    rememberParamsPerModel: true,
    inferenceParams: PIRATE,
    customPresets: [{ name: "Pirate", params: { ...DEFAULT_INFERENCE_PARAMS, ...PIRATE } }],
    activePreset: "Pirate",
    activePresetSource: "custom",
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const store = useChatRuntimeStore.getState();
  store.setParams(
    mergeBackendRecommendedInference({
      current: store.params,
      response: { inference: { temperature: 1, top_p: 0.9, top_k: 64 }, is_gguf: true, context_length: 8192 },
      modelId: QWEN,
      presetSource: store.activePresetSource,
      loadedContextLength: 8192,
    }),
    { fromModelDefaults: true, maxTokensCap: 8192 },
  );
  settingsHttp.release?.();
  await hydrating;

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.activePresetSource, "custom");
  hydrated.setActivePresetSource("modified");
  hydrated.setParams({ ...useChatRuntimeStore.getState().params, topK: 5 });
  const entry = useChatRuntimeStore.getState().paramsByModel[QWEN] ?? {};
  assert.equal(entry.topK, 5);
  assert.equal(entry.systemPrompt, undefined, "the preset's prompt is not filed as the model's");
  assert.equal(entry.temperature, undefined);
});
