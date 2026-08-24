// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

const QWEN38 = "unsloth/Qwen3.8-27B-GGUF";
const LEGACY_SNAPSHOT = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1.0,
  presencePenalty: 0.0,
  maxTokens: 8192,
  systemPrompt: "",
  systemVariables: "",
  fastMode: false,
};

test("hydration replaces and persists the stale Qwen3.8 default snapshot", async () => {
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      checkpoint: QWEN38,
      minP: 0,
      presencePenalty: 1.5,
    },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.params.minP, 0);
  assert.equal(hydrated.params.presencePenalty, 1.5);
  assert.equal(hydrated.paramsByModel[QWEN38]?.minP, 0);
  assert.equal(hydrated.paramsByModel[QWEN38]?.presencePenalty, 1.5);

  await new Promise((resolve) => setTimeout(resolve, 600));
  const migrationPut = settingsHttp.puts.find(
    (put) =>
      (put.inferenceParamsByModel as Record<string, unknown> | undefined)?.[
        QWEN38
      ] !== undefined,
  );
  assert.deepEqual(migrationPut, {
    inferenceParamsByModel: {
      [QWEN38]: { minP: 0, presencePenalty: 1.5 },
    },
  });
});

test("hydration upgrades a global-only non-thinking installation", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: false,
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
      maxTokens: 8192,
    },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.params.temperature, 0.7);
  assert.equal(hydrated.params.topP, 0.8);
  assert.equal(hydrated.params.minP, 0);
  assert.equal(hydrated.params.presencePenalty, 1.5);
  assert.deepEqual(
    settingsHttp.puts.find((put) => put.inferenceParams !== undefined),
    {
      inferenceParams: {
        temperature: 0.7,
        topP: 0.8,
        minP: 0,
        presencePenalty: 1.5,
      },
    },
  );
});

test("a confirming read preserves a newer edit from another tab", async () => {
  const newerSnapshot = {
    ...LEGACY_SNAPSHOT,
    presencePenalty: 0.4,
  };
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  const newerSettings = {
    ...legacySettings,
    inferenceParamsByModel: { [QWEN38]: newerSnapshot },
  };
  settingsHttp.settings = newerSettings;
  settingsHttp.getResponses = [legacySettings, newerSettings];
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[QWEN38]?.presencePenalty,
    0.4,
  );
  assert.equal(
    settingsHttp.puts.some(
      (put) =>
        (put.inferenceParamsByModel as Record<string, unknown> | undefined)?.[
          QWEN38
        ] !== undefined,
    ),
    false,
  );
});

test("returning from a custom preset retries the guarded migration", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      checkpoint: QWEN38,
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
    },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Creative",
    activePresetSource: "custom",
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setActivePreset("Default");
  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  await new Promise((resolve) => setTimeout(resolve, 50));

  const state = useChatRuntimeStore.getState();
  assert.equal(state.params.minP, 0);
  assert.equal(state.params.presencePenalty, 1.5);
  assert.equal(state.paramsByModel[QWEN38]?.presencePenalty, 1.5);
  assert.equal(
    settingsHttp.puts.some(
      (put) =>
        (put.inferenceParamsByModel as Record<string, unknown> | undefined)?.[
          QWEN38
        ] !== undefined,
    ),
    true,
  );
});
