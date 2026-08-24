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
const {
  flushPendingChatSettings,
  noteLoadedModelReasoningMode,
  useChatRuntimeStore,
} = await import(
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

test("active-model adoption retries a migration deferred during hydration", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "" },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[QWEN38]?.presencePenalty,
    0,
  );

  useChatRuntimeStore.getState().setCheckpoint(QWEN38);
  const active = useChatRuntimeStore.getState();
  active.setParams(
    {
      ...active.params,
      checkpoint: QWEN38,
      minP: 0,
      presencePenalty: 1.5,
    },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  const migrated = useChatRuntimeStore.getState();
  assert.equal(migrated.paramsByModel[QWEN38]?.minP, 0);
  assert.equal(migrated.paramsByModel[QWEN38]?.presencePenalty, 1.5);
  assert.equal(
    settingsHttp.puts.some((put) => put.inferenceParams !== undefined),
    false,
  );
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

test("atomic migration persistence rejects an edit after confirmation", async () => {
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  const newerSettings = {
    ...legacySettings,
    inferenceParamsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT, presencePenalty: 0.4 },
    },
  };
  settingsHttp.settings = legacySettings;
  settingsHttp.getResponses = [legacySettings, legacySettings];
  settingsHttp.puts.length = 0;
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = newerSettings;
  };
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
  assert.equal(settingsHttp.puts.length, 0);
});

test("a confirming read uses the latest reasoning mode", async () => {
  const thinkingSettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: true,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  const nonThinkingSettings = {
    ...thinkingSettings,
    reasoningEnabled: false,
  };
  settingsHttp.settings = nonThinkingSettings;
  settingsHttp.getResponses = [thinkingSettings, nonThinkingSettings];
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.reasoningEnabled, false);
  assert.equal(hydrated.params.temperature, 0.7);
  assert.equal(hydrated.params.topP, 0.8);
  assert.deepEqual(
    settingsHttp.puts.find(
      (put) =>
        (put.inferenceParamsByModel as Record<string, unknown> | undefined)?.[
          QWEN38
        ] !== undefined,
    ),
    {
      inferenceParamsByModel: {
        [QWEN38]: {
          temperature: 0.7,
          topP: 0.8,
          minP: 0,
          presencePenalty: 1.5,
        },
      },
    },
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

test("restoring the final modified field retries after the parameter edit", async () => {
  const modifiedSnapshot = {
    ...LEGACY_SNAPSHOT,
    presencePenalty: 0.4,
  };
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      ...modifiedSnapshot,
      checkpoint: QWEN38,
    },
    paramsByModel: { [QWEN38]: modifiedSnapshot },
    activePreset: "Default",
    activePresetSource: "modified",
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  useChatRuntimeStore.getState().setParams({
    ...useChatRuntimeStore.getState().params,
    presencePenalty: 0,
  });
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

test("a model switch during the confirming read leaves the former row untouched", async () => {
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  let releaseConfirmation!: (value: Record<string, unknown>) => void;
  const confirmation = new Promise<Record<string, unknown>>((resolve) => {
    releaseConfirmation = resolve;
  });
  settingsHttp.settings = legacySettings;
  settingsHttp.getResponses = [legacySettings, confirmation];
  settingsHttp.gets = 0;
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();
  while (settingsHttp.gets < 2) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "unsloth/Qwen3.6-9B-GGUF" },
    reasoningEnabled: false,
  }));
  releaseConfirmation(legacySettings);
  await hydration;

  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[QWEN38]?.presencePenalty,
    0,
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

test("resident-model adoption migrates a deferred global-only snapshot", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const state = useChatRuntimeStore.getState();
  state.setParams(
    { ...state.params, minP: 0, presencePenalty: 1.5 },
    {
      fromModelDefaults: true,
      migrateOwnedGlobalQwenDefaults: true,
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  const globalPut = settingsHttp.puts.find(
    (put) => put.inferenceParams !== undefined,
  );
  assert.deepEqual(globalPut?.inferenceParams, {
    minP: 0,
    presencePenalty: 1.5,
  });
});

test("resident-model adoption does not claim a global beside model memory", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const state = useChatRuntimeStore.getState();
  state.setParams(
    { ...state.params, minP: 0, presencePenalty: 1.5 },
    {
      fromModelDefaults: true,
      migrateOwnedGlobalQwenDefaults: true,
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  const persisted = settingsHttp.settings as {
    inferenceParams?: { minP?: number; presencePenalty?: number };
    inferenceParamsByModel?: Record<
      string,
      { presencePenalty?: number }
    >;
  };
  assert.equal(persisted.inferenceParams?.minP, 0.01);
  assert.equal(persisted.inferenceParams?.presencePenalty, 0);
  assert.equal(
    persisted.inferenceParamsByModel?.[QWEN38]?.presencePenalty,
    1.5,
  );
});

test("a retry cannot overwrite an edit made after its confirming read", async () => {
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = legacySettings;
  settingsHttp.puts.length = 0;
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = {
      ...legacySettings,
      inferenceParamsByModel: {
        [QWEN38]: { ...LEGACY_SNAPSHOT, presencePenalty: 0.4 },
      },
    };
  };
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(
    (
      settingsHttp.settings.inferenceParamsByModel as Record<
        string,
        Record<string, unknown>
      >
    )[QWEN38].presencePenalty,
    0.4,
  );
});

test("a retry revalidates the checkpoint after its confirming read", async () => {
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  let releaseConfirmation!: (value: Record<string, unknown>) => void;
  const confirmation = new Promise<Record<string, unknown>>((resolve) => {
    releaseConfirmation = resolve;
  });
  settingsHttp.settings = legacySettings;
  settingsHttp.getResponses = [confirmation];
  settingsHttp.gets = 0;
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  while (settingsHttp.gets < 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      checkpoint: "unsloth/Qwen3.6-9B-GGUF",
    },
    reasoningEnabled: false,
  }));
  releaseConfirmation(legacySettings);
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(
    (
      settingsHttp.settings.inferenceParamsByModel as Record<
        string,
        Record<string, unknown>
      >
    )[QWEN38].presencePenalty,
    0,
  );
});

test("a retry fences reasoning added after a confirming read", async () => {
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = legacySettings;
  settingsHttp.puts.length = 0;
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = { ...legacySettings, reasoningEnabled: false };
  };
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const state = useChatRuntimeStore.getState();
  state.setParams(
    { ...state.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(settingsHttp.settings.reasoningEnabled, false);
  assert.equal(
    (
      settingsHttp.settings.inferenceParamsByModel as Record<
        string,
        Record<string, unknown>
      >
    )[QWEN38].presencePenalty,
    0,
  );
});

test("deferred global ownership cannot follow a later checkpoint", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const state = useChatRuntimeStore.getState();
  state.setParams(
    { ...state.params, minP: 0, presencePenalty: 1.5 },
    {
      fromModelDefaults: true,
      migrateOwnedGlobalQwenDefaults: true,
    },
  );
  useChatRuntimeStore.setState((current) => ({
    params: {
      ...current.params,
      checkpoint: "unsloth/Qwen3.6-9B-GGUF",
    },
    reasoningEnabled: false,
  }));
  await new Promise((resolve) => setTimeout(resolve, 50));

  const persisted = settingsHttp.settings as {
    inferenceParams?: { presencePenalty?: number; minP?: number };
  };
  assert.equal(persisted.inferenceParams?.presencePenalty, 0);
  assert.equal(persisted.inferenceParams?.minP, 0.01);
});

test("local migration preserves an active thread's sampling override", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));
  const initial = useChatRuntimeStore.getState();
  initial.setActiveThreadId("thread-with-sampling");
  initial.applyThreadScopedSettings("thread-with-sampling", {
    presencePenalty: 0,
    reasoningEnabled: false,
  });

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  const migrated = useChatRuntimeStore.getState();
  assert.equal(migrated.params.presencePenalty, 0);
  assert.equal(migrated.paramsByModel[QWEN38]?.presencePenalty, 1.5);
  assert.equal(migrated.paramsByModel[QWEN38]?.temperature, 0.6);
  assert.equal(migrated.paramsByModel[QWEN38]?.topP, 0.95);
  migrated.applyThreadScopedSettings(null, {});
  assert.equal(useChatRuntimeStore.getState().params.minP, 0);
  assert.equal(useChatRuntimeStore.getState().params.presencePenalty, 1.5);
  migrated.setActiveThreadId(null);
});

test("hydration migrates the authoritative global when model memory is off", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: false,
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.rememberParamsPerModel, false);
  assert.equal(hydrated.params.minP, 0);
  assert.equal(hydrated.params.presencePenalty, 1.5);
  assert.deepEqual(
    settingsHttp.puts.find(
      (put) =>
        put.inferenceParams !== undefined &&
        put.inferenceParamsByModel !== undefined,
    ),
    {
      inferenceParamsByModel: {
        [QWEN38]: { minP: 0, presencePenalty: 1.5 },
      },
      inferenceParams: { minP: 0, presencePenalty: 1.5 },
    },
  );
});

test("a first user model load during hydration does not claim prior globals", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
  };
  settingsHttp.puts.length = 0;
  settingsHttp.hold();
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "" },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const loading = useChatRuntimeStore.getState();
  loading.setParams(
    {
      ...loading.params,
      checkpoint: QWEN38,
      minP: 0,
      presencePenalty: 1.5,
    },
    { fromModelDefaults: true },
  );
  settingsHttp.release?.();
  await hydration;
  settingsHttp.gate = null;
  settingsHttp.release = null;

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.params.minP, 0);
  assert.equal(hydrated.params.presencePenalty, 1.5);
  assert.equal(
    settingsHttp.puts.some((put) => put.inferenceParams !== undefined),
    false,
  );
});

test("a normalized migration patch stays within the loaded context", async () => {
  const lowerCaseKey = QWEN38.toLowerCase();
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [lowerCaseKey]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      ...LEGACY_SNAPSHOT,
      checkpoint: QWEN38,
      maxTokens: 4096,
    },
    paramsByModel: { [lowerCaseKey]: LEGACY_SNAPSHOT },
    ggufContextLength: 4096,
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  const migrated = useChatRuntimeStore.getState();
  assert.equal(migrated.params.maxTokens, 4096);
  assert.equal(migrated.paramsByModel[QWEN38]?.presencePenalty, 1.5);
});

test("deferred adoption migrates the authoritative global when memory is off", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: false,
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "" },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();
  const adopting = useChatRuntimeStore.getState();
  adopting.setParams(
    {
      ...adopting.params,
      checkpoint: QWEN38,
      minP: 0,
      presencePenalty: 1.5,
    },
    {
      fromModelDefaults: true,
      migrateOwnedGlobalQwenDefaults: true,
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  const globalPatch = settingsHttp.puts.find(
    (put) => put.inferenceParams !== undefined,
  );
  assert.deepEqual(globalPatch?.inferenceParams, {
    minP: 0,
    presencePenalty: 1.5,
  });
});

test("a retry fences optional global fields added after confirmation", async () => {
  await flushPendingChatSettings();
  const legacySettings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0,
      maxTokens: 8192,
    },
  };
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = legacySettings;
  settingsHttp.puts.length = 0;
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = {
      ...legacySettings,
      inferenceParams: { ...legacySettings.inferenceParams, topK: 40 },
    };
  };
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      ...LEGACY_SNAPSHOT,
      checkpoint: QWEN38,
      minP: 0,
      presencePenalty: 1.5,
    },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    {
      fromModelDefaults: true,
      migrateOwnedGlobalQwenDefaults: true,
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(
    (
      settingsHttp.settings.inferenceParams as Record<string, unknown>
    ).topK,
    40,
  );
  assert.equal(
    (
      settingsHttp.settings.inferenceParams as Record<string, unknown>
    ).presencePenalty,
    0,
  );
});

test("migration follows the reasoning mode established by the loaded model", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: true,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    supportsReasoning: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: false,
    settingsHydrated: true,
  }));
  noteLoadedModelReasoningMode(QWEN38, false);

  const active = useChatRuntimeStore.getState();
  active.setParams(
    {
      ...active.params,
      temperature: 0.7,
      topP: 0.8,
      minP: 0,
      presencePenalty: 1.5,
    },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.deepEqual(
    settingsHttp.puts.find(
      (put) => put.inferenceParamsByModel !== undefined,
    )?.inferenceParamsByModel,
    {
      [QWEN38]: {
        temperature: 0.7,
        topP: 0.8,
        minP: 0,
        presencePenalty: 1.5,
      },
    },
  );
  noteLoadedModelReasoningMode(QWEN38, true);
});

test("routine model-default refreshes skip migration reads without a candidate", async () => {
  settingsHttp.getResponses.length = 0;
  settingsHttp.gets = 0;
  settingsHttp.puts.length = 0;
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
  };
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      checkpoint: QWEN38,
      temperature: 0.6,
      topP: 0.95,
      topK: 20,
      minP: 0,
      repetitionPenalty: 1,
      presencePenalty: 1.5,
    },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    settingsHydrated: true,
  }));

  const qwen = useChatRuntimeStore.getState();
  qwen.setParams({ ...qwen.params }, { fromModelDefaults: true });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(settingsHttp.gets, 0);

  const llamaCheckpoint = "unsloth/Llama-3.2-3B-Instruct-GGUF";
  const current = useChatRuntimeStore.getState();
  current.setParams(
    { ...current.params, checkpoint: llamaCheckpoint },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(settingsHttp.gets, 0);
});

test("a retained settings write stays ahead of the conditional migration", async () => {
  await flushPendingChatSettings();
  settingsHttp.getResponses.length = 0;
  settingsHttp.gets = 0;
  settingsHttp.puts.length = 0;
  settingsHttp.putFailures = [{ status: 503 }];
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "custom",
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  await new Promise((resolve) => setTimeout(resolve, 50));

  // The failed PUT was requeued, so the confirming GET/CAS must not overtake it.
  assert.equal(settingsHttp.gets, 0);
  assert.equal(
    settingsHttp.puts.some(
      (put) =>
        (put.inferenceParamsByModel as Record<string, unknown> | undefined)?.[
          QWEN38
        ] !== undefined,
    ),
    false,
  );

  // A later successful settings flush drains the predecessor and resumes the
  // deferred, still-guarded migration in order.
  useChatRuntimeStore.getState().setActivePreset("Default");
  assert.equal(await flushPendingChatSettings(), true);
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(settingsHttp.gets, 1);
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
