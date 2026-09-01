// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The migration must never leave this tab showing sampling the server does not
// have. Each test here failed before the fix it guards.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { noteLoadedModelReasoningMode, useChatRuntimeStore } = await import(
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

function resetHttp(settings: Record<string, unknown>): void {
  settingsHttp.settings = settings;
  settingsHttp.getResponses.length = 0;
  settingsHttp.puts.length = 0;
  settingsHttp.beforeConditionalApply = null;
  settingsHttp.conditionalStatus = 200;
  settingsHttp.gate = null;
  settingsHttp.release = null;
}

function seedActiveQwen(): void {
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
}

function serverRow(key = QWEN38): Record<string, unknown> {
  return (
    settingsHttp.settings.inferenceParamsByModel as Record<
      string,
      Record<string, unknown>
    >
  )[key];
}

const LEGACY_SETTINGS = {
  activePreset: "Default",
  activePresetSource: "builtin-default",
  inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
};

test("a rejected retry leaves local sampling on the value the server kept", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  // Another tab commits presencePenalty=0.4 between the confirming GET and the
  // conditional write, so the compare-and-set is rejected.
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = {
      ...LEGACY_SETTINGS,
      inferenceParamsByModel: {
        [QWEN38]: { ...LEGACY_SNAPSHOT, presencePenalty: 0.4 },
      },
    };
  };
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(serverRow().presencePenalty, 0.4);
  // The local store must not advertise the rejected values.
  const after = useChatRuntimeStore.getState();
  assert.notEqual(after.params.presencePenalty, 1.5);
  assert.notEqual(
    (after.paramsByModel[QWEN38] as Record<string, unknown>).presencePenalty,
    1.5,
  );
});

test("an accepted retry still applies the migration locally", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 1);
  assert.equal(serverRow().presencePenalty, 1.5);
  assert.equal(serverRow().minP, 0);
  const after = useChatRuntimeStore.getState();
  assert.equal(after.params.presencePenalty, 1.5);
  assert.equal(after.params.minP, 0);
});

test("a backend without the conditional route persists nothing and shows nothing", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  // The desktop app adopts backends above a version floor, so an older one that
  // never learned this route is a supported install rather than an error.
  settingsHttp.conditionalStatus = 404;
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(serverRow().presencePenalty, 0);
  assert.equal(serverRow().minP, 0.01);
  const after = useChatRuntimeStore.getState();
  assert.notEqual(after.params.presencePenalty, 1.5);
});

test("the browser build's 405 is treated the same as an absent route", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  settingsHttp.conditionalStatus = 405;
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(serverRow().presencePenalty, 0);
  assert.notEqual(useChatRuntimeStore.getState().params.presencePenalty, 1.5);
});

test("a preset modified during hydration is not migrated on a stale read", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));
  settingsHttp.hold();
  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();
  // A slider moves while the settings GET is in flight. Its provenance write
  // sits behind the debounce, so the server still reads "builtin-default".
  useChatRuntimeStore.getState().setActivePresetSource("modified");
  settingsHttp.release?.();
  await hydration;

  assert.equal(useChatRuntimeStore.getState().activePresetSource, "modified");
  assert.deepEqual(
    settingsHttp.puts.filter((put) => "inferenceParamsByModel" in put),
    [],
  );
  assert.equal(serverRow().minP, 0.01);
  assert.equal(serverRow().presencePenalty, 0);
});

test("a normalized exact key cannot overwrite a row added after the read", async () => {
  const lower = QWEN38.toLowerCase();
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [lower]: LEGACY_SNAPSHOT },
  });
  const newerRow = { temperature: 0.31, presencePenalty: 0.4, maxTokens: 2048 };
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = {
      activePreset: "Default",
      activePresetSource: "builtin-default",
      inferenceParamsByModel: {
        [lower]: LEGACY_SNAPSHOT,
        [QWEN38]: newerRow,
      },
    };
  };
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [lower]: LEGACY_SNAPSHOT },
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

  // The exact-key row the other tab wrote must survive intact, and no
  // per-model patch may be sent for it at all.
  assert.deepEqual(serverRow(QWEN38), newerRow);
  assert.deepEqual(serverRow(lower), LEGACY_SNAPSHOT);
  assert.deepEqual(
    settingsHttp.puts.filter((put) => "inferenceParamsByModel" in put),
    [],
  );
});

test("a model id containing slashes stays one absence-fence path segment", async () => {
  const lower = QWEN38.toLowerCase();
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [lower]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [lower]: LEGACY_SNAPSHOT },
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

  // Nothing raced it, so the normalization applies under the exact spelling.
  assert.equal(serverRow(QWEN38).presencePenalty, 1.5);
});

test("the loaded model's reasoning mode survives hydration", async () => {
  // A small Qwen defaults to non-thinking while storage still says thinking.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: true,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: false,
    settingsHydrated: false,
  }));
  noteLoadedModelReasoningMode(QWEN38, false, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const after = useChatRuntimeStore.getState();
  // The pill and the sampling table have to agree; both follow the load.
  assert.equal(after.reasoningEnabled, false);
  assert.equal(after.params.temperature, 0.7);
  assert.equal(after.params.topP, 0.8);
  assert.equal(after.params.presencePenalty, 1.5);
  assert.equal(after.params.minP, 0);
});

test("a status refresh cannot outrank the installation's persisted reasoning", async () => {
  // Adoption of a resident model echoes whatever this browser had locally,
  // which before hydration is a local default, not the installation setting.
  // Only a load this browser performed may win.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: false,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));
  // This browser never loaded QWEN38: clear any load marker left by an earlier
  // case, then let a status merely report the model (fromLoad defaults false).
  noteLoadedModelReasoningMode("unsloth/Llama-3.2-3B-Instruct-GGUF", false);
  noteLoadedModelReasoningMode(QWEN38, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  // The server said thinking is off for this installation. It stays off.
  assert.equal(useChatRuntimeStore.getState().reasoningEnabled, false);
});

test("the post-load refresh does not drop the load's claim on the mode", async () => {
  // performLoad marks fromLoad, then awaits refresh(), whose status merge calls
  // noteLoadedModelReasoningMode again with the default false. Downgrading there
  // would let hydration replay the previous model's persisted toggle.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: true,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: false,
    settingsHydrated: false,
  }));
  noteLoadedModelReasoningMode(QWEN38, false, true);
  // The refresh that performLoad awaits, reporting the same model.
  noteLoadedModelReasoningMode(QWEN38, false);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const after = useChatRuntimeStore.getState();
  assert.equal(after.reasoningEnabled, false);
  assert.equal(after.params.temperature, 0.7);
  assert.equal(after.params.topP, 0.8);
});

test("a model without reasoning support is migrated as non-thinking", async () => {
  // The load-time overlay is gated on supportsReasoning, so a toggle left over
  // from the previous model must not pick the thinking table for the row.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: false,
    settingsHydrated: false,
  }));
  // A status reported this checkpoint, and it cannot reason.
  noteLoadedModelReasoningMode("unsloth/Llama-3.2-3B-Instruct-GGUF", false);
  useChatRuntimeStore.setState({ reasoningEnabled: true });
  noteLoadedModelReasoningMode(QWEN38, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  assert.equal(serverRow().temperature, 0.7);
  assert.equal(serverRow().topP, 0.8);
});

test("an empty stored model map does not fence off the migration", async () => {
  // sanitizeChatSettings drops inferenceParamsByModel: {}, so asserting its
  // absence would fence a key the row still has and the CAS would reject.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
    },
    inferenceParamsByModel: {},
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const global = settingsHttp.settings.inferenceParams as Record<string, number>;
  assert.equal(global.presencePenalty, 1.5);
  assert.equal(global.minP, 0);
});

test("case-distinct POSIX paths keep separate reasoning-mode records", async () => {
  // Two different files on a case-sensitive filesystem. The mode recorded for
  // one must not gate hydration or pick the migration table for the other.
  const upper = "/home/u/Models/qwen3.8-27b";
  const lower = "/home/u/models/qwen3.8-27b";
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: true,
    inferenceParamsByModel: { [lower]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: lower },
    paramsByModel: { [lower]: LEGACY_SNAPSHOT },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: false,
    supportsReasoning: true,
    settingsHydrated: false,
  }));
  // The load was for the other path entirely.
  noteLoadedModelReasoningMode(upper, false, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  // No record for this checkpoint, so the persisted toggle stands.
  assert.equal(useChatRuntimeStore.getState().reasoningEnabled, true);
});
