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
const {
  awaitPendingQwenDefaultsMigration,
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

function resetHttp(settings: Record<string, unknown>): void {
  settingsHttp.settings = settings;
  settingsHttp.getResponses.length = 0;
  settingsHttp.puts.length = 0;
  settingsHttp.beforeConditionalApply = null;
  settingsHttp.conditionalStatus = 200;
  settingsHttp.gate = null;
  settingsHttp.release = null;
  settingsHttp.putGate = null;
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

test("a checkpoint switch during the conditional write is not migrated", async () => {
  // The CAS commits for QWEN38, but the user moves to another Qwen before the
  // response resolves. Applying then would mark the second model's row migrated
  // while the server updated only the first.
  const other = "unsloth/Qwen3.6-9B-GGUF";
  resetHttp({ ...LEGACY_SETTINGS });
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.setState((state) => ({
      params: { ...state.params, checkpoint: other },
      paramsByModel: {
        ...state.paramsByModel,
        [other]: { ...LEGACY_SNAPSHOT },
      },
    }));
  };
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 60));

  // The row the user switched to keeps its legacy values locally.
  const after = useChatRuntimeStore.getState();
  const otherRow = after.paramsByModel[other] as Record<string, number>;
  assert.equal(otherRow.presencePenalty, 0);
  assert.equal(otherRow.minP, 0.01);
});

test("case-distinct POSIX ownership claims still conflict", async () => {
  // Two resident-model callbacks enqueue ownership before the microtask runs.
  // They are different files, so the competing claims must cancel each other
  // rather than the later one silently replacing the earlier.
  const upper = "/home/u/Models/qwen3.8-27b";
  const lower = "/home/u/models/qwen3.8-27b";
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
    },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: lower },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, checkpoint: upper },
    { fromModelDefaults: true, migrateOwnedGlobalQwenDefaults: true },
  );
  const second = useChatRuntimeStore.getState();
  second.setParams(
    { ...second.params, checkpoint: lower },
    { fromModelDefaults: true, migrateOwnedGlobalQwenDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 60));

  const global = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(global.presencePenalty, 0);
  assert.equal(global.minP, 0.01);
});

test("selecting an external Qwen migrates its dormant row", async () => {
  // No load and no status follows an external pick, so setCheckpoint is the
  // only chance to repair the row it just replayed.
  const external = `external::openrouter::${encodeURIComponent(
    "Qwen/Qwen3.8-27B",
  )}`;
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [external]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "unsloth/Llama-3.2-3B-Instruct-GGUF" },
    paramsByModel: { [external]: { ...LEGACY_SNAPSHOT } },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setCheckpoint(external, null);
  await new Promise((resolve) => setTimeout(resolve, 60));

  const row = serverRow(external);
  assert.equal(row.presencePenalty, 1.5);
  assert.equal(row.minP, 0);
  assert.equal(useChatRuntimeStore.getState().params.presencePenalty, 1.5);
});

test("the send barrier waits for a migration a model pick just scheduled", async () => {
  const external = `external::openrouter::${encodeURIComponent(
    "Qwen/Qwen3.6-9B",
  )}`;
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: { [external]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "unsloth/Llama-3.2-3B-Instruct-GGUF" },
    paramsByModel: { [external]: { ...LEGACY_SNAPSHOT } },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setCheckpoint(external, null);
  // What the run does before sending. No sleep: the point is that this alone
  // is enough, since the row is still legacy the microtask before it. Raced
  // against a deadline so a regression fails here instead of hanging CI.
  const settled = await Promise.race([
    awaitPendingQwenDefaultsMigration().then(() => "settled"),
    new Promise((resolve) => setTimeout(() => resolve("timeout"), 5000)),
  ]);
  assert.equal(settled, "settled", "the barrier never released");

  assert.equal(useChatRuntimeStore.getState().params.presencePenalty, 1.5);
  assert.equal(serverRow(external).presencePenalty, 1.5);
});

test("an edit racing the conditional write does not strand local on legacy", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  // The user moves a control after the confirming GET, while the write is out.
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.getState().setActivePresetSource("modified");
  };
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 60));

  // The server took the migration, so the tab must not keep generating from the
  // values it no longer holds.
  assert.equal(serverRow().presencePenalty, 1.5);
  const after = useChatRuntimeStore.getState();
  assert.equal(after.activePresetSource, "modified");
  assert.equal(after.params.presencePenalty, 1.5);
  assert.equal(after.params.minP, 0);
});

test("a decision field that sanitizes away blocks the write", async () => {
  // An explicit null cannot be asserted as a value or as an absence, so the
  // migration declines rather than writing unfenced.
  resetHttp({ ...LEGACY_SETTINGS, activePresetSource: null });
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 60));

  // The row itself, not the queue: an earlier case's debounced PUT can land in
  // this window and says nothing about whether the migration wrote.
  assert.equal(serverRow().presencePenalty, 0);
  assert.equal(serverRow().minP, 0.01);
});

test("an empty model map still migrates despite sanitizing away", async () => {
  // The companion to the case above: this one means what the migration already
  // assumes, so it must not be treated as unfenceable.
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
    supportsReasoning: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const global = settingsHttp.settings.inferenceParams as Record<string, number>;
  assert.equal(global.presencePenalty, 1.5);
});

test("a case-distinct external switch during the write is not migrated", async () => {
  // Provider-qualified ids are opaque, so these are two different models even
  // though normalizeModelIdentity would fold them together.
  const upper = `external::vendor::${encodeURIComponent("Vendor/Qwen3.8-27B")}`;
  const lower = `external::vendor::${encodeURIComponent("vendor/qwen3.8-27b")}`;
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: {
      [upper]: LEGACY_SNAPSHOT,
      [lower]: LEGACY_SNAPSHOT,
    },
  });
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.setState((state) => ({
      params: { ...state.params, checkpoint: lower },
    }));
  };
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: upper },
    paramsByModel: {
      [upper]: { ...LEGACY_SNAPSHOT },
      [lower]: { ...LEGACY_SNAPSHOT },
    },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 60));

  // Only the row the write was for; the other model keeps its own values.
  const lowerRow = useChatRuntimeStore.getState().paramsByModel[lower] as Record<
    string,
    number
  >;
  assert.equal(lowerRow.presencePenalty, 0);
  assert.equal(lowerRow.minP, 0.01);
});

test("an earlier retry settling does not clear a later retry's barrier", async () => {
  const QWEN36 = "unsloth/Qwen3.6-14B-GGUF";
  const base = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
  };
  resetHttp({ ...base });
  // Both confirming GETs are held, so the first retry finishes while the second
  // is still deciding.
  const holdGet = (): [Promise<Record<string, unknown>>, () => void] => {
    let release: () => void = () => undefined;
    const held = new Promise<Record<string, unknown>>((resolve) => {
      release = () => resolve(settingsHttp.settings);
    });
    return [held, release];
  };
  const [firstGet, releaseFirst] = holdGet();
  const [secondGet, releaseSecond] = holdGet();
  settingsHttp.getResponses.push(firstGet, secondGet);
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const first = useChatRuntimeStore.getState();
  first.setParams(
    { ...first.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 0));
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN36 },
  }));
  const second = useChatRuntimeStore.getState();
  second.setParams(
    { ...second.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 10));
  releaseFirst();
  await new Promise((resolve) => setTimeout(resolve, 30));
  setTimeout(releaseSecond, 40);

  // A barrier cleared by the wrong retry resolves here with the second row
  // still legacy; the deadline keeps that a failure rather than a hang.
  await Promise.race([
    awaitPendingQwenDefaultsMigration(),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  assert.equal(serverRow(QWEN36).presencePenalty, 1.5);
});

test("the send barrier follows a retry that replaces the one it captured", async () => {
  const QWEN36 = "unsloth/Qwen3.6-14B-GGUF";
  const base = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParamsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
  };
  resetHttp({ ...base });
  const holdGet = (): [Promise<Record<string, unknown>>, () => void] => {
    let release: () => void = () => undefined;
    const held = new Promise<Record<string, unknown>>((resolve) => {
      release = () => resolve(settingsHttp.settings);
    });
    return [held, release];
  };
  const [firstGet, releaseFirst] = holdGet();
  const [secondGet, releaseSecond] = holdGet();
  settingsHttp.getResponses.push(firstGet, secondGet);
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));

  const first = useChatRuntimeStore.getState();
  first.setParams(
    { ...first.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 0));
  // The send joins here, so it captures the first retry.
  const barrier = awaitPendingQwenDefaultsMigration();
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN36 },
  }));
  const second = useChatRuntimeStore.getState();
  second.setParams(
    { ...second.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 10));
  releaseFirst();
  setTimeout(releaseSecond, 60);

  await Promise.race([
    barrier,
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  // The checkpoint the send would generate from is the second one.
  assert.equal(serverRow(QWEN36).presencePenalty, 1.5);
});

test("a write outlasting the flush timeout rearms the migration", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  let releasePut: () => void = () => undefined;
  settingsHttp.putGate = new Promise<void>((resolve) => {
    releasePut = resolve;
  });
  seedActiveQwen();
  // An unrelated setting write, held open past the debounce so it is still on
  // the wire when the migration tries to land.
  const before = useChatRuntimeStore.getState();
  before.setAutoTitle(!before.autoTitle);
  await new Promise((resolve) => setTimeout(resolve, 600));

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  // Past the 2000 ms flush timeout with the ordinary write still outstanding.
  await new Promise((resolve) => setTimeout(resolve, 2600));
  assert.equal(serverRow().presencePenalty, 0);

  releasePut();
  settingsHttp.putGate = null;
  await new Promise((resolve) => setTimeout(resolve, 200));
  await Promise.race([
    awaitPendingQwenDefaultsMigration(),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  assert.equal(serverRow().presencePenalty, 1.5);
});

test("a status-only reasoning marker does not pick the migration table", async () => {
  // Status for a resident model lands before the settings GET, so the marker
  // carries this browser's local default rather than an answer about the model.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    reasoningEnabled: false,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: { ...LEGACY_SNAPSHOT } },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    supportsReasoning: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
  }));
  noteLoadedModelReasoningMode(QWEN38, true);

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 60));

  // The persisted toggle is off, so the non-thinking row is what the pill will
  // hydrate against.
  assert.equal(serverRow().temperature, 0.7);
  assert.equal(serverRow().topP, 0.8);
});

test("a null model map is fenced rather than treated as empty", async () => {
  // Only a genuinely empty map means "no per-model memory". A null is a key the
  // row holds, so another tab can replace it with rows this write never saw,
  // and the compare-and-set asserts neither its value nor its absence.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: { ...LEGACY_SNAPSHOT },
    inferenceParamsByModel: null,
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "custom",
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  await new Promise((resolve) => setTimeout(resolve, 80));

  const globals = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(globals.presencePenalty, 0);
  assert.equal(globals.minP, 0.01);
});

test("per-model memory enabled during the read cancels the global migration", async () => {
  const OTHER = "unsloth/Llama-3.2-3B-GGUF";
  // Scheduled while the global snapshot was the active model's to rewrite; the
  // confirming read shows per-model memory on, so it no longer is.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    inferenceParams: { ...LEGACY_SNAPSHOT },
    inferenceParamsByModel: { [OTHER]: { ...LEGACY_SNAPSHOT } },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "custom",
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  await new Promise((resolve) => setTimeout(resolve, 80));

  const globals = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(globals.presencePenalty, 0);
  assert.equal(globals.minP, 0.01);
});

test("a wedged backend does not hold a send open forever", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  // Neither the confirming GET nor the conditional write takes an abort signal.
  settingsHttp.hold();
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 20));

  const started = Date.now();
  let timedOut = false;
  await Promise.race([
    awaitPendingQwenDefaultsMigration(),
    new Promise((resolve) => {
      setTimeout(() => {
        timedOut = true;
        resolve(undefined);
      }, 6000);
    }),
  ]);
  settingsHttp.release?.();

  assert.equal(timedOut, false, "the barrier never released");
  assert.ok(Date.now() - started < 4000);
});

test("adopting the startup model clears the unowned pre-hydration mark", async () => {
  // Status finds a resident Qwen before hydration finishes. setCheckpoint marks
  // any pre-hydration switch as an interactive pick, and adoption is the signal
  // that says otherwise, so the global snapshot is still this model's.
  resetHttp({
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: { ...LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: "" },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: false,
  }));
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  useChatRuntimeStore.getState().setCheckpoint(QWEN38);
  const adopting = useChatRuntimeStore.getState();
  adopting.setParams(
    { ...adopting.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN38 },
    { fromModelDefaults: true, migrateOwnedGlobalQwenDefaults: true },
  );

  settingsHttp.release?.();
  await hydrating;
  await new Promise((resolve) => setTimeout(resolve, 60));

  const global = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(global.presencePenalty, 1.5);
});

test("a thread pin survives the race recovery path", async () => {
  // Applying a thread snapshot advances no installation mutation version, so
  // nothing in the recovery path would otherwise notice the pin.
  resetHttp({ ...LEGACY_SETTINGS });
  seedActiveQwen();
  useChatRuntimeStore
    .getState()
    .applyThreadScopedSettings("thread-1", { presencePenalty: 0.9 });
  useChatRuntimeStore.setState({ activeThreadId: "thread-1" });
  // Provenance moves while the conditional write is in flight, which is the
  // branch that copies migrated fields straight into the live params.
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.getState().setActivePresetSource("modified");
  };

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 80));

  assert.equal(
    useChatRuntimeStore.getState().params.presencePenalty,
    0.9,
    "the open chat's pinned value was replaced by the migrated default",
  );
});

test("an unreachable backend does not spin the migration rearm", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  // Every ordinary write is refused, so the patch is retained and each pass
  // flushes it again.
  settingsHttp.putFailures = Array.from({ length: 400 }, () => ({
    status: 503,
  }));
  seedActiveQwen();
  const before = useChatRuntimeStore.getState();
  before.setAutoTitle(!before.autoTitle);

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await new Promise((resolve) => setTimeout(resolve, 1500));

  assert.ok(
    settingsHttp.puts.length < 40,
    `rearmed without bound: ${settingsHttp.puts.length} writes`,
  );
  settingsHttp.putFailures = [];
});
