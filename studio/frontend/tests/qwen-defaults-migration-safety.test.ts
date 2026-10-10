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

function seedActiveQwen(overrides: Record<string, unknown> = {}): void {
  const hydratedSnapshot = { ...LEGACY_SNAPSHOT, minPMode: "custom" as const };
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...hydratedSnapshot, checkpoint: QWEN38 },
    paramsByModel: { [QWEN38]: hydratedSnapshot },
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: true,
    ...overrides,
  }));
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

async function adoptQwenDefaults(): Promise<void> {
  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(50);
}

function serverRow(key = QWEN38): Record<string, unknown> {
  return (
    settingsHttp.settings.inferenceParamsByModel as Record<
      string,
      Record<string, unknown>
    >
  )[key];
}

const BUILTIN_DEFAULT = {
  activePreset: "Default",
  activePresetSource: "builtin-default",
} as const;

const LEGACY_SETTINGS = {
  ...BUILTIN_DEFAULT,
  inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
};

test("a rejected retry leaves local sampling on the value the server kept", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = {
      ...LEGACY_SETTINGS,
      inferenceParamsByModel: {
        [QWEN38]: { ...LEGACY_SNAPSHOT, presencePenalty: 0.4 },
      },
    };
  };
  seedActiveQwen();

  await adoptQwenDefaults();

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(serverRow().presencePenalty, 0.4);
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

  await adoptQwenDefaults();

  assert.equal(settingsHttp.puts.length, 1);
  assert.equal(serverRow().presencePenalty, 1.5);
  assert.equal(serverRow().minP, 0);
  const after = useChatRuntimeStore.getState();
  assert.equal(after.params.presencePenalty, 1.5);
  assert.equal(after.params.minP, 0);
});

test("a backend without the conditional route persists nothing and shows nothing", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  // Desktop adopts older backends without this route, so a 404 is a supported install.
  settingsHttp.conditionalStatus = 404;
  seedActiveQwen();

  await adoptQwenDefaults();

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

  await adoptQwenDefaults();

  assert.equal(settingsHttp.puts.length, 0);
  assert.equal(serverRow().presencePenalty, 0);
  assert.notEqual(useChatRuntimeStore.getState().params.presencePenalty, 1.5);
});

test("a preset modified during hydration is not migrated on a stale read", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  seedActiveQwen({
    settingsHydrated: false,
  });
  settingsHttp.hold();
  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();
  // Provenance write sits behind the debounce, so the server still reads builtin-default.
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
    ...BUILTIN_DEFAULT,
    inferenceParamsByModel: { [lower]: LEGACY_SNAPSHOT },
  });
  const newerRow = { temperature: 0.31, presencePenalty: 0.4, maxTokens: 2048 };
  settingsHttp.beforeConditionalApply = () => {
    settingsHttp.settings = {
      ...BUILTIN_DEFAULT,
      inferenceParamsByModel: {
        [lower]: LEGACY_SNAPSHOT,
        [QWEN38]: newerRow,
      },
    };
  };
  seedActiveQwen({
    paramsByModel: { [lower]: LEGACY_SNAPSHOT },
  });

  await adoptQwenDefaults();

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
    ...BUILTIN_DEFAULT,
    inferenceParamsByModel: { [lower]: LEGACY_SNAPSHOT },
  });
  seedActiveQwen({
    paramsByModel: { [lower]: LEGACY_SNAPSHOT },
  });

  await adoptQwenDefaults();

  assert.equal(serverRow(QWEN38).presencePenalty, 1.5);
});

test("the loaded model's reasoning mode survives hydration", async () => {
  // A small Qwen defaults to non-thinking while storage still says thinking.
  resetHttp({
    ...BUILTIN_DEFAULT,
    reasoningEnabled: true,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  seedActiveQwen({
    reasoningEnabled: false,
    settingsHydrated: false,
  });
  noteLoadedModelReasoningMode(QWEN38, false, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const after = useChatRuntimeStore.getState();
  assert.equal(after.reasoningEnabled, false);
  assert.equal(after.params.temperature, 0.7);
  assert.equal(after.params.topP, 0.8);
  assert.equal(after.params.presencePenalty, 1.5);
  assert.equal(after.params.minP, 0);
});

test("a status refresh cannot outrank the installation's persisted reasoning", async () => {
  // Before hydration the echo is a local default, so only a load from this browser may win.
  resetHttp({
    ...BUILTIN_DEFAULT,
    reasoningEnabled: false,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  seedActiveQwen({
    settingsHydrated: false,
  });
  noteLoadedModelReasoningMode("unsloth/Llama-3.2-3B-Instruct-GGUF", false);
  noteLoadedModelReasoningMode(QWEN38, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  assert.equal(useChatRuntimeStore.getState().reasoningEnabled, false);
});

test("the post-load refresh does not drop the load's claim on the mode", async () => {
  // refresh() re-notes the mode with false; downgrading would replay the previous toggle.
  resetHttp({
    ...BUILTIN_DEFAULT,
    reasoningEnabled: true,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  seedActiveQwen({
    reasoningEnabled: false,
    settingsHydrated: false,
  });
  noteLoadedModelReasoningMode(QWEN38, false, true);
  noteLoadedModelReasoningMode(QWEN38, false);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const after = useChatRuntimeStore.getState();
  assert.equal(after.reasoningEnabled, false);
  assert.equal(after.params.temperature, 0.7);
  assert.equal(after.params.topP, 0.8);
});

test("a model without reasoning support is migrated as non-thinking", async () => {
  // The overlay is gated on supportsReasoning, so a leftover toggle must not pick the thinking table.
  resetHttp({
    ...BUILTIN_DEFAULT,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  seedActiveQwen({
    supportsReasoning: false,
    settingsHydrated: false,
  });
  noteLoadedModelReasoningMode("unsloth/Llama-3.2-3B-Instruct-GGUF", false);
  useChatRuntimeStore.setState({ reasoningEnabled: true });
  noteLoadedModelReasoningMode(QWEN38, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  assert.equal(serverRow().temperature, 0.7);
  assert.equal(serverRow().topP, 0.8);
});

test("an empty stored model map is declined, since it cannot be fenced", async () => {
  // An empty map matches any populated one by recursive subset, so this declines to write unfenced.
  resetHttp({
    ...BUILTIN_DEFAULT,
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
    },
    inferenceParamsByModel: {},
  });
  seedActiveQwen({
    paramsByModel: {},
    rememberParamsPerModel: false,
    settingsHydrated: false,
  });

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const global = settingsHttp.settings.inferenceParams as Record<string, number>;
  assert.equal(global.presencePenalty, 0);
  assert.equal(global.minP, 0.01);
});

test("case-distinct POSIX paths keep separate reasoning-mode records", async () => {
  // Two different files on a case-sensitive filesystem. The mode recorded for
  // one must not gate hydration or pick the migration table for the other.
  const upper = "/home/u/Models/qwen3.8-27b";
  const lower = "/home/u/models/qwen3.8-27b";
  resetHttp({
    ...BUILTIN_DEFAULT,
    reasoningEnabled: true,
    inferenceParamsByModel: { [lower]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: lower },
    paramsByModel: { [lower]: LEGACY_SNAPSHOT },
    ...BUILTIN_DEFAULT,
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: false,
    supportsReasoning: true,
    settingsHydrated: false,
  }));
  noteLoadedModelReasoningMode(upper, false, true);

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  assert.equal(useChatRuntimeStore.getState().reasoningEnabled, true);
});

test("a checkpoint switch during the conditional write is not migrated", async () => {
  // Applying after a switch would mark the second row migrated; the server updated only the first.
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
  await sleep(60);

  const after = useChatRuntimeStore.getState();
  const otherRow = after.paramsByModel[other] as Record<string, number>;
  assert.equal(otherRow.presencePenalty, 0);
  assert.equal(otherRow.minP, 0.01);
});

test("case-distinct POSIX ownership claims still conflict", async () => {
  // Different files, so competing ownership claims must cancel rather than the later one win.
  const upper = "/home/u/Models/qwen3.8-27b";
  const lower = "/home/u/models/qwen3.8-27b";
  resetHttp({
    ...BUILTIN_DEFAULT,
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
    ...BUILTIN_DEFAULT,
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
  await sleep(60);

  const global = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(global.presencePenalty, 0);
  assert.equal(global.minP, 0.01);
});

test("selecting an external Qwen migrates its dormant row", async () => {
  // No load or status follows an external pick, so setCheckpoint must repair the row.
  const external = `external::openrouter::${encodeURIComponent(
    "Qwen/Qwen3.8-27B",
  )}`;
  resetHttp({
    ...BUILTIN_DEFAULT,
    inferenceParamsByModel: { [external]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "unsloth/Llama-3.2-3B-Instruct-GGUF" },
    paramsByModel: { [external]: { ...LEGACY_SNAPSHOT } },
    ...BUILTIN_DEFAULT,
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setCheckpoint(external, null);
  await sleep(60);

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
    ...BUILTIN_DEFAULT,
    inferenceParamsByModel: { [external]: LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, checkpoint: "unsloth/Llama-3.2-3B-Instruct-GGUF" },
    paramsByModel: { [external]: { ...LEGACY_SNAPSHOT } },
    ...BUILTIN_DEFAULT,
    rememberParamsPerModel: true,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    supportsReasoning: true,
    settingsHydrated: true,
  }));

  useChatRuntimeStore.getState().setCheckpoint(external, null);
  // Raced against a deadline so a regression fails instead of hanging CI.
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
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.getState().setActivePresetSource("modified");
  };
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(60);

  assert.equal(serverRow().presencePenalty, 1.5);
  const after = useChatRuntimeStore.getState();
  assert.equal(after.activePresetSource, "modified");
  assert.equal(after.params.presencePenalty, 1.5);
  assert.equal(after.params.minP, 0);
});

test("a decision field that sanitizes away blocks the write", async () => {
  // An explicit null cannot be fenced as value or absence, so the migration declines.
  resetHttp({ ...LEGACY_SETTINGS, activePresetSource: null });
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(60);

  // Check the row, not the queue: an earlier case's debounced PUT can land here.
  assert.equal(serverRow().presencePenalty, 0);
  assert.equal(serverRow().minP, 0.01);
});

test("a case-distinct external switch during the write is not migrated", async () => {
  // Provider-qualified ids are opaque even though normalizeModelIdentity would fold them.
  const upper = `external::vendor::${encodeURIComponent("Vendor/Qwen3.8-27B")}`;
  const lower = `external::vendor::${encodeURIComponent("vendor/qwen3.8-27b")}`;
  resetHttp({
    ...BUILTIN_DEFAULT,
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
    ...BUILTIN_DEFAULT,
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
  await sleep(60);

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
    ...BUILTIN_DEFAULT,
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
  seedActiveQwen({
    paramsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
  });

  const first = useChatRuntimeStore.getState();
  first.setParams(
    { ...first.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(0);
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN36 },
  }));
  const second = useChatRuntimeStore.getState();
  second.setParams(
    { ...second.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(10);
  releaseFirst();
  await sleep(30);
  setTimeout(releaseSecond, 40);

  // The deadline keeps a wrongly cleared barrier a failure rather than a hang.
  await Promise.race([
    awaitPendingQwenDefaultsMigration(),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  assert.equal(serverRow(QWEN36).presencePenalty, 1.5);
});

test("the send barrier follows a retry that replaces the one it captured", async () => {
  const QWEN36 = "unsloth/Qwen3.6-14B-GGUF";
  const base = {
    ...BUILTIN_DEFAULT,
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
  seedActiveQwen({
    paramsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
  });

  const first = useChatRuntimeStore.getState();
  first.setParams(
    { ...first.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(0);
  const barrier = awaitPendingQwenDefaultsMigration();
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN36 },
  }));
  const second = useChatRuntimeStore.getState();
  second.setParams(
    { ...second.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(10);
  releaseFirst();
  setTimeout(releaseSecond, 60);

  await Promise.race([
    barrier,
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  assert.equal(serverRow(QWEN36).presencePenalty, 1.5);
});

test("a write outlasting the flush timeout rearms the migration", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  let releasePut: () => void = () => undefined;
  settingsHttp.putGate = new Promise<void>((resolve) => {
    releasePut = resolve;
  });
  seedActiveQwen();
  const before = useChatRuntimeStore.getState();
  before.setAutoTitle(!before.autoTitle);
  await sleep(600);

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  // Past the 2000 ms flush timeout with the ordinary write still outstanding.
  await sleep(2600);
  assert.equal(serverRow().presencePenalty, 0);

  releasePut();
  settingsHttp.putGate = null;
  await sleep(200);
  await Promise.race([
    awaitPendingQwenDefaultsMigration(),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
  assert.equal(serverRow().presencePenalty, 1.5);
});

test("a status-only reasoning marker does not pick the migration table", async () => {
  // Status lands before the settings GET, so the marker holds the local default.
  resetHttp({
    ...BUILTIN_DEFAULT,
    reasoningEnabled: false,
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  });
  seedActiveQwen({
    paramsByModel: { [QWEN38]: { ...LEGACY_SNAPSHOT } },
    supportsReasoning: true,
  });
  noteLoadedModelReasoningMode(QWEN38, true);

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(60);

  assert.equal(serverRow().temperature, 0.7);
  assert.equal(serverRow().topP, 0.8);
});

test("a null model map is fenced rather than treated as empty", async () => {
  // A null map is a key another tab can replace, and the CAS asserts neither value nor absence.
  resetHttp({
    ...BUILTIN_DEFAULT,
    inferenceParams: { ...LEGACY_SNAPSHOT },
    inferenceParamsByModel: null,
  });
  seedActiveQwen({
    paramsByModel: {},
    activePresetSource: "custom",
    rememberParamsPerModel: false,
    supportsReasoning: true,
  });

  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  await sleep(80);

  const globals = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(globals.presencePenalty, 0);
  assert.equal(globals.minP, 0.01);
});

test("per-model memory enabled during the read cancels the global migration", async () => {
  const OTHER = "unsloth/Llama-3.2-3B-GGUF";
  resetHttp({
    ...BUILTIN_DEFAULT,
    rememberParamsPerModel: true,
    inferenceParams: { ...LEGACY_SNAPSHOT },
    inferenceParamsByModel: { [OTHER]: { ...LEGACY_SNAPSHOT } },
  });
  seedActiveQwen({
    paramsByModel: {},
    activePresetSource: "custom",
    rememberParamsPerModel: false,
    supportsReasoning: true,
  });

  useChatRuntimeStore.getState().setActivePresetSource("builtin-default");
  await sleep(80);

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
  await sleep(20);

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
  // setCheckpoint marks pre-hydration switches interactive; adoption says otherwise.
  resetHttp({
    ...BUILTIN_DEFAULT,
    inferenceParams: { ...LEGACY_SNAPSHOT },
  });
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: "" },
    paramsByModel: {},
    ...BUILTIN_DEFAULT,
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
  await sleep(60);

  const global = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(global.presencePenalty, 1.5);
});

test("a thread pin survives the race recovery path", async () => {
  // Applying a thread snapshot advances no mutation version, so recovery would miss the pin.
  resetHttp({ ...LEGACY_SETTINGS });
  seedActiveQwen();
  useChatRuntimeStore
    .getState()
    .applyThreadScopedSettings("thread-1", { presencePenalty: 0.9 });
  useChatRuntimeStore.setState({ activeThreadId: "thread-1" });
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.getState().setActivePresetSource("modified");
  };

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(80);

  assert.equal(
    useChatRuntimeStore.getState().params.presencePenalty,
    0.9,
    "the open chat's pinned value was replaced by the migrated default",
  );
});

test("an unreachable backend does not spin the migration rearm", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
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
  await sleep(1500);

  assert.ok(
    settingsHttp.puts.length < 40,
    `rearmed without bound: ${settingsHttp.puts.length} writes`,
  );
  settingsHttp.putFailures = [];
});

test("a model switch during hydration still gets its own row migrated", async () => {
  const QWEN36 = "unsloth/Qwen3.6-14B-GGUF";
  const stored = {
    ...BUILTIN_DEFAULT,
    inferenceParamsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
  };
  resetHttp({ ...stored });
  let releaseConfirm: () => void = () => undefined;
  const confirming = new Promise<Record<string, unknown>>((resolve) => {
    releaseConfirm = () => resolve(settingsHttp.settings);
  });
  settingsHttp.getResponses.push(stored, confirming);
  seedActiveQwen({
    paramsByModel: {
      [QWEN38]: { ...LEGACY_SNAPSHOT },
      [QWEN36]: { ...LEGACY_SNAPSHOT },
    },
    supportsReasoning: true,
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  await sleep(20);
  useChatRuntimeStore.setState((state) => ({
    params: { ...state.params, ...LEGACY_SNAPSHOT, checkpoint: QWEN36 },
  }));
  releaseConfirm();
  await hydrating;
  await sleep(80);

  const row = (
    settingsHttp.settings.inferenceParamsByModel as Record<
      string,
      Record<string, number>
    >
  )[QWEN36];
  assert.equal(row.presencePenalty, 1.5);
  assert.equal(row.minP, 0);
});

test("a custom preset chosen mid-write keeps its own legacy-valued fields", async () => {
  resetHttp({ ...LEGACY_SETTINGS });
  useChatRuntimeStore.getState().applyThreadScopedSettings(null, {});
  useChatRuntimeStore.setState({ activeThreadId: null });
  // Applying a preset with presencePenalty 0 does not move its mutation counter.
  settingsHttp.beforeConditionalApply = () => {
    useChatRuntimeStore.getState().setActivePresetSource("custom");
  };
  seedActiveQwen();

  const active = useChatRuntimeStore.getState();
  active.setParams(
    { ...active.params, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  await sleep(80);

  const after = useChatRuntimeStore.getState().params;
  assert.equal(after.presencePenalty, 0);
  assert.equal(after.minP, 0.01);
});
