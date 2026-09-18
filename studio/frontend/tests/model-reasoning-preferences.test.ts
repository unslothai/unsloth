// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type { ExternalReasoningCapabilities } from "../src/features/chat/provider-capabilities.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storageData, storage, fireWindowEvent } = installLocalStorageFake();
register("./thread-sampling-resolver.mjs", import.meta.url);
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import("./helpers/store-stubs/chat-history-storage.ts");
const {
  createModelReasoningEffortStore,
  useModelReasoningEffortStore,
  pinnedReasoningEffort,
} = await import("../src/features/model-picker/components/model-selector/model-reasoning-effort.ts");
const { resolveExternalReasoningEffort } = await import("../src/features/chat/provider-capabilities.ts");

const KEY = "unsloth_model_reasoning_effort";
const MODEL = "external::review::reasoning-model";
const THREAD = "review-thread";
const CAPS: ExternalReasoningCapabilities = {
  supportsReasoning: true,
  reasoningStyle: "reasoning_effort",
  reasoningAlwaysOn: false,
  supportsReasoningOff: true,
  reasoningEffortLevels: ["low", "medium", "high"],
};
type RuntimeModule = typeof import("../src/features/chat/stores/chat-runtime-store.ts");
const runtimeUrl = new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url).href;

async function boot(scenario: string, paired = true): Promise<RuntimeModule> {
  storageData.clear();
  storageData.set("unsloth_chat_settings_imported_to_studio_db", "true");
  useModelReasoningEffortStore.getState().syncFromStorage();
  settingsHttp.settings = { rememberParamsPerModel: false, reasoningEffort: "medium" };
  settingsHttp.puts.length = 0;
  threadRows.reset();
  const runtime = await import(`${runtimeUrl}?scenario=${scenario}`) as RuntimeModule;
  const store = runtime.useChatRuntimeStore;
  await store.getState().hydratePersistedSettings();
  store.getState().setCheckpoint(MODEL, null);
  store.setState({ ...CAPS, reasoningEffort: "medium" });
  store.getState().setActiveThreadId(THREAD);
  if (paired) store.getState().applyThreadScopedSettings(THREAD, { reasoningEffort: "medium" });
  else runtime.beginThreadScopedPairing(THREAD);
  settingsHttp.puts.length = 0;
  return runtime;
}

function remotePin(runtime: RuntimeModule, effort: string | null): void {
  createModelReasoningEffortStore(storage).getState().setModelReasoningEffort(MODEL, effort);
  fireWindowEvent("storage", { key: `${KEY}::${MODEL}`, storageArea: storage });
  runtime.reconcilePinnedReasoningEffort({ checkpoint: MODEL, caps: CAPS, providerType: "openai" });
}

test("independent pin edits survive delayed storage events", () => {
  storageData.clear();
  const a = createModelReasoningEffortStore(storage);
  const b = createModelReasoningEffortStore(storage);
  a.getState().setModelReasoningEffort("model-a", "low");
  b.getState().setModelReasoningEffort("model-b", "high");
  a.getState().syncFromStorage();
  b.getState().syncFromStorage();
  const expected = { "model-a": "low", "model-b": "high" };
  assert.deepEqual(a.getState().effortByModel, expected);
  assert.deepEqual(b.getState().effortByModel, expected);
  assert.deepEqual(createModelReasoningEffortStore(storage).getState().effortByModel, expected);
});

test("clearing an unrelated pin does not erase a delayed edit", () => {
  storageData.clear();
  storageData.set(KEY, JSON.stringify({ "model-b": "medium" }));
  const a = createModelReasoningEffortStore(storage);
  const b = createModelReasoningEffortStore(storage);
  a.getState().setModelReasoningEffort("model-a", "high");
  b.getState().setModelReasoningEffort("model-b", null);
  assert.deepEqual(createModelReasoningEffortStore(storage).getState().effortByModel, { "model-a": "high" });
});

test("legacy pins load without writes and cleared pins cannot return", () => {
  storageData.clear();
  const legacy = JSON.stringify({ a: "low", b: "medium" });
  storageData.set(KEY, legacy);
  const a = createModelReasoningEffortStore(storage);
  assert.equal(storageData.size, 1);
  assert.deepEqual(a.getState().effortByModel, { a: "low", b: "medium" });
  a.getState().setModelReasoningEffort("a", null);
  const b = createModelReasoningEffortStore(storage);
  assert.deepEqual(b.getState().effortByModel, { b: "medium" });
  b.getState().setModelReasoningEffort("b", "high");
  assert.equal(storageData.get(KEY), legacy);
  assert.deepEqual(createModelReasoningEffortStore(storage).getState().effortByModel, { b: "high" });
});

test("same-model edits use the last persisted write", () => {
  storageData.clear();
  const a = createModelReasoningEffortStore(storage);
  const b = createModelReasoningEffortStore(storage);
  a.getState().setModelReasoningEffort("a", "low");
  b.getState().setModelReasoningEffort("a", "high");
  a.getState().syncFromStorage();
  assert.equal(a.getState().effortByModel.a, "high");
  a.getState().setModelReasoningEffort("a", null);
  b.getState().syncFromStorage();
  assert.deepEqual(b.getState().effortByModel, {});
});

test("malformed legacy storage does not hide newer pins", () => {
  storageData.clear();
  storageData.set(KEY, "invalid json");
  storageData.set(`${KEY}::a`, "high");
  assert.deepEqual(createModelReasoningEffortStore(storage).getState().effortByModel, { a: "high" });
});

test("unavailable storage keeps edits in memory", () => {
  const store = createModelReasoningEffortStore(null);
  store.getState().setModelReasoningEffort("a", "low");
  assert.equal(store.getState().effortByModel.a, "low");
});

test("an empty offered ladder rejects stored pins", () => {
  useModelReasoningEffortStore.setState({ effortByModel: { a: "high" } });
  assert.equal(pinnedReasoningEffort("a", []), null);
  assert.equal(pinnedReasoningEffort("a", ["high"]), "high");
});

test("restoration preserves a valid chat preference without changing selection defaults", () => {
  for (const [providerType, expected] of [["openai", "high"], ["anthropic", "high"], ["gemini", "medium"]]) {
    assert.equal(resolveExternalReasoningEffort({ caps: CAPS, providerType, current: "low" }), expected);
    assert.equal(resolveExternalReasoningEffort({ caps: CAPS, providerType, current: "low", restore: true }), "low");
    assert.equal(resolveExternalReasoningEffort({ caps: { ...CAPS, defaultEffort: "high" }, providerType, current: "low", restore: true }), "low");
  }
});

test("restoration clamps unavailable levels and ignores toggle-only pins", () => {
  assert.equal(resolveExternalReasoningEffort({ caps: { ...CAPS, reasoningEffortLevels: ["low", "high"] }, providerType: "openai", current: "medium", restore: true }), "low");
  assert.equal(resolveExternalReasoningEffort({ caps: { ...CAPS, reasoningStyle: "enable_thinking" }, providerType: "openai", current: "low", pinned: "high", restore: true }), "low");
});

test("a pin arriving during the debounce does not become the thread preference", async (t) => {
  const runtime = await boot("pending-edit");
  const store = runtime.useChatRuntimeStore;
  t.mock.timers.enable({ apis: ["setTimeout"] });
  store.getState().setReasoningEffort("low");
  t.mock.timers.tick(100);
  remotePin(runtime, "high");
  assert.equal(store.getState().reasoningEffort, "high");
  t.mock.timers.tick(300);
  await runtime.awaitStartedThreadScopedSettingsWrites();
  assert.equal(threadRows.rows.get(THREAD)?.reasoningEffort, "low");
  assert.equal(store.getState().reasoningEffort, "high");
  assert.equal(settingsHttp.puts.some((patch) => "reasoningEffort" in patch), false);
});

test("an ordinary pending edit persists without a pin", async (t) => {
  const runtime = await boot("ordinary-edit");
  t.mock.timers.enable({ apis: ["setTimeout"] });
  runtime.useChatRuntimeStore.getState().setReasoningEffort("low");
  t.mock.timers.tick(400);
  await runtime.awaitStartedThreadScopedSettingsWrites();
  assert.equal(threadRows.rows.get(THREAD)?.reasoningEffort, "low");
});

test("clearing a pin before the debounce restores the pending preference", async (t) => {
  const runtime = await boot("clear-before-save");
  const store = runtime.useChatRuntimeStore;
  t.mock.timers.enable({ apis: ["setTimeout"] });
  store.getState().setReasoningEffort("low");
  remotePin(runtime, "high");
  remotePin(runtime, null);
  assert.equal(store.getState().reasoningEffort, "low");
  t.mock.timers.tick(400);
  await runtime.awaitStartedThreadScopedSettingsWrites();
  assert.equal(threadRows.rows.get(THREAD)?.reasoningEffort, "low");
});

test("reconciliation advances the queue epoch only when effort changes", async () => {
  const runtime = await boot("queue-epoch");
  const store = runtime.useChatRuntimeStore;
  const initial = store.getState().queuedSettingsEpoch;
  remotePin(runtime, "high");
  assert.equal(store.getState().queuedSettingsEpoch, initial + 1);
  remotePin(runtime, "high");
  assert.equal(store.getState().queuedSettingsEpoch, initial + 1);
  remotePin(runtime, null);
  assert.equal(store.getState().reasoningEffort, "medium");
  assert.equal(store.getState().queuedSettingsEpoch, initial + 2);
});

test("an incoming thread keeps its own effort beneath an existing pin", async () => {
  const runtime = await boot("switch-thread");
  const store = runtime.useChatRuntimeStore;
  remotePin(runtime, "high");
  store.getState().setActiveThreadId("other-thread");
  store.getState().applyThreadScopedSettings("other-thread", { reasoningEffort: "low" });
  assert.equal(store.getState().reasoningEffort, "high");
  remotePin(runtime, null);
  assert.equal(store.getState().reasoningEffort, "low");
});

test("held effort edits survive a pin arriving before pairing completes", async (t) => {
  const runtime = await boot("held-edit", false);
  const store = runtime.useChatRuntimeStore;
  t.mock.timers.enable({ apis: ["setTimeout"] });
  store.getState().setReasoningEffort("low");
  remotePin(runtime, "high");
  store.getState().applyThreadScopedSettings(THREAD, { reasoningEffort: "medium" });
  t.mock.timers.tick(400);
  await runtime.awaitStartedThreadScopedSettingsWrites();
  assert.equal(threadRows.rows.get(THREAD)?.reasoningEffort, "low");
  assert.equal(store.getState().reasoningEffort, "high");
  store.getState().setActiveThreadId(null);
  store.getState().applyThreadScopedSettings(null, null);
  remotePin(runtime, null);
  assert.equal(store.getState().reasoningEffort, "medium");
});

test("leaving mid-pairing persists the requested effort rather than the pin", async () => {
  const runtime = await boot("held-leave", false);
  runtime.useChatRuntimeStore.getState().setReasoningEffort("low");
  remotePin(runtime, "high");
  runtime.useChatRuntimeStore.getState().setActiveThreadId("other-thread");
  await runtime.commitHeldThreadScopedEditsToTheirThread();
  assert.equal(threadRows.rows.get(THREAD)?.reasoningEffort, "low");
});

test("a capability downgrade releases an effort pin", async () => {
  const runtime = await boot("toggle-only");
  const store = runtime.useChatRuntimeStore;
  remotePin(runtime, "high");
  const caps = { ...CAPS, reasoningStyle: "enable_thinking" as const };
  store.setState(caps);
  runtime.reconcilePinnedReasoningEffort({ checkpoint: MODEL, caps, providerType: "openai" });
  assert.equal(store.getState().reasoningEffort, "medium");
  assert.equal(runtime.pinHoldsLiveEffort(), false);
});

test("a global effort edit survives pinning after leaving a saved chat", async () => {
  const runtime = await boot("global-after-thread");
  const store = runtime.useChatRuntimeStore;
  store.getState().setActiveThreadId(null);
  store.getState().applyThreadScopedSettings(null, null);
  store.getState().setReasoningEffort("low");
  remotePin(runtime, "high");
  remotePin(runtime, null);
  assert.equal(store.getState().reasoningEffort, "low");
});
