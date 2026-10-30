// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings → Agents → Run settings are views of settings Studio already has, never copies:
// a change in Agents must be what Run settings / Model auto-switch read, and the other way round.

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { store, fireWindowEvent } = installLocalStorageFake();
let updates = 0;
Object.assign((globalThis as unknown as { window: Record<string, unknown> }).window, {
  dispatchEvent: (event: { type: string }) => {
    fireWindowEvent(event.type, event);
    return true;
  },
});
(globalThis as unknown as { window: { addEventListener: (t: string, f: () => void) => void } }).window.addEventListener(
  "unsloth-per-model-config-updated",
  () => {
    updates += 1;
  },
);

const per = await import("../src/features/model-picker/model-config/per-model-config.ts");
const remembered = await import(
  "../src/features/model-picker/model-config/remembered-model-settings.ts"
);
const state = await import("../src/features/settings/components/desktop-app-run-settings-state.ts");
const { followSavedIdleDraft } = await import("../src/features/settings/components/idle-draft.ts");

const PATH =
  "/models/hub/models--unsloth--Qwen3.8-GGUF/snapshots/abc123";
const QUANT = "UD-Q4_K_XL";

type Synced = [string, string | null | undefined, unknown, unknown];
function fakeSync() {
  const calls: Synced[] = [];
  return {
    calls,
    deps: {
      sync: ((...args: Synced) => {
        calls.push(args);
      }) as never,
    },
  };
}

function info(overrides: Partial<import("../src/features/settings/components/desktop-app-run-settings-state.ts").DesktopAppModelSettings> = {}) {
  return {
    model: "unsloth/Qwen3.8-GGUF:UD-Q4_K_XL",
    config_id: PATH,
    alias_id: "unsloth/Qwen3.8-GGUF",
    gguf_variant: QUANT,
    is_gguf: true,
    custom_context_length: null,
    speculative_type: null,
    trained_context: 262_144,
    mtp_available: true,
    guard_budget_bytes: null,
    guard_context: null,
    loaded: false,
    loaded_context: null,
    ...overrides,
  };
}

test.beforeEach(() => {
  store.clear();
});

test("a loaded GGUF is keyed exactly as Run settings keys it (load path + quant)", () => {
  assert.deepEqual(remembered.rememberedSettingsKey(PATH, QUANT), {
    modelId: PATH,
    ggufVariant: QUANT,
  });
  // A loose .gguf has no variant in Run settings.
  assert.deepEqual(remembered.rememberedSettingsKey("/m/x-Q4_K_M.gguf", "Q4_K_M"), {
    modelId: "/m/x-Q4_K_M.gguf",
    ggufVariant: null,
  });
});

test("Agents → Run settings: the context slider writes the remembered Context Length and mirrors it", () => {
  const { calls, deps } = fakeSync();
  const before = updates;
  const patch = state.contextPatch(131_072, 262_144);
  assert.ok(remembered.rememberModelSettingsPatch(PATH, QUANT, patch, deps));
  // What Run settings reads (useStoredPerModelConfig → resolveInitialConfig).
  const runSettings = per.resolveInitialConfig(PATH, QUANT);
  assert.equal(runSettings.remembered, true);
  assert.equal(runSettings.config.customContextLength, 131_072);
  // Every open reader was told, in this tab.
  assert.ok(updates > before);
  // The same server mirror Run settings' Save makes, so a gateway load applies it.
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], PATH);
  assert.equal(calls[0][1], QUANT);
  assert.equal((calls[0][2] as { customContextLength: number }).customContextLength, 131_072);
});

test("an open Run settings panel adopts the Agents write instead of keeping a stale draft", () => {
  const atOpen = per.resolveInitialConfig(PATH, QUANT);
  remembered.rememberModelSettingsPatch(PATH, QUANT, { customContextLength: 65_536 }, fakeSync().deps);
  const now = per.resolveInitialConfig(PATH, QUANT);
  assert.deepEqual(per.peerStoredConfigPatch(atOpen, now), { customContextLength: 65_536 });
  // Its own save, already in the draft, is a no-op.
  assert.equal(per.peerStoredConfigPatch(now, per.resolveInitialConfig(PATH, QUANT)), null);
});

test("Run settings → Agents: a Remembered Context Length and speculative mode show up in Agents", () => {
  per.savePerModelConfig(PATH, QUANT, {
    ...per.DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 65_536,
    speculativeType: "off",
  });
  const local = remembered.readRememberedModelSettings(PATH, QUANT, "unsloth/Qwen3.8-GGUF");
  const fields = state.sharedModelFields(local, info());
  const ctx = state.contextView(fields, info());
  assert.equal(ctx?.value, 65_536);
  assert.equal(ctx?.auto, false);
  assert.equal(state.mtpView(fields, info()).on, false);
});

test("a record under the catalog spelling is read, and an edit keeps its other fields", () => {
  per.savePerModelConfig("unsloth/Qwen3.8-GGUF", QUANT, {
    ...per.DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 32_768,
    kvCacheDtype: "q8_0",
  });
  const local = remembered.readRememberedModelSettings(PATH, QUANT, "unsloth/Qwen3.8-GGUF");
  assert.equal(local?.customContextLength, 32_768);
  remembered.rememberModelSettingsPatch(PATH, QUANT, state.mtpPatch(true), fakeSync().deps, local);
  const saved = per.resolveInitialConfig(PATH, QUANT).config;
  assert.equal(saved.speculativeType, "mtp");
  assert.equal(saved.kvCacheDtype, "q8_0");
  assert.equal(saved.customContextLength, 32_768);
});

test("the slider's top is Auto (full trained context): it forgets the pin, keeping server launch flags", () => {
  remembered.rememberModelSettingsPatch(PATH, QUANT, { customContextLength: 8_192 }, fakeSync().deps);
  const { calls, deps } = fakeSync();
  remembered.rememberModelSettingsPatch(PATH, QUANT, state.contextPatch(262_144, 262_144), deps);
  assert.equal(per.resolveInitialConfig(PATH, QUANT).remembered, false);
  assert.equal(calls[0][2], null);
  assert.deepEqual(calls[0][3], { keepLaunchFlags: true });
  const ctx = state.contextView(state.sharedModelFields(null, info()), info());
  assert.equal(ctx?.auto, true);
  assert.equal(ctx?.effective, 262_144);
});

test("the default is the full trained context, capped only by the memory guard (and says so)", () => {
  const guarded = info({ trained_context: 1_048_576, guard_context: 40_960, guard_budget_bytes: 20 * 1024 ** 3 });
  const ctx = state.contextView(state.sharedModelFields(null, guarded), guarded);
  assert.equal(ctx?.max, 1_048_576);
  assert.equal(ctx?.value, 1_048_576);
  assert.equal(ctx?.guardCapped, true);
  assert.equal(ctx?.effective, 40_960);
  assert.equal(ctx?.belowClaude, true);
  assert.equal(ctx?.belowCodex, false);
});

test("Agents shows the server's value when this browser remembers nothing", () => {
  const fields = state.sharedModelFields(null, info({ custom_context_length: 98_304, speculative_type: "mtp" }));
  assert.equal(state.contextView(fields, info())?.value, 98_304);
  assert.equal(state.mtpView(fields, info()).on, true);
});

test("MTP: Auto reads as on only when the model has MTP heads", () => {
  const auto = state.sharedModelFields(null, info());
  assert.deepEqual(state.mtpView(auto, info()), { available: true, auto: true, on: true });
  assert.equal(state.mtpView(auto, info({ mtp_available: false })).on, false);
  assert.deepEqual(state.mtpPatch(false), { speculativeType: "off" });
});

test("guard estimate: linear in context, floored at the loader's fit minimum", () => {
  // Mirror of routes/desktop_app_models.guard_fitted_context, exercised in pytest; here the
  // view only consumes it.
  const capped = info({ guard_context: 8_192 });
  assert.equal(state.contextView(state.sharedModelFields(null, capped), capped)?.effective, 8_192);
});

const autoSwitch = (enabled: boolean, autoUnloadIdleSeconds: number) =>
  ({ enabled, autoUnloadIdleSeconds }) as import("../src/features/settings/api/openai-auto-switch.ts").OpenAIAutoSwitchSettings;

test("Keep loaded writes Model auto-switch's Unload after idle, as its own Save does", () => {
  assert.deepEqual(state.keepAliveUpdate(900, autoSwitch(false, 0)), {
    enabled: true,
    autoUnloadIdleSeconds: 900,
  });
  assert.deepEqual(state.keepAliveUpdate(0, autoSwitch(true, 900)), {
    enabled: true,
    autoUnloadIdleSeconds: 0,
  });
  // Floored like the backend's MIN_AUTO_UNLOAD_IDLE_SECONDS.
  assert.equal(state.keepAliveUpdate(10, null).autoUnloadIdleSeconds, 60);
});

test("Keep loaded reads the effective setting, and Model memory → Keep resident vetoes it", () => {
  assert.deepEqual(state.keepAliveView(autoSwitch(true, 1800), false), { seconds: 1800, vetoed: false });
  // Off auto-switch never unloads (get_auto_unload_idle_seconds).
  assert.equal(state.keepAliveView(autoSwitch(false, 1800), false).seconds, 0);
  assert.equal(state.keepAliveView(autoSwitch(true, 1800), true).vetoed, true);
  // A value typed in Settings (7 min) is still a choice here.
  assert.ok(state.keepAliveChoices(420).includes(420));
});

test("Model auto-switch follows a write from Agents live, without discarding a typed draft", () => {
  assert.equal(followSavedIdleDraft("900", autoSwitch(true, 900), autoSwitch(true, 1800)), "1800");
  assert.equal(followSavedIdleDraft("123", autoSwitch(true, 900), autoSwitch(true, 1800)), "123");
  const section = readSrc("features/settings/components/model-auto-switch-section.tsx");
  assert.match(section, /subscribeOpenAIAutoSwitchSettings\(/);
});

test("the Agents panel subscribes to every store it shows", () => {
  const panel = readSrc("features/settings/components/desktop-app-run-settings.tsx");
  for (const needle of [
    "PER_MODEL_CONFIG_UPDATED_EVENT",
    '"storage"',
    "subscribeOpenAIAutoSwitchSettings(",
    "subscribeModelMemorySettings(",
    "rememberModelSettingsPatch(",
    "updateOpenAIAutoSwitchSettings(",
  ]) {
    assert.ok(panel.includes(needle), needle);
  }
  // No storage of its own.
  assert.doesNotMatch(panel, /localStorage\.|useLocalStorage/);
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(page, /peerStoredConfigPatch\(previous, storedRecord\)/);
});

test("the default model is the loaded one, else the first a Desktop app uses", () => {
  const rows = [
    { id: "a", loaded: false },
    { id: "b", loaded: true },
  ];
  assert.equal(state.defaultRunSettingsModel(rows, ["a"]), "b");
  assert.equal(state.defaultRunSettingsModel([{ id: "a", loaded: false }, { id: "c", loaded: false }], ["c"]), "c");
});
