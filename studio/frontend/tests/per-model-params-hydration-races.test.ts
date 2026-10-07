// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Own file: the other hydration suite shares store state across tests.

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

const A = "unsloth/model-a";
const B = "unsloth/model-b";
const STATUS_CONTEXT_LENGTH = 131072;
const STATUS = {
  inference: { temperature: 0.9 },
  is_gguf: true,
  context_length: STATUS_CONTEXT_LENGTH,
} as never;

/** The store is a module singleton, so set every varied field explicitly. */
function reset(
  params: Record<string, unknown>,
  rest: Record<string, unknown> = {},
) {
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, ...params },
    rememberParamsPerModel: true,
    paramsByModel: {},
    settingsHydrated: false,
    ...rest,
  });
}

test("with the memory off, the saved shared settings still reach a new model", async () => {
  // With memory off there is no previous model, so the global set must not be suppressed.
  settingsHttp.settings = {
    rememberParamsPerModel: false,
    inferenceParams: { temperature: 0.22, systemPrompt: "shared" },
  };
  reset({ checkpoint: A }, { rememberParamsPerModel: false });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  const s = useChatRuntimeStore.getState();
  s.setParams(
    mergeBackendRecommendedInference({
      current: { ...s.params, checkpoint: B },
      response: STATUS,
      modelId: B,
      presetSource: s.activePresetSource,
      loadedContextLength: STATUS_CONTEXT_LENGTH,
    }),
    { fromModelDefaults: true },
  );
  settingsHttp.release?.();
  await hydrating;

  const { params } = useChatRuntimeStore.getState();
  assert.equal(params.temperature, 0.22);
  assert.equal(params.systemPrompt, "shared");
});

test("a model that loaded mid-flight keeps its own context", async () => {
  settingsHttp.settings = {
    inferenceParams: { maxSeqLength: 131072, temperature: 0.9 },
    inferenceParamsByModel: { [B]: { temperature: 0.2 } },
  };
  reset({ checkpoint: A, maxSeqLength: 131072 });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  const s = useChatRuntimeStore.getState();
  s.setParams(
    { ...s.params, checkpoint: B, maxSeqLength: 4096 },
    { fromModelDefaults: true, maxTokensCap: 4096 },
  );
  settingsHttp.release?.();
  await hydrating;

  const { params } = useChatRuntimeStore.getState();
  assert.equal(
    params.maxSeqLength,
    4096,
    "the loaded context survives hydration",
  );
  assert.equal(params.temperature, 0.2, "the entry still replays");
});

test("a pre-hydration edit survives on an install that has no model map", async () => {
  // Legacy settings carry only inferenceParams, with no entry to replay.
  settingsHttp.settings = { inferenceParams: { temperature: 0.55 } };
  reset({ checkpoint: A });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  const s = useChatRuntimeStore.getState();
  s.setParams({ ...s.params, temperature: 0.11 });
  settingsHttp.release?.();
  await hydrating;
  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.11);

  const s2 = useChatRuntimeStore.getState();
  s2.setParams(
    mergeBackendRecommendedInference({
      current: s2.params,
      response: STATUS,
      modelId: A,
      presetSource: s2.activePresetSource,
      loadedContextLength: STATUS_CONTEXT_LENGTH,
    }),
    { fromModelDefaults: true },
  );
  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    0.11,
    "the edit is not replaced by the backend recommendation",
  );
});

test("setCheckpoint clamps a replayed budget to the context it is given", () => {
  reset(
    { checkpoint: "small", maxTokens: 2048 },
    {
      settingsHydrated: true,
      rememberParamsPerModel: true,
      paramsByModel: { big: { maxTokens: 131072 } },
    },
  );
  useChatRuntimeStore
    .getState()
    .setCheckpoint("big", null, { maxTokensCap: 4096 });
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 4096);
});

test("the loaded context caps the budget even with nothing remembered", () => {
  // The cap describes the load, so it cannot depend on a replay happening.
  reset(
    { checkpoint: "small", maxTokens: 32768 },
    { settingsHydrated: true, paramsByModel: {} },
  );
  useChatRuntimeStore
    .getState()
    .setCheckpoint("fresh", null, { maxTokensCap: 8192 });
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});

test("the memory being off does not disable the loaded-context cap", () => {
  reset(
    { checkpoint: "small", maxTokens: 32768 },
    {
      settingsHydrated: true,
      rememberParamsPerModel: false,
      paramsByModel: {},
    },
  );
  useChatRuntimeStore
    .getState()
    .setCheckpoint("fresh", null, { maxTokensCap: 8192 });
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});

test("a model left before hydration keeps the globals it was running with", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.33, systemPrompt: "A's prompt" },
  };
  reset({ checkpoint: A });
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.getState().setParams(
    {
      ...useChatRuntimeStore.getState().params,
      checkpoint: B,
      temperature: 0.95,
      systemPrompt: "B's prompt",
    },
    { fromModelDefaults: true },
  );
  settingsHttp.release?.();
  await hydrating;

  useChatRuntimeStore.getState().setCheckpoint(A, null);
  const { params } = useChatRuntimeStore.getState();
  assert.equal(params.temperature, 0.33);
  assert.equal(params.systemPrompt, "A's prompt");
});
