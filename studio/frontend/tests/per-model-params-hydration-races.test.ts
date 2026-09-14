// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Startup races where the settings GET is still in flight. Own file: the other
// hydration suite shares store state across its tests, so an appended case picks
// up an earlier one's params.

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

/** Every field this file varies is set explicitly: the store is a module
 * singleton, so a value left behind by an earlier test silently changes the
 * next one's meaning. */
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
  // keepModelDefaults exists so a model that loaded mid-flight is not handed the
  // previous model's globals. With the memory OFF there is no previous model:
  // the global set is the one set the user keeps for everything, and suppressing
  // it strands the model on whatever the load happened to recommend.
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
  // The global maxSeqLength belongs to whichever model was used last. No entry
  // ever carries one, so the replay cannot put the right value back after the
  // global loop has overwritten it.
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
  // The upgrade path: settings written before this feature carry only
  // inferenceParams. The edit is fenced out of the global set either way, but
  // without an entry the next defaults update has nothing to replay and puts the
  // backend recommendation back over it.
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
  // Compare's ensureModelLoaded reaches the replay through setCheckpoint, which
  // is the one switch path that had no way to pass the context it just loaded.
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
  // The cap describes the load, so it cannot be conditional on a replay: compare
  // loading a fresh 8K model after a 32K one has no entry to replay and would
  // otherwise send the 32K budget.
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
  // The upgrade path again, from the other side: A is resident with only the
  // legacy global set to its name, and B replaces it before the GET returns.
  // Nothing could be filed for A at the time, so without this A ends up with no
  // entry and switching back inherits B's settings.
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
