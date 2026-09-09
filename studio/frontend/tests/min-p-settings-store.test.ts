// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storage } = installLocalStorageFake();
storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore: store } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const A = "external::vllm-a::model-a";
const B = "external::vllm-b::model-b";

function reset(settings: Record<string, unknown> = {}) {
  settingsHttp.settings = settings;
  settingsHttp.gate = null;
  store.setState({
    params: { ...DEFAULT_INFERENCE_PARAMS, checkpoint: A },
    paramsByModel: {},
    rememberParamsPerModel: true,
    settingsHydrated: false,
  });
}

test("fresh settings remain server-default through startup and unknown-provider selection", async () => {
  reset();
  await store.getState().hydratePersistedSettings();
  assert.equal(store.getState().params.minPMode, "server-default");
  store.getState().setCheckpoint(B);
  assert.equal(store.getState().params.minPMode, "server-default");
});

test("legacy global and model records retain independent intent through both switching routes", async () => {
  reset({
    inferenceParams: { minPMode: "server-default", minP: 0.1 },
    inferenceParamsByModel: {
      [A]: { minP: 0.01 },
      [B]: { minPMode: "server-default", minP: 0.2 },
    },
    customPresets: [{ name: "Legacy", params: { minP: 0 } }],
  });
  await store.getState().hydratePersistedSettings();
  assert.equal(store.getState().params.minPMode, "custom");
  assert.equal(store.getState().params.minP, 0.01);
  assert.equal(store.getState().customPresets[0]?.params.minPMode, "custom");
  store.getState().setCheckpoint(B);
  assert.equal(store.getState().params.minPMode, "server-default");
  assert.equal(store.getState().params.minP, 0.2);
  store
    .getState()
    .setParams(
      { ...store.getState().params, checkpoint: A },
      { fromModelDefaults: true },
    );
  assert.equal(store.getState().params.minPMode, "custom");
  assert.equal(store.getState().params.minP, 0.01);
  store.getState().setCheckpoint("local-model");
  store.getState().setCheckpoint("external::openai::other");
  store.getState().setCheckpoint(A);
  assert.equal(store.getState().params.minPMode, "custom");
  store.getState().setRememberParamsPerModel(false);
  store.getState().setCheckpoint(B);
  assert.equal(store.getState().params.minPMode, "custom");
  assert.equal(store.getState().params.minP, 0.01);
});

test("mode-only and same-value explicit edits fence the retained number against a delayed GET", async () => {
  for (const minPMode of ["server-default", "custom"] as const) {
    reset({
      inferenceParams: { minP: 0.7 },
      inferenceParamsByModel: { [A]: { minP: 0.8 } },
    });
    settingsHttp.hold();
    const hydrating = store.getState().hydratePersistedSettings();
    const current = store.getState();
    const epoch = current.queuedSettingsEpoch;
    current.setParams(
      { ...current.params, minPMode },
      { minPChoiceEdited: true },
    );
    settingsHttp.release?.();
    await hydrating;
    assert.equal(store.getState().params.minPMode, minPMode);
    assert.equal(store.getState().params.minP, DEFAULT_INFERENCE_PARAMS.minP);
    assert.equal(store.getState().paramsByModel[A]?.minPMode, minPMode);
    assert.equal(
      store.getState().paramsByModel[A]?.minP,
      DEFAULT_INFERENCE_PARAMS.minP,
    );
    assert.ok(store.getState().queuedSettingsEpoch > epoch);
  }
});
