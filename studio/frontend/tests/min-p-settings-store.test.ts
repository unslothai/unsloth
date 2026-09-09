// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake().store.set(
  "unsloth_chat_settings_imported_to_studio_db",
  "true",
);
register("./store-settings-resolver.mjs", import.meta.url);
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore: store } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const initialParams = store.getState().params;
const { shouldOfferMinPRecovery } = await import(
  "../src/features/chat/lib/min-p-recovery.ts"
);
const A = "external::vllm-a::model-a";
const B = "external::vllm-b::model-b";

function reset(settings: Record<string, unknown> = {}) {
  settingsHttp.settings = settings;
  store.setState({
    params: { ...initialParams, checkpoint: A },
    paramsByModel: {},
    rememberParamsPerModel: true,
    settingsHydrated: false,
  });
}

test("fresh settings delegate; saved legacy and explicit modes survive switching", async () => {
  reset();
  await store.getState().hydratePersistedSettings();
  assert.equal(store.getState().params.minPMode, "server-default");
  reset({
    inferenceParamsByModel: {
      [A]: { minP: 0.01 },
      [B]: { minP: 0, minPMode: "server-default" },
    },
  });
  await store.getState().hydratePersistedSettings();
  assert.equal(store.getState().params.minPMode, "custom");
  assert.equal(store.getState().params.minP, 0.01);
  store.getState().setCheckpoint(B);
  assert.equal(store.getState().params.minPMode, "server-default");
  assert.equal(store.getState().params.minP, 0);
});

test("an explicit zero survives a delayed settings read", async () => {
  reset({ inferenceParams: { minP: 0.7 } });
  settingsHttp.hold();
  const hydrating = store.getState().hydratePersistedSettings();
  const current = store.getState();
  current.setParams(
    { ...current.params, minP: 0, minPMode: "custom" },
    { minPChoiceEdited: true },
  );
  settingsHttp.release?.();
  await hydrating;
  assert.equal(store.getState().params.minP, 0);
  assert.equal(store.getState().params.minPMode, "custom");
});

test("recovery offers zero for delegation but not an explicit zero", () => {
  const error =
    "The min_p and logit_bias sampling parameters are not yet supported with speculative decoding";
  for (const minPMode of ["server-default", "custom"] as const) {
    const offered = shouldOfferMinPRecovery(error, "vllm", {
      minP: 0,
      minPMode,
    });
    assert.equal(offered, minPMode === "server-default");
  }
});
