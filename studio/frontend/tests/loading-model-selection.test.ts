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
const { useChatRuntimeStore: store } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

const pick = { id: "incoming-local", ggufVariant: null, nativePathToken: null };
const previous = "external::provider-a::model-a";
const selected = "external::provider-b::model-b";

test("a later hosted selection supersedes the local load even after returning to the original provider", () => {
  store.getState().setCheckpoint(previous);
  store.getState().setLoadingModelPick(pick);
  assert.equal(store.getState().loadingModelPick?.selectionSuperseded, false);
  store.getState().setCheckpoint(selected);
  assert.equal(store.getState().loadingModelPick?.selectionSuperseded, true);
  store.getState().setCheckpoint(previous);
  assert.equal(store.getState().loadingModelPick?.selectionSuperseded, true);
  assert.equal(store.getState().loadingModelPick?.id, pick.id);
  store.getState().clearLoadingModelPick(pick);
  assert.equal(store.getState().loadingModelPick, null);
});

test("sampling changes and repeated checkpoint reconciliation keep the incoming load selected", () => {
  store.getState().setCheckpoint(previous);
  store.getState().setLoadingModelPick(pick);
  store.getState().setParams({ ...store.getState().params, temperature: 0.4 });
  store.getState().setCheckpoint(previous);
  assert.equal(store.getState().loadingModelPick?.selectionSuperseded, false);
  store.getState().setLoadingModelPick(null);
  assert.equal(store.getState().loadingModelPick, null);
});
