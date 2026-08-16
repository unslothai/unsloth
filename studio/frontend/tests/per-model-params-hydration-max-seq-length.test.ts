// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hydration replay is a second copy of the replay rules, so it needs its own
// guard. Its own file: per-model-params-hydration.test.ts shares store state
// across its tests, and an appended case picks up an earlier one's temperature.

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

const QWEN = "unsloth/Qwen3.5-9B-GGUF";

test("hydration does not replay a maxSeqLength an entry happens to carry", async () => {
  // maxSeqLength is persisted but never remembered: the context belongs to the
  // load config. This app writes none into an entry, but the row accepts one.
  settingsHttp.settings = {
    inferenceParamsByModel: {
      [QWEN]: { temperature: 0.2, maxSeqLength: 131072 },
    },
  };
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      maxSeqLength: 4096,
    },
  });

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const { params } = useChatRuntimeStore.getState();
  assert.equal(
    params.temperature,
    0.2,
    "the remembered temperature should still replay",
  );
  assert.equal(
    params.maxSeqLength,
    4096,
    "the loaded context must survive the hydration replay",
  );
});
