// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The restored external pick is browser-local, but chat settings are
// installation-wide. Seeding the key before the store module loads is the whole
// point: the store reads it once while building its initial params.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const EXTERNAL_QWEN = `external::openrouter::${encodeURIComponent(
  "Qwen/Qwen3.8-27B",
)}`;

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
localStorageFake.set("unsloth_chat_last_external_checkpoint", EXTERNAL_QWEN);
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

test("a restored external checkpoint does not claim the shared global snapshot", async () => {
  // Another browser wrote this global for some other model. Nothing adopted a
  // resident model here, so the stale local pick must not migrate it.
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
    },
  };
  settingsHttp.puts.length = 0;

  assert.equal(
    useChatRuntimeStore.getState().params.checkpoint,
    EXTERNAL_QWEN,
    "the store restored the external pick",
  );
  useChatRuntimeStore.setState({
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  });

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const global = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(global.presencePenalty, 0);
  assert.equal(global.minP, 0.01);
  assert.deepEqual(
    settingsHttp.puts.filter((put) => "inferenceParams" in put),
    [],
  );
});

