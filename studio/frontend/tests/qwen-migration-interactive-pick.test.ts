// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Its own file because hydratePersistedSettings memoizes its promise for the
// life of the module, so a second call in an already-hydrated module returns
// without reading anything and would assert nothing.

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

const EXTERNAL_QWEN = `external::openrouter::${encodeURIComponent(
  "Qwen/Qwen3.8-27B",
)}`;

test("an external model picked during hydration does not claim the global", async () => {
  // setCheckpoint moves the checkpoint at once and records no adoption marker,
  // so ownership must not fall out of the absence of load markers. The global
  // here belongs to whatever another browser last used.
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
  useChatRuntimeStore.setState({
    rememberParamsPerModel: false,
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    settingsHydrated: false,
  });

  settingsHttp.hold();
  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.getState().setCheckpoint(EXTERNAL_QWEN);
  settingsHttp.release?.();
  await hydration;

  assert.equal(
    useChatRuntimeStore.getState().params.checkpoint,
    EXTERNAL_QWEN,
    "the pick landed",
  );
  const global = settingsHttp.settings.inferenceParams as Record<
    string,
    number
  >;
  assert.equal(global.presencePenalty, 0);
  assert.equal(global.minP, 0.01);
});
