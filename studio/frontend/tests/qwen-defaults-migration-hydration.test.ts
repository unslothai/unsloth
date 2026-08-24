// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

const QWEN38 = "unsloth/Qwen3.8-27B-GGUF";
const LEGACY_SNAPSHOT = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1.0,
  presencePenalty: 0.0,
  maxTokens: 8192,
  systemPrompt: "",
  systemVariables: "",
  fastMode: false,
};

test("hydration replaces and persists the stale Qwen3.8 default snapshot", async () => {
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [QWEN38]: LEGACY_SNAPSHOT },
  };
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      checkpoint: QWEN38,
      minP: 0,
      presencePenalty: 1.5,
    },
    paramsByModel: {},
    activePreset: "Default",
    activePresetSource: "builtin-default",
    rememberParamsPerModel: true,
    settingsHydrated: false,
  }));

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.params.minP, 0);
  assert.equal(hydrated.params.presencePenalty, 1.5);
  assert.equal(hydrated.paramsByModel[QWEN38]?.minP, 0);
  assert.equal(hydrated.paramsByModel[QWEN38]?.presencePenalty, 1.5);

  await new Promise((resolve) => setTimeout(resolve, 600));
  const migrationPut = settingsHttp.puts.find(
    (put) =>
      (put.inferenceParamsByModel as Record<string, unknown> | undefined)?.[
        QWEN38
      ] !== undefined,
  );
  assert.deepEqual(migrationPut, {
    inferenceParamsByModel: {
      [QWEN38]: { minP: 0, presencePenalty: 1.5 },
    },
    inferenceParams: { minP: 0, presencePenalty: 1.5 },
  });
});
