// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const {
  sanitizeChatSettings,
  savePersistedChatSettingsPatch,
} = await import(
  "../src/features/chat/utils/chat-settings-storage.ts"
);
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

function resetForHydration(): void {
  settingsHttp.getResponses.length = 0;
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.setState((state) => ({
    params: {
      ...state.params,
      checkpoint: "unsloth/Llama-3.2-3B-Instruct-GGUF",
      samplingFieldsExplicit: ["top_p"],
    },
    paramsByModel: {},
    reasoningAlwaysOn: false,
    reasoningEnabled: true,
    reasoningEffort: "high",
    preserveThinking: true,
    rememberParamsPerModel: false,
    settingsHydrated: false,
  }));
}

test("sampling masks survive the settings sanitizer for every inference-param container", () => {
  const sanitized = sanitizeChatSettings({
    inferenceParams: {
      samplingFieldsExplicit: [],
      temperature: 0.6,
    },
    inferenceParamsByModel: {
      llama: {
        samplingFieldsExplicit: [
          "temperature",
          "not_a_sampling_field",
          "temperature",
        ],
        temperature: 0.2,
      },
    },
    customPresets: [
      {
        name: "Focused",
        params: {
          samplingFieldsExplicit: ["reasoning_effort"],
          temperature: 0.4,
        },
      },
    ],
  });

  assert.deepEqual(sanitized.inferenceParams?.samplingFieldsExplicit, []);
  assert.deepEqual(
    sanitized.inferenceParamsByModel?.llama?.samplingFieldsExplicit,
    ["temperature"],
  );
  assert.deepEqual(
    sanitized.customPresets?.[0]?.params.samplingFieldsExplicit,
    ["reasoning_effort"],
  );
});

test("an empty automatic mask survives write, response sanitization, and hydration", async () => {
  const snapshot: PersistedChatSettings = {
    inferenceParams: {
      samplingFieldsExplicit: [],
      temperature: 0.6,
      topP: 0.95,
    },
    reasoningEnabled: false,
    reasoningEffort: "low",
    preserveThinking: false,
  };
  settingsHttp.settings = { ...snapshot };
  settingsHttp.puts.length = 0;

  const saved = await savePersistedChatSettingsPatch(snapshot);
  assert.deepEqual(
    (
      settingsHttp.puts[0]?.inferenceParams as
        | Record<string, unknown>
        | undefined
    )?.samplingFieldsExplicit,
    [],
  );
  assert.deepEqual(saved.inferenceParams?.samplingFieldsExplicit, []);

  resetForHydration();
  settingsHttp.settings = { ...saved };
  await useChatRuntimeStore.getState().hydratePersistedSettings();

  assert.deepEqual(
    useChatRuntimeStore.getState().params.samplingFieldsExplicit,
    [],
  );
});

test("legacy top-level reasoning scalars become explicit sampling fields", async () => {
  resetForHydration();
  settingsHttp.settings = {
    reasoningEnabled: false,
    reasoningEffort: "low",
    preserveThinking: false,
  };

  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.reasoningEnabled, false);
  assert.equal(hydrated.reasoningEffort, "low");
  assert.equal(hydrated.preserveThinking, false);
  assert.deepEqual(hydrated.params.samplingFieldsExplicit, [
    "enable_thinking",
    "reasoning_effort",
    "preserve_thinking",
  ]);
});

test("a reasoning edit made during hydration keeps its newer explicit mask", async () => {
  resetForHydration();
  settingsHttp.settings = {
    reasoningEnabled: false,
    reasoningEffort: "high",
    preserveThinking: false,
  };
  settingsHttp.hold();

  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.getState().setReasoningEffort("low");
  settingsHttp.release?.();
  await hydration;

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.reasoningEffort, "low");
  assert.deepEqual(hydrated.params.samplingFieldsExplicit, [
    "top_p",
    "reasoning_effort",
  ]);
});
