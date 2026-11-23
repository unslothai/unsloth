// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";
import * as zustandMiddleware from "zustand/middleware";

import * as audioCppCatalog from "../src/features/audio/audio-cpp-catalog.ts";
import * as sttContext from "../src/features/settings/stores/stt-context.ts";
import * as sttModelCatalog from "../src/features/settings/stores/stt-model-catalog.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { store, fireWindowEvent } = installLocalStorageFake();
const KEY = "unsloth_voice_settings";

const { useVoiceSettingsStore } = loadWithStubs<
  typeof import("../src/features/settings/stores/voice-settings-store.ts")
>(
  new URL("../src/features/settings/stores/voice-settings-store.ts", import.meta.url),
  {
    "@/lib/api-base": { isTauri: false },
    "../../audio/audio-cpp-catalog": audioCppCatalog,
    zustand,
    "zustand/middleware": zustandMiddleware,
    "./stt-model-catalog": sttModelCatalog,
    "./stt-context": sttContext,
    "../api/stt-device": { syncSttDevicePreference: () => undefined },
  },
);

const saved = () => JSON.parse(store.get(KEY) ?? "{}").state;

test("a save in this window keeps what another window added", () => {
  useVoiceSettingsStore.getState().setTtsRate(1.5);
  store.set(
    KEY,
    JSON.stringify({
      state: {
        ...saved(),
        dictionary: ["Zephyrine"],
        recentDictations: [{ id: "d1", text: "hello there", at: 1 }],
      },
      version: 1,
    }),
  );
  fireWindowEvent("storage", { key: KEY });
  useVoiceSettingsStore.getState().setAutoReadAloud(true);
  assert.deepEqual(saved().dictionary, ["Zephyrine"]);
  assert.deepEqual(
    saved().recentDictations.map((d: { text: string }) => d.text),
    ["hello there"],
  );
  assert.equal(saved().ttsRate, 1.5);
  assert.equal(saved().autoReadAloud, true);
});

test("writes to other keys leave the settings alone", () => {
  const before = useVoiceSettingsStore.getState().dictionary;
  fireWindowEvent("storage", { key: "unsloth_theme" });
  assert.equal(useVoiceSettingsStore.getState().dictionary, before);
});
