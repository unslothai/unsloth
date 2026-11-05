// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";
import * as zustandMiddleware from "zustand/middleware";

import * as audioCppCatalog from "../src/features/audio/audio-cpp-catalog.ts";
import {
  beginDictationSession,
  dictationProducedClip,
  markDictationClip,
} from "../src/features/chat/adapters/dictation-outcome.ts";
import { shouldSubmitDictation } from "../src/features/chat/utils/dictation-send.ts";
import {
  micClipFilename,
  micSendsAudioToModel,
} from "../src/features/chat/utils/mic-target.ts";
import * as sttModelCatalog from "../src/features/settings/stores/stt-model-catalog.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { store } = installLocalStorageFake();
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
    "../api/stt-device": { syncSttDevicePreference: () => undefined },
  },
);

const rehydrateWith = async (saved: Record<string, unknown>) => {
  store.set(KEY, JSON.stringify({ state: saved, version: 1 }));
  await useVoiceSettingsStore.persist.rehydrate();
};

test("the mic defaults to ASR dictation", () => {
  assert.equal(useVoiceSettingsStore.getInitialState().micTarget, "asr");
});

test("an older saved settings blob without the field stays on ASR", async () => {
  await rehydrateWith({ dictationEngine: "model" });
  assert.equal(useVoiceSettingsStore.getState().micTarget, "asr");
});

test("a saved model pick survives a reload, junk falls back to ASR", async () => {
  await rehydrateWith({ micTarget: "model" });
  assert.equal(useVoiceSettingsStore.getState().micTarget, "model");
  await rehydrateWith({ micTarget: "nonsense" });
  assert.equal(useVoiceSettingsStore.getState().micTarget, "asr");
});

test("ASR never sends audio to the model, even one that hears it", () => {
  assert.equal(micSendsAudioToModel("asr", true), false);
});

test("model mode needs a loaded model with audio input", () => {
  assert.equal(micSendsAudioToModel("model", true), true);
  assert.equal(micSendsAudioToModel("model", false), false);
});

test("the settings option is disabled without an audio-input model", () => {
  const source = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(
    source,
    /value="model"\s+data-testid="mic-target-model"\s+disabled=\{!modelHearsAudio && micTarget !== "model"\}/,
  );
  assert.match(source, /\?\.hasAudioInput === true/);
});

test("Compare and the adapter route only through the gate", () => {
  const source = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  assert.match(source, /if \(chatId === null \|\| !audioClipTarget\) return false;/);
  assert.match(
    source,
    /micSendsAudioToModel\(micTarget, activeModel\?\.hasAudioInput === true\)/,
  );
  assert.match(
    source,
    /if \(clipTarget && micRecordsForModel\(this\.chatId\)\) \{[\s\S]*?listenForAudioClip\(clipTarget\)/,
  );
});

test("a recorded clip is a WAV named for the attachment chip", () => {
  assert.equal(
    micClipFilename(new Date("2026-11-04T18:05:09.123Z")),
    "recording-2026-11-04T18-05-09.wav",
  );
});

test("a clip marks the session and resets on the next one", () => {
  beginDictationSession();
  assert.equal(dictationProducedClip(), false);
  markDictationClip();
  assert.equal(dictationProducedClip(), true);
  beginDictationSession();
  assert.equal(dictationProducedClip(), false);
});

test("stop-and-send submits an attached clip with no new text", () => {
  const base = {
    originComposer: "a",
    currentComposer: "a",
    producedTranscript: false,
    baseText: "",
    text: "",
  };
  assert.equal(shouldSubmitDictation({ ...base, producedClip: true }), true);
  assert.equal(shouldSubmitDictation(base), false);
  assert.equal(
    shouldSubmitDictation({ ...base, producedClip: true, currentComposer: "b" }),
    false,
  );
});
