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
  MIN_MIC_CLIP_BYTES,
  micClipRoom,
  micSendsAudioToModel,
} from "../src/features/chat/utils/mic-target.ts";
import { MAX_AUDIO_SIZE } from "../src/lib/audio-utils.ts";
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
    /if \(clipTarget && micRecordsForModel\(this\.chatId\)\) \{[\s\S]*?listenForAudioClip\(clipTarget\.add, room\);\s*discardClipWhenModelChanges\(session, useChatRuntimeStore\)/,
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

// The gate keys on hasAudioInput alone, so every way a model reaches the store must carry it:
// the models list row (GGUF, safetensors, MLX alike), the load response, and the status adoption.
test("the gate is family-agnostic: no model names in the routing", () => {
  const gate = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  const routing = gate.slice(gate.indexOf("export function micRecordsForModel"));
  assert.doesNotMatch(routing.slice(0, routing.indexOf("\n}\n")), /lfm|gemma|qwen|omni|audio_type/i);
});

test("every route into the chat store carries has_audio_input", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const row = runtime.slice(runtime.indexOf("function toChatModelRow"));
  assert.match(row.slice(0, 900), /hasAudioInput: Boolean\(model\.has_audio_input\)/);
  const sync = runtime.slice(runtime.indexOf("export function syncModelCapabilities"));
  assert.match(sync.slice(0, 2000), /hasAudioInput: Boolean\(resp\.has_audio_input\)/);
  const status = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
  assert.match(status, /hasAudioInput: status\.has_audio_input \?\? false/);
});

test("a recording gets the whole audio limit when nothing else is attached", () => {
  assert.deepEqual(micClipRoom([], 8, false), { maxBytes: MAX_AUDIO_SIZE });
  // Images do not count against the audio budget.
  const image = { name: "a.png", type: "image/png", size: 5_000_000 };
  assert.deepEqual(micClipRoom([image], 8, false), { maxBytes: MAX_AUDIO_SIZE });
});

test("a recording is capped by the audio already in the composer", () => {
  const clip = { name: "a.wav", type: "audio/wav", size: 10_000_000 };
  const m4a = { name: "b.m4a", type: "", size: 5_000_000 };
  assert.deepEqual(micClipRoom([clip, m4a], 8, false), {
    maxBytes: MAX_AUDIO_SIZE - 15_000_000,
  });
});

test("no room is refused with the attachment adapter's reason", () => {
  const clip = { name: "a.wav", type: "audio/wav", size: 1_000 };
  assert.match(
    (micClipRoom([clip], 1, false) as { reason: string }).reason,
    /one audio file per message/,
  );
  const full = { name: "a.wav", type: "audio/wav", size: MAX_AUDIO_SIZE };
  assert.match(
    (micClipRoom([full], 8, false) as { reason: string }).reason,
    /per-message limit/,
  );
  assert.match(
    (micClipRoom([], 8, true) as { reason: string }).reason,
    /staged audio clip/,
  );
});

test("a room too small for a usable clip is refused", () => {
  const nearlyFull = {
    name: "a.wav",
    type: "audio/wav",
    size: MAX_AUDIO_SIZE - MIN_MIC_CLIP_BYTES + 1,
  };
  assert.match(
    (micClipRoom([nearlyFull], 8, false) as { reason: string }).reason,
    /per-message limit/,
  );
  const justFits = { ...nearlyFull, size: MAX_AUDIO_SIZE - MIN_MIC_CLIP_BYTES };
  assert.deepEqual(micClipRoom([justFits], 8, false), {
    maxBytes: MIN_MIC_CLIP_BYTES,
  });
});

test("the audio attachment adapter refuses audio while a mic clip records", () => {
  const adapter = readSrc("features/chat/audio-attachment-adapter.ts");
  assert.match(
    adapter,
    /if \(micClipRecording\(\)\) \{[\s\S]*?throw new Error\(recordingReason\)/,
  );
});

test("the model clip path gates on PCM support, not MediaRecorder", () => {
  const source = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  assert.match(
    source,
    /if \(clipTarget && micRecordsForModel\(this\.chatId\)\) \{\s*if \(!PcmRecorder\.isSupported\(\)\)/,
  );
});
