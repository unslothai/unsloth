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
import {
  clipStillAttached,
  shouldSubmitDictation,
} from "../src/features/chat/utils/dictation-send.ts";
import {
  micClipFilename,
  MIN_MIC_CLIP_BYTES,
  micClipRoom,
  micSendsAudioToModel,
  pickAudioForModel,
} from "../src/features/chat/utils/mic-target.ts";
import { AUDIO_ATTACHMENT_ACCEPT, MAX_AUDIO_SIZE } from "../src/lib/audio-utils.ts";
import * as sttModelCatalog from "../src/features/settings/stores/stt-model-catalog.ts";
import * as sttContext from "../src/features/settings/stores/stt-context.ts";
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
    "./stt-context": sttContext,
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
  );  assert.match(source, /discardClipOnRelease\(session, clipTarget\.onRelease\)/);
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const register = thread.slice(thread.indexOf("setMicAudioClipTarget({"));
  assert.match(register.slice(0, 700), /\[aui, composerIdentity\]/);
});

test("a recorded clip is a WAV named for the attachment chip", () => {
  assert.equal(
    micClipFilename(new Date("2026-11-04T18:05:09.123Z")),
    "recording-2026-11-04T18-05-09.wav",
  );
});

test("a clip marks the session and resets on the next one", () => {
  beginDictationSession();
  assert.equal(dictationProducedClip(), null);
  const clip = new File([new Uint8Array(4)], "recording.wav", { type: "audio/wav" });
  markDictationClip(clip);
  assert.equal(dictationProducedClip(), clip);
  beginDictationSession();
  assert.equal(dictationProducedClip(), null);
});

test("a held stop-and-send drops the clip once it is removed from the composer", () => {
  const clip = new File([new Uint8Array(4)], "recording.wav", { type: "audio/wav" });
  const other = new File([new Uint8Array(4)], "recording.wav", { type: "audio/wav" });
  assert.equal(clipStillAttached(clip, [{ file: clip }]), true);
  assert.equal(clipStillAttached(clip, [{ file: other }, {}]), false);
  assert.equal(clipStillAttached(clip, []), false);
  assert.equal(clipStillAttached(null, [{ file: clip }]), false);
  const base = { originComposer: "a", currentComposer: "a", producedTranscript: false, baseText: "", text: "hi" };
  assert.equal(shouldSubmitDictation({ ...base, producedClip: clipStillAttached(clip, []) }), false);
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /producedClip: clipStillAttached\(\s*dictationProducedClip\(\),\s*aui\.composer\(\)\.getState\(\)\.attachments,\s*\)/,
  );
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

// Every route into the store must carry hasAudioInput: list row, load response, status adoption.
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

test("over plain HTTP the model mic picks recording files and attaches them", async () => {
  const listeners = new Map<string, () => void>();
  const input = {
    type: "",
    accept: "",
    multiple: false,
    hidden: false,
    files: null as File[] | null,
    removed: false,
    clicked: false,
    addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
    remove() {
      this.removed = true;
    },
    click() {
      this.clicked = true;
    },
  };
  let appended: unknown = null;
  const doc = {
    createElement: () => input,
    body: { append: (node: unknown) => (appended = node) },
  } as unknown as Document;
  const added: File[] = [];
  const clip = new File([new Uint8Array(4)], "note.m4a", { type: "" });
  const refused = new File([new Uint8Array(4)], "big.wav", { type: "audio/wav" });
  pickAudioForModel(async (file) => {
    added.push(file);
    if (file === refused) throw new Error("too big");
  }, doc);
  assert.equal(input.type, "file");
  assert.equal(input.accept, AUDIO_ATTACHMENT_ACCEPT);
  assert.equal(appended, input);
  assert.equal(input.clicked, true);
  input.files = [clip, refused];
  listeners.get("change")?.();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(added, [clip, refused]);
  assert.equal(input.removed, true);
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const fallback = thread.slice(thread.indexOf('currentDictationEntryMode() === "recording-file"'));
  assert.match(
    fallback.slice(0, 400),
    /if \(micRecordsForModel\(\)\) \{\s*pickAudioForModel\(\(file\) => aui\.composer\(\)\.addAttachment\(file\)\);\s*\} else \{\s*audioUpload\.openDialog\(\);/,
  );
});
