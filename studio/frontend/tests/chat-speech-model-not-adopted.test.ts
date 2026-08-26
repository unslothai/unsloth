// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio page loads a TTS model into the same single inference slot chat reads,
// and chat's local selection is re-derived from /status on every mount. So picking
// a voice on the Audio page silently made it the chat model, and the backend then
// answered the next chat turn by SYNTHESIZING the prompt instead of replying to it.
// Nothing refused it anywhere along the way. These pin the three places that now do.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type SpeechOnlyStatusInput,
  isSpeechOnlyStatus,
} from "../src/features/chat/lib/speech-only-status.ts";

const status = (fields: SpeechOnlyStatusInput): SpeechOnlyStatusInput => fields;

const readSource = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

test("a resident TTS model reads as speech-only", () => {
  for (const audioType of ["snac", "csm", "bicodec", "dac"]) {
    assert.equal(
      isSpeechOnlyStatus(status({ is_audio: true, audio_type: audioType })),
      true,
      audioType,
    );
  }
});

test("an ordinary chat model is not speech-only", () => {
  assert.equal(isSpeechOnlyStatus(status({})), false);
  assert.equal(isSpeechOnlyStatus(status({ is_audio: false })), false);
});

// Whisper answers with transcripts, not speech, and the chat route branches on it
// separately. It is the STT half of the Audio page and has its own guards.
test("whisper is not speech-only", () => {
  assert.equal(
    isSpeechOnlyStatus(status({ is_audio: true, audio_type: "whisper" })),
    false,
  );
});

// Gemma 3n takes audio IN and answers in text: an ordinary chat model that happens
// to listen. Treating it as speech-only would lock a real chat model out of chat.
test("an audio-input chat model is not speech-only", () => {
  assert.equal(
    isSpeechOnlyStatus(status({ is_audio: true, audio_type: "audio_vlm" })),
    false,
  );
});

test("chat does not adopt the server's model when it only speaks", () => {
  const source = readSource(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
  );
  // Guarded in tryAdoptServerActiveModel, not in resolveInferenceCheckpointId:
  // the loaded-models indicator and the API monitor share that resolver and must
  // go on naming a resident TTS model.
  const adopt = source.slice(
    source.indexOf("export async function tryAdoptServerActiveModel"),
  );
  assert.match(adopt, /isSpeechOnlyStatus\(status\)/);
  const resolver = source.slice(
    source.indexOf("export function resolveInferenceCheckpointId"),
    source.indexOf("function ensureActiveModelInStoreList"),
  );
  assert.doesNotMatch(
    resolver,
    /isSpeechOnlyStatus/,
    "the shared resolver must keep answering for a resident TTS model",
  );
});

test("the mount-time status sync treats a speech model as an empty slot", () => {
  const hook = readSource(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(
    hook,
    /const chatActiveModel =\s*\n?\s*statusRes\.active_model && !isSpeechOnlyStatus\(statusRes\);/,
  );
  // Both edges, or the eviction branch would stop clearing a stale pick.
  assert.match(hook, /if \(chatActiveModel && !isExternalSelectionActive\)/);
  assert.match(
    hook,
    /\} else if \(!chatActiveModel && !isExternalSelectionActive\)/,
  );
});

test("a TTS load announces its own runtime so chat re-reads the slot", () => {
  const events = readSource("../src/lib/model-lifecycle-events.ts");
  assert.match(events, /export type ModelRuntime =[^;]*"tts"/);

  // Chat ignores its own loads when reconciling, so a TTS load announced as
  // "chat" left chat naming a model the Audio page had already evicted.
  const audio = readSource("../src/features/audio/audio-page.tsx");
  assert.match(audio, /runtime: "tts",/);
  const hook = readSource(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(
    hook,
    /if \(runtime === "chat" \|\| runtime === "stt"\) return;/,
  );
  assert.doesNotMatch(hook, /runtime === "tts"\) return;/);
});

// tryAdoptServerActiveModel is not the only door into params.checkpoint. These two
// call sites resolve a resident model into the chat store on their own, so a guard
// that lived only in the adopt helper still let a speech model become the chat pick.

test("the Hub does not pin a speech model as the chat checkpoint", () => {
  const hub = readSource("../src/features/hub/hub-page.tsx");
  const adopt = hub.slice(
    hub.indexOf("adoptResidentModelStatus("),
    hub.indexOf("registerRefresh(", hub.indexOf("adoptResidentModelStatus(")),
  );
  assert.match(
    adopt,
    /checkpointId: isSpeechOnlyStatus\(status\)\s*\n?\s*\? null\s*\n?\s*: resolveInferenceCheckpointId\(status\),/,
  );
  // null, not an early return: the empty-slot branch is what clears the pick the
  // Audio load evicted, and skipping the call would leave it pointing at a 400.
  assert.doesNotMatch(adopt, /if \(isSpeechOnlyStatus\(status\)\) return/);
});

test("a queued local thread does not adopt a speech model either", () => {
  const adapter = readSource("../src/features/chat/api/chat-adapter.ts");
  const queued = adapter.slice(
    adapter.indexOf("async function resolveQueuedEmptyLocalModel"),
    adapter.indexOf("export function createOpenAIStreamAdapter"),
  );
  assert.match(
    queued,
    /const checkpoint = isSpeechOnlyStatus\(status\)\s*\n?\s*\? null\s*\n?\s*: resolveInferenceCheckpointId\(status\);/,
  );
});

test("the auto-load sweep skips every task chat cannot answer", () => {
  const adapter = readSource("../src/features/chat/api/chat-adapter.ts");
  const set = adapter.slice(
    adapter.indexOf("const NON_CHAT_TASKS"),
    adapter.indexOf("]);", adapter.indexOf("const NON_CHAT_TASKS")),
  );
  for (const task of [
    "text-to-image",
    "text-to-video",
    "image-diffusion-unsupported",
    "text-to-speech",
    "text-to-audio",
    "audio-to-audio",
    "automatic-speech-recognition",
  ]) {
    assert.ok(set.includes(`"${task}"`), task);
  }
  // Both the cached-repo and the on-disk filter, or one inventory still offers them.
  assert.equal(adapter.match(/NON_CHAT_TASKS\.has\(/g)?.length, 2);
});
