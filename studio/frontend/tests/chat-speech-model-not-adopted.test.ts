// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio page loads a TTS model into the single slot chat reads, and chat's local
// selection is re-derived from /status on every mount, so picking a voice silently made
// it the chat model and the next turn came back as SYNTHESIZED speech. Nothing refused
// it anywhere. These pin the places that now do.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type SpeechOnlyStatusInput,
  isIdleUnloadedStatus,
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

// Whisper answers with transcripts, and the chat route branches on it separately.
test("whisper is not speech-only", () => {
  assert.equal(
    isSpeechOnlyStatus(status({ is_audio: true, audio_type: "whisper" })),
    false,
  );
});

// Gemma 3n takes audio IN and answers in text: treating it as speech-only would lock a
// real chat model out of chat.
test("an audio-input chat model is not speech-only", () => {
  assert.equal(
    isSpeechOnlyStatus(status({ is_audio: true, audio_type: "audio_vlm" })),
    false,
  );
});

test("only an armed idle unload preserves an empty resident slot", () => {
  assert.equal(isIdleUnloadedStatus(status({ active_model: null }), true), true);
  assert.equal(isIdleUnloadedStatus(status({ active_model: null }), false), false);
  assert.equal(
    isIdleUnloadedStatus(
      status({ active_model: "my-voice", is_audio: true, audio_type: "snac" }),
      true,
    ),
    false,
  );
});

test("chat does not adopt the server's model when it only speaks", () => {
  const source = readSource(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
  );
  // In tryAdoptServerActiveModel, not resolveInferenceCheckpointId: the loaded-models
  // indicator and the API monitor share that resolver and must go on naming one.
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
    /const chatActiveModel =\s*statusRes\.active_model &&\s*!isSpeechOnlyStatus\(statusRes\) &&\s*!\(statusLoading && options\?\.externalChatSlotLoad\);/,
  );
  // Both edges, or the eviction branch would stop clearing a stale pick.
  assert.match(
    hook,
    /if \(\s*chatActiveModel &&\s*!isExternalSelectionActive &&\s*!selectionChanged\s*\)/,
  );
  assert.match(
    hook,
    /\} else if \(\s*!chatActiveModel &&\s*!isExternalSelectionActive &&\s*!selectionChanged\s*\)/,
  );
  assert.match(
    hook,
    /\(wasResident \|\| isSpeechOnlyStatus\(statusRes\)\)[\s\S]*selectedCheckpoint[\s\S]*!modelLoading/,
    "the first speech-only status must clear a persisted Chat pick",
  );
});

test("a TTS load announces its own runtime so chat re-reads the slot", () => {
  const events = readSource("../src/lib/model-lifecycle-events.ts");
  assert.match(events, /export type ModelRuntime =[^;]*"tts"/);

  // Chat ignores its own loads when reconciling, so a TTS load announced as "chat" left
  // chat naming a model the Audio page had evicted.
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
  assert.match(
    hook,
    /externalChatSlotLoad: runtime === "tts"/,
    "the shared loading lease must not hide TTS eviction from Chat",
  );
  assert.match(
    hook,
    /\(!modelLoading \|\| options\?\.externalChatSlotLoad\)/,
    "the TTS settle event runs before Audio releases the shared loading lease",
  );
});

test("chat re-reads status when a different tab returns to the foreground", () => {
  const hook = readSource(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(hook, /subscribeResidentStatusRefresh\(\(\) => \{/);
  assert.match(
    hook,
    /void refresh\(\{ includeLoras: false, preserveIdleUnloaded: true \}\);/,
  );
});

// tryAdoptServerActiveModel is not the only door into params.checkpoint: these two call
// sites resolve a resident model into the chat store on their own.

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
  // null, not an early return: the empty-slot branch clears the pick the Audio load
  // evicted, and skipping the call would leave it pointing at a 400.
  assert.doesNotMatch(adopt, /if \(isSpeechOnlyStatus\(status\)\) return/);
  // And say WHY it is null: a bare null reads as an idle eviction, which the helper
  // deliberately keeps. hub-resident-status pins that branch.
  assert.match(adopt, /speechOnly: isSpeechOnlyStatus\(status\),/);
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
