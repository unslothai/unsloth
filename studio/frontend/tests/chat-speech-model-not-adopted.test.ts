// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Picking a TTS voice on the Audio page used to silently become the chat model.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SpeechOnlyStatusInput,
  isIdleUnloadedStatus,
  isSpeechOnlyStatus,
} from "../src/features/chat/lib/speech-only-status.ts";

import { readText } from "./helpers/kit.ts";

const status = (fields: SpeechOnlyStatusInput): SpeechOnlyStatusInput => fields;

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

test("whisper is not speech-only", () => {
  assert.equal(
    isSpeechOnlyStatus(status({ is_audio: true, audio_type: "whisper" })),
    false,
  );
});

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
  const source = readText(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
  );
  // In tryAdoptServerActiveModel, not the shared resolver: the loaded-models indicator and
  // API monitor must keep naming the model.
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
  const hook = readText(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(
    hook,
    /const chatActiveModel =\s*statusRes\.active_model &&\s*!isSpeechOnlyStatus\(statusRes\) &&\s*!\(statusLoading && options\?\.externalChatSlotLoad\);/,
  );
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
  const events = readText("../src/lib/model-lifecycle-events.ts");
  assert.match(events, /export type ModelRuntime =[^;]*"tts"/);

  const audio = readText("../src/features/audio/hooks/use-audio-model-slot.ts");
  assert.match(audio, /runtime: "tts",/);
  const hook = readText(
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
  const hook = readText(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(hook, /subscribeResidentStatusRefresh\(\(\) => \{/);
  assert.match(
    hook,
    /void refresh\(\{ includeLoras: false, preserveIdleUnloaded: true \}\);/,
  );
});

test("the Hub does not pin a speech model as the chat checkpoint", () => {
  const hub = readText("../src/features/hub/hub-page.tsx");
  const adopt = hub.slice(
    hub.indexOf("adoptResidentModelStatus("),
    hub.indexOf("registerRefresh(", hub.indexOf("adoptResidentModelStatus(")),
  );
  assert.match(
    adopt,
    /checkpointId: isSpeechOnlyStatus\(status\)\s*\n?\s*\? null\s*\n?\s*: resolveInferenceCheckpointId\(status\),/,
  );
  // null, not an early return: the empty-slot branch must still clear the evicted pick.
  assert.doesNotMatch(adopt, /if \(isSpeechOnlyStatus\(status\)\) return/);
  assert.match(adopt, /speechOnly: isSpeechOnlyStatus\(status\),/);
});

test("a queued local thread does not adopt a speech model either", () => {
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
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
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
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
  assert.equal(adapter.match(/NON_CHAT_TASKS\.has\(/g)?.length, 2);
});
