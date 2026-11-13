// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  VOICE_CHAT_MODEL_VOICE,
  VOICE_DEFAULT_TTS_MODEL,
  voiceHearsAudio,
  wantsSpokenReply,
} from "../src/features/chat/voice/voice-models.ts";
import { readSrc } from "./helpers/kit.ts";

const state = (
  model: { id: string; hasAudioInput?: boolean },
  selectedVoiceModelId = VOICE_DEFAULT_TTS_MODEL,
  modelLoading = false,
) => ({
  models: [model],
  params: { checkpoint: model.id },
  selectedVoiceModelId,
  modelLoading,
});

// One row per audio-input family voice mode should hear; only the capability flag differs.
const AUDIO_MODELS = [
  "unsloth/gemma-4-E2B-it-GGUF",
  "unsloth/gemma-3n-E2B-it",
  "unsloth/gemma-4-E4B-it",
  "LiquidAI/LFM3-Omni-2B",
  "LiquidAI/LFM3-Omni-2B-GGUF",
  "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
  "someone/any-model-with-an-audio-mmproj-GGUF",
];

test("ASR stays the default: no model hears the turn unless the setting says so", () => {
  for (const id of AUDIO_MODELS) {
    assert.equal(voiceHearsAudio(state({ id, hasAudioInput: true }), "asr"), false, id);
  }
});

test("with the mic set to send audio, every audio-input model hears the turn", () => {
  for (const id of AUDIO_MODELS) {
    assert.equal(voiceHearsAudio(state({ id, hasAudioInput: true }), "model"), true, id);
  }
});

test("a model without audio input never hears it, whatever the setting or pick", () => {
  const text = { id: "unsloth/Qwen3-4B-GGUF", hasAudioInput: false };
  assert.equal(voiceHearsAudio(state(text), "model"), false);
  assert.equal(voiceHearsAudio(state(text, VOICE_CHAT_MODEL_VOICE), "asr"), false);
  assert.equal(voiceHearsAudio(state({ id: "x" }), "model"), false);
});

test("picking the chat model's own voice is itself the opt-in to be heard", () => {
  const lfm3 = { id: "LiquidAI/LFM3-Omni-2B-GGUF", hasAudioInput: true };
  assert.equal(voiceHearsAudio(state(lfm3, VOICE_CHAT_MODEL_VOICE), "asr"), true);
});

test("a model still loading hears nothing", () => {
  const gemma = { id: "unsloth/gemma-4-E2B-it-GGUF", hasAudioInput: true };
  assert.equal(voiceHearsAudio(state(gemma, VOICE_DEFAULT_TTS_MODEL, true), "model"), false);
});

test("speech comes back from the model only where it speaks and is the picked voice", () => {
  const base = { speaks: true, turnHasAudio: true, voiceModeOn: true };
  assert.equal(wantsSpokenReply({ ...base, selectedVoiceModelId: VOICE_CHAT_MODEL_VOICE }), true);
  // Voice mode with an audio.cpp voice: that voice reads the text, the model writes only text.
  assert.equal(wantsSpokenReply({ ...base, selectedVoiceModelId: VOICE_DEFAULT_TTS_MODEL }), false);
  // A model that hears but cannot speak (Gemma 4) never asks for speech.
  assert.equal(
    wantsSpokenReply({ ...base, speaks: false, selectedVoiceModelId: VOICE_CHAT_MODEL_VOICE }),
    false,
  );
  // Outside voice mode an audio turn to a speaking model is answered aloud, as before.
  assert.equal(
    wantsSpokenReply({ ...base, voiceModeOn: false, selectedVoiceModelId: VOICE_DEFAULT_TTS_MODEL }),
    true,
  );
  assert.equal(
    wantsSpokenReply({ ...base, turnHasAudio: false, selectedVoiceModelId: VOICE_CHAT_MODEL_VOICE }),
    false,
  );
});

test("the loop keeps and sends the turn's speech through the hearing gate, not the speaking one", () => {
  const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");
  assert.match(loop, /transcript &&\s*epoch === heldEpoch &&\s*voiceHearsAudio\(rt, useVoiceSettingsStore\.getState\(\)\.micTarget\)\s*\) \{\s*turnAudio\.add/);
  // An emptied edit drops its speech before this turn's is kept.
  assert.match(loop, /onCleared: \(\) => turnAudio\.clear\(\)/);
  assert.match(loop, /const editing = !heldEdit\.autoSend\(\);[\s\S]{0,400}turnAudio\.add\(merged, sampleRate\)/);
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /const turnAudio = voiceHearsAudio\(/);
  assert.doesNotMatch(engine, /const turnAudio = speaksWithChatModel\(\)/);
  // The text TTS is skipped only for a reply the chat model speaks itself.
  assert.match(engine, /spokenRunRef\.current =\s*speaksWithChatModel\(\) && lastUserTurnHasAudio/);
});

test("the request asks for speech only through wantsSpokenReply", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /\.\.\.\(wantsSpokenReply\(\{[\s\S]{0,300}\}\)\s*\? \{ modalities: \["text", "audio"\] \}/);
  assert.doesNotMatch(adapter, /activeModel\?\.speaks && turnAudioBase64/);
});

test("the gates name no model family", () => {
  const models = readSrc("features/chat/voice/voice-models.ts");
  for (const fn of ["export function voiceHearsAudio", "export function wantsSpokenReply"]) {
    const body = models.slice(models.indexOf(fn), models.indexOf("\n}\n", models.indexOf(fn)));
    assert.doesNotMatch(body, /lfm|gemma|qwen|omni|liquid/i, fn);
  }
});

test("speech of a turn cleared while STT ran never reaches the next turn", () => {
  const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");
  const add = loop.indexOf("turnAudio.add(merged, sampleRate)");
  const guard = loop.lastIndexOf("if (", add);
  assert.match(loop.slice(guard, add), /epoch === heldEpoch/);
});

test("voice mode takes the mic before the model-bound clip recorder", () => {
  const adapter = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  const body = adapter.slice(adapter.indexOf("private createSession()"));
  const voice = body.indexOf('getVoiceMode() === "active"');
  const clip = body.indexOf("micRecordsForModel(this.chatId)");
  assert.ok(voice > 0 && clip > 0 && voice < clip);
});

test("a session dropped by a clear re-arms the listener", () => {
  const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");
  const drop = loop.slice(loop.indexOf('why: "cleared"'));
  assert.match(drop.slice(0, 300), /finish\("stopped"\);[\s\S]*?requestVoiceResume\(\)[\s\S]*?return;/);
});

test("a transcript revealed across a Stop is dropped, not sent", () => {
  const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");
  const reveal = loop.slice(loop.indexOf("const mine = { shown: shown(transcript), rush: false };"));
  const submit = reveal.indexOf("requestVoiceSubmit(transcript)");
  const guard = reveal.indexOf("if (epoch !== heldEpoch) {");
  assert.ok(guard > 0 && guard < submit);
});

test("parked turns flush only once the voice actually loaded", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const effect = engine.slice(engine.indexOf("wasSlotLoadingRef.current = voiceSlotLoading;"));
  assert.match(
    effect.slice(0, 900),
    /if \(pendingForVoiceRef\.current\.length && useChatRuntimeStore\.getState\(\)\.voiceSlotLoaded\) \{[\s\S]*?submitRef\.current\(""\)/,
  );
});

test("Stop closes the open session before re-arming, as a thread reset does", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const handler = engine.slice(engine.indexOf("registerVoiceStopSpeaking(() => {"));
  const body = handler.slice(0, handler.indexOf("registerVoiceStopSpeaking(null)"));
  assert.match(body, /if \(composer\.getState\(\)\.dictation\) composer\.stopDictation\(\);\s*resumeListen\(\);/);
});
