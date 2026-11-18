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
