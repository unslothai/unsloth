// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice mode listens and speaks with audio.cpp models; a pick left over from the
// llama-server voice slot must not survive into a load that cannot work.

import assert from "node:assert/strict";
import test from "node:test";

import {
  VOICE_DEFAULT_STT_MODEL,
  VOICE_DEFAULT_TTS_MODEL,
  VOICE_TTS_MODELS,
  migrateVoiceTtsModel,
  voiceIdServes,
  voiceTtsCanClone,
} from "../src/features/chat/voice/voice-models.ts";
import { AUDIO_CPP_STT_MODELS } from "../src/features/settings/stores/stt-model-catalog.ts";

const POCKET = "audio-cpp/audio.cpp-gguf/PocketTTS-GGUF";

test("voice mode defaults to audio.cpp on both sides", () => {
  assert.equal(VOICE_DEFAULT_TTS_MODEL, "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF");
  assert.equal(VOICE_TTS_MODELS[0], VOICE_DEFAULT_TTS_MODEL);
  assert.ok(AUDIO_CPP_STT_MODELS.has(VOICE_DEFAULT_STT_MODEL));
});

test("only voices that speak plain text are offered", () => {
  assert.ok(VOICE_TTS_MODELS.includes(POCKET));
  // Clone-only and voice-conversion models need a reference clip for every line.
  assert.ok(!VOICE_TTS_MODELS.includes("audio-cpp/audio.cpp-gguf/Chatterbox-GGUF"));
  assert.ok(!VOICE_TTS_MODELS.includes("audio-cpp/audio.cpp-gguf/RVC-GGUF"));
  assert.equal(new Set(VOICE_TTS_MODELS).size, VOICE_TTS_MODELS.length);
});

test("a saved Audio voice is offered only for a voice that can clone", () => {
  assert.equal(voiceTtsCanClone(VOICE_DEFAULT_TTS_MODEL), false);
  assert.equal(voiceTtsCanClone(POCKET), true);
});

test("an Orpheus pick from the old voice slot falls back to Kokoro", () => {
  assert.equal(migrateVoiceTtsModel("unsloth/orpheus-3b-0.1-ft-GGUF"), VOICE_DEFAULT_TTS_MODEL);
  assert.equal(migrateVoiceTtsModel(""), VOICE_DEFAULT_TTS_MODEL);
  assert.equal(migrateVoiceTtsModel(null), VOICE_DEFAULT_TTS_MODEL);
  assert.equal(migrateVoiceTtsModel("audio-cpp/audio.cpp-gguf/RVC-GGUF"), VOICE_DEFAULT_TTS_MODEL);
  assert.equal(migrateVoiceTtsModel(POCKET), POCKET);
});

test("the serving list matches the voice by load path or repo id", () => {
  assert.ok(voiceIdServes(VOICE_DEFAULT_TTS_MODEL, VOICE_DEFAULT_TTS_MODEL));
  assert.ok(
    voiceIdServes(
      String.raw`C:\Users\me\.cache\huggingface\hub\snap\Kokoro-82M-GGUF`,
      VOICE_DEFAULT_TTS_MODEL,
    ),
  );
  assert.ok(!voiceIdServes("unsloth/Qwen3-1.7B-GGUF", VOICE_DEFAULT_TTS_MODEL));
  assert.ok(!voiceIdServes("", VOICE_DEFAULT_TTS_MODEL));
});
