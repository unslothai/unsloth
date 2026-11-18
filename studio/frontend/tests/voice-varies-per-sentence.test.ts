// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { voiceVariesPerSentence } from "../src/features/chat/voice/voice-models.ts";

test("voices heard changing speaker between sentences are named, steady ones are not", () => {
  for (const folder of [
    "OmniVoice-GGUF",
    "Breeze-TTS-2-GGUF",
    "MOSS-TTS-Nano-100M-GGUF",
    "VoxCPM1-GGUF",
  ]) {
    assert.ok(
      voiceVariesPerSentence(`audio-cpp/audio.cpp-gguf/${folder}`),
      folder,
    );
  }
  for (const folder of [
    "Kokoro-82M-GGUF",
    "PocketTTS-GGUF",
    "Supertonic-3-GGUF",
    "VoxCPM2-GGUF",
  ]) {
    assert.ok(
      !voiceVariesPerSentence(`audio-cpp/audio.cpp-gguf/${folder}`),
      folder,
    );
  }
  assert.ok(!voiceVariesPerSentence(""));
  assert.ok(!voiceVariesPerSentence(null));
});
