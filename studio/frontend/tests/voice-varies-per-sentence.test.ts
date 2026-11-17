// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

test("the panel says a new voice is drawn each session until a saved voice is picked", () => {
  const selector = readFileSync(
    new URL(
      "../src/components/assistant-ui/voice-model-selector.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    selector,
    /newVoiceEachSession: !savedVoiceId &&\s*\(spec\.family === "omnivoice" \|\| voiceVariesPerSentence\(modelId\)\),/,
  );
});

test("voice mode keeps the first sentence's voice for such a model", () => {
  const player = readFileSync(
    new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    player,
    /!voiceId && !selectedVoiceSpeaker && voiceVariesPerSentence\(selectedVoiceModelId\)\s*\?\s*await withVoicePin\(model, speak, async \(\) => \{[\s\S]*?VOICE_REFERENCE_TEXT/,
  );
  const store = readFileSync(
    new URL(
      "../src/features/chat/stores/chat-runtime-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(store, /if \(voiceMode === "off"\) clearVoicePin\(\);/);
});
