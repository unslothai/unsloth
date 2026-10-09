// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the voice picker offers no cached CSM GGUF, which /voice/load refuses", () => {
  const list = readSrc("features/chat/chat-page.tsx").match(
    /const TTS_REPO_KEYWORDS = \[([^\]]*)\]/,
  );
  assert.ok(list, "TTS_REPO_KEYWORDS not found");
  assert.doesNotMatch(list[1], /"csm"/);
});

test("the speaker picker shows for any SNAC voice, not only ids that say orpheus", () => {
  const picker = readSrc("components/assistant-ui/voice-name-picker.tsx");
  assert.match(
    picker,
    /const isSnac = codec\s*\? codec === "snac"\s*: voiceModelId\.toLowerCase\(\)\.includes\("orpheus"\);/,
  );
  // chat-page hands it the catalog's detected audio type for the selected voice.
  assert.match(
    readSrc("features/chat/chat-page.tsx"),
    /<VoiceNamePicker\s+codec=\{\s*ttsModels\.find\(\(m\) => m\.id === selectedVoiceModelId\)\s*\?\.audioType \?\? null\s*\}/,
  );
});

test("a cached voice keeps its detected codec, so a renamed SNAC voice gets the speaker picker", () => {
  const cached = readSrc("features/chat/chat-page.tsx").match(
    /setCachedGgufs\(([\s\S]*?)\);\n/,
  );
  assert.ok(cached, "setCachedGgufs call not found");
  assert.match(cached[1], /\.map\(\(c\) => \(\{[\s\S]*audioType: c\.audio_type \?\? null/);
});

test("the dictation mic resumes its AudioContext before expecting frames", () => {
  const adapter = readSrc(
    "features/chat/adapters/studio-whisper-dictation-adapter.ts",
  );
  assert.match(adapter, /audioCtx = new AudioCtx\(\);[\s\S]{0,400}await audioCtx\.resume\(\);/);
});

test("Voice is offered wherever the loop's Whisper adapter can listen, MediaRecorder or not", () => {
  const gate = readSrc("features/chat/voice/voice-engine.tsx").match(
    /export function useVoiceAvailable\(\): boolean \{([\s\S]*?)\n\}/,
  );
  assert.ok(gate, "useVoiceAvailable not found");
  assert.match(gate[1], /return StudioWhisperDictationAdapter\.isSupported\(\);/);
  // Active voice mode dispatches to that adapter.
  assert.match(
    readSrc("features/chat/adapters/studio-dictation-adapter.tsx"),
    /getVoiceMode\(\) === "active"\) \{\s*if \(StudioWhisperDictationAdapter\.isSupported\(\)\)/,
  );
});
