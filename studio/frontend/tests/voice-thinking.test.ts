// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  VOICE_CHAT_MODEL_VOICE,
  VOICE_DEFAULT_TTS_MODEL,
  cutAbortsSpeechRequest,
  reasoningForTurn,
  } from "../src/features/chat/voice/voice-models.ts";

test("a voice turn thinks only when voice mode was asked to", () => {
  const typed = { voiceModeOn: false, voiceThinking: false };
  assert.equal(reasoningForTurn({ enabled: true, ...typed }), true);
  assert.equal(reasoningForTurn({ enabled: false, ...typed }), false);
  assert.equal(reasoningForTurn({ enabled: false, voiceModeOn: false, voiceThinking: true }), false);

  const spoken = { voiceModeOn: true };
  assert.equal(reasoningForTurn({ enabled: true, voiceThinking: false, ...spoken }), false);
  assert.equal(reasoningForTurn({ enabled: true, voiceThinking: true, ...spoken }), true);
  assert.equal(reasoningForTurn({ enabled: false, voiceThinking: true, ...spoken }), false);
});

test("cutting a reply leaves an audio.cpp sentence to finish", () => {
  // Aborting would stop the voice's process, so the next reply waits for a reload.
  assert.equal(cutAbortsSpeechRequest(VOICE_DEFAULT_TTS_MODEL), false);
  assert.equal(cutAbortsSpeechRequest(null), false);
  // The chat model's own voice is a generation the chat server cancels in place.
  assert.equal(cutAbortsSpeechRequest(VOICE_CHAT_MODEL_VOICE), true);
});
