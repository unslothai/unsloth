// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  chatModelOwnsItsVoice,
  isSpeechLLMCheckpoint,
} from "../src/features/chat/voice/speech-llm.ts";

test("a renamed fine-tune is a speech model by its detected codec, not its name", () => {
  // No keyword in the id: the name pattern alone said "text model".
  assert.equal(isSpeechLLMCheckpoint("me/my-finetune-v3"), false);
  assert.equal(isSpeechLLMCheckpoint("me/my-finetune-v3", "snac"), true);
  assert.equal(isSpeechLLMCheckpoint("me/my-finetune-v3", "csm"), true);
});

test("the name pattern still catches a checkpoint with no inventory record", () => {
  assert.equal(isSpeechLLMCheckpoint("unsloth/orpheus-3b-0.1-ft", null), true);
  assert.equal(isSpeechLLMCheckpoint("unsloth/csm-1b", undefined), true);
  assert.equal(isSpeechLLMCheckpoint("unsloth/Qwen3-4B", null), false);
});

test("a non-speech audio type does not make a text model own its voice", () => {
  // Speech INPUT (whisper) and audio VLMs are not TTS codecs.
  assert.equal(isSpeechLLMCheckpoint("unsloth/Qwen3-4B", "whisper"), false);
  assert.equal(isSpeechLLMCheckpoint("unsloth/Qwen3-4B", "audio_vlm"), false);
});

test("chatModelOwnsItsVoice reads the active checkpoint's record from the store", () => {
  const models = [
    { id: "me/my-finetune-v3", audioType: "snac" },
    { id: "unsloth/Qwen3-4B", audioType: null },
  ];
  assert.equal(
    chatModelOwnsItsVoice({
      params: { checkpoint: "me/my-finetune-v3" },
      models,
    }),
    true,
  );
  assert.equal(
    chatModelOwnsItsVoice({
      params: { checkpoint: "unsloth/Qwen3-4B" },
      models,
    }),
    false,
  );
  // Not in the inventory yet: falls back to the name.
  assert.equal(
    chatModelOwnsItsVoice({
      params: { checkpoint: "unsloth/orpheus-3b-0.1-ft" },
      models,
    }),
    true,
  );
  assert.equal(
    chatModelOwnsItsVoice({ params: { checkpoint: "" }, models }),
    false,
  );
});
