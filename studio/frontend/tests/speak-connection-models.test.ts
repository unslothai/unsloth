// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { connectionSpeechModels } from "../src/features/audio/speak-connection-models.ts";

test("an OpenAI connection offers its speech models, not its chat models", () => {
  assert.deepEqual(
    connectionSpeechModels({ providerType: "openai", models: ["gpt-5.5"] }),
    ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
  );
});

test("a self-hosted connection offers the models it lists; hosted chat providers none", () => {
  assert.deepEqual(
    connectionSpeechModels({ providerType: "custom", models: ["kokoro"] }),
    ["kokoro"],
  );
  assert.deepEqual(
    connectionSpeechModels({
      providerType: "custom",
      models: [
        "speaches-ai/Kokoro-82M-v1.0-ONNX",
        "Systran/faster-whisper-tiny",
        "nvidia/parakeet-tdt-0.6b-v3",
        "Qwen/Qwen3-ASR-0.6B",
      ],
    }),
    ["speaches-ai/Kokoro-82M-v1.0-ONNX"],
  );
  assert.deepEqual(
    connectionSpeechModels({ providerType: "mistral", models: ["mistral-large"] }),
    [],
  );
});

test("an OpenAI-typed Azure connection offers its deployment names", () => {
  assert.deepEqual(
    connectionSpeechModels({
      providerType: "openai",
      baseUrl: "https://studio.openai.azure.com/openai/v1",
      models: ["speech-prod"],
    }),
    ["speech-prod"],
  );
  assert.deepEqual(
    connectionSpeechModels({
      providerType: "openai",
      baseUrl: "https://api.openai.com/v1",
      models: ["gpt-5.5"],
    }),
    ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
  );
});
