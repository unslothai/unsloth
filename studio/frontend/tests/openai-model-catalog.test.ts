// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  audioModelsFromCatalog,
  chatModelsFromCatalog,
} from "../src/features/settings/api/openai-model-catalog.ts";

test("non-chat entries never reach the chat example picker", () => {
  const models = chatModelsFromCatalog({
    data: [
      { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
      { id: "unsloth/Z-Image-Turbo-GGUF", loaded: true, quant: "Q4_K_S", task: "text-to-image" },
      { id: "Lightricks/LTX-2", loaded: false, task: "text-to-video" },
      { id: "tiny", loaded: false, task: "automatic-speech-recognition" },
      { id: "unsloth/orpheus-3b-0.1-ft-GGUF", loaded: false, quant: "Q8_0", task: "text-to-speech" },
      { id: "unsloth/Llama-3.2-1B-Instruct-GGUF", loaded: false, task: "text-generation" },
    ],
  });
  assert.deepEqual(models, [
    { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
    { id: "unsloth/Llama-3.2-1B-Instruct-GGUF", loaded: false, quant: undefined },
  ]);
});

test("older servers without a task field list every entry as before", () => {
  const models = chatModelsFromCatalog({ data: [{ id: "a", loaded: true }, { id: "b" }] });
  assert.deepEqual(models, [
    { id: "a", loaded: true, quant: undefined },
    { id: "b", loaded: false, quant: undefined },
  ]);
});

test("malformed bodies yield no models", () => {
  assert.deepEqual(chatModelsFromCatalog(null), []);
  assert.deepEqual(chatModelsFromCatalog({}), []);
  assert.deepEqual(chatModelsFromCatalog({ data: [{ id: 3 }, { id: "" }, {}] }), []);
});

test("the audio examples see only speech and transcription models", () => {
  const models = audioModelsFromCatalog({
    data: [
      { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
      { id: "unsloth/Z-Image-Turbo-GGUF", loaded: true, task: "text-to-image" },
      { id: "tiny", loaded: true, task: "automatic-speech-recognition" },
      { id: "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", loaded: false, task: "text-to-speech" },
      { id: "", task: "text-to-speech" },
    ],
  });
  assert.deepEqual(models, [
    { id: "tiny", loaded: true, task: "automatic-speech-recognition" },
    { id: "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", loaded: false, task: "text-to-speech" },
  ]);
  assert.deepEqual(audioModelsFromCatalog(null), []);
});

test("the audio examples keep the workflows a row advertises, separators included", () => {
  const models = audioModelsFromCatalog({
    data: [
      { id: "MiniMaxAI/MiniMax-Music3", loaded: true, task: "text-to-speech", audio_workflows: ["music"] },
      { id: "audio-cpp/audio.cpp-gguf/BS-RoFormer-GGUF", loaded: true, task: "audio-to-audio", audio_workflows: ["separate", 3] },
    ],
  });
  assert.deepEqual(models, [
    { id: "MiniMaxAI/MiniMax-Music3", loaded: true, task: "text-to-speech", workflows: ["music"] },
    { id: "audio-cpp/audio.cpp-gguf/BS-RoFormer-GGUF", loaded: true, task: "audio-to-audio", workflows: ["separate"] },
  ]);
});
