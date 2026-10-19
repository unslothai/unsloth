// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SELF_HOSTED_TYPES = new Set([
  "custom",
  "vllm",
  "llama_cpp",
  "lemonade",
  "ollama",
]);
const OPENAI_SPEECH_MODELS = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"];

export function connectionSpeechModels(connection: {
  providerType: string;
  models: readonly string[];
}): string[] {
  if (connection.providerType === "openai") return OPENAI_SPEECH_MODELS;
  return SELF_HOSTED_TYPES.has(connection.providerType)
    ? [...connection.models]
    : [];
}
