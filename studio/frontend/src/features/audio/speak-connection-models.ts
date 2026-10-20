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

function isOpenAIApi(baseUrl: string | undefined): boolean {
  if (!baseUrl) return true;
  try {
    return new URL(baseUrl).hostname.toLowerCase() === "api.openai.com";
  } catch {
    return false;
  }
}

export function connectionSpeechModels(connection: {
  providerType: string;
  baseUrl?: string;
  models: readonly string[];
}): string[] {
  if (connection.providerType === "openai" && isOpenAIApi(connection.baseUrl)) {
    return OPENAI_SPEECH_MODELS;
  }
  // Azure and other OpenAI-typed endpoints name models by deployment.
  return connection.providerType === "openai" ||
    SELF_HOSTED_TYPES.has(connection.providerType)
    ? [...connection.models]
    : [];
}
