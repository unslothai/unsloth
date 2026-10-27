// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SELF_HOSTED_TYPES = new Set(["custom", "vllm", "llama_cpp", "lemonade"]);
const OPENAI_SPEECH_MODELS = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"];
const SPEECH_MODEL_ID =
  /kokoro|tts|speech|orpheus|piper|chatterbox|parler|zonos|cosyvoice|(?:^|[\W_])(?:f5|csm|dia|bark|melo|sesame)(?:[\W_\d]|$)/i;
// A self-hosted server such as speaches lists its speech-to-text models beside its voices.
const TRANSCRIBE_MODEL_ID =
  /whisper|transcri|parakeet|(?:^|[\W_])(?:asr|stt)(?:[\W_]|$)/i;

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
  sharedByOwner?: boolean;
}): string[] {
  if (connection.providerType === "openai" && isOpenAIApi(connection.baseUrl)) {
    // An owner-shared connection refuses any model outside the list the owner allowed (403).
    return connection.sharedByOwner
      ? OPENAI_SPEECH_MODELS.filter((id) => connection.models.includes(id))
      : OPENAI_SPEECH_MODELS;
  }
  // Azure and other OpenAI-typed endpoints name models by deployment.
  return connection.providerType === "openai" ||
    SELF_HOSTED_TYPES.has(connection.providerType)
    ? connection.models.filter(
        (id) => SPEECH_MODEL_ID.test(id) && !TRANSCRIBE_MODEL_ID.test(id),
      )
    : [];
}
