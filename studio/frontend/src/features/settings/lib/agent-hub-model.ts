// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type HubModelTaskMetadata = {
  id?: string | null;
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
};

// Coding agents require a chat-completions model. When the Hub declares a
// primary pipeline, admit only pipelines that can consume conversational text;
// a missing pipeline remains eligible because many valid GGUF repos omit it.
const CHAT_GENERATION_TASKS: ReadonlySet<string> = new Set([
  "text-generation",
  "conversational",
  "image-text-to-text",
  "audio-text-to-text",
  "any-to-any",
]);

// These generation tasks belong to the Audio page and cannot serve coding-agent
// chat completions. Keep this narrower than generic audio tags so audio-capable
// chat and vision-language models remain available.
const SPEECH_ONLY_TAGS: ReadonlySet<string> = new Set([
  "text-to-speech",
  "automatic-speech-recognition",
]);

const CLASSIFIER_TASKS: ReadonlySet<string> = new Set([
  "text-classification",
  "token-classification",
  "zero-shot-classification",
]);

const RERANKER_TAGS: ReadonlySet<string> = new Set([
  "reranker",
  "reranking",
  "text-ranking",
  "cross-encoder",
]);

const RERANKER_ID_PATTERN =
  /(?:^|[-_./])(?:rerank(?:er|ing)?|classif(?:ier|ication))(?:$|[-_./])/i;

function normalize(value: string | null | undefined): string | null {
  const normalized = value?.toLowerCase().trim();
  return normalized || null;
}

export function isChatGenerativeHubModel(model: HubModelTaskMetadata): boolean {
  const pipeline = normalize(model.pipelineTag);
  return pipeline == null || CHAT_GENERATION_TASKS.has(pipeline);
}

export function isSpeechOnlyHubModel(model: HubModelTaskMetadata): boolean {
  const pipeline = normalize(model.pipelineTag);
  if (pipeline) {
    return SPEECH_ONLY_TAGS.has(pipeline);
  }
  return (model.tags ?? []).some((tag) => {
    const normalized = normalize(tag);
    return normalized != null && SPEECH_ONLY_TAGS.has(normalized);
  });
}

export function isClassifierOrRerankerHubModel(
  model: HubModelTaskMetadata,
): boolean {
  const pipeline = normalize(model.pipelineTag);
  if (
    pipeline &&
    (CLASSIFIER_TASKS.has(pipeline) || RERANKER_TAGS.has(pipeline))
  ) {
    return true;
  }
  if (RERANKER_ID_PATTERN.test(model.id ?? "")) {
    return true;
  }
  if (pipeline) {
    return false;
  }
  return (model.tags ?? []).some((tag) => {
    const normalized = normalize(tag);
    return (
      normalized != null &&
      (CLASSIFIER_TASKS.has(normalized) || RERANKER_TAGS.has(normalized))
    );
  });
}
