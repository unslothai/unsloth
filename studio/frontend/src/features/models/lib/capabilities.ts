// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type CapabilityKey =
  | "vision"
  | "audio"
  | "tools"
  | "reasoning"
  | "code"
  | "embedding"
  | "multilingual";

export interface Capability {
  key: CapabilityKey;
  label: string;
}

const VISION_TAGS = new Set([
  "vision",
  "vision-language",
  "image-text-to-text",
  "image-to-text",
  "visual-question-answering",
  "image-classification",
  "image-segmentation",
  "object-detection",
  "any-to-any",
  "multimodal",
  "vlm",
  "llava",
]);

const AUDIO_TAGS = new Set([
  "audio",
  "audio-text-to-text",
  "automatic-speech-recognition",
  "text-to-speech",
  "audio-classification",
  "voice-activity-detection",
  "asr",
  "tts",
]);

const TOOL_TAGS = new Set([
  "tool-use",
  "tools",
  "tool-calling",
  "function-calling",
  "function-call",
  "agent",
  "agentic",
]);

const REASONING_TAGS = new Set([
  "reasoning",
  "thinking",
  "chain-of-thought",
  "math-reasoning",
  "step-by-step",
]);

const CODE_TAGS = new Set([
  "code",
  "code-generation",
  "code-completion",
  "code-llm",
  "programming",
]);

const EMBEDDING_TAGS = new Set([
  "feature-extraction",
  "sentence-transformers",
  "sentence-similarity",
  "text-embeddings-inference",
  "embeddings",
]);

const COMMON_LANGUAGE_TAGS = new Set([
  "en", "fr", "de", "es", "it", "pt", "nl", "ru", "pl", "uk", "tr", "cs",
  "sv", "no", "da", "fi", "el", "ro", "bg", "hu", "hr", "sr", "sl", "sk",
  "zh", "ja", "ko", "ar", "he", "fa", "hi", "bn", "ta", "te", "th", "vi",
  "id", "ms", "tl", "sw", "ur", "uk", "ca", "eu", "is", "lv", "lt", "et",
]);

function lower(values: string[] | undefined): Set<string> {
  if (!values) return new Set();
  return new Set(values.map((v) => v.toLowerCase()));
}

export function detectCapabilities(
  tags: string[] | undefined,
  pipelineTag: string | undefined,
  modelId: string,
): Capability[] {
  const tagSet = lower(tags);
  const pipeline = pipelineTag?.toLowerCase();
  const lowerId = modelId.toLowerCase();
  const out: Capability[] = [];

  const hasAny = (set: Set<string>) =>
    [...set].some((t) => tagSet.has(t)) ||
    (pipeline ? set.has(pipeline) : false);

  if (
    hasAny(VISION_TAGS) ||
    /[-_/]vl[-_]|vision|multimodal|llava|kosmos|qwen2[-_]?vl|cogvlm/.test(
      lowerId,
    )
  ) {
    out.push({ key: "vision", label: "Vision" });
  }
  if (
    hasAny(AUDIO_TAGS) ||
    /audio|whisper|speech|tts|stt|parler|sesame/.test(lowerId)
  ) {
    out.push({ key: "audio", label: "Audio" });
  }
  if (hasAny(TOOL_TAGS) || /tool[-_]use|function[-_]call/.test(lowerId)) {
    out.push({ key: "tools", label: "Tool use" });
  }
  if (
    hasAny(REASONING_TAGS) ||
    /think|reason|qwq|deepseek[-_]?r|gpt[-_]?oss|o1[-_]|nemotron[-_]reasoning/.test(
      lowerId,
    )
  ) {
    out.push({ key: "reasoning", label: "Reasoning" });
  }
  if (
    hasAny(CODE_TAGS) ||
    /coder|code[-_]|starcoder|codellama|deepseek[-_]coder/.test(lowerId)
  ) {
    out.push({ key: "code", label: "Code" });
  }
  if (
    hasAny(EMBEDDING_TAGS) ||
    /embed|retriever|reranker|bge[-_]|e5[-_]/.test(lowerId)
  ) {
    out.push({ key: "embedding", label: "Embeddings" });
  }
  const languageTagCount = (tags ?? []).reduce((count, tag) => {
    return COMMON_LANGUAGE_TAGS.has(tag.toLowerCase()) ? count + 1 : count;
  }, 0);
  if (tagSet.has("multilingual") || languageTagCount >= 3) {
    out.push({ key: "multilingual", label: "Multilingual" });
  }
  return out;
}

export function detectLicense(tags: string[] | undefined): string | null {
  if (!tags) return null;
  const license = tags.find((t) => t.startsWith("license:"));
  return license ? license.slice("license:".length) : null;
}
