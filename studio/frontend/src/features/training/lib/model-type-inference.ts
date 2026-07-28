// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { EMBEDDING_TAGS } from "@/features/hub";
import type { ModelType } from "@/types/training";

export interface ModelTypeCapabilityFlags {
  isEmbedding?: boolean | null;
  isAudio?: boolean | null;
  isVision?: boolean | null;
}

export function inferTrainingModelTypeFromFlags({
  isEmbedding,
  isAudio,
  isVision,
}: ModelTypeCapabilityFlags): ModelType {
  if (isEmbedding) {
    return "embeddings";
  }
  if (isAudio) {
    return "audio";
  }
  if (isVision) {
    return "vision";
  }
  return "text";
}

export interface TrainingModelTypeMetadata {
  tags?: readonly string[] | null;
  pipelineTag?: string | null;
  identifiers?: readonly (string | null | undefined)[];
}

function hasEmbeddingHint({
  tags,
  pipelineTag,
}: TrainingModelTypeMetadata): boolean {
  if (pipelineTag && EMBEDDING_TAGS.has(pipelineTag.toLowerCase())) {
    return true;
  }
  return (tags ?? []).some((tag) => EMBEDDING_TAGS.has(tag.toLowerCase()));
}

const VISION_TAGS = new Set([
  "image-text-to-text",
  "image-to-text",
  "visual-question-answering",
  "video-text-to-text",
  "any-to-any",
  "multimodal",
  "vision",
]);
const AUDIO_TAGS = new Set([
  "automatic-speech-recognition",
  "audio-text-to-text",
  "text-to-speech",
  "text-to-audio",
  "audio-to-audio",
  "audio-classification",
]);
const SEP = "(?:^|[-_/. ])";
const END = "(?=$|[-_/. ])";
const VISION_NAME_RE = new RegExp(
  `${SEP}(?:vl|llava|pixtral|moondream|smolvlm|internvl|cogvlm|idefics|paligemma|vision)${END}`,
  "i",
);
const AUDIO_NAME_RE = new RegExp(
  `${SEP}(?:whisper|tts|parakeet|parler|musicgen|bark|orpheus|csm|voice|speech|audio)${END}`,
  "i",
);

function hasAny(tagSet: Set<string>, wanted: Set<string>): boolean {
  for (const tag of wanted) {
    if (tagSet.has(tag)) {
      return true;
    }
  }
  return false;
}

function detectTrainingCapabilities(opts: {
  id: string;
  tags?: readonly string[];
  pipelineTag?: string;
}): Pick<ModelTypeCapabilityFlags, "isAudio" | "isVision"> {
  const tagSet = new Set((opts.tags ?? []).map((tag) => tag.toLowerCase()));
  if (opts.pipelineTag) {
    tagSet.add(opts.pipelineTag.toLowerCase());
  }
  return {
    isAudio: hasAny(tagSet, AUDIO_TAGS) || AUDIO_NAME_RE.test(opts.id),
    isVision: hasAny(tagSet, VISION_TAGS) || VISION_NAME_RE.test(opts.id),
  };
}

export function trainingModelTypeFlagsFromMetadata(
  metadata: TrainingModelTypeMetadata,
): ModelTypeCapabilityFlags {
  const { tags, pipelineTag, identifiers = [] } = metadata;
  const capabilities = detectTrainingCapabilities({
    id: identifiers.filter(Boolean).join(" "),
    tags: tags ?? undefined,
    pipelineTag: pipelineTag ?? undefined,
  });
  return {
    isEmbedding: hasEmbeddingHint(metadata),
    isAudio: capabilities.isAudio,
    isVision: capabilities.isVision,
  };
}
