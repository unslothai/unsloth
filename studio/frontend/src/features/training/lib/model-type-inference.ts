// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { EMBEDDING_TAGS } from "@/features/hub";
import type { ModelType } from "@/types/training";
import {
  type TrainingModelModalityMetadata,
  inferTrainingModelModalityFlags,
} from "./model-modality-inference";

export interface ModelTypeCapabilityFlags {
  isEmbedding?: boolean | null;
  isAudio?: boolean | null;
  isVision?: boolean | null;
  hasModelTypeSignal?: boolean;
}

const TEXT_MODEL_TAGS = new Set(["text-generation"]);

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

export type TrainingModelTypeMetadata = TrainingModelModalityMetadata;

function hasEmbeddingHint({
  tags,
  pipelineTag,
}: TrainingModelTypeMetadata): boolean {
  if (pipelineTag && EMBEDDING_TAGS.has(pipelineTag.toLowerCase())) {
    return true;
  }
  return (tags ?? []).some((tag) => EMBEDDING_TAGS.has(tag.toLowerCase()));
}

function hasTextModelHint({
  tags,
  pipelineTag,
}: TrainingModelTypeMetadata): boolean {
  if (pipelineTag && TEXT_MODEL_TAGS.has(pipelineTag.toLowerCase())) {
    return true;
  }
  return (tags ?? []).some((tag) => TEXT_MODEL_TAGS.has(tag.toLowerCase()));
}

export function trainingModelTypeFlagsFromMetadata(
  metadata: TrainingModelTypeMetadata,
): ModelTypeCapabilityFlags {
  const capabilities = inferTrainingModelModalityFlags(metadata);
  const isEmbedding = hasEmbeddingHint(metadata);
  const hasModelTypeSignal =
    isEmbedding ||
    capabilities.isAudio ||
    capabilities.isVision ||
    hasTextModelHint(metadata);
  return {
    isEmbedding: hasModelTypeSignal ? isEmbedding : undefined,
    isAudio: hasModelTypeSignal ? capabilities.isAudio : undefined,
    isVision: hasModelTypeSignal ? capabilities.isVision : undefined,
    hasModelTypeSignal,
  };
}
