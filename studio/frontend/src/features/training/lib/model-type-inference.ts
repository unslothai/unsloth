// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { EMBEDDING_TAGS } from "@/features/hub";
import {
  type TrainingModelModalityMetadata,
  inferTrainingModelModalityFlags,
} from "./model-modality-inference";
import type { ModelTypeCapabilityFlags } from "./model-type-capabilities";

const TEXT_MODEL_TAGS = new Set(["text-generation"]);
const DECISION_MODEL_TAGS = new Set(["laya"]);

export type TrainingModelTypeMetadata = TrainingModelModalityMetadata;

function hasDecisionHint({ tags }: TrainingModelTypeMetadata): boolean {
  return (tags ?? []).some((tag) => DECISION_MODEL_TAGS.has(tag.toLowerCase()));
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
  const isDecision = hasDecisionHint(metadata);
  const hasModelTypeSignal =
    isDecision ||
    isEmbedding ||
    capabilities.isAudio ||
    capabilities.isVision ||
    hasTextModelHint(metadata);
  return {
    isDecision: hasModelTypeSignal ? isDecision : undefined,
    isEmbedding: hasModelTypeSignal ? isEmbedding : undefined,
    isAudio: hasModelTypeSignal ? capabilities.isAudio : undefined,
    isVision: hasModelTypeSignal ? capabilities.isVision : undefined,
    hasModelTypeSignal,
  };
}
