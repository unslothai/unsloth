


import { EMBEDDING_TAGS } from "@/features/hub";
import {
  type TrainingModelModalityMetadata,
  inferTrainingModelModalityFlags,
} from "./model-modality-inference";
import type { ModelTypeCapabilityFlags } from "./model-type-capabilities";

const TEXT_MODEL_TAGS = new Set(["text-generation"]);

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
