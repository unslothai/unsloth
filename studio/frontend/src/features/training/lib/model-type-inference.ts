// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MODEL_TYPE, type ModelType } from "../../../types/training.ts";
import {
  type CapabilityKey,
  detectCapabilities,
} from "../../../lib/model-capabilities.ts";

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
    return MODEL_TYPE.EMBEDDINGS;
  }
  if (isAudio) {
    return MODEL_TYPE.AUDIO;
  }
  if (isVision) {
    return MODEL_TYPE.VISION;
  }
  return MODEL_TYPE.TEXT;
}

export function inferTrainingModelTypeFromCapabilityKeys(
  capabilities: readonly CapabilityKey[],
): ModelType {
  return inferTrainingModelTypeFromFlags({
    isEmbedding: capabilities.includes("embedding"),
    isAudio: capabilities.includes("audio"),
    isVision: capabilities.includes("vision"),
  });
}

export interface TrainingModelTypeMetadata {
  tags?: readonly string[] | null;
  pipelineTag?: string | null;
  identifiers?: readonly (string | null | undefined)[];
}

export function inferTrainingModelTypeFromMetadata({
  tags,
  pipelineTag,
  identifiers = [],
}: TrainingModelTypeMetadata): ModelType {
  const capabilities = detectCapabilities(
    tags ? [...tags] : undefined,
    pipelineTag ?? undefined,
    identifiers.filter(Boolean).join(" "),
  ).map((capability) => capability.key);
  return inferTrainingModelTypeFromCapabilityKeys(capabilities);
}

export function resolvePickerInferredModelType(
  current: ModelType | null,
  inferred: ModelType,
): ModelType {
  if (
    inferred === MODEL_TYPE.TEXT &&
    (current === MODEL_TYPE.VISION ||
      current === MODEL_TYPE.AUDIO ||
      current === MODEL_TYPE.EMBEDDINGS)
  ) {
    return current;
  }
  return inferred;
}
