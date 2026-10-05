// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType } from "@/types/training";

export interface ModelTypeCapabilityFlags {
  isEmbedding?: boolean | null;
  isAudio?: boolean | null;
  isVision?: boolean | null;
  hasModelTypeSignal?: boolean;
}

export function inferTrainingModelTypeFromFlags({
  isEmbedding,
  isAudio,
  isVision,
}: ModelTypeCapabilityFlags): ModelType {
  if (isEmbedding) {
    return "embeddings";
  }
  if (isVision) {
    return "vision";
  }
  if (isAudio) {
    return "audio";
  }
  return "text";
}

export function resolveTrainingModelType({
  modelType,
  ...capabilities
}: ModelTypeCapabilityFlags & { modelType?: ModelType | null }): ModelType {
  if (modelType === "audio" && capabilities.isVision) {
    return "vision";
  }
  return modelType ?? inferTrainingModelTypeFromFlags(capabilities);
}
