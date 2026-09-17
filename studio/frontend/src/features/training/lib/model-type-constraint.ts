// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType } from "@/types/training";
import type { ModelTypeCapabilityFlags } from "./model-type-capabilities";

export function trainingModelMatchesTypeConstraint(
  capabilities: ModelTypeCapabilityFlags,
  requiredType: ModelType | undefined,
): boolean {
  if (requiredType === undefined || capabilities.hasModelTypeSignal !== true) {
    return true;
  }
  switch (requiredType) {
    case "embeddings":
      return capabilities.isEmbedding === true;
    case "audio":
      return capabilities.isAudio === true;
    case "vision":
      return capabilities.isVision === true;
    case "text":
      return !(
        capabilities.isEmbedding ||
        capabilities.isAudio ||
        capabilities.isVision
      );
    default:
      return false;
  }
}
