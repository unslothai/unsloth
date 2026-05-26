// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isGgufLike } from "../../../lib/model-identifiers.ts";
import { classifyUnslothSupport } from "../../../lib/unsloth-support.ts";
import type {
  ModelInventoryCapabilities,
  ModelInventoryFormat,
} from "../../inventory/types.ts";

const ADAPTER_ARTIFACT_PATTERN =
  /(?:^|[/\\])adapter_(?:config\.json|model\.safetensors)$/i;

export type TrainingModelValidationCandidate = {
  id: string;
  modelFormat?: ModelInventoryFormat | null;
  capabilities?: ModelInventoryCapabilities | null;
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  quantMethod?: string | null;
};

export type TrainingModelValidationResult =
  | { ok: true }
  | { ok: false; reason: string };

export function validateTrainingModelCandidate(
  candidate: TrainingModelValidationCandidate,
  options: { deviceType?: string | null } = {},
): TrainingModelValidationResult {
  const id = candidate.id.trim();
  if (!id) {
    return { ok: false, reason: "Enter a model id or local model path." };
  }
  if (candidate.modelFormat === "gguf" || isGgufLike(id)) {
    return { ok: false, reason: "GGUF models cannot be used for training." };
  }
  if (
    candidate.modelFormat === "adapter" ||
    ADAPTER_ARTIFACT_PATTERN.test(id)
  ) {
    return {
      ok: false,
      reason: "Adapter outputs cannot be used as base training models.",
    };
  }
  if (candidate.capabilities && !candidate.capabilities.canTrain) {
    return { ok: false, reason: "This on-device model is not trainable." };
  }
  const support = classifyUnslothSupport({
    modelId: id,
    pipelineTag: candidate.pipelineTag,
    tags: candidate.tags,
    libraryName: candidate.libraryName,
    quantMethod: candidate.quantMethod,
    deviceType: options.deviceType,
  });
  if (support.status === "unsupported") {
    return {
      ok: false,
      reason:
        support.reason ?? "This model format is not supported for training.",
    };
  }
  return { ok: true };
}
