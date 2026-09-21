// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingStartRequest } from "../types/api";
import type { TrainingConfigState } from "../types/config";
import { isUntrainableModelFormat } from "./model-support";

type TrainingStartIdentityConfig = Pick<
  TrainingConfigState,
  | "modelType"
  | "isVisionModel"
  | "isAudioModel"
  | "manualDatasetOptionsValid"
  | "userEditRevision"
>;

export function normalizeTrainingStartPayloadForComparison(
  payload: TrainingStartRequest,
): TrainingStartRequest {
  const normalized = { ...payload };
  normalized.model_format = isUntrainableModelFormat(payload.model_format)
    ? payload.model_format
    : null;
  return normalized;
}

export function createTrainingStartInputIdentity(
  payload: TrainingStartRequest,
  config: TrainingStartIdentityConfig,
) {
  return {
    payload: normalizeTrainingStartPayloadForComparison(payload),
    modelType: config.modelType,
    isVisionModel: config.isVisionModel,
    isAudioModel: config.isAudioModel,
    manualDatasetOptionsValid: config.manualDatasetOptionsValid,
    userEditRevision: config.userEditRevision,
  };
}
