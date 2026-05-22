// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAdapterMethod } from "@/types/training";
import { parseSliceValue } from "../api/mappers";
import type { TrainingConfigState } from "../types/config";

export interface StartValidationResult {
  ok: boolean;
  message: string | null;
}

function validateDatasetSelection(
  config: TrainingConfigState,
): StartValidationResult {
  if (config.datasetSource === "huggingface") {
    if (!config.dataset) {
      return { ok: false, message: "Select a Hugging Face dataset first." };
    }
  } else if (config.datasetSource === "upload") {
    if (!config.uploadedFile) {
      return { ok: false, message: "Select a local dataset first." };
    }
  } else {
    return { ok: false, message: "Unsupported dataset source." };
  }

  const sliceStart = parseSliceValue(config.datasetSliceStart);
  const sliceEnd = parseSliceValue(config.datasetSliceEnd);
  if (sliceStart !== null && sliceEnd !== null && sliceStart >= sliceEnd) {
    return { ok: false, message: "Dataset slice start must be below its end." };
  }

  return { ok: true, message: null };
}

function validateHyperparameters(
  config: TrainingConfigState,
): StartValidationResult {
  if (config.contextLength < 1) {
    return { ok: false, message: "Context length must be at least 1." };
  }
  if (config.batchSize < 1) {
    return { ok: false, message: "Batch size must be at least 1." };
  }
  if (config.gradientAccumulation < 1) {
    return { ok: false, message: "Gradient accumulation must be at least 1." };
  }
  // maxSteps === 0 selects epochs mode; both non-positive trains nothing.
  if (config.maxSteps <= 0 && config.epochs <= 0) {
    return { ok: false, message: "Set epochs or max steps above 0." };
  }
  if (isAdapterMethod(config.trainingMethod)) {
    if (config.loraRank < 1) {
      return { ok: false, message: "LoRA rank must be at least 1." };
    }
    if (config.loraAlpha < 1) {
      return { ok: false, message: "LoRA alpha must be at least 1." };
    }
  }

  return { ok: true, message: null };
}

export function validateTrainingConfig(
  config: TrainingConfigState,
): StartValidationResult {
  if (!config.selectedModel) {
    return { ok: false, message: "Select a base model first." };
  }

  const dataset = validateDatasetSelection(config);
  if (!dataset.ok) {
    return dataset;
  }

  return validateHyperparameters(config);
}
