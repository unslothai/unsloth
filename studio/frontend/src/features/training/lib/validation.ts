// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAdapterMethod } from "../../../types/training.ts";
import { parseSliceValueInput } from "./dataset-slices.ts";
import type { TrainingConfigState } from "../types/config.ts";

export interface StartValidationResult {
  ok: boolean;
  message: string | null;
}

const VALID_CONFIG: StartValidationResult = { ok: true, message: null };
const SELECT_HF_DATASET: StartValidationResult = {
  ok: false,
  message: "Select a Hugging Face dataset first.",
};
const SELECT_LOCAL_DATASET: StartValidationResult = {
  ok: false,
  message: "Select a local dataset first.",
};
const UNSUPPORTED_DATASET_SOURCE: StartValidationResult = {
  ok: false,
  message: "Unsupported dataset source.",
};
const INVALID_DATASET_SLICE: StartValidationResult = {
  ok: false,
  message:
    "Dataset slice values must be non-negative whole numbers, with start below end.",
};
const INVALID_CONTEXT_LENGTH: StartValidationResult = {
  ok: false,
  message: "Context length must be at least 1.",
};
const INVALID_BATCH_SIZE: StartValidationResult = {
  ok: false,
  message: "Batch size must be at least 1.",
};
const INVALID_GRADIENT_ACCUMULATION: StartValidationResult = {
  ok: false,
  message: "Gradient accumulation must be at least 1.",
};
const INVALID_TRAINING_LENGTH: StartValidationResult = {
  ok: false,
  message: "Set epochs or max steps above 0.",
};
const INVALID_LORA_RANK: StartValidationResult = {
  ok: false,
  message: "LoRA rank must be at least 1.",
};
const INVALID_LORA_ALPHA: StartValidationResult = {
  ok: false,
  message: "LoRA alpha must be at least 1.",
};
const INVALID_LORA_DROPOUT: StartValidationResult = {
  ok: false,
  message: "LoRA dropout must be between 0 and 1.",
};
const INVALID_LEARNING_RATE: StartValidationResult = {
  ok: false,
  message: "Learning rate must be above 0.",
};
const INVALID_EMBEDDING_LEARNING_RATE: StartValidationResult = {
  ok: false,
  message: "Embedding learning rate must be above 0 or left blank.",
};
const INVALID_WEIGHT_DECAY: StartValidationResult = {
  ok: false,
  message: "Weight decay must be 0 or higher.",
};
const INVALID_WARMUP_STEPS: StartValidationResult = {
  ok: false,
  message: "Warmup steps must be 0 or higher.",
};
const INVALID_SAVE_STEPS: StartValidationResult = {
  ok: false,
  message: "Save steps must be 0 or higher.",
};
const INVALID_EVAL_STEPS: StartValidationResult = {
  ok: false,
  message: "Eval steps must be between 0 and 1.",
};
const INVALID_RANDOM_SEED: StartValidationResult = {
  ok: false,
  message: "Random seed must be a whole number 0 or higher.",
};
const INVALID_LOG_FREQUENCY: StartValidationResult = {
  ok: false,
  message: "Log frequency must be at least 1.",
};
const SELECT_BASE_MODEL: StartValidationResult = {
  ok: false,
  message: "Select a base model first.",
};

function finite(value: number): boolean {
  return typeof value === "number" && Number.isFinite(value);
}

function finiteInteger(value: number): boolean {
  return finite(value) && Number.isInteger(value);
}

function validateDatasetSelection(
  config: TrainingConfigState,
): StartValidationResult {
  if (config.datasetSource === "huggingface") {
    if (!config.dataset) {
      return SELECT_HF_DATASET;
    }
  } else if (config.datasetSource === "upload") {
    if (!config.uploadedFile) {
      return SELECT_LOCAL_DATASET;
    }
  } else {
    return UNSUPPORTED_DATASET_SOURCE;
  }

  const sliceStart = parseSliceValueInput(config.datasetSliceStart);
  const sliceEnd = parseSliceValueInput(config.datasetSliceEnd);
  if (!sliceStart.ok || !sliceEnd.ok) {
    return INVALID_DATASET_SLICE;
  }
  if (
    sliceStart.value !== null &&
    sliceEnd.value !== null &&
    sliceStart.value >= sliceEnd.value
  ) {
    return INVALID_DATASET_SLICE;
  }

  return VALID_CONFIG;
}

function validateHyperparameters(
  config: TrainingConfigState,
): StartValidationResult {
  if (!finiteInteger(config.contextLength) || config.contextLength < 1) {
    return INVALID_CONTEXT_LENGTH;
  }
  if (
    finiteInteger(config.maxPositionEmbeddings ?? Number.NaN) &&
    config.maxPositionEmbeddings !== null &&
    config.contextLength > config.maxPositionEmbeddings
  ) {
    return {
      ok: false,
      message: `Context length cannot exceed this model's maximum of ${config.maxPositionEmbeddings.toLocaleString()}.`,
    };
  }
  if (!finiteInteger(config.batchSize) || config.batchSize < 1) {
    return INVALID_BATCH_SIZE;
  }
  if (
    !finiteInteger(config.gradientAccumulation) ||
    config.gradientAccumulation < 1
  ) {
    return INVALID_GRADIENT_ACCUMULATION;
  }
  if (!finite(config.epochs) || config.epochs < 0) {
    return INVALID_TRAINING_LENGTH;
  }
  if (!finiteInteger(config.maxSteps) || config.maxSteps < 0) {
    return INVALID_TRAINING_LENGTH;
  }
  if (config.maxSteps <= 0 && config.epochs <= 0) {
    return INVALID_TRAINING_LENGTH;
  }
  if (!finite(config.learningRate) || config.learningRate <= 0) {
    return INVALID_LEARNING_RATE;
  }
  if (
    config.embeddingLearningRate !== null &&
    (!finite(config.embeddingLearningRate) || config.embeddingLearningRate <= 0)
  ) {
    return INVALID_EMBEDDING_LEARNING_RATE;
  }
  if (!finite(config.weightDecay) || config.weightDecay < 0) {
    return INVALID_WEIGHT_DECAY;
  }
  if (!finiteInteger(config.warmupSteps) || config.warmupSteps < 0) {
    return INVALID_WARMUP_STEPS;
  }
  if (!finiteInteger(config.saveSteps) || config.saveSteps < 0) {
    return INVALID_SAVE_STEPS;
  }
  if (!finite(config.evalSteps) || config.evalSteps < 0 || config.evalSteps > 1) {
    return INVALID_EVAL_STEPS;
  }
  if (!finiteInteger(config.randomSeed) || config.randomSeed < 0) {
    return INVALID_RANDOM_SEED;
  }
  if (!finiteInteger(config.logFrequency) || config.logFrequency < 1) {
    return INVALID_LOG_FREQUENCY;
  }
  if (isAdapterMethod(config.trainingMethod)) {
    if (!finiteInteger(config.loraRank) || config.loraRank < 1) {
      return INVALID_LORA_RANK;
    }
    if (!finite(config.loraAlpha) || config.loraAlpha < 1) {
      return INVALID_LORA_ALPHA;
    }
    if (
      !finite(config.loraDropout) ||
      config.loraDropout < 0 ||
      config.loraDropout > 1
    ) {
      return INVALID_LORA_DROPOUT;
    }
  }

  return VALID_CONFIG;
}

export function validateTrainingConfig(
  config: TrainingConfigState,
): StartValidationResult {
  if (!config.selectedModel) {
    return SELECT_BASE_MODEL;
  }

  const dataset = validateDatasetSelection(config);
  if (!dataset.ok) {
    return dataset;
  }

  return validateHyperparameters(config);
}
