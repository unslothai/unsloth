// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState } from "../types/config";
import type { ModelDefaultsPatch } from "./model-defaults-edit-policy";

function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    // Do not coerce blank strings to 0.
    const trimmed = value.trim();
    if (trimmed === "") return undefined;
    const parsed = Number(trimmed);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function toBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  return undefined;
}

function toStringValue(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  return undefined;
}

function toStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const result = [
    ...new Set(
      value.filter((item): item is string => typeof item === "string"),
    ),
  ];
  return result.length > 0 ? result : undefined;
}

// The spellings studio/backend/core/training/trainer.py accepts, so a file means
// the same thing to the picker as it does to the trainer. A quoted "false" read
// as "leave it at the default" is how a config asking for no checkpointing ended
// up training with Unsloth GC.
const GRADIENT_CHECKPOINTING_ALIASES = new Map<
  string,
  TrainingConfigState["gradientCheckpointing"]
>([
  ["true", "true"],
  ["1", "true"],
  ["yes", "true"],
  ["false", "none"],
  ["0", "none"],
  ["no", "none"],
  ["none", "none"],
  ["off", "none"],
  ["unsloth", "unsloth"],
  ["mlx", "mlx"],
]);

function toGradientCheckpointing(
  value: unknown,
): TrainingConfigState["gradientCheckpointing"] | undefined {
  // Shipped YAML may decode this value as a boolean.
  if (typeof value === "boolean") return value ? "true" : "none";
  if (typeof value !== "string") return undefined;
  // Blank means absent here too, so it keeps whatever is selected.
  const resolved = GRADIENT_CHECKPOINTING_ALIASES.get(
    value.trim().toLowerCase(),
  );
  if (resolved === undefined) return undefined;
  // On Mac, map "unsloth" → "mlx" since Unsloth GC is GPU-only
  if (
    resolved === "unsloth" &&
    usePlatformStore.getState().deviceType === "mac"
  ) {
    return "mlx";
  }
  return resolved;
}

export function mapBackendModelConfigToTrainingPatch(
  config?: BackendModelConfig | null,
): ModelDefaultsPatch {
  if (!config) return {};

  const patch: ModelDefaultsPatch = {};
  const training = config.training;
  const lora = config.lora;
  const logging = config.logging;

  const maxSeqLength = toNumber(training?.max_seq_length);
  if (maxSeqLength !== undefined) patch.contextLength = maxSeqLength;

  const numEpochs = toNumber(training?.num_epochs);
  if (numEpochs !== undefined) patch.epochs = numEpochs;

  const learningRate = toNumber(training?.learning_rate);
  if (learningRate !== undefined) patch.learningRate = learningRate;

  // Preserve explicit null ("derive it") versus an absent or invalid value.
  if (Object.hasOwn(training ?? {}, "embedding_learning_rate")) {
    const raw = training?.embedding_learning_rate;
    if (raw === null) {
      patch.embeddingLearningRate = null;
    } else {
      const embeddingLearningRate = toNumber(raw);
      if (embeddingLearningRate !== undefined) {
        patch.embeddingLearningRate = embeddingLearningRate;
      }
    }
  }

  const optim = toStringValue(training?.optim);
  if (optim !== undefined) patch.optimizerType = optim;

  const lrSchedulerType = toStringValue(training?.lr_scheduler_type);
  if (lrSchedulerType !== undefined) patch.lrSchedulerType = lrSchedulerType;

  const batchSize = toNumber(training?.batch_size);
  if (batchSize !== undefined) patch.batchSize = batchSize;

  const gradAccum = toNumber(training?.gradient_accumulation_steps);
  if (gradAccum !== undefined) patch.gradientAccumulation = gradAccum;

  const warmupSteps = toNumber(training?.warmup_steps);
  if (warmupSteps !== undefined) patch.warmupSteps = warmupSteps;

  const maxSteps = toNumber(training?.max_steps);
  if (maxSteps !== undefined) patch.maxSteps = maxSteps;

  const saveSteps = toNumber(training?.save_steps);
  if (saveSteps !== undefined) patch.saveSteps = saveSteps;

  const evalSteps = toNumber(training?.eval_steps);
  if (evalSteps !== undefined) patch.evalSteps = evalSteps;

  const weightDecay = toNumber(training?.weight_decay);
  if (weightDecay !== undefined) patch.weightDecay = weightDecay;

  const randomSeed = toNumber(training?.random_seed);
  if (randomSeed !== undefined) patch.randomSeed = randomSeed;

  // Only patch when the config carries the key; model-switch reset lives in setSelectedModel.
  if (Object.hasOwn(training ?? {}, "vision_image_size")) {
    const raw = training?.vision_image_size;
    if (raw == null) {
      patch.visionImageSize = null;
    } else {
      // Mirror studio/backend/models/training.py:_check_vision_image_size: drop anything outside
      // [_MIN_VISION_IMAGE_SIZE, _MAX_VISION_IMAGE_SIZE] so the UI never shows a rejected value.
      const n = toNumber(raw);
      if (n !== undefined && Number.isInteger(n) && n >= 256 && n <= 2048) {
        patch.visionImageSize = n;
      } else {
        patch.visionImageSize = null;
      }
    }
  }

  const packing = toBoolean(training?.packing);
  if (packing !== undefined) patch.packing = packing;

  const trainOnCompletions = toBoolean(training?.train_on_completions);
  if (trainOnCompletions !== undefined) {
    patch.trainOnCompletions = trainOnCompletions;
  }

  const gradientCheckpointing = toGradientCheckpointing(
    training?.gradient_checkpointing,
  );
  if (gradientCheckpointing !== undefined) {
    patch.gradientCheckpointing = gradientCheckpointing;
  }

  const trustRemoteCode = toBoolean(training?.trust_remote_code);
  if (trustRemoteCode !== undefined) patch.trustRemoteCode = trustRemoteCode;

  const loraRank = toNumber(lora?.lora_r);
  if (loraRank !== undefined) patch.loraRank = loraRank;

  const loraAlpha = toNumber(lora?.lora_alpha);
  if (loraAlpha !== undefined) patch.loraAlpha = loraAlpha;

  const loraDropout = toNumber(lora?.lora_dropout);
  if (loraDropout !== undefined) patch.loraDropout = loraDropout;

  const targetModules = toStringArray(lora?.target_modules);
  if (targetModules !== undefined) patch.targetModules = targetModules;

  if (lora?.use_loftq === true) patch.loraVariant = "loftq";
  else if (lora?.use_rslora === true) patch.loraVariant = "rslora";
  else if (lora?.use_dora === true) patch.loraVariant = "dora";
  else if (lora) patch.loraVariant = "lora";

  const finetuneVisionLayers = toBoolean(lora?.finetune_vision_layers);
  if (finetuneVisionLayers !== undefined) {
    patch.finetuneVisionLayers = finetuneVisionLayers;
  }

  const finetuneLanguageLayers = toBoolean(lora?.finetune_language_layers);
  if (finetuneLanguageLayers !== undefined) {
    patch.finetuneLanguageLayers = finetuneLanguageLayers;
  }

  const finetuneAttentionModules = toBoolean(lora?.finetune_attention_modules);
  if (finetuneAttentionModules !== undefined) {
    patch.finetuneAttentionModules = finetuneAttentionModules;
  }

  const finetuneMLPModules = toBoolean(lora?.finetune_mlp_modules);
  if (finetuneMLPModules !== undefined) {
    patch.finetuneMLPModules = finetuneMLPModules;
  }

  const enableWandb = toBoolean(logging?.enable_wandb);
  if (enableWandb !== undefined) patch.enableWandb = enableWandb;

  const wandbProject = toStringValue(logging?.wandb_project);
  if (wandbProject !== undefined) patch.wandbProject = wandbProject;

  const enableTensorboard = toBoolean(logging?.enable_tensorboard);
  if (enableTensorboard !== undefined)
    patch.enableTensorboard = enableTensorboard;

  const tensorboardDir = toStringValue(logging?.tensorboard_dir);
  if (tensorboardDir !== undefined) patch.tensorboardDir = tensorboardDir;

  const logFrequency = toNumber(logging?.log_frequency);
  if (logFrequency !== undefined) patch.logFrequency = logFrequency;

  return patch;
}
