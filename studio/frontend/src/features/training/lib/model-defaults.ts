// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState } from "../types/config";
import { usePlatformStore } from "@/config/env";

type ModelDefaultsPatch = Partial<
  Pick<
    TrainingConfigState,
    | "epochs"
    | "contextLength"
    | "learningRate"
    | "optimizerType"
    | "lrSchedulerType"
    | "loraRank"
    | "loraAlpha"
    | "loraDropout"
    | "loraVariant"
    | "batchSize"
    | "gradientAccumulation"
    | "weightDecay"
    | "warmupSteps"
    | "maxSteps"
    | "saveSteps"
    | "evalSteps"
    | "packing"
    | "trainOnCompletions"
    | "gradientCheckpointing"
    | "randomSeed"
    | "enableWandb"
    | "wandbProject"
    | "enableTensorboard"
    | "tensorboardDir"
    | "logFrequency"
    | "finetuneVisionLayers"
    | "trustRemoteCode"
    | "finetuneLanguageLayers"
    | "finetuneAttentionModules"
    | "finetuneMLPModules"
    | "targetModules"
  >
>;

function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
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
  const result = value.filter((item): item is string => typeof item === "string");
  return result.length > 0 ? result : undefined;
}

function toGradientCheckpointing(
  value: unknown,
): TrainingConfigState["gradientCheckpointing"] | undefined {
  if (value === "none" || value === "true" || value === "unsloth" || value === "mlx") {
    // On Mac, map "unsloth" → "mlx" since Unsloth GC is GPU-only
    if (usePlatformStore.getState().deviceType === "mac" && value === "unsloth") {
      return "mlx";
    }
    return value;
  }
  return undefined;
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
  if (enableTensorboard !== undefined) patch.enableTensorboard = enableTensorboard;

  const tensorboardDir = toStringValue(logging?.tensorboard_dir);
  if (tensorboardDir !== undefined) patch.tensorboardDir = tensorboardDir;

  const logFrequency = toNumber(logging?.log_frequency);
  if (logFrequency !== undefined) patch.logFrequency = logFrequency;

  return patch;
}
