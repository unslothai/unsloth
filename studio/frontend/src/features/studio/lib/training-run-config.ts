// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingRunDetailResponse } from "@/features/training";
import { parseBackendTrainingMethod } from "@/features/training/lib/training-methods";
import type { TrainingMethod } from "@/types/training";

export type TrainingRunConfigOverride = {
  epochs?: number;
  batchSize?: number;
  learningRate?: string;
  maxSteps?: number | null;
  contextLength?: number;
  warmupSteps?: number | null;
  optimizerType?: string;
  loraRank?: number;
  loraAlpha?: number;
  loraDropout?: number;
  loraVariant?: string;
};

export function mapTrainingRunConfigOverride(
  detail: TrainingRunDetailResponse,
): TrainingRunConfigOverride | undefined {
  if (!detail.config) return undefined;
  const cfg = detail.config;
  return {
    epochs: cfg.num_epochs as number | undefined,
    batchSize: cfg.batch_size as number | undefined,
    learningRate: cfg.learning_rate as string | undefined,
    maxSteps: cfg.max_steps as number | null | undefined,
    contextLength: cfg.max_seq_length as number | undefined,
    warmupSteps: cfg.warmup_steps as number | null | undefined,
    optimizerType: cfg.optim as string | undefined,
    loraRank: cfg.lora_r as number | undefined,
    loraAlpha: cfg.lora_alpha as number | undefined,
    loraDropout: cfg.lora_dropout as number | undefined,
    loraVariant: cfg.use_rslora
      ? "rslora"
      : cfg.use_loftq
        ? "loftq"
      : "lora",
  };
}

export function mapTrainingRunMethod(
  detail: TrainingRunDetailResponse,
): TrainingMethod {
  const cfg = detail.config;
  return parseBackendTrainingMethod(cfg?.training_type, cfg?.load_in_4bit);
}
