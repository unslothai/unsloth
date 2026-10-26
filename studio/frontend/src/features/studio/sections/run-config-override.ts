// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseBackendTrainingMethod } from "@/features/training";

export interface RunConfigOverride {
  trainingMethod?: string;
  isDecision?: boolean;
  epochs?: number;
  batchSize?: number;
  learningRate?: string;
  maxSteps?: number;
  contextLength?: number;
  warmupSteps?: number;
  optimizerType?: string;
  loraRank?: number;
  loraAlpha?: number;
  loraDropout?: number;
  loraVariant?: string;
}

/** Shared by History and Current Run so both read the saved snapshot, not the form store. */
export function mapRunConfigToOverride(
  config: Record<string, unknown> | null | undefined,
): RunConfigOverride | undefined {
  if (!config) {
    return undefined;
  }
  return {
    trainingMethod: parseBackendTrainingMethod(
      config.training_type,
      config.load_in_4bit,
    ),
    isDecision: config.is_decision === true,
    epochs: config.num_epochs as number | undefined,
    batchSize: config.batch_size as number | undefined,
    learningRate: config.learning_rate as string | undefined,
    maxSteps: config.max_steps as number | undefined,
    contextLength: config.max_seq_length as number | undefined,
    warmupSteps: config.warmup_steps as number | undefined,
    optimizerType: config.optim as string | undefined,
    loraRank: config.lora_r as number | undefined,
    loraAlpha: config.lora_alpha as number | undefined,
    loraDropout: config.lora_dropout as number | undefined,
    loraVariant: config.use_rslora
      ? "rslora"
      : config.use_loftq
        ? "loftq"
        : config.use_dora
          ? "dora"
          : "lora",
  };
}
