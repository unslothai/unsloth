// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseBackendTrainingMethod } from "@/features/training";

/** Shape of the Training Config popover's data when it is driven by a saved
 * run snapshot instead of the editable form store. */
export interface RunConfigOverride {
  trainingMethod?: string;
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

/** Map a saved run's config (GET /api/train/runs/{id} `detail.config`) into the
 * Training Config popover's override shape. Shared by the History view and the
 * live Current Run view so both read the same authoritative run snapshot
 * instead of the editable form store (#6853). */
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
