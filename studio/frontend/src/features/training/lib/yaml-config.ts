// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import * as yaml from "js-yaml";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState } from "../types/config";

const EXPECTED_TOP_KEYS = new Set(["training", "lora", "logging", "inference"]);

/**
 * Parse a YAML string into a BackendModelConfig suitable for
 * `mapBackendModelConfigToTrainingPatch`. Throws on invalid input.
 */
export function parseYamlConfig(text: string): BackendModelConfig {
  const parsed = yaml.load(text);
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(
      "Invalid config: expected a YAML mapping with training/lora/logging sections",
    );
  }

  const raw = parsed as Record<string, unknown>;
  const unknownKeys = Object.keys(raw).filter(
    (k) => !EXPECTED_TOP_KEYS.has(k),
  );
  if (unknownKeys.length > 0) {
    console.warn("Ignored unknown YAML keys:", unknownKeys.join(", "));
  }

  // File import is authoritative: force vision_image_size = null when the
  // training section is missing/malformed/lacks the key, so a stale store
  // value can't survive an import. (Same-model defaults reloads preserve
  // user choice via Object.hasOwn in model-defaults.ts.)
  const rawTraining = raw.training;
  const isPlainTrainingObject =
    rawTraining != null &&
    typeof rawTraining === "object" &&
    !Array.isArray(rawTraining);
  let trainingObj: Record<string, unknown>;
  if (!isPlainTrainingObject) {
    trainingObj = { vision_image_size: null };
  } else {
    trainingObj = { ...(rawTraining as Record<string, unknown>) };
    if (!Object.hasOwn(trainingObj, "vision_image_size")) {
      trainingObj.vision_image_size = null;
    }
  }

  return {
    training: trainingObj as BackendModelConfig["training"],
    lora: (raw.lora ?? undefined) as BackendModelConfig["lora"],
    logging: (raw.logging ?? undefined) as BackendModelConfig["logging"],
  };
}

/**
 * Serialize the current training config state to a YAML string matching the
 * backend model-defaults schema.
 */
export function serializeConfigToYaml(
  state: TrainingConfigState,
  includeVisionFields: boolean,
  includeVisionImageSize: boolean = includeVisionFields,
): string {
  const lora: Record<string, unknown> = {
    lora_r: state.loraRank,
    lora_alpha: state.loraAlpha,
    lora_dropout: state.loraDropout,
    target_modules: state.targetModules,
    use_rslora: state.loraVariant === "rslora",
    use_loftq: state.loraVariant === "loftq",
    use_dora: state.loraVariant === "dora",
  };

  if (includeVisionFields) {
    lora.finetune_vision_layers = state.finetuneVisionLayers;
    lora.finetune_language_layers = state.finetuneLanguageLayers;
    lora.finetune_attention_modules = state.finetuneAttentionModules;
    lora.finetune_mlp_modules = state.finetuneMLPModules;
  }

  const training: Record<string, unknown> = {
    max_seq_length: state.contextLength,
    num_epochs: state.epochs,
    learning_rate: state.learningRate,
    batch_size: state.batchSize,
    gradient_accumulation_steps: state.gradientAccumulation,
    warmup_steps: state.warmupSteps,
    max_steps: state.maxSteps,
    save_steps: state.saveSteps,
    eval_steps: state.evalSteps,
    weight_decay: state.weightDecay,
    random_seed: state.randomSeed,
    packing: state.packing,
    train_on_completions: state.trainOnCompletions,
    gradient_checkpointing: state.gradientCheckpointing,
    optim: state.optimizerType,
    lr_scheduler_type: state.lrSchedulerType,
  };

  if (includeVisionImageSize) {
    training.vision_image_size = state.visionImageSize;
  }

  const config = {
    training,
    lora,
  };

  return yaml.dump(config, { lineWidth: -1, noRefs: true });
}
