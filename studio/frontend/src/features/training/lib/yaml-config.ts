// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import * as yaml from "js-yaml";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState } from "../types/config";

const EXPECTED_TOP_KEYS = new Set([
  "training",
  "lora",
  "logging",
  "inference",
  "checkpoint_backup",
]);

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
  const unknownKeys = Object.keys(raw).filter((k) => !EXPECTED_TOP_KEYS.has(k));
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

  const rawBackup = raw.checkpoint_backup;
  if (
    rawBackup != null &&
    (typeof rawBackup !== "object" || Array.isArray(rawBackup))
  ) {
    throw new Error("Invalid checkpoint_backup: expected a YAML mapping");
  }
  if (rawBackup && typeof rawBackup === "object") {
    const backup = rawBackup as Record<string, unknown>;
    const enabled = backup.enabled === true;
    const saveSteps = Number(trainingObj.save_steps ?? 0);
    const interval = Number(backup.interval_steps ?? 0);
    if (enabled && (!Number.isInteger(saveSteps) || saveSteps <= 0)) {
      throw new Error(
        "Interval-based backup requires local checkpoint save steps greater than zero.",
      );
    }
    if (
      enabled &&
      (!Number.isInteger(interval) ||
        interval <= 0 ||
        interval % saveSteps !== 0)
    ) {
      throw new Error(
        `Backup interval must be a multiple of local checkpoint save steps. With save_steps=${saveSteps}, use ${saveSteps}, ${saveSteps * 2}, ${saveSteps * 3}, ...`,
      );
    }
  }

  return {
    training: trainingObj as BackendModelConfig["training"],
    lora: (raw.lora ?? undefined) as BackendModelConfig["lora"],
    logging: (raw.logging ?? undefined) as BackendModelConfig["logging"],
    checkpoint_backup: (raw.checkpoint_backup ??
      undefined) as BackendModelConfig["checkpoint_backup"],
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
    embedding_learning_rate: state.embeddingLearningRate,
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

  const checkpointBackup = state.checkpointBackup;
  const config = {
    training,
    lora,
    // Include every non-secret logging field read by parseYamlConfig.
    logging: {
      enable_wandb: state.enableWandb,
      wandb_project: state.wandbProject,
      enable_tensorboard: state.enableTensorboard,
      tensorboard_dir: state.tensorboardDir,
      log_frequency: state.logFrequency,
    },
    checkpoint_backup: {
      enabled: checkpointBackup.enabled,
      provider: checkpointBackup.provider,
      repo_id: checkpointBackup.repoId,
      private: checkpointBackup.private,
      interval_steps: checkpointBackup.intervalSteps,
      strategy: checkpointBackup.strategy,
      keep_remote: checkpointBackup.keepRemote,
      upload_on_stop: checkpointBackup.uploadOnStop,
      upload_on_complete: checkpointBackup.uploadOnComplete,
    },
  };

  return yaml.dump(config, { lineWidth: -1, noRefs: true });
}
