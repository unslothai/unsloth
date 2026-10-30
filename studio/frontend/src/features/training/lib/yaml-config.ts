// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import * as yaml from "js-yaml";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState } from "../types/config";
import { drafterRequestFrom } from "./drafter-config";
import { effectiveTrainingObjective } from "./rl-roles";
import { type YamlRlSettings, parseRlSection } from "./rl-settings-config";

export type { YamlRlSettings };

const EXPECTED_TOP_KEYS = new Set([
  "training",
  "lora",
  "logging",
  "inference",
  "rl",
]);

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

  // Import is authoritative: a missing key forces vision_image_size = null.
  const rawTraining = raw.training;
  const isPlainTrainingObject =
    rawTraining != null &&
    typeof rawTraining === "object" &&
    !Array.isArray(rawTraining);
  let trainingObj: Record<string, unknown>;
  if (isPlainTrainingObject) {
    trainingObj = { ...(rawTraining as Record<string, unknown>) };
    if (!Object.hasOwn(trainingObj, "vision_image_size")) {
      trainingObj.vision_image_size = null;
    }
  } else {
    trainingObj = { vision_image_size: null };
  }

  return {
    training: trainingObj as BackendModelConfig["training"],
    lora: (raw.lora ?? undefined) as BackendModelConfig["lora"],
    logging: (raw.logging ?? undefined) as BackendModelConfig["logging"],
  };
}

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
    early_stopping_patience: state.earlyStoppingPatience,
    weight_decay: state.weightDecay,
    random_seed: state.randomSeed,
    packing: state.packing,
    train_on_completions: state.trainOnCompletions,
    gradient_checkpointing: state.gradientCheckpointing,
    offload_layers: state.offloadLayers,
    offload_vram_gb: state.offloadVramGb,
    offload_vram_gb_per_device: state.offloadVramGbPerDevice,
    prefetch_depth: state.prefetchDepth,
    optim: state.optimizerType,
    lr_scheduler_type: state.lrSchedulerType,
  };
  // Only when on, so a file exported without the feature reads exactly as before.
  if (state.drafterEnabled) {
    training.drafter = drafterRequestFrom(state);
  }

  if (includeVisionImageSize) {
    training.vision_image_size = state.visionImageSize;
  }

  const config: Record<string, unknown> = {
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
  };

  // What the run would train: a model that forces SFT exports no RL section.
  const objective = effectiveTrainingObjective(state);
  if (objective !== "sft") {
    config.rl = {
      objective,
      beta: state.rlBeta,
      max_prompt_length: state.rlMaxPromptLength,
      ...(objective === "grpo"
        ? {
            variant: state.grpoVariant,
            num_generations: state.grpoNumGenerations,
            max_completion_length: state.grpoMaxCompletionLength,
            temperature: state.grpoTemperature,
            epsilon_high: state.grpoEpsilonHigh,
            mask_truncated_completions: state.grpoMaskTruncatedCompletions,
            python_reward_isolation: state.grpoPythonRewardIsolation,
            enable_thinking: state.grpoEnableThinking,
            system_prompt: state.grpoSystemPrompt,
            rewards: state.grpoRewards,
          }
        : {}),
    };
  }

  return yaml.dump(config, { lineWidth: -1, noRefs: true });
}

/** The objective and RL settings from a saved YAML. A file with no rl section is SFT;
 * fields with the wrong type are dropped so the current value stays. */
export function parseYamlRlSettings(text: string): YamlRlSettings {
  const parsed = yaml.load(text) as Record<string, unknown> | null;
  return parseRlSection(parsed?.rl);
}
