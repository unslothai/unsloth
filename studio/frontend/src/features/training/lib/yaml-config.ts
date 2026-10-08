// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  GRPO_VARIANTS,
  type GrpoRewardSelection,
  type GrpoVariant,
  type TrainingObjective,
} from "@/types/training";
import * as yaml from "js-yaml";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState } from "../types/config";

const EXPECTED_TOP_KEYS = new Set([
  "training",
  "lora",
  "logging",
  "inference",
  "rl",
]);
const RL_OBJECTIVES = ["dpo", "orpo", "grpo"] as const;

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

  // File import is authoritative: force vision_image_size = null when the training section is
  // missing/malformed/lacks the key, so a stale store value can't survive an import. (Same-model
  // defaults reloads preserve user choice via Object.hasOwn in model-defaults.ts.)
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

  // SFT files stay exactly as before; the section only appears for RL objectives.
  if (state.trainingObjective !== "sft") {
    config.rl = {
      objective: state.trainingObjective,
      beta: state.rlBeta,
      max_prompt_length: state.rlMaxPromptLength,
      ...(state.trainingObjective === "grpo"
        ? {
            variant: state.grpoVariant,
            num_generations: state.grpoNumGenerations,
            max_completion_length: state.grpoMaxCompletionLength,
            temperature: state.grpoTemperature,
            epsilon_high: state.grpoEpsilonHigh,
            mask_truncated_completions: state.grpoMaskTruncatedCompletions,
            enable_thinking: state.grpoEnableThinking,
            system_prompt: state.grpoSystemPrompt,
            rewards: state.grpoRewards,
          }
        : {}),
    };
  }

  return yaml.dump(config, { lineWidth: -1, noRefs: true });
}

export interface YamlRlSettings {
  trainingObjective: TrainingObjective;
  rlBeta?: number | null;
  rlMaxPromptLength?: number | null;
  grpoVariant?: GrpoVariant;
  grpoNumGenerations?: number;
  grpoMaxCompletionLength?: number | null;
  grpoTemperature?: number;
  grpoEpsilonHigh?: number | null;
  grpoMaskTruncatedCompletions?: boolean;
  grpoEnableThinking?: boolean;
  grpoSystemPrompt?: string;
  grpoRewards?: GrpoRewardSelection[];
}

const isNum = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const numOrNull = (v: unknown): number | null | undefined =>
  v === null ? null : isNum(v) ? v : undefined;

/** The objective and RL settings from a saved YAML. A file with no rl section is SFT;
 * fields with the wrong type are dropped so the current value stays. */
export function parseYamlRlSettings(text: string): YamlRlSettings {
  const parsed = yaml.load(text) as Record<string, unknown> | null;
  return parseRlSection(parsed?.rl);
}

/** A saved run's objective and RL settings. The stored config keeps them as `objective`,
 * `rl_settings` (the same keys as the YAML rl section) and `reward_specs`. */
export function parseRunConfigRlSettings(
  config: Record<string, unknown>,
): YamlRlSettings {
  const settings = config.rl_settings;
  return parseRlSection({
    ...(settings !== null && typeof settings === "object" ? settings : {}),
    objective: config.objective,
    rewards: config.reward_specs,
  });
}

function parseRlSection(rl: unknown): YamlRlSettings {
  if (rl == null || typeof rl !== "object" || Array.isArray(rl)) {
    return { trainingObjective: "sft" };
  }
  const r = rl as Record<string, unknown>;
  const objective = RL_OBJECTIVES.find((o) => o === r.objective);
  if (!objective) {
    return { trainingObjective: "sft" };
  }
  const out: YamlRlSettings = { trainingObjective: objective };
  const set = <K extends keyof YamlRlSettings>(
    key: K,
    value: YamlRlSettings[K] | undefined,
  ) => {
    if (value !== undefined) {
      out[key] = value;
    }
  };
  // Out-of-range values are dropped, matching the backend's TrainingStartRequest bounds.
  const inRange = (v: unknown, lo: number, hi: number, loOpen = false) => {
    const n = numOrNull(v);
    return n == null || (n <= hi && (loOpen ? n > lo : n >= lo))
      ? n
      : undefined;
  };
  set("rlBeta", inRange(r.beta, 0, 10));
  set("rlMaxPromptLength", inRange(r.max_prompt_length, 16, Infinity));
  if (objective !== "grpo") {
    return out;
  }
  set(
    "grpoVariant",
    GRPO_VARIANTS.find((v) => v === r.variant),
  );
  set(
    "grpoNumGenerations",
    isNum(r.num_generations) &&
      r.num_generations >= 2 &&
      r.num_generations <= 16
      ? r.num_generations
      : undefined,
  );
  set(
    "grpoMaxCompletionLength",
    inRange(r.max_completion_length, 16, Infinity),
  );
  set(
    "grpoTemperature",
    isNum(r.temperature) && r.temperature > 0 && r.temperature <= 2
      ? r.temperature
      : undefined,
  );
  set("grpoEpsilonHigh", inRange(r.epsilon_high, 0, 1, true));
  if (typeof r.mask_truncated_completions === "boolean") {
    out.grpoMaskTruncatedCompletions = r.mask_truncated_completions;
  }
  if (typeof r.enable_thinking === "boolean") {
    out.grpoEnableThinking = r.enable_thinking;
  }
  if (typeof r.system_prompt === "string") {
    out.grpoSystemPrompt = r.system_prompt;
  } else if (r.system_prompt === null) {
    out.grpoSystemPrompt = "";
  }
  if (Array.isArray(r.rewards)) {
    out.grpoRewards = r.rewards.flatMap((item) => {
      const x = item as Record<string, unknown> | null;
      return x && typeof x.name === "string" && x.name
        ? [
            {
              name: x.name,
              weight:
                isNum(x.weight) && x.weight >= -10 && x.weight <= 10
                  ? x.weight
                  : 1,
            },
          ]
        : [];
    });
  }
  return out;
}
