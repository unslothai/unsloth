// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType } from "@/types/training";
import type { PipelineType } from "@huggingface/hub";
export const CONTEXT_LENGTHS = [
  512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
];

export const TARGET_MODULES = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
];

/** CPT trains embeddings via modules_to_save; keep them visible in the UI. */
export const CPT_EMBEDDING_MODULES = ["embed_tokens", "lm_head"] as const;

/** CPT requires embed_tokens and lm_head in addition to standard LoRA modules. */
export const CPT_TARGET_MODULES = [...TARGET_MODULES, ...CPT_EMBEDDING_MODULES];

const CPT_UI_TARGET_MODULES = ["all-linear", ...CPT_TARGET_MODULES] as const;

export function isCptAllLinearTargetModules(
  targetModules: readonly string[],
): boolean {
  const loraTargetModules = targetModules.filter(
    (module) =>
      !CPT_EMBEDDING_MODULES.some(
        (embeddingModule) => embeddingModule === module,
      ),
  );
  return (
    loraTargetModules.length === 1 && loraTargetModules[0] === "all-linear"
  );
}

/** Preserve all-linear model defaults for CPT. */
export function resolveCptTargetModules(
  currentTargetModules: readonly string[],
): string[] {
  if (isCptAllLinearTargetModules(currentTargetModules)) {
    return ["all-linear", ...CPT_EMBEDDING_MODULES];
  }
  return [...CPT_TARGET_MODULES];
}

export function getCptUiTargetModules(): readonly string[] {
  return CPT_UI_TARGET_MODULES;
}

export function isCptTargetModuleActive(
  targetModules: readonly string[],
  module: string,
): boolean {
  return module === "all-linear"
    ? isCptAllLinearTargetModules(targetModules)
    : targetModules.includes(module);
}

export function toggleCptTargetModule(
  targetModules: readonly string[],
  module: string,
): string[] {
  if (
    CPT_EMBEDDING_MODULES.some((embeddingModule) => embeddingModule === module)
  ) {
    return targetModules.includes(module)
      ? targetModules.filter((candidate) => candidate !== module)
      : [...targetModules, module];
  }

  if (module === "all-linear") {
    const embeddingModules = targetModules.filter((candidate) =>
      CPT_EMBEDDING_MODULES.some(
        (embeddingModule) => embeddingModule === candidate,
      ),
    );
    return isCptAllLinearTargetModules(targetModules)
      ? [...TARGET_MODULES, ...embeddingModules]
      : ["all-linear", ...embeddingModules];
  }

  const namedTargetModules = targetModules.filter(
    (candidate) => candidate !== "all-linear",
  );
  return targetModules.includes(module)
    ? namedTargetModules.filter((candidate) => candidate !== module)
    : [...namedTargetModules, module];
}

export const OPTIMIZER_OPTIONS: ReadonlyArray<{
  value: string;
  label: string;
}> = [
  { value: "adamw_8bit", label: "AdamW 8-bit" },
  { value: "paged_adamw_8bit", label: "Paged AdamW 8-bit" },
  { value: "adamw_bnb_8bit", label: "AdamW BNB 8-bit" },
  { value: "paged_adamw_32bit", label: "Paged AdamW 32-bit" },
  { value: "adamw_torch", label: "AdamW (PyTorch)" },
  { value: "adamw_torch_fused", label: "AdamW (PyTorch Fused)" },
];

// MLX trainer optimizers (Apple Silicon); must match SUPPORTED_MLX_OPTIMIZERS in unsloth-zoo's
// mlx/trainer.py. The CUDA/torch names above are remapped to AdamW on MLX.
export const MLX_OPTIMIZER_OPTIONS: ReadonlyArray<{
  value: string;
  label: string;
}> = [
  { value: "adamw", label: "AdamW" },
  { value: "adam", label: "Adam" },
  { value: "lion", label: "Lion" },
  { value: "muon", label: "Muon" },
  { value: "sgd", label: "SGD" },
  { value: "adafactor", label: "Adafactor" },
];

export const LR_SCHEDULER_OPTIONS: ReadonlyArray<{
  value: string;
  label: string;
}> = [
  { value: "linear", label: "Linear" },
  { value: "cosine", label: "Cosine" },
];

/** Method-aware learning rate defaults; the backend mirrors these in
 * studio/backend/assets/configs/. */
export const LR_DEFAULT_LORA = 2e-4;
export const LR_DEFAULT_FULL = 2e-5;
export const LR_DEFAULT_CPT = 5e-5;

export const DEFAULT_HYPERPARAMS = {
  epochs: 3,
  contextLength: 2048,
  visionImageSize: null as number | null,
  learningRate: LR_DEFAULT_LORA,
  // null = let backend auto-compute (lr/10 per Unsloth CPT recipe). Only used by CPT.
  embeddingLearningRate: null as number | null,
  optimizerType: "adamw_8bit",
  lrSchedulerType: "linear",
  loraRank: 16,
  loraAlpha: 32,
  loraDropout: 0.05,
  loraVariant: "lora" as const,
  batchSize: 4,
  gradientAccumulation: 8,
  weightDecay: 0.001,
  warmupSteps: 5,
  maxSteps: 60,
  saveSteps: 0,
  evalSteps: 0.0,
  packing: false,
  trainOnCompletions: false,
  // GRPO (RL) rollout defaults, matching Gemma3_(1B)-GRPO / Llama3.1_(8B)-GRPO.
  rewardFunctions: [
    { id: "exact_answer_match", weight: 2 },
    { id: "reasoning_format_match", weight: 1 },
    { id: "think_tag_structure", weight: 0.5 },
  ] as import("@/features/training/types/config").RewardFunctionSelection[],
  numGenerations: 4,
  maxPromptLength: 256,
  maxCompletionLength: 512,
  grpoTemperature: 1.0,
  grpoTopP: 1.0,
  grpoBeta: 0.04,
  gradientCheckpointing: "unsloth" as const,
  randomSeed: 3407,
  enableWandb: false,
  wandbToken: "",
  wandbProject: "llm-finetuning",
  enableTensorboard: false,
  tensorboardDir: "runs",
  logFrequency: 10,
  trustRemoteCode: false,
  finetuneVisionLayers: true,
  finetuneLanguageLayers: true,
  finetuneAttentionModules: true,
  finetuneMLPModules: true,
  targetModules: TARGET_MODULES,
  s3Config: null as import("@/types/training").S3Config | null,
};

export const MODEL_TYPE_TO_HF_TASKS: Record<
  ModelType,
  readonly PipelineType[]
> = {
  text: ["text-generation"],
  vision: [
    "image-text-to-text",
    "visual-question-answering",
    "document-question-answering",
    "image-to-text",
    "any-to-any",
  ],
  audio: [
    "text-to-speech",
    "automatic-speech-recognition",
    "audio-text-to-text",
    "text-to-audio",
  ],
  embeddings: ["feature-extraction"],
};

export const PRIORITY_TRAINING_MODELS: readonly string[] = [
  "unsloth/gemma-4-E2B-it",
  "unsloth/gemma-4-E4B-it",
  "unsloth/gemma-4-31B-it",
  "unsloth/gemma-4-26B-A4B-it",
  "unsloth/Qwen3.5-2B",
  "unsloth/Qwen3.5-9B",
  "unsloth/gpt-oss-20b",
  "unsloth/NVIDIA-Nemotron-3-Nano-4B",
  "unsloth/Qwen3-0.6B",
  "unsloth/gemma-3-4b-it",
  "unsloth/embeddinggemma-300m",
  "unsloth/orpheus-3b-0.1-ft",
  "unsloth/Llama-3.1-8B-Instruct",
  "unsloth/Llama-3.2-3B-Instruct",
];

/** Pin priority models to the top of a list of model IDs, preserving their defined order. */
export function applyPriorityOrdering(ids: string[]): string[] {
  const idSet = new Set(ids);
  const pinned = PRIORITY_TRAINING_MODELS.filter((id) => idSet.has(id));
  const pinnedSet = new Set(pinned);
  const rest = ids.filter((id) => !pinnedSet.has(id));
  return [...pinned, ...rest];
}
