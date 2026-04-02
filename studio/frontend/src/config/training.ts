// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType, StepConfig } from "@/types/training";
import type { PipelineType } from "@huggingface/hub";

export const STEPS: StepConfig[] = [
  {
    number: 1,
    title: "Model Type",
    subtitle: "Select type",
    description: "Choose the type of model you want to fine-tune",
  },
  {
    number: 2,
    title: "Model",
    subtitle: "Select model",
    description: "Choose a base model and training method",
  },
  {
    number: 3,
    title: "Dataset",
    subtitle: "Add dataset",
    description: "Select or upload a training dataset",
  },
  {
    number: 4,
    title: "Parameters",
    subtitle: "Configure",
    description: "Fine-tune your training hyperparameters",
  },
  {
    number: 5,
    title: "Summary",
    subtitle: "Review",
    description: "Review your configuration before starting",
  },
];

export const MODEL_TYPES: ReadonlyArray<{
  value: ModelType;
  label: string;
  description: string;
}> = [
  {
    value: "text",
    label: "Text",
    description: "Language models",
  },
    {
      value: "vision",
      label: "Vision",
      description: "Image understanding models",
    },
    {
      value: "audio",
      label: "Audio",
      description: "Audio and speech models",
    },
    {
      value: "embeddings",
      label: "Embeddings",
      description: "Text embedding models",
    },
  ];

export const CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

export const TARGET_MODULES = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
];

export const OPTIMIZER_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "adamw_8bit", label: "AdamW 8-bit" },
  { value: "paged_adamw_8bit", label: "Paged AdamW 8-bit" },
  { value: "adamw_bnb_8bit", label: "AdamW BNB 8-bit" },
  { value: "paged_adamw_32bit", label: "Paged AdamW 32-bit" },
  { value: "adamw_torch", label: "AdamW (PyTorch)" },
  { value: "adamw_torch_fused", label: "AdamW (PyTorch Fused)" },
];

export const LR_SCHEDULER_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "linear", label: "Linear" },
  { value: "cosine", label: "Cosine" },
];

/**
 * Method-aware learning rate defaults.
 * Backend mirrors these in the YAML configs under studio/backend/assets/configs/.
 */
export const LR_DEFAULT_LORA = 2e-4;
export const LR_DEFAULT_FULL = 2e-5;

export const DEFAULT_HYPERPARAMS = {
  epochs: 3,
  contextLength: 2048,
  learningRate: LR_DEFAULT_LORA,
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
  evalSteps: 0.00,
  packing: false,
  trainOnCompletions: false,
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
};

export const MODEL_TYPE_TO_HF_TASK: Record<ModelType, PipelineType> = {
  text: "text-generation",
  vision: "image-text-to-text",
  audio: "text-to-speech",
  embeddings: "feature-extraction",
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
