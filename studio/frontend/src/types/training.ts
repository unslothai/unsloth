// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "vision" | "audio" | "embeddings" | "text" | "decision";
export type TrainingMethod = "qlora" | "lora" | "full" | "cpt";

export function isTrainingMethod(value: unknown): value is TrainingMethod {
  return (
    value === "qlora" || value === "lora" || value === "full" || value === "cpt"
  );
}

export type TrainingObjective = "sft" | "dpo" | "orpo" | "grpo";

export const TRAINING_OBJECTIVES: readonly TrainingObjective[] = [
  "sft",
  "dpo",
  "orpo",
  "grpo",
];

export function isTrainingObjective(
  value: unknown,
): value is TrainingObjective {
  return TRAINING_OBJECTIVES.includes(value as TrainingObjective);
}

export type GrpoVariant = "dapo" | "dr_grpo" | "bnpo" | "grpo" | "gspo";

export const GRPO_VARIANTS: readonly GrpoVariant[] = [
  "dapo",
  "dr_grpo",
  "gspo",
  "bnpo",
  "grpo",
];

export interface GrpoRewardSelection {
  name: string;
  weight: number;
}

export function isAdapterMethod(method: TrainingMethod): boolean {
  return method === "lora" || method === "qlora" || method === "cpt";
}
export type DatasetSource = "huggingface" | "upload" | "s3";

/** S3 bucket configuration for loading datasets */
export interface S3Config {
  bucket: string;
  region: string;
  prefix?: string;
  accessKeyId?: string;
  secretAccessKey?: string;
  useIamRole?: boolean;
}
export type DatasetFormat = "auto" | "alpaca" | "chatml" | "sharegpt" | "raw";
export type GradientCheckpointing = "none" | "true" | "unsloth" | "mlx";
