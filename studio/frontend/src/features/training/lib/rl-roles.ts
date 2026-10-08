// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingObjective } from "@/types/training";

export type RlObjective = Exclude<TrainingObjective, "sft">;

/** RL needs a text dataset and a text or vision-language model; the backend refuses the rest. */
export function rlObjectiveSupported(s: {
  modelType: string | null;
  isEmbeddingModel?: boolean;
  isDatasetImage?: boolean | null;
  isDatasetAudio?: boolean;
}): boolean {
  return (
    !s.isEmbeddingModel &&
    !s.isDatasetImage &&
    !s.isDatasetAudio &&
    s.modelType !== "decision" &&
    s.modelType !== "embeddings" &&
    s.modelType !== "audio"
  );
}

/** The objective a run will actually use. */
export function effectiveTrainingObjective(s: {
  modelType: string | null;
  isEmbeddingModel?: boolean;
  isDatasetImage?: boolean | null;
  isDatasetAudio?: boolean;
  trainingObjective: TrainingObjective;
}): TrainingObjective {
  return rlObjectiveSupported(s) ? s.trainingObjective : "sft";
}
export type RlRole = "prompt" | "answer" | "chosen" | "rejected" | "system";

// Mirrors RL_ROLES / _REQUIRED_ROLES / _AUTO_ROLE_NAMES in studio/backend/core/training/rl.py.
export const RL_ROLES: Record<RlObjective, readonly RlRole[]> = {
  dpo: ["prompt", "chosen", "rejected", "system"],
  orpo: ["prompt", "chosen", "rejected", "system"],
  grpo: ["prompt", "answer", "system"],
};
export const RL_REQUIRED_ROLES: Record<RlObjective, readonly RlRole[]> = {
  dpo: ["prompt", "chosen", "rejected"],
  orpo: ["prompt", "chosen", "rejected"],
  grpo: ["prompt"],
};
const AUTO_NAMES: Record<RlRole, readonly string[]> = {
  prompt: ["prompt", "question", "instruction", "problem", "query", "input"],
  answer: ["answer", "solution", "final_answer", "target", "label"],
  chosen: ["chosen", "accepted", "preferred"],
  rejected: ["rejected", "dispreferred"],
  system: ["system", "system_prompt"],
};

/** Mapping limited to the objective's roles, with unmapped roles filled from column names. */
export function resolveRlMapping(
  objective: RlObjective,
  columns: readonly string[],
  mapping: Record<string, string>,
): Record<string, string> {
  const roles = RL_ROLES[objective];
  const out: Record<string, string> = {};
  for (const [column, role] of Object.entries(mapping)) {
    if (
      columns.includes(column) &&
      (roles as readonly string[]).includes(role) &&
      !Object.values(out).includes(role)
    ) {
      out[column] = role;
    }
  }
  for (const role of roles) {
    if (Object.values(out).includes(role)) {
      continue;
    }
    const column = columns.find(
      (c) => AUTO_NAMES[role].includes(c.toLowerCase()) && !(c in out),
    );
    if (column) {
      out[column] = role;
    }
  }
  return out;
}

export function missingRlRoles(
  objective: RlObjective,
  columns: readonly string[],
  mapping: Record<string, string>,
): RlRole[] {
  const mapped = new Set(
    Object.values(resolveRlMapping(objective, columns, mapping)),
  );
  return RL_REQUIRED_ROLES[objective].filter((role) => !mapped.has(role));
}
