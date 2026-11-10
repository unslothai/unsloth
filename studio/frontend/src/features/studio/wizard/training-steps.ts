// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingObjective } from "@/types/training";

export type TrainingStep =
  | "model"
  | "dataset"
  | "prompt"
  | "rewards"
  | "parameters";

/** Prompt format and rewards only apply to GRPO. */
export function trainingSteps(objective: TrainingObjective): TrainingStep[] {
  return objective === "grpo"
    ? ["model", "dataset", "prompt", "rewards", "parameters"]
    : ["model", "dataset", "parameters"];
}

/** The step to show: one the objective dropped falls through to the settings. */
export function resolveTrainingStep(
  step: TrainingStep,
  objective: TrainingObjective,
): TrainingStep {
  return trainingSteps(objective).includes(step) ? step : "parameters";
}

const TOUR_TARGET_STEP: Record<string, TrainingStep> = {
  "studio-model-picker": "model",
  "studio-method": "model",
  "studio-dataset": "dataset",
  "studio-params": "parameters",
};

export function trainingStepForTourTarget(target: string): TrainingStep | null {
  return TOUR_TARGET_STEP[target] ?? null;
}
