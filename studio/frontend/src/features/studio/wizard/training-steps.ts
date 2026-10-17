// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingObjective } from "@/types/training";

export type TrainingStep =
  | "model"
  | "dataset"
  | "prompt"
  | "rewards"
  | "parameters"
  | "review";

export function trainingSteps(objective: TrainingObjective): TrainingStep[] {
  return objective === "grpo"
    ? ["model", "dataset", "prompt", "rewards", "parameters", "review"]
    : ["model", "dataset", "parameters", "review"];
}

export function resolveTrainingStep(
  step: TrainingStep,
  objective: TrainingObjective,
): TrainingStep {
  return trainingSteps(objective).includes(step) ? step : "parameters";
}

export function trainingStepForTourTarget(target: string): TrainingStep {
  if (target === "studio-dataset") return "dataset";
  if (target === "studio-params") return "parameters";
  if (target === "studio-run-preview") return "review";
  return "model";
}
