// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_HYPERPARAMS } from "@/config/training";
import type { DrafterTrainingRequest } from "../types/api";
import { effectiveTrainingObjective } from "./rl-roles";
import type {
  DrafterDataSource,
  DrafterStrategy,
  TrainingConfigState,
} from "../types/config";

export type DrafterState = Pick<
  TrainingConfigState,
  | "drafterEnabled"
  | "drafterStrategy"
  | "drafterDataSource"
  | "drafterMaxExamples"
  | "drafterEpochs"
  | "drafterLearningRate"
  | "drafterDraftSteps"
>;

export const DRAFTER_DEFAULTS: DrafterState = {
  drafterEnabled: DEFAULT_HYPERPARAMS.drafterEnabled,
  drafterStrategy: DEFAULT_HYPERPARAMS.drafterStrategy,
  drafterDataSource: DEFAULT_HYPERPARAMS.drafterDataSource,
  drafterMaxExamples: DEFAULT_HYPERPARAMS.drafterMaxExamples,
  drafterEpochs: DEFAULT_HYPERPARAMS.drafterEpochs,
  drafterLearningRate: DEFAULT_HYPERPARAMS.drafterLearningRate,
  drafterDraftSteps: DEFAULT_HYPERPARAMS.drafterDraftSteps,
};

// Backend bounds (models/training.py DrafterTrainingConfig): a half-typed field must not 422 the Start.
const MAX_EXAMPLES = [16, 100000] as const;
const EPOCHS = [0.1, 20] as const;
const DRAFT_STEPS = [1, 8] as const;

function clamp(value: number, [lo, hi]: readonly [number, number]): number {
  return Math.min(Math.max(value, lo), hi);
}

export function isDrafterStrategy(value: unknown): value is DrafterStrategy {
  return value === "auto" || value === "mtp" || value === "eagle3";
}

export function isDrafterDataSource(
  value: unknown,
): value is DrafterDataSource {
  return value === "dataset" || value === "regenerate";
}

function finiteOr(value: unknown, fallback: number): number {
  const n = Number(value);
  return value !== null && value !== "" && Number.isFinite(n) ? n : fallback;
}

export function normalizeDrafterMaxExamples(value: unknown): number {
  return Math.floor(
    clamp(finiteOr(value, DRAFTER_DEFAULTS.drafterMaxExamples), MAX_EXAMPLES),
  );
}

export function normalizeDrafterEpochs(value: unknown): number {
  return clamp(finiteOr(value, DRAFTER_DEFAULTS.drafterEpochs), EPOCHS);
}

export function normalizeDrafterDraftSteps(value: unknown): number {
  return Math.floor(
    clamp(finiteOr(value, DRAFTER_DEFAULTS.drafterDraftSteps), DRAFT_STEPS),
  );
}

export function normalizeDrafterLearningRate(value: unknown): number | null {
  const n = Number(value);
  return value !== null && value !== "" && Number.isFinite(n) && n > 0
    ? n
    : null;
}

/** State from a saved run's / YAML file's `drafter` block; absent or malformed = feature off. */
export function drafterStateFrom(value: unknown): DrafterState {
  if (typeof value !== "object" || value === null)
    return { ...DRAFTER_DEFAULTS };
  const v = value as Record<string, unknown>;
  return {
    drafterEnabled: v.enabled === true,
    drafterStrategy: isDrafterStrategy(v.strategy)
      ? v.strategy
      : DRAFTER_DEFAULTS.drafterStrategy,
    drafterDataSource: isDrafterDataSource(v.data_source)
      ? v.data_source
      : DRAFTER_DEFAULTS.drafterDataSource,
    drafterMaxExamples: normalizeDrafterMaxExamples(v.max_examples),
    drafterEpochs: normalizeDrafterEpochs(v.epochs),
    drafterLearningRate: normalizeDrafterLearningRate(v.learning_rate),
    drafterDraftSteps: normalizeDrafterDraftSteps(v.draft_steps),
  };
}

/** The request block for these settings, enabled or not (YAML keeps the user's settings either way). */
export function drafterRequestFrom(
  state: DrafterState,
): DrafterTrainingRequest {
  return {
    enabled: state.drafterEnabled,
    strategy: state.drafterStrategy,
    data_source: state.drafterDataSource,
    max_examples: normalizeDrafterMaxExamples(state.drafterMaxExamples),
    epochs: normalizeDrafterEpochs(state.drafterEpochs),
    learning_rate: normalizeDrafterLearningRate(state.drafterLearningRate),
    draft_steps: normalizeDrafterDraftSteps(state.drafterDraftSteps),
  };
}

/** Runs the backend's drafter phase cannot take: embeddings, decision models, CPT, DPO / ORPO / GRPO, and image / audio fine-tunes. */
export function drafterEligible(
  config: Pick<
    TrainingConfigState,
    | "modelType"
    | "isEmbeddingModel"
    | "trainingMethod"
    | "isVisionModel"
    | "isDatasetImage"
    | "isAudioModel"
    | "isDatasetAudio"
    | "trainingObjective"
  >,
): boolean {
  return (
    effectiveTrainingObjective(config) === "sft" &&
    config.modelType !== "decision" &&
    config.modelType !== "embeddings" &&
    !config.isEmbeddingModel &&
    config.trainingMethod !== "cpt" &&
    // A vision / audio model on a text dataset trains the text path, which the drafter supports.
    !(config.isVisionModel && config.isDatasetImage === true) &&
    !(config.isAudioModel && config.isDatasetAudio)
  );
}
