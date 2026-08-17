// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CPT_TARGET_MODULES, DEFAULT_HYPERPARAMS } from "@/config/training";
import type {
  AdvancedSettingsBaseline,
  TrainingConfigState,
} from "@/features/training";
import { isAdapterMethod } from "@/types/training";

type AdvancedSettingsState = Pick<
  TrainingConfigState,
  "trainingMethod" | keyof AdvancedSettingsBaseline
>;

const SCALAR_DEFAULTS = {
  optimizerType: DEFAULT_HYPERPARAMS.optimizerType,
  lrSchedulerType: DEFAULT_HYPERPARAMS.lrSchedulerType,
  weightDecay: DEFAULT_HYPERPARAMS.weightDecay,
  warmupSteps: DEFAULT_HYPERPARAMS.warmupSteps,
  saveSteps: DEFAULT_HYPERPARAMS.saveSteps,
  evalSteps: DEFAULT_HYPERPARAMS.evalSteps,
  randomSeed: DEFAULT_HYPERPARAMS.randomSeed,
  packing: DEFAULT_HYPERPARAMS.packing,
  trainOnCompletions: DEFAULT_HYPERPARAMS.trainOnCompletions,
  gradientCheckpointing: DEFAULT_HYPERPARAMS.gradientCheckpointing,
  visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
  finetuneVisionLayers: DEFAULT_HYPERPARAMS.finetuneVisionLayers,
  finetuneLanguageLayers: DEFAULT_HYPERPARAMS.finetuneLanguageLayers,
  finetuneAttentionModules: DEFAULT_HYPERPARAMS.finetuneAttentionModules,
  finetuneMLPModules: DEFAULT_HYPERPARAMS.finetuneMLPModules,
} as const;

function sameValues(
  left: readonly string[],
  right: readonly string[],
): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();
  return sortedLeft.every((value, index) => value === sortedRight[index]);
}

function baselineValue<K extends keyof AdvancedSettingsBaseline>(
  baseline: AdvancedSettingsBaseline | null,
  key: K,
  fallback: AdvancedSettingsState[K],
): AdvancedSettingsState[K] {
  const value = baseline?.[key];
  return value === undefined ? fallback : value;
}

function countNonDefaultScalarSettings(
  state: AdvancedSettingsState,
  baseline: AdvancedSettingsBaseline | null,
): number {
  return Object.entries(SCALAR_DEFAULTS).reduce((total, [rawKey, value]) => {
    const key = rawKey as keyof typeof SCALAR_DEFAULTS;
    const expected = baselineValue(baseline, key, value);
    return state[key] === expected ? total : total + 1;
  }, 0);
}

function countNonDefaultLoraSettings(
  state: AdvancedSettingsState,
  baseline: AdvancedSettingsBaseline | null,
): number {
  const isCpt = state.trainingMethod === "cpt";
  const loraDefaults = {
    loraRank: isCpt
      ? 128
      : baselineValue(baseline, "loraRank", DEFAULT_HYPERPARAMS.loraRank),
    loraAlpha: isCpt
      ? 32
      : baselineValue(baseline, "loraAlpha", DEFAULT_HYPERPARAMS.loraAlpha),
    loraDropout: baselineValue(
      baseline,
      "loraDropout",
      DEFAULT_HYPERPARAMS.loraDropout,
    ),
    loraVariant: isCpt
      ? "rslora"
      : baselineValue(baseline, "loraVariant", DEFAULT_HYPERPARAMS.loraVariant),
  } as const;
  const count = Object.entries(loraDefaults).reduce(
    (total, [key, value]) =>
      state[key as keyof typeof loraDefaults] === value ? total : total + 1,
    0,
  );
  const defaultTargetModules = isCpt
    ? CPT_TARGET_MODULES
    : baselineValue(
        baseline,
        "targetModules",
        DEFAULT_HYPERPARAMS.targetModules,
      );
  return sameValues(state.targetModules, defaultTargetModules)
    ? count
    : count + 1;
}

export function countNonDefaultAdvancedSettings(
  state: AdvancedSettingsState,
  baseline: AdvancedSettingsBaseline | null = null,
): number {
  const scalarCount = countNonDefaultScalarSettings(state, baseline);
  if (!isAdapterMethod(state.trainingMethod)) {
    return scalarCount;
  }
  return scalarCount + countNonDefaultLoraSettings(state, baseline);
}
