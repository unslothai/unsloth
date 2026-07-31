// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CPT_TARGET_MODULES, DEFAULT_HYPERPARAMS } from "@/config/training";
import type { LoraVariant } from "@/features/training";
import {
  type GradientCheckpointing,
  type TrainingMethod,
  isAdapterMethod,
} from "@/types/training";

interface AdvancedSettingsState {
  trainingMethod: TrainingMethod;
  optimizerType: string;
  lrSchedulerType: string;
  weightDecay: number;
  warmupSteps: number;
  saveSteps: number;
  evalSteps: number;
  randomSeed: number;
  packing: boolean;
  trainOnCompletions: boolean;
  gradientCheckpointing: GradientCheckpointing;
  visionImageSize: number | null;
  finetuneVisionLayers: boolean;
  finetuneLanguageLayers: boolean;
  finetuneAttentionModules: boolean;
  finetuneMLPModules: boolean;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  loraVariant: LoraVariant;
  targetModules: string[];
}

function sameValues(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return (
    left.length === right.length && left.every((value) => right.includes(value))
  );
}

export function countNonDefaultAdvancedSettings(
  state: AdvancedSettingsState,
): number {
  const scalarDefaults = {
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
  let count = Object.entries(scalarDefaults).reduce(
    (total, [key, value]) =>
      state[key as keyof typeof scalarDefaults] === value ? total : total + 1,
    0,
  );

  if (!isAdapterMethod(state.trainingMethod)) {
    return count;
  }

  const isCpt = state.trainingMethod === "cpt";
  const loraDefaults = {
    loraRank: isCpt ? 128 : DEFAULT_HYPERPARAMS.loraRank,
    loraAlpha: isCpt ? 32 : DEFAULT_HYPERPARAMS.loraAlpha,
    loraDropout: DEFAULT_HYPERPARAMS.loraDropout,
    loraVariant: isCpt ? "rslora" : DEFAULT_HYPERPARAMS.loraVariant,
  } as const;
  count += Object.entries(loraDefaults).reduce(
    (total, [key, value]) =>
      state[key as keyof typeof loraDefaults] === value ? total : total + 1,
    0,
  );
  const defaultTargetModules = isCpt
    ? CPT_TARGET_MODULES
    : DEFAULT_HYPERPARAMS.targetModules;
  if (!sameValues(state.targetModules, defaultTargetModules)) {
    count += 1;
  }
  return count;
}
