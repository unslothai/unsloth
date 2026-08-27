// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CPT_TARGET_MODULES,
  DEFAULT_HYPERPARAMS,
  LR_DEFAULT_CPT,
  LR_DEFAULT_FULL,
  LR_DEFAULT_LORA,
  TARGET_MODULES,
} from "@/config/training";
import { isAdapterMethod } from "@/types/training";
import type { TrainingMethod } from "@/types/training";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import type { TrainingConfigState } from "../types/config";

type TrainingMethodStatePatch = Partial<
  Pick<
    TrainingConfigState,
    | "trainingMethod"
    | "learningRate"
    | "loraRank"
    | "loraAlpha"
    | "loraVariant"
    | "targetModules"
    | "datasetFormat"
    | "trainOnCompletions"
    | "trainingMethodProvenance"
  >
>;

function arraysEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

// A model whose default target modules differ from the generic Llama-style
// list (e.g. LFM2's "all-linear") has declared an architecture-specific LoRA
// target. CPT must not clobber that with the hard-coded Llama fallback.
function resolveCptTargetModules(currentTargetModules: string[]): string[] {
  const isModelSpecific =
    currentTargetModules.length > 0 &&
    !arraysEqual(currentTargetModules, TARGET_MODULES);
  return isModelSpecific ? currentTargetModules : CPT_TARGET_MODULES;
}

function getCptTrainingPatch(
  currentTargetModules: string[],
): TrainingMethodStatePatch {
  return {
    loraRank: 128,
    loraAlpha: 32,
    loraVariant: "rslora",
    targetModules: resolveCptTargetModules(currentTargetModules),
    datasetFormat: "raw",
    trainOnCompletions: false,
  };
}

export function getCptModelDefaultsPatch(
  currentTargetModules: string[],
): TrainingMethodStatePatch {
  return {
    ...getCptTrainingPatch(currentTargetModules),
    learningRate: LR_DEFAULT_CPT,
  };
}

function getRestoreFromCptPatch(): TrainingMethodStatePatch {
  return {
    loraRank: DEFAULT_HYPERPARAMS.loraRank,
    loraAlpha: DEFAULT_HYPERPARAMS.loraAlpha,
    loraVariant: DEFAULT_HYPERPARAMS.loraVariant,
    targetModules: TARGET_MODULES,
  };
}

function resolveTrainingMethodLearningRate(
  prevMethod: TrainingMethod,
  nextMethod: TrainingMethod,
  learningRateManuallySet: boolean,
  modelAdapterLearningRate: number | null,
): number | undefined {
  if (learningRateManuallySet) {
    return undefined;
  }

  const wasCpt = prevMethod === "cpt";
  const wasAdapter = isAdapterMethod(prevMethod);
  const nowAdapter = isAdapterMethod(nextMethod);

  if (nextMethod === "cpt") {
    return LR_DEFAULT_CPT;
  }
  if (wasCpt && nowAdapter) {
    return modelAdapterLearningRate ?? LR_DEFAULT_LORA;
  }
  if (wasAdapter && nowAdapter) {
    return undefined;
  }
  return nowAdapter
    ? (modelAdapterLearningRate ?? LR_DEFAULT_LORA)
    : LR_DEFAULT_FULL;
}

export function buildTrainingMethodPatch(
  state: Pick<
    TrainingConfigState,
    | "trainingMethod"
    | "trainingMethodProvenance"
    | "datasetFormat"
    | "targetModules"
  >,
  nextMethod: TrainingMethod,
): TrainingMethodStatePatch {
  const prevMethod = state.trainingMethod;
  const patch: TrainingMethodStatePatch = { trainingMethod: nextMethod };
  const provenance = { ...state.trainingMethodProvenance };

  if (prevMethod !== "cpt" && nextMethod === "cpt") {
    provenance.datasetFormatBeforeCpt = isRawTextDatasetFormat(
      state.datasetFormat,
    )
      ? null
      : state.datasetFormat;
    Object.assign(patch, getCptTrainingPatch(state.targetModules));
  }
  if (prevMethod === "cpt" && nextMethod !== "cpt") {
    Object.assign(patch, getRestoreFromCptPatch());
    if (provenance.datasetFormatBeforeCpt !== null) {
      patch.datasetFormat = provenance.datasetFormatBeforeCpt;
    }
    provenance.datasetFormatBeforeCpt = null;
  }

  const learningRate = resolveTrainingMethodLearningRate(
    prevMethod,
    nextMethod,
    provenance.learningRateManuallySet,
    provenance.modelAdapterLearningRate,
  );
  if (learningRate !== undefined) {
    patch.learningRate = learningRate;
  }
  patch.trainingMethodProvenance = provenance;
  return patch;
}
