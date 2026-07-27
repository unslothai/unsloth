// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type StartValidationResult,
  validateS3Source,
  validateTrainingConfig,
} from "../lib/validation";
import { useTrainingConfigStore } from "../stores/training-config-store";
import type { TrainingConfigState } from "../types/config";

export interface TrainingReadiness {
  isReady: boolean;
  isLoadingModel: boolean;
  isCheckingDataset: boolean;
  hasModel: boolean;
  hasDataset: boolean;
  isIncompatible: boolean;
  datasetUnverified: boolean;
  configValidation: StartValidationResult;
}

let readinessCache: Readonly<TrainingReadiness> | null = null;
let readinessStateCache: TrainingConfigState | null = null;

function stableReadiness(next: TrainingReadiness): Readonly<TrainingReadiness> {
  if (
    readinessCache &&
    readinessCache.isReady === next.isReady &&
    readinessCache.isLoadingModel === next.isLoadingModel &&
    readinessCache.isCheckingDataset === next.isCheckingDataset &&
    readinessCache.hasModel === next.hasModel &&
    readinessCache.hasDataset === next.hasDataset &&
    readinessCache.isIncompatible === next.isIncompatible &&
    readinessCache.datasetUnverified === next.datasetUnverified &&
    readinessCache.configValidation.ok === next.configValidation.ok &&
    readinessCache.configValidation.message === next.configValidation.message
  ) {
    return readinessCache;
  }
  readinessCache = Object.freeze(next);
  return readinessCache;
}

function selectTrainingReadiness(
  state: TrainingConfigState,
): Readonly<TrainingReadiness> {
  if (state === readinessStateCache && readinessCache) {
    return readinessCache;
  }
  const configValidation = validateTrainingConfig(state);
  const hasModel = !!state.selectedModel;
  const hasDataset =
    state.datasetSource === "upload"
      ? !!state.uploadedFile
      : state.datasetSource === "s3"
        ? validateS3Source(state).ok
        : !!state.dataset;
  const isLoadingModel = state.isLoadingModelDefaults || state.isCheckingVision;
  const isModelCapabilitiesSettled = hasModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled &&
    ((!state.isVisionModel && state.isDatasetImage === true) ||
      (!state.isAudioModel && state.isDatasetAudio === true));
  const hasDatasetModalityMetadata = state.isDatasetImage !== null;
  const modelHandlesAllModalities = state.isVisionModel && state.isAudioModel;
  const datasetUnverified =
    isModelCapabilitiesSettled &&
    !state.isCheckingDataset &&
    state.datasetCheckFailed &&
    !(hasDatasetModalityMetadata && modelHandlesAllModalities);

  const isReady =
    hasModel &&
    hasDataset &&
    !isLoadingModel &&
    !state.isCheckingDataset &&
    !isIncompatible &&
    configValidation.ok;

  const readiness = stableReadiness({
    isReady,
    isLoadingModel,
    isCheckingDataset: state.isCheckingDataset,
    hasModel,
    hasDataset,
    isIncompatible,
    datasetUnverified,
    configValidation,
  });
  readinessStateCache = state;
  return readiness;
}

export function useTrainingReadiness(): Readonly<TrainingReadiness> {
  return useTrainingConfigStore(selectTrainingReadiness);
}
