// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useShallow } from "zustand/react/shallow";
import { useTrainingConfigStore } from "../stores/training-config-store";
import {
  type StartValidationResult,
  validateTrainingConfig,
} from "../lib/validation";

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

export function useTrainingReadiness(): TrainingReadiness {
  const {
    selectedModel,
    isLoadingModelDefaults,
    isCheckingVision,
    isCheckingDataset,
    isVisionModel,
    isAudioModel,
    isDatasetImage,
    isDatasetAudio,
    datasetCheckFailed,
    datasetKnownCached,
    dataset,
    uploadedFile,
    datasetSource,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      isLoadingModelDefaults: s.isLoadingModelDefaults,
      isCheckingVision: s.isCheckingVision,
      isCheckingDataset: s.isCheckingDataset,
      isVisionModel: s.isVisionModel,
      isAudioModel: s.isAudioModel,
      isDatasetImage: s.isDatasetImage,
      isDatasetAudio: s.isDatasetAudio,
      datasetCheckFailed: s.datasetCheckFailed,
      datasetKnownCached: s.datasetKnownCached,
      dataset: s.dataset,
      uploadedFile: s.uploadedFile,
      datasetSource: s.datasetSource,
    })),
  );
  const configValidation = useTrainingConfigStore(
    useShallow((s) => validateTrainingConfig(s)),
  );

  const hasModel = !!selectedModel;
  const hasDataset =
    datasetSource === "upload" ? !!uploadedFile : !!dataset;
  const isLoadingModel = isLoadingModelDefaults || isCheckingVision;
  const isModelCapabilitiesSettled = hasModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled &&
    ((!isVisionModel && isDatasetImage === true) ||
      (!isAudioModel && isDatasetAudio === true));
  const modelHandlesAllModalities = isVisionModel && isAudioModel;
  const datasetUnverified =
    isModelCapabilitiesSettled &&
    !isCheckingDataset &&
    datasetCheckFailed &&
    !modelHandlesAllModalities &&
    !datasetKnownCached;

  const isReady =
    hasModel &&
    hasDataset &&
    !isLoadingModel &&
    !isCheckingDataset &&
    !isIncompatible &&
    !datasetUnverified &&
    configValidation.ok;

  return {
    isReady,
    isLoadingModel,
    isCheckingDataset,
    hasModel,
    hasDataset,
    isIncompatible,
    datasetUnverified,
    configValidation,
  };
}
