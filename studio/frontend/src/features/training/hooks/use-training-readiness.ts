


import { usePlatformStore } from "@/config/env";
import { useMemo } from "react";
import {
  type StartValidationResult,
  hasIncompatibleTrainingModalities,
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
  modelError: string | null;
  configValidation: StartValidationResult;
}

function deriveTrainingReadiness(
  state: TrainingConfigState,
  deviceType: string,
): TrainingReadiness {
  const configValidation = validateTrainingConfig(state, deviceType);
  const hasModel = !!state.selectedModel;
  const hasDataset =
    state.datasetSource === "upload"
      ? !!state.uploadedFile
      : state.datasetSource === "s3"
        ? validateS3Source(state).ok
        : !!state.dataset;
  const isLoadingModel = state.isLoadingModelDefaults || state.isCheckingVision;
  const modelError = state.modelDefaultsError;
  const isModelCapabilitiesSettled = hasModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled && hasIncompatibleTrainingModalities(state);
  const hasDatasetModalityMetadata = state.isDatasetImage !== null;
  const modelHandlesAllModalities = state.isVisionModel && state.isAudioModel;
  const datasetUnverified =
    isModelCapabilitiesSettled &&
    !state.isCheckingDataset &&
    state.datasetCheckFailed &&
    !(hasDatasetModalityMetadata && modelHandlesAllModalities);

  return {
    isReady:
      hasModel &&
      hasDataset &&
      !isLoadingModel &&
      !state.isCheckingDataset &&
      !isIncompatible &&
      configValidation.ok,
    isLoadingModel,
    isCheckingDataset: state.isCheckingDataset,
    hasModel,
    hasDataset,
    isIncompatible,
    datasetUnverified,
    modelError,
    configValidation,
  };
}

function readinessEqual(
  current: Readonly<TrainingReadiness>,
  next: TrainingReadiness,
): boolean {
  return (
    current.isReady === next.isReady &&
    current.isLoadingModel === next.isLoadingModel &&
    current.isCheckingDataset === next.isCheckingDataset &&
    current.hasModel === next.hasModel &&
    current.hasDataset === next.hasDataset &&
    current.isIncompatible === next.isIncompatible &&
    current.datasetUnverified === next.datasetUnverified &&
    current.modelError === next.modelError &&
    current.configValidation.ok === next.configValidation.ok &&
    current.configValidation.errorKey === next.configValidation.errorKey
  );
}

function createTrainingReadinessSelector(deviceType: string) {
  let cachedReadiness: Readonly<TrainingReadiness> | null = null;
  let cachedState: TrainingConfigState | null = null;

  return (state: TrainingConfigState): Readonly<TrainingReadiness> => {
    if (state === cachedState && cachedReadiness) {
      return cachedReadiness;
    }
    const next = deriveTrainingReadiness(state, deviceType);
    if (!(cachedReadiness && readinessEqual(cachedReadiness, next))) {
      cachedReadiness = Object.freeze(next);
    }
    cachedState = state;
    return cachedReadiness;
  };
}

export function useTrainingReadiness(): Readonly<TrainingReadiness> {
  const deviceType = usePlatformStore((state) => state.deviceType);
  const selector = useMemo(
    () => createTrainingReadinessSelector(deviceType),
    [deviceType],
  );
  return useTrainingConfigStore(selector);
}
