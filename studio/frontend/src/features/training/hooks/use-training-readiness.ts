// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import { useMemo } from "react";
import { effectiveTrainingObjective } from "../lib/rl-roles";
import {
  type StartValidationResult,
  hasIncompatibleTrainingModalities,
  validateS3Source,
  validateTrainingConfig,
} from "../lib/validation";
import { useRlWorkspaceStore } from "../stores/rl-workspace-store";
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
  // Roles or reward columns the RL objective needs and the dataset lacks.
  rlMissing: readonly string[];
}

export function deriveTrainingReadiness(
  state: TrainingConfigState,
  deviceType: string,
  isOwner: boolean,
  rlStartMissing: readonly string[] | null = null,
): TrainingReadiness {
  const configValidation = validateTrainingConfig(state, deviceType, isOwner);
  const hasModel = !!state.selectedModel;
  const hasDataset =
    state.datasetSource === "upload"
      ? !!state.uploadedFile
      : state.datasetSource === "s3"
        ? validateS3Source(state).ok
        : !!state.dataset;
  const isLoadingModel = state.isLoadingModelDefaults || state.isCheckingVision;
  const modelError = state.modelDefaultsError;
  // Without its config a decision model has no recipe or checkpoint, so the run would train the wrong one.
  const decisionConfigMissing =
    state.modelType === "decision" && modelError !== null;
  const isModelCapabilitiesSettled = hasModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled &&
    hasIncompatibleTrainingModalities(state, deviceType);
  const hasDatasetModalityMetadata = state.isDatasetImage !== null;
  const modelHandlesAllModalities = state.isVisionModel && state.isAudioModel;
  const datasetUnverified =
    isModelCapabilitiesSettled &&
    !state.isCheckingDataset &&
    state.datasetCheckFailed &&
    !(hasDatasetModalityMetadata && modelHandlesAllModalities);
  const rlMissing =
    effectiveTrainingObjective(state) !== "sft" &&
    state.trainingMethod !== "cpt"
      ? (rlStartMissing ?? [])
      : [];

  return {
    isReady:
      hasModel &&
      hasDataset &&
      !isLoadingModel &&
      !state.isCheckingDataset &&
      !isIncompatible &&
      !decisionConfigMissing &&
      rlMissing.length === 0 &&
      configValidation.ok,
    isLoadingModel,
    isCheckingDataset: state.isCheckingDataset,
    hasModel,
    hasDataset,
    isIncompatible,
    datasetUnverified,
    modelError,
    configValidation,
    rlMissing,
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
    current.configValidation.errorKey === next.configValidation.errorKey &&
    current.rlMissing.join("\n") === next.rlMissing.join("\n")
  );
}

function createTrainingReadinessSelector(
  deviceType: string,
  isOwner: boolean,
  rlStartMissing: readonly string[] | null,
) {
  let cachedReadiness: Readonly<TrainingReadiness> | null = null;
  let cachedState: TrainingConfigState | null = null;

  return (state: TrainingConfigState): Readonly<TrainingReadiness> => {
    if (state === cachedState && cachedReadiness) {
      return cachedReadiness;
    }
    const next = deriveTrainingReadiness(
      state,
      deviceType,
      isOwner,
      rlStartMissing,
    );
    if (!(cachedReadiness && readinessEqual(cachedReadiness, next))) {
      cachedReadiness = Object.freeze(next);
    }
    cachedState = state;
    return cachedReadiness;
  };
}

export function useTrainingReadiness(): Readonly<TrainingReadiness> {
  const deviceType = usePlatformStore((state) => state.deviceType);
  const isOwner = useIsAccountOwner();
  const rlStartMissing = useRlWorkspaceStore((state) => state.startMissing);
  const selector = useMemo(
    () => createTrainingReadinessSelector(deviceType, isOwner, rlStartMissing),
    [deviceType, isOwner, rlStartMissing],
  );
  return useTrainingConfigStore(selector);
}
