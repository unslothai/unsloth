// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CPT_TARGET_MODULES,
  DEFAULT_HYPERPARAMS,
  LR_DEFAULT_CPT,
  LR_DEFAULT_FULL,
  LR_DEFAULT_LORA,
} from "@/config/training";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's unrelated React exports.
import {
  getHfToken,
  useHfTokenStore,
} from "@/features/hub/stores/hf-token-store";
import { isTrainingMethod } from "@/types/training";
import type { DatasetFormat } from "@/types/training";
import type {
  TrainingConfigState,
  TrainingConfigStore,
  TrainingMethodProvenance,
} from "../types/config";
import {
  createHfBrowseDatasetSelection,
  createUploadBrowseDatasetSelection,
  datasetSourceInvariantPatch,
} from "./training-config-policy";

export const TRAINING_CONFIG_PERSISTENCE_NAME = "unsloth_training_config_v1";
export const TRAINING_CONFIG_PERSISTENCE_VERSION = 20;

const NON_PERSISTED_STATE_KEYS: ReadonlySet<keyof TrainingConfigState> =
  new Set([
    "userEditRevision",
    "isCheckingVision",
    "isLoadingModelDefaults",
    "modelDefaultsError",
    "isCheckingDataset",
    "isDatasetImage",
    "isDatasetAudio",
    "datasetCheckFailed",
    "manualDatasetOptionsValid",
    "trainOnCompletionsDefaultPendingFor",
    "maxPositionEmbeddings",
    "s3Config",
    "wandbToken",
  ]);

export function partializeTrainingConfig(
  state: TrainingConfigStore,
): Partial<TrainingConfigStore> {
  const partial = Object.fromEntries(
    Object.entries(state).filter(([key, value]) => {
      if (typeof value === "function") {
        return false;
      }
      if (
        key === "trainOnCompletions" &&
        typeof state.trainOnCompletionsDefaultPendingFor === "string" &&
        state.trainOnCompletionsDefaultPendingFor === state.selectedModel
      ) {
        return false;
      }
      return !NON_PERSISTED_STATE_KEYS.has(key as keyof TrainingConfigState);
    }),
  ) as Partial<TrainingConfigStore>;
  return {
    ...partial,
    ...datasetSourceInvariantPatch(state),
  };
}

type PersistedTrainingConfig = Record<string, unknown>;

function migrateThroughVersion8(
  state: PersistedTrainingConfig,
  version: number,
): void {
  if (
    version < 2 &&
    state.datasetSubset == null &&
    state.datasetConfig != null
  ) {
    state.datasetSubset = state.datasetConfig;
  }
  Reflect.deleteProperty(state, "datasetConfig");
  if (version < 3 && state.modelDefaultsAppliedFor == null) {
    state.modelDefaultsAppliedFor = null;
  }
  if (version < 4 && state.optimizerType == null) {
    state.optimizerType = DEFAULT_HYPERPARAMS.optimizerType;
  }
  if (version < 5 && state.lrSchedulerType == null) {
    state.lrSchedulerType = DEFAULT_HYPERPARAMS.lrSchedulerType;
  }
  if (version < 6 && state.datasetEvalSplit == null) {
    state.datasetEvalSplit = null;
  }
  if (version < 7) {
    state.datasetSliceStart ??= null;
    state.datasetSliceEnd ??= null;
  }
  if (version < 8) {
    state.datasetSystemPrompt ??= "";
    state.datasetLabelMapping ??= {};
    state.datasetAdvisorNotification ??= null;
  }
}

function migrateThroughVersion12(
  state: PersistedTrainingConfig,
  version: number,
): void {
  if (version < 9 && state.weightDecay === 0.01) {
    state.weightDecay = DEFAULT_HYPERPARAMS.weightDecay;
  }
  if (version < 10 && state.trainingMethod === "cpt") {
    state.loraRank = 128;
    state.loraAlpha = 32;
    state.loraVariant = "rslora";
    state.targetModules = CPT_TARGET_MODULES;
    state.datasetFormat = "raw";
    if (state.learningRate == null || state.learningRate === LR_DEFAULT_LORA) {
      state.learningRate = LR_DEFAULT_CPT;
    }
  }
  if (version < 11) {
    state.datasetStreaming ??= false;
  }
  if (version < 12) {
    const legacyToken =
      typeof state.hfToken === "string" ? state.hfToken.trim() : "";
    if (legacyToken && !getHfToken()) {
      useHfTokenStore.getState().setToken(legacyToken);
    }
    state.hfToken = undefined;
  }
}

function migrateThroughVersion16(
  state: PersistedTrainingConfig,
  version: number,
): void {
  if (version < 13) {
    state.modelKnownCached ??= false;
    state.modelLocalPath ??= null;
    state.modelFormat ??= null;
    state.datasetKnownCached ??= false;
    state.datasetLocalPath ??= null;
  }
  if (version < 14) {
    const dataset = typeof state.dataset === "string" ? state.dataset : null;
    const uploadedFile =
      typeof state.uploadedFile === "string" ? state.uploadedFile : null;
    state.browseDatasetSelection =
      state.datasetSource === "upload"
        ? createUploadBrowseDatasetSelection(uploadedFile)
        : createHfBrowseDatasetSelection(dataset, {
            knownCached: state.datasetKnownCached === true,
            localPath:
              typeof state.datasetLocalPath === "string"
                ? state.datasetLocalPath
                : null,
          });
  }
  if (version < 15) {
    state.isEmbeddingModel = state.modelType === "embeddings";
  }
  if (version < 16) {
    state.datasetUserTemplate = undefined;
    state.datasetAssistantTemplate = undefined;
  }
}

function migrateThroughVersion17(
  state: PersistedTrainingConfig,
  version: number,
): void {
  if (version < 17) {
    state.modelDefaultsAppliedFor =
      typeof state.selectedModel === "string" && state.selectedModel.length > 0
        ? state.selectedModel
        : null;
    state.advancedSettingsBaseline = null;
  }
}

function migrateThroughVersion18(
  state: PersistedTrainingConfig,
  version: number,
): void {
  if (version < 18) {
    Reflect.deleteProperty(state, "wandbToken");
  }
}

function defaultLearningRateForMethod(method: unknown): number {
  if (method === "full") {
    return LR_DEFAULT_FULL;
  }
  if (method === "cpt") {
    return LR_DEFAULT_CPT;
  }
  return LR_DEFAULT_LORA;
}

function inferLegacyLearningRateWasManuallySet(
  state: PersistedTrainingConfig,
): boolean {
  return (
    typeof state.learningRate === "number" &&
    Number.isFinite(state.learningRate) &&
    state.learningRate !== defaultLearningRateForMethod(state.trainingMethod)
  );
}

function migrateThroughVersion19(
  state: PersistedTrainingConfig,
  version: number,
): void {
  if (version < 19) {
    state.trainingMethodProvenance = {
      learningRateManuallySet: inferLegacyLearningRateWasManuallySet(state),
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: null,
    } satisfies TrainingMethodProvenance;
  }
}

function isDatasetFormat(value: unknown): value is DatasetFormat {
  return (
    value === "auto" ||
    value === "alpaca" ||
    value === "chatml" ||
    value === "sharegpt" ||
    value === "raw"
  );
}

function normalizeTrainingMethodProvenance(
  value: unknown,
  persistedState: PersistedTrainingConfig,
): TrainingMethodProvenance {
  const provenance =
    value !== null && typeof value === "object"
      ? (value as Partial<TrainingMethodProvenance>)
      : {};
  const modelAdapterLearningRate = provenance.modelAdapterLearningRate;
  const datasetFormatBeforeCpt = provenance.datasetFormatBeforeCpt;
  return {
    learningRateManuallySet:
      typeof provenance.learningRateManuallySet === "boolean"
        ? provenance.learningRateManuallySet
        : inferLegacyLearningRateWasManuallySet(persistedState),
    modelAdapterLearningRate:
      typeof modelAdapterLearningRate === "number" &&
      Number.isFinite(modelAdapterLearningRate) &&
      modelAdapterLearningRate > 0
        ? modelAdapterLearningRate
        : null,
    datasetFormatBeforeCpt:
      persistedState.trainingMethod === "cpt" &&
      isDatasetFormat(datasetFormatBeforeCpt) &&
      datasetFormatBeforeCpt !== "raw"
        ? datasetFormatBeforeCpt
        : null,
  };
}

export function migrateTrainingConfig(
  persisted: unknown,
  version: number,
): TrainingConfigStore {
  const state = persisted as PersistedTrainingConfig;
  migrateThroughVersion8(state, version);
  migrateThroughVersion12(state, version);
  migrateThroughVersion16(state, version);
  migrateThroughVersion17(state, version);
  migrateThroughVersion18(state, version);
  migrateThroughVersion19(state, version);
  return state as unknown as TrainingConfigStore;
}

export function mergeTrainingConfig(
  persisted: unknown,
  current: TrainingConfigStore,
): TrainingConfigStore {
  const persistedState = persisted as Partial<TrainingConfigState>;
  const persistedRecord = persisted as PersistedTrainingConfig;
  const modelDefaultsAppliedFor =
    typeof persistedState.modelDefaultsAppliedFor === "string" &&
    persistedState.modelDefaultsAppliedFor.length > 0 &&
    persistedState.modelDefaultsAppliedFor === persistedState.selectedModel
      ? persistedState.modelDefaultsAppliedFor
      : null;
  const advancedSettingsBaseline = modelDefaultsAppliedFor
    ? (persistedState.advancedSettingsBaseline ?? null)
    : null;
  const persistedTrainOnCompletions =
    typeof persistedState.trainOnCompletions === "boolean"
      ? persistedState.trainOnCompletions
      : undefined;
  const baselineTrainOnCompletions =
    typeof advancedSettingsBaseline?.trainOnCompletions === "boolean"
      ? advancedSettingsBaseline.trainOnCompletions
      : undefined;
  const trainOnCompletions =
    persistedTrainOnCompletions ??
    baselineTrainOnCompletions ??
    current.trainOnCompletions;
  const trainOnCompletionsDefaultPendingFor =
    persistedTrainOnCompletions === undefined &&
    baselineTrainOnCompletions === undefined
      ? modelDefaultsAppliedFor
      : null;
  const merged: TrainingConfigStore = {
    ...current,
    ...persistedState,
    wandbToken: current.wandbToken,
    trainOnCompletions,
    modelDefaultsAppliedFor,
    advancedSettingsBaseline,
    trainOnCompletionsDefaultPendingFor,
    trainingMethodProvenance: normalizeTrainingMethodProvenance(
      persistedState.trainingMethodProvenance,
      persistedRecord,
    ),
    trainingMethod: isTrainingMethod(persistedState.trainingMethod)
      ? persistedState.trainingMethod
      : current.trainingMethod,
  };
  return {
    ...merged,
    ...datasetSourceInvariantPatch(merged),
  };
}
