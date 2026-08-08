// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  hasSeparateStreamingEvalSplit,
  useTrainingConfigStore,
} from "./stores/training-config-store";
export {
  isTrainingRunActive,
  isTrainingStartPending,
  shouldShowTrainingView,
  useTrainingRuntimeStore,
} from "./stores/training-runtime-store";
export { useTrainingActions } from "./hooks/use-training-actions";

export {
  getTrainingRunDisplayTitle,
  getTrainingRunModelSubtitle,
  shouldShowTrainingArtifactsDeleted,
} from "./lib/run-display";
export {
  getTrainingMethodLabel,
  isRawTextDatasetFormat,
  isTrainingLoraVariantSupportedOnDevice,
  isTrainingMethodSupportedOnDevice,
  isTrainingModelTypeSupportedOnDevice,
  parseBackendTrainingMethod,
} from "./lib/training-methods";
export { useTrainingHistorySidebarItems } from "./hooks/use-training-history-sidebar";
export { useTrainingRuntimeLifecycle } from "./hooks/use-training-runtime-lifecycle";
export { useTrainingCompletionWatch } from "./hooks/use-training-completion-watch";
export {
  removeTrainingUnloadGuard,
  useTrainingUnloadGuard,
} from "./hooks/use-training-unload-guard";
export { useMaxStepsEpochsToggle } from "./hooks/use-max-steps-epochs-toggle";
export { HfDatasetSubsetSplitSelectors } from "./components/hf-dataset-subset-split-selectors";
export { useDatasetPreviewDialogStore } from "./stores/dataset-preview-dialog-store";
export {
  DatasetFormatError,
  aiAssistMapping,
  checkDatasetFormat,
  uploadNativeTrainingDataset,
  uploadTrainingDataset,
} from "./api/datasets-api";
export { clearDeletedDataset } from "./stores/training-config-store";
export type { CheckFormatResponse } from "./types/datasets";
export type {
  AdvancedSettingsBaseline,
  LoraVariant,
  TrainingConfigState,
} from "./types/config";
export { getModelConfig, listLocalModels } from "./api/models-api";
export type { LocalModelInfo, ModelConfigResponse } from "./api/models-api";
export type {
  TrainingPhase,
  TrainingViewData,
  TrainingSeriesPoint,
} from "./types/runtime";
export type {
  TrainingRunSummary,
  TrainingRunListResponse,
  TrainingRunMetrics,
  TrainingRunDetailResponse,
  TrainingRunDeleteResponse,
} from "./types/history";
export {
  listTrainingRuns,
  getTrainingRun,
  deleteTrainingRun,
  renameTrainingRun,
  HistoryRequestError,
} from "./api/history-api";
export {
  onTrainingRunUpdated,
  onTrainingRunDeleted,
  onTrainingRunsChanged,
  emitTrainingRunUpdated,
  emitTrainingRunDeleted,
  emitTrainingRunsChanged,
} from "./events";
export { parseYamlConfig, serializeConfigToYaml } from "./lib/yaml-config";
export {
  type StartValidationResult,
  validateTrainingConfig,
} from "./lib/validation";
export { useTrainingReadiness } from "./hooks/use-training-readiness";
export { useTrainingResourceNotices } from "./hooks/use-training-resource-notices";
export {
  cacheLocalPathMatchesSelection,
  cachedInventoryPathMatchesSelection,
} from "./lib/cache-reference";
export {
  createDatasetCacheUsabilityIdentity,
  datasetCacheUsabilityIdentitiesEqual,
  trainingDatasetCacheRejections,
} from "./lib/dataset-cache-rejection";
export type {
  DatasetCacheInventoryIdentity,
  DatasetCacheUsabilityIdentity,
} from "./lib/dataset-cache-rejection";
export { validateTrainingModelCandidate } from "./lib/freeform-model-validation";
export { isLocalTrainingModelSelection } from "./lib/model-selection";
export {
  isHuggingFaceDatasetSelected,
  resolveDeletedLocalDatasetSelection,
  shouldClearMissingLocalDatasetSelection,
} from "./lib/dataset-selection";
export {
  TRAINING_DATASET_UPLOAD_ACCEPT,
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
  classifyNativeTrainingDatasetDrop,
  isTrainingDatasetUploadPath,
  nativeDropPositionHitsBounds,
} from "./lib/native-dataset-drop";
export {
  isTrainableModelFormat,
  isUntrainableModelFormat,
} from "./lib/model-support";
export {
  inferTrainingModelTypeFromFlags,
  type ModelTypeCapabilityFlags,
} from "./lib/model-type-capabilities";
export { trainingModelTypeFlagsFromMetadata } from "./lib/model-type-inference";
export { trainingModelMatchesTypeConstraint } from "./lib/model-type-constraint";
export {
  buildCachedTrainingModelLookup,
  buildLocalTrainingModelLookup,
} from "./lib/training-picker-lookups";
export {
  TRAINING_METHOD_META,
  TRAINING_METHOD_ORDER,
} from "./lib/training-method-meta";
export {
  LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY,
  TRAINING_DATASET_PICKER_TAB_STORAGE_KEY,
  TRAINING_MODEL_PICKER_TAB_STORAGE_KEY,
  TRAINING_PARAM_MODE_STORAGE_KEY,
  TRAINING_UI_PREFERENCE_KEYS,
} from "./lib/training-ui-preferences";
