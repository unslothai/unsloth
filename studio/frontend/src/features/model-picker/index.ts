// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { ModelSelector } from "./components/model-selector";
export { FolderBrowser } from "./components/model-selector/folder-browser";
export { invalidateLlamaFlagCatalog } from "./api/llama-flags";
export { ModelRowMenu } from "./components/model-selector/model-row-menu";
export { formatFootprintBytes } from "./components/model-selector/pickers";
export {
  makePinRank,
  pinKey,
  usePinnedModelsStore,
} from "./components/model-selector/pinned-models";
export {
  pinnedReasoningEffort,
  useModelReasoningEffortStore,
} from "./components/model-selector/model-reasoning-effort";
export {
  hfModelFitsDevice,
  loadScopedGpu,
  searchRowFitsDevice,
} from "./components/model-selector/recommended-fit";
export {
  NumericValueInput,
  type NumericValueInputHandle,
  snapToStep,
} from "./components/numeric-value-input";
export { ModelConfigPage } from "./components/model-config-page";
export { SidebarModelConfig } from "./components/sidebar-model-config";
export type { ModelPickTarget } from "./components/model-selector/types";
export {
  fetchModelOverrides,
  modelOverrideKey,
  putModelOverride,
  syncModelOverride,
  type ApiModelOverride,
  type ApiModelOverrides,
} from "./api/model-overrides";
export { useActiveModelConfig } from "./hooks/use-active-model-config";
export { useVllmAvailable, useVllmPlatform } from "./hooks/use-vllm-available";
export { ManagedEngineOfferDialog } from "./components/managed-engine-offer-dialog";
export {
  confirmManagedEngineIfNeeded,
  type ManagedEngineOffer,
} from "./hooks/managed-engine-offer";
export type {
  DeletedModelRef,
  ExternalConnectionRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./components/model-selector";
export { modelConfigInstanceKey } from "./model-config/config-signature";
export { modelConfigDraftKey } from "./model-config/model-config-draft";
export { splitQuantSuffix } from "./model-config/model-identity";
export {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  modelConfigHandoffForDestination,
  modelConfigTarget,
  modelConfigTargetIsResident,
  modelConfigTargetMatchesSelection,
  requestModelConfigHandoff,
  useModelConfigHandoffStore,
  type ModelConfigFocus,
  type ModelConfigHandoffRequest,
} from "./model-config/model-config-handoff";
export {
  applyModelLoadConfigToRuntime,
  applyPerModelConfigToRuntime,
  currentRuntimePerModelConfig,
  perModelConfigsEqual,
} from "./model-config/apply-per-model-config";
export {
  DEFAULT_MAX_SEQ_LENGTH,
  DEFAULT_PER_MODEL_CONFIG,
  normalizeMaxSeqLength,
  type PerModelConfig,
  PER_MODEL_CONFIG_STORAGE_KEY,
  PER_MODEL_CONFIG_UPDATED_EVENT,
  adoptCachedRepoConfig,
  adoptLegacyConfigKey,
  isServedByLlamaCpp,
  resumesThought,
  contextPinPatch,
  listPerModelConfigs,
  isServedByMlx,
  residentIsServedByMlx,
  savedContextPin,
  loadedContextFields,
  presetLoadSettingNames,
  resolveInitialConfig,
  resolveResidentInitialConfig,
} from "./model-config/per-model-config";
export {
  SharedRunConfigLinkHandler,
  receiveSharedRunConfigUrls,
} from "./sharing";
export { useChatPickerInventory } from "./inventory/use-chat-picker-inventory";
export { modelDisplayName } from "./model-config/model-identity";
export { loraOptionLabel } from "./components/model-selector/row-meta";
export { PillTabs } from "./components/model-selector/pill-tabs";
export { usePinnedConnectedModelsStore } from "./components/model-selector/pinned-connected-models";
export { isMediaCatalogRepo } from "./components/model-selector/model-catalog";
export { ModelLoadSettingsAction } from "./components/model-selector/model-load-settings-action";
export { ConnectedModelSettingsDialog } from "./components/model-selector/connected-model-settings-dialog";
export { ConnectedModelInfoDialog } from "./components/model-selector/connected-model-info-dialog";
export { missingExternalModel } from "./components/model-selector/missing-external-model";
export {
  type ModelLoadTimes,
  loadStampId,
  loadedAt,
  useModelLoadTimes,
} from "./components/model-selector/model-usage";
export {
  chatGgufFit,
  estimateQuantBytes,
  isRecommendableFormat,
  paramsFromId,
} from "./components/model-selector/recommended-fit";
export {
  isOverBudget,
  VRAM_VERDICT,
} from "./components/model-selector/fit-verdict";
export type { PickerFitStatus } from "./components/model-selector/media-picker-fit";
export { pinDropAnchor } from "./components/model-selector/pinned-models";
export { usePinnedRowDrag } from "./components/model-selector/use-pinned-row-drag";
export { describeVariantListingError } from "./components/model-selector/variant-listing-error";
export { pickerOrderedGgufVariants } from "./components/model-selector/variant-order";
export { ggufVariantPickerLabel } from "./components/model-selector/variant-presentation";
export { connectedModelMarks } from "./components/model-selector/connected-model-meta";
export { mediaPageForTask } from "./components/model-selector/pickers";
export {
  AUDIO_CATALOG,
  artifactForRepoId,
} from "./components/model-selector/model-catalog";
export {
  audioPickIsRoutable,
  audioPipelineTagFor,
  curatedAudioInventoryTask,
  macMlxSttRowIsRunnable,
  nativeAudioCheckpointIsLoadable,
  taskForMediaPick,
} from "./components/model-selector/audio-picker-policy";
