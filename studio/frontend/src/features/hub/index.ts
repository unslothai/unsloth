// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  DownloadProgressBar,
  downloadManager,
  finishExternalJob,
  jobKeyOf,
  startExternalJob,
  subscribeJobListeners,
  type TransportMode,
  updateExternalJob,
  TRANSPORT_MODE_STORAGE_KEY,
  useDownloadManagerStore,
  useHttpPartialsResumable,
  useTransportMode,
} from "./download-manager";
export { HfTokenIndicator } from "./components/hf-token-indicator";
export { useHubDatasetSearch } from "./hooks/use-hub-dataset-search";
export {
  type HfModelResult,
  type HfSortKey,
  useHubModelSearch,
} from "./hooks/use-hub-model-search";
export { useHubInfiniteScroll } from "./hooks/use-hub-infinite-scroll";
export { useLatestRef } from "./hooks/use-latest-ref";
export { useOnlineStatus } from "./hooks/use-online-status";
export {
  INVENTORY_HINT_KIND,
  INVENTORY_HINT_KINDS,
  LOCAL_MODEL_SOURCE,
  LOCAL_MODEL_SOURCES,
  type BaseModelSource,
  type CachedDatasetRepo,
  type CachedGgufRepo,
  type CachedInventoryRow,
  type CachedModelRepo,
  type DeviceInventoryRows,
  type DeviceInventorySource,
  type DeviceInventorySourceState,
  type GgufVariantDetail,
  type GgufVariantsResponse,
  type HubInventory,
  type HubInventoryKind,
  type InventoryHint,
  type InventoryHintKind,
  type InventoryResourceFormatHint,
  type InventoryRow,
  type LocalDatasetInfo,
  type LocalInventoryRow,
  type LocalModelInfo,
  type LocalModelListResponse,
  type LocalSource,
  type ModelInventoryCapabilities,
  type ModelInventoryFormat,
  type ModelInventoryRuntime,
  type ResolvedInventoryResource,
  type ScanFolderInfo,
  addScanFolder,
  buildCachedInventoryRow,
  buildLocalInventoryRows,
  dedupeSameSourceHubCacheRows,
  defaultCapabilities,
  deleteCachedDataset,
  deleteCachedModel,
  fetchInventorySource,
  findCompleteHfCacheLocalRow,
  formatLocalUpdated,
  invalidateGgufVariantsCache,
  listCachedDatasets,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  listLocalDatasets,
  listLocalModels,
  listScanFolders,
  localSourceLabel,
  normalizeCapabilities,
  normalizeModelFormat,
  normalizeTimestamp,
  removeScanFolder,
  resolveInventoryResource,
  useDeviceInventorySources,
  useDeviceInventoryStore,
  useTokenScopedInventoryRequestOptions,
  useGgufVariantsCacheVersion,
  useGgufVariantsCacheVersions,
  useHubInventory,
} from "./inventory";
export { bumpInventoryVersion } from "./stores/inventory-events";
export {
  getHfToken,
  hfApiToken,
  mirrorHfTokenInto,
  useHfTokenStore,
} from "./stores/hf-token-store";
export { useInventoryVersion } from "./stores/inventory-events";
export { looksLikeLocalPath, localPathCacheKey } from "./lib/local-path";
export { scanFolderStatusCopy } from "./lib/scan-folder-status";
export type { ScanFolderStatus } from "./lib/scan-folder-status";
export { hubTokenHeader } from "./lib/hub-token-header";
export {
  ggufVariantsMatch,
  isOllamaLinkPath,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  publicModelId,
  residentModelIdMatches,
} from "./lib/model-identity";
export {
  formatBytes,
  formatRelativeShort,
  ownerOf,
  repoOf,
} from "./lib/format";
export { ggufVariantDisplayLabel } from "./lib/gguf-variant-sort";
export { EMBEDDING_TAGS, isGgufLike } from "./lib/hf-model-meta";
export { matchTokens, tokenizeQuery } from "./lib/search-text";
export {
  DeleteConfirmDialog,
  UpdateConfirmDialog,
} from "./catalog/download-card";
export {
  DeleteImpactSummary,
  useDeleteImpact,
} from "./catalog/delete-impact";
export { HubOptionMenu, type HubOption } from "./catalog/hub-option-menu";
export { DotTag } from "./catalog/dot-tag";
export { TransportConflictDialog } from "./catalog/transport-conflict-dialog";
export { TrainIcon } from "./components/train-icon";
export { isHiddenModelId } from "./lib/hidden-models";
export { classifyUnslothSupport, studioPageForTask } from "./lib/unsloth-support";
