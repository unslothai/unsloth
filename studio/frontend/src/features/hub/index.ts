// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  cancelModelDownload,
  downloadManager,
  jobKeyOf,
  subscribeJobListeners,
  useDownloadManagerStore,
  type TransportConflictInfo,
} from "./download-manager";
export {
  useHubInventory,
  type CachedInventoryRow,
  type GgufVariantDetail,
  type HubInventory,
  type HubInventoryKind,
  type InventoryRow,
  type LocalInventoryRow,
  type LocalSource,
  type ScanFolderInfo,
  addScanFolder,
  deleteCachedDataset,
  deleteCachedModel,
  formatLocalUpdated,
  invalidateGgufVariantsCache,
  listGgufVariants,
  listScanFolders,
  removeScanFolder,
  useGgufVariantsCacheVersion,
} from "./inventory";
export {
  type HfModelResult,
  type HfModelSearchChannel,
  type HfSortDirection,
  type HfSortKey,
  useHubModelSearch,
} from "./hooks/use-hub-model-search";
export { useOnlineStatus } from "./hooks/use-online-status";
export { useHubInfiniteScroll } from "./hooks/use-hub-infinite-scroll";
export { bumpInventoryVersion } from "./stores/inventory-events";
export {
  getHfToken,
  hfApiToken,
  mirrorHfTokenInto,
  useHfTokenStore,
} from "./stores/hf-token-store";
export { useInventoryVersion } from "./stores/inventory-events";
export { looksLikeLocalPath } from "./lib/local-path";
export { hubTokenHeader } from "./lib/hub-token-header";
export {
  ggufVariantsMatch,
  modelIdsMatch,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "./lib/model-identity";
export { formatBytes, formatRelativeShort } from "./lib/format";
export { ggufVariantDisplayLabel } from "./lib/gguf-variant-sort";
export {
  DeleteConfirmDialog,
  UpdateConfirmDialog,
} from "./catalog/download-card";
export { HubOptionMenu, type HubOption } from "./catalog/hub-option-menu";
export { DotTag } from "./catalog/dot-tag";
export { TransportConflictDialog } from "./catalog/transport-conflict-dialog";
export { TrainIcon } from "./components/train-icon";
export { isHiddenModelId } from "./lib/hidden-models";
export { classifyUnslothSupport } from "./lib/unsloth-support";
