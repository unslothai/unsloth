// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  INVENTORY_HINT_KIND,
  INVENTORY_HINT_KINDS,
  LOCAL_MODEL_SOURCE,
  LOCAL_MODEL_SOURCES,
  type InventoryHintKind,
  type LocalSource,
} from "./constants";
export {
  addScanFolder,
  browseFolders,
  listCachedDatasets,
  listCachedGguf,
  listCachedModels,
  deleteCachedDataset,
  deleteCachedModel,
  invalidateGgufVariantsCache,
  listGgufVariants,
  listLocalDatasets,
  listLocalModels,
  listScanFolders,
  removeScanFolder,
  type BrowseFoldersResponse,
  type CachedDatasetRepo,
  type CachedGgufRepo,
  type CachedModelRepo,
  type BaseModelSource,
  type GgufVariantDetail,
  type GgufVariantsResponse,
  type LocalDatasetInfo,
  type LocalModelInfo,
  type LocalModelListResponse,
  type ModelInventoryFormat,
  type ModelInventoryRuntime,
  type ScanFolderInfo,
} from "./api";
export {
  buildLocalInventoryRows,
  buildCachedInventoryRow,
  defaultCapabilities,
  formatLocalUpdated,
  localSourceLabel,
  normalizeCapabilities,
  normalizeModelFormat,
  normalizeRuntime,
  normalizeTimestamp,
} from "./view-models";
export {
  resolveInventoryResource,
  type InventoryResourceFormatHint,
  type ResolvedInventoryResource,
} from "./resource-resolver";
export {
  dedupeSameSourceHubCacheRows,
  findCompleteHfCacheLocalRow,
} from "./inventory-dedupe";
export {
  fetchInventorySource,
  useDeviceInventorySources,
  useDeviceInventoryStore,
  type DeviceInventoryRows,
  type DeviceInventorySource,
  type DeviceInventorySourceState,
} from "./use-device-inventory";
export {
  useHubInventory,
  type HubInventoryKind,
  type HubInventory,
} from "./use-hub-inventory";
export { useGgufVariantsCacheVersion } from "./use-gguf-variants-cache-version";
export type {
  CachedInventoryRow,
  InventoryHint,
  InventoryRow,
  LocalInventoryRow,
  ModelInventoryCapabilities,
} from "./types";
