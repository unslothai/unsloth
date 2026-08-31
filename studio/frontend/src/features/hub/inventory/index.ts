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
  listCachedDatasets,
  listCachedGguf,
  listCachedModels,
  deleteCachedDataset,
  deleteCachedModel,
  fetchDeleteImpact,
  fetchOrphanCompanions,
  invalidateGgufVariantsCache,
  listGgufVariants,
  listLocalDatasets,
  listLocalModels,
  listScanFolders,
  removeScanFolder,
  type CachedDatasetRepo,
  type CachedGgufRepo,
  type CachedModelRepo,
  type BaseModelSource,
  type CompanionAssetInfo,
  type DeleteImpact,
  type GgufVariantDetail,
  type GgufVariantsResponse,
  type LocalDatasetInfo,
  type LocalModelInfo,
  type LocalModelListResponse,
  type ModelInventoryFormat,
  type ModelInventoryRuntime,
  type OrphanCompanion,
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
} from "./view-models";
export {
  epochMillisecondsToSeconds,
  normalizeTimestamp,
} from "./inventory-timestamps";
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
  useTokenScopedInventoryRequestOptions,
  type DeviceInventoryRows,
  type DeviceInventorySource,
  type DeviceInventorySourceState,
} from "./use-device-inventory";
export { INVENTORY_FRESHNESS_WINDOW_MS } from "./inventory-freshness";
export {
  useHubInventory,
  type HubInventoryKind,
  type HubInventory,
} from "./use-hub-inventory";
export {
  useGgufVariantsCacheVersion,
  useGgufVariantsCacheVersions,
} from "./use-gguf-variants-cache-version";
export type {
  CachedInventoryRow,
  InventoryHint,
  InventoryRow,
  LocalInventoryRow,
  ModelInventoryCapabilities,
} from "./types";
