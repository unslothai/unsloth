// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  BaseModelSource,
  LocalSource,
  ModelInventoryCapabilities,
  ModelInventoryFormat,
} from "@/features/inventory";
import type { HfModelResult } from "@/hooks";
import type { Capability, CapabilityKey } from "./lib/capabilities";

export type {
  CachedInventoryRow,
  InventoryHint,
  InventoryRow,
  LocalInventoryRow,
  LocalSource,
} from "@/features/inventory";

export type ModelsTab = "discover" | "downloaded";

export type ResourceTypeFilter = "models" | "datasets";

export type ModelFormatFilter = "all" | "gguf" | "checkpoint";

export type CapabilityFilter = "all" | CapabilityKey;

export interface DiscoverRow {
  id: string;
  owner: string;
  repo: string;
  result: HfModelResult;
  isAvailableOnDevice: boolean;
  isPartialOnDevice: boolean;
  summary: string;
  capabilities: Capability[];
}

export type SelectedResourceSource = "huggingface" | "hub_cache" | LocalSource;

export type SelectedResourceCacheState =
  | "remote"
  | "cached"
  | "local"
  | "partial";

export interface SelectedResourceRef {
  repoId: string | null;
  localPath: string | null;
  source: SelectedResourceSource;
  cacheState: SelectedResourceCacheState;
  runId: string;
  trainId: string;
}

export interface SelectedModelView {
  id: string;
  kind: "discover" | "cache" | "local";
  resource: SelectedResourceRef;
  displayId: string;
  hubRepoId: string | null;
  owner: string;
  title: string;
  summary: string;
  sourceLabel: string;
  path: string | null;
  localSource?: LocalSource;
  isLocal: boolean;
  isGguf: boolean;
  requiresVariant?: boolean;
  modelFormat: ModelInventoryFormat | null;
  baseModel?: string | null;
  baseModelSource?: BaseModelSource | null;
  baseModelHubId?: string | null;
  baseModelSummary?: string | null;
  adapterType?: string | null;
  trainingMethod?: string | null;
  isDownloaded: boolean;
  isPartial?: boolean;
  runtimeCapabilities?: ModelInventoryCapabilities;
  capabilities: Capability[];
  license: string | null;
  pipelineTag?: string;
  libraryName?: string;
  downloads?: number;
  likes?: number;
  totalParams?: number;
  estimatedSizeBytes?: number;
  cachedBytes?: number;
  updatedAt?: string;
  localUpdatedAt?: number | null;
  tags?: string[];
  quantMethod?: string;
}
