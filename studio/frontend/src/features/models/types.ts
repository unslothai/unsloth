// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/chat/api/chat-api";
import type { HfModelResult } from "@/hooks";
import type { Capability, CapabilityKey } from "./lib/capabilities";

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
  summary: string;
  capabilities: Capability[];
}

export interface CachedInventoryRow {
  kind: "cache";
  id: string;
  repoId: string;
  owner: string;
  repo: string;
  isGguf: boolean;
  bytes: number;
}

export interface LocalInventoryRow {
  kind: "local";
  id: string;
  repoId: string | null;
  owner: string;
  title: string;
  source: LocalModelInfo["source"];
  sourceLabel: string;
  path: string;
  isGguf: boolean;
  updatedAt: number | null;
}

export type InventoryRow = CachedInventoryRow | LocalInventoryRow;

export interface SelectedModelView {
  id: string;
  kind: "discover" | "cache" | "local";
  displayId: string;
  hubRepoId: string | null;
  owner: string;
  title: string;
  summary: string;
  sourceLabel: string;
  path: string | null;
  isLocal: boolean;
  isGguf: boolean;
  isDownloaded: boolean;
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
}
