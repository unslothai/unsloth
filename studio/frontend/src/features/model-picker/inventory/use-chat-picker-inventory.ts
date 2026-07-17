// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CachedGgufRepo,
  CachedModelRepo,
  LocalModelInfo,
} from "@/features/chat";
import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  type LocalSource,
  isHiddenModelId,
  useHubInventory,
} from "@/features/hub";
import { useMemo } from "react";

const PICKER_LOCAL_SOURCES: ReadonlySet<LocalSource> = new Set([
  "lmstudio",
  "models_dir",
  "custom",
]);

function isCompleteCachedRow(row: CachedInventoryRow): boolean {
  return !row.partial && !row.liveDownload;
}

function toCachedGgufRepo(row: CachedInventoryRow): CachedGgufRepo {
  return {
    repo_id: row.repoId,
    size_bytes: row.bytes,
    cache_path: row.cachePath ?? "",
    last_modified: row.lastModified ?? undefined,
    has_vision: row.capabilities.supportsVision,
  };
}

function toCachedModelRepo(row: CachedInventoryRow): CachedModelRepo {
  return {
    repo_id: row.repoId,
    size_bytes: row.bytes,
    last_modified: row.lastModified ?? undefined,
  };
}

function toLocalModelInfo(row: LocalInventoryRow): LocalModelInfo {
  return {
    id: row.loadId,
    display_name: row.displayName ?? row.title,
    path: row.path,
    source: row.source as LocalModelInfo["source"],
    model_id: row.modelId ?? row.repoId,
    model_format: row.modelFormat,
    updated_at: row.updatedAt,
  };
}

export interface ChatPickerInventory {
  cachedGguf: CachedGgufRepo[];
  cachedModels: CachedModelRepo[];
  cachedReady: boolean;
  localModels: LocalModelInfo[];
  refreshInventory: () => Promise<void>;
}

export function useChatPickerInventory(
  options: { enabled?: boolean } = {},
): ChatPickerInventory {
  const inventory = useHubInventory({
    kind: "models",
    enabled: options.enabled,
    includeLocal: true,
  });

  const cachedGguf = useMemo(
    () =>
      inventory.cachedRows
        .filter(
          (row) =>
            row.modelFormat === "gguf" &&
            isCompleteCachedRow(row) &&
            !isHiddenModelId(row.repoId),
        )
        .map(toCachedGgufRepo),
    [inventory.cachedRows],
  );
  const cachedModels = useMemo(
    () =>
      inventory.cachedRows
        .filter(
          (row) =>
            row.modelFormat !== "gguf" &&
            isCompleteCachedRow(row) &&
            !isHiddenModelId(row.repoId),
        )
        .map(toCachedModelRepo),
    [inventory.cachedRows],
  );
  const localModels = useMemo(
    () =>
      inventory.localRows
        .filter(
          (row) =>
            PICKER_LOCAL_SOURCES.has(row.source) &&
            !isHiddenModelId(row.modelId, row.repoId, row.path),
        )
        .map(toLocalModelInfo),
    [inventory.localRows],
  );

  return {
    cachedGguf,
    cachedModels,
    cachedReady: inventory.downloadedReady,
    localModels,
    refreshInventory: inventory.refreshInventory,
  };
}
