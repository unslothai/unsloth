// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CachedGgufRepo,
  CachedModelRepo,
  LocalModelInfo,
} from "@/features/chat";
import {
  type CachedInventoryRow,
  isHiddenModelId,
  useHubInventory,
} from "@/features/hub";
import { useMemo } from "react";

import { buildPickerLocalModels } from "./chat-picker-inventory-sources";

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
      buildPickerLocalModels(inventory.cachedRows, inventory.localRows).filter(
        // Skip hidden infrastructure and embedding models after projection;
        // the projector preserves every identity used by the matcher.
        (row) => !isHiddenModelId(row.model_id, row.id, row.path),
      ),
    [inventory.cachedRows, inventory.localRows],
  );

  return {
    cachedGguf,
    cachedModels,
    cachedReady: inventory.downloadedReady,
    localModels,
    refreshInventory: inventory.refreshInventory,
  };
}
