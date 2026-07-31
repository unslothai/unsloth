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
  epochMillisecondsToSeconds,
  isHiddenModelId,
  studioPageForTask,
  useHubInventory,
} from "@/features/hub";
import { useMemo } from "react";
import { allowedHiddenModelIdMatches } from "../components/model-selector/audio-picker-policy";
import {
  completeCachedModelKeys,
  isHfCacheDuplicate,
} from "./chat-picker-inventory-sources";

const PICKER_LOCAL_SOURCES: ReadonlySet<LocalSource> = new Set([
  "lmstudio",
  "models_dir",
  "ollama",
  // A cached Hugging Face repo materialized on disk is a loadable local model; without it
  // the chat picker hid every model that only exists in the HF cache. Duplicates against a
  // complete cached entry are dropped below via isHfCacheDuplicate.
  "hf_cache",
  "custom",
]);

function isCompleteCachedRow(row: CachedInventoryRow): boolean {
  return !row.partial && !row.liveDownload;
}

function toCachedGgufRepo(row: CachedInventoryRow): CachedGgufRepo {
  return {
    repo_id: row.repoId,
    // Listed by repo id, loaded by the pinned id: dropping it sends the picker back down the ref.
    load_id: row.loadId,
    size_bytes: row.bytes,
    cache_path: row.cachePath ?? "",
    last_modified: epochMillisecondsToSeconds(row.lastModified),
    has_vision: row.capabilities.supportsVision,
    task: row.task ?? null,
    audio_type: row.audioType ?? null,
    has_variant_state: row.hasVariantState ?? false,
  };
}

function toCachedModelRepo(row: CachedInventoryRow): CachedModelRepo {
  return {
    repo_id: row.repoId,
    load_id: row.loadId,
    // Delete targets the copy the row describes; without it the request hits the active cache.
    cache_path: row.cachePath,
    size_bytes: row.bytes,
    last_modified: epochMillisecondsToSeconds(row.lastModified),
    task: row.task ?? null,
    audio_type: row.audioType ?? null,
    tags: row.tags,
    library_name: row.libraryName,
    // Carried through: the diffusion picker drops single-file checkpoint repos (loading one as a pipeline fails after the handoff), and undefined reads as "full pipeline".
    single_file: row.singleFile ?? false,
  };
}

function toLocalModelInfo(row: LocalInventoryRow): LocalModelInfo {
  return {
    id: row.loadId,
    display_name: row.displayName ?? row.title,
    // A cached HF repo on disk loads exactly like a models-dir entry; presenting it under its
    // own "hf_cache" source would leak an origin the chat loader/UI does not special-case.
    source: (row.source === "hf_cache"
      ? "models_dir"
      : row.source) as LocalModelInfo["source"],
    path: row.path,
    model_id: row.modelId ?? row.repoId,
    model_format: row.modelFormat,
    updated_at: epochMillisecondsToSeconds(row.updatedAt),
    task: row.task ?? null,
    audio_type: row.audioType ?? null,
  };
}

export interface ChatPickerInventory {
  cachedGguf: CachedGgufRepo[];
  cachedModels: CachedModelRepo[];
  cachedReady: boolean;
  localModels: LocalModelInfo[];
  refreshInventory: () => Promise<void>;
  refreshInventoryIfOlderThan: (maxAgeMs: number) => Promise<void>;
}

export function useChatPickerInventory(
  options: {
    enabled?: boolean;
    /** Exact task-page artifacts that may bypass chat's hidden-model list. */
    allowedHiddenModelIds?: ReadonlySet<string>;
  } = {},
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
            (!isHiddenModelId(row.repoId) ||
              allowedHiddenModelIdMatches(
                options.allowedHiddenModelIds,
                row.repoId,
              )),
        )
        .map(toCachedGgufRepo),
    [inventory.cachedRows, options.allowedHiddenModelIds],
  );
  const cachedModels = useMemo(
    () =>
      inventory.cachedRows
        .filter(
          (row) =>
            row.modelFormat !== "gguf" &&
            isCompleteCachedRow(row) &&
            // An sd.cpp companion mirror holds a VAE / text encoders and no denoiser. It has no
            // task, and a task of null is what every unclassified CHAT repo carries, so without
            // this it lands in the chat On Device list as a load that cannot succeed.
            !row.companion &&
            (!isHiddenModelId(row.repoId) ||
              allowedHiddenModelIdMatches(
                options.allowedHiddenModelIds,
                row.repoId,
              )),
        )
        .map(toCachedModelRepo),
    [inventory.cachedRows, options.allowedHiddenModelIds],
  );
  const localModels = useMemo(() => {
    // A cached model already surfaced above makes its hf_cache twin redundant; key the
    // complete cached entries once so the row filter can drop the duplicates.
    const completeCachedKeys = completeCachedModelKeys(inventory.cachedRows);
    return inventory.localRows
      .filter(
        (row) =>
          PICKER_LOCAL_SOURCES.has(row.source) &&
          // Skip non-chat rows (a folder with only config.json classifies "unknown" -> canChat false); selecting one would load a weightless path.
          // toLocalModelInfo drops capabilities, so this is the only place the guard can live. A row the backend classified as a generation task is
          // exempt: canChat is about the chat loader, and dropping it here hid every on-device diffusion model from the pickers that CAN load it.
          (row.capabilities.canChat ||
            studioPageForTask(row.task) !== undefined) &&
          (!isHiddenModelId(row.modelId, row.repoId, row.path) ||
            allowedHiddenModelIdMatches(
              options.allowedHiddenModelIds,
              row.modelId,
              row.repoId,
            )) &&
          !isHfCacheDuplicate(row, completeCachedKeys),
      )
      .map(toLocalModelInfo);
  }, [inventory.localRows, inventory.cachedRows, options.allowedHiddenModelIds]);

  return {
    cachedGguf,
    cachedModels,
    cachedReady: inventory.downloadedReady,
    localModels,
    refreshInventory: inventory.refreshInventory,
    refreshInventoryIfOlderThan: inventory.refreshInventoryIfOlderThan,
  };
}
