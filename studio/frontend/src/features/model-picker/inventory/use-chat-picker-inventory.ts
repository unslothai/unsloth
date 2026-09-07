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

const PICKER_LOCAL_SOURCES: ReadonlySet<LocalSource> = new Set([
  "lmstudio",
  "models_dir",
  "ollama",
  "custom",
]);

/** A row the picker lists. Partial snapshots stay: like the Hub, the picker shows them so they can
 *  be seen and deleted, and carries `partial` through so a click opens the download instead of
 *  claiming the weights are there. A live download is still dropped -- bytes are moving and the
 *  Downloads panel owns that row until they stop. */
function isListableCachedRow(row: CachedInventoryRow): boolean {
  return !row.liveDownload;
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
    // Listed but not loadable: the row renders a partial mark and its click opens the download.
    partial: row.partial,
    // What that mark is allowed to promise: a restart-only partial must not be called a resume.
    partial_resumable: row.partialResumable,
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
    // Listed but not loadable: the row renders a partial mark and its click opens the download.
    partial: row.partial,
    // What that mark is allowed to promise: a restart-only partial must not be called a resume.
    partial_resumable: row.partialResumable,
    task: row.task ?? null,
    audio_type: row.audioType ?? null,
    tags: row.tags,
    library_name: row.libraryName,
    // Carried through: the diffusion picker drops single-file checkpoint repos, since loading one as
    // a pipeline fails after the handoff, and undefined reads as "full pipeline".
    single_file: row.singleFile ?? false,
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
            isListableCachedRow(row) &&
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
            isListableCachedRow(row) &&
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
  const localModels = useMemo(
    () =>
      inventory.localRows
        .filter(
          (row) =>
            PICKER_LOCAL_SOURCES.has(row.source) &&
            // Skip non-chat rows (a folder with only config.json classifies "unknown"); selecting one would
            // load a weightless path. toLocalModelInfo drops capabilities, so this is the only place the
            // guard can live. A row the backend classified as a generation task is exempt: canChat is about
            // the chat loader, and dropping it here hid every on-device diffusion model from the pickers
            // that CAN load it.
            (row.capabilities.canChat ||
              studioPageForTask(row.task) !== undefined) &&
            (!isHiddenModelId(row.modelId, row.repoId, row.path) ||
              allowedHiddenModelIdMatches(
                options.allowedHiddenModelIds,
                row.modelId,
                row.repoId,
              )),
        )
        .map(toLocalModelInfo),
    [inventory.localRows, options.allowedHiddenModelIds],
  );

  return {
    cachedGguf,
    cachedModels,
    cachedReady: inventory.downloadedReady,
    localModels,
    refreshInventory: inventory.refreshInventory,
    refreshInventoryIfOlderThan: inventory.refreshInventoryIfOlderThan,
  };
}
