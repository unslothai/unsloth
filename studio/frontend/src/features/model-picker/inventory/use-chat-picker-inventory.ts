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
  type FamilyOverrideArtifactKind,
  taskOpaqueArtifactSupportsFamilyOverride,
} from "../components/model-selector/family-override";

const PICKER_LOCAL_SOURCES: ReadonlySet<LocalSource> = new Set([
  "lmstudio",
  "omlx",
  "models_dir",
  "ollama",
  "hermes",
  "custom",
]);

/** Partial snapshots stay (click opens the download); live downloads belong to the Downloads panel. */
function isListableCachedRow(row: CachedInventoryRow): boolean {
  return !row.liveDownload;
}

function toCachedGgufRepo(row: CachedInventoryRow): CachedGgufRepo {
  return {
    repo_id: row.repoId,
    // Listed by repo id, loaded by the pinned id.
    load_id: row.loadId,
    size_bytes: row.bytes,
    cache_path: row.cachePath ?? "",
    last_modified: epochMillisecondsToSeconds(row.lastModified),
    has_vision: row.capabilities.supportsVision,
    partial: row.partial,
    // A restart-only partial must not be called a resume.
    partial_resumable: row.partialResumable,
    task: row.task ?? null,
    audio_type: row.audioType ?? null,
    has_variant_state: row.hasVariantState ?? false,
  };
}

function toCachedModelRepo(row: CachedInventoryRow, opaqueKind?: FamilyOverrideArtifactKind): CachedModelRepo {
  return {
    repo_id: row.repoId,
    load_id: row.loadId,
    // Without it the delete hits the active cache.
    cache_path: row.cachePath,
    size_bytes: row.bytes,
    opaque: taskOpaqueArtifactSupportsFamilyOverride(row.task, row.artifact, opaqueKind),
    last_modified: epochMillisecondsToSeconds(row.lastModified),
    partial: row.partial,
    partial_resumable: row.partialResumable,
    task: row.task ?? null,
    audio_type: row.audioType ?? null,
    tags: row.tags,
    library_name: row.libraryName,
    // Single-file checkpoints fail as pipelines; undefined would read as "full pipeline".
    single_file: row.singleFile ?? false,
  };
}

function toLocalModelInfo(row: LocalInventoryRow, opaqueKind?: FamilyOverrideArtifactKind): LocalModelInfo {
  return {
    id: row.loadId,
    display_name: row.displayName ?? row.title,
    path: row.path,
    source: row.source as LocalModelInfo["source"],
    model_id: row.modelId ?? row.repoId,
    model_format: row.modelFormat,
    opaque: taskOpaqueArtifactSupportsFamilyOverride(row.task, row.artifact, opaqueKind),
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
    allowedHiddenModelIds?: ReadonlySet<string>;
    opaqueKind?: FamilyOverrideArtifactKind;
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
            // An sd.cpp companion mirror has no denoiser and no task, so it would look like an unclassified chat repo.
            !row.companion &&
            (!isHiddenModelId(row.repoId) ||
              allowedHiddenModelIdMatches(
                options.allowedHiddenModelIds,
                row.repoId,
              )),
        )
        .map((row) => toCachedModelRepo(row, options.opaqueKind)),
    [inventory.cachedRows, options.allowedHiddenModelIds, options.opaqueKind],
  );
  const localModels = useMemo(
    () =>
      inventory.localRows
        .filter(
          (row) =>
            PICKER_LOCAL_SOURCES.has(row.source) &&
            // Skip non-chat rows (a weightless path), except generation-task rows the diffusion pickers can load.
            // toLocalModelInfo drops capabilities, so this is the only place for the guard.
            (row.capabilities.canChat ||
              studioPageForTask(row.task) !== undefined ||
              taskOpaqueArtifactSupportsFamilyOverride(row.task, row.artifact, options.opaqueKind)) &&
            (!isHiddenModelId(row.modelId, row.repoId, row.path) ||
              allowedHiddenModelIdMatches(
                options.allowedHiddenModelIds,
                row.modelId,
                row.repoId,
              )),
        )
        .map((row) => toLocalModelInfo(row, options.opaqueKind)),
    [inventory.localRows, options.allowedHiddenModelIds, options.opaqueKind],
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
