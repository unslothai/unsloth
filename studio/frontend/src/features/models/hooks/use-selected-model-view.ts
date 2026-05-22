// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { detectCapabilities, detectLicense } from "../lib/capabilities";
import { buildSummary, localSourceLabel, toHfModelResult } from "../lib/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  SelectedModelView,
} from "../types";

type HfResult = ReturnType<typeof toHfModelResult>;

export function useSelectedModelView({
  selectedDiscoverRow,
  selectedCachedRow,
  selectedLocalRow,
  selectedHfResult,
  isDatasetMode,
}: {
  selectedDiscoverRow: DiscoverRow | null;
  selectedCachedRow: CachedInventoryRow | null;
  selectedLocalRow: LocalInventoryRow | null;
  selectedHfResult: HfResult;
  isDatasetMode: boolean;
}): SelectedModelView | null {
  return useMemo<SelectedModelView | null>(() => {
    if (selectedDiscoverRow) {
      const onDevicePath =
        selectedCachedRow?.cachePath ?? selectedLocalRow?.path ?? null;
      return {
        id: selectedDiscoverRow.id,
        kind: "discover",
        displayId: selectedDiscoverRow.id,
        hubRepoId: selectedDiscoverRow.result.id,
        owner: selectedDiscoverRow.owner,
        title: selectedDiscoverRow.repo,
        summary: selectedDiscoverRow.summary,
        sourceLabel: selectedDiscoverRow.isAvailableOnDevice
          ? "On device"
          : "Hugging Face",
        path: onDevicePath,
        isLocal: false,
        isGguf: selectedDiscoverRow.result.isGguf,
        isDownloaded:
          selectedDiscoverRow.isAvailableOnDevice &&
          !selectedDiscoverRow.isPartialOnDevice,
        isPartial: selectedDiscoverRow.isPartialOnDevice,
        capabilities: selectedDiscoverRow.capabilities,
        license: detectLicense(selectedDiscoverRow.result.tags),
        pipelineTag: selectedDiscoverRow.result.pipelineTag,
        libraryName: selectedDiscoverRow.result.libraryName,
        downloads: selectedDiscoverRow.result.downloads,
        likes: selectedDiscoverRow.result.likes,
        totalParams: selectedDiscoverRow.result.totalParams,
        estimatedSizeBytes: selectedDiscoverRow.result.estimatedSizeBytes,
        cachedBytes: selectedCachedRow?.bytes,
        updatedAt: selectedDiscoverRow.result.updatedAt,
        tags: selectedDiscoverRow.result.tags,
        quantMethod: selectedDiscoverRow.result.quantMethod,
      };
    }

    if (selectedCachedRow) {
      const cachedSummary = isDatasetMode
        ? "Cached dataset, ready to use."
        : selectedCachedRow.isGguf
          ? "Cached GGUF repository ready for local inference."
          : "Cached checkpoint repository ready for local inference.";
      const mergedTags = selectedHfResult?.tags ?? selectedCachedRow.tags;
      const mergedPipelineTag =
        selectedHfResult?.pipelineTag ?? selectedCachedRow.pipelineTag ?? undefined;
      const mergedLibraryName =
        selectedHfResult?.libraryName ?? selectedCachedRow.libraryName ?? undefined;
      const mergedQuantMethod =
        selectedHfResult?.quantMethod ?? selectedCachedRow.quantMethod ?? undefined;
      return {
        id: selectedCachedRow.repoId,
        kind: "cache",
        displayId: selectedCachedRow.repoId,
        hubRepoId: selectedCachedRow.repoId,
        owner: selectedCachedRow.owner,
        title: selectedCachedRow.repo,
        summary: selectedHfResult
          ? buildSummary(selectedHfResult)
          : cachedSummary,
        sourceLabel: "Hub cache",
        path: selectedCachedRow.cachePath ?? null,
        isLocal: false,
        isGguf: selectedCachedRow.isGguf,
        isDownloaded: !selectedCachedRow.partial,
        isPartial: selectedCachedRow.partial ?? false,
        capabilities: detectCapabilities(
          mergedTags,
          mergedPipelineTag,
          selectedCachedRow.repoId,
        ),
        license: detectLicense(mergedTags),
        pipelineTag: mergedPipelineTag,
        libraryName: mergedLibraryName,
        downloads: selectedHfResult?.downloads,
        likes: selectedHfResult?.likes,
        totalParams: selectedHfResult?.totalParams,
        estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
        cachedBytes: selectedCachedRow.bytes,
        updatedAt: selectedHfResult?.updatedAt,
        tags: mergedTags,
        quantMethod: mergedQuantMethod,
      };
    }

    if (selectedLocalRow) {
      const localDisplayId = selectedLocalRow.repoId ?? selectedLocalRow.id;
      const isPartialHubCache =
        selectedLocalRow.source === "hf_cache" &&
        !!selectedLocalRow.partial &&
        !!selectedLocalRow.repoId;

      if (isPartialHubCache && selectedLocalRow.repoId) {
        return {
          id: selectedLocalRow.repoId,
          kind: "cache",
          displayId: selectedLocalRow.repoId,
          hubRepoId: selectedLocalRow.repoId,
          owner: selectedLocalRow.owner,
          title: selectedLocalRow.title,
          summary: selectedHfResult
            ? buildSummary(selectedHfResult)
            : "Partial download. Resume to finish or delete to free space.",
          sourceLabel: "Hub cache",
          path: selectedLocalRow.path,
          isLocal: false,
          isGguf: selectedLocalRow.isGguf,
          isDownloaded: false,
          isPartial: true,
          capabilities: detectCapabilities(
            selectedHfResult?.tags,
            selectedHfResult?.pipelineTag,
            selectedLocalRow.repoId,
          ),
          license: detectLicense(selectedHfResult?.tags),
          pipelineTag: selectedHfResult?.pipelineTag,
          libraryName: selectedHfResult?.libraryName,
          downloads: selectedHfResult?.downloads,
          likes: selectedHfResult?.likes,
          totalParams: selectedHfResult?.totalParams,
          estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
          updatedAt: selectedHfResult?.updatedAt,
          localUpdatedAt: selectedLocalRow.updatedAt,
          tags: selectedHfResult?.tags,
          quantMethod: selectedHfResult?.quantMethod,
        };
      }

      return {
        id: selectedLocalRow.id,
        kind: "local",
        displayId: localDisplayId,
        hubRepoId: selectedLocalRow.repoId,
        owner: selectedLocalRow.owner,
        title: selectedLocalRow.title,
        summary: selectedHfResult
          ? buildSummary(selectedHfResult)
          : `${localSourceLabel(selectedLocalRow.source)} · ${
              selectedLocalRow.isGguf ? "local GGUF" : "local checkpoint"
            }`,
        sourceLabel: selectedLocalRow.sourceLabel,
        path: selectedLocalRow.path,
        localSource: selectedLocalRow.source,
        isLocal: true,
        isGguf: selectedLocalRow.isGguf,
        isDownloaded: true,
        capabilities: detectCapabilities(
          selectedHfResult?.tags,
          selectedHfResult?.pipelineTag,
          selectedLocalRow.repoId ?? selectedLocalRow.title,
        ),
        license: detectLicense(selectedHfResult?.tags),
        pipelineTag: selectedHfResult?.pipelineTag,
        libraryName: selectedHfResult?.libraryName,
        downloads: selectedHfResult?.downloads,
        likes: selectedHfResult?.likes,
        totalParams: selectedHfResult?.totalParams,
        estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
        updatedAt: selectedHfResult?.updatedAt,
        localUpdatedAt: selectedLocalRow.updatedAt,
        tags: selectedHfResult?.tags,
        quantMethod: selectedHfResult?.quantMethod,
      };
    }

    return null;
  }, [
    isDatasetMode,
    selectedCachedRow,
    selectedDiscoverRow,
    selectedHfResult,
    selectedLocalRow,
  ]);
}
