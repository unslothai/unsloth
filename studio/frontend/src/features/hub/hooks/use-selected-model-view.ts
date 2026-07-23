// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { detectCapabilities, detectLicense } from "../lib/model-capabilities";
import {
  buildSummary,
  localSourceLabel,
  toHfModelResult,
} from "../lib/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  SelectedResourceRef,
  SelectedModelView,
} from "../types";

type HfResult = ReturnType<typeof toHfModelResult>;

function detectViewCapabilities(
  tags: string[] | undefined,
  pipelineTag: string | undefined,
  ...identifiers: Array<string | null | undefined>
) {
  const modelId = identifiers.filter(Boolean).join(" ");
  return detectCapabilities(tags, pipelineTag, modelId);
}

function cachedResource(row: CachedInventoryRow): SelectedResourceRef {
  return {
    repoId: row.repoId,
    localPath: row.cachePath ?? null,
    source: "hub_cache",
    cacheState: row.partial ? "partial" : "cached",
    runId: row.loadId,
    trainId: row.loadId,
  };
}

function localResource(
  row: LocalInventoryRow,
  repoId: string | null = row.repoId,
): SelectedResourceRef {
  const cacheState = row.partial
    ? "partial"
    : row.source === "hf_cache"
      ? "cached"
      : "local";
  const id =
    row.source === "hf_cache" &&
    row.activeCache !== false &&
    repoId &&
    !row.partial
      ? repoId
      : row.loadId;
  return {
    repoId,
    localPath: row.path,
    source: row.source,
    cacheState,
    runId: id,
    trainId: id,
  };
}

function remoteResource(row: DiscoverRow): SelectedResourceRef {
  return {
    repoId: row.result.id,
    localPath: null,
    source: "huggingface",
    cacheState: row.isPartialOnDevice ? "partial" : "remote",
    runId: row.result.id,
    trainId: row.result.id,
  };
}

function localFormatLabel(row: LocalInventoryRow): string {
  if (row.modelFormat === "gguf") return "local GGUF";
  if (row.modelFormat === "adapter") return "local adapter";
  if (row.modelFormat === "safetensors") return "local safetensors model";
  if (row.modelFormat === "checkpoint") return "local checkpoint";
  return "local model";
}

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
      if (
        !selectedCachedRow &&
        selectedLocalRow &&
        selectedLocalRow.source !== "hf_cache"
      ) {
        const resource = localResource(
          selectedLocalRow,
          selectedDiscoverRow.result.id,
        );
        return {
          id: resource.runId,
          kind: "local",
          resource,
          displayId: selectedDiscoverRow.id,
          hubRepoId: selectedDiscoverRow.result.id,
          owner: selectedDiscoverRow.owner,
          title: selectedDiscoverRow.repo,
          summary: selectedHfResult
            ? buildSummary(selectedHfResult)
            : selectedDiscoverRow.summary,
          sourceLabel: selectedLocalRow.sourceLabel,
          path: selectedLocalRow.path,
          localSource: selectedLocalRow.source,
          isLocal: true,
          isGguf: selectedLocalRow.isGguf || selectedDiscoverRow.result.isGguf,
          requiresVariant: selectedLocalRow.capabilities.requiresVariant,
          modelFormat:
            selectedLocalRow.modelFormat !== "unknown"
              ? selectedLocalRow.modelFormat
              : selectedDiscoverRow.result.isGguf
                ? "gguf"
                : null,
          baseModel: selectedDiscoverRow.result.baseModel ?? null,
          baseModelSource: selectedDiscoverRow.result.baseModel
            ? "huggingface"
            : null,
          baseModelHubId: selectedDiscoverRow.result.baseModel ?? null,
          isDownloaded: !selectedLocalRow.partial,
          isPartial: selectedLocalRow.partial ?? false,
          partialTransport: selectedLocalRow.partialTransport ?? null,
          runtimeCapabilities: selectedLocalRow.capabilities,
          capabilities: selectedDiscoverRow.capabilities,
          license: detectLicense(selectedDiscoverRow.result.tags),
          pipelineTag: selectedDiscoverRow.result.pipelineTag,
          libraryName: selectedDiscoverRow.result.libraryName,
          gated: selectedDiscoverRow.result.gated,
          private: selectedDiscoverRow.result.private,
          downloads: selectedDiscoverRow.result.downloads,
          downloadsAllTime: selectedDiscoverRow.result.downloadsAllTime,
          likes: selectedDiscoverRow.result.likes,
          totalParams: selectedDiscoverRow.result.totalParams,
          estimatedSizeBytes: selectedDiscoverRow.result.estimatedSizeBytes,
          updatedAt: selectedDiscoverRow.result.updatedAt,
          createdAt: selectedDiscoverRow.result.createdAt,
          localUpdatedAt: selectedLocalRow.updatedAt,
          tags: selectedDiscoverRow.result.tags,
          quantMethod: selectedDiscoverRow.result.quantMethod,
        };
      }

      const onDevicePath =
        selectedCachedRow?.cachePath ?? selectedLocalRow?.path ?? null;
      const resource = selectedCachedRow
        ? cachedResource(selectedCachedRow)
        : selectedLocalRow?.source === "hf_cache"
          ? localResource(selectedLocalRow, selectedDiscoverRow.result.id)
          : remoteResource(selectedDiscoverRow);
      const isResolvedPartial = resource.cacheState === "partial";
      const isResolvedOnDevice =
        resource.cacheState === "cached" || resource.cacheState === "local";
      const resolvedModelFormat =
        selectedCachedRow?.modelFormat ??
        (selectedLocalRow?.modelFormat &&
        selectedLocalRow.modelFormat !== "unknown"
          ? selectedLocalRow.modelFormat
          : selectedDiscoverRow.result.isGguf
            ? "gguf"
            : null);
      return {
        id: selectedDiscoverRow.id,
        kind: "discover",
        resource,
        displayId: selectedDiscoverRow.id,
        hubRepoId: selectedDiscoverRow.result.id,
        owner: selectedDiscoverRow.owner,
        title: selectedDiscoverRow.repo,
        summary: selectedDiscoverRow.summary,
        sourceLabel: isResolvedOnDevice
          ? "On device"
          : isResolvedPartial
            ? "Partial on device"
            : "Hugging Face",
        path: onDevicePath,
        isLocal: false,
        isGguf:
          selectedCachedRow?.isGguf ??
          selectedLocalRow?.isGguf ??
          selectedDiscoverRow.result.isGguf,
        requiresVariant:
          selectedCachedRow?.capabilities.requiresVariant ??
          selectedLocalRow?.capabilities.requiresVariant ??
          selectedDiscoverRow.result.isGguf,
        modelFormat: resolvedModelFormat,
        baseModel: selectedDiscoverRow.result.baseModel ?? null,
        baseModelSource: selectedDiscoverRow.result.baseModel
          ? "huggingface"
          : null,
        baseModelHubId: selectedDiscoverRow.result.baseModel ?? null,
        isDownloaded: isResolvedOnDevice,
        isPartial: isResolvedPartial,
        partialTransport:
          selectedCachedRow?.partialTransport ??
          selectedLocalRow?.partialTransport ??
          null,
        runtimeCapabilities:
          selectedCachedRow?.capabilities ?? selectedLocalRow?.capabilities,
        capabilities: selectedDiscoverRow.capabilities,
        license: detectLicense(selectedDiscoverRow.result.tags),
        pipelineTag: selectedDiscoverRow.result.pipelineTag,
        libraryName: selectedDiscoverRow.result.libraryName,
        gated: selectedDiscoverRow.result.gated,
        private: selectedDiscoverRow.result.private,
        downloads: selectedDiscoverRow.result.downloads,
        downloadsAllTime: selectedDiscoverRow.result.downloadsAllTime,
        likes: selectedDiscoverRow.result.likes,
        totalParams: selectedDiscoverRow.result.totalParams,
        estimatedSizeBytes: selectedDiscoverRow.result.estimatedSizeBytes,
        cachedBytes: selectedCachedRow?.bytes,
        updatedAt: selectedDiscoverRow.result.updatedAt,
        createdAt: selectedDiscoverRow.result.createdAt,
        tags: selectedDiscoverRow.result.tags,
        quantMethod: selectedDiscoverRow.result.quantMethod,
      };
    }

    if (selectedCachedRow) {
      const resource = cachedResource(selectedCachedRow);
      const cachedSummary = isDatasetMode
        ? "Cached dataset, ready to use."
        : selectedCachedRow.isGguf
          ? "Cached GGUF repository ready for local inference."
          : "Cached checkpoint repository ready for local inference.";
      const mergedTags = selectedHfResult?.tags ?? selectedCachedRow.tags;
      const mergedPipelineTag =
        selectedHfResult?.pipelineTag ??
        selectedCachedRow.pipelineTag ??
        undefined;
      const mergedLibraryName =
        selectedHfResult?.libraryName ??
        selectedCachedRow.libraryName ??
        undefined;
      const mergedQuantMethod =
        selectedHfResult?.quantMethod ??
        selectedCachedRow.quantMethod ??
        undefined;
      const mergedBaseModel = selectedHfResult?.baseModel ?? null;
      return {
        id: selectedCachedRow.repoId,
        kind: "cache",
        resource,
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
        requiresVariant: selectedCachedRow.capabilities.requiresVariant,
        modelFormat: selectedCachedRow.modelFormat,
        baseModel: mergedBaseModel,
        baseModelSource: mergedBaseModel ? "huggingface" : null,
        baseModelHubId: mergedBaseModel,
        isDownloaded: !selectedCachedRow.partial,
        isPartial: selectedCachedRow.partial ?? false,
        partialTransport: selectedCachedRow.partialTransport ?? null,
        runtimeCapabilities: selectedCachedRow.capabilities,
        capabilities: detectViewCapabilities(
          mergedTags,
          mergedPipelineTag,
          selectedCachedRow.repoId,
          selectedCachedRow.loadId,
          selectedCachedRow.repo,
        ),
        license: detectLicense(mergedTags),
        pipelineTag: mergedPipelineTag,
        libraryName: mergedLibraryName,
        gated: selectedHfResult?.gated,
        private: selectedHfResult?.private,
        downloads: selectedHfResult?.downloads,
        downloadsAllTime: selectedHfResult?.downloadsAllTime,
        likes: selectedHfResult?.likes,
        totalParams: selectedHfResult?.totalParams,
        estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
        cachedBytes: selectedCachedRow.bytes,
        updatedAt: selectedHfResult?.updatedAt,
        createdAt: selectedHfResult?.createdAt,
        tags: mergedTags,
        quantMethod: mergedQuantMethod,
      };
    }

    if (selectedLocalRow) {
      const localHubRepoId =
        selectedLocalRow.source === "hf_cache" ? selectedLocalRow.repoId : null;
      const resource = localResource(selectedLocalRow, localHubRepoId);
      const localDisplayId = selectedLocalRow.repoId ?? selectedLocalRow.loadId;
      const isPartialHubCache =
        selectedLocalRow.source === "hf_cache" &&
        !!selectedLocalRow.partial &&
        !!selectedLocalRow.repoId;
      const mergedTags = selectedHfResult?.tags ?? selectedLocalRow.tags;
      const mergedPipelineTag =
        selectedHfResult?.pipelineTag ??
        selectedLocalRow.pipelineTag ??
        undefined;
      const mergedLibraryName =
        selectedHfResult?.libraryName ??
        selectedLocalRow.libraryName ??
        undefined;
      const mergedQuantMethod =
        selectedHfResult?.quantMethod ??
        selectedLocalRow.quantMethod ??
        undefined;
      const baseModelSummary =
        selectedLocalRow.baseModel && selectedHfResult
          ? buildSummary(selectedHfResult)
          : null;
      const localHubMetadata = localHubRepoId ? selectedHfResult : null;

      if (isPartialHubCache && selectedLocalRow.repoId) {
        return {
          id: selectedLocalRow.repoId,
          kind: "cache",
          resource,
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
          requiresVariant: selectedLocalRow.capabilities.requiresVariant,
          modelFormat: selectedLocalRow.modelFormat,
          isDownloaded: false,
          isPartial: true,
          partialTransport: selectedLocalRow.partialTransport ?? null,
          runtimeCapabilities: selectedLocalRow.capabilities,
          capabilities: detectViewCapabilities(
            mergedTags,
            mergedPipelineTag,
            selectedLocalRow.repoId,
            selectedLocalRow.loadId,
            selectedLocalRow.title,
            selectedLocalRow.path,
          ),
          license: detectLicense(mergedTags),
          pipelineTag: mergedPipelineTag,
          libraryName: mergedLibraryName,
          gated: selectedHfResult?.gated,
          private: selectedHfResult?.private,
          downloads: selectedHfResult?.downloads,
          downloadsAllTime: selectedHfResult?.downloadsAllTime,
          likes: selectedHfResult?.likes,
          totalParams: selectedHfResult?.totalParams,
          estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
          updatedAt: selectedHfResult?.updatedAt,
          createdAt: selectedHfResult?.createdAt,
          localUpdatedAt: selectedLocalRow.updatedAt,
          tags: mergedTags,
          quantMethod: mergedQuantMethod,
        };
      }

      return {
        id: selectedLocalRow.loadId,
        kind: "local",
        resource,
        displayId: localDisplayId,
        hubRepoId: localHubRepoId,
        owner: selectedLocalRow.owner,
        title: selectedLocalRow.title,
        summary: `${localSourceLabel(selectedLocalRow.source)} · ${localFormatLabel(
          selectedLocalRow,
        )}`,
        sourceLabel: selectedLocalRow.sourceLabel,
        path: selectedLocalRow.path,
        localSource: selectedLocalRow.source,
        isLocal: true,
        isGguf: selectedLocalRow.isGguf,
        requiresVariant: selectedLocalRow.capabilities.requiresVariant,
        modelFormat: selectedLocalRow.modelFormat,
        baseModel: selectedLocalRow.baseModel ?? null,
        baseModelSource: selectedLocalRow.baseModelSource ?? null,
        baseModelHubId: selectedLocalRow.baseModelHubId ?? null,
        baseModelSummary,
        adapterType: selectedLocalRow.adapterType ?? null,
        trainingMethod: selectedLocalRow.trainingMethod ?? null,
        isDownloaded: true,
        runtimeCapabilities: selectedLocalRow.capabilities,
        capabilities: detectViewCapabilities(
          mergedTags,
          mergedPipelineTag,
          selectedLocalRow.repoId ?? selectedLocalRow.title,
          selectedLocalRow.baseModel,
          selectedLocalRow.loadId,
          selectedLocalRow.path,
        ),
        license: detectLicense(mergedTags),
        pipelineTag: mergedPipelineTag,
        libraryName: mergedLibraryName,
        gated: localHubMetadata?.gated,
        private: localHubMetadata?.private,
        downloads: localHubMetadata?.downloads,
        downloadsAllTime: localHubMetadata?.downloadsAllTime,
        likes: localHubMetadata?.likes,
        totalParams: localHubMetadata?.totalParams,
        estimatedSizeBytes: localHubMetadata?.estimatedSizeBytes,
        updatedAt: localHubMetadata?.updatedAt,
        createdAt: localHubMetadata?.createdAt,
        localUpdatedAt: selectedLocalRow.updatedAt,
        tags: mergedTags,
        quantMethod: mergedQuantMethod,
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
