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
        return {
          id: selectedLocalRow.loadId,
          loadId: selectedLocalRow.loadId,
          kind: "local",
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
          partialResumable: selectedLocalRow.partialResumable === true,
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
      const isResolvedPartial = selectedCachedRow
        ? Boolean(selectedCachedRow.partial)
        : selectedLocalRow?.source === "hf_cache"
          ? Boolean(selectedLocalRow.partial)
          : selectedDiscoverRow.isPartialOnDevice;
      const isResolvedOnDevice = selectedCachedRow
        ? !selectedCachedRow.partial
        : selectedLocalRow?.source === "hf_cache" && !selectedLocalRow.partial;
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
        loadId: selectedCachedRow?.loadId ?? selectedLocalRow?.loadId ?? null,
        kind: "discover",
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
        partialResumable:
          (selectedCachedRow ?? selectedLocalRow)?.partialResumable === true,
        capabilities: selectedDiscoverRow.capabilities,
        license: detectLicense(selectedDiscoverRow.result.tags),
        pipelineTag: selectedDiscoverRow.result.pipelineTag,
        // From the matched on-device row, like every field above: its inventory task is the
        // only record of the modality when the Hub metadata has no pipeline tag or only the
        // generic text-generation one.
        task: selectedCachedRow?.task ?? selectedLocalRow?.task ?? null,
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
        loadId: selectedCachedRow.loadId,
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
        requiresVariant: selectedCachedRow.capabilities.requiresVariant,
        modelFormat: selectedCachedRow.modelFormat,
        baseModel: mergedBaseModel,
        baseModelSource: mergedBaseModel ? "huggingface" : null,
        baseModelHubId: mergedBaseModel,
        isDownloaded: !selectedCachedRow.partial,
        isPartial: selectedCachedRow.partial ?? false,
        partialTransport: selectedCachedRow.partialTransport ?? null,
        partialResumable: selectedCachedRow.partialResumable === true,
        capabilities: detectViewCapabilities(
          mergedTags,
          mergedPipelineTag,
          selectedCachedRow.repoId,
          selectedCachedRow.loadId,
          selectedCachedRow.repo,
        ),
        license: detectLicense(mergedTags),
        pipelineTag: mergedPipelineTag,
        task: selectedCachedRow.task ?? null,
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
          loadId: selectedLocalRow.loadId,
          kind: "cache",
          displayId: selectedLocalRow.repoId,
          hubRepoId: selectedLocalRow.repoId,
          owner: selectedLocalRow.owner,
          title: selectedLocalRow.title,
          summary: selectedHfResult
            ? buildSummary(selectedHfResult)
            : "Partial download. Finish it from the card below, or delete it to free space.",
          sourceLabel: "Hub cache",
          path: selectedLocalRow.path,
          isLocal: false,
          isGguf: selectedLocalRow.isGguf,
          requiresVariant: selectedLocalRow.capabilities.requiresVariant,
          modelFormat: selectedLocalRow.modelFormat,
          isDownloaded: false,
          isPartial: true,
          partialTransport: selectedLocalRow.partialTransport ?? null,
          partialResumable: selectedLocalRow.partialResumable === true,
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
          task: selectedLocalRow.task ?? null,
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
        loadId: selectedLocalRow.loadId,
        kind: "local",
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
        task: selectedLocalRow.task ?? null,
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
