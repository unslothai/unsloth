// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  findCanonicalHubResourceId,
  hubResourceIdsEqual,
} from "@/components/resource-picker/hub-resource-id";
import { pathDisplayName } from "@/components/resource-picker/path-display-name";
import {
  type CachedInventoryRow,
  type HfModelResult,
  type LocalInventoryRow,
  type LocalSource,
  type ModelInventoryFormat,
  repoOf,
} from "@/features/hub";
import {
  type ModelTypeCapabilityFlags,
  cacheLocalPathMatchesSelection,
  isLocalTrainingModelSelection,
  trainingModelTypeFlagsFromMetadata,
  type validateTrainingModelCandidate,
} from "@/features/training";

export interface TrainModelDeviceItem {
  key: string;
  id: string;
  title: string;
  path: string;
  source: LocalSource;
  sourceLabel: string;
  knownCached: boolean;
  localPath: string | null;
  modelFormat: ModelInventoryFormat | null;
  modelTypeFlags: ModelTypeCapabilityFlags;
}

function trainModelSourceWeight(source: LocalSource): number {
  switch (source) {
    case "hf_cache":
      return 0;
    case "models_dir":
      return 1;
    case "custom":
      return 2;
    case "lmstudio":
      return 3;
    case "ollama":
      return 4;
    default:
      return 5;
  }
}

export function compareTrainModelDeviceItems(
  a: TrainModelDeviceItem,
  b: TrainModelDeviceItem,
): number {
  const titleCmp = (a.title || a.id).localeCompare(b.title || b.id, undefined, {
    sensitivity: "base",
  });
  if (titleCmp !== 0) {
    return titleCmp;
  }
  const sourceCmp =
    trainModelSourceWeight(a.source) - trainModelSourceWeight(b.source);
  if (sourceCmp !== 0) {
    return sourceCmp;
  }
  const pathCmp = a.path.localeCompare(b.path, undefined, {
    sensitivity: "base",
  });
  if (pathCmp !== 0) {
    return pathCmp;
  }
  return a.key.localeCompare(b.key);
}

export function trainModelDeviceItemMatchesSelection({
  item,
  selectedModel,
  selectedLocalPath,
  selectedFormat,
}: {
  item: TrainModelDeviceItem;
  selectedModel: string | null;
  selectedLocalPath: string | null;
  selectedFormat: ModelInventoryFormat | null;
}): boolean {
  if (
    !selectedModel ||
    (item.source === "hf_cache"
      ? !hubResourceIdsEqual(selectedModel, item.id)
      : selectedModel !== item.id)
  ) {
    return false;
  }
  if (
    selectedFormat &&
    item.modelFormat &&
    selectedFormat !== item.modelFormat
  ) {
    return false;
  }
  if (selectedLocalPath?.trim()) {
    return cacheLocalPathMatchesSelection(item.localPath, selectedLocalPath);
  }
  return true;
}

export function trainModelSelectionDisplayName({
  selectedModel,
  knownCached,
  selectedLocalPath,
  selectedFormat,
  deviceItems,
}: {
  selectedModel: string | null;
  knownCached: boolean;
  selectedLocalPath: string | null;
  selectedFormat: ModelInventoryFormat | null;
  deviceItems: readonly TrainModelDeviceItem[];
}): string | null {
  if (!selectedModel) {
    return null;
  }
  if (
    !isLocalTrainingModelSelection({
      model: selectedModel,
      knownCached,
      localPath: selectedLocalPath,
    })
  ) {
    return repoOf(selectedModel);
  }
  const selectedDeviceTitle = deviceItems.find((item) =>
    trainModelDeviceItemMatchesSelection({
      item,
      selectedModel,
      selectedLocalPath,
      selectedFormat,
    }),
  )?.title;
  return (
    selectedDeviceTitle ??
    pathDisplayName(selectedLocalPath?.trim() || selectedModel)
  );
}

export function toCachedTrainModelDeviceItem(
  row: CachedInventoryRow,
  sourceLabel: string,
): TrainModelDeviceItem {
  return {
    key: row.id,
    id: row.repoId,
    title: row.repoId,
    path: row.cachePath ?? row.repoId,
    source: "hf_cache",
    sourceLabel,
    knownCached: true,
    localPath: row.cachePath ?? null,
    modelFormat: row.modelFormat,
    modelTypeFlags: trainingModelTypeFlagsFromMetadata({
      tags: row.tags,
      pipelineTag: row.pipelineTag,
      identifiers: [row.repoId, row.repo],
    }),
  };
}

export function toLocalTrainModelDeviceItem(
  row: LocalInventoryRow,
): TrainModelDeviceItem {
  const cachedRepoId = row.source === "hf_cache" ? row.repoId?.trim() : null;
  return {
    key: row.id,
    id: cachedRepoId || row.loadId,
    title: row.repoId ?? row.title,
    path: row.path,
    source: row.source,
    sourceLabel: row.sourceLabel,
    knownCached: row.source === "hf_cache",
    localPath: row.path,
    modelFormat: row.modelFormat,
    modelTypeFlags: trainingModelTypeFlagsFromMetadata({
      tags: row.tags,
      pipelineTag: row.pipelineTag,
      identifiers: [row.repoId, row.loadId, row.title, row.path],
    }),
  };
}

export function hubTrainingModelCandidate(
  id: string,
  result: HfModelResult | undefined,
  cached: CachedInventoryRow | undefined,
  cachedLocal: LocalInventoryRow | undefined,
): Parameters<typeof validateTrainingModelCandidate>[0] {
  return {
    id,
    modelFormat: cached?.modelFormat ?? cachedLocal?.modelFormat ?? null,
    capabilities: cached?.capabilities ?? cachedLocal?.capabilities ?? null,
    pipelineTag:
      result?.pipelineTag ?? cached?.pipelineTag ?? cachedLocal?.pipelineTag,
    tags: result?.tags ?? cached?.tags ?? cachedLocal?.tags,
    libraryName:
      result?.libraryName ?? cached?.libraryName ?? cachedLocal?.libraryName,
    quantMethod:
      result?.quantMethod ?? cached?.quantMethod ?? cachedLocal?.quantMethod,
  };
}

export function hubTrainingModelTypeFlags(
  id: string,
  result: HfModelResult | undefined,
  cached: CachedInventoryRow | undefined,
  cachedLocal: LocalInventoryRow | undefined,
): ModelTypeCapabilityFlags {
  if (result) {
    return trainingModelTypeFlagsFromMetadata({
      tags: result.tags,
      pipelineTag: result.pipelineTag,
      identifiers: [result.id],
    });
  }
  if (cached) {
    return trainingModelTypeFlagsFromMetadata({
      tags: cached.tags,
      pipelineTag: cached.pipelineTag,
      identifiers: [cached.repoId, cached.repo],
    });
  }
  if (cachedLocal) {
    return trainingModelTypeFlagsFromMetadata({
      tags: cachedLocal.tags,
      pipelineTag: cachedLocal.pipelineTag,
      identifiers: [
        cachedLocal.repoId,
        cachedLocal.loadId,
        cachedLocal.title,
        cachedLocal.path,
      ],
    });
  }
  return trainingModelTypeFlagsFromMetadata({ identifiers: [id] });
}

export function hasExactTrainingModelMatch(
  query: string,
  tab: "device" | "hub",
  hubIds: readonly string[],
  deviceItems: readonly TrainModelDeviceItem[],
): boolean {
  if (!query) {
    return false;
  }
  if (tab === "hub") {
    return findCanonicalHubResourceId(query, hubIds) !== undefined;
  }
  return findExactTrainModelDeviceItem(query, deviceItems) !== undefined;
}

export function findExactTrainModelDeviceItem(
  query: string,
  deviceItems: readonly TrainModelDeviceItem[],
): TrainModelDeviceItem | undefined {
  return deviceItems.find(
    (item) =>
      item.id === query ||
      (item.source === "hf_cache" &&
        item.id.toLowerCase() === query.trim().toLowerCase()) ||
      cacheLocalPathMatchesSelection(item.path, query),
  );
}
