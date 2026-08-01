// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolveDevicePickerItem } from "@/components/resource-picker/device-item-match";
import { findCanonicalHubResourceId } from "@/components/resource-picker/hub-resource-id";
import type {
  CachedInventoryRow,
  HfModelResult,
  LocalInventoryRow,
  LocalSource,
} from "@/features/hub";
import {
  type ModelTypeCapabilityFlags,
  cacheLocalPathMatchesSelection,
  trainingModelTypeFlagsFromMetadata,
  type validateTrainingModelCandidate,
} from "@/features/training";
import {
  type TrainModelDisplayCandidate,
  toTrainModelDisplayCandidate,
} from "../lib/train-model-selection-display";

export interface TrainModelDeviceItem extends TrainModelDisplayCandidate {
  readonly key: string;
  readonly path: string;
  readonly sourceLabel: string;
  readonly knownCached: boolean;
  readonly modelTypeFlags: ModelTypeCapabilityFlags;
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
  sourceLabel: string,
): TrainModelDeviceItem {
  return {
    ...toTrainModelDisplayCandidate(row),
    key: row.id,
    path: row.path,
    sourceLabel,
    knownCached: row.source === "hf_cache",
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
  return resolveExactTrainModelDeviceItem(query, deviceItems).kind !== "none";
}

export function resolveExactTrainModelDeviceItem(
  query: string,
  deviceItems: readonly TrainModelDeviceItem[],
) {
  return resolveDevicePickerItem({
    query,
    items: deviceItems,
    canonicalMatch: (item, candidate) =>
      item.id === candidate ||
      (item.source === "hf_cache" &&
        item.id.toLowerCase() === candidate.trim().toLowerCase()) ||
      cacheLocalPathMatchesSelection(item.path, candidate),
    title: (item) => item.title,
  });
}
