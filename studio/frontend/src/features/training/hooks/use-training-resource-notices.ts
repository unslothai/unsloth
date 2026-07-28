// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  useHubInventory,
} from "@/features/hub";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { cacheLocalPathMatchesSelection } from "../lib/cache-reference";
import { isLocalTrainingModelSelection } from "../lib/model-selection";
import { isUntrainableModelFormat } from "../lib/model-support";
import {
  type TrainingResourceNotice,
  completeResourceSet,
  resolveTrainingResourceNotice,
} from "../lib/resource-availability";
import { useTrainingConfigStore } from "../stores/training-config-store";

const EMPTY_RESOURCE_SET = new Set<string>();

type ModelInventoryRow = CachedInventoryRow | LocalInventoryRow;

function modelRowPath(row: ModelInventoryRow): string | null {
  return row.kind === "cache" ? (row.cachePath ?? null) : row.path;
}

function modelRowMatchesSelection(
  row: ModelInventoryRow,
  model: string | null,
  localPath: string | null,
): boolean {
  const key = model?.trim().toLowerCase();
  if (!key) {
    return false;
  }
  if (
    row.id.toLowerCase() !== key &&
    row.loadId.toLowerCase() !== key &&
    row.repoId?.toLowerCase() !== key
  ) {
    return false;
  }
  return (
    !localPath || cacheLocalPathMatchesSelection(modelRowPath(row), localPath)
  );
}

function usableModelRow(
  row: ModelInventoryRow,
  requiresQuantizedCache: boolean,
): boolean {
  if (isUntrainableModelFormat(row.modelFormat)) {
    return false;
  }
  if (row.kind !== "cache" && row.source !== "hf_cache") {
    return false;
  }
  return !requiresQuantizedCache || !!row.quantMethod;
}

function resolveModelNotice({
  id,
  isLocal,
  knownCached,
  localPath,
  requiresQuantizedCache,
  rows,
  partialSet,
}: {
  id: string | null;
  isLocal: boolean;
  knownCached: boolean;
  localPath: string | null;
  requiresQuantizedCache: boolean;
  rows: readonly ModelInventoryRow[];
  partialSet: ReadonlySet<string>;
}): TrainingResourceNotice | null {
  const usableRows = rows.filter((row) =>
    usableModelRow(row, requiresQuantizedCache),
  );
  const selectedCacheIsUsable =
    !(knownCached && requiresQuantizedCache) ||
    usableRows.some((row) => modelRowMatchesSelection(row, id, localPath));
  return resolveTrainingResourceNotice({
    kind: "model",
    id,
    isLocal,
    knownCached: knownCached && selectedCacheIsUsable,
    localPath: selectedCacheIsUsable ? localPath : null,
    completeSet: selectedCacheIsUsable
      ? completeResourceSet(usableRows)
      : EMPTY_RESOURCE_SET,
    partialSet,
  });
}

function resolveDatasetNotice({
  id,
  isHf,
  knownCached,
  localPath,
  streaming,
  rows,
  partialSet,
}: {
  id: string | null;
  isHf: boolean;
  knownCached: boolean;
  localPath: string | null;
  streaming: boolean;
  rows: readonly ModelInventoryRow[];
  partialSet: ReadonlySet<string>;
}): TrainingResourceNotice | null {
  if (streaming) {
    return null;
  }
  return resolveTrainingResourceNotice({
    kind: "dataset",
    id,
    isLocal: !isHf,
    knownCached,
    localPath,
    completeSet: completeResourceSet(rows),
    partialSet,
  });
}

export function useTrainingResourceNotices(): TrainingResourceNotice[] {
  const {
    selectedModel,
    modelKnownCached,
    modelLocalPath,
    trainingMethod,
    datasetSource,
    dataset,
    datasetKnownCached,
    datasetLocalPath,
    datasetStreaming,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      modelKnownCached: s.modelKnownCached,
      modelLocalPath: s.modelLocalPath,
      trainingMethod: s.trainingMethod,
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      datasetKnownCached: s.datasetKnownCached,
      datasetLocalPath: s.datasetLocalPath,
      datasetStreaming: s.datasetStreaming,
    })),
  );

  const modelIsLocal = isLocalTrainingModelSelection({
    model: selectedModel,
    knownCached: modelKnownCached,
    localPath: modelLocalPath,
  });
  const datasetIsHf = datasetSource === "huggingface" && !!dataset;
  const modelInventory = useHubInventory({
    kind: "models",
    enabled: !!selectedModel && !modelIsLocal,
  });
  const datasetInventory = useHubInventory({
    kind: "datasets",
    enabled: datasetIsHf && !datasetStreaming,
  });

  return useMemo(() => {
    const modelNotice = resolveModelNotice({
      id: selectedModel,
      isLocal: modelIsLocal,
      knownCached: modelKnownCached,
      localPath: modelLocalPath,
      requiresQuantizedCache: trainingMethod === "qlora",
      rows: [...modelInventory.cachedRows, ...modelInventory.localRows],
      partialSet: modelInventory.partialSet,
    });
    const datasetNotice = resolveDatasetNotice({
      id: dataset,
      isHf: datasetIsHf,
      knownCached: datasetKnownCached,
      localPath: datasetLocalPath,
      streaming: datasetStreaming,
      rows: [...datasetInventory.cachedRows, ...datasetInventory.localRows],
      partialSet: datasetInventory.partialSet,
    });
    return [modelNotice, datasetNotice].filter(
      (notice): notice is TrainingResourceNotice => notice !== null,
    );
  }, [
    selectedModel,
    modelIsLocal,
    modelKnownCached,
    modelLocalPath,
    trainingMethod,
    modelInventory.cachedRows,
    modelInventory.localRows,
    modelInventory.partialSet,
    dataset,
    datasetIsHf,
    datasetKnownCached,
    datasetLocalPath,
    datasetStreaming,
    datasetInventory.cachedRows,
    datasetInventory.localRows,
    datasetInventory.partialSet,
  ]);
}
