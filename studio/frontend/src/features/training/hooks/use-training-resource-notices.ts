// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type InventoryRow, useHubInventory } from "@/features/hub";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { isLocalTrainingModelSelection } from "../lib/model-selection";
import {
  type TrainingResourceNotice,
  completeResourceSet,
  resolveTrainingResourceNotice,
} from "../lib/resource-availability";
import { useTrainingConfigStore } from "../stores/training-config-store";

function usableModelRow(row: InventoryRow): boolean {
  if (!row.capabilities.canTrain) {
    return false;
  }
  if (row.kind !== "cache" && row.source !== "hf_cache") {
    return false;
  }
  return true;
}

function resolveModelNotice({
  id,
  isLocal,
  knownCached,
  localPath,
  rows,
  partialSet,
}: {
  id: string | null;
  isLocal: boolean;
  knownCached: boolean;
  localPath: string | null;
  rows: readonly InventoryRow[];
  partialSet: ReadonlySet<string>;
}): TrainingResourceNotice | null {
  const usableRows = rows.filter(usableModelRow);
  return resolveTrainingResourceNotice({
    kind: "model",
    id,
    isLocal,
    knownCached,
    localPath,
    completeSet: completeResourceSet(usableRows),
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
  rows: readonly InventoryRow[];
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
