// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type DatasetDisplayCandidate,
  datasetSelectionDisplayName,
} from "@/features/dataset-picker";
import {
  type ModelInventoryFormat,
  buildLocalInventoryRows,
  useDeviceInventorySources,
} from "@/features/hub";
import {
  toTrainModelDisplayCandidate,
  trainModelSelectionDisplayName,
} from "@/features/train-model-picker";
import { isLocalTrainingModelSelection } from "@/features/training";
import { useMemo } from "react";

export function useTrainingResourceDisplayNames({
  selectedModel,
  modelKnownCached,
  modelLocalPath,
  modelFormat,
  datasetSource,
  dataset,
  uploadedFile,
}: {
  selectedModel: string | null;
  modelKnownCached: boolean;
  modelLocalPath: string | null;
  modelFormat: ModelInventoryFormat | null;
  datasetSource: "huggingface" | "upload" | "s3";
  dataset: string | null;
  uploadedFile: string | null;
}): {
  modelName: string | null;
  datasetName: string | null;
} {
  const modelIsLocal = isLocalTrainingModelSelection({
    model: selectedModel,
    knownCached: modelKnownCached,
    localPath: modelLocalPath,
  });
  const localModels = useDeviceInventorySources(["localModels"], {
    enabled: modelIsLocal,
  }).localModels.rows;
  const localDatasets = useDeviceInventorySources(["localDatasets"], {
    enabled: datasetSource === "upload" && uploadedFile !== null,
  }).localDatasets.rows;
  const modelCandidates = useMemo(
    () =>
      buildLocalInventoryRows(localModels).map(toTrainModelDisplayCandidate),
    [localModels],
  );
  const datasetCandidates = useMemo<DatasetDisplayCandidate[]>(
    () =>
      localDatasets.map((item) => ({
        path: item.path,
        title: item.label || item.id,
      })),
    [localDatasets],
  );

  return {
    modelName: trainModelSelectionDisplayName({
      selectedModel,
      knownCached: modelKnownCached,
      selectedLocalPath: modelLocalPath,
      selectedFormat: modelFormat,
      candidates: modelCandidates,
    }),
    datasetName: datasetSelectionDisplayName({
      source: datasetSource,
      dataset,
      uploadedFile,
      candidates: datasetCandidates,
    }),
  };
}
