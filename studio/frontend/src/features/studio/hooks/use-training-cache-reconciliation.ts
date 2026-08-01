// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type CachedDatasetRepo,
  type CachedModelRepo,
  type LocalModelInfo,
  type ModelInventoryFormat,
  fetchInventorySource,
  useHfTokenStore,
  useInventoryVersion,
} from "@/features/hub";
import {
  cachedInventoryPathMatchesSelection,
  isLocalTrainingModelSelection,
  isTrainableModelFormat,
  useTrainingConfigStore,
} from "@/features/training";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";

type ModelCacheReference = {
  localPath: string | null;
  modelFormat: ModelInventoryFormat | null;
};

function findDatasetReference(
  rows: readonly CachedDatasetRepo[],
  dataset: string,
): CachedDatasetRepo | undefined {
  const key = dataset.toLowerCase();
  return rows.find((row) => !row.partial && row.repo_id.toLowerCase() === key);
}

function modelIdentityMatches(
  row: CachedModelRepo | LocalModelInfo,
  key: string,
): boolean {
  return (
    ("repo_id" in row && row.repo_id.toLowerCase() === key) ||
    ("model_id" in row && row.model_id?.toLowerCase() === key) ||
    row.load_id?.toLowerCase() === key ||
    ("id" in row && row.id.toLowerCase() === key)
  );
}

function findModelReference(
  cachedRows: readonly CachedModelRepo[],
  localRows: readonly LocalModelInfo[],
  model: string,
): ModelCacheReference | null {
  const key = model.toLowerCase();
  const cachedMatch = cachedRows.find(
    (row) =>
      !row.partial &&
      isTrainableModelFormat(row.model_format) &&
      modelIdentityMatches(row, key),
  );
  if (cachedMatch) {
    return {
      localPath: cachedMatch.cache_path ?? null,
      modelFormat: cachedMatch.model_format ?? null,
    };
  }
  const localMatch = localRows.find(
    (row) =>
      row.source === "hf_cache" &&
      !row.partial &&
      isTrainableModelFormat(row.model_format) &&
      modelIdentityMatches(row, key),
  );
  if (!localMatch) {
    return null;
  }
  return {
    localPath: localMatch.path,
    modelFormat: localMatch.model_format ?? null,
  };
}

function reconcileDatasetReference(
  rows: readonly CachedDatasetRepo[],
  expectedDataset: string,
  expectedLocalPath: string | null,
  wasKnownCached: boolean,
): void {
  const current = useTrainingConfigStore.getState();
  if (
    current.datasetSource !== "huggingface" ||
    current.dataset !== expectedDataset
  ) {
    return;
  }
  const reference = findDatasetReference(rows, expectedDataset);
  if (wasKnownCached) {
    if (!current.datasetKnownCached) {
      return;
    }
    if (
      reference &&
      cachedInventoryPathMatchesSelection(
        reference.cache_path,
        expectedLocalPath,
      )
    ) {
      return;
    }
    current.clearSelectedDatasetCacheReference(
      expectedDataset,
      expectedLocalPath,
    );
    toast.warning(translate("studio.wizard.cachedDatasetGoneTitle"), {
      description: translate("studio.wizard.cachedDatasetGoneDescription"),
    });
    return;
  }
  if (!current.datasetKnownCached && reference) {
    current.setSelectedDatasetCacheReference(
      expectedDataset,
      reference.cache_path ?? null,
    );
  }
}

function reconcileModelReference(
  cachedRows: readonly CachedModelRepo[],
  localRows: readonly LocalModelInfo[],
  expectedModel: string,
  expectedLocalPath: string | null,
  wasKnownCached: boolean,
): void {
  const current = useTrainingConfigStore.getState();
  if (current.selectedModel !== expectedModel) {
    return;
  }
  const reference = findModelReference(cachedRows, localRows, expectedModel);
  if (wasKnownCached) {
    if (!current.modelKnownCached) {
      return;
    }
    if (
      reference &&
      cachedInventoryPathMatchesSelection(
        reference.localPath,
        expectedLocalPath,
      )
    ) {
      return;
    }
    current.clearSelectedModelCacheReference(expectedModel, expectedLocalPath);
    toast.warning(translate("studio.wizard.cachedModelGoneTitle"), {
      description: translate("studio.wizard.cachedModelGoneDescription"),
    });
    return;
  }
  if (!current.modelKnownCached && reference) {
    current.setSelectedModelCacheReference(expectedModel, reference);
  }
}

export function useTrainingCacheReconciliation(): void {
  const inventoryVersion = useInventoryVersion();
  const hfToken = useHfTokenStore((s) => s.token);
  const {
    selectedModel,
    modelKnownCached,
    modelLocalPath,
    datasetSource,
    dataset,
    datasetKnownCached,
    datasetLocalPath,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      modelKnownCached: s.modelKnownCached,
      modelLocalPath: s.modelLocalPath,
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      datasetKnownCached: s.datasetKnownCached,
      datasetLocalPath: s.datasetLocalPath,
    })),
  );

  useEffect(() => {
    if (datasetSource !== "huggingface" || !dataset) {
      return;
    }
    let cancelled = false;
    const expectedDataset = dataset;
    const expectedLocalPath = datasetLocalPath;
    const wasKnownCached = datasetKnownCached;
    fetchInventorySource("cachedDatasets", { inventoryVersion, hfToken })
      .then((rows) => {
        if (cancelled) {
          return;
        }
        reconcileDatasetReference(
          rows,
          expectedDataset,
          expectedLocalPath,
          wasKnownCached,
        );
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [
    inventoryVersion,
    hfToken,
    datasetSource,
    dataset,
    datasetKnownCached,
    datasetLocalPath,
  ]);

  useEffect(() => {
    if (
      !selectedModel ||
      isLocalTrainingModelSelection({
        model: selectedModel,
        knownCached: modelKnownCached,
        localPath: modelLocalPath,
      })
    ) {
      return;
    }
    let cancelled = false;
    const expectedModel = selectedModel;
    const expectedLocalPath = modelLocalPath;
    const wasKnownCached = modelKnownCached;
    Promise.all([
      fetchInventorySource("cachedModels", { inventoryVersion, hfToken }),
      fetchInventorySource("localModels", { inventoryVersion, hfToken }),
    ])
      .then(([cachedRows, localRows]) => {
        if (cancelled) {
          return;
        }
        reconcileModelReference(
          cachedRows,
          localRows,
          expectedModel,
          expectedLocalPath,
          wasKnownCached,
        );
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [
    inventoryVersion,
    hfToken,
    selectedModel,
    modelKnownCached,
    modelLocalPath,
  ]);
}
