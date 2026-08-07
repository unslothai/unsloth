


import {
  type CachedDatasetRepo,
  type CachedModelRepo,
  type LocalModelInfo,
  type ModelInventoryFormat,
  fetchInventorySource,
  normalizeModelIdentity,
  useDeviceInventoryStore,
  useHfTokenStore,
  useInventoryVersion,
  useTokenScopedInventoryRequestOptions,
} from "@/features/hub";
import {
  type DatasetCacheInventoryIdentity,
  type DatasetCacheUsabilityIdentity,
  cachedInventoryPathMatchesSelection,
  createDatasetCacheUsabilityIdentity,
  datasetCacheUsabilityIdentitiesEqual,
  isLocalTrainingModelSelection,
  isTrainableModelFormat,
  trainingDatasetCacheRejections,
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

function datasetReferenceLoadPath(
  reference: CachedDatasetRepo,
): string | undefined {
  return reference.load_cache_path ?? reference.cache_path;
}

function datasetReferenceUsabilityIdentity(
  reference: CachedDatasetRepo,
  dataset: string,
  expected: DatasetCacheUsabilityIdentity,
): DatasetCacheUsabilityIdentity {
  return createDatasetCacheUsabilityIdentity({
    dataset,
    cachePath: datasetReferenceLoadPath(reference),
    subset: expected.subset,
    split: expected.split,
    streaming: expected.streaming,
  });
}

function datasetReferenceInventoryIdentity(
  reference: CachedDatasetRepo,
): DatasetCacheInventoryIdentity {
  return {
    cachePath: datasetReferenceLoadPath(reference),
    sizeBytes: reference.size_bytes,
    partial: reference.partial,
    partialTransport: reference.partial_transport,
  };
}

function modelIdentityMatches(
  row: CachedModelRepo | LocalModelInfo,
  key: string,
): boolean {
  return (
    ("repo_id" in row && normalizeModelIdentity(row.repo_id) === key) ||
    ("model_id" in row &&
      row.model_id != null &&
      normalizeModelIdentity(row.model_id) === key) ||
    (row.load_id != null && normalizeModelIdentity(row.load_id) === key) ||
    ("id" in row && normalizeModelIdentity(row.id) === key)
  );
}

function findModelReference(
  cachedRows: readonly CachedModelRepo[],
  localRows: readonly LocalModelInfo[],
  model: string,
): ModelCacheReference | null {
  const key = normalizeModelIdentity(model);
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
  expectedIdentity: DatasetCacheUsabilityIdentity,
  wasKnownCached: boolean,
): void {
  const current = useTrainingConfigStore.getState();
  if (
    current.datasetSource !== "huggingface" ||
    current.dataset !== expectedDataset
  ) {
    return;
  }
  const currentIdentity = createDatasetCacheUsabilityIdentity({
    dataset: expectedDataset,
    cachePath: current.datasetLocalPath,
    subset: current.datasetSubset,
    split: current.datasetSplit,
    streaming: current.datasetStreaming,
  });
  if (
    !datasetCacheUsabilityIdentitiesEqual(currentIdentity, expectedIdentity)
  ) {
    return;
  }
  if (current.datasetKnownCached !== wasKnownCached) {
    return;
  }
  const reference = findDatasetReference(rows, expectedDataset);
  if (!reference) {
    trainingDatasetCacheRejections.reset(expectedDataset);
    if (!wasKnownCached) {
      return;
    }
    current.clearSelectedDatasetCacheReference(
      expectedDataset,
      expectedIdentity.cachePath,
    );
    toast.warning(translate("studio.wizard.cachedDatasetGoneTitle"), {
      description: translate("studio.wizard.cachedDatasetGoneDescription"),
    });
    return;
  }

  const candidateIdentity = datasetReferenceUsabilityIdentity(
    reference,
    expectedDataset,
    expectedIdentity,
  );
  const inventoryIdentity = datasetReferenceInventoryIdentity(reference);
  if (wasKnownCached) {
    if (
      cachedInventoryPathMatchesSelection(
        datasetReferenceLoadPath(reference),
        expectedIdentity.cachePath,
      )
    ) {
      trainingDatasetCacheRejections.observe(
        candidateIdentity,
        inventoryIdentity,
      );
      return;
    }
    if (
      !trainingDatasetCacheRejections.shouldPromote(
        candidateIdentity,
        inventoryIdentity,
      )
    ) {
      current.clearSelectedDatasetCacheReference(
        expectedDataset,
        expectedIdentity.cachePath,
      );
      return;
    }
    current.setSelectedDatasetCacheReference(
      expectedDataset,
      datasetReferenceLoadPath(reference) ?? null,
    );
    return;
  }

  if (
    !trainingDatasetCacheRejections.shouldPromote(
      candidateIdentity,
      inventoryIdentity,
    )
  ) {
    return;
  }
  current.setSelectedDatasetCacheReference(
    expectedDataset,
    datasetReferenceLoadPath(reference) ?? null,
  );
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
    if (reference) {
      if (
        cachedInventoryPathMatchesSelection(
          reference.localPath,
          expectedLocalPath,
        )
      ) {
        return;
      }
      current.setSelectedModelCacheReference(expectedModel, reference);
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
  const cachedDatasetRows = useDeviceInventoryStore(
    (s) => s.cachedDatasets.rows,
  );
  const hfToken = useHfTokenStore((s) => s.token);
  const modelInventoryOptions = useTokenScopedInventoryRequestOptions(
    inventoryVersion,
    hfToken,
  );
  const {
    selectedModel,
    modelKnownCached,
    modelLocalPath,
    datasetSource,
    dataset,
    datasetSubset,
    datasetSplit,
    datasetStreaming,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      modelKnownCached: s.modelKnownCached,
      modelLocalPath: s.modelLocalPath,
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      datasetSubset: s.datasetSubset,
      datasetSplit: s.datasetSplit,
      datasetStreaming: s.datasetStreaming,
    })),
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: row replacement is an intentional force-refresh trigger even when the global version is unchanged
  useEffect(() => {
    if (datasetSource !== "huggingface" || !dataset) {
      return;
    }
    const current = useTrainingConfigStore.getState();
    if (
      current.datasetSource !== "huggingface" ||
      current.dataset !== dataset ||
      current.datasetSubset !== datasetSubset ||
      current.datasetSplit !== datasetSplit ||
      current.datasetStreaming !== datasetStreaming
    ) {
      return;
    }
    let cancelled = false;
    const expectedDataset = dataset;
    const expectedIdentity = createDatasetCacheUsabilityIdentity({
      dataset: expectedDataset,
      cachePath: current.datasetLocalPath,
      subset: current.datasetSubset,
      split: current.datasetSplit,
      streaming: current.datasetStreaming,
    });
    const expectedValidation =
      trainingDatasetCacheRejections.beginValidation(expectedIdentity);
    const wasKnownCached = current.datasetKnownCached;
    fetchInventorySource("cachedDatasets", { inventoryVersion })
      .then((rows) => {
        if (
          cancelled ||
          !trainingDatasetCacheRejections.isValidationCurrent(
            expectedValidation,
          )
        ) {
          return;
        }
        reconcileDatasetReference(
          rows,
          expectedDataset,
          expectedIdentity,
          wasKnownCached,
        );
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [
    inventoryVersion,
    cachedDatasetRows,
    datasetSource,
    dataset,
    datasetSubset,
    datasetSplit,
    datasetStreaming,
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
      fetchInventorySource("cachedModels", modelInventoryOptions),
      fetchInventorySource("localModels", modelInventoryOptions),
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
  }, [modelInventoryOptions, selectedModel, modelKnownCached, modelLocalPath]);
}
