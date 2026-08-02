// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type InventoryResourceFormatHint,
  findCompleteHfCacheLocalRow,
  resolveInventoryResource,
} from "@/features/hub/inventory";
import type { HfModelResult } from "@/features/hub/hooks/use-hub-model-search";
import {
  useCallback,
  useDeferredValue,
  useMemo,
  useState,
} from "react";
import {
  resolveDiscoverSelection,
  resolveDownloadedSelection,
} from "../lib/selection-resolution";
import { ownerOf, repoOf } from "@/features/hub/lib/format";
import { buildDiscoverRows, isGgufLike } from "../lib/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  SelectedModelView,
} from "../types";
import { useSelectedModelMetadata } from "./use-selected-model-metadata";
import { useSelectedModelView } from "./use-selected-model-view";

export interface ModelsSelection {
  selectedId: string | null;
  setSelected: (id: string | null) => void;
  selectedModel: SelectedModelView | null;
  metadataUnavailable: boolean;
  selectionHiddenByFilters: boolean;
}

function stubDiscoverRow(id: string): DiscoverRow {
  return {
    id,
    owner: ownerOf(id),
    repo: repoOf(id),
    result: { id, downloads: 0, likes: 0, isGguf: isGgufLike(id) },
    isAvailableOnDevice: false,
    isPartialOnDevice: false,
    summary: "",
    capabilities: [],
  };
}

function preferredLocalRow(
  rows: readonly LocalInventoryRow[],
): LocalInventoryRow | null {
  return (
    rows.find((row) => !row.partial && row.source !== "hf_cache") ??
    rows.find((row) => !row.partial) ??
    rows[0] ??
    null
  );
}

export function useModelsSelection({
  isDiscoverTab,
  isDatasetMode,
  discoverRows,
  cachedRows,
  localRows,
  filteredDiscoverRows,
  filteredCachedRows,
  filteredLocalRows,
  results,
  accessToken,
  online,
}: {
  isDiscoverTab: boolean;
  isDatasetMode: boolean;
  discoverRows: DiscoverRow[];
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  filteredDiscoverRows: DiscoverRow[];
  filteredCachedRows: CachedInventoryRow[];
  filteredLocalRows: LocalInventoryRow[];
  results: HfModelResult[];
  accessToken: string | undefined;
  online: boolean;
}): ModelsSelection {
  // Per-tab selection so switching keeps each list's highlighted row.
  const [discoverSelectedId, setDiscoverSelectedId] = useState<string | null>(
    null,
  );
  const [downloadedSelectedId, setDownloadedSelectedId] = useState<
    string | null
  >(null);
  const [discoverSelectedSnapshot, setDiscoverSelectedSnapshot] =
    useState<DiscoverRow | null>(null);

  const [prevDatasetMode, setPrevDatasetMode] = useState(isDatasetMode);
  if (prevDatasetMode !== isDatasetMode) {
    setPrevDatasetMode(isDatasetMode);
    setDiscoverSelectedId(null);
    setDiscoverSelectedSnapshot(null);
  }

  const discoverRowById = useMemo(
    () => new Map(discoverRows.map((row) => [row.id, row])),
    [discoverRows],
  );
  const cachedRowById = useMemo(
    () => new Map(cachedRows.map((row) => [row.id, row])),
    [cachedRows],
  );
  const localRowById = useMemo(
    () => new Map(localRows.map((row) => [row.id, row])),
    [localRows],
  );
  const localRowsByRepo = useMemo(() => {
    const map = new Map<string, LocalInventoryRow[]>();
    for (const row of localRows) {
      if (!row.repoId) {
        continue;
      }
      const key = row.repoId.toLowerCase();
      const rows = map.get(key);
      if (rows) {
        rows.push(row);
      } else {
        map.set(key, [row]);
      }
    }
    return map;
  }, [localRows]);
  const resultByRepo = useMemo(
    () => new Map(results.map((row) => [row.id.toLowerCase(), row])),
    [results],
  );

  const discoverResolution = useMemo(
    () =>
      resolveDiscoverSelection({
        selectedId: discoverSelectedId,
        discoverRows,
        filteredDiscoverRows,
        selectedSnapshotId: discoverSelectedSnapshot?.id ?? null,
      }),
    [
      discoverRows,
      discoverSelectedId,
      discoverSelectedSnapshot,
      filteredDiscoverRows,
    ],
  );
  const downloadedResolution = useMemo(
    () =>
      resolveDownloadedSelection({
        selectedId: downloadedSelectedId,
        cachedRows,
        localRows,
        filteredCachedRows,
        filteredLocalRows,
      }),
    [
      cachedRows,
      downloadedSelectedId,
      filteredCachedRows,
      filteredLocalRows,
      localRows,
    ],
  );
  const selectedId = isDiscoverTab
    ? discoverResolution.selectedId
    : downloadedResolution.selectedId;
  const selectionHiddenByFilters = isDiscoverTab
    ? discoverResolution.hiddenByFilters
    : downloadedResolution.hiddenByFilters;
  const detailSelectedId = useDeferredValue(selectedId);
  const setSelected = useCallback(
    (id: string | null) => {
      if (isDiscoverTab) {
        setDiscoverSelectedId(id);
        setDiscoverSelectedSnapshot(
          id ? (discoverRowById.get(id) ?? stubDiscoverRow(id)) : null,
        );
      } else {
        setDownloadedSelectedId(id);
      }
    },
    [discoverRowById, isDiscoverTab],
  );

  const selectedDiscoverRow = useMemo(
    () =>
      isDiscoverTab && detailSelectedId
        ? (discoverRowById.get(detailSelectedId) ??
          (discoverSelectedSnapshot?.id === detailSelectedId
            ? discoverSelectedSnapshot
            : null))
        : null,
    [
      detailSelectedId,
      discoverRowById,
      discoverSelectedSnapshot,
      isDiscoverTab,
    ],
  );

  const selectedDiscoverResource = useMemo(() => {
    if (!selectedDiscoverRow) return null;
    const formatHint: InventoryResourceFormatHint = isDatasetMode
      ? null
      : selectedDiscoverRow.result.isGguf
        ? "gguf"
        : "non-gguf";
    return resolveInventoryResource({
      repoId: selectedDiscoverRow.result.id,
      cachedRows,
      localRows,
      formatHint,
    });
  }, [cachedRows, isDatasetMode, localRows, selectedDiscoverRow]);

  const selectedCachedRow = useMemo(() => {
    if (selectedDiscoverResource) return selectedDiscoverResource.cachedRow;
    if (!detailSelectedId) return null;
    const cached = cachedRowById.get(detailSelectedId) ?? null;
    if (!cached?.partial) return cached;
    const completeLocal = findCompleteHfCacheLocalRow(cached, localRows);
    return completeLocal ? null : cached;
  }, [cachedRowById, detailSelectedId, localRows, selectedDiscoverResource]);

  const selectedLocalRow = useMemo(() => {
    if (selectedDiscoverResource) return selectedDiscoverResource.localRow;
    if (!detailSelectedId) return null;
    const direct = localRowById.get(detailSelectedId);
    if (direct) return direct;
    const cached = cachedRowById.get(detailSelectedId);
    if (cached?.partial) {
      return findCompleteHfCacheLocalRow(cached, localRows);
    }
    const repoKey = isDiscoverTab
      ? selectedDiscoverRow?.result.id.toLowerCase()
      : detailSelectedId.toLowerCase();
    return repoKey ? preferredLocalRow(localRowsByRepo.get(repoKey) ?? []) : null;
  }, [
    detailSelectedId,
    cachedRowById,
    isDiscoverTab,
    localRowById,
    localRows,
    localRowsByRepo,
    selectedDiscoverResource,
    selectedDiscoverRow,
  ]);

  const selectedHubRepoId =
    selectedDiscoverRow?.result.id ??
    selectedCachedRow?.repoId ??
    (selectedLocalRow?.source === "hf_cache"
      ? selectedLocalRow.repoId
      : null) ??
    null;
  const selectedMetadataRepoId =
    selectedHubRepoId ?? selectedLocalRow?.baseModelHubId ?? null;

  const selectedResultFromFeed = useMemo(
    () =>
      selectedMetadataRepoId
        ? (resultByRepo.get(selectedMetadataRepoId.toLowerCase()) ?? null)
        : null,
    [resultByRepo, selectedMetadataRepoId],
  );

  const selectedRepoMetadata = useSelectedModelMetadata(selectedMetadataRepoId, {
    accessToken,
    enabled: !isDatasetMode && !selectedResultFromFeed,
    online,
  });
  const selectedHfResult = selectedResultFromFeed ?? selectedRepoMetadata.result;
  const metadataUnavailable =
    !isDatasetMode &&
    !selectedResultFromFeed &&
    !!selectedHubRepoId &&
    selectedRepoMetadata.error;

  const selectedDiscoverRowForView = useMemo(() => {
    if (!selectedDiscoverRow) return null;
    if (discoverRowById.has(selectedDiscoverRow.id)) return selectedDiscoverRow;
    if (
      !isDatasetMode &&
      selectedHfResult &&
      selectedHfResult.id.toLowerCase() === selectedDiscoverRow.id.toLowerCase()
    ) {
      return (
        buildDiscoverRows([selectedHfResult], cachedRows, localRows)[0] ??
        selectedDiscoverRow
      );
    }
    return selectedDiscoverRow;
  }, [
    selectedDiscoverRow,
    discoverRowById,
    isDatasetMode,
    selectedHfResult,
    cachedRows,
    localRows,
  ]);

  const selectedModel = useSelectedModelView({
    selectedDiscoverRow: selectedDiscoverRowForView,
    selectedCachedRow,
    selectedLocalRow,
    selectedHfResult,
    isDatasetMode,
  });

  return {
    selectedId,
    setSelected,
    selectedModel,
    metadataUnavailable,
    selectionHiddenByFilters,
  };
}
