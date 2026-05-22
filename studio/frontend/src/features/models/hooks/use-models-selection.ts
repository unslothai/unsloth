// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useState } from "react";
import type { HfModelResult } from "@/hooks";
import { cachedModelInfo } from "@/lib/hf-cache";
import { toHfModelResult } from "../lib/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  SelectedModelView,
} from "../types";
import { useSelectedModelView } from "./use-selected-model-view";

export interface ModelsSelection {
  selectedId: string | null;
  setSelected: (id: string) => void;
  selectedModel: SelectedModelView | null;
  metadataUnavailable: boolean;
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
}): ModelsSelection {
  // Per-tab selection so switching keeps each list's highlighted row.
  const [discoverSelectedId, setDiscoverSelectedId] = useState<string | null>(
    null,
  );
  const [downloadedSelectedId, setDownloadedSelectedId] = useState<
    string | null
  >(null);
  const effectiveDiscoverSelectedId =
    discoverSelectedId &&
    filteredDiscoverRows.some((row) => row.id === discoverSelectedId)
      ? discoverSelectedId
      : (filteredDiscoverRows[0]?.id ?? null);
  const downloadedStillSelected =
    downloadedSelectedId != null &&
    (filteredCachedRows.some((row) => row.id === downloadedSelectedId) ||
      filteredLocalRows.some((row) => row.id === downloadedSelectedId));
  const effectiveDownloadedSelectedId = downloadedStillSelected
    ? downloadedSelectedId
    : (filteredCachedRows[0]?.id ?? filteredLocalRows[0]?.id ?? null);
  const selectedId = isDiscoverTab
    ? effectiveDiscoverSelectedId
    : effectiveDownloadedSelectedId;
  const setSelected = useCallback(
    (id: string) => {
      if (isDiscoverTab) setDiscoverSelectedId(id);
      else setDownloadedSelectedId(id);
    },
    [isDiscoverTab],
  );

  const [metadataState, setMetadataState] = useState<{
    repoId: string;
    result: ReturnType<typeof toHfModelResult>;
    error: boolean;
  }>(() => ({ repoId: "", result: null, error: false }));

  const selectedDiscoverRow = useMemo(
    () =>
      isDiscoverTab && selectedId
        ? (discoverRows.find((row) => row.id === selectedId) ?? null)
        : null,
    [discoverRows, selectedId, isDiscoverTab],
  );

  const selectedCachedRow = useMemo(
    () =>
      selectedId
        ? (cachedRows.find((row) => row.id === selectedId) ?? null)
        : null,
    [cachedRows, selectedId],
  );

  const selectedLocalRow = useMemo(
    () =>
      selectedId
        ? (localRows.find((row) => row.id === selectedId) ?? null)
        : null,
    [localRows, selectedId],
  );

  const selectedHubRepoId =
    selectedDiscoverRow?.result.id ??
    selectedCachedRow?.repoId ??
    selectedLocalRow?.repoId ??
    null;

  const selectedResultFromFeed = useMemo(
    () =>
      selectedHubRepoId
        ? (results.find(
            (row) => row.id.toLowerCase() === selectedHubRepoId.toLowerCase(),
          ) ?? null)
        : null,
    [results, selectedHubRepoId],
  );

  useEffect(() => {
    let cancelled = false;

    if (isDatasetMode || !selectedHubRepoId || selectedResultFromFeed) return;

    void cachedModelInfo({
      name: selectedHubRepoId,
      ...(accessToken ? { accessToken } : {}),
    })
      .then((result) => {
        if (cancelled) return;
        setMetadataState({
          repoId: selectedHubRepoId,
          result: toHfModelResult(result),
          error: false,
        });
      })
      .catch(() => {
        if (cancelled) return;
        setMetadataState({
          repoId: selectedHubRepoId,
          result: null,
          error: true,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [isDatasetMode, selectedHubRepoId, selectedResultFromFeed, accessToken]);

  const selectedRepoMetadata =
    metadataState.repoId === selectedHubRepoId ? metadataState.result : null;
  const selectedHfResult = selectedResultFromFeed ?? selectedRepoMetadata;
  const metadataUnavailable =
    !isDatasetMode &&
    !selectedResultFromFeed &&
    !!selectedHubRepoId &&
    metadataState.repoId === selectedHubRepoId &&
    metadataState.error;

  const selectedModel = useSelectedModelView({
    selectedDiscoverRow,
    selectedCachedRow,
    selectedLocalRow,
    selectedHfResult,
    isDatasetMode,
  });

  return { selectedId, setSelected, selectedModel, metadataUnavailable };
}
