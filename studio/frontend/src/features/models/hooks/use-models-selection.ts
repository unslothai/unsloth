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
  const selectedId = isDiscoverTab ? discoverSelectedId : downloadedSelectedId;
  const setSelected = useCallback(
    (id: string) => {
      if (isDiscoverTab) setDiscoverSelectedId(id);
      else setDownloadedSelectedId(id);
    },
    [isDiscoverTab],
  );

  const [selectedRepoMetadata, setSelectedRepoMetadata] =
    useState<ReturnType<typeof toHfModelResult>>(null);
  const [metadataErrorRepo, setMetadataErrorRepo] = useState<string | null>(
    null,
  );

  useEffect(() => {
    if (isDiscoverTab) {
      if (
        discoverSelectedId &&
        filteredDiscoverRows.some((row) => row.id === discoverSelectedId)
      )
        return;
      setDiscoverSelectedId(filteredDiscoverRows[0]?.id ?? null);
      return;
    }
    const stillSelected =
      downloadedSelectedId != null &&
      (filteredCachedRows.some((row) => row.id === downloadedSelectedId) ||
        filteredLocalRows.some((row) => row.id === downloadedSelectedId));
    if (stillSelected) return;
    setDownloadedSelectedId(
      filteredCachedRows[0]?.id ?? filteredLocalRows[0]?.id ?? null,
    );
  }, [
    isDiscoverTab,
    filteredCachedRows,
    filteredDiscoverRows,
    filteredLocalRows,
    discoverSelectedId,
    downloadedSelectedId,
  ]);

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

    if (!selectedHubRepoId) {
      setSelectedRepoMetadata(null);
      return;
    }

    if (selectedResultFromFeed) {
      setSelectedRepoMetadata(selectedResultFromFeed);
      return;
    }

    setSelectedRepoMetadata(null);
    void cachedModelInfo({
      name: selectedHubRepoId,
      ...(accessToken ? { accessToken } : {}),
    })
      .then((result) => {
        if (cancelled) return;
        setSelectedRepoMetadata(toHfModelResult(result));
        setMetadataErrorRepo((repo) =>
          repo === selectedHubRepoId ? null : repo,
        );
      })
      .catch(() => {
        if (cancelled) return;
        setMetadataErrorRepo(selectedHubRepoId);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedHubRepoId, selectedResultFromFeed, accessToken]);

  const selectedHfResult = selectedResultFromFeed ?? selectedRepoMetadata;
  const metadataUnavailable =
    !isDatasetMode &&
    !selectedResultFromFeed &&
    !!selectedHubRepoId &&
    metadataErrorRepo === selectedHubRepoId;

  const selectedModel = useSelectedModelView({
    selectedDiscoverRow,
    selectedCachedRow,
    selectedLocalRow,
    selectedHfResult,
    isDatasetMode,
  });

  return { selectedId, setSelected, selectedModel, metadataUnavailable };
}
