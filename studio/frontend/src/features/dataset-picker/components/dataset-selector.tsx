// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hubResourceIdsEqual } from "@/components/resource-picker/hub-resource-id";
import { PICKER_FOCUS_VISIBLE_CLASS } from "@/components/resource-picker/picker-focus";
import { PickerShell } from "@/components/resource-picker/picker-shell";
import { useHfErrorToast } from "@/components/resource-picker/use-hf-error-toast";
import { usePickerHubPagination } from "@/components/resource-picker/use-picker-hub-pagination";
import { usePickerState } from "@/components/resource-picker/use-picker-state";
import {
  type CachedInventoryRow,
  hfApiToken,
  matchTokens,
  tokenizeQuery,
  useHfTokenStore,
  useHubDatasetSearch,
  useHubInfiniteScroll,
  useHubInventory,
  useOnlineStatus,
} from "@/features/hub";
import {
  cacheLocalPathMatchesSelection,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, Database02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo } from "react";
import { datasetDisplayName } from "../lib/display";
import {
  type DatasetDeviceItem,
  DatasetDeviceList,
  DatasetHubList,
} from "./dataset-selector-lists";

const DATASET_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.datasetPickerTab";

const TRIGGER_BASE = cn(
  "hub-menu-trigger field-soft inline-flex h-9 w-full cursor-pointer select-none items-center gap-1.5 rounded-[12px] px-3 text-ui-12p5 text-muted-foreground transition-colors",
  PICKER_FOCUS_VISIBLE_CLASS,
);

function findExactLocalDataset(
  query: string,
  deviceItems: readonly DatasetDeviceItem[],
): Extract<DatasetDeviceItem, { kind: "local" }> | undefined {
  return deviceItems.find(
    (item): item is Extract<DatasetDeviceItem, { kind: "local" }> =>
      item.kind === "local" && cacheLocalPathMatchesSelection(item.path, query),
  );
}

function findExactCachedDataset(
  query: string,
  deviceItems: readonly DatasetDeviceItem[],
): Extract<DatasetDeviceItem, { kind: "cached" }> | undefined {
  return deviceItems.find(
    (item): item is Extract<DatasetDeviceItem, { kind: "cached" }> =>
      item.kind === "cached" && hubResourceIdsEqual(item.repoId, query),
  );
}

function hasExactDatasetMatch(
  query: string,
  tab: "device" | "hub",
  hubItems: readonly { id: string }[],
  deviceItems: readonly DatasetDeviceItem[],
): boolean {
  if (!query) {
    return false;
  }
  if (tab === "hub") {
    return hubItems.some((item) => hubResourceIdsEqual(item.id, query));
  }
  return (
    findExactCachedDataset(query, deviceItems) !== undefined ||
    findExactLocalDataset(query, deviceItems) !== undefined
  );
}

function selectedDatasetDisplay({
  source,
  dataset,
  uploadedFile,
  localTitle,
}: {
  source: "huggingface" | "upload" | "s3";
  dataset: string | null;
  uploadedFile: string | null;
  localTitle: string | null;
}): string | null {
  if (source === "upload") {
    return uploadedFile
      ? (localTitle ?? datasetDisplayName(uploadedFile))
      : null;
  }
  return dataset ? datasetDisplayName(dataset) : null;
}

export function DatasetSelector({
  triggerDataTour = "studio-dataset-picker",
}: {
  triggerDataTour?: string;
}) {
  const t = useT();
  const dataset = useTrainingConfigStore((s) => s.dataset);
  const uploadedFile = useTrainingConfigStore((s) => s.uploadedFile);
  const datasetSource = useTrainingConfigStore((s) => s.datasetSource);
  const modelType = useTrainingConfigStore((s) => s.modelType);
  const selectHfDataset = useTrainingConfigStore((s) => s.selectHfDataset);
  const selectLocalDataset = useTrainingConfigStore(
    (s) => s.selectLocalDataset,
  );
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const picker = usePickerState({
    storageKey: DATASET_PICKER_TAB_STORAGE_KEY,
    hfToken,
    online,
  });
  const { closePicker } = picker;
  const {
    cachedRows,
    localRows,
    downloadedReady,
    inventoryError,
    inventoryWarning,
    refreshInventory,
  } = useHubInventory({ kind: "datasets", enabled: picker.open });
  const isLoadingLocal = !downloadedReady;
  const localError =
    inventoryError && localRows.length === 0 && cachedRows.length === 0
      ? t("studio.datasetPicker.couldntScan")
      : null;
  const retryLocalDatasets = useCallback(() => {
    refreshInventory().catch(() => undefined);
  }, [refreshInventory]);

  const cachedDatasetById = useMemo(() => {
    const map = new Map<string, CachedInventoryRow>();
    for (const row of cachedRows) {
      if (!row.partial) {
        map.set(row.repoId.toLowerCase(), row);
      }
    }
    return map;
  }, [cachedRows]);

  const selectHubDataset = useCallback(
    (id: string) => {
      const cached = cachedDatasetById.get(id.trim().toLowerCase());
      const canonicalId = cached?.repoId ?? id.trim();
      selectHfDataset(canonicalId, {
        knownCached: cached !== undefined,
        localPath: cached?.cachePath ?? null,
      });
    },
    [selectHfDataset, cachedDatasetById],
  );

  const deviceItems = useMemo<DatasetDeviceItem[]>(() => {
    const cachedItems: DatasetDeviceItem[] = cachedRows
      .filter((d) => !d.partial)
      .map((d) => ({
        kind: "cached",
        key: `cached:${d.repoId}`,
        title: d.repoId,
        detail: t("studio.datasetPicker.hfCacheLabel"),
        repoId: d.repoId,
        cachePath: d.cachePath ?? null,
      }));
    const localItems: DatasetDeviceItem[] = localRows.map((d) => ({
      kind: "local",
      key: `local:${d.path}`,
      title: d.title || d.id,
      detail: d.sourceLabel,
      path: d.path,
    }));
    return [...cachedItems, ...localItems].sort((a, b) =>
      a.title.localeCompare(b.title),
    );
  }, [cachedRows, localRows, t]);

  const pickerView = picker.getViewState({
    hasDeviceItems: deviceItems.length > 0,
    isLoadingDevice: isLoadingLocal,
  });
  const { activeQuery, handleOpenChange, handleQueryChange, tab } = pickerView;

  const {
    results: hfResults,
    isLoading: isLoadingHf,
    isLoadingMore: isLoadingHfMore,
    fetchMore: fetchMoreHf,
    retry: retryHf,
    error: hfError,
    scannedCount: scannedHfCount,
    hasMore: hasMoreHf,
  } = useHubDatasetSearch(picker.debouncedHubQuery, {
    modelType,
    enabled: online && picker.open && tab === "hub",
    accessToken: hfApiToken(picker.debouncedHfToken),
  });

  const hubSearchActive = online && picker.open && tab === "hub";
  useHfErrorToast(hubSearchActive ? hfError : null, "datasets");

  const hubItems = useMemo(() => {
    if (
      datasetSource !== "huggingface" ||
      !dataset ||
      hfResults.some((item) => hubResourceIdsEqual(item.id, dataset))
    ) {
      return hfResults;
    }
    return [...hfResults, { id: dataset }];
  }, [dataset, datasetSource, hfResults]);

  const hubPagination = usePickerHubPagination({
    enabled: hubSearchActive,
    fetchMore: fetchMoreHf,
    hasMore: hasMoreHf,
    isFetching: isLoadingHf || isLoadingHfMore,
    resetKey: picker.debouncedHubQuery,
    resultCount: hfResults.length,
    scannedCount: scannedHfCount,
  });

  const { scrollRef, sentinelRef } = useHubInfiniteScroll(
    hubPagination.fetchMore,
    hubPagination.signal,
    hubPagination.options,
  );

  const filteredDevice = useMemo(() => {
    const tokens = tokenizeQuery(picker.deviceQuery);
    if (tokens.length === 0) {
      return deviceItems;
    }
    return deviceItems.filter((item) =>
      matchTokens(
        item.kind === "cached" ? item.repoId : `${item.title} ${item.path}`,
        tokens,
      ),
    );
  }, [deviceItems, picker.deviceQuery]);

  const hasExactMatch = hasExactDatasetMatch(
    activeQuery,
    tab,
    hubItems,
    deviceItems,
  );
  const showUseThis = activeQuery.length > 0 && !hasExactMatch;
  const useThisLabel =
    tab === "hub"
      ? t("studio.datasetPicker.useAsHubDataset")
      : t("studio.datasetPicker.useAsLocalPath");

  const commitRaw = (raw: string) => {
    const next = raw.trim();
    if (!next) {
      return;
    }
    if (tab === "hub") {
      selectHubDataset(next);
    } else {
      selectLocalDataset(next);
    }
    closePicker();
  };

  const commitExactQuery = useCallback(
    (query: string) => {
      if (tab === "hub") {
        const item = hubItems.find((candidate) =>
          hubResourceIdsEqual(candidate.id, query),
        );
        if (!item) {
          return false;
        }
        selectHubDataset(item.id);
        closePicker();
        return true;
      }
      const cached = findExactCachedDataset(query, deviceItems);
      if (cached) {
        selectHfDataset(cached.repoId, {
          knownCached: true,
          localPath: cached.cachePath,
        });
        closePicker();
        return true;
      }
      const item = findExactLocalDataset(query, deviceItems);
      if (!item) {
        return false;
      }
      selectLocalDataset(item.path);
      closePicker();
      return true;
    },
    [
      closePicker,
      deviceItems,
      hubItems,
      selectHfDataset,
      selectHubDataset,
      selectLocalDataset,
      tab,
    ],
  );

  const selectedLocalDatasetTitle = useMemo(() => {
    if (datasetSource !== "upload" || !uploadedFile) {
      return null;
    }
    const selected = deviceItems.find(
      (item) =>
        item.kind === "local" &&
        cacheLocalPathMatchesSelection(item.path, uploadedFile),
    );
    return selected?.title ?? null;
  }, [datasetSource, uploadedFile, deviceItems]);

  const display = selectedDatasetDisplay({
    source: datasetSource,
    dataset,
    uploadedFile,
    localTitle: selectedLocalDatasetTitle,
  });

  return (
    <PickerShell
      open={picker.open}
      onOpenChange={handleOpenChange}
      tab={tab}
      onTabChange={picker.handleTabChange}
      hubQuery={picker.hubQuery}
      deviceQuery={picker.deviceQuery}
      activeQuery={activeQuery}
      onQueryChange={handleQueryChange}
      online={online}
      noun={t("studio.datasetPicker.noun")}
      isHubLoading={isLoadingHf}
      showUseThis={showUseThis}
      useThisLabel={useThisLabel}
      onUseThis={() => commitRaw(activeQuery)}
      onExactQueryCommit={commitExactQuery}
      placeholder={{
        hub: t("studio.datasetPicker.hubPlaceholder"),
        device: t("studio.datasetPicker.devicePlaceholder"),
      }}
      scrollRef={scrollRef}
      trigger={
        <button
          type="button"
          data-tour={triggerDataTour}
          className={cn(TRIGGER_BASE, "justify-between")}
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <HugeiconsIcon
              icon={Database02Icon}
              strokeWidth={1.75}
              className="size-3.5 shrink-0"
            />
            <span
              className={cn(
                "truncate font-medium",
                display ? "text-foreground" : "text-muted-foreground",
              )}
            >
              {display ?? t("studio.datasetPicker.selectDataset")}
            </span>
          </span>
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            strokeWidth={1.25}
            className="size-3.5 shrink-0 text-muted-foreground"
          />
        </button>
      }
      deviceContent={
        <DatasetDeviceList
          items={filteredDevice}
          isLoading={isLoadingLocal}
          error={localError}
          warning={inventoryWarning}
          hasQuery={activeQuery.length > 0}
          onRetry={retryLocalDatasets}
          selectedLocalPath={datasetSource === "upload" ? uploadedFile : null}
          selectedHfRepoId={datasetSource === "huggingface" ? dataset : null}
          onPick={(item) => {
            if (item.kind === "local") {
              selectLocalDataset(item.path);
            } else {
              selectHfDataset(item.repoId, {
                knownCached: true,
                localPath: item.cachePath,
              });
            }
            closePicker();
          }}
        />
      }
      hubContent={
        <DatasetHubList
          items={hubItems}
          isLoading={isLoadingHf}
          isLoadingMore={isLoadingHfMore}
          value={dataset}
          hasQuery={activeQuery.length > 0}
          error={hfError}
          onPick={(id) => {
            selectHubDataset(id);
            closePicker();
          }}
          onRetry={retryHf}
          sentinelRef={sentinelRef}
        />
      }
    />
  );
}
