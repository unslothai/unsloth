// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolveDevicePickerItem } from "@/components/resource-picker/device-item-match";
import {
  hubResourceIdsEqual,
  isValidHubResourceId,
  validateHubResourceId,
} from "@/components/resource-picker/hub-resource-id";
import { PICKER_TRIGGER_CLASS } from "@/components/resource-picker/picker-focus";
import {
  type PickerExactQueryCommitResult,
  PickerShell,
} from "@/components/resource-picker/picker-shell";
import {
  PICKER_TAB,
  type PickerTab,
} from "@/components/resource-picker/picker-tab-state";
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
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, Database02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useMemo } from "react";
import { datasetSelectionDisplayName } from "../lib/display";
import {
  type DatasetDeviceItem,
  DatasetDeviceList,
  DatasetHubList,
} from "./dataset-selector-lists";

const DATASET_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.datasetPickerTab";

function resolveExactDatasetDeviceItem(
  query: string,
  deviceItems: readonly DatasetDeviceItem[],
) {
  return resolveDevicePickerItem({
    query,
    items: deviceItems,
    canonicalMatch: (item, candidate) =>
      item.kind === "cached"
        ? hubResourceIdsEqual(item.repoId, candidate)
        : cacheLocalPathMatchesSelection(item.path, candidate),
    title: (item) => item.title,
  });
}

function hasExactDatasetMatch(
  query: string,
  tab: PickerTab,
  hubItems: readonly { id: string }[],
  deviceItems: readonly DatasetDeviceItem[],
): boolean {
  if (!query) {
    return false;
  }
  if (tab === PICKER_TAB.hub) {
    return hubItems.some((item) => hubResourceIdsEqual(item.id, query));
  }
  return resolveExactDatasetDeviceItem(query, deviceItems).kind !== "none";
}

export function DatasetSelector() {
  const t = useT();
  const navigate = useNavigate();
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
    inventorySettled,
    inventoryError,
    inventoryWarning,
    refreshInventory,
  } = useHubInventory({ kind: "datasets" });
  const localError =
    inventoryError && localRows.length === 0 && cachedRows.length === 0
      ? t("studio.datasetPicker.couldntScan")
      : null;
  const isLoadingLocal = localError === null && !inventorySettled;
  const retryLocalDatasets = useCallback(() => {
    refreshInventory().catch(() => undefined);
  }, [refreshInventory]);
  const openDataRecipes = useCallback(() => {
    closePicker();
    navigate({ to: "/data-recipes" }).catch(() => undefined);
  }, [closePicker, navigate]);

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
      const validation = validateHubResourceId(id);
      if (!validation.ok) {
        toast.error(t("studio.datasetPicker.cantUseDataset"), {
          description: t("studio.datasetPicker.reasonInvalidHubId"),
        });
        return false;
      }
      const cached = cachedDatasetById.get(validation.id.toLowerCase());
      const canonicalId = cached?.repoId ?? validation.id;
      selectHfDataset(canonicalId, {
        knownCached: cached !== undefined,
        localPath: cached?.cachePath ?? null,
      });
      return true;
    },
    [selectHfDataset, cachedDatasetById, t],
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
    const localItems: DatasetDeviceItem[] = localRows
      .filter((d) => !d.partial)
      .map((d) => ({
        kind: "local",
        key: `local:${d.path}`,
        title: d.title || d.id,
        detail:
          d.datasetSource === "recipe"
            ? t("studio.datasetPicker.sourceRecipe")
            : d.datasetSource === "upload"
              ? t("studio.datasetPicker.sourceUpload")
              : t("studio.datasetPicker.sourceLocal"),
        path: d.path,
      }));
    return [...cachedItems, ...localItems].sort((a, b) =>
      a.title.localeCompare(b.title, undefined, { sensitivity: "base" }),
    );
  }, [cachedRows, localRows, t]);

  const hasDeviceItems = deviceItems.length > 0;
  const pickerView = picker.getViewState({
    hasDeviceItems,
    isDeviceInventorySettled: inventorySettled,
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
    enabled: online && picker.open && tab === PICKER_TAB.hub,
    accessToken: hfApiToken(picker.debouncedHfToken),
  });

  const hubSearchActive = online && picker.open && tab === PICKER_TAB.hub;
  useHfErrorToast(hubSearchActive ? hfError : null, "datasets");

  const hubItems = useMemo(() => {
    const selectableResults = hfResults.filter((item) =>
      isValidHubResourceId(item.id),
    );
    if (
      datasetSource !== "huggingface" ||
      !dataset ||
      !isValidHubResourceId(dataset) ||
      selectableResults.some((item) => hubResourceIdsEqual(item.id, dataset))
    ) {
      return selectableResults;
    }
    return [...selectableResults, { id: dataset }];
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

  const { scrollRef, sentinelRef, fetchMoreManually } = useHubInfiniteScroll(
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
  const showUseThis =
    tab === PICKER_TAB.hub &&
    activeQuery.trim().length > 0 &&
    !hasExactMatch &&
    isValidHubResourceId(activeQuery);

  const commitHubQuery = (raw: string) => {
    const next = raw.trim();
    if (!next) {
      return;
    }
    if (!selectHubDataset(next)) {
      return;
    }
    closePicker();
  };

  const commitExactQuery = useCallback(
    (query: string): PickerExactQueryCommitResult => {
      if (tab === PICKER_TAB.hub) {
        const item = hubItems.find((candidate) =>
          hubResourceIdsEqual(candidate.id, query),
        );
        if (!item) {
          return { kind: "unhandled" };
        }
        if (selectHubDataset(item.id)) {
          closePicker();
        }
        return { kind: "handled" };
      }
      const resolution = resolveExactDatasetDeviceItem(query, deviceItems);
      if (resolution.kind === "ambiguous") {
        return {
          kind: "ambiguous",
          focusValue:
            resolution.firstItem.kind === "cached"
              ? resolution.firstItem.repoId
              : resolution.firstItem.path,
        };
      }
      if (resolution.kind === "none") {
        return { kind: "unhandled" };
      }
      const item = resolution.item;
      if (item.kind === "cached") {
        selectHfDataset(item.repoId, {
          knownCached: true,
          localPath: item.cachePath,
          preferLocalCache: true,
        });
        closePicker();
        return { kind: "handled" };
      }
      selectLocalDataset(item.path);
      closePicker();
      return { kind: "handled" };
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

  const display = datasetSelectionDisplayName({
    source: datasetSource,
    dataset,
    uploadedFile,
    candidates: localRows,
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
      useThisLabel={t("studio.datasetPicker.useAsHubDataset")}
      onUseThis={() => commitHubQuery(activeQuery)}
      onExactQueryCommit={commitExactQuery}
      placeholder={{
        hub: t("studio.datasetPicker.hubPlaceholder"),
        device: t("studio.datasetPicker.devicePlaceholder"),
      }}
      scrollRef={scrollRef}
      trigger={
        <button
          type="button"
          data-tour="studio-dataset-picker"
          title={
            (datasetSource === "upload" ? uploadedFile : dataset) ?? undefined
          }
          aria-label={`${t("studio.wizard.datasetLabel")}: ${
            display ?? t("studio.datasetPicker.selectDataset")
          }`}
          className={cn(PICKER_TRIGGER_CLASS, "w-full justify-between")}
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
          onOpenDataRecipes={openDataRecipes}
          selectedLocalPath={datasetSource === "upload" ? uploadedFile : null}
          selectedHfRepoId={datasetSource === "huggingface" ? dataset : null}
          onPick={(item) => {
            if (item.kind === "local") {
              selectLocalDataset(item.path);
            } else {
              selectHfDataset(item.repoId, {
                knownCached: true,
                localPath: item.cachePath,
                preferLocalCache: true,
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
            if (selectHubDataset(id)) {
              closePicker();
            }
          }}
          onLoadMore={fetchMoreManually}
          onRetry={retryHf}
          showLoadMore={hasMoreHf}
          sentinelRef={sentinelRef}
        />
      }
    />
  );
}
