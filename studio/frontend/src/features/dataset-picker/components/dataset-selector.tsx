// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PickerShell } from "@/components/resource-picker/picker-shell";
import { isHfAuthError } from "@/components/resource-picker/picker-tab-state";
import { RetryButton } from "@/components/resource-picker/picker-tab-toggle";
import { SelectablePickerItem } from "@/components/resource-picker/selectable-picker-item";
import { useHfErrorToast } from "@/components/resource-picker/use-hf-error-toast";
import { usePickerState } from "@/components/resource-picker/use-picker-state";
import { Spinner } from "@/components/ui/spinner";
import {
  type CachedInventoryRow,
  hfApiToken,
  matchTokens,
  tokenizeQuery,
  useHfTokenStore,
  useHubDatasetSearch,
  useHubInfiniteScroll,
  useHubInventory,
  useLatestRef,
  useOnlineStatus,
} from "@/features/hub";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, Database02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type RefObject, useCallback, useMemo } from "react";
import { datasetDisplayName } from "../lib/display";

const DATASET_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.datasetPickerTab";

const TRIGGER_BASE = cn(
  "hub-menu-trigger field-soft inline-flex h-9 w-full cursor-pointer select-none items-center gap-1.5 rounded-[12px] px-3 text-ui-12p5 text-muted-foreground transition-colors",
  "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
);

type DeviceDatasetItem =
  | {
      kind: "local";
      key: string;
      title: string;
      detail: string;
      path: string;
    }
  | {
      kind: "cached";
      key: string;
      title: string;
      detail: string;
      repoId: string;
      cachePath: string | null;
    };

function hasExactDatasetMatch(
  query: string,
  tab: "device" | "hub",
  hubItems: readonly { id: string }[],
  deviceItems: readonly DeviceDatasetItem[],
): boolean {
  if (!query) {
    return false;
  }
  if (tab === "hub") {
    return hubItems.some((item) => item.id === query);
  }
  return deviceItems.some((item) =>
    item.kind === "cached" ? item.repoId === query : item.path === query,
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
      selectHfDataset(id, {
        knownCached: cached !== undefined,
        localPath: cached?.cachePath ?? null,
      });
    },
    [selectHfDataset, cachedDatasetById],
  );

  const deviceItems = useMemo<DeviceDatasetItem[]>(() => {
    const cachedItems: DeviceDatasetItem[] = cachedRows
      .filter((d) => !d.partial)
      .map((d) => ({
        kind: "cached",
        key: `cached:${d.repoId}`,
        title: d.repoId,
        detail: t("studio.datasetPicker.hfCacheLabel"),
        repoId: d.repoId,
        cachePath: d.cachePath ?? null,
      }));
    const localItems: DeviceDatasetItem[] = localRows.map((d) => ({
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
  const { activeQuery, handleQueryChange, tab } = pickerView;

  const {
    results: hfResults,
    isLoading: isLoadingHf,
    isLoadingMore: isLoadingHfMore,
    fetchMore: fetchMoreHf,
    retry: retryHf,
    error: hfError,
  } = useHubDatasetSearch(picker.debouncedHubQuery, {
    modelType,
    enabled: online && picker.open && tab === "hub",
    accessToken: hfApiToken(picker.debouncedHfToken),
  });

  const hubSearchActive = online && picker.open && tab === "hub";
  const hubSearchActiveRef = useLatestRef(hubSearchActive);
  const fetchMoreHfRef = useLatestRef(fetchMoreHf);
  useHfErrorToast(hubSearchActive ? hfError : null, "datasets");

  const fetchMoreOpenHf = useCallback(() => {
    if (!hubSearchActiveRef.current) {
      return;
    }
    fetchMoreHfRef.current();
  }, [hubSearchActiveRef, fetchMoreHfRef]);

  const { scrollRef, sentinelRef } = useHubInfiniteScroll(
    fetchMoreOpenHf,
    hfResults.length,
    { enabled: hubSearchActive },
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
    hfResults,
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
    picker.closePicker();
  };

  const selectedLocalDatasetTitle = useMemo(() => {
    if (datasetSource !== "upload" || !uploadedFile) {
      return null;
    }
    const selected = deviceItems.find(
      (item) => item.kind === "local" && item.path === uploadedFile,
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
      onOpenChange={picker.handleOpenChange}
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
        <DeviceList
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
            picker.closePicker();
          }}
        />
      }
      hubContent={
        <HubList
          items={hfResults}
          isLoading={isLoadingHf}
          isLoadingMore={isLoadingHfMore}
          value={dataset}
          hasQuery={activeQuery.length > 0}
          error={hfError}
          onPick={(id) => {
            selectHubDataset(id);
            picker.closePicker();
          }}
          onRetry={retryHf}
          sentinelRef={sentinelRef}
        />
      }
    />
  );
}

function DeviceList({
  items,
  isLoading,
  error,
  warning,
  hasQuery,
  onRetry,
  selectedLocalPath,
  selectedHfRepoId,
  onPick,
}: {
  items: DeviceDatasetItem[];
  isLoading: boolean;
  error: string | null;
  warning: boolean;
  hasQuery: boolean;
  onRetry: () => void;
  selectedLocalPath: string | null;
  selectedHfRepoId: string | null;
  onPick: (item: DeviceDatasetItem) => void;
}) {
  const t = useT();
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.datasetPicker.scanningLocal")}
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-ui-12p5 font-medium text-foreground">
            {t("studio.datasetPicker.couldntScan")}
          </p>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
    if (hasQuery) {
      return null;
    }
    return (
      <div className="px-4 py-8 text-center text-xs text-muted-foreground">
        {t("studio.datasetPicker.noLocalDatasets")}
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {items.map((item) => {
        const active =
          item.kind === "local"
            ? selectedLocalPath === item.path
            : selectedHfRepoId === item.repoId;
        return (
          <li key={item.key}>
            <SelectablePickerItem
              active={active}
              onSelect={() => onPick(item)}
              values={[item.kind === "local" ? item.path : item.repoId]}
            >
              <span className="block min-w-0 flex-1 truncate">
                {item.title}
              </span>
              <span className="ml-auto shrink-0 text-ui-10 text-muted-foreground">
                {item.detail}
              </span>
            </SelectablePickerItem>
          </li>
        );
      })}
      {warning && (
        <li className="px-2 py-1 text-ui-10p5 text-muted-foreground/80">
          {t("studio.datasetPicker.someLocationsUnscanned")}
        </li>
      )}
    </ul>
  );
}

function HubList({
  items,
  isLoading,
  isLoadingMore,
  value,
  hasQuery,
  error,
  onPick,
  onRetry,
  sentinelRef,
}: {
  items: ReadonlyArray<{ id: string; downloads?: number | null }>;
  isLoading: boolean;
  isLoadingMore: boolean;
  value: string | null;
  hasQuery: boolean;
  error: string | null;
  onPick: (id: string) => void;
  onRetry: () => void;
  sentinelRef: RefObject<HTMLDivElement | null>;
}) {
  const t = useT();
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.datasetPicker.searchingHub")}
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      const isAuth = isHfAuthError(error);
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-ui-12p5 font-medium text-foreground">
            {isAuth
              ? t("studio.datasetPicker.tokenRejectedTitle")
              : t("studio.datasetPicker.hubUnreachable")}
          </p>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {isAuth ? t("studio.datasetPicker.tokenRejectedBody") : error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
    if (hasQuery) {
      return null;
    }
    return (
      <div className="px-4 py-8 text-center text-xs text-muted-foreground">
        {t("studio.datasetPicker.noDatasetsFound")}
      </div>
    );
  }
  return (
    <>
      <ul className="flex flex-col gap-0.5 p-0.5">
        {items.map((d) => {
          const active = value === d.id;
          return (
            <li key={d.id}>
              <SelectablePickerItem
                active={active}
                onSelect={() => onPick(d.id)}
                values={[d.id]}
              >
                <span className="block min-w-0 flex-1 truncate">{d.id}</span>
              </SelectablePickerItem>
            </li>
          );
        })}
      </ul>
      <div ref={sentinelRef} className="h-px" />
      {isLoadingMore && (
        <div className="flex items-center justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      )}
    </>
  );
}
