// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { type CachedInventoryRow, useHubInventory } from "@/features/inventory";
import { useTrainingConfigStore } from "@/features/training";
import {
  useHfDatasetSearch,
  useInfiniteScroll,
  useLatestRef,
  useOnlineStatus,
} from "@/hooks";
import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { ArrowDown01Icon, Database02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type RefObject, useCallback, useMemo } from "react";
import { PickerShell } from "./hub-picker-shell";
import {
  RetryButton,
  isHfAuthError,
  looksLikeLocalPath,
} from "./picker-tab-toggle";
import { SelectablePickerItem } from "./selectable-picker-item";
import { useHfErrorToast } from "./use-hf-error-toast";
import { useHubPickerState } from "./use-hub-picker-state";

const DATASET_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.datasetPickerTab";

const TRIGGER_BASE = cn(
  "menu-trigger field-soft inline-flex h-9 w-full items-center gap-1.5 rounded-[12px] px-3 text-[12.5px] text-muted-foreground transition-colors",
  "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
);

const PATH_SEPARATOR_RE = /[\\/]/;
const UPLOADED_DATASET_HASH_PREFIX_RE = /^[0-9a-f]{32}_(.+)$/i;

function datasetDisplayName(value: string): string {
  const leaf = value.split(PATH_SEPARATOR_RE).pop() ?? value;
  return leaf.replace(UPLOADED_DATASET_HASH_PREFIX_RE, "$1");
}

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

export function TrainDatasetPicker() {
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
  const picker = useHubPickerState({
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
      ? "Couldn't scan local datasets"
      : null;
  const retryLocalDatasets = useCallback(() => {
    void refreshInventory();
  }, [refreshInventory]);

  const cachedDatasetById = useMemo(() => {
    const map = new Map<string, CachedInventoryRow>();
    for (const row of cachedRows) {
      if (!row.partial) map.set(row.repoId.toLowerCase(), row);
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
        detail: "Hugging Face cache",
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
  }, [cachedRows, localRows]);

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
  } = useHfDatasetSearch(picker.debouncedHubQuery, {
    modelType,
    enabled: online && picker.open && tab === "hub",
    accessToken: picker.debouncedHfToken || undefined,
  });

  const hubSearchActive = online && picker.open && tab === "hub";
  const hubSearchActiveRef = useLatestRef(hubSearchActive);
  const fetchMoreHfRef = useLatestRef(fetchMoreHf);
  useHfErrorToast(hubSearchActive ? hfError : null, "datasets");

  const fetchMoreOpenHf = useCallback(() => {
    if (!hubSearchActiveRef.current) return;
    fetchMoreHfRef.current();
  }, []);

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMoreOpenHf,
    hfResults.length,
    hubSearchActive,
  );

  const filteredDevice = useMemo(() => {
    const tokens = tokenizeQuery(picker.deviceQuery);
    if (tokens.length === 0) return deviceItems;
    return deviceItems.filter((item) =>
      matchTokens(
        item.kind === "cached" ? item.repoId : `${item.title} ${item.path}`,
        tokens,
      ),
    );
  }, [deviceItems, picker.deviceQuery]);

  const hasExactMatch =
    activeQuery.length === 0
      ? false
      : tab === "hub"
        ? hfResults.some((r) => r.id === activeQuery)
        : deviceItems.some(
            (item) =>
              (item.kind === "cached" && item.repoId === activeQuery) ||
              (item.kind === "local" && item.path === activeQuery),
          );
  const showUseThis =
    activeQuery.length > 0 &&
    !hasExactMatch &&
    (tab === "hub" || looksLikeLocalPath(activeQuery));
  const useThisLabel =
    tab === "hub" ? "Use as Hugging Face dataset" : "Use as local path";

  const commitRaw = (raw: string) => {
    const next = raw.trim();
    if (!next) return;
    if (tab === "hub") selectHubDataset(next);
    else selectLocalDataset(next);
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

  const display =
    datasetSource === "upload"
      ? uploadedFile
        ? (selectedLocalDatasetTitle ?? datasetDisplayName(uploadedFile))
        : null
      : dataset
        ? datasetDisplayName(dataset)
        : null;

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
      noun="datasets"
      isHubLoading={isLoadingHf}
      showUseThis={showUseThis}
      useThisLabel={useThisLabel}
      onUseThis={() => commitRaw(activeQuery)}
      placeholder={{
        hub: "Search Hugging Face datasets...",
        device: "Search local datasets...",
      }}
      scrollRef={scrollRef}
      trigger={
        <button
          type="button"
          data-tour="studio-dataset"
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
              {display ?? "Select dataset"}
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
            if (item.kind === "local") selectLocalDataset(item.path);
            else {
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
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> Scanning datasets on this device…
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-[12.5px] font-medium text-foreground">
            Couldn't scan local datasets
          </p>
          <p className="text-[11px] leading-snug text-muted-foreground">
            {error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
    if (hasQuery) return null;
    return (
      <div className="px-4 py-8 text-center text-xs text-muted-foreground">
        Nothing on this device yet. Download a dataset from the Hub, build one
        in Recipes, or upload a file.
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
            <SelectablePickerItem active={active} onSelect={() => onPick(item)}>
              <span className="block min-w-0 flex-1 cursor-text select-text truncate">
                {item.title}
              </span>
              <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                {item.kind === "cached" ? "HF cache" : item.detail}
              </span>
            </SelectablePickerItem>
          </li>
        );
      })}
      {warning && (
        <li className="px-2 py-1 text-[10.5px] text-muted-foreground/80">
          Some dataset locations could not be scanned.
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
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> Searching Hugging Face…
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      const isAuth = isHfAuthError(error);
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-[12.5px] font-medium text-foreground">
            {isAuth
              ? "Hugging Face token rejected"
              : "Couldn't reach Hugging Face"}
          </p>
          <p className="text-[11px] leading-snug text-muted-foreground">
            {isAuth
              ? "Update your token in Settings → Hugging Face, then retry."
              : error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
    if (hasQuery) return null;
    return (
      <div className="px-4 py-8 text-center text-xs text-muted-foreground">
        No datasets found.
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {items.map((d) => {
        const active = value === d.id;
        return (
          <li key={d.id}>
            <SelectablePickerItem active={active} onSelect={() => onPick(d.id)}>
              <span className="block min-w-0 flex-1 cursor-text select-text truncate">
                {d.id}
              </span>
            </SelectablePickerItem>
          </li>
        );
      })}
      <div ref={sentinelRef} className="h-px" />
      {isLoadingMore && (
        <div className="flex items-center justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      )}
    </ul>
  );
}
