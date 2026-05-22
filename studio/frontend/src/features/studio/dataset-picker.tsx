// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { useTrainingConfigStore } from "@/features/training";
import {
  type CachedDatasetRepo,
  listCachedDatasets,
  listLocalDatasets,
  type LocalDatasetInfo,
} from "@/features/training";
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useInfiniteScroll,
} from "@/hooks";
import { useInventoryVersion } from "@/features/models";
import { cn } from "@/lib/utils";
import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  ArrowDown01Icon,
  ArrowRight01Icon,
  Database02Icon,
  FolderSearchIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  PICKER_TABS,
  type PickerTab,
  PickerTabToggle,
  RetryButton,
  isHfAuthError,
  looksLikeLocalPath,
} from "./picker-tab-toggle";
import { useHfErrorToast } from "./use-hf-error-toast";

const TRIGGER_BASE = cn(
  "menu-trigger field-soft inline-flex h-9 w-full items-center gap-1.5 rounded-[12px] px-3 text-[12.5px] text-muted-foreground transition-colors",
  "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
);

function datasetDisplayName(value: string): string {
  return value.split("/").pop() ?? value;
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

  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<PickerTab>("hub");
  const [hubQuery, setHubQuery] = useState("");
  const [deviceQuery, setDeviceQuery] = useState("");

  const [localDatasets, setLocalDatasets] = useState<LocalDatasetInfo[]>([]);
  const [cachedDatasets, setCachedDatasets] = useState<CachedDatasetRepo[]>([]);
  const [isLoadingLocal, setIsLoadingLocal] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [localRetryToken, setLocalRetryToken] = useState(0);
  const inventoryVersion = useInventoryVersion();
  const loadedInventoryVersionRef = useRef<number>(-1);

  const debouncedHubQuery = useDebouncedValue(hubQuery);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);

  const {
    results: hfResults,
    isLoading: isLoadingHf,
    isLoadingMore: isLoadingHfMore,
    fetchMore: fetchMoreHf,
    retry: retryHf,
    error: hfError,
  } = useHfDatasetSearch(debouncedHubQuery, {
    modelType,
    enabled: open && tab === "hub",
    accessToken: debouncedHfToken || undefined,
  });

  useHfErrorToast(hfError, "datasets");

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMoreHf,
    hfResults.length,
    open && tab === "hub",
  );

  useEffect(() => {
    if (!open || tab !== "device") return;
    if (loadedInventoryVersionRef.current === inventoryVersion) return;
    let cancelled = false;
    setIsLoadingLocal(true);
    setLocalError(null);
    void Promise.all([listLocalDatasets(), listCachedDatasets()])
      .then(([localResponse, cached]) => {
        if (cancelled) return;
        setLocalDatasets(localResponse.datasets);
        setCachedDatasets(cached);
        loadedInventoryVersionRef.current = inventoryVersion;
      })
      .catch((err) => {
        if (cancelled) return;
        setLocalError(
          err instanceof Error ? err.message : "Couldn't scan local datasets",
        );
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoadingLocal(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, tab, inventoryVersion, localRetryToken]);

  const retryLocalDatasets = useCallback(() => {
    loadedInventoryVersionRef.current = -1;
    setLocalRetryToken((token) => token + 1);
  }, []);

  const deviceItems = useMemo<DeviceDatasetItem[]>(() => {
    const cachedItems: DeviceDatasetItem[] = cachedDatasets
      .filter((d) => !d.partial)
      .map((d) => ({
        kind: "cached",
        key: `cached:${d.repo_id}`,
        title: d.repo_id,
        detail: "Hugging Face cache",
        repoId: d.repo_id,
      }));
    const localItems: DeviceDatasetItem[] = localDatasets.map((d) => ({
      kind: "local",
      key: `local:${d.path}`,
      title: d.label || d.id,
      detail: d.path,
      path: d.path,
    }));
    return [...cachedItems, ...localItems].sort((a, b) =>
      a.title.localeCompare(b.title),
    );
  }, [cachedDatasets, localDatasets]);

  const filteredDevice = useMemo(() => {
    const tokens = tokenizeQuery(deviceQuery);
    if (tokens.length === 0) return deviceItems;
    return deviceItems.filter((item) =>
      matchTokens(
        item.kind === "cached" ? item.repoId : `${item.title} ${item.path}`,
        tokens,
      ),
    );
  }, [deviceItems, deviceQuery]);

  const activeQuery = (tab === "hub" ? hubQuery : deviceQuery).trim();
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
    if (tab === "hub") selectHfDataset(next);
    else selectLocalDataset(next);
    setOpen(false);
  };

  const display =
    datasetSource === "upload"
      ? uploadedFile
        ? datasetDisplayName(uploadedFile)
        : null
      : dataset
        ? datasetDisplayName(dataset)
        : null;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild={true}>
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
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        collisionPadding={16}
        className="w-[min(420px,calc(100vw-2rem))] rounded-2xl p-4"
      >
        <PickerTabToggle
          tab={tab}
          options={PICKER_TABS}
          onTabChange={setTab}
        />
        <div className="mt-2.5 flex flex-col gap-2">
          <div className="relative">
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={1.75}
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              autoFocus={true}
              value={tab === "hub" ? hubQuery : deviceQuery}
              onChange={(e) =>
                tab === "hub"
                  ? setHubQuery(e.target.value)
                  : setDeviceQuery(e.target.value)
              }
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                e.preventDefault();
                if (showUseThis) commitRaw(activeQuery);
              }}
              placeholder={
                tab === "hub"
                  ? "Search Hugging Face datasets..."
                  : "Search local datasets..."
              }
              className="field-soft h-9 rounded-full pl-9 text-[12.5px]"
            />
            {tab === "hub" && isLoadingHf && (
              <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            )}
          </div>

          <div
            ref={scrollRef}
            className="max-h-[320px] overflow-y-auto overscroll-contain rounded-[10px] [scrollbar-width:thin]"
          >
            {showUseThis && (
              <button
                type="button"
                onClick={() => commitRaw(activeQuery)}
                className="mb-1 flex w-full items-center gap-2 rounded-[8px] border border-dashed border-primary/30 bg-primary/[0.04] px-2.5 py-2 text-left text-[12.5px] transition-colors hover:bg-primary/[0.08]"
              >
                <HugeiconsIcon
                  icon={tab === "hub" ? Search01Icon : FolderSearchIcon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0 text-primary"
                />
                <span className="flex min-w-0 flex-1 flex-col leading-tight">
                  <span className="truncate font-medium text-foreground">
                    {activeQuery}
                  </span>
                  <span className="text-[10.5px] text-muted-foreground/80">
                    {useThisLabel}
                  </span>
                </span>
                <HugeiconsIcon
                  icon={ArrowRight01Icon}
                  strokeWidth={1.5}
                  className="size-3.5 shrink-0 text-muted-foreground/70"
                />
              </button>
            )}
            {tab === "device" ? (
              <DeviceList
                items={filteredDevice}
                isLoading={isLoadingLocal}
                error={localError}
                hasQuery={activeQuery.length > 0}
                onRetry={retryLocalDatasets}
                selectedLocalPath={
                  datasetSource === "upload" ? uploadedFile : null
                }
                selectedHfRepoId={
                  datasetSource === "huggingface" ? dataset : null
                }
                onPick={(item) => {
                  if (item.kind === "local") selectLocalDataset(item.path);
                  else selectHfDataset(item.repoId, { knownCached: true });
                  setOpen(false);
                }}
              />
            ) : (
              <HubList
                items={hfResults}
                isLoading={isLoadingHf}
                isLoadingMore={isLoadingHfMore}
                value={dataset}
                hasQuery={activeQuery.length > 0}
                error={hfError}
                onPick={(id) => {
                  selectHfDataset(id);
                  setOpen(false);
                }}
                onRetry={retryHf}
                sentinelRef={sentinelRef}
              />
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function DeviceList({
  items,
  isLoading,
  error,
  hasQuery,
  onRetry,
  selectedLocalPath,
  selectedHfRepoId,
  onPick,
}: {
  items: DeviceDatasetItem[];
  isLoading: boolean;
  error: string | null;
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
            <button
              type="button"
              onClick={() => onPick(item)}
              className={cn(
                "flex w-full items-center gap-2 rounded-[8px] px-2 py-1.5 text-left text-[12.5px] transition-colors hover:bg-foreground/[0.05]",
                active && "bg-foreground/[0.06]",
              )}
            >
              <span className="block min-w-0 flex-1 truncate">
                {item.title}
              </span>
              <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                {item.kind === "cached" ? "HF cache" : "Local"}
              </span>
            </button>
          </li>
        );
      })}
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
  sentinelRef: React.RefObject<HTMLDivElement | null>;
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
            {isAuth ? "Hugging Face token rejected" : "Couldn't reach Hugging Face"}
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
            <button
              type="button"
              onClick={() => onPick(d.id)}
              className={cn(
                "flex w-full items-center gap-2 rounded-[8px] px-2 py-1.5 text-left text-[12.5px] transition-colors hover:bg-foreground/[0.05]",
                active && "bg-foreground/[0.06]",
              )}
            >
              <span className="block min-w-0 flex-1 truncate">{d.id}</span>
            </button>
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
