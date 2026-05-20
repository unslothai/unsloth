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
} from "@/features/training/api/datasets-api";
import type { LocalDatasetInfo } from "@/features/training/types/datasets";
import { useDebouncedValue, useHfDatasetSearch } from "@/hooks";
import { useInventoryVersion } from "@/features/models";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  ArrowDown01Icon,
  Database02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  PICKER_TABS,
  type PickerTab,
  PickerTabToggle,
  isHfAuthError,
} from "./picker-tab-toggle";

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
  const inventoryVersion = useInventoryVersion();
  const loadedInventoryVersionRef = useRef<number>(-1);

  const debouncedHubQuery = useDebouncedValue(hubQuery);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);

  const {
    results: hfResults,
    isLoading: isLoadingHf,
    error: hfError,
  } = useHfDatasetSearch(debouncedHubQuery, {
    enabled: open && tab === "hub",
    accessToken: debouncedHfToken || undefined,
  });

  const lastToastedErrorRef = useRef<string | null>(null);
  useEffect(() => {
    if (!hfError) {
      lastToastedErrorRef.current = null;
      return;
    }
    if (lastToastedErrorRef.current === hfError) return;
    lastToastedErrorRef.current = hfError;
    if (isHfAuthError(hfError)) {
      toast.error("Hugging Face token rejected", {
        id: "hf-token-rejected",
        description:
          "Your token was refused. Update it in Settings → Hugging Face to search datasets.",
      });
    } else {
      toast.error("Couldn't reach Hugging Face", {
        id: "hf-search-failed",
        description: hfError,
      });
    }
  }, [hfError]);

  useEffect(() => {
    if (!open || tab !== "device") return;
    if (loadedInventoryVersionRef.current === inventoryVersion) return;
    let cancelled = false;
    setIsLoadingLocal(true);
    void Promise.all([
      listLocalDatasets().catch(() => ({ datasets: [] as LocalDatasetInfo[] })),
      listCachedDatasets().catch(() => [] as CachedDatasetRepo[]),
    ])
      .then(([localResponse, cached]) => {
        if (cancelled) return;
        setLocalDatasets(localResponse.datasets);
        setCachedDatasets(cached);
        loadedInventoryVersionRef.current = inventoryVersion;
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoadingLocal(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, tab, inventoryVersion]);

  const deviceItems = useMemo<DeviceDatasetItem[]>(() => {
    const cachedItems: DeviceDatasetItem[] = cachedDatasets.map((d) => ({
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
    const q = deviceQuery.trim().toLowerCase();
    if (!q) return deviceItems;
    return deviceItems.filter((item) => {
      const haystack =
        item.kind === "cached"
          ? item.repoId.toLowerCase()
          : `${item.title} ${item.path}`.toLowerCase();
      return haystack.includes(q);
    });
  }, [deviceItems, deviceQuery]);

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

          <div className="max-h-[320px] overflow-y-auto overscroll-contain rounded-[10px] [scrollbar-width:thin]">
            {tab === "device" ? (
              <DeviceList
                items={filteredDevice}
                isLoading={isLoadingLocal}
                selectedLocalPath={
                  datasetSource === "upload" ? uploadedFile : null
                }
                selectedHfRepoId={
                  datasetSource === "huggingface" ? dataset : null
                }
                onPick={(item) => {
                  if (item.kind === "local") selectLocalDataset(item.path);
                  else selectHfDataset(item.repoId);
                  setOpen(false);
                }}
              />
            ) : (
              <HubList
                items={hfResults}
                isLoading={isLoadingHf}
                value={dataset}
                error={hfError}
                onPick={(id) => {
                  selectHfDataset(id);
                  setOpen(false);
                }}
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
  selectedLocalPath,
  selectedHfRepoId,
  onPick,
}: {
  items: DeviceDatasetItem[];
  isLoading: boolean;
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
  value,
  error,
  onPick,
}: {
  items: ReadonlyArray<{ id: string; downloads?: number | null }>;
  isLoading: boolean;
  value: string | null;
  error: string | null;
  onPick: (id: string) => void;
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
        </div>
      );
    }
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
    </ul>
  );
}
