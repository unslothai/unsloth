// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub page: browse + download surface only. Per-model config / chat-runtime /
// "use in train" integrations live outside this PR's scope and are intentionally
// not wired here.
import { useHubInventory } from "@/features/hub/inventory";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useGpuInfo } from "@/hooks/use-gpu-info";
import {
  type HfSortDirection,
  type HfSortKey,
} from "@/features/hub/hooks/use-hub-model-search";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { useIsHubDesktop } from "@/features/hub/hooks/use-is-hub-desktop";
import { useHubInfiniteScroll } from "@/features/hub/hooks/use-hub-infinite-scroll";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
} from "react";
import { ModelInspector } from "./catalog/model-inspector";
import {
  ModelsCatalog,
  type ModelsCatalogHandlers,
  type ModelsCatalogPagination,
  type ModelsCatalogState,
} from "./catalog/models-catalog";
import { ModelsHeader } from "./catalog/models-header";
import { ModelsToolbar } from "./catalog/models-toolbar";
import { ExternalLinkConfirmDialog } from "./catalog/external-link-confirm-dialog";
import { OnDeviceFoldersDialog } from "./catalog/on-device-folders-dialog";
import { useDiscoverSearch } from "./hooks/use-discover-search";
import { useHubModelVram } from "./hooks/use-hub-model-vram";
import { useModelsSelection } from "./hooks/use-models-selection";
import {
  type ChannelId,
  type ChannelPreset,
  findChannel,
} from "./lib/channels";
import { inventoryRowMatches, tokenizeQuery } from "./lib/inventory-search";
import {
  buildDiscoverRows,
  matchesCapability,
  matchesFormat,
} from "./lib/view-models";
import type {
  CachedInventoryRow,
  CapabilityFilter,
  DiscoverRow,
  LocalInventoryRow,
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
  SelectedModelView,
} from "./types";

const MODELS_TAB_STORAGE_KEY = "unsloth.hub.modelsTab";

const DEFAULT_DISCOVER_CHANNEL: ChannelId = "unsloth-trending";

type ModelLoadOptions = { ggufVariant?: string; expectedBytes?: number };

function discoveryInventorySignature(
  cachedRows: readonly CachedInventoryRow[],
  localRows: readonly LocalInventoryRow[],
): string {
  const parts: string[] = [];
  for (const row of cachedRows) {
    parts.push(
      `c:${row.repoId.toLowerCase()}:${row.modelFormat}:${row.partial ? "p" : "c"}`,
    );
  }
  for (const row of localRows) {
    parts.push(
      `l:${(row.repoId ?? row.id).toLowerCase()}:${row.modelFormat}:${row.partial ? "p" : "c"}`,
    );
  }
  return parts.sort().join("|");
}

function readModelsTabPreference(): ModelsTab | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const value = window.localStorage.getItem(MODELS_TAB_STORAGE_KEY);
    return value === "discover" || value === "downloaded" ? value : null;
  } catch {
    return null;
  }
}

function writeModelsTabPreference(tab: ModelsTab): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(MODELS_TAB_STORAGE_KEY, tab);
  } catch {
    return;
  }
}

function useModelsTabState(): {
  tab: ModelsTab;
  setTab: (tab: ModelsTab) => void;
} {
  const navigate = useNavigate();
  const search = useSearch({ from: "/hub" });
  const urlTab: ModelsTab | null = search.tab ?? null;
  const [fallbackTab, setFallbackTab] = useState<ModelsTab>(
    () => readModelsTabPreference() ?? "downloaded",
  );
  const tab = urlTab ?? fallbackTab;

  useEffect(() => {
    if (urlTab !== null) return;
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, tab: fallbackTab }),
      replace: true,
    });
  }, [urlTab, fallbackTab, navigate]);

  const setTab = useCallback(
    (next: ModelsTab) => {
      setFallbackTab(next);
      writeModelsTabPreference(next);
      void navigate({
        to: "/hub",
        search: (prev) => ({ ...prev, tab: next }),
        replace: true,
      });
    },
    [navigate],
  );

  return { tab, setTab };
}

function partitionByMatch<T extends CachedInventoryRow | LocalInventoryRow>(
  rows: T[],
  tokens: readonly string[],
): T[] {
  if (tokens.length === 0) return rows;
  const matches: T[] = [];
  const rest: T[] = [];
  for (const row of rows) {
    if (inventoryRowMatches(row, tokens)) matches.push(row);
    else rest.push(row);
  }
  return [...matches, ...rest];
}

function selectedRepoMatchesRuntime(
  selectedModel: SelectedModelView | null,
  runtimeId: string | null,
  ggufVariant: string | null,
): boolean {
  if (!selectedModel || !runtimeId) return false;
  if (!modelIdsMatch(runtimeId, selectedModel.resource.runId)) return false;
  if (selectedModel.modelFormat === "gguf") {
    const localPath =
      selectedModel.resource.localPath ?? selectedModel.path ?? "";
    return ggufVariant !== null || localPath.toLowerCase().endsWith(".gguf");
  }
  return ggufVariant === null;
}

export function ModelsPage() {
  const navigate = useNavigate();
  const gpu = useGpuInfo();
  const online = useOnlineStatus();

  // Hub does not own the chat runtime or training state. "Use in chat" simply
  // navigates to /chat with no model handoff in this PR's scope; the active
  // checkpoint / variant integration lands when the Hub-aware picker ships.
  const activeCheckpoint: string | null = null;
  const activeGgufVariant: string | null = null;
  // Stubs so the inspector + load-status memo still typecheck while the
  // chat-runtime integration is deferred. Cast through `as` so TS keeps the
  // shape instead of narrowing to `never`.
  const loadingModel = null as { id: string } | null;
  const loadProgress = null as {
    phase?: "downloading" | "starting";
  } | null;

  const { tab, setTab: setModelsTab } = useModelsTabState();
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<HfSortKey>(
    () => findChannel(DEFAULT_DISCOVER_CHANNEL)?.sort ?? "trendingScore",
  );
  const [direction, setDirection] = useState<HfSortDirection>("desc");
  const [resourceType, setResourceType] =
    useState<ResourceTypeFilter>("models");
  const [activeChannelId, setActiveChannelId] = useState<ChannelId | null>(
    DEFAULT_DISCOVER_CHANNEL,
  );
  const activeChannel: ChannelPreset | null = useMemo(
    () => findChannel(activeChannelId),
    [activeChannelId],
  );
  const [discoverFormat, setDiscoverFormat] = useState<ModelFormatFilter>(
    () => findChannel(DEFAULT_DISCOVER_CHANNEL)?.format ?? "gguf",
  );
  const [downloadedFormat, setDownloadedFormat] =
    useState<ModelFormatFilter>("all");
  const isDiscoverTab = tab === "discover";
  const isDatasetMode = resourceType === "datasets";
  const formatFilter = isDiscoverTab ? discoverFormat : downloadedFormat;
  const setFormatFilter = useCallback(
    (next: ModelFormatFilter) => {
      if (isDiscoverTab) {
        setDiscoverFormat(next);
      } else {
        setDownloadedFormat(next);
      }
    },
    [isDiscoverTab],
  );
  const [capabilityFilter, setCapabilityFilter] =
    useState<CapabilityFilter>("all");
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(false);
  const isHubDesktop = useIsHubDesktop();
  const [foldersDialogOpen, setFoldersDialogOpen] = useState(false);
  const [discoverFetchIntent, setDiscoverFetchIntent] = useState(0);

  const handleTabChange = useCallback(
    (next: ModelsTab) => {
      setMobileInspectorOpen(false);
      setModelsTab(next);
    },
    [setModelsTab],
  );

  const handleResourceTypeChange = useCallback(
    (next: ResourceTypeFilter) => {
      if (next === resourceType) return;
      setResourceType(next);
      setDownloadedFormat("all");
      setCapabilityFilter("all");
      if (next === "models") {
        const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
        setActiveChannelId(DEFAULT_DISCOVER_CHANNEL);
        setDiscoverFormat(preset?.format ?? "gguf");
        setSortBy(preset?.sort ?? "trendingScore");
        setDirection("desc");
      } else {
        setActiveChannelId(null);
      }
    },
    [resourceType],
  );

  const handleChannelSelect = useCallback((next: ChannelId | null) => {
    setActiveChannelId(next);
    setCapabilityFilter("all");
    setQuery("");
    const preset = findChannel(next);
    if (preset) {
      setDiscoverFormat(preset.format);
      setSortBy(preset.sort);
      setDirection("desc");
    }
  }, []);

  const handleSortChange = useCallback((next: HfSortKey) => {
    setSortBy(next);
    if (next === "trendingScore") setDirection("desc");
  }, []);

  const debouncedQuery = useDebouncedValue(query);
  const deferredDebouncedQuery = useDeferredValue(debouncedQuery);
  const hfToken = useHfTokenStore((s) => s.token);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const effectiveSort: HfSortKey = sortBy;
  const deferredFormatFilter = useDeferredValue(formatFilter);
  const deferredCapabilityFilter = useDeferredValue(capabilityFilter);

  const {
    results,
    datasetResults,
    scannedCount,
    isLoading,
    isLoadingMore,
    hasMore,
    fetchMore,
    searchError,
    handleRetrySearch,
  } = useDiscoverSearch({
    debouncedQuery,
    accessToken: debouncedHfToken || undefined,
    isDiscoverTab,
    isDatasetMode,
    sortBy: effectiveSort,
    direction,
    activeChannel,
    online,
  });

  const {
    cachedRows: effectiveCachedRows,
    localRows: effectiveLocalRows,
    availableSet,
    partialSet,
    downloadedReady,
    inventoryError,
    inventoryWarning,
    refreshInventory,
  } = useHubInventory({ kind: isDatasetMode ? "datasets" : "models" });

  const modelDiscoveryInventorySignature = useMemo(
    () =>
      discoveryInventorySignature(effectiveCachedRows, effectiveLocalRows),
    [effectiveCachedRows, effectiveLocalRows],
  );
  const modelDiscoverRows = useMemo<DiscoverRow[]>(
    () => buildDiscoverRows(results, effectiveCachedRows, effectiveLocalRows),
    [results, modelDiscoveryInventorySignature],
  );

  const datasetDiscoverRows = useMemo<DiscoverRow[]>(() => {
    if (!isDatasetMode) return [];
    return datasetResults.map((ds) => {
      const owner = ds.id.includes("/") ? ds.id.split("/")[0] : "Hub";
      const repo = ds.id.includes("/")
        ? ds.id.split("/").slice(1).join("/")
        : ds.id;
      const summaryParts: string[] = [];
      if (ds.taskCategories.length > 0)
        summaryParts.push(ds.taskCategories.slice(0, 2).join(", "));
      if (ds.totalExamples)
        summaryParts.push(`${ds.totalExamples.toLocaleString()} rows`);
      else if (ds.sizeCategory) summaryParts.push(ds.sizeCategory);
      const lower = ds.id.toLowerCase();
      return {
        id: ds.id,
        owner,
        repo,
        result: {
          id: ds.id,
          downloads: ds.downloads,
          likes: ds.likes,
          isGguf: false,
          tags: ds.plainTags,
        },
        isAvailableOnDevice: availableSet.has(lower),
        isPartialOnDevice: partialSet.has(lower),
        summary: summaryParts.join(" · ") || "Dataset",
        capabilities: [],
      };
    });
  }, [isDatasetMode, datasetResults, availableSet, partialSet]);

  const discoverRows = isDatasetMode ? datasetDiscoverRows : modelDiscoverRows;

  const filteredDiscoverRows = useMemo(
    () =>
      discoverRows.filter((row) => {
        if (isDatasetMode) return true;
        return (
          matchesFormat(row.result.isGguf, deferredFormatFilter) &&
          matchesCapability(row.capabilities, deferredCapabilityFilter)
        );
      }),
    [
      discoverRows,
      isDatasetMode,
      deferredFormatFilter,
      deferredCapabilityFilter,
    ],
  );

  const inventoryTokens = useMemo(
    () => (isDiscoverTab ? [] : tokenizeQuery(deferredDebouncedQuery)),
    [isDiscoverTab, deferredDebouncedQuery],
  );
  // Format filter is a deliberate scope narrowing — hard-filter it out.
  // The text query, by contrast, drives dim-not-filter on the On Device tab
  // (see ModelsCatalog) so selection survives typing. Matching rows are
  // partitioned to the top so the dim becomes a tail of the list, not noise
  // the user has to scan past.
  const filteredCachedRows = useMemo(
    () =>
      partitionByMatch(
        effectiveCachedRows.filter(
          (row) =>
            isDatasetMode ||
            matchesFormat(row.modelFormat, deferredFormatFilter),
        ),
        inventoryTokens,
      ),
    [effectiveCachedRows, isDatasetMode, deferredFormatFilter, inventoryTokens],
  );

  const filteredLocalRows = useMemo(
    () =>
      partitionByMatch(
        effectiveLocalRows.filter(
          (row) =>
            isDatasetMode ||
            matchesFormat(row.modelFormat, deferredFormatFilter),
        ),
        inventoryTokens,
      ),
    [effectiveLocalRows, isDatasetMode, deferredFormatFilter, inventoryTokens],
  );

  const filterResetSignature = useMemo(
    () =>
      JSON.stringify([
        deferredDebouncedQuery,
        resourceType,
        deferredFormatFilter,
        deferredCapabilityFilter,
        effectiveSort,
        direction,
        activeChannelId,
      ]),
    [
      deferredDebouncedQuery,
      resourceType,
      deferredFormatFilter,
      deferredCapabilityFilter,
      effectiveSort,
      direction,
      activeChannelId,
    ],
  );
  const handleClearFilters = useCallback(() => {
    if (isDiscoverTab) {
      setActiveChannelId(null);
      setDiscoverFormat("all");
    } else {
      setDownloadedFormat("all");
    }
    setCapabilityFilter("all");
  }, [isDiscoverTab]);
  const handleDiscoverFetchIntent = useCallback(() => {
    setDiscoverFetchIntent((value) => value + 1);
  }, []);

  const {
    scrollRef,
    sentinelRef,
    manualFetchAvailable: discoverManualFetchAvailable,
    fetchMoreManually: fetchMoreDiscoverManually,
  } = useHubInfiniteScroll(
    fetchMore,
    // Drive re-evaluation off the raw fetched count, not the filtered one —
    // the page-level format/capability filters can reject every incoming
    // row, leaving filteredDiscoverRows.length stalled while results keep
    // growing. Using the raw count guarantees the auto-fire effect re-runs
    // after each fetch lands so we don't dead-end on aggressive filters.
    scannedCount,
    {
      enabled: online && isDiscoverTab && hasMore,
      isFetching: isLoading || isLoadingMore,
      resultCount: filteredDiscoverRows.length,
      maxAutoFillFetches: 5,
      manualFetchAfterAutoFill: true,
      onFetchIntent: handleDiscoverFetchIntent,
      resetKey: filterResetSignature,
    },
  );

  const {
    selectedId,
    setSelected,
    selectedModel,
    metadataUnavailable,
    selectionHiddenByFilters,
  } = useModelsSelection({
    isDiscoverTab,
    isDatasetMode,
    discoverRows,
    cachedRows: effectiveCachedRows,
    localRows: effectiveLocalRows,
    filteredDiscoverRows,
    filteredCachedRows,
    filteredLocalRows,
    results,
    accessToken: debouncedHfToken || undefined,
    online,
  });

  const handleSelect = useCallback(
    (id: string) => {
      setSelected(id);
      // Only flip the mobile-drawer state when the catalog is the mobile
      // single-pane layout. On lg+ both panes are visible, so setting state
      // here just triggers an extra ModelsPage render that ripples through
      // the catalog's virtualizer during selection.
      if (
        typeof window !== "undefined" &&
        window.matchMedia("(max-width: 1023.98px)").matches
      ) {
        setMobileInspectorOpen(true);
      }
    },
    [setSelected],
  );
  const handleQueryChange = useCallback(
    (next: string) => {
      if (activeChannelId) setActiveChannelId(null);
      setQuery(next);
    },
    [activeChannelId],
  );
  const handleManageLocalFolders = useCallback(
    () => setFoldersDialogOpen(true),
    [],
  );
  const handleSwitchDevice = useCallback(
    () => handleTabChange("downloaded"),
    [handleTabChange],
  );

  const isActive = useMemo(
    () =>
      selectedRepoMatchesRuntime(
        selectedModel,
        activeCheckpoint,
        activeGgufVariant,
      ),
    [activeCheckpoint, activeGgufVariant, selectedModel],
  );

  const isLoadingThisModel = useMemo(() => {
    if (!loadingModel) return false;
    return selectedRepoMatchesRuntime(selectedModel, loadingModel.id, null);
  }, [loadingModel, selectedModel]);

  const { vramInfo, minMemory } = useHubModelVram(selectedModel, gpu);

  const gpuLabel = gpu.available
    ? `${Math.floor(gpu.memoryTotalGb)} GB`
    : "Unavailable";
  const ramLabel =
    gpu.systemRamAvailableGb > 0
      ? `${Math.floor(gpu.systemRamAvailableGb)} GB`
      : "Unavailable";

  // Out of scope for this PR: handleLoad / handleLoadLocal,
  // handleUseInChat with runtime check, handleTrain.
  // Hub right now is browse + download only; once the Hub-aware chat picker
  // and Hub-aware train picker ship, those handlers wire in there.
  const handleLoad = useCallback(
    (_opts: ModelLoadOptions) => undefined,
    [],
  );
  const handleLoadLocal = useCallback(
    (_opts: ModelLoadOptions = {}) => undefined,
    [],
  );
  const handleUseInChat = useCallback(() => {
    void navigate({ to: "/chat" });
  }, [navigate]);
  const handleTrain = useCallback(() => {
    // Hub → train integration ships in a later PR.
  }, []);

  const inspectorRuntime = useMemo(
    () => ({
      isActive,
      activeGgufVariant,
      isLoadingThisModel,
      loadingPhase: loadProgress?.phase,
      minMemory,
      vramInfo,
      gpuGb: gpu.available ? gpu.memoryTotalGb : undefined,
      systemRamGb:
        gpu.systemRamAvailableGb > 0 ? gpu.systemRamAvailableGb : undefined,
    }),
    [
      isActive,
      activeGgufVariant,
      isLoadingThisModel,
      loadProgress?.phase,
      minMemory,
      vramInfo,
      gpu.available,
      gpu.memoryTotalGb,
      gpu.systemRamAvailableGb,
    ],
  );

  const inspectorActions = useMemo(
    () => ({
      onLoad: handleLoad,
      onLoadLocal: handleLoadLocal,
      onUseInChat: handleUseInChat,
      onTrain: handleTrain,
      onInventoryChange: refreshInventory,
    }),
    [
      handleLoad,
      handleLoadLocal,
      handleUseInChat,
      handleTrain,
      refreshInventory,
    ],
  );

  const catalogState = useMemo<ModelsCatalogState>(
    () => ({
      tab,
      discoverRows: filteredDiscoverRows,
      cachedRows: filteredCachedRows,
      localRows: filteredLocalRows,
      selectedId,
      isLoading,
      downloadedReady,
      inventoryError,
      inventoryWarning,
      query,
      activeCheckpoint,
      activeGgufVariant,
      searchError,
      online,
      isDataset: isDatasetMode,
      inventoryTokens,
      scannedCount,
      loadingIntentCount: discoverFetchIntent,
      hasMore,
      manualFetchAvailable: discoverManualFetchAvailable,
      hasActiveFilters:
        deferredFormatFilter !== "all" || deferredCapabilityFilter !== "all",
    }),
    [
      tab,
      filteredDiscoverRows,
      filteredCachedRows,
      filteredLocalRows,
      selectedId,
      isLoading,
      downloadedReady,
      inventoryError,
      inventoryWarning,
      query,
      activeCheckpoint,
      activeGgufVariant,
      searchError,
      online,
      isDatasetMode,
      inventoryTokens,
      scannedCount,
      discoverFetchIntent,
      hasMore,
      discoverManualFetchAvailable,
      deferredFormatFilter,
      deferredCapabilityFilter,
    ],
  );

  const catalogPagination = useMemo<ModelsCatalogPagination>(
    () => ({
      scrollRef,
      sentinelRef,
      isLoadingMore,
    }),
    [scrollRef, sentinelRef, isLoadingMore],
  );

  const catalogHandlers = useMemo<ModelsCatalogHandlers>(
    () => ({
      onSelect: handleSelect,
      onFetchMore: fetchMoreDiscoverManually,
      onClearFilters: handleClearFilters,
      onRetry: handleRetrySearch,
      onInventoryChange: refreshInventory,
      onSwitchDevice: handleSwitchDevice,
    }),
    [
      handleSelect,
      fetchMoreDiscoverManually,
      handleClearFilters,
      handleRetrySearch,
      refreshInventory,
      handleSwitchDevice,
    ],
  );

  return (
    <div className="hub-page flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[1180px] flex-1 min-h-0 flex-col gap-6 px-5 pt-8 pb-16 sm:px-9 sm:pt-10 sm:pb-24">
        <ModelsHeader
          cachedCount={effectiveCachedRows.length}
          localCount={effectiveLocalRows.length}
          isDataset={isDatasetMode}
          gpuLabel={gpuLabel}
          ramLabel={ramLabel}
          activeCheckpoint={activeCheckpoint}
          activeGgufVariant={activeGgufVariant}
          onEject={() => undefined}
        />

        <section className="elevated-card flex min-h-0 flex-1 flex-col overflow-hidden bg-card">
          <div className="hub-side-surface shrink-0 px-4 pt-4 pb-6">
            <ModelsToolbar
              tab={tab}
              onTabChange={handleTabChange}
              query={query}
              onQueryChange={handleQueryChange}
              isLoading={isLoading}
              sortBy={effectiveSort}
              onSortChange={handleSortChange}
              resourceType={resourceType}
              onResourceTypeChange={handleResourceTypeChange}
              formatFilter={formatFilter}
              onFormatFilterChange={setFormatFilter}
              capabilityFilter={capabilityFilter}
              onCapabilityFilterChange={setCapabilityFilter}
              activeChannelId={activeChannelId}
              onChannelSelect={handleChannelSelect}
              onRefresh={handleRetrySearch}
              onManageLocalFolders={handleManageLocalFolders}
            />
          </div>

          <div className="flex min-h-0 flex-1 flex-col lg:grid lg:grid-cols-[360px_minmax(0,1fr)] xl:grid-cols-[400px_minmax(0,1fr)] 2xl:grid-cols-[440px_minmax(0,1fr)]">
            <div
              className={cn(
                "hub-side-surface flex min-h-0 min-w-0 flex-1 flex-col border-b border-border lg:flex-initial lg:border-b-0 lg:border-r",
                mobileInspectorOpen && "hidden lg:flex",
              )}
            >
              <ModelsCatalog
                state={catalogState}
                pagination={catalogPagination}
                handlers={catalogHandlers}
              />
            </div>

            <div
              className={cn(
                "hub-side-surface flex min-h-0 min-w-0 flex-1 flex-col lg:flex-initial",
                !mobileInspectorOpen && "hidden lg:flex",
              )}
            >
              <button
                type="button"
                onClick={() => setMobileInspectorOpen(false)}
                className="relative z-10 flex h-10 shrink-0 cursor-pointer select-none items-center gap-1.5 border-b border-border bg-transparent px-4 text-[12.5px] font-medium text-muted-foreground transition-colors hover:text-foreground lg:hidden"
              >
                <HugeiconsIcon
                  icon={ArrowLeft01Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                Back to list
              </button>
              <div className="flex min-h-0 flex-1 flex-col">
                {(isHubDesktop || mobileInspectorOpen) && (
                  <ModelInspector
                    model={selectedModel}
                    isDataset={isDatasetMode}
                    metadataUnavailable={metadataUnavailable}
                    selectionHiddenByFilters={selectionHiddenByFilters}
                    runtime={inspectorRuntime}
                    actions={inspectorActions}
                  />
                )}
              </div>
            </div>
          </div>
        </section>
        <OnDeviceFoldersDialog
          open={foldersDialogOpen}
          onOpenChange={setFoldersDialogOpen}
          onInventoryChange={refreshInventory}
        />
        <ExternalLinkConfirmDialog />
      </div>
    </div>
  );
}
