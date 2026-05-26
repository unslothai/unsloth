// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  resolveInitialConfig,
  useChatModelRuntime,
  useChatRuntimeStore,
} from "@/features/chat";
import { useHubInventory } from "@/features/inventory";
import {
  inferTrainingModelTypeFromCapabilityKeys,
  useTrainingConfigStore,
} from "@/features/training";
import {
  type HfSortDirection,
  type HfSortKey,
  useDebouncedValue,
  useGpuInfo,
  useInfiniteScroll,
  useOnlineStatus,
} from "@/hooks";
import { ggufVariantsMatch, modelIdsMatch } from "@/lib/model-identity";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useCallback, useDeferredValue, useMemo, useState } from "react";
import { ModelInspector } from "./components/model-inspector";
import {
  ModelsCatalog,
  type ModelsCatalogHandlers,
  type ModelsCatalogPagination,
  type ModelsCatalogState,
} from "./components/models-catalog";
import { ModelsHeader } from "./components/models-header";
import { ModelsToolbar } from "./components/models-toolbar";
import { OnDeviceFoldersDialog } from "./components/on-device-folders-dialog";
import { useDiscoverSearch } from "./hooks/use-discover-search";
import { useHubModelVram } from "./hooks/use-hub-model-vram";
import { useModelsSelection } from "./hooks/use-models-selection";
import {
  type ChannelId,
  type ChannelPreset,
  findChannel,
} from "./lib/channels";
import { inventoryRowMatches, tokenizeQuery } from "./lib/inventory-search";
import { cacheOptionsForTraining } from "./lib/training-cache-options";
import {
  type TrainingSelectionConflictSnapshot,
  createTrainingSelectionTarget,
  getTrainingSelectionConflict,
  trainingSelectionConflictEqual,
} from "./lib/training-conflict";
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

type ModelLoadOptions = { ggufVariant?: string; expectedBytes?: number };
type PendingTrainRequest = {
  model: SelectedModelView;
  isDatasetMode: boolean;
  conflict: TrainingSelectionConflictSnapshot;
};

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

function currentTrainingSelectionConflict(
  target: SelectedModelView,
  isDatasetMode: boolean,
): TrainingSelectionConflictSnapshot | null {
  return getTrainingSelectionConflict(
    useTrainingConfigStore.getState(),
    createTrainingSelectionTarget(target, isDatasetMode),
  );
}

function useModelsTabState(): {
  tab: ModelsTab;
  setTab: (tab: ModelsTab) => void;
} {
  const navigate = useNavigate();
  const search = useSearch({ from: "/models" });
  const urlTab: ModelsTab | null = search.tab ?? null;
  const [fallbackTab, setFallbackTab] = useState<ModelsTab>(
    () => readModelsTabPreference() ?? "downloaded",
  );
  const tab = urlTab ?? fallbackTab;

  const setTab = useCallback(
    (next: ModelsTab) => {
      setFallbackTab(next);
      writeModelsTabPreference(next);
      void navigate({
        to: "/models",
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

function showUnsupportedTrainingModelToast(): void {
  toast.info("This model can't be selected for training", {
    description: "GGUF and adapter models are available for chat only.",
  });
}

export function ModelsPage() {
  const navigate = useNavigate();
  const gpu = useGpuInfo();
  const online = useOnlineStatus();
  const { selectModel, ejectModel, loadingModel, loadProgress } =
    useChatModelRuntime();

  const activeCheckpoint = useChatRuntimeStore(
    (state) => state.params.checkpoint,
  );
  const activeGgufVariant = useChatRuntimeStore(
    (state) => state.activeGgufVariant,
  );

  const { tab, setTab: setModelsTab } = useModelsTabState();
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<HfSortKey>("trendingScore");
  const [direction, setDirection] = useState<HfSortDirection>("desc");
  const [resourceType, setResourceType] =
    useState<ResourceTypeFilter>("models");
  const [activeChannelId, setActiveChannelId] = useState<ChannelId | null>(
    null,
  );
  const activeChannel: ChannelPreset | null = useMemo(
    () => findChannel(activeChannelId),
    [activeChannelId],
  );
  const [discoverFormat, setDiscoverFormat] =
    useState<ModelFormatFilter>("gguf");
  const [downloadedFormat, setDownloadedFormat] =
    useState<ModelFormatFilter>("all");
  const isDiscoverTab = tab === "discover";
  const isDatasetMode = resourceType === "datasets";
  // When a channel is active, the channel's recipe drives the format on the
  // discover tab. Manual format changes from the toolbar deactivate the
  // channel so the user has direct control again.
  const formatFilter = isDiscoverTab
    ? (activeChannel?.format ?? discoverFormat)
    : downloadedFormat;
  const setFormatFilter = useCallback(
    (next: ModelFormatFilter) => {
      if (isDiscoverTab) {
        setActiveChannelId(null);
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
  const [foldersDialogOpen, setFoldersDialogOpen] = useState(false);
  const [discoverFetchIntent, setDiscoverFetchIntent] = useState(0);
  const [pendingTrainRequest, setPendingTrainRequest] =
    useState<PendingTrainRequest | null>(null);

  const handleTabChange = useCallback(
    (next: ModelsTab) => {
      setMobileInspectorOpen(false);
      if (next === "downloaded") setActiveChannelId(null);
      setModelsTab(next);
    },
    [setModelsTab],
  );

  const handleResourceTypeChange = useCallback(
    (next: ResourceTypeFilter) => {
      if (next === resourceType) return;
      setResourceType(next);
      setDiscoverFormat("gguf");
      setDownloadedFormat("all");
      setCapabilityFilter("all");
      setActiveChannelId(null);
    },
    [resourceType],
  );

  const handleChannelSelect = useCallback((next: ChannelId | null) => {
    setActiveChannelId(next);
    setCapabilityFilter("all");
    setQuery("");
  }, []);

  const handleSortChange = useCallback((next: HfSortKey) => {
    setActiveChannelId(null);
    setSortBy(next);
    if (next === "trendingScore") setDirection("desc");
  }, []);

  const debouncedQuery = useDebouncedValue(query);
  const deferredDebouncedQuery = useDeferredValue(debouncedQuery);
  const hfToken = useHfTokenStore((s) => s.token);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const effectiveSort: HfSortKey = activeChannel?.sort ?? sortBy;
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

  const modelDiscoverRows = useMemo<DiscoverRow[]>(
    () => buildDiscoverRows(results, availableSet, partialSet),
    [results, availableSet, partialSet],
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
    setFormatFilter("all");
    setCapabilityFilter("all");
  }, [setFormatFilter]);
  const handleDiscoverFetchIntent = useCallback(() => {
    setDiscoverFetchIntent((value) => value + 1);
  }, []);

  const {
    scrollRef,
    sentinelRef,
    manualFetchAvailable: discoverManualFetchAvailable,
    fetchMoreManually: fetchMoreDiscoverManually,
  } = useInfiniteScroll(
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
    if (
      (loadingModel.modelFormat ?? selectedModel?.modelFormat ?? null) !==
      (selectedModel?.modelFormat ?? null)
    ) {
      return false;
    }
    return selectedRepoMatchesRuntime(
      selectedModel,
      loadingModel.id,
      loadingModel.ggufVariant ?? null,
    );
  }, [loadingModel, selectedModel]);

  const { vramInfo, minMemory } = useHubModelVram(selectedModel, gpu);

  const gpuLabel = gpu.available
    ? `${Math.floor(gpu.memoryTotalGb)} GB`
    : "Unavailable";
  const ramLabel =
    gpu.systemRamAvailableGb > 0
      ? `${Math.floor(gpu.systemRamAvailableGb)} GB`
      : "Unavailable";

  const loadSelectedModel = useCallback(
    async (
      opts: ModelLoadOptions = {},
      { refreshAfter = false }: { refreshAfter?: boolean } = {},
    ) => {
      if (!selectedModel) return;
      const runtimeId = selectedModel.resource.runId;
      const ggufVariant = opts.ggufVariant ?? null;
      const sameModel =
        modelIdsMatch(activeCheckpoint, runtimeId) &&
        ggufVariantsMatch(ggufVariant, activeGgufVariant);
      await selectModel({
        id: runtimeId,
        ggufVariant: opts.ggufVariant,
        modelFormat: selectedModel.modelFormat,
        isLora: false,
        isDownloaded: selectedModel.isDownloaded,
        isPartial: selectedModel.isPartial,
        localPath: selectedModel.resource.localPath,
        preferLocalCache:
          selectedModel.resource.cacheState === "cached" &&
          selectedModel.isDownloaded &&
          !selectedModel.isPartial,
        source: selectedModel.resource.cacheState === "local" ? "local" : "hub",
        expectedBytes: opts.expectedBytes,
        forceReload: sameModel,
        config: resolveInitialConfig(runtimeId, ggufVariant).config,
      });
      if (refreshAfter) void refreshInventory();
    },
    [
      selectedModel,
      activeCheckpoint,
      activeGgufVariant,
      selectModel,
      refreshInventory,
    ],
  );

  const handleLoad = useCallback(
    (opts: ModelLoadOptions) => loadSelectedModel(opts, { refreshAfter: true }),
    [loadSelectedModel],
  );

  const handleLoadLocal = useCallback(
    (opts: ModelLoadOptions = {}) => loadSelectedModel(opts),
    [loadSelectedModel],
  );

  const handleUseInChat = useCallback(() => {
    const runtime = useChatRuntimeStore.getState();
    if (
      !selectedRepoMatchesRuntime(
        selectedModel,
        runtime.params.checkpoint,
        runtime.activeGgufVariant,
      )
    ) {
      return;
    }
    void navigate({ to: "/chat" });
  }, [navigate, selectedModel]);

  const applyTrainSelection = useCallback(
    (target: SelectedModelView, datasetMode: boolean) => {
      const { resource } = target;
      const repoId = resource.repoId ?? target.hubRepoId ?? target.id;
      const store = useTrainingConfigStore.getState();
      if (datasetMode) {
        if (resource.cacheState === "local" && resource.localPath) {
          store.selectLocalDataset(resource.localPath);
        } else {
          store.selectHfDataset(repoId, cacheOptionsForTraining(resource));
        }
      } else {
        if (target.modelFormat === "gguf" || target.modelFormat === "adapter") {
          showUnsupportedTrainingModelToast();
          return;
        }
        const inferredType = inferTrainingModelTypeFromCapabilityKeys(
          target.capabilities.map((c) => c.key),
        );
        store.selectTrainingModel(resource.trainId, inferredType, {
          ...cacheOptionsForTraining(resource),
          modelFormat: target.modelFormat,
        });
      }
      void navigate({ to: "/studio" });
    },
    [navigate],
  );

  const handleTrain = useCallback(() => {
    if (!selectedModel) return;
    if (
      !isDatasetMode &&
      (selectedModel.modelFormat === "gguf" ||
        selectedModel.modelFormat === "adapter")
    ) {
      showUnsupportedTrainingModelToast();
      return;
    }
    const conflict = currentTrainingSelectionConflict(
      selectedModel,
      isDatasetMode,
    );
    if (conflict) {
      setPendingTrainRequest({ model: selectedModel, isDatasetMode, conflict });
      return;
    }
    applyTrainSelection(selectedModel, isDatasetMode);
  }, [selectedModel, isDatasetMode, applyTrainSelection]);

  const handleConfirmTrainReplace = useCallback(() => {
    const request = pendingTrainRequest;
    if (!request) return;
    const conflict = currentTrainingSelectionConflict(
      request.model,
      request.isDatasetMode,
    );
    if (
      conflict &&
      !trainingSelectionConflictEqual(conflict, request.conflict)
    ) {
      setPendingTrainRequest({ ...request, conflict });
      return;
    }
    setPendingTrainRequest(null);
    applyTrainSelection(request.model, request.isDatasetMode);
  }, [pendingTrainRequest, applyTrainSelection]);

  const handleTrainDialogOpenChange = useCallback((open: boolean) => {
    if (!open) setPendingTrainRequest(null);
  }, []);

  const trainConflictLabel = pendingTrainRequest?.isDatasetMode
    ? "dataset"
    : "model";

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
          onEject={() => void ejectModel()}
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
                <ModelInspector
                  model={selectedModel}
                  isDataset={isDatasetMode}
                  metadataUnavailable={metadataUnavailable}
                  selectionHiddenByFilters={selectionHiddenByFilters}
                  runtime={inspectorRuntime}
                  actions={inspectorActions}
                />
              </div>
            </div>
          </div>
        </section>
        <OnDeviceFoldersDialog
          open={foldersDialogOpen}
          onOpenChange={setFoldersDialogOpen}
          onInventoryChange={refreshInventory}
        />
        <AlertDialog
          open={pendingTrainRequest !== null}
          onOpenChange={handleTrainDialogOpenChange}
        >
          <AlertDialogContent size="sm">
            <AlertDialogHeader>
              <AlertDialogTitle>
                Replace training {trainConflictLabel}?
              </AlertDialogTitle>
              <AlertDialogDescription>
                Studio currently has{" "}
                <span className="break-all font-medium text-foreground">
                  {pendingTrainRequest?.conflict.id}
                </span>{" "}
                selected as the training {trainConflictLabel}. Continuing will
                replace it with{" "}
                <span className="break-all font-medium text-foreground">
                  {pendingTrainRequest?.model.displayId}
                </span>{" "}
                and refresh related training metadata.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Keep Draft</AlertDialogCancel>
              <AlertDialogAction onClick={handleConfirmTrainReplace}>
                Replace and Open Studio
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}
