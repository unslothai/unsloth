// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate, useSearch } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatModelRuntime, useChatRuntimeStore } from "@/features/chat";
import { useTrainingConfigStore } from "@/features/training";
import type { ModelType } from "@/types/training";
import {
  type HfSortDirection,
  type HfSortKey,
  useDebouncedValue,
  useGpuInfo,
  useHfDatasetSearch,
  useHfModelSearch,
  useInfiniteScroll,
  useOnlineStatus,
} from "@/hooks";
import { cachedModelInfo } from "@/lib/hf-cache";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { checkVramFit, estimateLoadingVram } from "@/lib/vram";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ChannelId,
  type ChannelPreset,
  findChannel,
} from "./lib/channels";
import { formatBytes } from "./lib/format";
import { inventoryRowMatches, tokenizeQuery } from "./lib/inventory-search";
import {
  buildDiscoverRows,
  matchesCapability,
  matchesFormat,
  toHfModelResult,
} from "./lib/view-models";
import { ModelInspector } from "./components/model-inspector";
import { ModelsCatalog } from "./components/models-catalog";
import { ModelsHeader } from "./components/models-header";
import { ModelsToolbar } from "./components/models-toolbar";
import { useHubInventory } from "./hooks/use-hub-inventory";
import { useFilterStarvedPause } from "./hooks/use-filter-starved-pause";
import { useSelectedModelView } from "./hooks/use-selected-model-view";
import type {
  CachedInventoryRow,
  CapabilityFilter,
  DiscoverRow,
  LocalInventoryRow,
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
} from "./types";

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

export function ModelsPage() {
  const navigate = useNavigate();
  const search = useSearch({ from: "/models" });
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

  const tab: ModelsTab = search.tab ?? "discover";
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<HfSortKey>("trendingScore");
  const [direction, setDirection] = useState<HfSortDirection>("desc");
  const [resourceType, setResourceType] =
    useState<ResourceTypeFilter>("models");
  const [activeChannelId, setActiveChannelId] = useState<ChannelId | null>(null);
  const activeChannel: ChannelPreset | null = useMemo(
    () => findChannel(activeChannelId),
    [activeChannelId],
  );
  const [discoverFormat, setDiscoverFormat] =
    useState<ModelFormatFilter>("gguf");
  const [downloadedFormat, setDownloadedFormat] =
    useState<ModelFormatFilter>("all");
  // When a channel is active, the channel's recipe drives the format on the
  // discover tab. Manual format changes from the toolbar deactivate the
  // channel so the user has direct control again.
  const formatFilter =
    tab === "discover"
      ? (activeChannel?.format ?? discoverFormat)
      : downloadedFormat;
  const setFormatFilter = useCallback(
    (next: ModelFormatFilter) => {
      if (tab === "discover") {
        setActiveChannelId(null);
        setDiscoverFormat(next);
      } else {
        setDownloadedFormat(next);
      }
    },
    [tab],
  );
  const [capabilityFilter, setCapabilityFilter] =
    useState<CapabilityFilter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(false);
  const handleSelect = useCallback((id: string) => {
    setSelectedId(id);
    setMobileInspectorOpen(true);
  }, []);
  const handleTabChange = useCallback(
    (next: ModelsTab) => {
      setMobileInspectorOpen(false);
      if (next === "downloaded") setActiveChannelId(null);
      void navigate({
        to: "/models",
        search: (prev) => ({ ...prev, tab: next }),
      });
    },
    [navigate],
  );

  const handleResourceTypeChange = useCallback(
    (next: ResourceTypeFilter) => {
      setResourceType((prev) => {
        if (prev !== next) {
          setDiscoverFormat("gguf");
          setDownloadedFormat("all");
          setCapabilityFilter("all");
          setActiveChannelId(null);
        }
        return next;
      });
    },
    [],
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
  const hfToken = useHfTokenStore((s) => s.token);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const isDiscoverTab = tab === "discover";
  const effectiveSort: HfSortKey = activeChannel?.sort ?? sortBy;
  const channelOption = useMemo(
    () =>
      activeChannel
        ? {
            owner: activeChannel.owner,
            tags: activeChannel.tags,
            query: activeChannel.query,
            idSuffix: activeChannel.idSuffix,
          }
        : null,
    [activeChannel],
  );
  const modelSearch = useHfModelSearch(debouncedQuery, {
    accessToken: debouncedHfToken || undefined,
    sortBy: effectiveSort,
    sortDirection: direction,
    pinUnslothFirst:
      effectiveSort === "trendingScore" &&
      direction === "desc" &&
      !activeChannel,
    enabled: isDiscoverTab && resourceType !== "datasets",
    keepUnsupportedTags: true,
    channel: channelOption,
  });
  const datasetSearch = useHfDatasetSearch(debouncedQuery, {
    accessToken: debouncedHfToken || undefined,
    enabled: isDiscoverTab && resourceType === "datasets",
    sortBy: effectiveSort,
    sortDirection: direction,
  });
  const isDatasetMode = resourceType === "datasets";
  const results = isDatasetMode ? [] : modelSearch.results;
  const isLoading = isDatasetMode ? datasetSearch.isLoading : modelSearch.isLoading;
  const isLoadingMore = isDatasetMode
    ? datasetSearch.isLoadingMore
    : modelSearch.isLoadingMore;
  const fetchMore = isDatasetMode ? datasetSearch.fetchMore : modelSearch.fetchMore;
  const searchError = isDatasetMode ? datasetSearch.error : modelSearch.error;
  const retrySearch = isDatasetMode ? datasetSearch.retry : modelSearch.retry;

  const handleRetrySearch = useCallback(() => {
    if (!online) {
      toast.error("You're offline", {
        description: "Reconnect to the internet to browse Hugging Face.",
      });
      return;
    }
    retrySearch();
    toast.message("Retrying…", {
      description: "Reaching Hugging Face for the latest models.",
    });
  }, [online, retrySearch]);

  const lastErrorRef = useRef<string | null>(null);
  useEffect(() => {
    if (!searchError) {
      lastErrorRef.current = null;
      return;
    }
    if (lastErrorRef.current === searchError) return;
    lastErrorRef.current = searchError;
    toast.error(
      online
        ? "Couldn't reach Hugging Face"
        : "You're offline",
      {
        description: online
          ? searchError
          : "Reconnect to the internet to browse models.",
        action: { label: "Retry", onClick: handleRetrySearch },
      },
    );
  }, [searchError, online, handleRetrySearch]);

  const wasOfflineRef = useRef(!online);
  useEffect(() => {
    if (online && wasOfflineRef.current) {
      toast.success("Back online", {
        description: "Refreshing the discovery feed.",
      });
      retrySearch();
    }
    wasOfflineRef.current = !online;
  }, [online, retrySearch]);

  const {
    cachedRows: effectiveCachedRows,
    localRows: effectiveLocalRows,
    availableSet,
    partialSet,
    downloadedReady,
    inventoryError,
    refreshInventory,
  } = useHubInventory(isDatasetMode);
  const [selectedRepoMetadata, setSelectedRepoMetadata] =
    useState<ReturnType<typeof toHfModelResult>>(null);

  const modelDiscoverRows = useMemo<DiscoverRow[]>(
    () => buildDiscoverRows(results, availableSet, partialSet),
    [results, availableSet, partialSet],
  );

  const datasetDiscoverRows = useMemo<DiscoverRow[]>(() => {
    if (!isDatasetMode) return [];
    return datasetSearch.results.map((ds) => {
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
  }, [isDatasetMode, datasetSearch.results, availableSet, partialSet]);

  const discoverRows = isDatasetMode ? datasetDiscoverRows : modelDiscoverRows;

  const filteredDiscoverRows = useMemo(
    () =>
      discoverRows.filter((row) => {
        if (isDatasetMode) return true;
        return (
          matchesFormat(row.result.isGguf, formatFilter) &&
          matchesCapability(row.capabilities, capabilityFilter)
        );
      }),
    [discoverRows, isDatasetMode, formatFilter, capabilityFilter],
  );

  const inventoryTokens = useMemo(
    () => (isDiscoverTab ? [] : tokenizeQuery(query)),
    [isDiscoverTab, query],
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
          (row) => isDatasetMode || matchesFormat(row.isGguf, formatFilter),
        ),
        inventoryTokens,
      ),
    [effectiveCachedRows, isDatasetMode, formatFilter, inventoryTokens],
  );

  const filteredLocalRows = useMemo(
    () =>
      partitionByMatch(
        effectiveLocalRows.filter(
          (row) => isDatasetMode || matchesFormat(row.isGguf, formatFilter),
        ),
        inventoryTokens,
      ),
    [effectiveLocalRows, isDatasetMode, formatFilter, inventoryTokens],
  );

  const { filterPaused, handleKeepSearching } = useFilterStarvedPause({
    isDiscoverTab,
    scannedCount: discoverRows.length,
    filteredCount: filteredDiscoverRows.length,
    resetDeps: [
      debouncedQuery,
      resourceType,
      formatFilter,
      capabilityFilter,
      effectiveSort,
      direction,
      activeChannelId,
    ],
  });

  const handleClearFilters = useCallback(() => {
    setFormatFilter("all");
    setCapabilityFilter("all");
  }, [setFormatFilter]);

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    // Drive re-evaluation off the raw fetched count, not the filtered one —
    // the page-level format/capability filters can reject every incoming
    // row, leaving filteredDiscoverRows.length stalled while results keep
    // growing. Using the raw count guarantees the auto-fire effect re-runs
    // after each fetch lands so we don't dead-end on aggressive filters.
    discoverRows.length,
    isDiscoverTab && !filterPaused,
  );

  useEffect(() => {
    const nextIds =
      tab === "discover"
        ? filteredDiscoverRows.map((row) => row.id)
        : [
            ...filteredCachedRows.map((row) => row.id),
            ...filteredLocalRows.map((row) => row.id),
          ];

    if (selectedId && nextIds.includes(selectedId)) return;
    setSelectedId(nextIds[0] ?? null);
  }, [
    tab,
    filteredCachedRows,
    filteredDiscoverRows,
    filteredLocalRows,
    selectedId,
  ]);

  const selectedDiscoverRow = useMemo(
    () =>
      selectedId
        ? (discoverRows.find((row) => row.id === selectedId) ?? null)
        : null,
    [discoverRows, selectedId],
  );

  const selectedCachedRow = useMemo(
    () =>
      selectedId
        ? (effectiveCachedRows.find((row) => row.id === selectedId) ?? null)
        : null,
    [effectiveCachedRows, selectedId],
  );

  const selectedLocalRow = useMemo(
    () =>
      selectedId
        ? (effectiveLocalRows.find((row) => row.id === selectedId) ?? null)
        : null,
    [effectiveLocalRows, selectedId],
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
      ...(debouncedHfToken ? { accessToken: debouncedHfToken } : {}),
    })
      .then((result) => {
        if (cancelled) return;
        setSelectedRepoMetadata(toHfModelResult(result));
      })
      .catch(() => {
        if (cancelled) return;
        setSelectedRepoMetadata(null);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedHubRepoId, selectedResultFromFeed, debouncedHfToken]);

  const selectedHfResult = selectedResultFromFeed ?? selectedRepoMetadata;

  const selectedModel = useSelectedModelView({
    selectedDiscoverRow,
    selectedCachedRow,
    selectedLocalRow,
    selectedHfResult,
    isDatasetMode,
  });

  const isActive = useMemo(
    () =>
      Boolean(
        selectedModel &&
          activeCheckpoint &&
          activeCheckpoint.toLowerCase() === selectedModel.id.toLowerCase(),
      ),
    [activeCheckpoint, selectedModel],
  );

  const isLoadingThisModel = useMemo(
    () =>
      Boolean(
        selectedModel &&
          loadingModel &&
          loadingModel.id.toLowerCase() === selectedModel.id.toLowerCase(),
      ),
    [loadingModel, selectedModel],
  );

  const vramInfo = useMemo(() => {
    if (!selectedModel || selectedModel.isGguf || !selectedModel.totalParams) {
      return null;
    }
    const est = estimateLoadingVram(selectedModel.totalParams, "qlora");
    if (!gpu.available) {
      return { est, status: "fits" as const };
    }
    const status = checkVramFit(est, gpu.memoryTotalGb);
    return status ? { est, status } : null;
  }, [gpu, selectedModel]);

  const minMemory = useMemo(() => {
    if (!selectedModel) return null;
    if (selectedModel.isGguf) {
      if (selectedModel.cachedBytes)
        return formatBytes(selectedModel.cachedBytes);
      if (selectedModel.estimatedSizeBytes)
        return formatBytes(selectedModel.estimatedSizeBytes);
      return null;
    }
    if (selectedModel.estimatedSizeBytes)
      return formatBytes(selectedModel.estimatedSizeBytes);
    if (vramInfo) return `~${vramInfo.est} GB`;
    if (selectedModel.cachedBytes)
      return formatBytes(selectedModel.cachedBytes);
    return null;
  }, [selectedModel, vramInfo]);

  const gpuLabel = gpu.available
    ? `${Math.floor(gpu.memoryTotalGb)} GB`
    : "Unavailable";
  const ramLabel = gpu.available
    ? `${Math.floor(gpu.systemRamAvailableGb)} GB`
    : "Unavailable";

  async function handleLoad(opts: {
    ggufVariant?: string;
    expectedBytes?: number;
  }) {
    if (!selectedModel) return;
    await selectModel({
      id: selectedModel.id,
      ggufVariant: opts.ggufVariant,
      isLora: false,
      isDownloaded: true,
      expectedBytes: opts.expectedBytes,
    });
    refreshInventory();
  }

  async function handleLoadLocal() {
    if (!selectedModel) return;
    await selectModel({
      id: selectedModel.id,
      isLora: false,
      isDownloaded: true,
    });
  }

  function handleUseInChat() {
    void navigate({ to: "/chat" });
  }

  function handleTrain() {
    if (!selectedModel) return;
    const repoId = selectedModel.hubRepoId ?? selectedModel.id;
    const store = useTrainingConfigStore.getState();
    if (isDatasetMode) {
      if (selectedModel.kind === "local" && selectedModel.path) {
        store.selectLocalDataset(selectedModel.path);
      } else {
        store.selectHfDataset(repoId);
      }
    } else {
      const caps = selectedModel.capabilities.map((c) => c.key);
      const inferredType: ModelType = caps.includes("vision")
        ? "vision"
        : caps.includes("audio")
          ? "audio"
          : caps.includes("embedding")
            ? "embeddings"
            : "text";
      if (store.modelType !== inferredType) {
        store.setModelType(inferredType);
      }
      store.setSelectedModel(repoId);
    }
    void navigate({ to: "/studio" });
  }

  return (
    <div className="hub-page flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[1180px] flex-1 min-h-0 flex-col gap-6 px-5 pt-8 pb-16 sm:px-9 sm:pt-10 sm:pb-24">
        <ModelsHeader
          cachedCount={effectiveCachedRows.length}
          localCount={effectiveLocalRows.length}
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
              onQueryChange={(next) => {
                if (activeChannelId) setActiveChannelId(null);
                setQuery(next);
              }}
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
                tab={tab}
                discoverRows={filteredDiscoverRows}
                cachedRows={filteredCachedRows}
                localRows={filteredLocalRows}
                selectedId={selectedId}
                onSelect={handleSelect}
                isLoading={isLoading}
                isLoadingMore={isLoadingMore}
                downloadedReady={downloadedReady}
                inventoryError={inventoryError}
                query={query}
                scrollRef={scrollRef}
                sentinelRef={sentinelRef}
                activeCheckpoint={activeCheckpoint}
                searchError={searchError}
                online={online}
                isDataset={isDatasetMode}
                inventoryTokens={inventoryTokens}
                filterPaused={filterPaused}
                scannedCount={discoverRows.length}
                hasActiveFilters={
                  formatFilter !== "all" || capabilityFilter !== "all"
                }
                onKeepSearching={handleKeepSearching}
                onClearFilters={handleClearFilters}
                onRetry={handleRetrySearch}
                onInventoryChange={refreshInventory}
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
                isActive={isActive}
                activeGgufVariant={activeGgufVariant}
                isLoadingThisModel={isLoadingThisModel}
                loadingPhase={loadProgress?.phase}
                minMemory={minMemory}
                vramInfo={vramInfo}
                gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                systemRamGb={
                  gpu.available ? gpu.systemRamAvailableGb : undefined
                }
                onLoad={handleLoad}
                onLoadLocal={() => void handleLoadLocal()}
                onUseInChat={handleUseInChat}
                onTrain={handleTrain}
                onInventoryChange={refreshInventory}
              />
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
