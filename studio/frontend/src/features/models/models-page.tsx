// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type CachedGgufRepo,
  type CachedModelRepo,
  listCachedGguf,
  listCachedModels,
  listLocalModels,
} from "@/features/chat/api/chat-api";
import { listLocalDatasets } from "@/features/training/api/datasets-api";
import type { LocalDatasetInfo } from "@/features/training/types/datasets";
import { useChatModelRuntime, useChatRuntimeStore } from "@/features/chat";
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
import { checkVramFit, estimateLoadingVram } from "@/lib/vram";
import { detectCapabilities, detectLicense } from "./lib/capabilities";
import { formatBytes } from "./lib/format";
import {
  buildDiscoverRows,
  buildLocalInventoryRows,
  buildSummary,
  localSourceLabel,
  matchesCapability,
  matchesFormat,
  toHfModelResult,
} from "./lib/view-models";
import { ModelInspector } from "./components/model-inspector";
import { ModelsCatalog } from "./components/models-catalog";
import { ModelsHeader } from "./components/models-header";
import { ModelsToolbar } from "./components/models-toolbar";
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

  const [tab, setTab] = useState<ModelsTab>("discover");
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<HfSortKey>("trendingScore");
  const [direction, setDirection] = useState<HfSortDirection>("desc");
  const [resourceType, setResourceType] =
    useState<ResourceTypeFilter>("models");
  const [formatFilter, setFormatFilter] = useState<ModelFormatFilter>("all");
  const [capabilityFilter, setCapabilityFilter] =
    useState<CapabilityFilter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const debouncedQuery = useDebouncedValue(query);
  const modelSearch = useHfModelSearch(debouncedQuery, {
    sortBy,
    sortDirection: direction,
    pinUnslothFirst: sortBy === "trendingScore" && direction === "desc",
  });
  const datasetSearch = useHfDatasetSearch(debouncedQuery, {
    enabled: resourceType === "datasets",
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

  const [cachedGguf, setCachedGguf] = useState<CachedGgufRepo[]>([]);
  const [cachedModels, setCachedModels] = useState<CachedModelRepo[]>([]);
  const [localRows, setLocalRows] = useState<LocalInventoryRow[]>([]);
  const [localDatasets, setLocalDatasets] = useState<LocalDatasetInfo[]>([]);
  const [downloadedReady, setDownloadedReady] = useState(false);
  const [selectedRepoMetadata, setSelectedRepoMetadata] =
    useState<ReturnType<typeof toHfModelResult>>(null);

  const refreshInventory = useCallback(() => {
    void listCachedGguf()
      .then(setCachedGguf)
      .catch(() => {});
    void listCachedModels()
      .then(setCachedModels)
      .catch(() => {});
    void listLocalModels()
      .then((response) => {
        setLocalRows(buildLocalInventoryRows(response.models));
      })
      .catch(() => {});
    void listLocalDatasets()
      .then((response) => {
        setLocalDatasets(response.datasets);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    let mounted = true;
    let pending = 4;
    const done = () => {
      pending -= 1;
      if (pending === 0 && mounted) {
        setDownloadedReady(true);
      }
    };

    void listCachedGguf()
      .then((rows) => {
        if (mounted) setCachedGguf(rows);
      })
      .catch(() => {})
      .finally(done);

    void listCachedModels()
      .then((rows) => {
        if (mounted) setCachedModels(rows);
      })
      .catch(() => {})
      .finally(done);

    void listLocalModels()
      .then((response) => {
        if (mounted) setLocalRows(buildLocalInventoryRows(response.models));
      })
      .catch(() => {})
      .finally(done);

    void listLocalDatasets()
      .then((response) => {
        if (mounted) setLocalDatasets(response.datasets);
      })
      .catch(() => {})
      .finally(done);

    return () => {
      mounted = false;
    };
  }, []);

  const cachedRows = useMemo<CachedInventoryRow[]>(() => {
    const rows: CachedInventoryRow[] = [
      ...cachedGguf.map((row) => ({
        kind: "cache" as const,
        id: row.repo_id,
        repoId: row.repo_id,
        owner: row.repo_id.includes("/") ? row.repo_id.split("/")[0] : "Hub",
        repo: row.repo_id.includes("/")
          ? row.repo_id.split("/").slice(1).join("/")
          : row.repo_id,
        isGguf: true,
        bytes: row.size_bytes,
      })),
      ...cachedModels.map((row) => ({
        kind: "cache" as const,
        id: row.repo_id,
        repoId: row.repo_id,
        owner: row.repo_id.includes("/") ? row.repo_id.split("/")[0] : "Hub",
        repo: row.repo_id.includes("/")
          ? row.repo_id.split("/").slice(1).join("/")
          : row.repo_id,
        isGguf: false,
        bytes: row.size_bytes,
      })),
    ];

    return rows.sort((a, b) => a.repoId.localeCompare(b.repoId));
  }, [cachedGguf, cachedModels]);

  const localDatasetRows = useMemo<LocalInventoryRow[]>(() => {
    return localDatasets
      .map((ds) => {
        const repoId = ds.id.includes("/") ? ds.id : null;
        const owner = repoId ? ds.id.split("/")[0] : "Local";
        return {
          kind: "local" as const,
          id: ds.id,
          repoId,
          owner,
          title: ds.label || ds.id,
          source: "custom" as const,
          sourceLabel: "Local dataset",
          path: ds.path,
          isGguf: false,
          updatedAt: ds.updated_at ?? null,
        };
      })
      .sort((a, b) => a.title.localeCompare(b.title));
  }, [localDatasets]);

  const effectiveCachedRows = isDatasetMode ? [] : cachedRows;
  const effectiveLocalRows = isDatasetMode ? localDatasetRows : localRows;

  const availableSet = useMemo(() => {
    const set = new Set<string>();
    for (const row of effectiveCachedRows) set.add(row.repoId.toLowerCase());
    for (const row of effectiveLocalRows) {
      if (row.repoId) set.add(row.repoId.toLowerCase());
    }
    return set;
  }, [effectiveCachedRows, effectiveLocalRows]);

  const modelDiscoverRows = useMemo<DiscoverRow[]>(
    () => buildDiscoverRows(results, availableSet),
    [results, availableSet],
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
        isAvailableOnDevice: false,
        summary: summaryParts.join(" · ") || "Dataset",
        capabilities: [],
      };
    });
  }, [isDatasetMode, datasetSearch.results]);

  const discoverRows = isDatasetMode ? datasetDiscoverRows : modelDiscoverRows;

  const filteredDiscoverRows = useMemo(
    () =>
      discoverRows.filter(
        (row) =>
          matchesFormat(row.result.isGguf, formatFilter) &&
          matchesCapability(row.capabilities, capabilityFilter),
      ),
    [discoverRows, formatFilter, capabilityFilter],
  );

  const inventoryNeedle = query.trim().toLowerCase();
  const filteredCachedRows = useMemo(
    () =>
      effectiveCachedRows.filter((row) => {
        if (!isDatasetMode && !matchesFormat(row.isGguf, formatFilter))
          return false;
        if (!inventoryNeedle) return true;
        return row.repoId.toLowerCase().includes(inventoryNeedle);
      }),
    [effectiveCachedRows, isDatasetMode, inventoryNeedle, formatFilter],
  );

  const filteredLocalRows = useMemo(
    () =>
      effectiveLocalRows.filter((row) => {
        if (!isDatasetMode && !matchesFormat(row.isGguf, formatFilter))
          return false;
        if (!inventoryNeedle) return true;
        const haystack = [
          row.title,
          row.owner,
          row.sourceLabel,
          row.path,
          row.repoId ?? "",
        ]
          .join(" ")
          .toLowerCase();
        return haystack.includes(inventoryNeedle);
      }),
    [effectiveLocalRows, isDatasetMode, inventoryNeedle, formatFilter],
  );

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    filteredDiscoverRows.length,
    tab === "discover",
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
    void cachedModelInfo({ name: selectedHubRepoId })
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
  }, [selectedHubRepoId, selectedResultFromFeed]);

  const selectedHfResult = selectedResultFromFeed ?? selectedRepoMetadata;

  const selectedModel = useMemo<SelectedModelView | null>(() => {
    if (selectedDiscoverRow) {
      return {
        id: selectedDiscoverRow.id,
        kind: "discover",
        displayId: selectedDiscoverRow.id,
        hubRepoId: selectedDiscoverRow.result.id,
        owner: selectedDiscoverRow.owner,
        title: selectedDiscoverRow.repo,
        summary: selectedDiscoverRow.summary,
        sourceLabel: selectedDiscoverRow.isAvailableOnDevice
          ? "On device"
          : "Hugging Face",
        path: null,
        isLocal: false,
        isGguf: selectedDiscoverRow.result.isGguf,
        isDownloaded: selectedDiscoverRow.isAvailableOnDevice,
        capabilities: selectedDiscoverRow.capabilities,
        license: detectLicense(selectedDiscoverRow.result.tags),
        pipelineTag: selectedDiscoverRow.result.pipelineTag,
        libraryName: selectedDiscoverRow.result.libraryName,
        downloads: selectedDiscoverRow.result.downloads,
        likes: selectedDiscoverRow.result.likes,
        totalParams: selectedDiscoverRow.result.totalParams,
        estimatedSizeBytes: selectedDiscoverRow.result.estimatedSizeBytes,
        updatedAt: selectedDiscoverRow.result.updatedAt,
      };
    }

    if (selectedCachedRow) {
      return {
        id: selectedCachedRow.repoId,
        kind: "cache",
        displayId: selectedCachedRow.repoId,
        hubRepoId: selectedCachedRow.repoId,
        owner: selectedCachedRow.owner,
        title: selectedCachedRow.repo,
        summary: selectedHfResult
          ? buildSummary(selectedHfResult)
          : selectedCachedRow.isGguf
            ? "Cached GGUF repository ready for local inference."
            : "Cached checkpoint repository ready for local inference.",
        sourceLabel: "Hub cache",
        path: null,
        isLocal: false,
        isGguf: selectedCachedRow.isGguf,
        isDownloaded: true,
        capabilities: detectCapabilities(
          selectedHfResult?.tags,
          selectedHfResult?.pipelineTag,
          selectedCachedRow.repoId,
        ),
        license: detectLicense(selectedHfResult?.tags),
        pipelineTag: selectedHfResult?.pipelineTag,
        libraryName: selectedHfResult?.libraryName,
        downloads: selectedHfResult?.downloads,
        likes: selectedHfResult?.likes,
        totalParams: selectedHfResult?.totalParams,
        estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
        cachedBytes: selectedCachedRow.bytes,
        updatedAt: selectedHfResult?.updatedAt,
      };
    }

    if (selectedLocalRow) {
      const localDisplayId = selectedLocalRow.repoId ?? selectedLocalRow.id;
      return {
        id: selectedLocalRow.id,
        kind: "local",
        displayId: localDisplayId,
        hubRepoId: selectedLocalRow.repoId,
        owner: selectedLocalRow.owner,
        title: selectedLocalRow.title,
        summary: selectedHfResult
          ? buildSummary(selectedHfResult)
          : `${localSourceLabel(selectedLocalRow.source)} · ${
              selectedLocalRow.isGguf ? "local GGUF" : "local checkpoint"
            }`,
        sourceLabel: selectedLocalRow.sourceLabel,
        path: selectedLocalRow.path,
        isLocal: true,
        isGguf: selectedLocalRow.isGguf,
        isDownloaded: true,
        capabilities: detectCapabilities(
          selectedHfResult?.tags,
          selectedHfResult?.pipelineTag,
          selectedLocalRow.repoId ?? selectedLocalRow.title,
        ),
        license: detectLicense(selectedHfResult?.tags),
        pipelineTag: selectedHfResult?.pipelineTag,
        libraryName: selectedHfResult?.libraryName,
        downloads: selectedHfResult?.downloads,
        likes: selectedHfResult?.likes,
        totalParams: selectedHfResult?.totalParams,
        estimatedSizeBytes: selectedHfResult?.estimatedSizeBytes,
        updatedAt: selectedHfResult?.updatedAt,
        localUpdatedAt: selectedLocalRow.updatedAt,
      };
    }

    return null;
  }, [
    selectedCachedRow,
    selectedDiscoverRow,
    selectedHfResult,
    selectedLocalRow,
  ]);

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

  const gpuLabel = gpu.available ? `${gpu.memoryTotalGb} GB` : "Unavailable";
  const ramLabel = gpu.available
    ? `${gpu.systemRamAvailableGb.toFixed(0)} GB`
    : "Unavailable";
  const downloadedCount = effectiveCachedRows.length + effectiveLocalRows.length;

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

  return (
    <div className="min-h-full bg-background">
      <div className="flex flex-col gap-3 px-4 py-4 sm:px-5 xl:px-6">
        <ModelsHeader
          cachedCount={effectiveCachedRows.length}
          localCount={effectiveLocalRows.length}
          gpuLabel={gpuLabel}
          ramLabel={ramLabel}
          activeCheckpoint={activeCheckpoint}
          activeGgufVariant={activeGgufVariant}
          onEject={() => void ejectModel()}
        />

        <section className="flex flex-col overflow-hidden rounded-[24px] border border-border/70 bg-[#f7f7f9] dark:bg-card/60">
          <div className="border-b border-border/70 p-3">
            <ModelsToolbar
              tab={tab}
              onTabChange={setTab}
              query={query}
              onQueryChange={setQuery}
              isLoading={isLoading}
              sortBy={sortBy}
              direction={direction}
              onSortChange={(next) => {
                setSortBy(next);
                if (next === "trendingScore") setDirection("desc");
              }}
              onDirectionToggle={() =>
                setDirection((value) => (value === "desc" ? "asc" : "desc"))
              }
              resourceType={resourceType}
              onResourceTypeChange={setResourceType}
              formatFilter={formatFilter}
              onFormatFilterChange={setFormatFilter}
              capabilityFilter={capabilityFilter}
              onCapabilityFilterChange={setCapabilityFilter}
              discoverCount={filteredDiscoverRows.length}
              downloadedCount={downloadedCount}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] xl:grid-cols-[400px_minmax(0,1fr)] 2xl:grid-cols-[440px_minmax(0,1fr)]">
            <div className="min-w-0 border-b border-border/70 bg-foreground/[0.015] dark:bg-transparent lg:border-b-0 lg:border-r">
              <ModelsCatalog
                tab={tab}
                discoverRows={filteredDiscoverRows}
                cachedRows={filteredCachedRows}
                localRows={filteredLocalRows}
                selectedId={selectedId}
                onSelect={setSelectedId}
                isLoading={isLoading}
                isLoadingMore={isLoadingMore}
                downloadedReady={downloadedReady}
                query={query}
                scrollRef={scrollRef}
                sentinelRef={sentinelRef}
                activeCheckpoint={activeCheckpoint}
                searchError={searchError}
                online={online}
                onRetry={handleRetrySearch}
              />
            </div>

            <div className="bg-background dark:bg-background/50">
              <ModelInspector
                model={selectedModel}
                isDataset={isDatasetMode}
                isActive={isActive}
                activeGgufVariant={activeGgufVariant}
                isLoadingThisModel={isLoadingThisModel}
                loadingPhase={loadProgress?.phase}
                minMemory={minMemory}
                vramInfo={vramInfo}
                gpuLabel={
                  gpu.available
                    ? `${gpu.memoryTotalGb} GB available`
                    : "GPU data unavailable"
                }
                gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                systemRamGb={
                  gpu.available ? gpu.systemRamAvailableGb : undefined
                }
                onLoad={handleLoad}
                onLoadLocal={() => void handleLoadLocal()}
                onUseInChat={handleUseInChat}
                onInventoryChange={refreshInventory}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
