// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  getInferenceStatus,
  isExternalModelId,
  useChatModelRuntime,
  useChatRuntimeStore,
} from "@/features/chat";
import { useHubInventory } from "@/features/hub";
import type {
  HfModelSearchChannel,
  HfSortDirection,
  HfSortKey,
} from "@/features/hub";
import { useOnlineStatus } from "@/features/hub";
import { useHubInfiniteScroll } from "@/features/hub";
import { ggufVariantsMatch, modelIdsMatch } from "@/features/hub";
import { hfApiToken, useHfTokenStore } from "@/features/hub";
import {
  applyModelLoadConfigToRuntime,
  currentRuntimePerModelConfig,
  hfModelFitsDevice,
  resolveInitialConfig,
} from "@/features/model-picker";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useGpuInfo } from "@/hooks/use-gpu-info";
import { cn } from "@/lib/utils";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ExternalLinkConfirmDialog } from "./catalog/external-link-confirm-dialog";
import { HubDetailView } from "./catalog/hub-detail-view";
import { HubFeed } from "./catalog/hub-feed";
import { HubTopBar } from "./catalog/hub-top-bar";
import {
  ModelsCatalog,
  type ModelsCatalogHandlers,
  type ModelsCatalogPagination,
  type ModelsCatalogState,
} from "./catalog/models-catalog";
import { ModelsHeader } from "./catalog/models-header";
import {
  type AllModelsView,
  HubListHeader,
  type InventorySort,
  InventorySortControl,
  InventoryTypeFilterControl,
  ResultListHeader,
} from "./catalog/models-table";
import {
  type ModelTypeFilter,
  matchesModelType,
} from "./lib/model-type-filter";
import { ModelsToolbar } from "./catalog/models-toolbar";
import { OnDeviceFoldersDialog } from "./catalog/on-device-folders-dialog";
import { OwnerScopeToggle } from "./catalog/owner-scope-toggle";
import { useDiscoverSearch } from "./hooks/use-discover-search";
import { useFeedWriteBack } from "./hooks/use-feed-write-back";
import { useHubFeed } from "./hooks/use-hub-feed";
import { useHubModelVram } from "./hooks/use-hub-model-vram";
import { useHiddenEmbeddingModelIds } from "./hooks/use-hidden-embedding-models";
import { useModelsSelection } from "./hooks/use-models-selection";
import {
  CHANNEL_TO_SECTION,
  type ChannelId,
  type ChannelPreset,
  HUB_SECTION_TITLE,
  type HubSection,
  SECTION_TO_CHANNEL,
  findChannel,
} from "./lib/channels";
import {
  isConfiguredHiddenModelId,
  isHiddenModelId,
} from "./lib/hidden-models";
import { inventoryRowMatches, tokenizeQuery } from "./lib/inventory-search";
import { resolveOwnerProviderLogo } from "./lib/provider-logos";
import {
  buildDiscoverRows,
  detectResultFormat,
  isUnslothFinetunable,
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
const ALL_MODELS_VIEW_STORAGE_KEY = "unsloth.hub.allModelsView";
const INVENTORY_SORT_STORAGE_KEY = "unsloth.hub.inventorySort";
const OWNER_SCOPE_STORAGE_KEY = "unsloth.hub.ownerScope";

/** Discover browsing scope: the whole Hub (default) or only the unsloth org. */
export type OwnerScope = "unsloth" | "all";

function readOwnerScopePreference(): OwnerScope {
  if (typeof window === "undefined") {
    return "all";
  }
  try {
    const value = window.localStorage.getItem(OWNER_SCOPE_STORAGE_KEY);
    // Default to the whole Hub; only honor an explicit "unsloth" preference.
    return value === "unsloth" ? "unsloth" : "all";
  } catch {
    return "all";
  }
}

function writeOwnerScopePreference(scope: OwnerScope): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(OWNER_SCOPE_STORAGE_KEY, scope);
  } catch {
    return;
  }
}

const DEFAULT_DISCOVER_CHANNEL: ChannelId = "unsloth-trending";
const FEED_LIST_CHANNEL_ID: ChannelId = "unsloth-latest";

type DiscoverMode = "feed" | "channel-list" | "search";

type ModelLoadOptions = { ggufVariant?: string; expectedBytes?: number };

// Focused list heading stays "Models"/"Datasets" regardless of filters; only
// search changes the label.
function buildFocusedHeading({
  query,
  channel,
  isDataset,
}: {
  query: string;
  channel: ChannelPreset | null;
  isDataset: boolean;
}): string {
  const trimmed = query.trim();
  if (trimmed) return `Results for "${trimmed}"`;
  if (channel && channel.id !== DEFAULT_DISCOVER_CHANNEL) return channel.label;
  return isDataset ? "Datasets" : "Models";
}

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

// "All models" defaults to list layout; list-vs-grid choice persists across sessions.
function readAllModelsViewPreference(): AllModelsView {
  if (typeof window === "undefined") {
    return "split";
  }
  try {
    const value = window.localStorage.getItem(ALL_MODELS_VIEW_STORAGE_KEY);
    return value === "grid" || value === "two" || value === "split"
      ? value
      : "split";
  } catch {
    return "split";
  }
}

function readInventorySortPreference(): InventorySort {
  if (typeof window === "undefined") {
    return "recent";
  }
  try {
    const value = window.localStorage.getItem(INVENTORY_SORT_STORAGE_KEY);
    return value === "name" || value === "size" || value === "recent"
      ? value
      : "recent";
  } catch {
    return "recent";
  }
}

function writeInventorySortPreference(sort: InventorySort): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(INVENTORY_SORT_STORAGE_KEY, sort);
  } catch {
    return;
  }
}

function writeAllModelsViewPreference(view: AllModelsView): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(ALL_MODELS_VIEW_STORAGE_KEY, view);
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
  const hasModelDeepLink =
    typeof search.model === "string" && search.model.length > 0;
  const [fallbackTab, setFallbackTab] = useState<ModelsTab>(
    () => readModelsTabPreference() ?? "discover",
  );
  const tab = urlTab ?? (hasModelDeepLink ? "discover" : fallbackTab);

  useEffect(() => {
    if (urlTab !== null) return;
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, tab }),
      replace: true,
    });
  }, [urlTab, tab, navigate]);

  const setTab = useCallback(
    (next: ModelsTab) => {
      setFallbackTab(next);
      writeModelsTabPreference(next);
      void navigate({
        to: "/hub",
        search: (prev) => ({
          ...prev,
          tab: next,
          section: undefined,
          model: undefined,
        }),
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
  const deviceType = usePlatformStore((s) => s.deviceType);
  const hubSearch = useSearch({ from: "/hub" });
  const urlModel = hubSearch.model ?? null;

  const { selectModel, loadingModel, loadProgress, ejectModel } =
    useChatModelRuntime();
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const activeCheckpoint =
    checkpoint && !isExternalModelId(checkpoint) ? checkpoint : null;
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  // Shared with the chat model selector: list only models sized for this device.
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);

  useEffect(() => {
    let cancelled = false;
    void getInferenceStatus()
      .then((status) => {
        if (cancelled || !status.active_model) return;
        const store = useChatRuntimeStore.getState();
        if (
          !isExternalModelId(store.params.checkpoint) &&
          (!modelIdsMatch(store.params.checkpoint, status.active_model) ||
            !ggufVariantsMatch(
              store.activeGgufVariant,
              status.gguf_variant ?? null,
            ))
        ) {
          store.setCheckpoint(status.active_model, status.gguf_variant ?? null);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const { tab, setTab: setModelsTab } = useModelsTabState();
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<HfSortKey>(
    () => findChannel(DEFAULT_DISCOVER_CHANNEL)?.sort ?? "trendingScore",
  );
  const [direction, setDirection] = useState<HfSortDirection>("desc");
  const [ownerScope, setOwnerScopeState] = useState<OwnerScope>(
    readOwnerScopePreference,
  );
  const setOwnerScope = useCallback((scope: OwnerScope) => {
    setOwnerScopeState(scope);
    writeOwnerScopePreference(scope);
  }, []);
  const urlResourceType: ResourceTypeFilter =
    hubSearch.kind === "datasets" ? "datasets" : "models";
  const [resourceType, setResourceType] =
    useState<ResourceTypeFilter>(urlResourceType);
  // Resync on URL kind change (back/forward); it is only seeded on mount.
  useEffect(() => {
    setResourceType((current) =>
      current === urlResourceType ? current : urlResourceType,
    );
  }, [urlResourceType]);
  const [discoverFormat, setDiscoverFormat] = useState<ModelFormatFilter>(
    () => findChannel(DEFAULT_DISCOVER_CHANNEL)?.format ?? "gguf",
  );
  const [downloadedFormat, setDownloadedFormat] =
    useState<ModelFormatFilter>("all");
  const isDiscoverTab = tab === "discover";
  const isDatasetMode = resourceType === "datasets";
  const hiddenEmbeddingModelIds = useHiddenEmbeddingModelIds(!isDatasetMode);
  const urlSection = hubSearch.section ?? null;
  const isModelDiscover = isDiscoverTab && !isDatasetMode;
  const sectionChannelId: ChannelId | null = urlSection
    ? SECTION_TO_CHANNEL[urlSection]
    : null;
  const activeChannelId: ChannelId | null = isModelDiscover
    ? sectionChannelId
    : null;
  const activeChannel: ChannelPreset | null = useMemo(
    () => findChannel(activeChannelId),
    [activeChannelId],
  );
  const formatFilter = isDiscoverTab ? discoverFormat : downloadedFormat;
  const setFormatFilter = useCallback(
    (next: ModelFormatFilter) => {
      if (isDiscoverTab) {
        setDiscoverFormat(next);
        if (urlSection) {
          // Exit the section so its preset does not re-impose its own format.
          void navigate({
            to: "/hub",
            search: (prev) => ({ ...prev, section: undefined }),
          });
        }
      } else {
        setDownloadedFormat(next);
      }
    },
    [isDiscoverTab, urlSection, navigate],
  );
  const [capabilityFilter, setCapabilityFilter] =
    useState<CapabilityFilter>("all");
  const [allModelsView, setAllModelsViewState] = useState<AllModelsView>(
    readAllModelsViewPreference,
  );
  // Remembers the last non-split view so "Back to Hub" can drop out of split
  // mode into the prior browsing layout (defaults to the two-column grid).
  const lastNonSplitViewRef = useRef<AllModelsView>("two");
  const setAllModelsView = useCallback(
    (view: AllModelsView) => {
      setAllModelsViewState(view);
      writeAllModelsViewPreference(view);
      // Leaving split view drops the inline preview so the user lands on the
      // full hub list, not the model detail overlay.
      if (view !== "split") {
        void navigate({
          to: "/hub",
          search: (prev) => ({ ...prev, model: undefined }),
        });
      }
    },
    [navigate],
  );
  const [inventorySort, setInventorySortState] = useState<InventorySort>(
    readInventorySortPreference,
  );
  const setInventorySort = useCallback((sort: InventorySort) => {
    setInventorySortState(sort);
    writeInventorySortPreference(sort);
  }, []);
  const [inventoryTypeFilter, setInventoryTypeFilter] =
    useState<ModelTypeFilter>("all");
  const [foldersDialogOpen, setFoldersDialogOpen] = useState(false);
  const [discoverFetchIntent, setDiscoverFetchIntent] = useState(0);
  const [sortBrowseActive, setSortBrowseActive] = useState(false);

  const handleTabChange = useCallback(
    (next: ModelsTab) => {
      setSortBrowseActive(false);
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
      setSortBrowseActive(false);
      if (next === "models") {
        const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
        setDiscoverFormat(preset?.format ?? "gguf");
        setSortBy(preset?.sort ?? "trendingScore");
        setDirection("desc");
      }
      void navigate({
        to: "/hub",
        search: (prev) => ({
          ...prev,
          kind: next === "datasets" ? "datasets" : undefined,
          section: undefined,
          model: undefined,
        }),
        replace: true,
      });
    },
    [resourceType, navigate],
  );

  const handleOpenList = useCallback(
    (section: HubSection) => {
      if (urlSection === section) return;
      const preset = findChannel(SECTION_TO_CHANNEL[section]);
      if (preset) {
        setDiscoverFormat(preset.format);
        setSortBy(preset.sort);
        setDirection("desc");
      }
      setCapabilityFilter("all");
      setSortBrowseActive(false);
      // Clear search: an active query outranks the section in `mode`, hiding
      // the curated list behind global results otherwise.
      setQuery("");
      void navigate({
        to: "/hub",
        search: (prev) => ({ ...prev, section, model: undefined }),
      });
    },
    [urlSection, navigate],
  );
  const handleBackToFeed = useCallback(() => {
    const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
    setDiscoverFormat(preset?.format ?? "gguf");
    setSortBy(preset?.sort ?? "trendingScore");
    setDirection("desc");
    setCapabilityFilter("all");
    setSortBrowseActive(false);
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, section: undefined }),
    });
  }, [navigate]);
  const handleResetToDiscover = useCallback(() => {
    const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
    setModelsTab("discover");
    setResourceType("models");
    setQuery("");
    setCapabilityFilter("all");
    setDownloadedFormat("all");
    setDiscoverFormat(preset?.format ?? "gguf");
    setSortBy(preset?.sort ?? "trendingScore");
    setDirection("desc");
    setSortBrowseActive(false);
    void navigate({
      to: "/hub",
      // Assert the tab here too: fires alongside setModelsTab's navigation, and
      // spreading prev could otherwise restore the old tab.
      search: (prev) => ({
        ...prev,
        tab: "discover",
        section: undefined,
        model: undefined,
        kind: undefined,
      }),
    });
  }, [navigate, setModelsTab]);

  const handleSortChange = useCallback(
    (next: HfSortKey) => {
      setSortBy(next);
      if (next === "trendingScore") setDirection("desc");
      setSortBrowseActive(true);
      if (urlSection) {
        void navigate({
          to: "/hub",
          search: (prev) => ({ ...prev, section: undefined }),
          replace: true,
        });
      }
    },
    [urlSection, navigate],
  );

  const debouncedQuery = useDebouncedValue(query);
  const deferredDebouncedQuery = useDeferredValue(debouncedQuery);
  const hfToken = useHfTokenStore((s) => s.token);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const apiHfToken = hfApiToken(debouncedHfToken);
  const deferredFormatFilter = useDeferredValue(formatFilter);
  const deferredCapabilityFilter = useDeferredValue(capabilityFilter);

  const hasQuery = deferredDebouncedQuery.trim() !== "";
  const mode: DiscoverMode = isModelDiscover
    ? hasQuery
      ? "search"
      : urlSection != null
        ? "channel-list"
        : sortBrowseActive
          ? "search"
          : "feed"
    : "search";
  const isFeedMode = mode === "feed";
  const isChannelListMode = mode === "channel-list";
  const isSortBrowseMode =
    sortBrowseActive && isModelDiscover && !hasQuery && urlSection == null;

  const liveListChannel = useMemo<ChannelPreset | null>(() => {
    if (isChannelListMode) return activeChannel;
    if (isFeedMode) return findChannel(FEED_LIST_CHANNEL_ID);
    return null;
  }, [isChannelListMode, isFeedMode, activeChannel]);

  const effectiveSort: HfSortKey =
    isFeedMode && liveListChannel ? liveListChannel.sort : sortBy;
  const effectiveDirection: HfSortDirection = isFeedMode ? "desc" : direction;
  // The format dropdown always filters the visible list, including the feed's
  // "Latest" list, so the default (GGUF) hides fp8/safetensors and picking a
  // format actually changes the rows.
  const effectiveDiscoverFormat: ModelFormatFilter = deferredFormatFilter;

  const listChannel = useMemo<HfModelSearchChannel | null>(() => {
    if (!liveListChannel) return null;
    return {
      owner: liveListChannel.owner,
      tags: liveListChannel.tags,
      query: liveListChannel.query,
      idSuffix: liveListChannel.idSuffix,
    };
  }, [liveListChannel]);

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
    accessToken: apiHfToken,
    isDiscoverTab,
    isDatasetMode,
    sortBy: effectiveSort,
    direction: effectiveDirection,
    channel: listChannel,
    ownerScope,
    online,
  });

  useFeedWriteBack({
    channelId: isChannelListMode ? activeChannelId : null,
    results,
    isLoading,
    accessToken: apiHfToken,
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
    () => discoveryInventorySignature(effectiveCachedRows, effectiveLocalRows),
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
          private: ds.private,
          gated: ds.gated,
          updatedAt: ds.updatedAt,
          createdAt: ds.createdAt,
          downloadsAllTime: ds.downloadsAllTime,
          isGguf: false,
          tags: ds.plainTags,
        },
        isAvailableOnDevice: availableSet.has(lower),
        isPartialOnDevice: partialSet.has(lower),
        summary: summaryParts.join(" · ") || ds.prettyName || "Dataset",
        capabilities: [],
      };
    });
  }, [isDatasetMode, datasetResults, availableSet, partialSet]);

  const discoverRows = isDatasetMode ? datasetDiscoverRows : modelDiscoverRows;

  const filteredDiscoverRows = useMemo(() => {
    if (isDatasetMode) return discoverRows;
    return discoverRows.filter(
      (row) =>
        !isHiddenModelId(row.id) &&
        !isConfiguredHiddenModelId(hiddenEmbeddingModelIds, row.id) &&
        // The default feed only shows models with a provider logo.
        (!isFeedMode ||
          resolveOwnerProviderLogo(row.owner, row.repo) !== null) &&
        matchesFormat(detectResultFormat(row.result), effectiveDiscoverFormat) &&
        matchesCapability(row.capabilities, deferredCapabilityFilter) &&
        (!activeChannel?.finetunableOnly || isUnslothFinetunable(row.result)) &&
        // Models already on disk stay visible regardless of device fit,
        // matching the chat model selector.
        (!fitOnDeviceOnly ||
          row.isAvailableOnDevice ||
          hfModelFitsDevice(row.result, gpu)),
    );
  }, [
    discoverRows,
    hiddenEmbeddingModelIds,
    isDatasetMode,
    isFeedMode,
    effectiveDiscoverFormat,
    deferredCapabilityFilter,
    activeChannel,
    fitOnDeviceOnly,
    gpu,
  ]);

  const listRows = filteredDiscoverRows;

  const hubFeed = useHubFeed({
    accessToken: apiHfToken,
    online,
    enabled: isFeedMode,
    deviceType,
  });

  const feedTrendingRows = useMemo(
    () =>
      buildDiscoverRows(
        hubFeed.trending.results,
        effectiveCachedRows,
        effectiveLocalRows,
      )
        .filter(
          (row) =>
            !isHiddenModelId(row.id) &&
            !isConfiguredHiddenModelId(hiddenEmbeddingModelIds, row.id),
        )
        .filter((row) => matchesFormat(row.result.isGguf, "gguf"))
        // Same fit filter as the main Discover list, so the feed carousel
        // honors the toggle too.
        .filter(
          (row) =>
            !fitOnDeviceOnly ||
            row.isAvailableOnDevice ||
            hfModelFitsDevice(row.result, gpu),
        ),
    [
      hubFeed.trending.results,
      hiddenEmbeddingModelIds,
      modelDiscoveryInventorySignature,
      fitOnDeviceOnly,
      gpu,
    ],
  );
  const feedRows = useMemo(() => {
    if (!isFeedMode) return [];
    const seen = new Set<string>();
    const merged: DiscoverRow[] = [];
    for (const row of [...feedTrendingRows, ...filteredDiscoverRows]) {
      if (seen.has(row.id)) continue;
      seen.add(row.id);
      merged.push(row);
    }
    return merged;
  }, [isFeedMode, feedTrendingRows, filteredDiscoverRows]);
  const feedResults = useMemo(
    () => feedRows.map((row) => row.result),
    [feedRows],
  );
  const selectionDiscoverRows = isFeedMode ? feedRows : discoverRows;
  const selectionFilteredDiscoverRows = isFeedMode
    ? feedRows
    : filteredDiscoverRows;
  const selectionResults = isFeedMode ? feedResults : results;

  const inventoryTokens = useMemo(
    () => (isDiscoverTab ? [] : tokenizeQuery(deferredDebouncedQuery)),
    [isDiscoverTab, deferredDebouncedQuery],
  );
  // Server cache rows already apply variant-aware infra hiding. Optimistic
  // rows are not server-confirmed, so apply the client filter first.
  const isVisibleInventoryRow = useCallback(
    (row: CachedInventoryRow | LocalInventoryRow) => {
      if (row.kind === "cache") {
        return (
          !row.optimistic ||
          (!isHiddenModelId(row.id, row.repoId, row.cachePath) &&
            !isConfiguredHiddenModelId(
              hiddenEmbeddingModelIds,
              row.id,
              row.repoId,
              row.cachePath,
            ))
        );
      }
      // Local rows may lack a repo id, so also check path and title.
      return (
        !isHiddenModelId(row.id, row.repoId, row.path, row.title) ||
        (inventoryTokens.length > 0 && inventoryRowMatches(row, inventoryTokens))
      );
    },
    [hiddenEmbeddingModelIds, inventoryTokens],
  );
  // Format filter is a deliberate scope narrowing, so hard-filter it out. The
  // text query instead drives dim-not-filter on On Device (see ModelsCatalog) so
  // selection survives typing; matching rows are partitioned to the top.
  const filteredCachedRows = useMemo(
    () =>
      partitionByMatch(
        effectiveCachedRows.filter(
          (row) =>
            // Hidden-model filtering is model-only; datasets bypass it (and the
            // format filter) the way Discover does, so a dataset whose
            // id/title/path happens to contain an infra needle is not dropped.
            isDatasetMode ||
            (matchesFormat(row.modelFormat, deferredFormatFilter) &&
              matchesModelType(row, inventoryTypeFilter) &&
              isVisibleInventoryRow(row)),
        ),
        inventoryTokens,
      ),
    [
      effectiveCachedRows,
      isDatasetMode,
      deferredFormatFilter,
      inventoryTypeFilter,
      inventoryTokens,
      isVisibleInventoryRow,
    ],
  );

  const filteredLocalRows = useMemo(
    () =>
      partitionByMatch(
        effectiveLocalRows.filter(
          (row) =>
            // Hidden-model filtering is model-only; datasets bypass it (and the
            // format filter) the way Discover does, so a dataset whose
            // id/title/path happens to contain an infra needle is not dropped.
            isDatasetMode ||
            (matchesFormat(row.modelFormat, deferredFormatFilter) &&
              matchesModelType(row, inventoryTypeFilter) &&
              isVisibleInventoryRow(row)),
        ),
        inventoryTokens,
      ),
    [
      effectiveLocalRows,
      isDatasetMode,
      deferredFormatFilter,
      inventoryTypeFilter,
      inventoryTokens,
      isVisibleInventoryRow,
    ],
  );

  // Header tallies exclude infra/hidden models so the count matches the On
  // Device list (a fresh install with only the bge embedder cached reads 0,
  // not 1 over an empty list). Reuse isVisibleInventoryRow so a hidden row
  // revealed by an active search is counted too, and datasets (never infra)
  // keep their full count, mirroring the row filter above.
  const visibleCachedCount = useMemo(
    () =>
      effectiveCachedRows.filter(
        (row) => isDatasetMode || isVisibleInventoryRow(row),
      ).length,
    [effectiveCachedRows, isDatasetMode, isVisibleInventoryRow],
  );
  const visibleLocalCount = useMemo(
    () =>
      effectiveLocalRows.filter(
        (row) => isDatasetMode || isVisibleInventoryRow(row),
      ).length,
    [effectiveLocalRows, isDatasetMode, isVisibleInventoryRow],
  );

  const filterResetSignature = useMemo(
    () =>
      JSON.stringify([
        deferredDebouncedQuery,
        resourceType,
        deferredFormatFilter,
        deferredCapabilityFilter,
        effectiveSort,
        effectiveDirection,
        activeChannelId,
        ownerScope,
      ]),
    [
      deferredDebouncedQuery,
      resourceType,
      deferredFormatFilter,
      deferredCapabilityFilter,
      effectiveSort,
      effectiveDirection,
      activeChannelId,
      ownerScope,
    ],
  );
  const handleClearFilters = useCallback(() => {
    if (isDiscoverTab) {
      setDiscoverFormat("all");
      if (urlSection) {
        void navigate({
          to: "/hub",
          search: (prev) => ({ ...prev, section: undefined }),
          replace: true,
        });
      }
    } else {
      setDownloadedFormat("all");
    }
    setCapabilityFilter("all");
  }, [isDiscoverTab, urlSection, navigate]);
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
    // Re-evaluate off the raw fetched count, not the filtered one: aggressive
    // format/capability filters can reject every row, stalling
    // filteredDiscoverRows.length while results grow. The raw count guarantees
    // the auto-fire effect re-runs after each fetch lands.
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
    discoverRows: selectionDiscoverRows,
    cachedRows: effectiveCachedRows,
    localRows: effectiveLocalRows,
    filteredDiscoverRows: selectionFilteredDiscoverRows,
    filteredCachedRows,
    filteredLocalRows,
    results: selectionResults,
    accessToken: apiHfToken,
    online,
  });

  const handleSelect = useCallback(
    (id: string) => {
      setSelected(id);
      void navigate({
        to: "/hub",
        search: (prev) => ({ ...prev, model: id }),
      });
    },
    [setSelected, navigate],
  );
  const handleCloseDetail = useCallback(() => {
    // From split view, "Back to Hub" exits the master-detail layout and returns
    // to the main hub feed (not the filtered list): leave split mode, reset
    // discover state to defaults, and clear the inline preview and channel.
    if (allModelsView === "split") {
      const next = lastNonSplitViewRef.current;
      setAllModelsViewState(next);
      writeAllModelsViewPreference(next);
      const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
      setDiscoverFormat(preset?.format ?? "gguf");
      setSortBy(preset?.sort ?? "trendingScore");
      setDirection("desc");
      setCapabilityFilter("all");
      setSortBrowseActive(false);
      setQuery("");
      void navigate({
        to: "/hub",
        search: (prev) => ({ ...prev, model: undefined, section: undefined }),
      });
      return;
    }
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, model: undefined }),
    });
  }, [navigate, allModelsView]);
  const handleQueryChange = useCallback(
    (next: string) => {
      if (next.trim() === "") {
        const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
        setCapabilityFilter("all");
        setSortBrowseActive(false);
        if (preset) {
          setDiscoverFormat(preset.format);
          setSortBy(preset.sort);
          setDirection("desc");
        }
      }
      if (urlModel || urlSection) {
        void navigate({
          to: "/hub",
          search: (prev) => ({
            ...prev,
            model: undefined,
            section: undefined,
          }),
          replace: true,
        });
      }
      setQuery(next);
    },
    [urlSection, urlModel, navigate],
  );

  useEffect(() => {
    if (urlModel !== selectedId) {
      setSelected(urlModel);
    }
  }, [urlModel, selectedId, setSelected]);

  // Track the last non-split layout so leaving split mode restores it.
  useEffect(() => {
    if (allModelsView !== "split") {
      lastNonSplitViewRef.current = allModelsView;
    }
  }, [allModelsView]);

  // Split view previews the first row so the detail pane isn't empty. Feed mode
  // included: split view hides the trending feed and shows only the master list,
  // so previewing its first row lands on a real README instead of a placeholder.
  useEffect(() => {
    if (allModelsView !== "split" || urlModel) return;
    // Use the filtered rows the master pane renders, not raw inventory, so the
    // preview never lands on a filtered-out row.
    const firstId = isDiscoverTab
      ? listRows[0]?.id
      : (filteredCachedRows[0]?.id ?? filteredLocalRows[0]?.id);
    if (!firstId) return;
    setSelected(firstId);
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, model: firstId }),
      replace: true,
    });
  }, [
    allModelsView,
    urlModel,
    isDiscoverTab,
    listRows,
    filteredCachedRows,
    filteredLocalRows,
    setSelected,
    navigate,
  ]);

  useEffect(() => {
    if (!isModelDiscover || !sectionChannelId) return;
    const preset = findChannel(sectionChannelId);
    if (!preset) return;
    setDiscoverFormat(preset.format);
    setSortBy(preset.sort);
    setDirection("desc");
    setCapabilityFilter("all");
  }, [isModelDiscover, sectionChannelId]);
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
    if (!loadingModel || !selectedModel) return false;
    return modelIdsMatch(loadingModel.id, selectedModel.resource.runId);
  }, [loadingModel, selectedModel]);

  const { vramInfo, minMemory } = useHubModelVram(selectedModel, gpu);

  const gpuLabel = gpu.available
    ? `${Math.round(gpu.memoryTotalGb)} GiB`
    : "Unavailable";
  const ramLabel =
    gpu.systemRamTotalGb > 0
      ? `${Math.round(gpu.systemRamTotalGb)} GiB`
      : "Unavailable";
  const coreLabel =
    gpu.cpuCore > 0 && gpu.cpuThread > 0
      ? `${gpu.cpuCore}/${gpu.cpuThread}`
      : "Unavailable";

  const openNewChat = useCallback(() => {
    void navigate({ to: "/chat", search: { new: crypto.randomUUID() } });
  }, [navigate]);
  const runSelectedModel = useCallback(
    (opts: ModelLoadOptions, isDownloaded: boolean) => {
      if (!selectedModel) return;
      const runId = selectedModel.resource.runId;
      const resolvedConfig = resolveInitialConfig(runId, opts.ggufVariant);
      const rememberedConfig = resolvedConfig.remembered
        ? resolvedConfig.config
        : null;
      const previousConfig = currentRuntimePerModelConfig({
        includeMaxSeqLength: true,
      });
      const hasAppliedConfig = applyModelLoadConfigToRuntime(rememberedConfig);
      void selectModel({
        id: runId,
        ggufVariant: opts.ggufVariant,
        isDownloaded,
        expectedBytes: opts.expectedBytes,
        keepSpeculative: hasAppliedConfig,
        throwOnError: true,
        previousConfig,
      })
        .then(() => {
          // Read fresh: the load is async, so the checkpoint may have changed.
          const store = useChatRuntimeStore.getState();
          if (!modelIdsMatch(store.params.checkpoint, runId)) {
            store.setCheckpoint(runId, opts.ggufVariant ?? null);
          }
        })
        .catch(() => undefined);
      openNewChat();
    },
    [openNewChat, selectModel, selectedModel],
  );
  const handleLoad = useCallback(
    (opts: ModelLoadOptions) =>
      runSelectedModel(opts, selectedModel?.isDownloaded ?? true),
    [runSelectedModel, selectedModel],
  );
  const handleLoadLocal = useCallback(
    (opts: ModelLoadOptions = {}) => runSelectedModel(opts, true),
    [runSelectedModel],
  );
  const handleTrain = useCallback(() => {
    // Hub → train integration ships in a later PR.
  }, []);
  const handleSearchHub = useCallback(
    (next: string) => {
      const trimmed = next.trim();
      if (!trimmed) return;
      setModelsTab("discover");
      setResourceType("models");
      setDiscoverFormat("all");
      setCapabilityFilter("all");
      setSortBrowseActive(false);
      // Base models come from other publishers, so search the whole Hub.
      setOwnerScope("all");
      setQuery(trimmed);
      void navigate({
        to: "/hub",
        search: (prev) => ({
          ...prev,
          tab: "discover",
          section: undefined,
          model: undefined,
          kind: undefined,
        }),
      });
    },
    [navigate, setModelsTab, setOwnerScope],
  );

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
      onUseInChat: openNewChat,
      onTrain: handleTrain,
      onInventoryChange: refreshInventory,
      onSearchHub: handleSearchHub,
    }),
    [
      handleLoad,
      handleLoadLocal,
      openNewChat,
      handleTrain,
      handleSearchHub,
      refreshInventory,
    ],
  );

  const catalogState = useMemo<ModelsCatalogState>(
    () => ({
      tab,
      discoverRows: listRows,
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
        !isFeedMode &&
        (deferredFormatFilter !== "all" || deferredCapabilityFilter !== "all"),
    }),
    [
      tab,
      isFeedMode,
      listRows,
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

  const focusedHeadingText = useMemo(
    () =>
      buildFocusedHeading({
        query: deferredDebouncedQuery,
        channel: activeChannel,
        isDataset: isDatasetMode,
      }),
    [deferredDebouncedQuery, activeChannel, isDatasetMode],
  );

  const listCount = listRows.length;
  const channelSection = activeChannelId
    ? CHANNEL_TO_SECTION[activeChannelId]
    : null;
  const catalogHeader = useMemo(() => {
    if (!isDiscoverTab) return null;
    if (isFeedMode) {
      return (
        <div className="flex flex-col gap-6 pt-6">
          {allModelsView !== "split" && (
            <HubFeed
              trending={{
                rows: feedTrendingRows,
                isLoading: hubFeed.trending.isLoading,
              }}
              deviceType={deviceType}
              isDataset={isDatasetMode}
              onSelect={handleSelect}
              onOpenChannel={handleOpenList}
            />
          )}
          <div className="flex flex-col gap-3">
            <HubListHeader
              title={HUB_SECTION_TITLE.latest}
              count={listCount}
              view={allModelsView}
              onViewChange={setAllModelsView}
              onRefresh={handleRetrySearch}
              isRefreshing={isLoading}
            />
            {allModelsView === "grid" && listCount > 0 && (
              <ResultListHeader isDataset={isDatasetMode} />
            )}
          </div>
        </div>
      );
    }
    const ownerToggle = isDatasetMode ? undefined : (
      <OwnerScopeToggle value={ownerScope} onChange={setOwnerScope} />
    );
    // Compact pill so it stays beside the view-mode tabs even in the narrow
    // split pane instead of dropping to its own row.
    return (
      <div className="flex flex-col gap-3 pt-6">
        {isChannelListMode ? (
          <HubListHeader
            title={
              channelSection ? HUB_SECTION_TITLE[channelSection] : "Models"
            }
            count={listCount}
            view={allModelsView}
            onViewChange={setAllModelsView}
            onBack={handleBackToFeed}
            actions={ownerToggle}
          />
        ) : (
          <HubListHeader
            title={focusedHeadingText}
            view={allModelsView}
            onViewChange={setAllModelsView}
            onBack={isSortBrowseMode ? handleBackToFeed : undefined}
            actions={ownerToggle}
          />
        )}
        {allModelsView === "grid" && listCount > 0 && (
          <ResultListHeader isDataset={isDatasetMode} />
        )}
      </div>
    );
  }, [
    isDiscoverTab,
    isFeedMode,
    isChannelListMode,
    isSortBrowseMode,
    channelSection,
    handleBackToFeed,
    focusedHeadingText,
    listCount,
    allModelsView,
    setAllModelsView,
    ownerScope,
    setOwnerScope,
    isDatasetMode,
    feedTrendingRows,
    hubFeed.trending.isLoading,
    deviceType,
    handleSelect,
    handleOpenList,
  ]);

  const downloadedHeader = useMemo(() => {
    // Compact pills so they stay beside the view-mode tabs even in the narrow
    // split pane instead of dropping to their own row.
    const controls = (
      <div className="flex min-w-0 items-center gap-1.5">
        <InventoryTypeFilterControl
          value={inventoryTypeFilter}
          onChange={setInventoryTypeFilter}
        />
        <InventorySortControl
          value={inventorySort}
          onChange={setInventorySort}
        />
      </div>
    );
    return (
      <HubListHeader
        title="On device"
        count={visibleCachedCount + visibleLocalCount}
        view={allModelsView}
        onViewChange={setAllModelsView}
        actions={controls}
      />
    );
  }, [
    visibleCachedCount,
    visibleLocalCount,
    allModelsView,
    setAllModelsView,
    inventorySort,
    setInventorySort,
    inventoryTypeFilter,
  ]);

  const detailOpen = urlModel !== null;
  const splitMode = allModelsView === "split";

  return (
    <div className="hub-page flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden bg-background">
      <HubTopBar>
        <ModelsHeader
          cachedCount={visibleCachedCount}
          localCount={visibleLocalCount}
          isDataset={isDatasetMode}
          gpuLabel={gpuLabel}
          ramLabel={ramLabel}
          coreLabel={coreLabel}
          activeCheckpoint={activeCheckpoint}
          activeGgufVariant={activeGgufVariant}
          onTitleClick={handleResetToDiscover}
          onEject={() => void ejectModel()}
        />
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
          fitOnDeviceOnly={fitOnDeviceOnly}
          onFitOnDeviceOnlyChange={setFitOnDeviceOnly}
          onManageLocalFolders={handleManageLocalFolders}
          onOpenFineTune={() => handleOpenList("finetune")}
        />
      </HubTopBar>

      <div
        className={cn(
          "relative flex min-h-0 min-w-0 flex-1 basis-0",
          // Split mode shares the top bar's centered 1100px measure so the
          // master list lines up with the heading instead of hugging the edge.
          splitMode
            ? "flex-col lg:mx-auto lg:w-full lg:max-w-[1100px] lg:flex-row"
            : "flex-col",
        )}
      >
        <div
          className={cn(
            "flex min-h-0 flex-col",
            // Split mode keeps the catalog as a fixed-width master pane on large
            // screens; otherwise it fills the area and the detail view overlays it.
            splitMode
              ? "flex-1 lg:w-[460px] lg:max-w-[44%] lg:flex-none lg:shrink-0 lg:border-r lg:border-border/60"
              : "flex-1",
            detailOpen && !splitMode && "pointer-events-none",
          )}
          aria-hidden={(detailOpen && !splitMode) || undefined}
        >
          <ModelsCatalog
            state={catalogState}
            pagination={catalogPagination}
            handlers={catalogHandlers}
            header={catalogHeader}
            downloadedHeader={downloadedHeader}
            resetScrollKey={filterResetSignature}
            discoverView={allModelsView}
            inventorySort={inventorySort}
          />
        </div>

        {splitMode ? (
          detailOpen ? (
            <div className="hub-canvas z-20 flex min-h-0 flex-col max-lg:absolute max-lg:inset-0 lg:relative lg:min-w-0 lg:flex-1">
              <HubDetailView
                model={selectedModel}
                isDataset={isDatasetMode}
                metadataUnavailable={metadataUnavailable}
                selectionHiddenByFilters={selectionHiddenByFilters}
                runtime={inspectorRuntime}
                actions={inspectorActions}
                onBack={handleCloseDetail}
                compact={true}
              />
            </div>
          ) : (
            <div className="hidden min-h-0 flex-1 items-center justify-center px-6 text-center text-[13px] text-muted-foreground lg:flex">
              Select a model to preview its details.
            </div>
          )
        ) : (
          detailOpen && (
            <div className="hub-canvas absolute inset-0 z-20 flex min-h-0 flex-col">
              <HubDetailView
                model={selectedModel}
                isDataset={isDatasetMode}
                metadataUnavailable={metadataUnavailable}
                selectionHiddenByFilters={selectionHiddenByFilters}
                runtime={inspectorRuntime}
                actions={inspectorActions}
                onBack={handleCloseDetail}
              />
            </div>
          )
        )}
      </div>

      <OnDeviceFoldersDialog
        open={foldersDialogOpen}
        onOpenChange={setFoldersDialogOpen}
        onInventoryChange={refreshInventory}
      />
      <ExternalLinkConfirmDialog />
    </div>
  );
}
