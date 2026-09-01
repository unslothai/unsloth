// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  applyActiveModelStatusToStore,
  getInferenceStatus,
  isExternalModelId,
  isSpeechOnlyStatus,
  listGgufVariants,
  resolveInferenceCheckpointId,
  useChatModelRuntime,
  useChatRuntimeStore,
} from "@/features/chat";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { useHubInfiniteScroll } from "@/features/hub";
import {
  type ModelPickTarget,
  type PerModelConfig,
  adoptLegacyConfigKey,
  applyModelLoadConfigToRuntime,
  applyPerModelConfigToRuntime,
  currentRuntimePerModelConfig,
  hfModelFitsDevice,
  loadScopedGpu,
  resolveInitialConfig,
  useActiveModelConfig,
} from "@/features/model-picker";
import { loadOpenAIAutoSwitchSettings } from "@/features/settings";
import { taskForMediaPick } from "@/features/model-picker/components/model-selector/audio-picker-policy";
import { diffusionRouteSearch } from "@/lib/diffusion-route-search";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useGpuInfo, useInferenceGpuInfo } from "@/hooks/use-gpu-info";
import { useVramBudgetFraction } from "@/hooks/use-vram-budget-fraction";
import { toast } from "@/lib/toast";
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
import { HubModelSettingsView } from "./catalog/hub-model-settings-view";
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
import { ModelsToolbar } from "./catalog/models-toolbar";
import { FreeUpSpaceDialog } from "./catalog/free-up-space-dialog";
import { OnDeviceFoldersDialog } from "./catalog/on-device-folders-dialog";
import { OwnerScopeToggle } from "./catalog/owner-scope-toggle";
import { useDiscoverSearch } from "./hooks/use-discover-search";
import { useFeedWriteBack } from "./hooks/use-feed-write-back";
import { useHiddenEmbeddingModelIds } from "./hooks/use-hidden-embedding-models";
import { useHubFeed } from "./hooks/use-hub-feed";
import type {
  HfModelResult,
  HfModelSearchChannel,
  HfSortDirection,
  HfSortKey,
} from "./hooks/use-hub-model-search";
import { useHubModelVram } from "./hooks/use-hub-model-vram";
import { useModelsSelection } from "./hooks/use-models-selection";
import { resolveSelectionUrlSync } from "./lib/selection-resolution";
import { useHubInventory } from "./inventory";
import { LOCAL_MODEL_SOURCE } from "./inventory/constants";
import { settingsGgufVariantForRow } from "./inventory/settings-identity";
import { adoptResidentModelStatus } from "./lib/adopt-inference-status";
import { subscribeResidentStatusRefresh } from "./lib/resident-status-refresh";
import {
  type RefreshSupersession,
  registerRefresh,
  supersedingRefresh,
} from "./lib/superseded-refresh";
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
import { looksLikeLocalPath, routableToMediaPage } from "./lib/local-path";
import {
  ggufVariantsMatch,
  modelIdsMatch,
  residentModelIdMatches,
} from "./lib/model-identity";
import {
  type ModelTypeFilter,
  matchesModelType,
} from "./lib/model-type-filter";
import { resolveOwnerProviderLogo } from "./lib/provider-logos";
import { studioPageForTask } from "./lib/unsloth-support";
import { fingerprintToken } from "./lib/token-fingerprint";
import {
  buildDiscoverRows,
  detectResultFormat,
  isUnslothFinetunable,
  matchesCapability,
  matchesFormat,
} from "./lib/view-models";
import { hfApiToken, useHfTokenStore } from "./stores/hf-token-store";
import { isChannelEntryFresh, useHubFeedStore } from "./stores/hub-feed-store";
import type {
  CachedInventoryRow,
  CapabilityFilter,
  DiscoverRow,
  LocalInventoryRow,
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
  SelectedModelView,
  SelectedResourceRef,
} from "./types";

// What per-model settings are keyed by, which is not always what the loader is handed: a repo
// cached outside the active HF cache loads by snapshot path while the picker keys it by repo
// id. The row decides that, not the view it is shown in; `hub_cache` marks exactly those.
function modelConfigIdentity(
  kind: SelectedModelView["kind"],
  resource: SelectedResourceRef,
): string {
  if (kind !== "cache" && resource.source !== "hub_cache") {
    return resource.runId;
  }
  return resource.repoId ?? resource.runId;
}

const MODELS_TAB_STORAGE_KEY = "unsloth.hub.modelsTab";
const ALL_MODELS_VIEW_STORAGE_KEY = "unsloth.hub.allModelsView";
const INVENTORY_SORT_STORAGE_KEY = "unsloth.hub.inventorySort";
const OWNER_SCOPE_STORAGE_KEY = "unsloth.hub.ownerScope";

// Iconless models (no provider logo, e.g. Ornith, Inkling) show once they clear this many likes.
const MIN_ICONLESS_MODEL_LIKES = 30;

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

// Focused list heading stays "Models"/"Datasets" regardless of filters; only search relabels it.
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
  // The saved VRAM Budget the loader admits against. The "Fits on device" filter scored against
  // the 0.97 default without it, so lowering the setting moved the badges and not the filter.
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  const inferenceGpu = useInferenceGpuInfo();
  // One fit answer for every Hub row, picked the way the task pickers pick it: by the runtime that
  // PLACES the row. An image or video repo goes to the diffusion planner, on one torch device,
  // under the media rule, so judging it by llama.cpp's budget kept a 52 GiB media GGUF that clears
  // 62.1 GiB on a 64 GiB Mac and blows the planner's 44.8. Everything else keeps the GGUF
  // backend's own inventory, which is the one thing llama.cpp rows must be judged against.
  const rowFitsDevice = useCallback(
    (result: HfModelResult) => {
      const mediaRow = studioPageForTask(result.pipelineTag) !== undefined;
      const source = loadScopedGpu(
        mediaRow || !result.isGguf ? gpu : inferenceGpu,
        mediaRow,
      );
      return hfModelFitsDevice(result, source, {
        budgetFraction,
        gpuCount: source.deviceCount,
        mediaLoad: mediaRow,
        hostPooledMemory: gpu.loadDeviceSharesHostMemory,
      });
    },
    [budgetFraction, gpu, inferenceGpu],
  );
  // Browser reachability, which is what every client here asks about: the
  // selected model's metadata and the cached feed each issue their own request.
  // On the discovery phase they would stay blocked at "probing" until a
  // *listing* succeeded, and the Downloaded tab has no Retry to make that happen.
  const online = useOnlineStatus();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const hubSearch = useSearch({ from: "/hub" });
  const urlModel = hubSearch.model ?? null;
  const preferredGgufFile = hubSearch.file ?? null;

  const preferredGgufFileIntent = hubSearch.intent ?? 0;
  const { selectModel, loadingModel, loadProgress, ejectModel } =
    useChatModelRuntime();
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const residentCheckpoint = useChatRuntimeStore((s) => s.residentCheckpoint);
  // Resident, not merely picked. An image or video load evicts the chat model
  // and leaves the pick alone, so the cards kept saying "Loaded" for weights the
  // backend had already released. `undefined` is "no status read yet", which
  // stays as it was rather than flashing "On device" on every launch.
  const activeCheckpoint =
    checkpoint && !isExternalModelId(checkpoint) && residentCheckpoint !== null
      ? checkpoint
      : null;
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const activeLoadedContextLength = useChatRuntimeStore(
    (s) => s.loadedContextLength,
  );
  const [initialResidentStatusSettled, setInitialResidentStatusSettled] =
    useState(false);
  // Live settings of the loaded model, so its page shows what it is running with.
  const { config: activeModelConfig } = useActiveModelConfig();
  // Shared with the chat model selector: list only models sized for this device.
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);

  // Drops a response that lands after a newer read started, or after unmount.
  const residentStatusSeq = useRef(0);
  // The newest read started, so a dropped response can resolve with that instead of nothing. The
  // settings handlers' own guard counts settings opens only: a focus refresh supersedes without bumping.
  const residentStatusSupersession = useRef<RefreshSupersession>({
    latest: null,
  });
  // /status cannot say whether an empty answer is an idle eviction that will reload or a real
  // unload, and only this endpoint knows. Read alongside every status read rather than cached,
  // since the idle timeout is editable while this page stays mounted. Last answer wins on
  // failure, not the default: "disarmed" clears the checkpoint, so a blip would discard a selection.
  const idleUnloadArmed = useRef(false);
  const readIdleUnloadArmed = useCallback(
    (): Promise<boolean> =>
      loadOpenAIAutoSwitchSettings()
        .then((settings) => {
          idleUnloadArmed.current = settings.idleUnloadActive;
          return idleUnloadArmed.current;
        })
        .catch(() => idleUnloadArmed.current),
    [],
  );
  // Returns the read so a caller that needs the answer first can wait for it.
  const refreshResidentModelStatus = useCallback((): Promise<void> => {
    const seq = ++residentStatusSeq.current;
    const read = Promise.all([getInferenceStatus(), readIdleUnloadArmed()])
      .then(([status, idleUnloadArmed]) => {
        // A newer read owns the store, so this writes nothing; resolving with the refresh that took
        // it over keeps an awaiting caller off the pre-switch store the await was there to replace.
        if (seq !== residentStatusSeq.current)
          return supersedingRefresh(residentStatusSupersession.current, seq);
        const store = useChatRuntimeStore.getState();
        adoptResidentModelStatus(
          {
            // The loadable identifier: a GGUF off disk loads by path, and two files sharing a stem collapse.
            // Null for a speech model: this page is the other writer of params.checkpoint,
            // so adopting one here made it the chat model just as the mount sync did.
            checkpointId: isSpeechOnlyStatus(status)
              ? null
              : resolveInferenceCheckpointId(status),
            speechOnly: isSpeechOnlyStatus(status),
            ggufVariant: status.gguf_variant ?? null,
          },
          {
            checkpoint: store.params.checkpoint,
            checkpointIsExternal: isExternalModelId(store.params.checkpoint),
            activeGgufVariant: store.activeGgufVariant,
            modelLoading: store.modelLoading,
            idleUnloadArmed,
          },
          {
            setCheckpoint: (checkpointId, ggufVariant) => {
              store.setCheckpoint(checkpointId, ggufVariant);
            },
            clearCheckpoint: () => {
              store.clearCheckpoint();
            },
            // The one entry point that has applied no status yet, so settings would read defaults.
            applyStatus: (previous) => {
              applyActiveModelStatusToStore(status, {
                previousCheckpoint: previous.checkpoint ?? undefined,
                previousGgufVariant: previous.ggufVariant,
              });
            },
          },
        );
      })
      .catch(() => undefined);
    // In start order, synchronously with the sequence number, so a dropped response finds the newest read.
    registerRefresh(residentStatusSupersession.current, seq, read);
    return read;
  }, [readIdleUnloadArmed]);

  // Mount, then whenever this tab could have missed an API-driven switch. Re-reading is safe.
  useEffect(() => {
    let active = true;
    void refreshResidentModelStatus().finally(() => {
      if (active) setInitialResidentStatusSettled(true);
    });
    const unsubscribe = subscribeResidentStatusRefresh(
      refreshResidentModelStatus,
    );
    return () => {
      active = false;
      // A response still in flight adopts nothing once the Hub is gone.
      residentStatusSeq.current += 1;
      unsubscribe();
    };
  }, [refreshResidentModelStatus]);

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
  // Remembers the last non-split view so "Back to Hub" drops into the prior browsing layout.
  const lastNonSplitViewRef = useRef<AllModelsView>("two");
  const setAllModelsView = useCallback(
    (view: AllModelsView) => {
      setAllModelsViewState(view);
      writeAllModelsViewPreference(view);
      // Leaving split view drops the inline preview so the user lands on the full hub list.
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
  const [freeUpSpaceOpen, setFreeUpSpaceOpen] = useState(false);
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
      // Clear search: an active query outranks the section in `mode`, hiding the curated list.
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
      // Assert the tab here too: it fires alongside setModelsTab's navigation, and spreading prev would restore the old tab.
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
  const tokenFingerprint = useMemo(
    () => fingerprintToken(apiHfToken),
    [apiHfToken],
  );
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
  // The format dropdown always filters the visible list, including the feed's "Latest" list, so
  // the default (GGUF) hides fp8/safetensors and picking a format actually changes the rows.
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
    searchFailure,
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
  });

  const cachedListEntry = useHubFeedStore((state) =>
    liveListChannel ? state.channels[liveListChannel.id] : undefined,
  );
  const visibleResults =
    results.length === 0 &&
    liveListChannel &&
    isChannelEntryFresh(cachedListEntry, liveListChannel.id, tokenFingerprint)
      ? (cachedListEntry?.results ?? results)
      : results;

  useFeedWriteBack({
    channelId: liveListChannel?.id ?? null,
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
    inventorySettled,
    inventoryError,
    inventoryWarning,
    refreshInventory,
  } = useHubInventory({ kind: isDatasetMode ? "datasets" : "models" });

  const reloadReadySent = useRef(false);
  useEffect(() => {
    if (
      reloadReadySent.current ||
      !initialResidentStatusSettled ||
      (isDiscoverTab ? isLoading : !inventorySettled)
    ) {
      return;
    }
    reloadReadySent.current = true;
    window.dispatchEvent(new Event("unsloth:app-shell-ready"));
  }, [
    initialResidentStatusSettled,
    inventorySettled,
    isDiscoverTab,
    isLoading,
  ]);

  const modelDiscoveryInventorySignature = useMemo(
    () => discoveryInventorySignature(effectiveCachedRows, effectiveLocalRows),
    [effectiveCachedRows, effectiveLocalRows],
  );
  const modelDiscoverRows = useMemo<DiscoverRow[]>(
    () =>
      buildDiscoverRows(
        visibleResults,
        effectiveCachedRows,
        effectiveLocalRows,
      ),
    [visibleResults, modelDiscoveryInventorySignature],
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
        // Feed shows logo'd models, plus iconless ones above the likes threshold.
        (!isFeedMode ||
          resolveOwnerProviderLogo(row.owner, row.repo) !== null ||
          (row.result.likes ?? 0) >= MIN_ICONLESS_MODEL_LIKES) &&
        matchesFormat(
          detectResultFormat(row.result),
          effectiveDiscoverFormat,
        ) &&
        matchesCapability(row.capabilities, deferredCapabilityFilter) &&
        (!activeChannel?.finetunableOnly || isUnslothFinetunable(row.result)) &&
        // Models already on disk stay visible regardless of device fit, matching the chat selector.
        (!fitOnDeviceOnly ||
          row.isAvailableOnDevice ||
          rowFitsDevice(row.result)),
    );
  }, [
    rowFitsDevice,
    discoverRows,
    hiddenEmbeddingModelIds,
    isDatasetMode,
    isFeedMode,
    effectiveDiscoverFormat,
    deferredCapabilityFilter,
    activeChannel,
    fitOnDeviceOnly,
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
        // Same fit filter as the main Discover list, so the feed carousel honors the toggle too.
        .filter(
          (row) =>
            !fitOnDeviceOnly ||
            row.isAvailableOnDevice ||
            rowFitsDevice(row.result),
        ),
    [
      budgetFraction,
      rowFitsDevice,
      hubFeed.trending.results,
      hiddenEmbeddingModelIds,
      modelDiscoveryInventorySignature,
      fitOnDeviceOnly,
      gpu,
      inferenceGpu,
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
  const selectionResults = isFeedMode ? feedResults : visibleResults;

  const inventoryTokens = useMemo(
    () => (isDiscoverTab ? [] : tokenizeQuery(deferredDebouncedQuery)),
    [isDiscoverTab, deferredDebouncedQuery],
  );
  // Server cache rows already apply variant-aware infra hiding; optimistic rows are not confirmed.
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
        (inventoryTokens.length > 0 &&
          inventoryRowMatches(row, inventoryTokens))
      );
    },
    [hiddenEmbeddingModelIds, inventoryTokens],
  );
  // Format filter is a deliberate scope narrowing, so hard-filter it out. The text query instead
  // drives dim-not-filter on On Device so selection survives typing; matches sort to the top.
  const filteredCachedRows = useMemo(
    () =>
      partitionByMatch(
        effectiveCachedRows.filter(
          (row) =>
            // Hidden-model filtering is model-only; datasets bypass it (and the format filter) as Discover
            // does, so a dataset whose id/title/path contains an infra needle is not dropped.
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
            // Hidden-model filtering is model-only; datasets bypass it (and the format filter) as Discover
            // does, so a dataset whose id/title/path contains an infra needle is not dropped.
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

  // Header tallies exclude infra/hidden models so the count matches the On Device list (a fresh
  // install with only the bge embedder cached reads 0, not 1 over an empty list). Reuse
  // isVisibleInventoryRow so a search-revealed row is counted, and datasets keep their full count.
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
        inventoryTypeFilter,
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
      inventoryTypeFilter,
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
      setInventoryTypeFilter("all");
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
    // Re-evaluate off the raw fetched count, not the filtered one: aggressive filters can reject
    // every row, stalling filteredDiscoverRows.length while results grow.
    scannedCount,
    {
      enabled: online && isDiscoverTab && hasMore,
      // No phase gate: the footer renders on hasMore and so outlives the failed
      // page, and fetchMoreDiscoverManual clears the backoff itself.
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
    selectionInputId,
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
    downloadedReady,
    results: selectionResults,
    accessToken: apiHfToken,
    online,
  });

  const handleSelect = useCallback(
    (id: string) => {
      void navigate({
        to: "/hub",
        search: (prev) => ({ ...prev, model: id, file: undefined }),
      });
    },
    [navigate],
  );
  const handleCloseDetail = useCallback(() => {
    // From split view, "Back to Hub" returns to the main hub feed (not the filtered list): leave
    // split mode, reset discover state, and clear the inline preview and channel.
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
    const sync = resolveSelectionUrlSync({
      isDiscoverTab,
      urlModel,
      selectionInputId,
      resolvedSelectedId: selectedId,
      resolvedModelFormat: selectedModel?.modelFormat ?? null,
    });
    if (sync?.action === "select") {
      setSelected(sync.selectedId);
    } else if (sync?.action === "replace") {
      void navigate({
        to: "/hub",
        search: (prev) => ({
          ...prev,
          model: sync.selectedId,
          file: sync.preserveGgufFile ? prev.file : undefined,
        }),
        replace: true,
      });
    }
  }, [
    isDiscoverTab,
    navigate,
    selectedId,
    selectedModel?.modelFormat,
    selectionInputId,
    setSelected,
    urlModel,
  ]);

  // Track the last non-split layout so leaving split mode restores it.
  useEffect(() => {
    if (allModelsView !== "split") {
      lastNonSplitViewRef.current = allModelsView;
    }
  }, [allModelsView]);

  // Split view previews the first row so the detail pane isn't empty. Feed mode included: split
  // view shows only the master list, so its first row lands on a real README, not a placeholder.
  useEffect(() => {
    if (allModelsView !== "split" || urlModel) return;
    // Use the filtered rows the master pane renders so the preview never lands on a filtered-out row.
    const firstId = isDiscoverTab
      ? listRows[0]?.id
      : (filteredCachedRows[0]?.id ?? filteredLocalRows[0]?.id);
    if (!firstId) return;
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, model: firstId, file: undefined }),
      replace: true,
    });
  }, [
    allModelsView,
    urlModel,
    isDiscoverTab,
    listRows,
    filteredCachedRows,
    filteredLocalRows,
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
  const handleFreeUpSpace = useCallback(() => setFreeUpSpaceOpen(true), []);
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
    ? `${Math.round(gpu.dedicatedMemoryTotalGb)} GiB`
    : "Unavailable";
  const gpuSharedLabel =
    gpu.available && gpu.memorySharedGb > 0
      ? `${Math.round(gpu.memorySharedGb)} GiB`
      : null;
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
      // An image / video model is run by its own page, not by chat: loading it here evicted
      // the resident chat model for a llama.cpp load that could only fail. Same resolution
      // and destination the chat picker uses, so both surfaces route a pick identically.
      // `task` is not optional here: only CachedModelRepo carries pipelineTag, so every
      // cached GGUF repo (the reported MiniMax-H3 case) reports its modality on `task`.
      const mediaPage = studioPageForTask(
        taskForMediaPick(selectedModel.pipelineTag, selectedModel.task) ?? undefined,
      );
      // The target pages read a routed `model` as a Hub id, so a runId that is a PATH would
      // arrive as a repo that does not exist -- prefer the Hub id, which loads the same copy
      // since the loader reuses whichever cache root holds it. That covers a filesystem row
      // (left on today's route, and the backend preflight now refuses it by name) and a
      // cached repo the inventory pinned to its snapshot directory, whose symlinked entries
      // the pages' containment check rejects anyway.
      const routeId = runId && !looksLikeLocalPath(runId) ? runId : selectedModel.hubRepoId;
      if (
        mediaPage &&
        routableToMediaPage(selectedModel.kind, selectedModel.localSource) &&
        routeId
      ) {
        void navigate({
          to: `/${mediaPage}`,
          // `quant` is consumed verbatim as a gguf filename, so a label rides `ggufQuant`.
          search: diffusionRouteSearch(routeId, {
            ggufVariant: opts.ggufVariant ?? null,
          }),
        });
        return;
      }
      const configIdentity = modelConfigIdentity(
        selectedModel.kind,
        selectedModel.resource,
      );
      // A cached repo used to be keyed by its snapshot path (its runId); move that record over first.
      adoptLegacyConfigKey(configIdentity, runId, opts.ggufVariant);
      const resolvedConfig = resolveInitialConfig(
        configIdentity,
        opts.ggufVariant,
      );
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
        // The runtime store is not enough: applyPerModelConfigToRuntime has no field
        // for the launch flags, and /load only inherits them from the SAME resident
        // model, so a cold launch or a switch from another model ran without the
        // arguments this model was remembered with.
        ...(rememberedConfig ? { config: rememberedConfig } : {}),
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
    [navigate, openNewChat, selectModel, selectedModel],
  );
  const handleLoad = useCallback(
    (opts: ModelLoadOptions) =>
      runSelectedModel(opts, selectedModel?.isDownloaded ?? true),
    [runSelectedModel, selectedModel],
  );

  // Full-page per-model settings. Local state, not a URL param: a deep link would need the row re-resolved.
  const [settingsTarget, setSettingsTarget] = useState<ModelPickTarget | null>(
    null,
  );
  // Opening settings is where a stale read costs something: the editor is seeded once and Apply
  // reloads with what it seeded. Resolving a target is openModelSettings' own read.
  useEffect(() => {
    if (settingsTarget) void refreshResidentModelStatus();
  }, [settingsTarget, refreshResidentModelStatus]);
  // Bumped per open so a slow lookup for an abandoned row cannot land on the chosen one.
  const settingsOpenSeq = useRef(0);
  const openModelSettings = useCallback(
    async (row: CachedInventoryRow | LocalInventoryRow) => {
      const openSeq = ++settingsOpenSeq.current;
      // Before anything reads the store: the status effect watches settingsTarget. Every path out
      // seeds the editor from the store and Apply reloads with that, so a stale read misconfigures.
      await refreshResidentModelStatus();
      if (settingsOpenSeq.current !== openSeq) return;
      // loadId is what the loader accepts; repoId is only a display/API alias.
      const id = row.loadId;
      // Every name this row answers to.
      const rowAliases =
        row.kind === "local"
          ? [id, row.repoId, row.path]
          : [id, row.repoId, row.cachePath];
      // Cached repo rows carry no quant, and a null variant keys the config to `repo::` while the loader reads `repo::Q4_K_M`.
      let ggufVariant = settingsGgufVariantForRow(row);
      if (!ggufVariant && row.isGguf && row.capabilities.requiresVariant) {
        // A local row only carries a repo id inside the HF cache, so a plain folder of quants has
        // none while still needing one; the listing scans a path in the same position.
        const repoId =
          row.kind === "cache" ? row.repoId : (row.repoId ?? row.path ?? null);
        if (repoId) {
          try {
            const res = await listGgufVariants(repoId, hfApiToken(hfToken), {
              preferLocalCache: true,
              localPath:
                row.kind === "local" ? row.path : (row.cachePath ?? null),
            });
            const downloaded = res.variants.filter((v) => v.downloaded);
            // Re-read after this await too: the lookup hits the network, and a switch during it would
            // leave the branch below on the displaced model's quant. Against the server, not the store:
            // nothing pushes an API switch into this tab.
            await refreshResidentModelStatus();
            if (settingsOpenSeq.current !== openSeq) return;
            const settled = useChatRuntimeStore.getState();
            const settledCheckpoint =
              settled.params.checkpoint &&
              !isExternalModelId(settled.params.checkpoint)
                ? settled.params.checkpoint
                : null;
            const settledIsActive = rowAliases.some((alias) =>
              modelIdsMatch(alias, settledCheckpoint),
            );
            ggufVariant =
              // Loaded quant, then repo default, then whatever is on disk, mirroring LocalOnDeviceCard.
              (settledIsActive
                ? downloaded.find((v) =>
                    ggufVariantsMatch(v.quant, settled.activeGgufVariant),
                  )?.quant
                : undefined) ??
              downloaded.find((v) =>
                ggufVariantsMatch(v.quant, res.default_variant),
              )?.quant ??
              downloaded[0]?.quant ??
              null;
          } catch {
            ggufVariant = null;
          }
        }
        if (!ggufVariant) {
          // A model needing a quant cannot be configured without one: the picker matches variants exactly.
          toast.error("Couldn't determine which quant to configure.", {
            description:
              "Settings for this model are per quant. Check the connection or the model's cache, then try again.",
          });
          return;
        }
      }
      // The lookup is async: without this, whichever call finished last would win.
      if (settingsOpenSeq.current !== openSeq) {
        return;
      }
      // A repo in a previous cache loads by snapshot path, so `id` ends in the revision hash.
      const configId = row.kind === "cache" ? row.repoId : id;
      const leaf = configId.split(/[\\/]/).filter(Boolean).pop() ?? configId;
      setSettingsTarget({
        id,
        configId,
        displayName: ggufVariant ? `${leaf} · ${ggufVariant}` : leaf,
        ggufVariant,
        isGguf: row.isGguf,
        apiLoadable:
          row.isGguf &&
          (row.kind !== "local" || row.source !== LOCAL_MODEL_SOURCE.OLLAMA),
        meta: {
          source: "local",
          isLora: row.modelFormat === "adapter",
          ggufVariant: ggufVariant ?? undefined,
          isGguf: row.isGguf,
          // A partial opens settings too; claiming complete would skip download progress.
          isDownloaded: !row.partial,
          // Not on inventory rows; ModelConfigPage reads the GGUF header itself.
          contextLength: null,
        },
      });
    },
    [activeCheckpoint, activeGgufVariant, hfToken, refreshResidentModelStatus],
  );
  // Applying loads the model with exactly these settings, already persisted by ModelConfigPage.
  const runSettingsTarget = useCallback(
    (config: PerModelConfig) => {
      const target = settingsTarget;
      if (!target) return;
      const previousConfig = currentRuntimePerModelConfig({
        includeMaxSeqLength: true,
      });
      applyPerModelConfigToRuntime(config);
      setSettingsTarget(null);
      void selectModel({
        id: target.id,
        source: "local",
        ggufVariant: target.ggufVariant ?? undefined,
        isGguf: target.isGguf,
        // A partial row opens settings too; claiming complete would skip download progress.
        isDownloaded: target.meta.isDownloaded,
        isLora: target.meta.isLora,
        keepSpeculative: true,
        forceReload: true,
        // The submitted config, not only its echo in the runtime store: the store
        // does not carry llamaExtraArgs, so without this the load omits the field
        // and the route keeps the resident server's old list. Applying an edit, or
        // clearing the box, would then do nothing on this page.
        config,
        previousConfig,
      }).catch(() => undefined);
    },
    [selectModel, settingsTarget],
  );
  const handleLoadLocal = useCallback(
    (opts: ModelLoadOptions = {}) => runSelectedModel(opts, true),
    [runSelectedModel],
  );
  const handleTrain = useCallback(() => {
    // Hub → train integration ships in a later PR.
  }, []);
  // Opened from the on-device card, which passes in the quant it already resolved.
  const openSelectedModelSettings = useCallback(
    async (ggufVariant: string | null, quantIsUserPicked = false) => {
      if (!selectedModel) return;
      // Shared with openModelSettings: another row's pending lookup must not land here.
      const openSeq = ++settingsOpenSeq.current;
      // Unconditional, as in openModelSettings: the editor seeds from the store and Apply reloads it.
      await refreshResidentModelStatus();
      if (settingsOpenSeq.current !== openSeq) return;
      let variant = ggufVariant;
      // The card derived this quant from the store's active variant. A user-picked quant stands.
      if (!quantIsUserPicked) {
        const settled = useChatRuntimeStore.getState();
        const settledCheckpoint =
          settled.params.checkpoint &&
          !isExternalModelId(settled.params.checkpoint)
            ? settled.params.checkpoint
            : null;
        // Every name this model answers to, as the row menu path matches them.
        const aliases = [
          selectedModel.resource.runId,
          selectedModel.resource.repoId,
          selectedModel.resource.localPath,
        ];
        if (
          settled.activeGgufVariant &&
          aliases.some((alias) => modelIdsMatch(alias, settledCheckpoint))
        ) {
          variant = settled.activeGgufVariant;
        }
      }
      // The card passes null while its lookup is pending or failed, so guard as openModelSettings does.
      if (!variant && selectedModel.isGguf && selectedModel.requiresVariant) {
        toast.error("Couldn't determine which quant to configure.", {
          description:
            "Settings for this model are per quant. Check the connection or the model's cache, then try again.",
        });
        return;
      }
      const id = selectedModel.resource.runId;
      const configId = modelConfigIdentity(
        selectedModel.kind,
        selectedModel.resource,
      );
      // As in runSelectedModel: the editor seeds from this key, so the move must happen first.
      adoptLegacyConfigKey(configId, id, variant);
      const leaf = configId.split(/[\\/]/).filter(Boolean).pop() ?? configId;
      setSettingsTarget({
        id,
        configId,
        displayName: variant ? `${leaf} · ${variant}` : leaf,
        ggufVariant: variant,
        isGguf: selectedModel.isGguf,
        apiLoadable:
          selectedModel.isGguf &&
          selectedModel.localSource !== LOCAL_MODEL_SOURCE.OLLAMA,
        meta: {
          source: "local",
          isLora: selectedModel.modelFormat === "adapter",
          ggufVariant: variant ?? undefined,
          isGguf: selectedModel.isGguf,
          isDownloaded: selectedModel.isDownloaded,
          contextLength: null,
        },
      });
    },
    [selectedModel, refreshResidentModelStatus],
  );
  // Whether the settings page is open on the loaded model, so it can show the live launch config.
  // A GGUF off disk loads by path but is reported by its public id, so the row's path and its
  // settings identity are both offered as aliases; a loose .gguf carries no variant.
  const settingsTargetIsStandaloneFile =
    settingsTarget !== null &&
    settingsTarget.ggufVariant == null &&
    settingsTarget.id.toLowerCase().endsWith(".gguf");
  const settingsTargetIsResident =
    settingsTarget !== null &&
    residentModelIdMatches(
      activeCheckpoint,
      settingsTarget.id,
      settingsTarget.configId,
    ) &&
    (settingsTargetIsStandaloneFile ||
      ggufVariantsMatch(activeGgufVariant, settingsTarget.ggufVariant));
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
      gpuGb: inferenceGpu.available ? inferenceGpu.memoryTotalGb : undefined,
      gpuCount: inferenceGpu.deviceCount,
      systemRamGb:
        inferenceGpu.systemRamAvailableGb > 0
          ? inferenceGpu.systemRamAvailableGb
          : undefined,
    }),
    [
      isActive,
      activeGgufVariant,
      isLoadingThisModel,
      loadProgress?.phase,
      minMemory,
      vramInfo,
      inferenceGpu.available,
      inferenceGpu.memoryTotalGb,
      inferenceGpu.deviceCount,
      inferenceGpu.systemRamAvailableGb,
    ],
  );

  const inspectorActions = useMemo(
    () => ({
      onLoad: handleLoad,
      onLoadLocal: handleLoadLocal,
      onUseInChat: openNewChat,
      onEject: () => void ejectModel(),
      onTrain: handleTrain,
      onInventoryChange: refreshInventory,
      onSearchHub: handleSearchHub,
      onOpenSettings: openSelectedModelSettings,
    }),
    [
      handleLoad,
      handleLoadLocal,
      openNewChat,
      ejectModel,
      handleTrain,
      handleSearchHub,
      refreshInventory,
      openSelectedModelSettings,
    ],
  );

  const catalogState = useMemo<ModelsCatalogState>(
    () => {
      const typeFilterActive =
        !isDatasetMode && inventoryTypeFilter !== "all";
      return {
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
        searchFailure,
        online,
        isDataset: isDatasetMode,
        inventoryTokens,
        scannedCount,
        loadingIntentCount: discoverFetchIntent,
        hasMore,
        manualFetchAvailable: discoverManualFetchAvailable,
        hasActiveFilters:
          !isFeedMode &&
          (deferredFormatFilter !== "all" ||
            deferredCapabilityFilter !== "all" ||
            (tab === "downloaded" && typeFilterActive)),
        typeFilterActive,
      };
    },
    [
      tab,
      selectedId,
      isLoading,
      downloadedReady,
      inventoryError,
      inventoryWarning,
      query,
      activeCheckpoint,
      activeGgufVariant,
      searchError,
      searchFailure,
      online,
      inventoryTokens,
      scannedCount,
      hasMore,
      isFeedMode,
      listRows,
      filteredCachedRows,
      filteredLocalRows,
      isDatasetMode,
      discoverFetchIntent,
      discoverManualFetchAvailable,
      deferredFormatFilter,
      deferredCapabilityFilter,
      inventoryTypeFilter,
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
      onOpenModelSettings: openModelSettings,
    }),
    [
      handleSelect,
      fetchMoreDiscoverManually,
      handleClearFilters,
      handleRetrySearch,
      refreshInventory,
      handleSwitchDevice,
      openModelSettings,
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
    // Compact pill so it stays beside the view-mode tabs even in the narrow split pane.
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
    // Compact pills so they stay beside the view-mode tabs even in the narrow split pane.
    const controls = (
      <div className="flex min-w-0 items-center gap-1.5">
        {!isDatasetMode && (
          <InventoryTypeFilterControl
            value={inventoryTypeFilter}
            onChange={setInventoryTypeFilter}
          />
        )}
        <InventorySortControl
          value={inventorySort}
          onChange={setInventorySort}
        />
      </div>
    );
    return (
      <HubListHeader
        title="On device"
        count={filteredCachedRows.length + filteredLocalRows.length}
        view={allModelsView}
        onViewChange={setAllModelsView}
        actions={controls}
      />
    );
  }, [
    filteredCachedRows,
    filteredLocalRows,
    allModelsView,
    setAllModelsView,
    inventorySort,
    setInventorySort,
    inventoryTypeFilter,
    isDatasetMode,
  ]);

  const detailOpen = urlModel !== null;
  const splitMode = allModelsView === "split";
  // Unreachable under an opaque overlay: the detail view (full-page only) or settings.
  const catalogCovered = (detailOpen && !splitMode) || settingsTarget !== null;

  return (
    <div className="hub-page flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden bg-background">
      <HubTopBar>
        <ModelsHeader
          cachedCount={visibleCachedCount}
          localCount={visibleLocalCount}
          isDataset={isDatasetMode}
          gpuLabel={gpuLabel}
          gpuSharedLabel={gpuSharedLabel}
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
          onFreeUpSpace={handleFreeUpSpace}
          onOpenFineTune={() => handleOpenList("finetune")}
        />
      </HubTopBar>

      <div
        className={cn(
          "relative flex min-h-0 min-w-0 flex-1 basis-0",
          // Split mode shares the top bar's centered --hub-measure column so the list lines up.
          splitMode
            ? "flex-col lg:mx-auto lg:w-full lg:max-w-[var(--hub-measure)] lg:flex-row"
            : "flex-col",
        )}
      >
        <div
          className={cn(
            "flex min-h-0 flex-col",
            // Split mode keeps the catalog as a master pane that grows off a 460px floor; otherwise it
            // fills the area and the detail view overlays it.
            splitMode
              ? "flex-1 lg:w-[clamp(460px,32%,620px)] lg:max-w-[44%] lg:flex-none lg:shrink-0 lg:border-r lg:border-border/60"
              : "flex-1",
            // A full-bleed opaque overlay, so the catalog leaves the tab order; else tabbing walks behind it.
            catalogCovered && "pointer-events-none",
          )}
          aria-hidden={catalogCovered || undefined}
          inert={catalogCovered || undefined}
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
            <div
              className="hub-canvas z-20 flex min-h-0 flex-col max-lg:absolute max-lg:inset-0 lg:relative lg:min-w-0 lg:flex-1"
              inert={settingsTarget !== null || undefined}
            >
              <HubDetailView
                model={selectedModel}
                preferredGgufFile={preferredGgufFile}

                preferredGgufFileIntent={preferredGgufFileIntent}
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
            <div className="hidden min-h-0 flex-1 items-center justify-center px-6 text-center text-ui-13 text-muted-foreground lg:flex">
              Select a model to preview its details.
            </div>
          )
        ) : (
          detailOpen && (
            <div
              className="hub-canvas absolute inset-0 z-20 flex min-h-0 flex-col"
              inert={settingsTarget !== null || undefined}
            >
              <HubDetailView
                model={selectedModel}
                preferredGgufFile={preferredGgufFile}

                preferredGgufFileIntent={preferredGgufFileIntent}
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

        {/* Above the detail overlay (z-30), so settings do not stack behind a preview. */}
        {settingsTarget && (
          <div className="hub-canvas absolute inset-0 z-30 flex min-h-0 flex-col">
            <HubModelSettingsView
              target={settingsTarget}
              loadedConfig={settingsTargetIsResident ? activeModelConfig : null}
              loadedContextLength={
                settingsTargetIsResident ? activeLoadedContextLength : null
              }
              onBack={() => setSettingsTarget(null)}
              onRun={runSettingsTarget}
              compact={splitMode}
            />
          </div>
        )}
      </div>

      <OnDeviceFoldersDialog
        open={foldersDialogOpen}
        onOpenChange={setFoldersDialogOpen}
        onInventoryChange={refreshInventory}
      />
      <FreeUpSpaceDialog
        open={freeUpSpaceOpen}
        onOpenChange={setFreeUpSpaceOpen}
        onChange={refreshInventory}
      />
      <ExternalLinkConfirmDialog />
    </div>
  );
}
