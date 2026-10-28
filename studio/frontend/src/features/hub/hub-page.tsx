// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PipelineType } from "@huggingface/hub";
import { useAppShellReadySignal } from "@/components/app-readiness";
import { matchesHubAudioTask } from "@/lib/hub-audio-task";
import { useHfEndpoint, useHubSource } from "@/lib/hf-endpoint";
import { usePlatformStore } from "@/config/env";
import {
  applyActiveModelStatusToStore,
  getInferenceStatus,
  isExternalModelId,
  isSpeechOnlyStatus,
  resolveInferenceCheckpointId,
  useChatRuntimeStore,
} from "@/features/chat";
import { audioPickSearch } from "@/features/audio/route-search";
import { useHubInfiniteScroll } from "@/features/hub";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import {
  type ModelConfigHandoffRequest,
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  hfModelFitsDevice,
  loadScopedGpu,
  requestModelConfigHandoff,
} from "@/features/model-picker";
import { taskForMediaPick } from "@/features/model-picker/components/model-selector/audio-picker-policy";
import { type NpuModel, type NpuPickerSource, useNpuStatus } from "@/features/npu";
import { loadOpenAIAutoSwitchSettings } from "@/features/settings";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useGpuInfo, useInferenceGpuInfo } from "@/hooks/use-gpu-info";
import { useIsCompact } from "@/hooks/use-mobile";
import { useVramBudgetFraction } from "@/hooks/use-vram-budget-fraction";
import { diffusionRouteSearch } from "@/lib/diffusion-route-search";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
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
import { toast } from "sonner";
import { ExternalLinkConfirmDialog } from "./catalog/external-link-confirm-dialog";
import { FreeUpSpaceDialog } from "./catalog/free-up-space-dialog";
import { HubDetailView } from "./catalog/hub-detail-view";
import { HubFeed } from "./catalog/hub-feed";
import { HubTopBar } from "./catalog/hub-top-bar";
import {
  HUB_SPLIT_CUSTOM_WIDTH_CLASS,
  HubSplitDivider,
  useHubSplitWidth,
} from "./components/hub-split-divider";
import { buildHubTourSteps } from "./tour";
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
} from "./catalog/models-table";
import {
  HUB_UPLOADERS,
  type HubUploader,
  hasModelSearchFilters,
  type ModelSearchFilters,
} from "./lib/model-search-filters";
import { ModelsToolbar } from "./catalog/models-toolbar";
import { NpuCatalogList } from "./catalog/npu-catalog-list";
import { OnDeviceFoldersDialog } from "./catalog/on-device-folders-dialog";
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
import { adoptResidentModelStatus } from "./lib/adopt-inference-status";
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
import { routableToMediaPage } from "./lib/local-path";
import { residentModelIdMatches } from "./lib/model-identity";
import {
  createHubModelConfigHandoff,
  type HubModelRunSelection,
  hubAudioRunRefused,
  hubAudioTask,
} from "./lib/model-run-selection";
import { inventoryRowFormat } from "./catalog/inventory-sort";
import { matchesInventoryCapabilities } from "./lib/model-type-filter";
import { ownerOf, repoOf } from "./lib/format";
import { resolveOwnerProviderLogo } from "./lib/provider-logos";
import { subscribeResidentStatusRefresh } from "./lib/resident-status-refresh";
import {
  type RefreshSupersession,
  registerRefresh,
  supersedingRefresh,
} from "./lib/superseded-refresh";
import {
  type DatasetModality,
  matchesDatasetFilters,
} from "./lib/dataset-filters";
import {
  detectCapabilities,
  type HubCapabilityFilter,
} from "./lib/model-capabilities";
import { studioPageForTask } from "./lib/unsloth-support";
import {
  buildDiscoverRows,
  detectResultFormat,
  discoveryInventorySignature,
  hasRecommendedCheckpointLikes,
  isUnslothFinetunable,
  matchesHubCapability,
  matchesFormat,
  matchesRecommended,
  matchesRecommendedDiffusion,
} from "./lib/view-models";
import { hfApiToken, useHfTokenStore } from "./stores/hf-token-store";
import {
  feedIdentity,
  isChannelEntryFresh,
  useHubFeedStore,
} from "./stores/hub-feed-store";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
  SelectedModelView,
} from "./types";

const NO_SEARCH_FILTERS: ModelSearchFilters = {};
const UNSLOTH_UPLOADER: readonly HubUploader[] = ["unsloth"];

const ALL_MODELS_VIEW_STORAGE_KEY = "unsloth.hub.allModelsView";
const INVENTORY_SORT_STORAGE_KEY = "unsloth.hub.inventorySort";
const RUN_CONFIG_REFRESH_TIMEOUT_MS = 5_000;

// Under Recommended, iconless models (no provider logo, e.g. Ornith, Inkling) show once they clear this many likes.
const MIN_ICONLESS_MODEL_LIKES = 30;

// Image and video models skip Recommended's likes rules: new diffusion uploads rarely clear them.
function isDiffusionResult(result: HfModelResult): boolean {
  return detectCapabilities(result.tags, result.pipelineTag, result.id).some(
    (c) => c.key === "diffusion",
  );
}

function isDiffusionRow(row: DiscoverRow): boolean {
  return row.capabilities.some((c) => c.key === "diffusion");
}

// Recommended curates browsing to logo'd models plus well-liked iconless ones, and to well-liked
// checkpoints; a typed search and every other format show all matches.
function matchesDiscoverFormat(
  result: HfModelResult,
  format: ModelFormatFilter,
  browsing: boolean,
  diffusion: boolean,
  quantFormats: readonly string[],
): boolean {
  if (format !== "recommended") {
    return matchesFormat(detectResultFormat(result), format);
  }
  if (diffusion) return matchesRecommendedDiffusion(result, quantFormats);
  return (
    (!browsing ||
      ((resolveOwnerProviderLogo(ownerOf(result.id), repoOf(result.id)) !==
        null ||
        (result.likes ?? 0) >= MIN_ICONLESS_MODEL_LIKES) &&
        hasRecommendedCheckpointLikes(result))) &&
    matchesRecommended(result, quantFormats)
  );
}

function withoutGgufTag(
  tags: readonly string[] | undefined,
): readonly string[] | undefined {
  const kept = tags?.filter((tag) => tag !== "gguf");
  return kept?.length ? kept : undefined;
}

async function waitForRunConfigRefresh(
  refresh: Promise<void>,
  signal: AbortSignal,
): Promise<void> {
  if (signal.aborted) return;
  let timeoutId: number | undefined;
  let onAbort: (() => void) | undefined;
  try {
    await Promise.race([
      refresh,
      new Promise<void>((resolve) => {
        timeoutId = window.setTimeout(resolve, RUN_CONFIG_REFRESH_TIMEOUT_MS);
      }),
      new Promise<void>((resolve) => {
        onAbort = resolve;
        signal.addEventListener("abort", onAbort, { once: true });
      }),
    ]);
  } finally {
    if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    if (onAbort) signal.removeEventListener("abort", onAbort);
  }
}

class RunConfigOpenCoordinator {
  private active: AbortController | null = null;

  begin(): AbortController {
    this.cancel();
    const controller = new AbortController();
    this.active = controller;
    return controller;
  }

  cancel(): void {
    this.active?.abort();
    this.active = null;
  }

  finish(controller: AbortController): void {
    if (this.active === controller) this.active = null;
  }
}

const DEFAULT_DISCOVER_CHANNEL: ChannelId = "unsloth-trending";
const FEED_LIST_CHANNEL_ID: ChannelId = "unsloth-latest";

type DiscoverMode = "feed" | "channel-list" | "search";

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

// "All models" defaults to split layout; the choice persists across sessions.
function readAllModelsViewPreference(): AllModelsView {
  if (typeof window === "undefined") {
    return "split";
  }
  try {
    const value = window.localStorage.getItem(ALL_MODELS_VIEW_STORAGE_KEY);
    // "grid" was the removed compact list; keep those users on a card layout.
    if (value === "grid") {
      window.localStorage.setItem(ALL_MODELS_VIEW_STORAGE_KEY, "two");
      return "two";
    }
    return value === "two" || value === "split" ? value : "split";
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
    return value === "name" ||
      value === "size" ||
      value === "format" ||
      value === "recent"
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
  // Opening the Hub without an explicit ?tab= always lands on Discover; the last tab is not remembered.
  const tab = urlTab ?? "discover";

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
  runtimeIsMlx = false,
): boolean {
  if (!selectedModel || !runtimeId) return false;
  if (
    !residentModelIdMatches(
      runtimeId,
      selectedModel.id,
      selectedModel.loadId,
      selectedModel.hubRepoId,
      selectedModel.path,
      selectedModel.displayId,
    )
  ) {
    return false;
  }
  if (selectedModel.modelFormat === "gguf") {
    const localPath = selectedModel.path ?? "";
    return ggufVariant !== null || localPath.toLowerCase().endsWith(".gguf");
  }
  // An MLX load of a precision folder (6-bit/) reports the folder as its variant.
  return ggufVariant === null || runtimeIsMlx;
}

export function ModelsPage() {
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const gpu = useGpuInfo();
  // Use the saved VRAM budget so the fit filter matches the badges.
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  const inferenceGpu = useInferenceGpuInfo();
  // Judge fit by the runtime that places the row: media repos use the diffusion planner's rule,
  // everything else the GGUF backend's inventory.
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
  // Browser reachability, not listing success, or the Downloaded tab stays stuck at "probing".
  const online = useOnlineStatus();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const hubSearch = useSearch({ from: "/hub" });
  const urlModel = hubSearch.model ?? null;
  const preferredGgufFile = hubSearch.file ?? null;

  const preferredGgufFileIntent = hubSearch.intent ?? 0;
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const residentCheckpoint = useChatRuntimeStore((s) => s.residentCheckpoint);
  const activeCheckpoint =
    checkpoint && !isExternalModelId(checkpoint) && residentCheckpoint !== null
      ? checkpoint
      : null;
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const loadedIsMlx = useChatRuntimeStore((s) => s.loadedIsMlx);
  const loadingModel = useChatRuntimeStore((s) => s.loadingModelPick);
  const [initialResidentStatusSettled, setInitialResidentStatusSettled] =
    useState(false);
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);

  const residentStatusSeq = useRef(0);
  const residentStatusSupersession = useRef<RefreshSupersession>({
    latest: null,
  });
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
  const refreshResidentModelStatus = useCallback(
    (
      { preserveIdleUnloaded = true }: {
        preserveIdleUnloaded?: boolean;
      } = {},
    ): Promise<void> => {
      const seq = ++residentStatusSeq.current;
      const selected = useChatRuntimeStore.getState().params.checkpoint;
      const read = Promise.all([
        getInferenceStatus(
          undefined,
          selected && !isExternalModelId(selected) ? selected : undefined,
        ),
        readIdleUnloadArmed(),
      ])
        .then(([status, idleUnloadArmed]) => {
          if (seq !== residentStatusSeq.current)
            return supersedingRefresh(residentStatusSupersession.current, seq);
          const store = useChatRuntimeStore.getState();
          adoptResidentModelStatus(
            {
              // Null for speech models: this page also writes params.checkpoint and would make it the chat model.
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
              idleUnloadArmed: preserveIdleUnloaded && idleUnloadArmed,
            },
            {
              setCheckpoint: (checkpointId, ggufVariant) => {
                store.setCheckpoint(checkpointId, ggufVariant);
              },
              clearCheckpoint: () => {
                store.clearCheckpoint();
              },
              applyStatus: (previous) => {
                applyActiveModelStatusToStore(status, {
                  previousCheckpoint: previous.checkpoint ?? undefined,
                  previousGgufVariant: previous.ggufVariant,
                  adoptingExistingServerModel:
                    previous.checkpoint === null || previous.checkpoint === "",
                });
              },
            },
          );
        })
        .catch(() => undefined);
      registerRefresh(residentStatusSupersession.current, seq, read);
      return read;
    },
    [readIdleUnloadArmed],
  );

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
      residentStatusSeq.current += 1;
      unsubscribe();
    };
  }, [refreshResidentModelStatus]);

  useEffect(
    () =>
      subscribeModelLifecycle(({ runtime }) => {
        if (runtime === "chat" || runtime === "stt") return;
        void refreshResidentModelStatus({ preserveIdleUnloaded: false });
      }),
    [refreshResidentModelStatus],
  );

  const { tab, setTab: setModelsTab } = useModelsTabState();
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<HfSortKey>(
    () => findChannel(DEFAULT_DISCOVER_CHANNEL)?.sort ?? "trendingScore",
  );
  const [direction, setDirection] = useState<HfSortDirection>("desc");
  // Null until the user picks: then Unsloth's uploads, or every uploader once a search is typed.
  const [uploaderPick, setUploaderPick] = useState<HubUploader[] | null>(null);
  // Until the popover's first change from the defaults, a capability pick keeps just that one.
  const [firstPickArmed, setFirstPickArmed] = useState(true);
  const [datasetModalities, setDatasetModalities] = useState<
    DatasetModality[]
  >([]);
  const handlePopoverModalitiesChange = useCallback(
    (next: DatasetModality[]) => {
      setFirstPickArmed(false);
      setDatasetModalities(next);
    },
    [],
  );
  const handlePopoverCapabilitiesChange = useCallback(
    (next: HubCapabilityFilter[]) => {
      setFirstPickArmed(false);
      setCapabilityFilters(next);
    },
    [],
  );
  const handlePopoverSearchFiltersChange = useCallback(
    (next: ModelSearchFilters) => {
      setFirstPickArmed(false);
      setSearchFilters(next);
    },
    [],
  );
  const handlePopoverUploadersChange = useCallback((next: HubUploader[]) => {
    setFirstPickArmed(false);
    setUploaderPick(next);
  }, []);
  const urlResourceType: ResourceTypeFilter =
    hubSearch.kind === "datasets" ? "datasets" : "models";
  const [resourceType, setResourceType] =
    useState<ResourceTypeFilter>(urlResourceType);
  // Resync on URL kind change (back/forward); it is only seeded on mount.
  const syncedUrlKind = useRef(urlResourceType);
  useEffect(() => {
    if (syncedUrlKind.current === urlResourceType) {
      return;
    }
    syncedUrlKind.current = urlResourceType;
    setResourceType((current) =>
      current === urlResourceType ? current : urlResourceType,
    );
    // Back/forward skips handleResourceTypeChange, so the other kind's hidden bounds go here too.
    setSearchFilters({});
    setDatasetModalities([]);
  }, [urlResourceType]);
  const [discoverFormat, setDiscoverFormat] = useState<ModelFormatFilter>(
    () => findChannel(DEFAULT_DISCOVER_CHANNEL)?.format ?? "recommended",
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
  const [npuStatus, setNpuStatus] = useNpuStatus();
  const npuSource: NpuPickerSource | undefined = npuStatus?.supported
    ? { status: npuStatus, onStatusChange: setNpuStatus }
    : undefined;
  const formatFilter = isDiscoverTab ? discoverFormat : downloadedFormat;
  const npuCatalogSource =
    formatFilter === "npu" && !isDatasetMode ? npuSource : undefined;
  const showNpuCatalog = npuCatalogSource !== undefined;
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
  const hubSource = useHubSource();
  const [storedSearchFilters, setSearchFilters] = useState<ModelSearchFilters>(
    {},
  );
  // Several capabilities show models with any of them; none shows every model.
  const [capabilityFilters, setCapabilityFilters] = useState<
    HubCapabilityFilter[]
  >([]);
  // Pipeline tasks from an audio page's Search Hub link; they apply while the Audio filter is on.
  const [linkedAudioTask, setLinkedAudioTask] = useState<
    PipelineType[] | null
  >(null);
  const [allModelsView, setAllModelsViewState] = useState<AllModelsView>(
    readAllModelsViewPreference,
  );
  const isCompactViewport = useIsCompact();
  // The split view's auto-picked first row; an explicit selection or leaving split clears it.
  const autoPreviewIdRef = useRef<string | null>(null);
  // Remembers the last non-split view so "Back to Hub" drops into the prior browsing layout.
  const lastNonSplitViewRef = useRef<AllModelsView>("two");
  const setAllModelsView = useCallback(
    (view: AllModelsView) => {
      setAllModelsViewState(view);
      writeAllModelsViewPreference(view);
      if (view !== "split") {
        autoPreviewIdRef.current = null;
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
  const [inventoryCapabilities, setInventoryCapabilities] = useState<
    HubCapabilityFilter[]
  >([]);
  const [inventoryFirstPickArmed, setInventoryFirstPickArmed] = useState(true);
  const handleInventoryCapabilitiesChange = useCallback(
    (next: HubCapabilityFilter[]) => {
      setInventoryFirstPickArmed(false);
      setInventoryCapabilities(next);
    },
    [],
  );
  const [foldersDialogOpen, setFoldersDialogOpen] = useState(false);
  const [freeUpSpaceOpen, setFreeUpSpaceOpen] = useState(false);
  const [discoverFetchIntent, setDiscoverFetchIntent] = useState(0);
  const [sortBrowseActive, setSortBrowseActive] = useState(false);

  const handleTabChange = useCallback(
    (next: ModelsTab) => {
      setSortBrowseActive(false);
      // The filters control is Discover-only, so a hidden filter must not reapply on return.
      setSearchFilters({});
      setDatasetModalities([]);
      setUploaderPick(null);
      setFirstPickArmed(true);
      setModelsTab(next);
    },
    [setModelsTab],
  );

  const handleResourceTypeChange = useCallback(
    (next: ResourceTypeFilter) => {
      if (next === resourceType) return;
      setResourceType(next);
      setDownloadedFormat("all");
      setCapabilityFilters([]);
      setDatasetModalities([]);
      setSearchFilters({});
      setUploaderPick(null);
      setFirstPickArmed(true);
      setSortBrowseActive(false);
      if (next === "models") {
        const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
        setDiscoverFormat(preset?.format ?? "recommended");
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
      setCapabilityFilters([]);
      setDatasetModalities([]);
      setSearchFilters({});
      setUploaderPick(null);
      setFirstPickArmed(true);
      setSortBrowseActive(false);
      // An active query outranks the section in `mode`, hiding the curated list.
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
    setDiscoverFormat(preset?.format ?? "recommended");
    setSortBy(preset?.sort ?? "trendingScore");
    setDirection("desc");
    setCapabilityFilters([]);
    setDatasetModalities([]);
    setSearchFilters({});
    setUploaderPick(null);
    setFirstPickArmed(true);
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
    setCapabilityFilters([]);
    setDatasetModalities([]);
    setSearchFilters({});
    setUploaderPick(null);
    setFirstPickArmed(true);
    setDownloadedFormat("all");
    setDiscoverFormat(preset?.format ?? "recommended");
    setSortBy(preset?.sort ?? "trendingScore");
    setDirection("desc");
    setSortBrowseActive(false);
    void navigate({
      to: "/hub",
      // Set the tab explicitly: spreading prev would restore the old tab.
      search: (prev) => ({
        ...prev,
        tab: "discover",
        section: undefined,
        model: undefined,
        kind: undefined,
      }),
    });
  }, [navigate, setModelsTab]);

  // A capability link opens the sorted Discover list with that filter on (the feed ignores it).
  // The param is consumed so later filter changes stick and a repeat link re-applies.
  const urlCapability = hubSearch.capability ?? null;
  const urlTaskKey = hubSearch.audioTask?.join("|") ?? "";
  useEffect(() => {
    if (!urlCapability) return;
    setResourceType("models");
    setQuery("");
    setDiscoverFormat("all");
    setCapabilityFilters([urlCapability]);
    setLinkedAudioTask(
      urlCapability === "audio" && urlTaskKey
        ? (urlTaskKey.split("|") as PipelineType[])
        : null,
    );
    setSortBrowseActive(true);
    // The media pages run curated Unsloth uploads, so start there.
    setUploaderPick([...UNSLOTH_UPLOADER]);
    setFirstPickArmed(false);
    void navigate({
      to: "/hub",
      search: (prev) => ({
        ...prev,
        tab: "discover",
        kind: undefined,
        capability: undefined,
        audioTask: undefined,
        section: undefined,
        model: undefined,
      }),
      replace: true,
    });
  }, [urlCapability, urlTaskKey, navigate]);
  // Picking another filter drops the link's tasks, so returning to Audio shows every audio model.
  if (!capabilityFilters.includes("audio") && linkedAudioTask !== null) {
    setLinkedAudioTask(null);
  }

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
  const uploaders =
    uploaderPick ?? (debouncedQuery.trim() ? HUB_UPLOADERS : UNSLOTH_UPLOADER);
  const ownerScope =
    uploaders.length === 1 && uploaders[0] === "unsloth" ? "unsloth" : "all";
  // The publisher and config.json lookups are Hugging Face endpoints.
  const searchFilters = useMemo(
    () =>
      hubSource === "modelscope"
        ? NO_SEARCH_FILTERS
        : ownerScope === "all" && uploaders.length < HUB_UPLOADERS.length
          ? { ...storedSearchFilters, uploaders: [...uploaders] }
          : storedSearchFilters,
    [hubSource, ownerScope, uploaders, storedSearchFilters],
  );
  const hasSearchFilters = hasModelSearchFilters(searchFilters);
  const hfToken = useHfTokenStore((s) => s.token);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const apiHfToken = hfApiToken(debouncedHfToken);
  const hubHfEndpoint = useHfEndpoint();
  const tokenFingerprint = useMemo(
    () => feedIdentity(hubHfEndpoint, apiHfToken),
    [hubHfEndpoint, apiHfToken],
  );
  const deferredFormatFilter = useDeferredValue(formatFilter);
  const deferredCapabilityFilters = useDeferredValue(capabilityFilters);

  const hasQuery = deferredDebouncedQuery.trim() !== "";
  const mode: DiscoverMode = isModelDiscover
    ? hasQuery
      ? "search"
      : urlSection != null
        ? "channel-list"
        : sortBrowseActive || hasSearchFilters || uploaderPick !== null
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

  const requestedSort: HfSortKey =
    isFeedMode && liveListChannel ? liveListChannel.sort : sortBy;
  // ModelScope has neither a creation-date sort nor trending; its closest is last modified.
  const effectiveSort: HfSortKey =
    hubSource === "modelscope" &&
    (requestedSort === "createdAt" || requestedSort === "fresh")
      ? "lastModified"
      : isDatasetMode && requestedSort === "fresh"
        ? "createdAt"
        : requestedSort;
  const effectiveDirection: HfSortDirection = isFeedMode ? "desc" : direction;
  // The format dropdown always filters the visible list, including the feed's "Latest" list, so
  // the default (Recommended) hides unrunnable checkpoints and picking a format changes the rows.
  const effectiveDiscoverFormat: ModelFormatFilter = deferredFormatFilter;
  const checkpointQuantFormats = gpu.checkpointQuantFormats;
  // Image / video NVFP4 also needs the build's NVFP4 switch: the backend refuses *-NVFP4 diffusion repos while it is off.
  const diffusionQuantFormats = useMemo(
    () =>
      gpu.nvfp4Diffusion
        ? checkpointQuantFormats
        : checkpointQuantFormats.filter((format) => format !== "nvfp4"),
    [checkpointQuantFormats, gpu.nvfp4Diffusion],
  );

  // A GGUF-tagged channel query would never fetch the FP8 / NVFP4 rows Recommended admits here.
  const widenRecommendedChannel =
    effectiveDiscoverFormat === "recommended" &&
    checkpointQuantFormats.length > 0;
  const listChannel = useMemo<HfModelSearchChannel | null>(() => {
    if (!liveListChannel) return null;
    return {
      owner: liveListChannel.owner,
      tags: widenRecommendedChannel
        ? withoutGgufTag(liveListChannel.tags)
        : liveListChannel.tags,
      query: liveListChannel.query,
      idSuffix: liveListChannel.idSuffix,
    };
  }, [liveListChannel, widenRecommendedChannel]);

  // checkpointQuantFormats is a superset of diffusionQuantFormats, so this never skips a row the list shows.
  const searchBrowsing = debouncedQuery.trim() === "";
  // Keyed by contents: a system poll hands out a new array, which would restart the search.
  const quantFormatsKey = checkpointQuantFormats.join(",");
  const keepSearchResult = useMemo(() => {
    const quantFormats = quantFormatsKey ? quantFormatsKey.split(",") : [];
    return (result: HfModelResult) =>
      matchesDiscoverFormat(
        result,
        effectiveDiscoverFormat,
        searchBrowsing,
        isDiffusionResult(result),
        quantFormats,
      );
  }, [effectiveDiscoverFormat, searchBrowsing, quantFormatsKey]);

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
    isDiscoverTab: isDiscoverTab && !showNpuCatalog,
    isDatasetMode,
    sortBy: effectiveSort,
    direction: effectiveDirection,
    channel: listChannel,
    ownerScope,
    filters: searchFilters,
    keepResult: keepSearchResult,
    task:
      // Asked of the Hub only when Audio is the one filter: with others it would drop their models.
      capabilityFilters.length === 1 && capabilityFilters[0] === "audio"
        ? (linkedAudioTask ?? undefined)
        : undefined,
  });

  const cachedListEntry = useHubFeedStore((state) =>
    liveListChannel ? state.channels[liveListChannel.id] : undefined,
  );
  const visibleResults =
    !hasSearchFilters &&
    results.length === 0 &&
    liveListChannel &&
    isChannelEntryFresh(cachedListEntry, liveListChannel.id, tokenFingerprint)
      ? (cachedListEntry?.results ?? results)
      : results;

  useFeedWriteBack({
    channelId: hasSearchFilters ? null : liveListChannel?.id ?? null,
    results,
    isLoading,
    accessToken: apiHfToken,
  });

  const {
    cachedRows: effectiveCachedRows,
    localRows: effectiveLocalRows,
    availableSet,
    partialSet,
    downloadingSet,
    downloadedReady,
    inventorySettled,
    inventoryError,
    inventoryWarning,
    refreshInventory,
  } = useHubInventory({
    kind: isDatasetMode ? "datasets" : "models",
    includeUnslothModels: true,
  });

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
    signalReady();
  }, [
    signalReady,
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

  const deferredDatasetModalities = useDeferredValue(datasetModalities);
  const { minRows, maxRows } = storedSearchFilters;
  const datasetDiscoverRows = useMemo<DiscoverRow[]>(() => {
    if (!isDatasetMode) return [];
    return datasetResults
      .filter((ds) =>
        matchesDatasetFilters(ds, deferredDatasetModalities, minRows, maxRows),
      )
      .map((ds) => {
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
          isDownloadingOnDevice: partialSet.has(lower) && downloadingSet.has(lower),
          summary: summaryParts.join(" · ") || ds.prettyName || "Dataset",
          capabilities: [],
        };
      });
  }, [
    isDatasetMode,
    datasetResults,
    deferredDatasetModalities,
    minRows,
    maxRows,
    availableSet,
    partialSet,
    downloadingSet,
  ]);

  const discoverRows = isDatasetMode ? datasetDiscoverRows : modelDiscoverRows;

  const filteredDiscoverRows = useMemo(() => {
    if (isDatasetMode) return discoverRows;
    return discoverRows.filter(
      (row) =>
        !isHiddenModelId(row.id) &&
        !isConfiguredHiddenModelId(hiddenEmbeddingModelIds, row.id) &&
        matchesDiscoverFormat(
          row.result,
          effectiveDiscoverFormat,
          !hasQuery,
          isDiffusionRow(row),
          isDiffusionRow(row) ? diffusionQuantFormats : checkpointQuantFormats,
        ) &&
        (deferredCapabilityFilters.length === 0 ||
          deferredCapabilityFilters.some(
            (key) =>
              matchesHubCapability(row, key) &&
              (key !== "audio" ||
                matchesHubAudioTask(row.result.pipelineTag, linkedAudioTask)),
          )) &&
        (!activeChannel?.finetunableOnly || isUnslothFinetunable(row.result)) &&
        // Models on disk stay visible regardless of fit, matching the chat selector.
        (!fitOnDeviceOnly ||
          row.isAvailableOnDevice ||
          rowFitsDevice(row.result)),
    );
  }, [
    rowFitsDevice,
    discoverRows,
    hiddenEmbeddingModelIds,
    isDatasetMode,
    hasQuery,
    effectiveDiscoverFormat,
    checkpointQuantFormats,
    diffusionQuantFormats,
    deferredCapabilityFilters,
    linkedAudioTask,
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
        // Catches low-like safetensors repos cross-tagged "gguf" for a stray .gguf export.
        .filter(
          (row) => isDiffusionRow(row) || hasRecommendedCheckpointLikes(row.result),
        )
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
  // Server rows already apply variant-aware infra hiding; optimistic rows are not confirmed.
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
      return (
        !isHiddenModelId(row.id, row.repoId, row.path, row.title) ||
        (inventoryTokens.length > 0 &&
          inventoryRowMatches(row, inventoryTokens))
      );
    },
    [hiddenEmbeddingModelIds, inventoryTokens],
  );
  // The text query dims rather than filters on On Device so selection survives typing.
  const filteredCachedRows = useMemo(
    () =>
      partitionByMatch(
        effectiveCachedRows.filter(
          (row) =>
            // Datasets bypass hidden-model and format filtering, as Discover does.
            isDatasetMode ||
            (deferredFormatFilter !== "unsloth" &&
              matchesFormat(inventoryRowFormat(row), deferredFormatFilter) &&
              matchesInventoryCapabilities(row, inventoryCapabilities) &&
              isVisibleInventoryRow(row)),
        ),
        inventoryTokens,
      ),
    [
      effectiveCachedRows,
      isDatasetMode,
      deferredFormatFilter,
      inventoryCapabilities,
      inventoryTokens,
      isVisibleInventoryRow,
    ],
  );

  const filteredLocalRows = useMemo(
    () =>
      partitionByMatch(
        effectiveLocalRows.filter(
          (row) =>
            isDatasetMode ||
            ((deferredFormatFilter === "unsloth"
              ? row.source === "unsloth"
              : matchesFormat(inventoryRowFormat(row), deferredFormatFilter)) &&
              matchesInventoryCapabilities(row, inventoryCapabilities) &&
              isVisibleInventoryRow(row)),
        ),
        inventoryTokens,
      ),
    [
      effectiveLocalRows,
      isDatasetMode,
      deferredFormatFilter,
      inventoryCapabilities,
      inventoryTokens,
      isVisibleInventoryRow,
    ],
  );

  // Exclude infra/hidden models so the header count matches the On Device list.
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
        deferredCapabilityFilters,
        deferredDatasetModalities,
        searchFilters,
        // ModelScope drops searchFilters, but the dataset rows are filtered here either way.
        minRows,
        maxRows,
        inventoryCapabilities,
        effectiveSort,
        effectiveDirection,
        activeChannelId,
        ownerScope,
      ]),
    [
      deferredDebouncedQuery,
      resourceType,
      deferredFormatFilter,
      deferredCapabilityFilters,
      deferredDatasetModalities,
      searchFilters,
      minRows,
      maxRows,
      inventoryCapabilities,
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
      setInventoryCapabilities([]);
      setInventoryFirstPickArmed(true);
    }
    setCapabilityFilters([]);
    setDatasetModalities([]);
    setSearchFilters({});
    setUploaderPick(null);
    setFirstPickArmed(true);
  }, [isDiscoverTab, urlSection, navigate]);
  const handleResetInventoryFilters = useCallback(() => {
    setInventoryCapabilities([]);
    setInventoryFirstPickArmed(true);
  }, []);
  const handleResetPopoverFilters = useCallback(() => {
    setCapabilityFilters([]);
    setDatasetModalities([]);
    setSearchFilters({});
    setUploaderPick(null);
    setFirstPickArmed(true);
  }, []);
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
    // Use the raw fetched count: aggressive filters can stall the filtered length while results grow.
    scannedCount,
    {
      enabled: online && isDiscoverTab && hasMore,
      // No phase gate: the footer outlives the failed page and the manual fetch clears the backoff.
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
  const [runConfigOpenCoordinator] = useState(
    () => new RunConfigOpenCoordinator(),
  );
  const [runConfigOpening, setRunConfigOpening] = useState<{
    modelId: string;
    controller: AbortController;
  } | null>(null);
  useEffect(() => {
    return () => runConfigOpenCoordinator.cancel();
  }, [runConfigOpenCoordinator, selectedId]);

  const handleSelect = useCallback(
    (id: string) => {
      autoPreviewIdRef.current = null;
      runConfigOpenCoordinator.cancel();
      setRunConfigOpening(null);
      void navigate({
        to: "/hub",
        search: (prev) => ({ ...prev, model: id, file: undefined }),
      });
    },
    [navigate, runConfigOpenCoordinator],
  );
  const handleCloseDetail = useCallback(() => {
    runConfigOpenCoordinator.cancel();
    setRunConfigOpening(null);
    autoPreviewIdRef.current = null;
    // From split view, "Back to Hub" returns to the main hub feed (not the filtered list): leave
    // split mode, reset discover state, and clear the inline preview and channel.
    if (allModelsView === "split" && !isCompactViewport) {
      const next = lastNonSplitViewRef.current;
      setAllModelsViewState(next);
      writeAllModelsViewPreference(next);
      const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
      setDiscoverFormat(preset?.format ?? "recommended");
      setSortBy(preset?.sort ?? "trendingScore");
      setDirection("desc");
      setCapabilityFilters([]);
      setDatasetModalities([]);
      setSearchFilters({});
      setUploaderPick(null);
      setFirstPickArmed(true);
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
  }, [navigate, allModelsView, isCompactViewport, runConfigOpenCoordinator]);
  const handleQueryChange = useCallback(
    (next: string) => {
      if (next.trim() === "") {
        const preset = findChannel(DEFAULT_DISCOVER_CHANNEL);
        setCapabilityFilters([]);
        setDatasetModalities([]);
        setSearchFilters({});
        setUploaderPick(null);
        setFirstPickArmed(true);
        setSortBrowseActive(false);
        if (preset) {
          setDiscoverFormat(preset.format);
          setSortBy(preset.sort);
          setDirection("desc");
        }
      } else if (query.trim() === "") {
        // A new search starts on every uploader (Unsloth's still listed first), whatever was picked.
        setUploaderPick(null);
      }
      // Clearing the split view's auto-picked preview only re-picks it: 2 navigations per key (#15221).
      const openedModel = !!urlModel && urlModel !== autoPreviewIdRef.current;
      if (openedModel || urlSection) {
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
    [query, urlSection, urlModel, navigate],
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

  useEffect(() => {
    if (allModelsView !== "split") {
      lastNonSplitViewRef.current = allModelsView;
    }
  }, [allModelsView]);

  // Narrowing below lg drops the auto-picked preview so the rows show; a model the user opened stays.
  useEffect(() => {
    if (!isCompactViewport || allModelsView !== "split") return;
    if (!urlModel || urlModel !== autoPreviewIdRef.current) return;
    autoPreviewIdRef.current = null;
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, model: undefined, file: undefined }),
      replace: true,
    });
  }, [isCompactViewport, allModelsView, urlModel, navigate]);

  // Split view previews the first row so the detail pane isn't empty. Feed mode included: split
  // view shows only the master list, so its first row lands on a real README, not a placeholder.
  useEffect(() => {
    if (allModelsView !== "split" || urlModel || isCompactViewport) return;
    // Use the filtered rows the master pane renders so the preview never lands on a filtered-out row.
    const firstId = isDiscoverTab
      ? listRows[0]?.id
      : (filteredCachedRows[0]?.id ?? filteredLocalRows[0]?.id);
    if (!firstId) return;
    autoPreviewIdRef.current = firstId;
    void navigate({
      to: "/hub",
      search: (prev) => ({ ...prev, model: firstId, file: undefined }),
      replace: true,
    });
  }, [
    allModelsView,
    urlModel,
    isCompactViewport,
    isDiscoverTab,
    listRows,
    filteredCachedRows,
    filteredLocalRows,
    navigate,
  ]);

  useEffect(() => {
    // A capability link clears the section itself; applying the preset would reset its filter.
    if (!isModelDiscover || !sectionChannelId || urlCapability) return;
    const preset = findChannel(sectionChannelId);
    if (!preset) return;
    setDiscoverFormat(preset.format);
    setSortBy(preset.sort);
    setDirection("desc");
    setCapabilityFilters([]);
    setDatasetModalities([]);
    setSearchFilters({});
    setUploaderPick(null);
    setFirstPickArmed(true);
  }, [isModelDiscover, sectionChannelId, urlCapability]);
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
        loadedIsMlx === true,
      ),
    [activeCheckpoint, activeGgufVariant, loadedIsMlx, selectedModel],
  );

  const isLoadingThisModel = useMemo(() => {
    if (!loadingModel || !selectedModel) return false;
    return residentModelIdMatches(
      loadingModel.id,
      selectedModel.id,
      selectedModel.loadId,
      selectedModel.hubRepoId,
      selectedModel.path,
      selectedModel.displayId,
    );
  }, [loadingModel, selectedModel]);

  const { vramInfo } = useHubModelVram(selectedModel, gpu);

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
  const handleSearchHub = useCallback(
    (next: string) => {
      const trimmed = next.trim();
      if (!trimmed) return;
      setModelsTab("discover");
      setResourceType("models");
      setDiscoverFormat("all");
      setCapabilityFilters([]);
      setDatasetModalities([]);
      setSearchFilters({});
      setSortBrowseActive(false);
      // Base models come from other publishers, so search the whole Hub.
      setUploaderPick(null);
      setFirstPickArmed(true);
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
    [navigate, setModelsTab],
  );
  const openRunSettingsInChat = useCallback(
    (request: ModelConfigHandoffRequest) => {
      const chatRuntime = useChatRuntimeStore.getState();
      chatRuntime.setActiveThreadId(null);
      chatRuntime.setActiveProjectId(null);
      chatRuntime.setIncognito(false);
      requestModelConfigHandoff(request);
      const { requestId } = request;
      void navigate({ to: "/chat", search: { new: requestId } }).catch(() => {
        clearModelConfigHandoff(requestId);
      });
    },
    [navigate],
  );

  const handleRunNpu = useCallback(
    (model: NpuModel) =>
      openRunSettingsInChat({
        requestId: createModelConfigHandoffRequestId(),
        id: model.model_path,
        displayName: model.id,
        meta: {
          source: "local",
          isLora: false,
          isDownloaded: true,
          isVision: model.supports_vision,
          contextLength: model.max_context_length,
        },
      }),
    [openRunSettingsInChat],
  );

  const handleRun = useCallback(
    async (
      selection: HubModelRunSelection,
      mediaPage: ReturnType<typeof studioPageForTask> | "audio",
    ) => {
      if (!selectedModel) return;
      if (mediaPage) {
        runConfigOpenCoordinator.cancel();
        setRunConfigOpening(null);
        if (
          !selectedModel.hubRepoId ||
          !routableToMediaPage(selectedModel.kind, selectedModel.localSource)
        ) {
          return;
        }
        void navigate({
          to: `/${mediaPage}`,
          search:
            mediaPage === "audio"
              ? audioPickSearch(selectedModel.hubRepoId, {
                  ...selection,
                  task: hubAudioTask(
                    selectedModel,
                    taskForMediaPick(
                      selectedModel.pipelineTag,
                      selectedModel.task,
                    ),
                  ),
                  audioType: selectedModel.audioType,
                  isGguf: selectedModel.isGguf,
                  loadId: selectedModel.loadId,
                })
              : diffusionRouteSearch(selectedModel.hubRepoId, selection),
        });
        return;
      }
      if (
        hubAudioRunRefused(
          selectedModel,
          taskForMediaPick(selectedModel.pipelineTag, selectedModel.task),
        )
      ) {
        toast.error(
          `${selectedModel.hubRepoId} is not an audio model Unsloth can run yet. The Audio page lists the families it supports.`,
          { duration: 7000 },
        );
        return;
      }
      const controller = runConfigOpenCoordinator.begin();
      setRunConfigOpening({ modelId: selectedModel.id, controller });
      try {
        await waitForRunConfigRefresh(
          refreshResidentModelStatus(),
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const requestId = createModelConfigHandoffRequestId();
        const request = createHubModelConfigHandoff({
          requestId,
          model: selectedModel,
          selection,
        });
        if (!request) {
          toast.error("Couldn't open run settings.", {
            description:
              "The model or selected quantization is no longer available to run.",
          });
          return;
        }
        openRunSettingsInChat(request);
      } finally {
        runConfigOpenCoordinator.finish(controller);
        setRunConfigOpening((current) =>
          current?.controller === controller ? null : current,
        );
      }
    },
    [
      navigate,
      openRunSettingsInChat,
      refreshResidentModelStatus,
      runConfigOpenCoordinator,
      selectedModel,
    ],
  );

  const inspectorRuntime = useMemo(
    () => ({
      isActive,
      activeGgufVariant,
      isLoadingThisModel,
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
      vramInfo,
      inferenceGpu.available,
      inferenceGpu.memoryTotalGb,
      inferenceGpu.deviceCount,
      inferenceGpu.systemRamAvailableGb,
    ],
  );

  const inspectorActions = useMemo(
    () => ({
      onInventoryChange: refreshInventory,
      onSearchHub: handleSearchHub,
      onRun: handleRun,
      runConfigPending: runConfigOpening?.modelId === selectedModel?.id,
    }),
    [
      handleRun,
      handleSearchHub,
      refreshInventory,
      runConfigOpening,
      selectedModel?.id,
    ],
  );

  const catalogState = useMemo<ModelsCatalogState>(
    () => {
      const typeFilterActive =
        !isDatasetMode && inventoryCapabilities.length > 0;
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
          ((isModelDiscover && hasSearchFilters) ||
            // A search lists every uploader, so picking Unsloth narrows it.
            (isModelDiscover &&
              uploaderPick !== null &&
              ownerScope === "unsloth" &&
              debouncedQuery.trim() !== "") ||
            deferredFormatFilter !== "all" ||
            deferredCapabilityFilters.length > 0 ||
            (isDatasetMode &&
              (deferredDatasetModalities.length > 0 ||
                minRows !== undefined ||
                maxRows !== undefined)) ||
            (tab === "downloaded" && typeFilterActive)),
        typeFilterActive,
        // A single format filter makes every row's format dot the same; Recommended mixes GGUF and checkpoints.
        showFormatDots:
          isDatasetMode ||
          deferredFormatFilter === "all" ||
          deferredFormatFilter === "recommended",
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
      deferredCapabilityFilters,
      deferredDatasetModalities,
      minRows,
      maxRows,
      hasSearchFilters,
      uploaderPick,
      ownerScope,
      debouncedQuery,
      inventoryCapabilities,
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
          </div>
        </div>
      );
    }
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
          />
        ) : (
          <HubListHeader
            title={focusedHeadingText}
            view={allModelsView}
            onViewChange={setAllModelsView}
            onBack={isSortBrowseMode ? handleBackToFeed : undefined}
          />
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
    isDatasetMode,
    feedTrendingRows,
    hubFeed.trending.isLoading,
    deviceType,
    handleSelect,
    handleOpenList,
  ]);

  const downloadedHeader = useMemo(() => {
    return (
      <HubListHeader
        title="On device"
        count={filteredCachedRows.length + filteredLocalRows.length}
        view={allModelsView}
        onViewChange={setAllModelsView}
      />
    );
  }, [
    filteredCachedRows,
    filteredLocalRows,
    allModelsView,
    setAllModelsView,
  ]);

  // NPU rows have no detail view; one left open would cover their list.
  const detailOpen = urlModel !== null && !showNpuCatalog;
  const splitMode = allModelsView === "split" && !showNpuCatalog;
  const catalogCovered = detailOpen && !splitMode;

  const catalogPaneRef = useRef<HTMLDivElement>(null);
  const splitWidth = useHubSplitWidth();

  const tour = useGuidedTourController({
    id: "hub",
    steps: useMemo(() => buildHubTourSteps({ catalogCovered }), [catalogCovered]),
  });

  return (
    <div className="hub-page flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden bg-background">
      <GuidedTour {...tour.tourProps} />
      <HubTopBar>
        <ModelsHeader
          cachedCount={visibleCachedCount}
          localCount={visibleLocalCount}
          isDataset={isDatasetMode}
          gpuLabel={gpuLabel}
          gpuSharedLabel={gpuSharedLabel}
          ramLabel={ramLabel}
          coreLabel={coreLabel}
          onTitleClick={handleResetToDiscover}
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
          searchFilters={storedSearchFilters}
          onSearchFiltersChange={handlePopoverSearchFiltersChange}
          uploaders={uploaders}
          uploadersChanged={uploaderPick !== null}
          onUploadersChange={handlePopoverUploadersChange}
          onResetFilters={handleResetPopoverFilters}
          capabilityFilters={capabilityFilters}
          onCapabilityFiltersChange={handlePopoverCapabilitiesChange}
          isolateFirstCapability={firstPickArmed}
          datasetModalities={datasetModalities}
          onDatasetModalitiesChange={handlePopoverModalitiesChange}
          fitOnDeviceOnly={fitOnDeviceOnly}
          onFitOnDeviceOnlyChange={setFitOnDeviceOnly}
          onManageLocalFolders={handleManageLocalFolders}
          onFreeUpSpace={handleFreeUpSpace}
          onOpenFineTune={() => handleOpenList("finetune")}
          npuAvailable={npuSource !== undefined}
          inventorySort={inventorySort}
          onInventorySortChange={setInventorySort}
          inventoryCapabilities={inventoryCapabilities}
          onInventoryCapabilitiesChange={handleInventoryCapabilitiesChange}
          isolateFirstInventoryCapability={inventoryFirstPickArmed}
          onResetInventoryFilters={handleResetInventoryFilters}
        />
      </HubTopBar>

      <div
        className={cn(
          "relative flex min-h-0 min-w-0 flex-1 basis-0",
          splitMode
            ? "flex-col lg:mx-auto lg:w-full lg:max-w-[var(--hub-measure)] lg:flex-row"
            : "flex-col",
        )}
      >
        <div
          ref={catalogPaneRef}
          data-tour="hub-catalog"
          className={cn(
            "flex min-h-0 flex-col",
            splitMode
              ? cn(
                  "flex-1 lg:flex-none lg:shrink-0 lg:border-r lg:border-border/60",
                  splitWidth.width === null
                    ? "lg:w-[clamp(460px,32%,620px)] lg:max-w-[44%]"
                    : HUB_SPLIT_CUSTOM_WIDTH_CLASS,
                )
              : "flex-1",
            // Opaque overlay: remove the catalog from the tab order.
            catalogCovered && "pointer-events-none",
          )}
          style={splitMode ? splitWidth.style : undefined}
          aria-hidden={catalogCovered || undefined}
          inert={catalogCovered || undefined}
        >
          {npuCatalogSource ? (
            <NpuCatalogList
              source={npuCatalogSource}
              query={query}
              onDevice={!isDiscoverTab}
              onRun={handleRunNpu}
            />
          ) : (
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
          )}
        </div>

        {splitMode && !isCompactViewport && (
          <HubSplitDivider
            paneRef={catalogPaneRef}
            width={splitWidth.width}
            onWidthChange={splitWidth.setWidth}
          />
        )}
        {splitMode ? (
          detailOpen ? (
            <div
              className="hub-canvas z-20 flex min-h-0 flex-col max-lg:absolute max-lg:inset-0 lg:relative lg:min-w-0 lg:flex-1"
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
              data-tour="hub-detail"
              className="hub-canvas absolute inset-0 z-20 flex min-h-0 flex-col"
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
      </div>

      <OnDeviceFoldersDialog
        open={foldersDialogOpen}
        onOpenChange={setFoldersDialogOpen}
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
