// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  MODEL_TYPE_TO_HF_TASKS,
  PRIORITY_TRAINING_MODELS,
  applyPriorityOrdering,
} from "@/config/training";
import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  type LocalSource,
  type ModelInventoryFormat,
  useHubInventory,
} from "@/features/inventory";
import {
  TRAINING_METHOD_DESCRIPTIONS,
  TRAINING_METHOD_DOTS,
  TRAINING_METHOD_HINTS,
  TRAINING_METHOD_LABELS,
  inferTrainingModelTypeFromMetadata,
  resolvePickerInferredModelType,
  useTrainingConfigStore,
} from "@/features/training";
import { cacheLocalPathMatchesSelection } from "@/features/training/lib/cache-reference";
import { validateTrainingModelCandidate } from "@/features/training/lib/freeform-model-validation";
import {
  type HfModelResult,
  classifyUnslothSupport,
  useGpuInfo,
  useHfModelSearch,
  useInfiniteScroll,
  useLatestRef,
  useOnlineStatus,
} from "@/hooks";
import { extractParamLabel, modelShortName } from "@/lib/format";
import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import { cn, formatCompact } from "@/lib/utils";
import {
  type VramFitStatus,
  type TrainingMethod as VramTrainingMethod,
  buildModelVramMap,
} from "@/lib/vram";
import { useHfTokenStore } from "@/stores/hf-token-store";
import type { ModelType, TrainingMethod } from "@/types/training";
import {
  ArrowDown01Icon,
  ChipIcon,
  FolderSearchIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type RefObject, useCallback, useMemo, useState } from "react";
import { toast } from "sonner";
import { useShallow } from "zustand/react/shallow";
import { PickerShell } from "./hub-picker-shell";
import {
  RetryButton,
  isHfAuthError,
  looksLikeLocalPath,
} from "./picker-tab-toggle";
import { SelectablePickerItem } from "./selectable-picker-item";
import {
  buildCachedTrainingModelLookup,
  buildLocalTrainingModelLookup,
} from "./training-picker-lookups";
import { useHfErrorToast } from "./use-hf-error-toast";
import { useHubPickerState } from "./use-hub-picker-state";

const MODEL_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.modelPickerTab";

const TRIGGER_BASE = cn(
  "menu-trigger field-soft inline-flex h-9 items-center gap-1.5 rounded-[12px] px-3 text-[12.5px] text-muted-foreground transition-colors",
  "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
);

type TrainModelDeviceItem = {
  key: string;
  id: string;
  title: string;
  path: string;
  source: LocalSource;
  sourceLabel: string;
  knownCached: boolean;
  localPath: string | null;
  modelFormat: ModelInventoryFormat | null;
  modelType: ModelType;
};

function trainModelSourceWeight(source: LocalSource): number {
  switch (source) {
    case "hf_cache":
      return 0;
    case "models_dir":
      return 1;
    case "custom":
      return 2;
    case "lmstudio":
      return 3;
    case "ollama":
      return 4;
    default:
      return 5;
  }
}

function compareTrainModelDeviceItems(
  a: TrainModelDeviceItem,
  b: TrainModelDeviceItem,
): number {
  const titleCmp = (a.title || a.id).localeCompare(b.title || b.id, undefined, {
    sensitivity: "base",
  });
  if (titleCmp !== 0) {
    return titleCmp;
  }
  const sourceCmp =
    trainModelSourceWeight(a.source) - trainModelSourceWeight(b.source);
  if (sourceCmp !== 0) {
    return sourceCmp;
  }
  const pathCmp = a.path.localeCompare(b.path, undefined, {
    sensitivity: "base",
  });
  if (pathCmp !== 0) {
    return pathCmp;
  }
  return a.key.localeCompare(b.key);
}

function trainModelDeviceItemMatchesSelection({
  item,
  selectedModel,
  selectedLocalPath,
  selectedFormat,
}: {
  item: TrainModelDeviceItem;
  selectedModel: string | null;
  selectedLocalPath: string | null;
  selectedFormat: ModelInventoryFormat | null;
}): boolean {
  if (!selectedModel || selectedModel !== item.id) {
    return false;
  }
  if (
    selectedFormat &&
    item.modelFormat &&
    selectedFormat !== item.modelFormat
  ) {
    return false;
  }
  if (selectedLocalPath?.trim()) {
    return cacheLocalPathMatchesSelection(item.localPath, selectedLocalPath);
  }
  return true;
}

export function TrainModelPicker() {
  const gpu = useGpuInfo();
  const {
    selectedModel,
    modelLocalPath,
    modelFormat: selectedModelFormat,
    setSelectedModel,
    selectTrainingModel,
    modelType,
    trainingMethod,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      modelLocalPath: s.modelLocalPath,
      modelFormat: s.modelFormat,
      setSelectedModel: s.setSelectedModel,
      selectTrainingModel: s.selectTrainingModel,
      modelType: s.modelType,
      trainingMethod: s.trainingMethod,
    })),
  );
  const [selectedDeviceKey, setSelectedDeviceKey] = useState<string | null>(
    null,
  );
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const picker = useHubPickerState({
    storageKey: MODEL_PICKER_TAB_STORAGE_KEY,
    hfToken,
    online,
  });
  const task = modelType ? MODEL_TYPE_TO_HF_TASKS[modelType] : undefined;
  const {
    cachedRows,
    localRows,
    downloadedReady,
    inventoryError,
    inventoryWarning,
    refreshInventory,
  } = useHubInventory({ kind: "models", enabled: picker.open });
  const isLoadingLocalModels = !downloadedReady;
  const localModelsError =
    inventoryError && cachedRows.length === 0 && localRows.length === 0
      ? "Couldn't scan local models"
      : null;
  const retryLocalModels = useCallback(() => {
    refreshInventory().catch(() => undefined);
  }, [refreshInventory]);

  const deviceType = usePlatformStore((s) => s.deviceType);
  const isTrainableLocalRow = useCallback(
    (row: LocalInventoryRow) => {
      if (row.partial) {
        return false;
      }
      if (row.source === "lmstudio" || row.source === "ollama") {
        return false;
      }
      if (!row.capabilities.canTrain) {
        return false;
      }
      if (
        classifyUnslothSupport({
          modelId: row.repoId ?? row.loadId,
          pipelineTag: row.pipelineTag,
          tags: row.tags,
          libraryName: row.libraryName,
          quantMethod: row.quantMethod,
          deviceType,
        }).status === "unsupported"
      ) {
        return false;
      }
      return true;
    },
    [deviceType],
  );

  const isTrainableCachedRow = useCallback(
    (row: CachedInventoryRow) => {
      if (row.partial || !row.capabilities.canTrain) {
        return false;
      }
      if (
        classifyUnslothSupport({
          modelId: row.repoId,
          pipelineTag: row.pipelineTag,
          tags: row.tags,
          libraryName: row.libraryName,
          quantMethod: row.quantMethod,
          deviceType,
        }).status === "unsupported"
      ) {
        return false;
      }
      return true;
    },
    [deviceType],
  );

  const cachedModelById = useMemo(() => {
    const map = new Map<string, CachedInventoryRow>();
    for (const row of cachedRows) {
      if (!isTrainableCachedRow(row)) {
        continue;
      }
      map.set(row.repoId.toLowerCase(), row);
    }
    return map;
  }, [cachedRows, isTrainableCachedRow]);

  const cachedModelByLookup = useMemo(() => {
    return buildCachedTrainingModelLookup(cachedRows, isTrainableCachedRow);
  }, [cachedRows, isTrainableCachedRow]);

  const nonPartialUntrainableCachedModelByLookup = useMemo(() => {
    return buildCachedTrainingModelLookup(
      cachedRows.filter((row) => !row.partial),
      (row) => !isTrainableCachedRow(row),
    );
  }, [cachedRows, isTrainableCachedRow]);

  const localModelByLookup = useMemo(() => {
    return buildLocalTrainingModelLookup(localRows, isTrainableLocalRow);
  }, [localRows, isTrainableLocalRow]);

  const nonPartialUntrainableLocalModelByLookup = useMemo(() => {
    return buildLocalTrainingModelLookup(
      localRows.filter((row) => !row.partial),
      (row) => !isTrainableLocalRow(row),
    );
  }, [localRows, isTrainableLocalRow]);

  const localModelByRepo = useMemo(() => {
    const map = new Map<string, LocalInventoryRow>();
    for (const row of localRows) {
      if (!(isTrainableLocalRow(row) && row.repoId)) {
        continue;
      }
      const key = row.repoId.toLowerCase();
      const existing = map.get(key);
      if (
        !existing ||
        (existing.source === "hf_cache" && row.source !== "hf_cache")
      ) {
        map.set(key, row);
      }
    }
    return map;
  }, [localRows, isTrainableLocalRow]);

  const trainableLocalModels = useMemo<TrainModelDeviceItem[]>(
    () =>
      [
        ...cachedRows.filter(isTrainableCachedRow).map((row) => ({
          key: row.id,
          id: row.loadId,
          title: row.repoId,
          path: row.cachePath ?? row.repoId,
          source: "hf_cache" as const,
          sourceLabel: "HF cache",
          knownCached: true,
          localPath: row.cachePath ?? null,
          modelFormat: row.modelFormat,
          modelType: inferTrainingModelTypeFromMetadata({
            tags: row.tags,
            pipelineTag: row.pipelineTag,
            identifiers: [row.repoId, row.repo],
          }),
        })),
        ...localRows.filter(isTrainableLocalRow).map((row) => ({
          key: row.id,
          id: row.loadId,
          title: row.repoId ?? row.title,
          path: row.path,
          source: row.source,
          sourceLabel: row.sourceLabel,
          knownCached: row.source === "hf_cache",
          localPath: row.path,
          modelFormat: row.modelFormat,
          modelType: inferTrainingModelTypeFromMetadata({
            tags: row.tags,
            pipelineTag: row.pipelineTag,
            identifiers: [row.repoId, row.loadId, row.title, row.path],
          }),
        })),
      ].sort(compareTrainModelDeviceItems),
    [cachedRows, localRows, isTrainableCachedRow, isTrainableLocalRow],
  );

  const pickerView = picker.getViewState({
    hasDeviceItems: trainableLocalModels.length > 0,
    isLoadingDevice: isLoadingLocalModels,
  });
  const { activeQuery, handleQueryChange, tab } = pickerView;

  const filteredLocalModels = useMemo(() => {
    const tokens = tokenizeQuery(picker.deviceQuery);
    if (tokens.length === 0) {
      return trainableLocalModels;
    }
    return trainableLocalModels.filter((m) =>
      matchTokens(`${m.id} ${m.title} ${m.path} ${m.sourceLabel}`, tokens),
    );
  }, [trainableLocalModels, picker.deviceQuery]);

  const selectedDeviceItemKey = useMemo(() => {
    const matches = filteredLocalModels.filter((item) =>
      trainModelDeviceItemMatchesSelection({
        item,
        selectedModel,
        selectedLocalPath: modelLocalPath,
        selectedFormat: selectedModelFormat,
      }),
    );
    if (matches.length === 0) {
      return null;
    }
    if (
      selectedDeviceKey &&
      matches.some((item) => item.key === selectedDeviceKey)
    ) {
      return selectedDeviceKey;
    }
    return matches[0].key;
  }, [
    filteredLocalModels,
    modelLocalPath,
    selectedDeviceKey,
    selectedModel,
    selectedModelFormat,
  ]);

  const {
    results: hfResults,
    isLoading: isLoadingHf,
    isLoadingMore: isLoadingHfMore,
    fetchMore: fetchMoreHf,
    retry: retryHf,
    error: hfError,
  } = useHfModelSearch(picker.debouncedHubQuery, {
    task,
    accessToken: picker.debouncedHfToken || undefined,
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
    enabled: online && picker.open && tab === "hub",
  });

  const hubSearchActive = online && picker.open && tab === "hub";
  const hubSearchActiveRef = useLatestRef(hubSearchActive);
  const fetchMoreHfRef = useLatestRef(fetchMoreHf);
  useHfErrorToast(hubSearchActive ? hfError : null, "models");

  const hubResultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    const seen = new Set(ids.map((id) => id.toLowerCase()));
    if (
      selectedModel &&
      !looksLikeLocalPath(selectedModel) &&
      !seen.has(selectedModel.toLowerCase())
    ) {
      ids.push(selectedModel);
    }
    return applyPriorityOrdering(ids);
  }, [hfResults, selectedModel]);

  const hfResultById = useMemo(() => {
    const map = new Map<string, HfModelResult>();
    for (const result of hfResults) {
      map.set(result.id.toLowerCase(), result);
    }
    return map;
  }, [hfResults]);

  const vramMap = useMemo(() => {
    const fitMap = buildModelVramMap(
      hfResults,
      trainingMethod as VramTrainingMethod,
      gpu,
    );
    const map = new Map<
      string,
      { est: number; status: VramFitStatus | null; detail: string | null }
    >();
    for (const r of hfResults) {
      const detail = r.totalParams
        ? formatCompact(r.totalParams)
        : extractParamLabel(r.id);
      const fit = fitMap.get(r.id);
      map.set(r.id, {
        est: fit?.est ?? 0,
        status: fit?.status ?? null,
        detail,
      });
    }
    return map;
  }, [hfResults, gpu, trainingMethod]);

  const fetchMoreOpenHf = useCallback(() => {
    if (!hubSearchActiveRef.current) {
      return;
    }
    fetchMoreHfRef.current();
  }, []);

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMoreOpenHf,
    hfResults.length,
    hubSearchActive,
  );

  function pick(
    id: string,
    options?: {
      knownCached?: boolean;
      localPath?: string | null;
      modelFormat?: ModelInventoryFormat | null;
    },
    inferredModelType?: ModelType | null,
  ) {
    const next = id.trim();
    if (!next) {
      return;
    }
    const nextModelType = inferredModelType
      ? resolvePickerInferredModelType(modelType, inferredModelType)
      : modelType;
    if (nextModelType) {
      selectTrainingModel(next, nextModelType, options);
    } else {
      setSelectedModel(next, options);
    }
    picker.closePicker();
  }

  function pickHubModel(id: string) {
    const key = id.trim().toLowerCase();
    const result = hfResultById.get(key);
    const cached = cachedModelByLookup.get(key) ?? cachedModelById.get(key);
    const local = localModelByLookup.get(key) ?? localModelByRepo.get(key);
    const validationCached =
      cached ?? nonPartialUntrainableCachedModelByLookup.get(key);
    const validationLocal =
      local ?? nonPartialUntrainableLocalModelByLookup.get(key);
    const validation = validateTrainingModelCandidate(
      {
        id,
        modelFormat:
          validationCached?.modelFormat ?? validationLocal?.modelFormat ?? null,
        capabilities:
          validationCached?.capabilities ?? validationLocal?.capabilities ?? null,
        pipelineTag:
          result?.pipelineTag ??
          validationCached?.pipelineTag ??
          validationLocal?.pipelineTag,
        tags: result?.tags ?? validationCached?.tags ?? validationLocal?.tags,
        libraryName:
          result?.libraryName ??
          validationCached?.libraryName ??
          validationLocal?.libraryName,
        quantMethod:
          result?.quantMethod ??
          validationCached?.quantMethod ??
          validationLocal?.quantMethod,
      },
      { deviceType },
    );
    if (!validation.ok) {
      toast.error("Can't use model for training", {
        description: validation.reason,
      });
      return;
    }
    const inferredType = result
      ? inferTrainingModelTypeFromMetadata({
          tags: result.tags,
          pipelineTag: result.pipelineTag,
          identifiers: [result.id],
        })
      : cached
        ? inferTrainingModelTypeFromMetadata({
            tags: cached.tags,
            pipelineTag: cached.pipelineTag,
            identifiers: [cached.repoId, cached.repo],
          })
        : local
          ? inferTrainingModelTypeFromMetadata({
              tags: local.tags,
              pipelineTag: local.pipelineTag,
              identifiers: [
                local.repoId,
                local.loadId,
                local.title,
                local.path,
              ],
            })
          : inferTrainingModelTypeFromMetadata({ identifiers: [id] });
    pick(
      id,
      {
        knownCached: cached !== undefined || local?.source === "hf_cache",
        localPath: cached?.cachePath ?? local?.path ?? null,
        modelFormat: cached?.modelFormat ?? local?.modelFormat ?? null,
      },
      inferredType,
    );
  }

  function pickFreeformModel(id: string) {
    if (tab === "hub") {
      pickHubModel(id);
      return;
    }
    const key = id.trim().toLowerCase();
    const cached = cachedModelByLookup.get(key);
    const local = localModelByLookup.get(key);
    const validationCached =
      cached ?? nonPartialUntrainableCachedModelByLookup.get(key);
    const validationLocal =
      local ?? nonPartialUntrainableLocalModelByLookup.get(key);
    const validation = validateTrainingModelCandidate(
      {
        id,
        modelFormat:
          validationCached?.modelFormat ?? validationLocal?.modelFormat ?? null,
        capabilities:
          validationCached?.capabilities ?? validationLocal?.capabilities ?? null,
        pipelineTag: validationCached?.pipelineTag ?? validationLocal?.pipelineTag,
        tags: validationCached?.tags ?? validationLocal?.tags,
        libraryName: validationCached?.libraryName ?? validationLocal?.libraryName,
        quantMethod: validationCached?.quantMethod ?? validationLocal?.quantMethod,
      },
      { deviceType },
    );
    if (!validation.ok) {
      toast.error("Can't use model for training", {
        description: validation.reason,
      });
      return;
    }
    pick(
      id,
      undefined,
      inferTrainingModelTypeFromMetadata({ identifiers: [id] }),
    );
  }

  const display = selectedModel ? modelShortName(selectedModel) : null;
  const hasExactMatch =
    activeQuery.length === 0
      ? false
      : tab === "hub"
        ? hubResultIds.some((id) => id === activeQuery)
        : trainableLocalModels.some(
            (m) => m.id === activeQuery || m.path === activeQuery,
          );
  const showUseThis =
    activeQuery.length > 0 &&
    !hasExactMatch &&
    (tab === "hub" || looksLikeLocalPath(activeQuery));
  const useThisLabel =
    tab === "hub" ? "Use as Hugging Face model" : "Use as local path";

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
      noun="models"
      isHubLoading={isLoadingHf}
      showUseThis={showUseThis}
      useThisLabel={useThisLabel}
      onUseThis={() => pickFreeformModel(activeQuery)}
      placeholder={{
        hub: "Search or paste a Hugging Face id...",
        device: "Search local models or paste a folder path...",
      }}
      scrollRef={scrollRef}
      trigger={
        <button
          type="button"
          data-tour="studio-model"
          className={cn(TRIGGER_BASE, "w-full min-w-[180px] justify-between")}
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <HugeiconsIcon
              icon={ChipIcon}
              strokeWidth={1.75}
              className="size-3.5 shrink-0"
            />
            <span
              className={cn(
                "truncate font-medium",
                display ? "text-foreground" : "text-muted-foreground",
              )}
            >
              {display ?? "Select model"}
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
          items={filteredLocalModels}
          isLoading={isLoadingLocalModels}
          error={localModelsError}
          warning={inventoryWarning}
          activeKey={selectedDeviceItemKey}
          hasQuery={activeQuery.length > 0}
          onPick={(model) => {
            setSelectedDeviceKey(model.key);
            pick(
              model.id,
              {
                knownCached: model.knownCached,
                localPath: model.localPath,
                modelFormat: model.modelFormat,
              },
              model.modelType,
            );
          }}
          onRetry={retryLocalModels}
        />
      }
      hubContent={
        <HubList
          ids={hubResultIds}
          value={selectedModel}
          vramMap={vramMap}
          isLoading={isLoadingHf}
          isLoadingMore={isLoadingHfMore}
          gpuTotalGb={gpu.available ? gpu.memoryTotalGb : null}
          hasQuery={activeQuery.length > 0}
          error={hfError}
          onPick={pickHubModel}
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
  activeKey,
  hasQuery,
  onPick,
  onRetry,
}: {
  items: TrainModelDeviceItem[];
  isLoading: boolean;
  error: string | null;
  warning: boolean;
  activeKey: string | null;
  hasQuery: boolean;
  onPick: (model: TrainModelDeviceItem) => void;
  onRetry: () => void;
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> Scanning local models…
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-[12.5px] font-medium text-foreground">
            Couldn't scan local models
          </p>
          <p className="text-[11px] leading-snug text-muted-foreground">
            {error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
    // When the user has typed a custom query the parent renders a
    // "Use as local path" affordance above this list — don't compete with
    // a contradictory "no models found" empty state in that case.
    if (hasQuery) {
      return null;
    }
    return (
      <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
        <HugeiconsIcon
          icon={FolderSearchIcon}
          strokeWidth={1.5}
          className="size-5 text-muted-foreground/70"
        />
        <p className="text-xs text-muted-foreground">No local models found.</p>
        <p className="text-[10.5px] text-muted-foreground/70">
          Paste a folder path above or switch to Hugging Face.
        </p>
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {items.map((m) => (
        <li key={m.key}>
          <SelectablePickerItem
            active={activeKey === m.key}
            onSelect={() => onPick(m)}
          >
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <span className="block min-w-0 flex-1 cursor-text select-text truncate">
                  {m.title || m.id}
                </span>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs break-all">
                {m.path}
              </TooltipContent>
            </Tooltip>
            <span className="ml-2 shrink-0 rounded-[6px] border border-border/60 px-1.5 py-0.5 text-[10px] font-medium leading-none text-muted-foreground">
              {m.sourceLabel}
            </span>
          </SelectablePickerItem>
        </li>
      ))}
      {warning && (
        <li className="px-2 py-1 text-[10.5px] text-muted-foreground/80">
          Some local locations could not be scanned.
        </li>
      )}
    </ul>
  );
}

function HubList({
  ids,
  value,
  vramMap,
  isLoading,
  isLoadingMore,
  gpuTotalGb,
  hasQuery,
  error,
  onPick,
  onRetry,
  sentinelRef,
}: {
  ids: string[];
  value: string | null;
  vramMap: Map<
    string,
    { est: number; status: VramFitStatus | null; detail: string | null }
  >;
  isLoading: boolean;
  isLoadingMore: boolean;
  gpuTotalGb: number | null;
  hasQuery: boolean;
  error: string | null;
  onPick: (id: string) => void;
  onRetry: () => void;
  sentinelRef: RefObject<HTMLDivElement | null>;
}) {
  if (isLoading && ids.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> Searching Hugging Face…
      </div>
    );
  }
  if (ids.length === 0) {
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
    if (hasQuery) {
      return null;
    }
    return (
      <div className="px-4 py-8 text-center text-xs text-muted-foreground">
        No models found.
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {ids.map((id) => {
        const fit = vramMap.get(id);
        const exceeds = fit?.status === "exceeds";
        const tight = fit?.status === "tight";
        return (
          <li key={id}>
            <SelectablePickerItem
              active={value === id}
              onSelect={() => onPick(id)}
            >
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <span
                    className={cn(
                      "block min-w-0 flex-1 cursor-text select-text truncate",
                      exceeds && "text-muted-foreground",
                    )}
                  >
                    {id}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="left" className="max-w-xs break-all">
                  {id}
                  {fit && fit.est > 0 && gpuTotalGb != null && (
                    <span className="mt-1 block text-[10px]">
                      {exceeds
                        ? `Needs ~${fit.est}GB VRAM (GPU: ${gpuTotalGb}GB)`
                        : tight
                          ? `~${fit.est}GB VRAM (tight on ${gpuTotalGb}GB)`
                          : `~${fit.est}GB VRAM`}
                    </span>
                  )}
                </TooltipContent>
              </Tooltip>
              <span className="ml-auto flex shrink-0 items-center gap-1.5">
                {exceeds && (
                  <span className="rounded bg-red-50 px-1.5 py-0.5 text-[9px] font-semibold text-red-700 dark:bg-red-950 dark:text-red-400">
                    OOM
                  </span>
                )}
                {tight && (
                  <span className="text-[9px] font-semibold text-amber-500">
                    TIGHT
                  </span>
                )}
                {fit?.detail && (
                  <span className="text-[10px] text-muted-foreground">
                    {fit.detail}
                  </span>
                )}
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

export function TrainingMethodSelect() {
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const setTrainingMethod = useTrainingConfigStore((s) => s.setTrainingMethod);
  return (
    <Select
      value={trainingMethod}
      onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
    >
      <SelectTrigger
        animateRadius={false}
        icon={ArrowDown01Icon}
        iconStrokeWidth={1.25}
        iconClassName="size-3.5"
        className={cn(TRIGGER_BASE, "w-full min-w-[148px] justify-between")}
        data-tour="studio-method"
      >
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className={cn(
              "size-2 shrink-0 rounded-full",
              TRAINING_METHOD_DOTS[trainingMethod] ?? "bg-muted-foreground",
            )}
          />
          <span className="truncate font-medium text-foreground">
            {TRAINING_METHOD_LABELS[trainingMethod] ?? trainingMethod}
          </span>
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        avoidCollisions={false}
        className="menu-instant menu-soft-surface rounded-[14px] ring-0"
      >
        {(["qlora", "lora", "full", "cpt"] as TrainingMethod[]).map(
          (method) => (
            <Tooltip key={method} delayDuration={300}>
              <TooltipTrigger asChild={true}>
                <SelectItem value={method}>
                  <span className="flex items-center gap-2">
                    <span
                      aria-hidden="true"
                      className={cn(
                        "size-2 shrink-0 rounded-full",
                        TRAINING_METHOD_DOTS[method],
                      )}
                    />
                    {TRAINING_METHOD_DESCRIPTIONS[method]}
                  </span>
                </SelectItem>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                sideOffset={10}
                className="max-w-[220px] text-[11.5px] leading-snug"
              >
                {TRAINING_METHOD_HINTS[method]}
              </TooltipContent>
            </Tooltip>
          ),
        )}
      </SelectContent>
    </Select>
  );
}
