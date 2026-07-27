// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PickerShell } from "@/components/resource-picker/picker-shell";
import { isHfAuthError } from "@/components/resource-picker/picker-tab-state";
import { RetryButton } from "@/components/resource-picker/picker-tab-toggle";
import { SelectablePickerItem } from "@/components/resource-picker/selectable-picker-item";
import { useHfErrorToast } from "@/components/resource-picker/use-hf-error-toast";
import { usePickerState } from "@/components/resource-picker/use-picker-state";
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
  type HfModelResult,
  type LocalInventoryRow,
  type LocalSource,
  type ModelInventoryFormat,
  classifyUnslothSupport,
  hfApiToken,
  matchTokens,
  repoOf,
  tokenizeQuery,
  useHfTokenStore,
  useHubInfiniteScroll,
  useHubInventory,
  useHubModelSearch,
  useLatestRef,
  useOnlineStatus,
} from "@/features/hub";
import {
  type ModelTypeCapabilityFlags,
  buildCachedTrainingModelLookup,
  buildLocalTrainingModelLookup,
  cacheLocalPathMatchesSelection,
  isLocalTrainingModelSelection,
  resolvePickerInferredModelType,
  trainingModelTypeFlagsFromMetadata,
  useTrainingConfigStore,
  validateTrainingModelCandidate,
} from "@/features/training";
import { useGpuInfo } from "@/hooks";
import { useT } from "@/i18n";
import { extractParamLabel } from "@/lib/model-size";
import { toast } from "@/lib/toast";
import { cn, formatCompact } from "@/lib/utils";
import {
  type VramFitStatus,
  type TrainingMethod as VramTrainingMethod,
  buildModelVramMap,
} from "@/lib/vram";
import {
  ArrowDown01Icon,
  ChipIcon,
  FolderSearchIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type RefObject, useCallback, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { TRAIN_PICKER_TRIGGER_CLASS } from "./train-picker-trigger";

const MODEL_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.modelPickerTab";

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
  modelTypeFlags: ModelTypeCapabilityFlags;
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

function hubTrainingModelCandidate(
  id: string,
  result: HfModelResult | undefined,
  cached: CachedInventoryRow | undefined,
  cachedLocal: LocalInventoryRow | undefined,
): Parameters<typeof validateTrainingModelCandidate>[0] {
  return {
    id,
    modelFormat: cached?.modelFormat ?? cachedLocal?.modelFormat ?? null,
    capabilities: cached?.capabilities ?? cachedLocal?.capabilities ?? null,
    pipelineTag:
      result?.pipelineTag ?? cached?.pipelineTag ?? cachedLocal?.pipelineTag,
    tags: result?.tags ?? cached?.tags ?? cachedLocal?.tags,
    libraryName:
      result?.libraryName ?? cached?.libraryName ?? cachedLocal?.libraryName,
    quantMethod:
      result?.quantMethod ?? cached?.quantMethod ?? cachedLocal?.quantMethod,
  };
}

function hubTrainingModelTypeFlags(
  id: string,
  result: HfModelResult | undefined,
  cached: CachedInventoryRow | undefined,
  cachedLocal: LocalInventoryRow | undefined,
): ModelTypeCapabilityFlags {
  if (result) {
    return trainingModelTypeFlagsFromMetadata({
      tags: result.tags,
      pipelineTag: result.pipelineTag,
      identifiers: [result.id],
    });
  }
  if (cached) {
    return trainingModelTypeFlagsFromMetadata({
      tags: cached.tags,
      pipelineTag: cached.pipelineTag,
      identifiers: [cached.repoId, cached.repo],
    });
  }
  if (cachedLocal) {
    return trainingModelTypeFlagsFromMetadata({
      tags: cachedLocal.tags,
      pipelineTag: cachedLocal.pipelineTag,
      identifiers: [
        cachedLocal.repoId,
        cachedLocal.loadId,
        cachedLocal.title,
        cachedLocal.path,
      ],
    });
  }
  return trainingModelTypeFlagsFromMetadata({ identifiers: [id] });
}

function hasExactTrainingModelMatch(
  query: string,
  tab: "device" | "hub",
  hubIds: readonly string[],
  deviceItems: readonly TrainModelDeviceItem[],
): boolean {
  if (!query) {
    return false;
  }
  if (tab === "hub") {
    return hubIds.some((id) => id === query);
  }
  return deviceItems.some((item) => item.id === query || item.path === query);
}

export function TrainModelSelector({
  triggerDataTour = "studio-model-picker",
}: {
  triggerDataTour?: string;
}) {
  const t = useT();
  const gpu = useGpuInfo();
  const {
    selectedModel,
    modelKnownCached,
    modelLocalPath,
    modelFormat: selectedModelFormat,
    selectTrainingModel,
    modelType,
    trainingMethod,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      modelKnownCached: s.modelKnownCached,
      modelLocalPath: s.modelLocalPath,
      modelFormat: s.modelFormat,
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
  const picker = usePickerState({
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
      ? t("studio.modelPicker.couldntScan")
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

  const trainableLocalModels = useMemo<TrainModelDeviceItem[]>(
    () =>
      [
        ...cachedRows.filter(isTrainableCachedRow).map((row) => ({
          key: row.id,
          id: row.loadId,
          title: row.repoId,
          path: row.cachePath ?? row.repoId,
          source: "hf_cache" as const,
          sourceLabel: t("studio.modelPicker.hfCacheLabel"),
          knownCached: true,
          localPath: row.cachePath ?? null,
          modelFormat: row.modelFormat,
          modelTypeFlags: trainingModelTypeFlagsFromMetadata({
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
          modelTypeFlags: trainingModelTypeFlagsFromMetadata({
            tags: row.tags,
            pipelineTag: row.pipelineTag,
            identifiers: [row.repoId, row.loadId, row.title, row.path],
          }),
        })),
      ].sort(compareTrainModelDeviceItems),
    [cachedRows, localRows, isTrainableCachedRow, isTrainableLocalRow, t],
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
  } = useHubModelSearch(picker.debouncedHubQuery, {
    task,
    accessToken: hfApiToken(picker.debouncedHfToken),
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
    ownerScope: picker.debouncedHubQuery.trim() ? "all" : "unsloth",
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
      !isLocalTrainingModelSelection({
        model: selectedModel,
        knownCached: modelKnownCached,
        localPath: modelLocalPath,
      }) &&
      !seen.has(selectedModel.toLowerCase())
    ) {
      ids.push(selectedModel);
    }
    return applyPriorityOrdering(ids);
  }, [hfResults, modelKnownCached, modelLocalPath, selectedModel]);

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
  }, [hubSearchActiveRef, fetchMoreHfRef]);

  const { scrollRef, sentinelRef } = useHubInfiniteScroll(
    fetchMoreOpenHf,
    hfResults.length,
    { enabled: hubSearchActive },
  );

  function pick(
    id: string,
    options?: {
      knownCached?: boolean;
      localPath?: string | null;
      modelFormat?: ModelInventoryFormat | null;
    },
    inferredFlags?: ModelTypeCapabilityFlags | null,
  ) {
    const next = id.trim();
    if (!next) {
      return;
    }
    const nextModelType = inferredFlags
      ? resolvePickerInferredModelType(modelType, inferredFlags)
      : modelType;
    selectTrainingModel(next, nextModelType ?? null, options);
    picker.closePicker();
  }

  function pickHubModel(id: string) {
    const key = id.trim().toLowerCase();
    const result = hfResultById.get(key);
    const cached = cachedModelByLookup.get(key);
    const local = localModelByLookup.get(key);
    const cachedLocal = local?.source === "hf_cache" ? local : undefined;
    const validation = validateTrainingModelCandidate(
      hubTrainingModelCandidate(id, result, cached, cachedLocal),
      { deviceType },
    );
    if (!validation.ok) {
      toast.error(t("studio.modelPicker.cantUseModel"), {
        description: validation.reasonText ?? t(validation.reasonKey),
      });
      return;
    }
    const inferredFlags = hubTrainingModelTypeFlags(
      id,
      result,
      cached,
      cachedLocal,
    );
    pick(
      id,
      {
        knownCached: cached !== undefined || cachedLocal !== undefined,
        localPath: cached?.cachePath ?? cachedLocal?.path ?? null,
        modelFormat: cached?.modelFormat ?? cachedLocal?.modelFormat ?? null,
      },
      inferredFlags,
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
          validationCached?.capabilities ??
          validationLocal?.capabilities ??
          null,
        pipelineTag:
          validationCached?.pipelineTag ?? validationLocal?.pipelineTag,
        tags: validationCached?.tags ?? validationLocal?.tags,
        libraryName:
          validationCached?.libraryName ?? validationLocal?.libraryName,
        quantMethod:
          validationCached?.quantMethod ?? validationLocal?.quantMethod,
      },
      { deviceType },
    );
    if (!validation.ok) {
      toast.error(t("studio.modelPicker.cantUseModel"), {
        description: validation.reasonText ?? t(validation.reasonKey),
      });
      return;
    }
    pick(
      id,
      { knownCached: false, localPath: id, modelFormat: null },
      trainingModelTypeFlagsFromMetadata({ identifiers: [id] }),
    );
  }

  const display = selectedModel ? repoOf(selectedModel) : null;
  const hasExactMatch = hasExactTrainingModelMatch(
    activeQuery,
    tab,
    hubResultIds,
    trainableLocalModels,
  );
  const showUseThis = activeQuery.length > 0 && !hasExactMatch;
  const useThisLabel =
    tab === "hub"
      ? t("studio.modelPicker.useAsHubModel")
      : t("studio.modelPicker.useAsLocalPath");

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
      noun={t("studio.modelPicker.noun")}
      isHubLoading={isLoadingHf}
      showUseThis={showUseThis}
      useThisLabel={useThisLabel}
      onUseThis={() => pickFreeformModel(activeQuery)}
      placeholder={{
        hub: t("studio.modelPicker.hubPlaceholder"),
        device: t("studio.modelPicker.devicePlaceholder"),
      }}
      scrollRef={scrollRef}
      trigger={
        <button
          type="button"
          data-tour={triggerDataTour}
          className={cn(
            TRAIN_PICKER_TRIGGER_CLASS,
            "w-full min-w-[180px] justify-between",
          )}
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
              {display ?? t("studio.modelPicker.selectModel")}
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
              model.modelTypeFlags,
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
  const t = useT();
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.modelPicker.scanningLocal")}
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-ui-12p5 font-medium text-foreground">
            {t("studio.modelPicker.couldntScan")}
          </p>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
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
        <p className="text-xs text-muted-foreground">
          {t("studio.modelPicker.noLocalModels")}
        </p>
        <p className="text-ui-10p5 text-muted-foreground/70">
          {t("studio.modelPicker.noLocalModelsHint")}
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
            values={[m.id, m.path]}
          >
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <span className="block min-w-0 flex-1 truncate">
                  {m.title || m.id}
                </span>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs break-all">
                {m.path}
              </TooltipContent>
            </Tooltip>
            <span className="ml-2 shrink-0 rounded-[6px] border border-border/60 px-1.5 py-0.5 text-ui-10 font-medium leading-none text-muted-foreground">
              {m.sourceLabel}
            </span>
          </SelectablePickerItem>
        </li>
      ))}
      {warning && (
        <li className="px-2 py-1 text-ui-10p5 text-muted-foreground/80">
          {t("studio.modelPicker.someLocationsUnscanned")}
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
  const t = useT();
  if (isLoading && ids.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.modelPicker.searchingHub")}
      </div>
    );
  }
  if (ids.length === 0) {
    if (error) {
      const isAuth = isHfAuthError(error);
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-ui-12p5 font-medium text-foreground">
            {isAuth
              ? t("studio.modelPicker.tokenRejectedTitle")
              : t("studio.modelPicker.hubUnreachable")}
          </p>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {isAuth ? t("studio.modelPicker.tokenRejectedBody") : error}
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
        {t("studio.modelPicker.noModelsFound")}
      </div>
    );
  }
  return (
    <>
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
                values={[id]}
              >
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <span
                      className={cn(
                        "block min-w-0 flex-1 truncate",
                        exceeds && "text-muted-foreground",
                      )}
                    >
                      {id}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="max-w-xs break-all">
                    {id}
                    {fit && fit.est > 0 && gpuTotalGb != null && (
                      <span className="mt-1 block text-ui-10">
                        {exceeds
                          ? t("studio.modelPicker.vramNeeds", {
                              est: fit.est,
                              total: gpuTotalGb,
                            })
                          : tight
                            ? t("studio.modelPicker.vramTight", {
                                est: fit.est,
                                total: gpuTotalGb,
                              })
                            : t("studio.modelPicker.vramApprox", {
                                est: fit.est,
                              })}
                      </span>
                    )}
                  </TooltipContent>
                </Tooltip>
                <span className="ml-auto flex shrink-0 items-center gap-1.5">
                  {exceeds && (
                    <span className="rounded bg-red-50 px-1.5 py-0.5 text-ui-9 font-semibold text-red-700 dark:bg-red-950 dark:text-red-400">
                      OOM
                    </span>
                  )}
                  {tight && (
                    <span className="text-ui-9 font-semibold text-amber-500">
                      TIGHT
                    </span>
                  )}
                  {fit?.detail && (
                    <span className="text-ui-10 text-muted-foreground">
                      {fit.detail}
                    </span>
                  )}
                </span>
              </SelectablePickerItem>
            </li>
          );
        })}
      </ul>
      <div ref={sentinelRef} className="h-px" />
      {isLoadingMore && (
        <div className="flex items-center justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      )}
    </>
  );
}
