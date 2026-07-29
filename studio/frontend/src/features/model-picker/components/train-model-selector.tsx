// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { findCanonicalHubResourceId } from "@/components/resource-picker/hub-resource-id";
import {
  type PickerExactQueryCommitResult,
  PickerShell,
} from "@/components/resource-picker/picker-shell";
import { useHfErrorToast } from "@/components/resource-picker/use-hf-error-toast";
import { usePickerHubPagination } from "@/components/resource-picker/use-picker-hub-pagination";
import { usePickerState } from "@/components/resource-picker/use-picker-state";
import { usePlatformStore } from "@/config/env";
import {
  PRIORITY_TRAINING_MODELS,
  applyPriorityOrdering,
} from "@/config/training";
import {
  type CachedInventoryRow,
  type HfModelResult,
  type LocalInventoryRow,
  type ModelInventoryFormat,
  classifyUnslothSupport,
  hfApiToken,
  matchTokens,
  normalizeModelIdentity,
  tokenizeQuery,
  useHfTokenStore,
  useHubInfiniteScroll,
  useHubInventory,
  useHubModelSearch,
  useOnlineStatus,
} from "@/features/hub";
import {
  type ModelTypeCapabilityFlags,
  buildCachedTrainingModelLookup,
  buildLocalTrainingModelLookup,
  inferTrainingModelTypeFromFlags,
  isLocalTrainingModelSelection,
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
  type TrainingMethod as VramTrainingMethod,
  buildModelVramMap,
} from "@/lib/vram";
import { ArrowDown01Icon, ChipIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  toCachedTrainModelDisplayCandidate,
  toTrainModelDisplayCandidate,
  trainModelDisplayCandidateMatchesSelection,
  trainModelSelectionDisplayName,
} from "../lib/train-model-selection-display";
import {
  TrainModelDeviceList,
  TrainModelHubList,
  type TrainModelVramView,
} from "./train-model-picker-lists";
import {
  type TrainModelDeviceItem,
  compareTrainModelDeviceItems,
  hasExactTrainingModelMatch,
  hubTrainingModelCandidate,
  hubTrainingModelTypeFlags,
  resolveExactTrainModelDeviceItem,
  toCachedTrainModelDeviceItem,
  toLocalTrainModelDeviceItem,
} from "./train-model-picker-view-model";
import { TRAIN_PICKER_TRIGGER_CLASS } from "./train-picker-trigger";

const MODEL_PICKER_TAB_STORAGE_KEY = "unsloth.studio.train.modelPickerTab";

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
    trainingMethod,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      modelKnownCached: s.modelKnownCached,
      modelLocalPath: s.modelLocalPath,
      modelFormat: s.modelFormat,
      selectTrainingModel: s.selectTrainingModel,
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
        ...cachedRows
          .filter(isTrainableCachedRow)
          .map((row) =>
            toCachedTrainModelDeviceItem(
              row,
              t("studio.modelPicker.hfCacheLabel"),
            ),
          ),
        ...localRows
          .filter(isTrainableLocalRow)
          .map(toLocalTrainModelDeviceItem),
      ].sort(compareTrainModelDeviceItems),
    [cachedRows, localRows, isTrainableCachedRow, isTrainableLocalRow, t],
  );
  const localModelDisplayCandidates = useMemo(
    () => [
      ...cachedRows.map(toCachedTrainModelDisplayCandidate),
      ...localRows.map(toTrainModelDisplayCandidate),
    ],
    [cachedRows, localRows],
  );

  const pickerView = picker.getViewState({
    hasDeviceItems: trainableLocalModels.length > 0,
    isLoadingDevice: isLoadingLocalModels,
  });
  const { activeQuery, handleOpenChange, handleQueryChange, tab } = pickerView;

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
      trainModelDisplayCandidateMatchesSelection({
        candidate: item,
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
    scannedCount: scannedHfCount,
    hasMore: hasMoreHf,
  } = useHubModelSearch(picker.debouncedHubQuery, {
    accessToken: hfApiToken(picker.debouncedHfToken),
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
    ownerScope: picker.debouncedHubQuery.trim() ? "all" : "unsloth",
    enabled: online && picker.open && tab === "hub",
  });

  const hubSearchActive = online && picker.open && tab === "hub";
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
    const map = new Map<string, TrainModelVramView>();
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

  const hubPagination = usePickerHubPagination({
    enabled: hubSearchActive,
    fetchMore: fetchMoreHf,
    hasMore: hasMoreHf,
    isFetching: isLoadingHf || isLoadingHfMore,
    resetKey: picker.debouncedHubQuery,
    resultCount: hfResults.length,
    scannedCount: scannedHfCount,
  });

  const { scrollRef, sentinelRef } = useHubInfiniteScroll(
    hubPagination.fetchMore,
    hubPagination.signal,
    hubPagination.options,
  );

  function pick(
    id: string,
    options: {
      knownCached?: boolean;
      localPath?: string | null;
      modelFormat?: ModelInventoryFormat | null;
    },
    inferredFlags: ModelTypeCapabilityFlags,
  ) {
    const next = id.trim();
    if (!next) {
      return;
    }
    selectTrainingModel(
      next,
      inferTrainingModelTypeFromFlags(inferredFlags),
      options,
    );
    picker.closePicker();
  }

  function pickHubModel(id: string) {
    const key = normalizeModelIdentity(id);
    const result = hfResultById.get(key);
    const cached = cachedModelByLookup.get(key);
    const local = localModelByLookup.get(key);
    const cachedLocal = local?.source === "hf_cache" ? local : undefined;
    const canonicalId =
      result?.id || cached?.repoId || cachedLocal?.repoId?.trim() || id.trim();
    const validation = validateTrainingModelCandidate(
      hubTrainingModelCandidate(canonicalId, result, cached, cachedLocal),
      { deviceType },
    );
    if (!validation.ok) {
      toast.error(t("studio.modelPicker.cantUseModel"), {
        description: validation.reasonText ?? t(validation.reasonKey),
      });
      return;
    }
    const inferredFlags = hubTrainingModelTypeFlags(
      canonicalId,
      result,
      cached,
      cachedLocal,
    );
    pick(
      canonicalId,
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
    const key = normalizeModelIdentity(id);
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

  function pickDeviceModel(model: TrainModelDeviceItem) {
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
  }

  function commitExactQuery(query: string): PickerExactQueryCommitResult {
    if (tab === "hub") {
      const canonicalId = findCanonicalHubResourceId(query, hubResultIds);
      if (!canonicalId) {
        return { kind: "unhandled" };
      }
      pickHubModel(canonicalId);
      return { kind: "handled" };
    }
    const resolution = resolveExactTrainModelDeviceItem(
      query,
      trainableLocalModels,
    );
    if (resolution.kind === "ambiguous") {
      return {
        kind: "ambiguous",
        focusValue: resolution.firstItem.path,
      };
    }
    if (resolution.kind === "none") {
      return { kind: "unhandled" };
    }
    pickDeviceModel(resolution.item);
    return { kind: "handled" };
  }

  const display = useMemo(
    () =>
      trainModelSelectionDisplayName({
        selectedModel,
        knownCached: modelKnownCached,
        selectedLocalPath: modelLocalPath,
        selectedFormat: selectedModelFormat,
        candidates: localModelDisplayCandidates,
      }),
    [
      modelKnownCached,
      modelLocalPath,
      selectedModel,
      selectedModelFormat,
      localModelDisplayCandidates,
    ],
  );
  const hasExactMatch = hasExactTrainingModelMatch(
    activeQuery,
    tab,
    hubResultIds,
    trainableLocalModels,
  );
  const showUseThis = activeQuery.trim().length > 0 && !hasExactMatch;
  const useThisLabel =
    tab === "hub"
      ? t("studio.modelPicker.useAsHubModel")
      : t("studio.modelPicker.useAsLocalPath");

  return (
    <PickerShell
      open={picker.open}
      onOpenChange={handleOpenChange}
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
      onExactQueryCommit={commitExactQuery}
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
        <TrainModelDeviceList
          items={filteredLocalModels}
          isLoading={isLoadingLocalModels}
          error={localModelsError}
          warning={inventoryWarning}
          activeKey={selectedDeviceItemKey}
          hasQuery={activeQuery.length > 0}
          onPick={pickDeviceModel}
          onRetry={retryLocalModels}
        />
      }
      hubContent={
        <TrainModelHubList
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
