// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
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
import {
  MODEL_TYPE_TO_HF_TASK,
  PRIORITY_TRAINING_MODELS,
  applyPriorityOrdering,
} from "@/config/training";
import {
  type LocalModelInfo,
  listLocalModels,
  useTrainingConfigStore,
} from "@/features/training";
import {
  extractParamLabel,
  modelShortName,
  useInventoryVersion,
} from "@/features/models";
import { usePlatformStore } from "@/config/env";
import {
  classifyUnslothSupport,
  useDebouncedValue,
  useGpuInfo,
  useHfModelSearch,
  useInfiniteScroll,
} from "@/hooks";
import { cn, formatCompact } from "@/lib/utils";
import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import {
  type VramFitStatus,
  type TrainingMethod as VramTrainingMethod,
  buildModelVramMap,
} from "@/lib/vram";
import { useHfTokenStore } from "@/stores/hf-token-store";
import type { TrainingMethod } from "@/types/training";
import {
  TRAINING_METHOD_DESCRIPTIONS,
  TRAINING_METHOD_DOTS,
  TRAINING_METHOD_HINTS,
  TRAINING_METHOD_LABELS,
} from "@/features/training/lib/training-method-meta";
import {
  ArrowDown01Icon,
  ArrowRight01Icon,
  ChipIcon,
  FolderSearchIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  PICKER_TABS,
  type PickerTab,
  PickerTabToggle,
  RetryButton,
  isHfAuthError,
  looksLikeLocalPath,
} from "./picker-tab-toggle";
import { useHfErrorToast } from "./use-hf-error-toast";

const TRIGGER_BASE = cn(
  "menu-trigger field-soft inline-flex h-9 items-center gap-1.5 rounded-[12px] px-3 text-[12.5px] text-muted-foreground transition-colors",
  "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
);

export function TrainModelPicker() {
  const gpu = useGpuInfo();
  const { selectedModel, setSelectedModel, modelType, trainingMethod } =
    useTrainingConfigStore(
      useShallow((s) => ({
        selectedModel: s.selectedModel,
        setSelectedModel: s.setSelectedModel,
        modelType: s.modelType,
        trainingMethod: s.trainingMethod,
      })),
    );
  const hfToken = useHfTokenStore((s) => s.token);

  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<PickerTab>("hub");
  const [hubQuery, setHubQuery] = useState("");
  const [deviceQuery, setDeviceQuery] = useState("");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(true);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);
  const [localRetryToken, setLocalRetryToken] = useState(0);
  const inventoryVersion = useInventoryVersion();
  const loadedInventoryVersionRef = useRef<number>(-1);

  const debouncedHubQuery = useDebouncedValue(hubQuery);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const task = modelType ? MODEL_TYPE_TO_HF_TASK[modelType] : undefined;

  const {
    results: hfResults,
    isLoading: isLoadingHf,
    isLoadingMore: isLoadingHfMore,
    fetchMore: fetchMoreHf,
    retry: retryHf,
    error: hfError,
  } = useHfModelSearch(debouncedHubQuery, {
    task,
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
    enabled: open && tab === "hub",
  });

  useHfErrorToast(hfError, "models");

  useEffect(() => {
    if (!open || tab !== "device") return;
    if (loadedInventoryVersionRef.current === inventoryVersion) return;
    const controller = new AbortController();
    setIsLoadingLocalModels(true);
    setLocalModelsError(null);
    void listLocalModels(controller.signal)
      .then((models) => {
        if (controller.signal.aborted) return;
        setLocalModels(models);
        loadedInventoryVersionRef.current = inventoryVersion;
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setLocalModelsError(
          err instanceof Error ? err.message : "Couldn't scan local models",
        );
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, [open, tab, inventoryVersion, localRetryToken]);

  const retryLocalModels = useCallback(() => {
    loadedInventoryVersionRef.current = -1;
    setLocalRetryToken((token) => token + 1);
  }, []);

  const deviceType = usePlatformStore((s) => s.deviceType);
  const trainableLocalModels = useMemo(
    () =>
      localModels.filter((m) => {
        if (m.partial) return false;
        // GGUF + LM Studio: never trainable in Unsloth.
        if (m.source === "lmstudio") return false;
        if (m.path.toLowerCase().endsWith(".gguf")) return false;
        if (/(?:^|[-_./])gguf(?:$|[-_./])/i.test(m.id)) return false;
        if (/(?:^|[-_./])gguf(?:$|[-_./])/i.test(m.path)) return false;
        // Mirror Hub's classifier: format-detect (mlx on non-Mac, awq, gptq,
        // exl2, onnx, openvino, coreml, tflite, ctranslate2, …) using both
        // the model id and the on-disk basename so paths like
        // "/models/llama-3-mlx" are caught even when the id is just "llama-3".
        const pathBase = m.path.split(/[/\\]/).pop() ?? "";
        for (const candidate of [m.id, pathBase, m.model_id ?? ""]) {
          if (!candidate) continue;
          if (
            classifyUnslothSupport({ modelId: candidate, deviceType })
              .status === "unsupported"
          ) {
            return false;
          }
        }
        return true;
      }),
    [localModels, deviceType],
  );

  const filteredLocalModels = useMemo(() => {
    const tokens = tokenizeQuery(deviceQuery);
    if (tokens.length === 0) return trainableLocalModels;
    return trainableLocalModels.filter((m) =>
      matchTokens(`${m.id} ${m.display_name} ${m.path}`, tokens),
    );
  }, [trainableLocalModels, deviceQuery]);

  const hubResultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (
      selectedModel &&
      !looksLikeLocalPath(selectedModel) &&
      !ids.includes(selectedModel)
    ) {
      ids.push(selectedModel);
    }
    return applyPriorityOrdering(ids);
  }, [hfResults, selectedModel]);

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

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMoreHf,
    hfResults.length,
    open && tab === "hub",
  );

  function pick(id: string) {
    const next = id.trim();
    if (!next) return;
    setSelectedModel(next);
    setOpen(false);
  }

  const display = selectedModel ? modelShortName(selectedModel) : null;
  const activeQuery = (tab === "hub" ? hubQuery : deviceQuery).trim();
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
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild={true}>
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
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        collisionPadding={16}
        className="w-[min(420px,calc(100vw-2rem))] rounded-2xl p-4"
      >
        <PickerTabToggle
          tab={tab}
          options={PICKER_TABS}
          onTabChange={setTab}
        />
        <div className="mt-2.5 flex flex-col gap-2">
          <div className="relative">
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={1.75}
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              autoFocus={true}
              value={tab === "hub" ? hubQuery : deviceQuery}
              onChange={(e) =>
                tab === "hub"
                  ? setHubQuery(e.target.value)
                  : setDeviceQuery(e.target.value)
              }
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                e.preventDefault();
                if (showUseThis) pick(activeQuery);
              }}
              placeholder={
                tab === "hub"
                  ? "Search or paste a Hugging Face id..."
                  : "Search local models or paste a folder path..."
              }
              className="field-soft h-9 rounded-full pl-9 text-[12.5px]"
            />
            {tab === "hub" && isLoadingHf && (
              <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            )}
          </div>

          <div
            ref={scrollRef}
            className="max-h-[320px] overflow-y-auto overscroll-contain rounded-[10px] [scrollbar-width:thin]"
          >
            {showUseThis && (
              <button
                type="button"
                onClick={() => pick(activeQuery)}
                className="mb-1 flex w-full items-center gap-2 rounded-[8px] border border-dashed border-primary/30 bg-primary/[0.04] px-2.5 py-2 text-left text-[12.5px] transition-colors hover:bg-primary/[0.08]"
              >
                <HugeiconsIcon
                  icon={tab === "hub" ? Search01Icon : FolderSearchIcon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0 text-primary"
                />
                <span className="flex min-w-0 flex-1 flex-col leading-tight">
                  <span className="truncate font-medium text-foreground">
                    {activeQuery}
                  </span>
                  <span className="text-[10.5px] text-muted-foreground/80">
                    {useThisLabel}
                  </span>
                </span>
                <HugeiconsIcon
                  icon={ArrowRight01Icon}
                  strokeWidth={1.5}
                  className="size-3.5 shrink-0 text-muted-foreground/70"
                />
              </button>
            )}
            {tab === "device" ? (
              <DeviceList
                items={filteredLocalModels}
                isLoading={isLoadingLocalModels}
                error={localModelsError}
                value={selectedModel}
                hasQuery={activeQuery.length > 0}
                onPick={pick}
                onRetry={retryLocalModels}
              />
            ) : (
              <HubList
                ids={hubResultIds}
                value={selectedModel}
                vramMap={vramMap}
                isLoading={isLoadingHf}
                isLoadingMore={isLoadingHfMore}
                gpuTotalGb={gpu.available ? gpu.memoryTotalGb : null}
                hasQuery={activeQuery.length > 0}
                error={hfError}
                onPick={pick}
                onRetry={retryHf}
                sentinelRef={sentinelRef}
              />
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function DeviceList({
  items,
  isLoading,
  error,
  value,
  hasQuery,
  onPick,
  onRetry,
}: {
  items: LocalModelInfo[];
  isLoading: boolean;
  error: string | null;
  value: string | null;
  hasQuery: boolean;
  onPick: (id: string) => void;
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
    if (hasQuery) return null;
    return (
      <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
        <HugeiconsIcon
          icon={FolderSearchIcon}
          strokeWidth={1.5}
          className="size-5 text-muted-foreground/70"
        />
        <p className="text-xs text-muted-foreground">
          No local models found.
        </p>
        <p className="text-[10.5px] text-muted-foreground/70">
          Paste a folder path above or switch to Hugging Face.
        </p>
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {items.map((m) => (
        <li key={m.id}>
          <button
            type="button"
            onClick={() => onPick(m.id)}
            className={cn(
              "flex w-full items-center gap-2 rounded-[8px] px-2 py-1.5 text-left text-[12.5px] transition-colors hover:bg-foreground/[0.05]",
              value === m.id && "bg-foreground/[0.06]",
            )}
          >
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <span className="block min-w-0 flex-1 truncate">
                  {m.display_name || m.id}
                </span>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs break-all">
                {m.path}
              </TooltipContent>
            </Tooltip>
            <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
              {m.source === "hf_cache"
                ? "HF cache"
                : m.source === "custom"
                  ? "Custom"
                  : "Local"}
            </span>
          </button>
        </li>
      ))}
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
  sentinelRef: React.RefObject<HTMLDivElement | null>;
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
            {isAuth ? "Hugging Face token rejected" : "Couldn't reach Hugging Face"}
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
    if (hasQuery) return null;
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
            <button
              type="button"
              onClick={() => onPick(id)}
              className={cn(
                "flex w-full items-center gap-2 rounded-[8px] px-2 py-1.5 text-left text-[12.5px] transition-colors hover:bg-foreground/[0.05]",
                value === id && "bg-foreground/[0.06]",
              )}
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
            </button>
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
