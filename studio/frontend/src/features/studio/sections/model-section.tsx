// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
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
  useDebouncedValue,
  useGpuInfo,
  useHfTokenValidation,
} from "@/hooks";
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import { useHubInfiniteScroll } from "@/features/hub/hooks/use-hub-infinite-scroll";
import { extractParamLabel } from "@/lib/model-size";
import { formatCompact } from "@/lib/utils";
import {
  type VramFitStatus,
  type TrainingMethod as VramTrainingMethod,
  buildModelVramMap,
} from "@/lib/vram";
import type { TrainingMethod } from "@/types/training";
import {
  ChipIcon,
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { translate, useT } from "@/i18n";

const METHOD_DOTS: Record<string, string> = {
  qlora: "bg-emerald-400",
  lora: "bg-blue-400",
  full: "bg-amber-400",
  cpt: "bg-purple-400",
};

const DARK_TRIGGER =
  "w-full bg-foreground text-background hover:bg-foreground/90 dark:bg-foreground dark:text-background dark:hover:bg-foreground dark:has-[[data-slot=input-group-control]:focus-visible]:bg-foreground [&_svg]:text-background/50";
const DARK_CONTENT =
  "bg-foreground text-background shadow-xl border-background/10 [--accent:rgba(255,255,255,0.1)] [--accent-foreground:white] dark:[--accent:rgba(2,6,23,0.08)] dark:[--accent-foreground:rgb(2,6,23)] [&_[data-slot=select-item]]:text-white/80 dark:[&_[data-slot=select-item]]:text-slate-900 [&_[data-slot=select-scroll-up-button]]:bg-foreground [&_[data-slot=select-scroll-down-button]]:bg-foreground";
const DARK_COMBOBOX_CONTENT =
  "bg-foreground text-background shadow-xl border-background/10 [--accent:rgba(255,255,255,0.1)] [--accent-foreground:white] dark:[--accent:rgba(2,6,23,0.08)] dark:[--accent-foreground:rgb(2,6,23)] dark:[&_[data-slot=combobox-item]]:text-slate-900 dark:[&_.text-muted-foreground]:text-slate-500";

export function ModelSection() {
  const t = useT();
  const gpu = useGpuInfo();

  const {
    modelType,
    selectedModel,
    setSelectedModel,
    trainingMethod,
    setTrainingMethod,
    hfToken,
    setHfToken,
  } = useTrainingConfigStore(
    useShallow(
      ({
        modelType,
        selectedModel,
        setSelectedModel,
        trainingMethod,
        setTrainingMethod,
        hfToken,
        setHfToken,
      }) => ({
        modelType,
        selectedModel,
        setSelectedModel,
        trainingMethod,
        setTrainingMethod,
        hfToken,
        setHfToken,
      }),
    ),
  );

  const [inputValue, setInputValue] = useState("");
  const [localModelInput, setLocalModelInput] = useState("");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(true);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);
  const selectingRef = useRef(false);
  const debouncedQuery = useDebouncedValue(inputValue);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);

  function handleModelSelect(id: string | null) {
    selectingRef.current = true;
    setSelectedModel(id);
  }

  function handleInputChange(val: string) {
    if (selectingRef.current) {
      selectingRef.current = false;
      return;
    }
    setInputValue(val);
  }

  function applyLocalModel(value: string) {
    const next = value.trim();
    if (!next) return;
    setSelectedModel(next);
  }

  useEffect(() => {
    const controller = new AbortController();
    void listLocalModels(controller.signal)
      .then((models) => {
        if (controller.signal.aborted) return;
        setLocalModels(models);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setLocalModelsError(
          error instanceof Error
            ? error.message
            : translate("studio.model.failedToLoadLocalModels"),
        );
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, []);
  const task = modelType ? MODEL_TYPE_TO_HF_TASK[modelType] : undefined;
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    scannedCount,
    error: hfSearchError,
  } = useHubModelSearch(debouncedQuery, {
    task,
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
    // Curated unsloth listing by default, but a typed query searches the whole
    // Hub (unsloth floated first) so non-unsloth base models stay selectable.
    ownerScope: debouncedQuery.trim() ? "all" : "unsloth",
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const resultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (selectedModel && !ids.includes(selectedModel)) {
      ids.push(selectedModel);
    }

    return applyPriorityOrdering(ids);
  }, [hfResults, selectedModel]);

  // Filter out GGUF models — they can't be used for training
  const trainableLocalModels = useMemo(
    () =>
      localModels.filter((m) => {
        if (m.source === "lmstudio") return false;
        if (m.path.endsWith(".gguf")) return false;
        if (m.id.toLowerCase().includes("-gguf")) return false;
        return true;
      }),
    [localModels],
  );

  const localMetaById = useMemo(() => {
    const map = new Map<string, LocalModelInfo>();
    for (const model of trainableLocalModels) map.set(model.id, model);
    return map;
  }, [trainableLocalModels]);

  const localResultIds = useMemo(() => {
    const ids = trainableLocalModels.map((model) => model.id);
    const manual = localModelInput.trim();
    if (manual && !ids.includes(manual)) {
      ids.unshift(manual);
    }
    return ids;
  }, [localModelInput, localModels]);

  const localFilteredIds = useMemo(() => {
    const q = localModelInput.trim().toLowerCase();
    if (!q) return localResultIds;
    return localResultIds.filter((id) => {
      const meta = localMetaById.get(id);
      if (id.toLowerCase().includes(q)) return true;
      if (meta?.display_name.toLowerCase().includes(q)) return true;
      if (meta?.path.toLowerCase().includes(q)) return true;
      return false;
    });
  }, [localMetaById, localModelInput, localResultIds]);

  // VRAM fit status per model, keyed by id for O(1) render lookups.
  // Recomputes on training-method change (QLoRA=4-bit vs LoRA/Full=fp16).
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

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const localComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useHubInfiniteScroll(fetchMore, scannedCount, {
    isFetching: isLoading || isLoadingMore,
    resultCount: hfResults.length,
    resetKey: debouncedQuery,
  });

  return (
    <div data-tour="studio-model" className="w-full min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={ChipIcon} className="size-5" />}
        title={t("studio.model.title")}
        description={t("studio.model.description")}
        accent="emerald"
        featured={true}
        badge={t("studio.model.fasterTrainingBadge")}
        className="shadow-border ring-border"
      >
        <div className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div
            data-tour="studio-local-model"
            className="flex min-w-0 flex-col gap-2"
          >
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              {t("studio.model.localModel")}
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-foreground/70 hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  {t("studio.model.localModelTooltip")}
                </TooltipContent>
              </Tooltip>
            </span>
            <div ref={localComboboxAnchorRef} className="min-w-0">
              <Combobox
                items={localResultIds}
                filteredItems={localFilteredIds}
                filter={null}
                value={localModelInput || null}
                onValueChange={(id) => {
                  const next = id ?? "";
                  setLocalModelInput(next);
                  if (next) setSelectedModel(next);
                }}
                onInputValueChange={setLocalModelInput}
                itemToStringValue={(id) => id}
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder={
                    isLoadingLocalModels
                      ? t("studio.model.scanningLocalAndCachedModels")
                      : "./models/my-model"
                  }
                  className={`${DARK_TRIGGER} [&_input]:text-background [&_input]:placeholder:text-background/40`}
                  onBlur={() => applyLocalModel(localModelInput)}
                  onKeyDown={(event) => {
                    if (event.key !== "Enter") return;
                    event.preventDefault();
                    applyLocalModel(localModelInput);
                  }}
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={FolderSearchIcon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent
                  anchor={localComboboxAnchorRef}
                  className={DARK_COMBOBOX_CONTENT}
                >
                  {isLoadingLocalModels ? (
                    <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                      <Spinner className="size-4" /> {t("studio.model.scanning")}
                    </div>
                  ) : localModelsError ? (
                    <div className="px-3 py-2 text-xs text-red-500">
                      {localModelsError}
                    </div>
                  ) : (
                    <ComboboxEmpty>{t("studio.model.noLocalModelsFound")}</ComboboxEmpty>
                  )}
                  <ComboboxList className="p-1">
                    {(id: string) => {
                      const model = localMetaById.get(id);
                      const source =
                        model?.source === "hf_cache"
                          ? t("studio.model.hfCache")
                          : model?.source === "lmstudio"
                            ? "LM Studio"
                            : model?.source === "custom"
                              ? t("studio.model.customFolders")
                              : t("studio.model.localDir");
                      return (
                        <ComboboxItem key={id} value={id} className="gap-2">
                          <Tooltip>
                            <TooltipTrigger asChild={true}>
                              <span className="block min-w-0 flex-1 truncate">
                                {model?.display_name ?? id}
                              </span>
                            </TooltipTrigger>
                            <TooltipContent
                              side="left"
                              className="max-w-xs break-all"
                            >
                              {model?.path ?? id}
                            </TooltipContent>
                          </Tooltip>
                          <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                            {source}
                          </span>
                        </ComboboxItem>
                      );
                    }}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </div>
            {isLoadingLocalModels ? (
              <p className="text-[10px] text-muted-foreground">
                {t("studio.model.scanningLocalModels")}
              </p>
            ) : localModelsError ? (
              <p className="text-[10px] text-red-500">{localModelsError}</p>
            ) : (
              <p className="text-[10px] text-muted-foreground">
                {trainableLocalModels.length > 0
                  ? t("studio.model.localModelsFound", {
                      count: trainableLocalModels.length,
                    })
                  : t("studio.model.noLocalModelsFoundManual")}
              </p>
            )}
          </div>

          <div
            data-tour="studio-base-model"
            className="flex min-w-0 flex-col gap-2"
          >
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              {t("studio.model.huggingFaceModel")}
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-foreground/70 hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  {t("studio.model.huggingFaceModelTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.model.readMore")}
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
            <div
              ref={comboboxAnchorRef}
              className="min-w-0"
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                if (!(event.target instanceof HTMLInputElement)) return;
                event.preventDefault();
                if (hfResults.length > 0) {
                  handleModelSelect(hfResults[0].id);
                } else {
                  const text = event.target.value.trim();
                  if (text) handleModelSelect(text);
                }
              }}
            >
              <Combobox
                items={resultIds}
                filteredItems={resultIds}
                filter={null}
                value={selectedModel}
                onValueChange={handleModelSelect}
                onInputValueChange={handleInputChange}
                itemToStringValue={(id) => id}
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder={t("studio.model.searchModels")}
                  className="w-full leading-5"
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent anchor={comboboxAnchorRef}>
                  {isLoading ? (
                    <div className="flex items-center justify-center py-4 gap-2 text-xs text-muted-foreground">
                      <Spinner className="size-4" /> {t("studio.model.searching")}
                    </div>
                  ) : (
                    <ComboboxEmpty>{t("studio.model.noModelsFound")}</ComboboxEmpty>
                  )}
                  <div
                    ref={scrollRef}
                    className="hover-scrollbar max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
                  >
                    <ComboboxList className="p-1 !max-h-none !overflow-visible">
                      {(id: string) => {
                        const entry = vramMap.get(id);
                        const detail = entry?.detail ?? null;
                        const fitStatus = entry?.status ?? null;
                        const vramEst = entry?.est ?? null;
                        const exceeds = fitStatus === "exceeds";

                        return (
                          <ComboboxItem
                            key={id}
                            value={id}
                            className="gap-2"
                          >
                            <Tooltip>
                              <TooltipTrigger asChild={true}>
                                <span
                                  className={`block min-w-0 flex-1 truncate ${exceeds ? "!text-gray-500 dark:!text-gray-400" : ""}`}
                                >
                                  {id}
                                </span>
                              </TooltipTrigger>
                              <TooltipContent
                                side="left"
                                className="max-w-xs break-all"
                              >
                                {id}
                                {vramEst != null &&
                                  vramEst > 0 &&
                                  gpu.available && (
                                    <span className="block text-[10px] mt-1">
                                      {exceeds
                                        ? t("studio.model.needsVram", {
                                            vram: vramEst,
                                            gpu: gpu.memoryTotalGb,
                                          })
                                        : fitStatus === "tight"
                                          ? t("studio.model.tightVram", {
                                              vram: vramEst,
                                              gpu: gpu.memoryTotalGb,
                                            })
                                          : t("studio.model.vramEstimate", {
                                              vram: vramEst,
                                            })}
                                    </span>
                                  )}
                              </TooltipContent>
                            </Tooltip>
                            <span className="ml-auto flex items-center gap-1.5 shrink-0">
                              {fitStatus === "exceeds" && (
                                <span className="text-[9px] font-medium !text-red-700 !bg-red-50 dark:!text-red-400 dark:!bg-red-950 px-1.5 py-0.5 rounded">
                                  OOM
                                </span>
                              )}
                              {fitStatus === "tight" && (
                                <span className="text-[9px] font-medium !text-amber-400">
                                  TIGHT
                                </span>
                              )}
                              {detail && (
                                <span className="text-[10px] text-muted-foreground">
                                  {detail}
                                </span>
                              )}
                            </span>
                          </ComboboxItem>
                        );
                      }}
                    </ComboboxList>
                    <div ref={sentinelRef} className="h-px" />
                    {isLoadingMore && (
                      <div className="flex items-center justify-center py-2">
                        <Spinner className="size-3.5 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                </ComboboxContent>
              </Combobox>
            </div>
          </div>

          <div
            data-tour="studio-method"
            className="flex min-w-0 flex-col gap-2"
          >
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              {t("studio.model.method")}
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-foreground/70 hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  {t("studio.model.methodTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.model.readMore")}
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
            <Select
              value={trainingMethod}
              onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
            >
              <SelectTrigger className={DARK_TRIGGER}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent
                position="popper"
                className={`${DARK_CONTENT} w-[var(--radix-select-trigger-width)]`}
              >
                <SelectItem value="qlora">
                  <span className="flex items-center gap-2">
                    <span
                      className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.qlora}`}
                    />
                    QLoRA (4-bit)
                  </span>
                </SelectItem>
                <SelectItem value="lora">
                  <span className="flex items-center gap-2">
                    <span
                      className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.lora}`}
                    />
                    LoRA (16-bit)
                  </span>
                </SelectItem>
                <SelectItem value="full">
                  <span className="flex items-center gap-2">
                    <span
                      className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.full}`}
                    />
                    {t("studio.model.fullFineTune")}
                  </span>
                </SelectItem>
                <SelectItem value="cpt">
                  <span className="flex items-center gap-2">
                    <span
                      className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.cpt}`}
                    />
                    {t("studio.model.continuedPretraining")}
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex min-w-0 flex-col gap-2">
            <span className="text-xs font-medium text-muted-foreground">
              {t("studio.model.huggingFaceTokenOptional")}
            </span>
            <InputGroup>
              <InputGroupAddon>
                <HugeiconsIcon icon={Key01Icon} className="size-4" />
              </InputGroupAddon>
              <InputGroupInput
                type="password"
                autoComplete="new-password"
                name="hf-token"
                placeholder="hf_..."
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
              />
            </InputGroup>
            {(tokenValidationError ?? hfSearchError) && (
              <p className="text-xs text-destructive">
                {tokenValidationError ?? hfSearchError}
                {" — "}
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  {t("studio.model.getOrUpdateToken")}
                </a>
              </p>
            )}
            {isCheckingToken && (
              <p className="text-xs text-muted-foreground">
                {t("studio.model.checkingToken")}
              </p>
            )}
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
