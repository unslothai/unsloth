// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { FolderBrowser } from "@/components/assistant-ui/model-selector/folder-browser";
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
  type ScanFolderInfo,
  addScanFolder,
  listRecommendedFolders,
  listScanFolders,
  removeScanFolder,
} from "@/features/chat/api/chat-api";
import {
  type LocalModelInfo,
  listLocalModels,
  useTrainingConfigStore,
} from "@/features/training";
import {
  useDebouncedValue,
  useGpuInfo,
  useHfModelSearch,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import { extractParamLabel } from "@/lib/model-size";
import { formatCompact } from "@/lib/utils";
import {
  type VramFitStatus,
  type TrainingMethod as VramTrainingMethod,
  buildModelVramMap,
} from "@/lib/vram";
import type { TrainingMethod } from "@/types/training";
import {
  Add01Icon,
  Cancel01Icon,
  ChipIcon,
  Folder02Icon,
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { translate, useT } from "@/i18n";
import { toast } from "@/lib/toast";

const METHOD_DOTS: Record<string, string> = {
  qlora: "bg-emerald-400",
  lora: "bg-blue-400",
  full: "bg-amber-400",
  cpt: "bg-purple-400",
};

const DARK_TRIGGER =
  "w-full bg-foreground text-background hover:bg-foreground/90 dark:bg-foreground dark:text-background dark:hover:bg-foreground [&_svg]:text-background/50";
const DARK_CONTENT =
  "bg-foreground text-background shadow-xl border-background/10 [--accent:rgba(255,255,255,0.1)] [--accent-foreground:white] dark:[--accent:rgba(2,6,23,0.08)] dark:[--accent-foreground:rgb(2,6,23)] [&_[data-slot=select-item]]:text-white/80 dark:[&_[data-slot=select-item]]:text-slate-900 [&_[data-slot=select-scroll-up-button]]:bg-foreground [&_[data-slot=select-scroll-down-button]]:bg-foreground";
const DARK_COMBOBOX_CONTENT =
  "bg-foreground text-background shadow-xl border-background/10 dark:[--accent:rgba(2,6,23,0.08)] dark:[--accent-foreground:rgb(2,6,23)] dark:[&_[data-slot=combobox-item]]:text-slate-900 dark:[&_.text-muted-foreground]:text-slate-500";

let _scanFoldersCache: ScanFolderInfo[] = [];

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
  const [customFoldersCollapsed, setCustomFoldersCollapsed] = useState(false);
  const [scanFolders, setScanFolders] =
    useState<ScanFolderInfo[]>(_scanFoldersCache);
  const [folderInput, setFolderInput] = useState("");
  const [folderError, setFolderError] = useState<string | null>(null);
  const [showFolderInput, setShowFolderInput] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);
  const [recommendedFolders, setRecommendedFolders] = useState<string[]>([]);
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

  const refreshLocalModelsList = useCallback((signal?: AbortSignal) => {
    return listLocalModels(signal)
      .then((models) => {
        if (signal?.aborted) return;
        setLocalModels(models);
        setLocalModelsError(null);
      })
      .catch((error) => {
        if (signal?.aborted) return;
        setLocalModelsError(
          error instanceof Error
            ? error.message
            : translate("studio.model.failedToLoadLocalModels"),
        );
      });
  }, []);

  const refreshScanFolders = useCallback(() => {
    listScanFolders()
      .then((folders) => {
        _scanFoldersCache = folders;
        setScanFolders(folders);
      })
      .catch(() => {});
  }, []);

  const handleAddFolder = useCallback(async (overridePath?: string) => {
    const raw = overridePath !== undefined ? overridePath : folderInput;
    const trimmed = raw.trim();
    if (!trimmed || folderLoading) return;
    setFolderError(null);
    setFolderLoading(true);
    const fromBrowser = overridePath !== undefined;
    try {
      const created = await addScanFolder(trimmed);
      const next = _scanFoldersCache.some(
        (folder) => folder.id === created.id || folder.path === created.path,
      )
        ? _scanFoldersCache
        : [..._scanFoldersCache, created];
      _scanFoldersCache = next;
      setScanFolders(next);
      setFolderInput("");
      setShowFolderInput(false);
      void refreshLocalModelsList();
      void refreshScanFolders();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to add folder";
      setFolderError(message);
      if (fromBrowser) {
        toast.error("Couldn't add folder", { description: message });
      }
    } finally {
      setFolderLoading(false);
    }
  }, [folderInput, folderLoading, refreshLocalModelsList, refreshScanFolders]);

  const handleRemoveFolder = useCallback(async (id: number) => {
    try {
      await removeScanFolder(id);
      const next = _scanFoldersCache.filter((folder) => folder.id !== id);
      _scanFoldersCache = next;
      setScanFolders(next);
      void refreshScanFolders();
      void refreshLocalModelsList();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to remove folder",
      );
      void refreshScanFolders();
    }
  }, [refreshLocalModelsList, refreshScanFolders]);

  useEffect(() => {
    const controller = new AbortController();
    void refreshLocalModelsList(controller.signal)
      .finally(() => {
        if (controller.signal.aborted) return;
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, [refreshLocalModelsList]);

  useEffect(() => {
    refreshScanFolders();
    listRecommendedFolders()
      .then(setRecommendedFolders)
      .catch(() => {});
  }, [refreshScanFolders]);
  const task = modelType ? MODEL_TYPE_TO_HF_TASK[modelType] : undefined;
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfModelSearch(debouncedQuery, {
    task,
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
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
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

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
                  className="w-full bg-foreground text-background [&_input]:text-background [&_input]:placeholder:text-background/40 [&_svg]:text-background/50 hover:bg-foreground/90"
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
                  <div className="flex items-center gap-1 px-2.5 py-1.5">
                    <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      <HugeiconsIcon icon={Folder02Icon} className="size-3" />
                      {t("studio.model.customFolders")}
                    </span>
                    <div className="flex items-center gap-0.5">
                      <button
                        type="button"
                        aria-label={
                          showFolderInput
                            ? "Cancel adding folder"
                            : "Add scan folder by path"
                        }
                        title={
                          showFolderInput ? "Cancel" : "Add by typing a path"
                        }
                        onClick={() => {
                          setShowFolderInput((open) => {
                            if (open) {
                              setFolderInput("");
                              setFolderError(null);
                            }
                            return !open;
                          });
                        }}
                        className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-background dark:hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={showFolderInput ? Cancel01Icon : Add01Icon}
                          className="size-3"
                        />
                      </button>
                      <button
                        type="button"
                        aria-label="Browse for a folder on the server"
                        title="Browse folders on the server"
                        onClick={() => setShowFolderBrowser(true)}
                        className="shrink-0 rounded p-0.5 text-muted-foreground/60 transition-colors hover:text-background dark:hover:text-foreground"
                      >
                        <HugeiconsIcon icon={Search01Icon} className="size-2.5" />
                      </button>
                    </div>
                    <div className="ml-auto">
                      <button
                        type="button"
                        aria-label={
                          customFoldersCollapsed
                            ? "Expand custom folders"
                            : "Collapse custom folders"
                        }
                        title={customFoldersCollapsed ? "Expand" : "Collapse"}
                        onClick={() => setCustomFoldersCollapsed((v) => !v)}
                        className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-background dark:hover:text-foreground"
                      >
                        {customFoldersCollapsed ? (
                          <ChevronRightIcon className="size-3" />
                        ) : (
                          <ChevronDownIcon className="size-3" />
                        )}
                      </button>
                    </div>
                  </div>

                  {!customFoldersCollapsed &&
                    scanFolders.map((folder) => (
                      <div
                        key={folder.id}
                        className="group flex items-center gap-1.5 px-2.5 py-0.5"
                      >
                        <HugeiconsIcon
                          icon={Folder02Icon}
                          className="size-3 shrink-0 text-muted-foreground/40"
                        />
                        <span
                          className="min-w-0 flex-1 truncate font-mono text-[10px] text-muted-foreground/70"
                          title={folder.path}
                        >
                          {folder.path}
                        </span>
                        <button
                          type="button"
                          onClick={() => handleRemoveFolder(folder.id)}
                          aria-label={`Remove folder ${folder.path}`}
                          className="shrink-0 rounded p-1 text-background/70 transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:bg-destructive/10 focus-visible:text-destructive dark:text-foreground/70"
                        >
                          <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                        </button>
                      </div>
                    ))}

                  {!customFoldersCollapsed && (() => {
                    const registered = new Set(
                      scanFolders.map((folder) => folder.path),
                    );
                    const unregistered = recommendedFolders.filter(
                      (path) => !registered.has(path),
                    );
                    if (unregistered.length === 0) return null;
                    return (
                      <div className="flex flex-wrap gap-1 px-2.5 pb-0.5">
                        {unregistered.map((path) => (
                          <button
                            key={path}
                            type="button"
                            onClick={() => void handleAddFolder(path)}
                            disabled={folderLoading}
                            title={`Add ${path}`}
                            className="rounded-full border border-dashed border-background/20 px-2 py-0.5 font-mono text-[10px] text-muted-foreground/70 transition-colors hover:border-background/40 hover:bg-background/10 hover:text-background disabled:opacity-40 dark:hover:border-foreground/30 dark:hover:bg-accent dark:hover:text-foreground"
                          >
                            <span className="text-[11px] font-semibold">+</span>{" "}
                            {path.length > 30 ? `...${path.slice(-27)}` : path}
                          </button>
                        ))}
                      </div>
                    );
                  })()}

                  {!customFoldersCollapsed && showFolderInput && (
                    <div className="px-2.5 pb-1 pt-0.5">
                      <div className="flex items-center gap-1">
                        <HugeiconsIcon
                          icon={Folder02Icon}
                          className="size-3 shrink-0 text-muted-foreground/40"
                        />
                        <input
                          value={folderInput}
                          onChange={(event) => {
                            setFolderInput(event.target.value);
                            setFolderError(null);
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              event.preventDefault();
                              void handleAddFolder();
                            }
                            if (event.key === "Escape") {
                              event.preventDefault();
                              event.stopPropagation();
                              setShowFolderInput(false);
                              setFolderInput("");
                              setFolderError(null);
                            }
                          }}
                          placeholder="/path/to/models"
                          className="h-6 min-w-0 flex-1 rounded border border-background/20 bg-transparent px-1.5 font-mono text-[10px] text-background outline-none placeholder:text-muted-foreground/40 focus:border-background/40 dark:border-border/50 dark:text-foreground dark:focus:border-foreground/20"
                          disabled={folderLoading}
                          autoFocus={true}
                        />
                        <button
                          type="button"
                          onClick={() => setShowFolderBrowser(true)}
                          disabled={folderLoading}
                          aria-label="Browse for folder"
                          title="Browse folders on the server"
                          className="flex h-6 shrink-0 items-center justify-center rounded border border-background/20 px-1.5 text-muted-foreground transition-colors hover:bg-background/10 hover:text-background disabled:opacity-40 dark:border-border/50 dark:hover:bg-accent dark:hover:text-foreground"
                        >
                          <HugeiconsIcon icon={Search01Icon} className="size-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            void handleAddFolder();
                          }}
                          disabled={folderLoading || !folderInput.trim()}
                          className="h-6 shrink-0 rounded border border-background/20 px-1.5 text-[10px] text-muted-foreground transition-colors hover:bg-background/10 hover:text-background disabled:opacity-40 dark:border-border/50 dark:hover:bg-accent"
                        >
                          Add
                        </button>
                      </div>
                      {folderError && (
                        <p className="px-0.5 pt-0.5 text-[10px] text-destructive">
                          {folderError}
                        </p>
                      )}
                    </div>
                  )}

                  <FolderBrowser
                    open={showFolderBrowser}
                    onOpenChange={setShowFolderBrowser}
                    initialPath={folderInput.trim() || undefined}
                    onSelect={(picked) => {
                      setFolderInput(picked);
                      setFolderError(null);
                      void handleAddFolder(picked);
                    }}
                  />

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
                    className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
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
