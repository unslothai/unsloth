// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
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
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  listLocalModels,
  type LocalModelInfo,
  useTrainingConfigStore,
} from "@/features/training";
import {
  useDebouncedValue,
  useHfModelSearch,
  useHfTokenValidation,
} from "@/hooks";
import {
  AlertCircleIcon,
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  PackageIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { collapseAnim } from "./anim";
import type { ModelCheckpoints } from "./api/export-api";
import {
  cleanupExport,
  exportBase,
  exportGGUF,
  exportLoRA,
  exportMerged,
  fetchCheckpoints,
  loadCheckpoint,
} from "./api/export-api";
import { ExportDialog } from "./components/export-dialog";
import { MethodPicker } from "./components/method-picker";
import { QuantPicker } from "./components/quant-picker";
import {
  type ExportMethod,
  GUIDE_STEPS,
  getEstimatedSize,
} from "./constants";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { useI18n } from "@/features/i18n";
import { getExportTourSteps } from "./tour";

const SEARCH_INPUT_REASONS = new Set(["input-change", "input-paste", "input-clear"]);

export function ExportPage() {
  const { t } = useI18n();
  const { hfToken, setHfToken } = useTrainingConfigStore(
    useShallow((s) => ({
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
    })),
  );

  // ---- API-driven checkpoint state ----
  const [models, setModels] = useState<ModelCheckpoints[]>([]);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(true);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);

  const [selectedModelIdx, setSelectedModelIdx] = useState<string | null>(null);
  const [checkpoint, setCheckpoint] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<"checkpoint" | "model">(
    "checkpoint",
  );
  const [modelSource, setModelSource] = useState<"hf" | "local">("hf");
  const [hfExportTrustRemoteCode, setHfExportTrustRemoteCode] =
    useState(true);
  const [modelInput, setModelInput] = useState("");
  const [selectedSourceModel, setSelectedSourceModel] = useState<string | null>(
    null,
  );
  const [localModelInput, setLocalModelInput] = useState("");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(true);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);
  const debouncedModelQuery = useDebouncedValue(modelInput);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);

  const [exportMethod, setExportMethod] = useState<ExportMethod | null>(null);
  const [quantLevels, setQuantLevels] = useState<string[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);

  const [destination, setDestination] = useState<"local" | "hub">("local");
  const [hfUsername, setHfUsername] = useState("");
  const [modelName, setModelName] = useState("");
  const [privateRepo, setPrivateRepo] = useState(false);

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportSuccess, setExportSuccess] = useState(false);
  // Resolved on-disk path of the most recent successful export, surfaced
  // on the Export Complete screen so the user can find their model
  // without digging through the server log. Null for Hub-only pushes.
  const [exportOutputPath, setExportOutputPath] = useState<string | null>(null);

  const hfComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const localComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const selectingHfModelRef = useRef(false);
  const hfModelInputRef = useRef("");
  const localModelInputRef = useRef("");

  const tour = useGuidedTourController({
    id: "export",
    steps: getExportTourSteps(t),
  });

  // ---- Fetch checkpoints on mount ----
  useEffect(() => {
    let cancelled = false;
    setLoadingCheckpoints(true);
    setCheckpointError(null);
    fetchCheckpoints()
      .then((data) => {
        if (!cancelled) {
          setModels(data.models);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setCheckpointError(
            err instanceof Error ? err.message : t("export.page.error.loadCheckpoints"),
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingCheckpoints(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- Fetch local models for direct export ----
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
          error instanceof Error ? error.message : t("export.page.error.loadLocalModels"),
        );
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, []);

  // ---- Derived state ----
  const selectedModelData = useMemo(
    () =>
      selectedModelIdx != null
        ? models.find((m) => m.name === selectedModelIdx) ?? null
        : null,
    [models, selectedModelIdx],
  );

  const checkpointsForModel = useMemo(
    () => selectedModelData?.checkpoints ?? [],
    [selectedModelData],
  );

  // Derive training info from selected model's API metadata
  const baseModelName = selectedModelData?.base_model ?? "—";
  const isAdapter = !!selectedModelData?.peft_type;
  const isQuantized = !!selectedModelData?.is_quantized;
  const loraRank = selectedModelData?.lora_rank ?? null;
  const trainingMethodLabel = selectedModelData?.peft_type
    ? t("export.page.trainingMethod.lora")
    : t("export.page.trainingMethod.full");
  const sourceBaseModelName = sourceMode === "model"
    ? selectedSourceModel ?? "—"
    : baseModelName;

  const {
    results: hfResults,
    isLoading: isLoadingHfModels,
    error: hfSearchError,
  } = useHfModelSearch(debouncedModelQuery, {
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
  });
  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const hfResultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (
      selectedSourceModel &&
      modelSource === "hf" &&
      !ids.includes(selectedSourceModel)
    ) {
      ids.push(selectedSourceModel);
    }
    return ids;
  }, [hfResults, modelSource, selectedSourceModel]);

  const exportableLocalModels = useMemo(
    () =>
      localModels.filter((m) => {
        if (m.path.endsWith(".gguf")) return false;
        if (m.id.toLowerCase().includes("-gguf")) return false;
        return true;
      }),
    [localModels],
  );

  const localMetaById = useMemo(() => {
    const map = new Map<string, LocalModelInfo>();
    for (const model of exportableLocalModels) map.set(model.id, model);
    return map;
  }, [exportableLocalModels]);

  const localResultIds = useMemo(() => {
    const ids = exportableLocalModels.map((model) => model.id);
    const manual = localModelInput.trim();
    if (manual && !ids.includes(manual)) {
      ids.unshift(manual);
    }
    return ids;
  }, [exportableLocalModels, localModelInput]);

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

  const exportGuideSteps = useMemo(
    () =>
      sourceMode === "model"
        ? [
            t("export.page.guide.model.1"),
            t("export.page.guide.model.2"),
            t("export.page.guide.model.3"),
            t("export.page.guide.model.4"),
            t("export.page.guide.model.5"),
          ]
        : GUIDE_STEPS,
    [sourceMode],
  );

  // Reset checkpoint when the selected model changes
  useEffect(() => {
    setCheckpoint(null);
  }, [selectedModelIdx]);

  // Auto-reset export method if incompatible with the selected model type
  useEffect(() => {
    if (!isAdapter && (exportMethod === "merged" || exportMethod === "lora")) {
      setExportMethod(null);
    }
    // Quantized non-PEFT models can't export to any format
    if (!isAdapter && isQuantized && exportMethod !== null) {
      setExportMethod(null);
    }
  }, [isAdapter, isQuantized, exportMethod]);

  const handleSourceModeSwitch = useCallback(
    (next: "checkpoint" | "model") => {
      setSourceMode(next);
      if (next === "model") {
        setExportMethod("gguf");
      }
      setSelectedSourceModel(null);
      setLocalModelInput("");
      setModelInput("");
    },
    [],
  );

  useEffect(() => {
    setSelectedSourceModel(null);
    setLocalModelInput("");
    setModelInput("");
  }, [modelSource]);

  useEffect(() => {
    hfModelInputRef.current = modelInput;
  }, [modelInput]);

  useEffect(() => {
    localModelInputRef.current = localModelInput;
  }, [localModelInput]);

  const handleMethodChange = (method: ExportMethod) => {
    setExportMethod(method);
    if (method !== "gguf") {
      setQuantLevels([]);
    }
  };

  const estimatedSize = getEstimatedSize(exportMethod, quantLevels);
  const selectedExportSource =
    sourceMode === "checkpoint" ? checkpoint : selectedSourceModel;
  const canExport = !!(
    selectedExportSource &&
    exportMethod &&
    (exportMethod !== "gguf" || quantLevels.length > 0)
  );

  const applyHfSourceModel = useCallback((value: string) => {
    const next = value.trim();
    setModelInput(next);
    setSelectedSourceModel(next || null);
  }, []);

  const handleHfSourceModelSelect = useCallback((id: string | null) => {
    selectingHfModelRef.current = true;
    const next = id ?? "";
    hfModelInputRef.current = next;
    setModelInput(next);
    setSelectedSourceModel(id);
  }, []);

  const handleHfSourceInputChange = useCallback(
    (value: string, eventDetails?: { reason?: string }) => {
      hfModelInputRef.current = value;
      if (selectingHfModelRef.current) {
        selectingHfModelRef.current = false;
        return;
      }
      if (!SEARCH_INPUT_REASONS.has(eventDetails?.reason ?? "")) {
        return;
      }
      setModelInput(value);
      if (value.trim() === "") {
        setSelectedSourceModel(null);
      }
    },
    [],
  );

  const applyLocalSourceModel = useCallback((value: string) => {
    const next = value.trim();
    setLocalModelInput(next);
    setSelectedSourceModel(next || null);
  }, []);

  const handleLocalSourceInputChange = useCallback(
    (value: string, eventDetails?: { reason?: string }) => {
      localModelInputRef.current = value;
      if (!SEARCH_INPUT_REASONS.has(eventDetails?.reason ?? "")) {
        return;
      }
      setLocalModelInput(value);
      if (value.trim() === "") {
        setSelectedSourceModel(null);
      }
    },
    [],
  );

  // ---- Export handler ----
  const handleExport = useCallback(async () => {
    const source = sourceMode === "checkpoint" ? checkpoint : selectedSourceModel;
    if (!source) return;

    const selectedCp = sourceMode === "checkpoint"
      ? checkpointsForModel.find((cp) => cp.display_name === checkpoint)
      : null;
    if (sourceMode === "checkpoint" && !selectedCp) return;
    const checkpointPath = selectedCp?.path;

    setExporting(true);
    setExportError(null);
    setExportSuccess(false);
    setExportOutputPath(null);

    // For GGUF, use a flat folder like "exports/gemma-3-4b-it-finetune-gguf"
    // For other formats, nest under training-run/checkpoint
    const saveDir =
      exportMethod === "gguf"
        ? `${(sourceBaseModelName.split("/").pop() ?? selectedModelIdx ?? "model")
          .replace(/[^a-zA-Z0-9._-]/g, "-")}-gguf`
        : `${selectedModelIdx ?? "model"}/${checkpoint}`;
    const pushToHub = destination === "hub";
    const repoId = pushToHub && hfUsername && modelName
      ? `${hfUsername}/${modelName}`
      : undefined;
    const token = pushToHub && hfToken ? hfToken : undefined;

    try {
      // 1. Load model source
      if (sourceMode === "checkpoint") {
        if (!checkpointPath) return;
        await loadCheckpoint({ checkpoint_path: checkpointPath });
      } else {
        await loadCheckpoint({
          checkpoint_path: source,
          load_in_4bit: false,
          trust_remote_code:
            modelSource === "hf" ? hfExportTrustRemoteCode : true,
        });
      }

      // 2. Run export based on method. Capture the resolved output_path
      // (when the backend wrote a local copy) so the success screen can
      // show the user the realpath of their saved model. For multi-quant
      // GGUF runs, the directory is the same for every quant so we just
      // keep the last response.
      let lastOutputPath: string | null = null;
      if (exportMethod === "merged") {
        if (isAdapter) {
          const resp = await exportMerged({
            save_directory: saveDir,
            push_to_hub: pushToHub,
            repo_id: repoId,
            hf_token: token,
            private: privateRepo,
          });
          lastOutputPath = resp.details?.output_path ?? null;
        } else {
          const resp = await exportBase({
            save_directory: saveDir,
            push_to_hub: pushToHub,
            repo_id: repoId,
            hf_token: token,
            private: privateRepo,
            base_model_id: selectedModelData?.base_model,
          });
          lastOutputPath = resp.details?.output_path ?? null;
        }
      } else if (exportMethod === "gguf") {
        for (const quant of quantLevels) {
          const resp = await exportGGUF({
            save_directory: saveDir,
            quantization_method: quant,
            push_to_hub: pushToHub,
            repo_id: repoId,
            hf_token: token,
          });
          lastOutputPath = resp.details?.output_path ?? lastOutputPath;
        }
      } else if (exportMethod === "lora") {
        const resp = await exportLoRA({
          save_directory: saveDir,
          push_to_hub: pushToHub,
          repo_id: repoId,
          hf_token: token,
          private: privateRepo,
        });
        lastOutputPath = resp.details?.output_path ?? null;
      }

      setExportOutputPath(lastOutputPath);
      setExportSuccess(true);
    } catch (err) {
      setExportError(
        err instanceof Error ? err.message : t("export.page.error.exportFailed"),
      );
    } finally {
      try {
        await cleanupExport();
      } catch {
        // cleanup is best-effort
      }
      setExporting(false);
    }
  }, [
    checkpoint,
    checkpointsForModel,
    sourceMode,
    selectedSourceModel,
    selectedModelIdx,
    selectedModelData,
    exportMethod,
    isAdapter,
    sourceBaseModelName,
    quantLevels,
    destination,
    hfUsername,
    modelName,
    hfToken,
    privateRepo,
    modelSource,
    hfExportTrustRemoteCode,
  ]);

  // ---- Render ----
  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto max-w-7xl px-4 py-4 sm:px-6">
        <GuidedTour {...tour.tourProps} />

        <div className="mb-8 flex flex-col gap-0.5">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t("export.page.title")}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t("export.page.subtitle")}
          </p>
        </div>

        <SectionCard
          icon={<HugeiconsIcon icon={PackageIcon} className="size-5" />}
          title={t("export.page.config.title")}
          description={t("export.page.config.description")}
          accent="emerald"
          featured={true}
          className="shadow-border ring-1 ring-border"
        >
          {/* Loading / error states */}
          {loadingCheckpoints && (
            <div className="flex items-center gap-2 py-6 justify-center text-sm text-muted-foreground">
              <Spinner className="size-4" />
              {t("export.page.loadingCheckpoints")}
            </div>
          )}

          {checkpointError && (
            <div className="flex items-center gap-2 py-6 justify-center text-sm text-destructive">
              <HugeiconsIcon icon={AlertCircleIcon} className="size-4" />
              {checkpointError}
            </div>
          )}

          {!loadingCheckpoints && !checkpointError && (
            <>
              {/* Top row: Dropdowns + metadata | Guide */}
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2 md:gap-8">
                <div className="flex flex-col gap-2">
                  <div className="flex items-end justify-between">
                    <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                      {sourceMode === "checkpoint"
                        ? t("export.page.trainingRun")
                        : t("export.page.modelSource")}
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
                          {sourceMode === "checkpoint"
                            ? t("export.page.trainingRunHint")
                            : t("export.page.modelSourceHint")}
                        </TooltipContent>
                      </Tooltip>
                    </label>
                    <button
                      type="button"
                      onClick={() =>
                        handleSourceModeSwitch(
                          sourceMode === "checkpoint" ? "model" : "checkpoint",
                        )
                      }
                      className="text-xs text-primary underline cursor-pointer leading-none"
                    >
                      {sourceMode === "checkpoint"
                        ? t("export.page.useHfOrLocal")
                        : t("export.page.useTrainingCheckpoints")}
                    </button>
                  </div>

                  <AnimatePresence mode="wait" initial={false}>
                  {sourceMode === "checkpoint" ? (
                    <motion.div
                      key="checkpoint"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
                      className="flex flex-col gap-2 overflow-visible"
                    >
                      <div data-tour="export-training-run" className="flex flex-col gap-2">
                        <Select
                          value={selectedModelIdx ?? ""}
                          onValueChange={setSelectedModelIdx}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue
                              placeholder={
                                models.length === 0
                                  ? t("export.page.noTrainingRuns")
                                  : t("export.page.selectTrainingRun")
                              }
                            />
                          </SelectTrigger>
                          <SelectContent>
                            {models.map((m) => {
                              const tsMatch = m.name.match(/_(\d{10,})$/);
                              const displayName = tsMatch
                                ? m.name.slice(0, tsMatch.index)
                                : m.name;
                              const timeStr = tsMatch
                                ? new Date(Number(tsMatch[1]) * 1000).toLocaleString(
                                    undefined,
                                    {
                                      dateStyle: "medium",
                                      timeStyle: "short",
                                    },
                                  )
                                : null;
                              return (
                                <SelectItem key={m.name} value={m.name}>
                                  <span className="flex items-center gap-2">
                                    {displayName}
                                    <span className="text-muted-foreground text-xs">
                                      {t("export.page.checkpointCount")
                                        .replace("{count}", String(m.checkpoints.length))
                                        .replace("{suffix}", m.checkpoints.length !== 1 ? "s" : "")}
                                    </span>
                                    {timeStr && (
                                      <span className="text-muted-foreground text-xs">
                                        · {timeStr}
                                      </span>
                                    )}
                                  </span>
                                </SelectItem>
                              );
                            })}
                          </SelectContent>
                        </Select>
                      </div>

                      <div data-tour="export-checkpoint" className="flex flex-col gap-2">
                        <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                          {t("export.page.checkpoint")}
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
                              {t("export.page.checkpointHint")}{" "}
                              <a
                                href="https://unsloth.ai/docs/basics/inference-and-deployment"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-primary underline"
                              >
                                {t("export.page.readMore")}
                              </a>
                            </TooltipContent>
                          </Tooltip>
                        </label>
                        <Select
                          value={checkpoint ?? ""}
                          onValueChange={setCheckpoint}
                          disabled={!selectedModelIdx}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue
                              placeholder={
                                !selectedModelIdx
                                  ? t("export.page.selectTrainingRunFirst")
                                  : checkpointsForModel.length === 0
                                    ? t("export.page.noCheckpoints")
                                    : t("export.page.selectCheckpoint")
                              }
                            />
                          </SelectTrigger>
                          <SelectContent>
                            {checkpointsForModel.map((cp) => (
                              <SelectItem key={cp.path} value={cp.display_name}>
                                <span className="flex items-center gap-2">
                                  {cp.display_name}
                                  {cp.loss != null && (
                                    <span className="text-muted-foreground text-xs">
                                      {t("export.page.loss")}: {cp.loss.toFixed(4)}
                                    </span>
                                  )}
                                </span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="model"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
                      className="flex flex-col gap-2 overflow-visible"
                    >
                      <div className="flex gap-2">
                        <Button
                          variant={modelSource === "hf" ? "dark" : "outline"}
                          className="flex-1"
                          onClick={() => setModelSource("hf")}
                        >
                          {t("export.page.hf")}
                        </Button>
                        <Button
                          variant={modelSource === "local" ? "dark" : "outline"}
                          className="flex-1"
                          onClick={() => setModelSource("local")}
                        >
                          {t("export.page.localModel")}
                        </Button>
                      </div>

                      {modelSource === "hf" ? (
                        <>
                          <div className="flex flex-col gap-2">
                            <label className="text-xs font-medium text-muted-foreground">
                              {t("export.page.hfModel")}
                            </label>
                            <div ref={hfComboboxAnchorRef}>
                              <Combobox
                                items={hfResultIds}
                                filteredItems={hfResultIds}
                                filter={null}
                                value={modelInput || selectedSourceModel || null}
                                onValueChange={handleHfSourceModelSelect}
                                onInputValueChange={handleHfSourceInputChange}
                                itemToStringValue={(id) => id}
                                autoHighlight={true}
                              >
                                <ComboboxInput
                                  placeholder={t("export.page.searchModels")}
                                  className="w-full"
                                  onBlur={() =>
                                    applyHfSourceModel(hfModelInputRef.current)
                                  }
                                  onKeyDown={(event) => {
                                    if (event.key !== "Enter") return;
                                    event.preventDefault();
                                    applyHfSourceModel(hfModelInputRef.current);
                                  }}
                                >
                                  <InputGroupAddon>
                                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                                  </InputGroupAddon>
                                </ComboboxInput>
                                <ComboboxContent anchor={hfComboboxAnchorRef}>
                                  {isLoadingHfModels ? (
                                    <div className="flex items-center justify-center py-4 gap-2 text-xs text-muted-foreground">
                                      <Spinner className="size-4" /> {t("export.page.searching")}
                                    </div>
                                  ) : (
                                    <ComboboxEmpty>{t("export.page.noModels")}</ComboboxEmpty>
                                  )}
                                  <ComboboxList className="p-1 !max-h-none !overflow-visible">
                                    {(id: string) => (
                                      <ComboboxItem key={id} value={id} className="gap-2">
                                        <span className="block min-w-0 flex-1 truncate">
                                          {id}
                                        </span>
                                      </ComboboxItem>
                                    )}
                                  </ComboboxList>
                                </ComboboxContent>
                              </Combobox>
                            </div>
                            {(tokenValidationError ?? hfSearchError) && (
                              <p className="text-xs text-destructive">
                                {tokenValidationError ?? hfSearchError}
                              </p>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <Switch
                              id="hf-export-trust-remote-code"
                              size="sm"
                              checked={hfExportTrustRemoteCode}
                              onCheckedChange={setHfExportTrustRemoteCode}
                              disabled={exporting}
                            />
                            <label
                              htmlFor="hf-export-trust-remote-code"
                              className="cursor-pointer text-xs font-medium text-muted-foreground hover:text-foreground"
                            >
                              {t("export.page.trustRemoteCode")}
                            </label>
                            <Tooltip>
                              <TooltipTrigger asChild={true}>
                                <button
                                  type="button"
                                  className="text-muted-foreground hover:text-foreground -m-1 inline-flex rounded p-1"
                                  aria-label={t("export.page.trustRemoteCodeAbout")}
                                >
                                  <HugeiconsIcon
                                    icon={InformationCircleIcon}
                                    className="size-3.5"
                                  />
                                </button>
                              </TooltipTrigger>
                              <TooltipContent
                                side="top"
                                className="max-w-[260px] text-xs"
                              >
                                {t("export.page.trustRemoteCodeHint")}
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-medium text-muted-foreground">
                              {t("export.page.hfTokenOptional")}
                            </label>
                            <InputGroup>
                              <InputGroupAddon>
                                <HugeiconsIcon icon={Key01Icon} className="size-4" />
                              </InputGroupAddon>
                              <InputGroupInput
                                type="password"
                                autoComplete="new-password"
                                name="hf-token-export-source"
                                placeholder="hf_..."
                                value={hfToken}
                                onChange={(e) => setHfToken(e.target.value)}
                              />
                            </InputGroup>
                            {isCheckingToken && (
                              <p className="text-xs text-muted-foreground">{t("export.page.checkingToken")}</p>
                            )}
                          </div>
                        </>
                      ) : (
                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-medium text-muted-foreground">
                            {t("export.page.localModelPath")}
                          </label>
                          <div ref={localComboboxAnchorRef}>
                            <Combobox
                              items={localResultIds}
                              filteredItems={localFilteredIds}
                              filter={null}
                              value={localModelInput || null}
                              onValueChange={(id) => {
                                const next = id ?? "";
                                localModelInputRef.current = next;
                                setLocalModelInput(next);
                                setSelectedSourceModel(next || null);
                              }}
                              onInputValueChange={handleLocalSourceInputChange}
                              itemToStringValue={(id) => id}
                              autoHighlight={true}
                            >
                              <ComboboxInput
                                placeholder={
                                  isLoadingLocalModels
                                    ? t("export.page.scanningLocalAndCache")
                                    : t("export.page.localModelPlaceholder")
                                }
                                className="w-full"
                                onBlur={() => applyLocalSourceModel(localModelInputRef.current)}
                                onKeyDown={(event) => {
                                  if (event.key !== "Enter") return;
                                  event.preventDefault();
                                  applyLocalSourceModel(localModelInputRef.current);
                                }}
                              >
                                <InputGroupAddon>
                                  <HugeiconsIcon icon={FolderSearchIcon} className="size-4" />
                                </InputGroupAddon>
                              </ComboboxInput>
                              <ComboboxContent anchor={localComboboxAnchorRef}>
                                {isLoadingLocalModels ? (
                                  <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                                    <Spinner className="size-4" /> {t("export.page.scanning")}
                                  </div>
                                ) : localModelsError ? (
                                  <div className="px-3 py-2 text-xs text-red-500">
                                    {localModelsError}
                                  </div>
                                ) : (
                                  <ComboboxEmpty>{t("export.page.noLocalModels")}</ComboboxEmpty>
                                )}
                                <ComboboxList className="p-1 !max-h-none !overflow-visible">
                                  {(id: string) => {
                                    const model = localMetaById.get(id);
                                    const source =
                                      model?.source === "hf_cache"
                                        ? t("export.page.sourceHfCache")
                                        : model?.source === "custom"
                                          ? t("export.page.sourceCustomFolders")
                                          : t("export.page.sourceLocalDir");
                                    return (
                                      <ComboboxItem key={id} value={id} className="gap-2">
                                        <span className="block min-w-0 flex-1 truncate">
                                          {model?.display_name ?? id}
                                        </span>
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
                              {t("export.page.scanningLocal")}
                            </p>
                          ) : localModelsError ? (
                            <p className="text-[10px] text-red-500">{localModelsError}</p>
                          ) : (
                            <p className="text-[10px] text-muted-foreground">
                              {exportableLocalModels.length > 0
                                ? t("export.page.localModelsFound").replace(
                                    "{count}",
                                    String(exportableLocalModels.length),
                                  )
                                : t("export.page.noLocalModelsManual")}
                            </p>
                          )}
                        </div>
                      )}

                      <div className="rounded-xl bg-muted/50 p-3">
                        <p className="text-[11px] text-muted-foreground">
                          {t("export.page.directModelGgufOnly")}
                        </p>
                      </div>
                    </motion.div>
                  )}
                  </AnimatePresence>

                  {sourceMode === "checkpoint" && (
                    <div className="rounded-xl bg-muted/50 p-3 flex flex-col gap-2">
                      <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                        {t("export.page.trainingInfo")}
                      </span>
                      <div className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("export.page.summary.baseModel")}</span>
                          <span className="font-medium">{baseModelName}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("export.page.summary.method")}</span>
                          <span className="font-medium">
                            {trainingMethodLabel}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("export.page.summary.checkpoints")}</span>
                          <span className="font-medium">
                            {checkpointsForModel.length}
                          </span>
                        </div>
                        {isAdapter && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">{t("export.page.summary.loraRank")}</span>
                            <span className="font-medium">{loraRank}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-2.5">
                  <span className="text-xs font-medium text-muted-foreground">
                    {t("export.page.quickGuide")}
                  </span>
                  <ol className="flex flex-col gap-3">
                    {exportGuideSteps.map((step, i) => (
                      <li
                        key={step}
                        className="flex items-start gap-2 text-xs text-muted-foreground"
                      >
                        <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold">
                          {i + 1}
                        </span>
                        {step}
                      </li>
                    ))}
                  </ol>
                </div>
              </div>

              <MethodPicker
                value={exportMethod}
                onChange={handleMethodChange}
                disabledMethods={
                  !isAdapter && isQuantized
                    ? ["merged", "lora", "gguf"]
                    : !isAdapter || sourceMode === "model"
                      ? ["merged", "lora"]
                      : []
                }
                disabledReason={
                  !isAdapter && isQuantized
                    ? t("export.page.disabledReason.prequantized")
                    : sourceMode === "model"
                      ? t("export.page.disabledReason.ggufOnly")
                      : !isAdapter
                        ? t("export.page.disabledReason.fullFinetune")
                        : undefined
                }
              />

              <AnimatePresence>
                {exportMethod === "gguf" && (
                  <motion.div {...collapseAnim} className="overflow-visible">
                    <QuantPicker value={quantLevels} onChange={setQuantLevels} />
                  </motion.div>
                )}
              </AnimatePresence>

              <Separator />
              <div className="flex items-center justify-end">
                {/* TODO: unhide once estimated size comes from the backend API */}
                {/* <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                  <span>Est. size: {estimatedSize} · Free disk space: 120 GB</span>
                </div> */}
                <Button
                  data-tour="export-cta"
                  disabled={!canExport}
                  onClick={() => { setExportSuccess(false); setExportError(null); setDialogOpen(true); }}
                >
                  {t("export.page.title")}
                </Button>
              </div>
            </>
          )}
        </SectionCard>
      </main>

      <ExportDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        exportMethod={exportMethod}
        quantLevels={quantLevels}
        estimatedSize={estimatedSize}
        checkpoint={selectedExportSource}
        baseModelName={sourceBaseModelName}
        isAdapter={sourceMode === "checkpoint" && isAdapter}
        destination={destination}
        onDestinationChange={setDestination}
        hfUsername={hfUsername}
        onHfUsernameChange={setHfUsername}
        modelName={modelName}
        onModelNameChange={setModelName}
        hfToken={hfToken}
        onHfTokenChange={setHfToken}
        privateRepo={privateRepo}
        onPrivateRepoChange={setPrivateRepo}
        onExport={handleExport}
        exporting={exporting}
        exportError={exportError}
        exportSuccess={exportSuccess}
        exportOutputPath={exportOutputPath}
      />
    </div>
  );
}
