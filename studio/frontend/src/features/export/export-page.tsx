// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { useDebouncedValue, useHfTokenValidation } from "@/hooks";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  PackageIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { useSearch } from "@tanstack/react-router";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { ModelCheckpoints } from "./api/export-api";
import { fetchCheckpoints } from "./api/export-api";
import { ExportRunPanel } from "./components/export-run-panel";
import { MethodPicker } from "./components/method-picker";
import { QuantPicker } from "./components/quant-picker";
import {
  EXPORT_METHODS,
  type ExportMethod,
  GUIDE_STEPS,
  MERGED_FORMATS,
  type MergedFormat,
  QUANT_OPTIONS,
  buildQuantSizeLabels,
  getEstimatedSize,
} from "./constants";
import {
  isExportPanelActive,
  useExportRuntimeStore,
} from "./stores/export-runtime-store";
import { useExportSizeEstimate } from "./hooks/use-export-size-estimate";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { exportTourSteps } from "./tour";

const SEARCH_INPUT_REASONS = new Set(["input-change", "input-paste", "input-clear"]);

type SourceTab = "local" | "checkpoint" | "hf";
type SourceMode = "checkpoint" | "model";


function safePathSegment(
  value: string | null | undefined,
  fallback = "model",
  maxLength = 250,
): string {
  const safe = (value ?? "")
    .replace(/[^a-zA-Z0-9._-]/g, "-")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, maxLength)
    .replace(/[._-]+$/g, "");
  return safe || fallback;
}

function buildRelativeSaveDirectory(
  exportMethod: ExportMethod | null,
  sourceMode: SourceMode,
  sourceBaseModelName: string,
  selectedModelIdx: string | null,
  checkpoint: string | null,
): string {
  if (exportMethod === "gguf") {
    const rawName =
      sourceMode === "checkpoint"
        ? checkpoint ?? selectedModelIdx ?? sourceBaseModelName
        : sourceBaseModelName;
    return `${safePathSegment(rawName)}-GGUF`;
  }
  return `${selectedModelIdx ?? "model"}/${checkpoint}`;
}

function siblingGgufDirectory(sourcePath: string): string | null {
  const trimmed = sourcePath.trim().replace(/[\\/]+$/, "");
  if (!trimmed) return null;
  const slash = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  // Lowercase `_gguf` matches the backend's intermediate dir (core/export/export.py);
  // `_GGUF` would relocate+delete that sibling.
  if (slash < 0) return `${trimmed}_gguf`;
  const parent =
    slash === 0 || (slash === 2 && /^[A-Za-z]:/.test(trimmed))
      ? trimmed.slice(0, slash + 1)
      : trimmed.slice(0, slash);
  const name = trimmed.slice(slash + 1);
  if (!name) return null;
  const sep = parent.endsWith("/") || parent.endsWith("\\")
    ? ""
    : trimmed.includes("\\")
      ? "\\"
      : "/";
  return `${parent}${sep}${name}_gguf`;
}

export function ExportPage() {
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
  const [sourceMode, setSourceMode] = useState<SourceMode>("checkpoint");
  const [modelSource, setModelSource] = useState<"hf" | "local">("hf");
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

  // Seed the method + quants from a live run so that navigating away and back
  // (which remounts this page) keeps the method card selected and the run panel
  // showing its logs/progress. The run itself lives in the global store; only
  // this form state is local and would otherwise reset to null on remount.
  const [exportMethod, setExportMethod] = useState<ExportMethod | null>(() => {
    const s = useExportRuntimeStore.getState();
    return isExportPanelActive(s) && s.summary ? s.summary.method : null;
  });
  const [quantLevels, setQuantLevels] = useState<string[]>(() => {
    const s = useExportRuntimeStore.getState();
    return isExportPanelActive(s) && s.summary?.method === "gguf"
      ? s.summary.quantLevels
      : [];
  });
  // GGUF importance matrix (required for the IQ quants) and merged-export precision.
  const [useImatrix, setUseImatrix] = useState(false);
  const [mergedFormat, setMergedFormat] = useState<MergedFormat>("16-bit (FP16)");
  // IQ quants are imatrix-only, so force it on when one is selected; otherwise we would submit
  // an IQ quant with no imatrix and llama.cpp would reject it.
  const requiresImatrix = quantLevels.some(
    (q) => QUANT_OPTIONS.find((o) => o.value === q)?.imatrix,
  );
  const effectiveImatrix = useImatrix || requiresImatrix;

  // Whether the inline export panel is expanded. The panel also shows itself
  // whenever a run is active/terminal (see `panelActive`), so it survives
  // navigation even though this local flag resets on remount.
  const [panelOpen, setPanelOpen] = useState(false);

  const [destination, setDestination] = useState<"local" | "hub">("local");
  const [customSaveDirectory, setCustomSaveDirectory] = useState<string | null>(
    null,
  );
  const [hfUsername, setHfUsername] = useState("");
  const [modelName, setModelName] = useState("");
  const [privateRepo, setPrivateRepo] = useState(false);

  // Export run state lives in the global runtime store so it keeps running and
  // streaming in the background, in parallel with training and inference.
  const runExport = useExportRuntimeStore((s) => s.runExport);
  const resetExportRun = useExportRuntimeStore((s) => s.reset);
  const isExporting = useExportRuntimeStore((s) => s.isExporting);
  const panelActive = useExportRuntimeStore(isExportPanelActive);

  const hfComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const localComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const selectingHfModelRef = useRef(false);
  const hfModelInputRef = useRef("");
  const localModelInputRef = useRef("");

  const tour = useGuidedTourController({
    id: "export",
    steps: exportTourSteps,
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
            err instanceof Error ? err.message : "Failed to load checkpoints",
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

  // Apply the ?run= deep link (e.g. from a finished run's "Export to GGUF"
  // button) once its run appears in the checkpoint list: select the run and
  // default to GGUF. The main checkpoint is auto-selected further below, after
  // the model-change effect that clears the checkpoint.
  const { run: preselectRun } = useSearch({ from: "/export" });
  const appliedRunRef = useRef<string | null>(null);
  useEffect(() => {
    if (!preselectRun) {
      // Deep link cleared (e.g. navigated to /export via the sidebar): stop
      // treating the previously preselected run specially.
      appliedRunRef.current = null;
      return;
    }
    if (models.length === 0) return;
    if (appliedRunRef.current === preselectRun) return;
    const match = models.find((m) => m.name === preselectRun);
    if (!match) return;
    appliedRunRef.current = preselectRun;
    setSourceMode("checkpoint");
    setSelectedModelIdx(match.name);
    setExportMethod("gguf");
  }, [preselectRun, models]);

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
          error instanceof Error ? error.message : "Failed to load local models",
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
    ? "LoRA / QLoRA"
    : "Full Fine-tune";
  const sourceBaseModelName = sourceMode === "model"
    ? selectedSourceModel ?? "—"
    : baseModelName;

  // For a full fine-tune checkpoint the weights live in the checkpoint dir
  // itself (its base_model may be a local/custom path that can't be sized), so
  // size that dir; for LoRA adapters the export merges into the base model.
  const sizeTargetModel = useMemo(() => {
    if (sourceMode === "checkpoint" && !isAdapter) {
      const cp = checkpointsForModel.find((c) => c.display_name === checkpoint);
      if (cp?.path) {
        return cp.path;
      }
    }
    return sourceBaseModelName;
  }, [sourceMode, isAdapter, checkpointsForModel, checkpoint, sourceBaseModelName]);

  // Real (MoE-aware) fp16 size, used to scale the GGUF quant estimates.
  const { fp16Bytes } = useExportSizeEstimate(sizeTargetModel, debouncedHfToken);
  const quantSizeLabels = useMemo(
    () => buildQuantSizeLabels(fp16Bytes),
    [fp16Bytes],
  );

  const {
    results: hfResults,
    isLoading: isLoadingHfModels,
    error: hfSearchError,
  } = useHubModelSearch(debouncedModelQuery, {
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
    // Curated unsloth listing by default, but a typed query searches the whole
    // Hub (unsloth floated first) so non-unsloth base models stay selectable.
    ownerScope: debouncedModelQuery.trim() ? "all" : "unsloth",
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
            "Select a Hugging Face or local model to export from",
            "GGUF is used for non-finetuned model exports",
            "Pick one or more GGUF quantization levels",
            "Click Export and choose your destination",
            "Test your model and compare outputs in Chat",
          ]
        : GUIDE_STEPS,
    [sourceMode],
  );
  const sourceTab: SourceTab =
    sourceMode === "checkpoint" ? "checkpoint" : modelSource;

  // Reset checkpoint when the selected model changes
  useEffect(() => {
    setCheckpoint(null);
  }, [selectedModelIdx]);

  // For a ?run= deep link, default to the run's main checkpoint. Declared after
  // the reset effect above so it runs last and isn't clobbered back to null.
  useEffect(() => {
    if (appliedRunRef.current == null) return;
    if (appliedRunRef.current !== selectedModelIdx) return;
    if (checkpoint != null || checkpointsForModel.length === 0) return;
    setCheckpoint(checkpointsForModel[0].display_name);
  }, [selectedModelIdx, checkpoint, checkpointsForModel]);

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

  const handleSourceTabChange = useCallback((next: string) => {
    if (next === "checkpoint") {
      setSourceMode("checkpoint");
    } else if (next === "hf" || next === "local") {
      setSourceMode("model");
      setModelSource(next);
      setExportMethod("gguf");
    } else {
      return;
    }
    setSelectedSourceModel(null);
    setLocalModelInput("");
    setModelInput("");
    hfModelInputRef.current = "";
    localModelInputRef.current = "";
  }, []);

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

  const estimatedSize = getEstimatedSize(exportMethod, quantLevels, fp16Bytes);
  const selectedExportSource =
    sourceMode === "checkpoint" ? checkpoint : selectedSourceModel;
  const defaultSaveDirectory = useMemo(() => {
    const relative = buildRelativeSaveDirectory(
      exportMethod,
      sourceMode,
      sourceBaseModelName,
      selectedModelIdx,
      checkpoint,
    );
    if (
      exportMethod === "gguf" &&
      sourceMode === "model" &&
      modelSource === "local" &&
      selectedSourceModel
    ) {
      const localModel = localMetaById.get(selectedSourceModel);
      if (localModel && (localModel.source === "models_dir" || localModel.source === "custom")) {
        return siblingGgufDirectory(localModel.path) ?? relative;
      }
    }
    return relative;
  }, [
    checkpoint,
    exportMethod,
    localMetaById,
    modelSource,
    selectedModelIdx,
    selectedSourceModel,
    sourceBaseModelName,
    sourceMode,
  ]);
  const saveDirectory = customSaveDirectory?.trim() || defaultSaveDirectory;
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

  useEffect(() => {
    setCustomSaveDirectory(null);
  }, [checkpoint, exportMethod, modelSource, selectedModelIdx, selectedSourceModel, sourceMode]);

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

  // ---- Export handlers ----
  // Assemble the run params from the current form and hand off to the global
  // runtime store, which drives load -> export -> cleanup in the background.
  const handleStart = useCallback(async () => {
    const source = sourceMode === "checkpoint" ? checkpoint : selectedSourceModel;
    if (!source || !exportMethod) return;
    // A GGUF export with no quant selected runs zero exports yet would still
    // settle as success with no file; require at least one (mirrors canExport).
    if (exportMethod === "gguf" && quantLevels.length === 0) return;

    const selectedCp = sourceMode === "checkpoint"
      ? checkpointsForModel.find((cp) => cp.display_name === checkpoint)
      : null;
    if (sourceMode === "checkpoint" && !selectedCp) return;
    const checkpointPath = selectedCp?.path ?? null;

    const pushToHub = destination === "hub";
    const repoId = pushToHub && hfUsername && modelName
      ? `${hfUsername}/${modelName}`
      : undefined;
    const token = pushToHub && hfToken ? hfToken : undefined;
    const methodLabel =
      EXPORT_METHODS.find((m) => m.value === exportMethod)?.title ?? exportMethod;
    const adapterExport = sourceMode === "checkpoint" && isAdapter;

    // Consent gate for an HF source's custom (auto_map) code, run before we hand
    // off to runExport (which performs the load in the background). A local
    // checkpoint/model the user exported is trusted by default.
    let trustRemoteCode = modelSource !== "hf";
    let approvedRemoteCodeFingerprint: string | null = null;
    if (sourceMode !== "checkpoint") {
      const remoteCodeOk = await confirmRemoteCodeIfNeeded({
        modelName: source,
        hfToken: hfToken || null,
        // An HF source can need trust_remote_code via its YAML default with no
        // auto_map to review; signal it so a YAML-only model does not export
        // with it false.
        requiresTrustRemoteCode: modelSource === "hf",
        onApprove: (fingerprint) => {
          trustRemoteCode = true;
          approvedRemoteCodeFingerprint = fingerprint;
        },
      });
      if (!remoteCodeOk) return;
    }

    void runExport({
      sourceMode,
      checkpointPath,
      source,
      modelSource,
      trustRemoteCode,
      approvedRemoteCodeFingerprint,
      loadToken: hfToken || null,
      exportMethod,
      isAdapter: adapterExport,
      quantLevels,
      useImatrix: effectiveImatrix,
      mergedFormat,
      saveDirectory,
      destination,
      repoId,
      token,
      privateRepo,
      baseModelId: selectedModelData?.base_model ?? undefined,
      summary: {
        baseModelName: sourceBaseModelName,
        checkpointLabel: selectedExportSource,
        methodLabel,
        method: exportMethod,
        quantLevels,
        destination,
      },
    });
  }, [
    checkpoint,
    checkpointsForModel,
    sourceMode,
    selectedSourceModel,
    selectedModelData,
    selectedExportSource,
    sourceBaseModelName,
    exportMethod,
    isAdapter,
    quantLevels,
    effectiveImatrix,
    mergedFormat,
    destination,
    saveDirectory,
    hfUsername,
    modelName,
    hfToken,
    privateRepo,
    modelSource,
    runExport,
  ]);

  // Open the inline panel into a fresh config state. Clears any previous
  // terminal run (a still-running export cannot be cleared, so it stays).
  const handleOpenPanel = useCallback(() => {
    if (!isExporting) {
      resetExportRun();
    }
    setPanelOpen(true);
  }, [isExporting, resetExportRun]);

  // Collapse the panel. Only reachable from config / terminal states (never
  // mid-run), so resetting the store back to idle is safe.
  const handleClosePanel = useCallback(() => {
    resetExportRun();
    setPanelOpen(false);
  }, [resetExportRun]);

  const showPanel = panelOpen || panelActive;

  // Bring the panel into view when it opens and offer a scroll-down affordance
  // (like Chat) when its end is below the fold.
  const panelEndRef = useRef<HTMLDivElement>(null);
  const [panelEndVisible, setPanelEndVisible] = useState(true);

  useEffect(() => {
    if (!showPanel) return;
    const id = window.setTimeout(() => {
      panelEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, 60);
    return () => window.clearTimeout(id);
  }, [showPanel]);

  useEffect(() => {
    const el = panelEndRef.current;
    if (!showPanel || !el) return;
    // The scroll-down button is also gated on showPanel, so there is no need to
    // reset visibility when the panel closes; the observer self-corrects on open.
    const obs = new IntersectionObserver(
      ([entry]) => setPanelEndVisible(entry.isIntersecting),
      { rootMargin: "0px 0px -40px 0px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [showPanel]);

  // ---- Render ----
  return (
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto max-w-7xl px-5 py-8 sm:px-9">
        <GuidedTour {...tour.tourProps} />

        <div className="mb-8 flex flex-col gap-0.5">
          <h1 className="text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]">
            Export Model
          </h1>
          <p className="text-sm text-muted-foreground">
            Export fine-tuned or base models for deployment
          </p>
        </div>

        <SectionCard
          icon={<HugeiconsIcon icon={PackageIcon} className="size-5" />}
          title="Export Configuration"
          description="Select source, method, and quantization"
          accent="emerald"
          featured={true}
          className="ring-0 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-none"
        >
          {/* Loading / error states */}
          {loadingCheckpoints && (
            <div className="flex items-center gap-2 py-6 justify-center text-sm text-muted-foreground">
              <Spinner className="size-4" />
              Loading checkpoints…
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
                <div className="flex flex-col gap-3">
                  <div className="flex flex-col gap-2">
                    <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                      Source
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
                          Choose a local model, fine-tuned checkpoint, or
                          Hugging Face model to export.
                        </TooltipContent>
                      </Tooltip>
                    </label>
                    <Tabs
                      value={sourceTab}
                      onValueChange={handleSourceTabChange}
                      className="w-full"
                    >
                      <TabsList
                        unstyled={true}
                        className="hub-menu-trigger hub-tab-toggle relative inline-flex h-9 w-full items-center rounded-full"
                      >
                        <TabsTrigger
                          value="local"
                          indicatorClassName="hub-tab-toggle-pill rounded-full"
                          className="h-9 rounded-full border-0 px-3 text-[12.5px] text-muted-foreground hover:text-foreground data-active:text-foreground data-[state=active]:text-foreground"
                        >
                          Local Model
                        </TabsTrigger>
                        <TabsTrigger
                          value="checkpoint"
                          indicatorClassName="hub-tab-toggle-pill rounded-full"
                          className="h-9 rounded-full border-0 px-3 text-[12.5px] text-muted-foreground hover:text-foreground data-active:text-foreground data-[state=active]:text-foreground"
                        >
                          Fine-tuned
                        </TabsTrigger>
                        <TabsTrigger
                          value="hf"
                          indicatorClassName="hub-tab-toggle-pill rounded-full"
                          className="h-9 rounded-full border-0 px-3 text-[12.5px] text-muted-foreground hover:text-foreground data-active:text-foreground data-[state=active]:text-foreground"
                        >
                          Hugging Face
                        </TabsTrigger>
                      </TabsList>
                    </Tabs>
                  </div>

                  {sourceMode === "checkpoint" ? (
                    <div className="flex flex-col gap-2 overflow-visible">
                      <div data-tour="export-training-run" className="flex flex-col gap-2">
                        <label className="text-xs font-medium text-muted-foreground">
                          Training Run
                        </label>
                        <Select
                          value={selectedModelIdx ?? ""}
                          onValueChange={setSelectedModelIdx}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue
                              placeholder={
                                models.length === 0
                                  ? "No training runs found"
                                  : "Select a training run…"
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
                                      {m.checkpoints.length} checkpoint
                                      {m.checkpoints.length !== 1 ? "s" : ""}
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
                          Checkpoint
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
                              Choose a saved checkpoint to export. Lower loss
                              generally means better quality.{" "}
                              <a
                                href="https://unsloth.ai/docs/basics/inference-and-deployment"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-primary underline"
                              >
                                Read more
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
                                  ? "Select a training run first"
                                  : checkpointsForModel.length === 0
                                    ? "No checkpoints found"
                                    : "Select a checkpoint…"
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
                                      loss: {cp.loss.toFixed(4)}
                                    </span>
                                  )}
                                </span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-2 overflow-visible">
                      {modelSource === "hf" ? (
                        <>
                          <div className="flex flex-col gap-2">
                            <label className="text-xs font-medium text-muted-foreground">
                              Hugging Face Model
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
                                  placeholder="Search models..."
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
                                      <Spinner className="size-4" /> Searching…
                                    </div>
                                  ) : (
                                    <ComboboxEmpty>No models found</ComboboxEmpty>
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
                          {/* No persistent "trust remote code" toggle: custom code is
                              consented per model via the load-time review dialog. */}
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-medium text-muted-foreground">
                              Hugging Face Token (Optional)
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
                              <p className="text-xs text-muted-foreground">Checking token…</p>
                            )}
                          </div>
                        </>
                      ) : (
                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-medium text-muted-foreground">
                            Local Model Path
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
                                    ? "Scanning local and cached models..."
                                    : "./models/my-model"
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
                                    <Spinner className="size-4" /> Scanning...
                                  </div>
                                ) : localModelsError ? (
                                  <div className="px-3 py-2 text-xs text-red-500">
                                    {localModelsError}
                                  </div>
                                ) : (
                                  <ComboboxEmpty>No local models found</ComboboxEmpty>
                                )}
                                <ComboboxList className="p-1 !max-h-none !overflow-visible">
                                  {(id: string) => {
                                    const model = localMetaById.get(id);
                                    const source =
                                      model?.source === "hf_cache"
                                        ? "HF cache"
                                        : model?.source === "custom"
                                          ? "Custom Folders"
                                          : "Local dir";
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
                              Scanning local models...
                            </p>
                          ) : localModelsError ? (
                            <p className="text-[10px] text-red-500">{localModelsError}</p>
                          ) : (
                            <p className="text-[10px] text-muted-foreground">
                              {exportableLocalModels.length > 0
                                ? `${exportableLocalModels.length} local/cached models found`
                                : "No local models found. Enter path manually."}
                            </p>
                          )}
                        </div>
                      )}

                      <div className="rounded-xl bg-foreground/[0.04] p-3">
                        <p className="text-[11px] text-muted-foreground">
                          Direct model exports currently support GGUF only.
                        </p>
                      </div>
                    </div>
                  )}

                  {sourceMode === "checkpoint" && (
                    <div className="rounded-xl bg-foreground/[0.04] p-3 flex flex-col gap-2">
                      <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                        Training Info
                      </span>
                      <div className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Base Model</span>
                          <span className="font-medium">{baseModelName}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Method</span>
                          <span className="font-medium">
                            {trainingMethodLabel}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Checkpoints</span>
                          <span className="font-medium">
                            {checkpointsForModel.length}
                          </span>
                        </div>
                        {isAdapter && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">LoRA Rank</span>
                            <span className="font-medium">{loraRank}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-2.5">
                  <span className="text-xs font-medium text-muted-foreground">
                    Quick Guide
                  </span>
                  <ol className="flex flex-col gap-3">
                    {exportGuideSteps.map((step, i) => (
                      <li
                        key={step}
                        className="flex items-start gap-2 text-xs text-muted-foreground"
                      >
                        <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-foreground/10 text-[10px] font-semibold">
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
                    ? "Pre-quantized (BNB 4-bit) models cannot be exported without LoRA adapters"
                    : sourceMode === "model"
                      ? "Only GGUF export is available for direct model export"
                      : !isAdapter
                        ? "Not available for full fine-tune checkpoints (no LoRA adapters)"
                        : undefined
                }
              />

              {exportMethod === "merged" && isAdapter && (
                <div className="space-y-2">
                  <div className="text-sm font-medium">Precision</div>
                  <div className="flex flex-wrap gap-2">
                    {MERGED_FORMATS.map((f) => (
                      <Button
                        key={f.value}
                        type="button"
                        variant={mergedFormat === f.value ? "default" : "outline"}
                        size="sm"
                        onClick={() => setMergedFormat(f.value)}
                        title={f.hint}
                      >
                        {f.label}
                      </Button>
                    ))}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {MERGED_FORMATS.find((f) => f.value === mergedFormat)?.hint}
                  </div>
                </div>
              )}

              {exportMethod === "gguf" && (
                <>
                  <QuantPicker
                    value={quantLevels}
                    onChange={setQuantLevels}
                    sizes={quantSizeLabels}
                  />
                  <div className="flex items-center justify-between gap-3 rounded-lg border p-3">
                    <div className="space-y-0.5">
                      <div className="text-sm font-medium">
                        Importance matrix (imatrix)
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {requiresImatrix
                          ? "Required for the selected IQ low-bit quant. Auto-downloads the upstream Unsloth imatrix for the base model."
                          : "Improves quant quality and unlocks the IQ low-bit quants. Auto-downloads the upstream Unsloth imatrix for the base model."}
                      </div>
                    </div>
                    <Switch
                      checked={effectiveImatrix}
                      onCheckedChange={setUseImatrix}
                      disabled={requiresImatrix}
                    />
                  </div>
                </>
              )}
              {estimatedSize && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                  <span>Est. size: {estimatedSize}</span>
                </div>
              )}

              <Separator />
              {showPanel && (
                    <ExportRunPanel
                      exportMethod={exportMethod}
                      quantLevels={quantLevels}
                      checkpoint={selectedExportSource}
                      baseModelName={sourceBaseModelName}
                      isAdapter={sourceMode === "checkpoint" && isAdapter}
                      destination={destination}
                      onDestinationChange={setDestination}
                      saveDirectory={saveDirectory}
                      defaultSaveDirectory={defaultSaveDirectory}
                      saveDirectoryOverridden={!!customSaveDirectory}
                      onSaveDirectoryChange={setCustomSaveDirectory}
                      hfUsername={hfUsername}
                      onHfUsernameChange={setHfUsername}
                      modelName={modelName}
                      onModelNameChange={setModelName}
                      hfToken={hfToken}
                      onHfTokenChange={setHfToken}
                      privateRepo={privateRepo}
                      onPrivateRepoChange={setPrivateRepo}
                      onStart={handleStart}
                      onClose={handleClosePanel}
                    />
                )}
              {showPanel && (
                <div ref={panelEndRef} aria-hidden="true" className="h-px w-full" />
              )}
              {showPanel && !panelEndVisible && (
                <button
                  type="button"
                  onClick={() =>
                    panelEndRef.current?.scrollIntoView({
                      behavior: "smooth",
                      block: "end",
                    })
                  }
                  aria-label="Scroll to export output"
                  className="fixed bottom-6 right-6 z-30 flex size-10 items-center justify-center rounded-full border border-border/60 bg-background/90 text-foreground shadow-md backdrop-blur transition-colors hover:bg-muted"
                >
                  <HugeiconsIcon icon={ArrowDown01Icon} className="size-5" />
                </button>
              )}
              {!showPanel && (
                <div className="flex items-center justify-end">
                  <Button
                    data-tour="export-cta"
                    disabled={!canExport}
                    onClick={handleOpenPanel}
                  >
                    Export Model
                  </Button>
                </div>
              )}
            </>
          )}
        </SectionCard>
      </main>
    </div>
  );
}
