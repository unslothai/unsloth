// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HfTokenIndicator } from "@/components/hf-token-indicator";
import { SectionCardHeadlessContext } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { notifyInventoryChanged } from "@/stores/inventory-events";
import {
  HfDatasetSubsetSplitSelectors,
  parseYamlConfig,
  serializeConfigToYaml,
  uploadTrainingDataset,
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingReadiness,
} from "@/features/training";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  Archive04Icon,
  BrainIcon,
  CleanIcon,
  CloudUploadIcon,
  Database02Icon,
  FloppyDiskIcon,
  Rocket01Icon,
  Settings05Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type CSSProperties,
  type ReactNode,
  useCallback,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { useShallow } from "zustand/react/shallow";
import { TrainDatasetPicker } from "./dataset-picker";
import { ParamsSection } from "./sections/params-section";
import { TrainingMethodSelect, TrainModelPicker } from "./training-pickers";

const UPLOAD_ACCEPT = ".csv,.jsonl,.json,.parquet";

function SectionBox({
  title,
  description,
  icon,
  iconColor,
  titleAction,
  children,
  dataTour,
}: {
  title?: string;
  description?: string;
  icon?: IconSvgElement;
  iconColor?: string;
  titleAction?: ReactNode;
  children: ReactNode;
  dataTour?: string;
}) {
  return (
    <section
      data-tour={dataTour}
      className="elevated-card flex flex-col gap-4 bg-card p-5"
    >
      {title && (
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            {icon && (
              <span
                className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
                style={
                  iconColor
                    ? ({ "--chip-color": iconColor } as CSSProperties)
                    : undefined
                }
              >
                <HugeiconsIcon
                  icon={icon}
                  strokeWidth={1.5}
                  className="size-[18px]"
                />
              </span>
            )}
            <div className="min-w-0">
              <h3 className="select-none text-[13.5px] font-semibold leading-[18px] tracking-[-0.012em] text-foreground">
                {title}
              </h3>
              {description && (
                <p className="text-[11.5px] leading-[15px] text-muted-foreground/85">
                  {description}
                </p>
              )}
            </div>
          </div>
          {titleAction && <div className="shrink-0">{titleAction}</div>}
        </div>
      )}
      <div className="min-w-0">{children}</div>
    </section>
  );
}

type ParamMode = "simple" | "advanced";

const PARAM_MODE_KEY = "unsloth_train_param_mode";

function readParamMode(): ParamMode {
  if (typeof window === "undefined") return "simple";
  try {
    return window.localStorage.getItem(PARAM_MODE_KEY) === "advanced"
      ? "advanced"
      : "simple";
  } catch {
    return "simple";
  }
}

function useParamMode(): [ParamMode, (next: ParamMode) => void] {
  const [mode, setMode] = useState<ParamMode>(readParamMode);
  const update = useCallback((next: ParamMode) => {
    setMode(next);
    try {
      window.localStorage.setItem(PARAM_MODE_KEY, next);
    } catch {
      // ignore quota / disabled storage
    }
  }, []);
  return [mode, update];
}

function ParamModeToggle({
  mode,
  onChange,
}: {
  mode: ParamMode;
  onChange: (next: ParamMode) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Parameter mode"
      className="menu-trigger tab-toggle relative inline-flex h-8 w-[170px] shrink-0 items-center rounded-full"
    >
      <span
        aria-hidden="true"
        className={cn(
          "tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 w-1/2 rounded-full transition-transform duration-200 ease-out",
          mode === "advanced" ? "translate-x-full" : "translate-x-0",
        )}
      />
      {(["simple", "advanced"] as const).map((value) => {
        const active = mode === value;
        return (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(value)}
            className={cn(
              "relative z-10 inline-flex h-8 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] capitalize transition-colors",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {value}
          </button>
        );
      })}
    </div>
  );
}

function SetupField({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-[11px] font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {label}
      </span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function ModelPanel() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_180px_200px]">
      <div className="sm:col-span-2 xl:col-span-1">
        <SetupField label="Model">
          <TrainModelPicker />
        </SetupField>
      </div>
      <SetupField label="Method">
        <TrainingMethodSelect />
      </SetupField>
      <SetupField label="Hugging Face token">
        <HfTokenIndicator showLabel={true} />
      </SetupField>
    </div>
  );
}

function DatasetPanel() {
  const {
    datasetSource,
    dataset,
    datasetSubset,
    datasetSplit,
    datasetEvalSplit,
    setDatasetSubset,
    setDatasetSplit,
    setDatasetEvalSplit,
    selectLocalDataset,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      datasetSubset: s.datasetSubset,
      datasetSplit: s.datasetSplit,
      datasetEvalSplit: s.datasetEvalSplit,
      setDatasetSubset: s.setDatasetSubset,
      setDatasetSplit: s.setDatasetSplit,
      setDatasetEvalSplit: s.setDatasetEvalSplit,
      selectLocalDataset: s.selectLocalDataset,
    })),
  );
  const hfToken = useHfTokenStore((s) => s.token);

  const showHfHelpers = datasetSource === "huggingface" && !!dataset;

  const handleUploaded = useCallback(
    (path: string) => {
      selectLocalDataset(path);
    },
    [selectLocalDataset],
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2 lg:gap-5">
        <SetupField label="Hugging Face dataset">
          <TrainDatasetPicker />
        </SetupField>
        <SetupField label="Or upload a local file">
          <DatasetDropZone onUploaded={handleUploaded} />
        </SetupField>
      </div>

      {showHfHelpers && (
        <HfDatasetSubsetSplitSelectors
          variant="studio"
          enabled={true}
          datasetName={dataset}
          accessToken={hfToken || undefined}
          datasetSubset={datasetSubset}
          setDatasetSubset={setDatasetSubset}
          datasetSplit={datasetSplit}
          setDatasetSplit={setDatasetSplit}
          datasetEvalSplit={datasetEvalSplit}
          setDatasetEvalSplit={setDatasetEvalSplit}
        />
      )}
    </div>
  );
}

function DatasetDropZone({
  onUploaded,
}: {
  onUploaded: (path: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const dragCounterRef = useRef(0);

  const upload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      try {
        const result = await uploadTrainingDataset(file);
        notifyInventoryChanged();
        onUploaded(result.stored_path);
        toast.success("Dataset uploaded", { description: result.filename });
      } catch (err) {
        toast.error("Upload failed", {
          description: err instanceof Error ? err.message : undefined,
        });
      } finally {
        setIsUploading(false);
      }
    },
    [onUploaded],
  );

  function onFileChosen(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void upload(file);
  }

  function onDragEnter(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    dragCounterRef.current += 1;
    if (dragCounterRef.current === 1) setIsDragOver(true);
  }
  function onDragLeave(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
    if (dragCounterRef.current === 0) setIsDragOver(false);
  }
  function onDragOver(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  }
  function onDrop(event: React.DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    dragCounterRef.current = 0;
    setIsDragOver(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) void upload(file);
  }

  return (
    <>
      <button
        type="button"
        disabled={isUploading}
        onClick={() => fileInputRef.current?.click()}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        className={cn(
          "group relative flex h-9 w-full select-none items-center justify-center gap-2 rounded-[12px] border border-dashed px-3 text-center transition-colors",
          "border-foreground/15 dark:border-white/15",
          "hover:border-foreground/30 hover:bg-foreground/[0.02] dark:hover:border-white/30 dark:hover:bg-white/[0.025]",
          isDragOver &&
            "border-foreground/45 bg-foreground/[0.04] dark:border-white/40 dark:bg-white/[0.05]",
          isUploading && "cursor-progress opacity-80",
        )}
      >
        <HugeiconsIcon
          icon={CloudUploadIcon}
          strokeWidth={1.5}
          className="size-3.5 text-muted-foreground"
        />
        <span className="truncate text-[12.5px] text-foreground/85">
          {isUploading
            ? "Uploading…"
            : isDragOver
              ? "Release to upload"
              : "Drop a file, or click to browse"}
        </span>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept={UPLOAD_ACCEPT}
        className="hidden"
        onChange={onFileChosen}
      />
    </>
  );
}

function resolveStartTrainingError(input: {
  startError: string | null | undefined;
  isIncompatible: boolean;
  isAudioModel: boolean;
  isDatasetAudio: boolean | null | undefined;
  datasetUnverified: boolean;
  hasModel: boolean;
  hasDataset: boolean;
  configValidation: { ok: boolean; message?: string | null };
}): string | null {
  const {
    startError,
    isIncompatible,
    isAudioModel,
    isDatasetAudio,
    datasetUnverified,
    hasModel,
    hasDataset,
    configValidation,
  } = input;
  if (startError) return startError;
  if (isIncompatible) {
    return !isAudioModel && isDatasetAudio === true
      ? "This model does not support audio. Switch to an audio-capable model or choose a non-audio dataset."
      : "Text model is not compatible with a multimodal dataset. Switch to a vision model or choose a text-only dataset.";
  }
  if (datasetUnverified) {
    return "Couldn't verify the dataset is compatible with this model. Check your connection or Hugging Face token, then reselect the dataset.";
  }
  if (!hasModel || !hasDataset) return null;
  if (!configValidation.ok && configValidation.message) {
    return configValidation.message;
  }
  return null;
}

export function StartTrainingCta() {
  const { isAudioModel, isDatasetAudio } = useTrainingConfigStore(
    useShallow((s) => ({
      isAudioModel: s.isAudioModel,
      isDatasetAudio: s.isDatasetAudio,
    })),
  );
  const {
    isReady,
    isLoadingModel,
    isCheckingDataset,
    isIncompatible,
    datasetUnverified,
    datasetMetadataStale,
    hasModel,
    hasDataset,
    configValidation,
  } = useTrainingReadiness();
  const { isStarting, startError, startTrainingRun } = useTrainingActions();

  const disabled = isStarting || !isReady;

  const buttonLabel = isStarting
    ? "Starting training…"
    : isLoadingModel
      ? "Loading model…"
        : isCheckingDataset
          ? "Checking dataset…"
          : !hasModel && !hasDataset
            ? "Choose model and dataset"
            : !hasModel
              ? "Choose a model"
              : !hasDataset
                ? "Choose a dataset"
                : datasetMetadataStale
                  ? "Start with cached metadata"
                  : "Start training";

  const errorMessage = resolveStartTrainingError({
    startError,
    isIncompatible,
    isAudioModel,
    isDatasetAudio,
    datasetUnverified,
    hasModel,
    hasDataset,
    configValidation,
  });

  return (
    <div className="flex flex-col gap-2">
      <Button
        data-tour="studio-start"
        size="lg"
        className={cn(
          "h-11 w-full justify-center rounded-xl text-[13.5px] font-semibold tracking-tight",
          "bg-primary text-primary-foreground shadow-sm",
          "hover:bg-primary/90",
          "disabled:bg-foreground/[0.08] disabled:text-muted-foreground disabled:shadow-none dark:disabled:bg-white/[0.06]",
          "transition-colors duration-200",
        )}
        onClick={() => void startTrainingRun()}
        disabled={disabled}
      >
        <HugeiconsIcon
          icon={Rocket01Icon}
          strokeWidth={1.75}
          className="size-4"
        />
        {buttonLabel}
      </Button>
      {errorMessage && (
        <p className="text-[11.5px] leading-relaxed text-destructive">
          {errorMessage}
        </p>
      )}
      {!errorMessage && datasetMetadataStale && (
        <p className="text-[11.5px] leading-relaxed text-muted-foreground">
          Using cached dataset metadata because the dataset could not be rechecked online.
        </p>
      )}
    </div>
  );
}

function ConfigActions() {
  const selectedModel = useTrainingConfigStore((s) => s.selectedModel);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const config = parseYamlConfig(reader.result as string);
        useTrainingConfigStore.getState().applyConfigPatch(config);
        toast.success("Config loaded", { description: file.name });
      } catch (err) {
        toast.error("Failed to load config", {
          description:
            err instanceof Error ? err.message : "Invalid YAML file",
        });
      }
    };
    reader.onerror = () => {
      toast.error("Failed to read file");
    };
    reader.readAsText(file);
  };

  const handleSaveConfig = () => {
    const state = useTrainingConfigStore.getState();
    const yamlStr = serializeConfigToYaml(state, state.isVisionModel);
    const blob = new Blob([yamlStr], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;

    const model = (state.selectedModel ?? "model").split("/").pop();
    const method = state.trainingMethod ?? "qlora";
    const dataset = (state.dataset ?? "dataset").split("/").pop();
    const timestamp = new Date()
      .toISOString()
      .replace(/[:T]/g, "-")
      .slice(0, 19);
    a.download = `${model}_${method}_${dataset}_${timestamp}.yaml`;

    a.click();
    URL.revokeObjectURL(url);
  };

  const handleResetConfig = () => {
    useTrainingConfigStore.getState().resetToModelDefaults();
    toast.success("Parameters reset to model defaults");
  };

  return (
    <div className="flex flex-wrap gap-2">
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className="h-9 cursor-pointer rounded-lg"
            onClick={() => fileInputRef.current?.click()}
          >
            <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
            Load YAML
          </Button>
        </TooltipTrigger>
        <TooltipContent>Load a saved YAML config</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            data-tour="studio-save"
            variant="outline"
            size="sm"
            className="h-9 cursor-pointer rounded-lg"
            onClick={handleSaveConfig}
          >
            <HugeiconsIcon icon={Archive04Icon} className="size-3.5" />
            Save YAML
          </Button>
        </TooltipTrigger>
        <TooltipContent>Download current config as YAML</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className="h-9 cursor-pointer rounded-lg"
            onClick={handleResetConfig}
            disabled={!selectedModel}
          >
            <HugeiconsIcon icon={CleanIcon} className="size-3.5" />
            Reset to defaults
          </Button>
        </TooltipTrigger>
        <TooltipContent>Reset to model defaults</TooltipContent>
      </Tooltip>
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml"
        className="hidden"
        onChange={handleFileUpload}
      />
    </div>
  );
}

export function TrainingWizard() {
  const [paramMode, setParamMode] = useParamMode();
  return (
    <SectionCardHeadlessContext.Provider value={true}>
      <div className="flex flex-col gap-5">
        <SectionBox
          title="Model"
          description="Select model and training method"
          icon={BrainIcon}
          iconColor="#7abf85"
          dataTour="studio-model"
        >
          <ModelPanel />
        </SectionBox>

        <SectionBox
          title="Dataset"
          description="Select or upload training data"
          icon={Database02Icon}
          iconColor="#e7828c"
          dataTour="studio-dataset"
        >
          <DatasetPanel />
        </SectionBox>

        <SectionBox
          title="Parameters"
          description="Configure training parameters"
          icon={Settings05Icon}
          iconColor="#8a7cce"
          dataTour="studio-params"
          titleAction={
            <ParamModeToggle mode={paramMode} onChange={setParamMode} />
          }
        >
          <ParamsSection mode={paramMode} />
        </SectionBox>

        <SectionBox
          title="Configuration"
          description="Save and load configurations"
          icon={FloppyDiskIcon}
          iconColor="#6ab7be"
        >
          <ConfigActions />
        </SectionBox>
      </div>
    </SectionCardHeadlessContext.Provider>
  );
}
