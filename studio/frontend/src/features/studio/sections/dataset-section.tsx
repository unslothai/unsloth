// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
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
import { usePlatformStore } from "@/config/env";
import { DatasetSelector } from "@/features/dataset-picker";
import {
  bumpInventoryVersion,
  hfApiToken,
  looksLikeLocalPath,
  useHfTokenStore,
} from "@/features/hub";
import {
  formatUploadSize,
  getCachedUploadLimitBytes,
  getCachedUploadLimitLabel,
  loadUploadLimitSettings,
  subscribeUploadLimitSettings,
} from "@/features/settings";
import {
  HfDatasetSubsetSplitSelectors,
  type LocalDatasetInfo,
  hasSeparateStreamingEvalSplit,
  listLocalDatasets,
  uploadTrainingDataset,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
} from "@/features/training";
import { translate, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  CloudUploadIcon,
  FileAttachmentIcon,
  InformationCircleIcon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { DocumentUploadRedirectDialog } from "./document-upload-redirect-dialog";
import { S3ConfigForm } from "./s3-config-form";

const TRAINING_UPLOAD_EXTENSIONS = [
  ".csv",
  ".jsonl",
  ".json",
  ".parquet",
  ".pdf",
  ".docx",
  ".txt",
] as const;
const TRAINING_UPLOAD_EXTENSION_SET = new Set<string>(
  TRAINING_UPLOAD_EXTENSIONS,
);
const TRAINING_UPLOAD_ACCEPT = TRAINING_UPLOAD_EXTENSIONS.join(",");
const TRAINING_UPLOAD_LABEL = "CSV, JSONL, JSON, Parquet, PDF, DOCX, TXT";
const TRAINING_DATASET_UPLOAD_LABEL = "CSV, JSONL, JSON, Parquet";
const DOCUMENT_REDIRECT_LABEL = "PDF/DOCX/TXT open Learning Recipes";
const DOCUMENT_REDIRECT_EXTENSIONS = new Set([".pdf", ".docx", ".txt"]);

const OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY =
  "data-recipes:open-learning-recipes";

function getFileExtension(fileName: string) {
  const extensionStart = fileName.lastIndexOf(".");
  return extensionStart >= 0
    ? fileName.slice(extensionStart).toLowerCase()
    : "";
}

function isLikelyLocalDatasetRef(value: string) {
  return (
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.includes("\\") ||
    /\.(jsonl|json|csv|parquet)$/i.test(value)
  );
}

function deriveLocalDatasetName(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  const parquetIndex = parts.lastIndexOf("parquet-files");
  if (parquetIndex > 0) return parts[parquetIndex - 1];
  const basename = parts[parts.length - 1] ?? path;
  // Strip UUID prefix from uploaded files (format: {32hex}_{original})
  const uuidPrefixMatch = basename.match(/^[a-f0-9]{32}_(.+)$/);
  if (uuidPrefixMatch) return uuidPrefixMatch[1];
  return basename;
}

function formatUpdatedDate(timestamp: number | null): string {
  if (typeof timestamp !== "number") return "--";
  return new Date(timestamp * 1000).toLocaleDateString();
}

function normalizeSliceInput(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!/^\d+$/.test(trimmed)) return null;
  return trimmed;
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
      {children}
    </span>
  );
}

export function DatasetPanel() {
  const t = useT();
  const navigate = useNavigate();
  const {
    dataset,
    datasetSource,
    selectHfDataset,
    selectLocalDataset,
    selectS3Source,
    datasetFormat,
    setDatasetFormat,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
    datasetEvalSplit,
    setDatasetEvalSplit,
    datasetStreaming,
    setDatasetStreaming,
    trainOnCompletions,
    maxSteps,
    evalSteps,
    isVisionModel,
    isAudioModel,
    isEmbeddingModel,
    isDatasetImage,
    isDatasetAudio,
    uploadedFile,
    uploadedEvalFile,
    setUploadedEvalFile,
    modelType,
    datasetSliceStart,
    setDatasetSliceStart,
    datasetSliceEnd,
    setDatasetSliceEnd,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      dataset: s.dataset,
      datasetSource: s.datasetSource,
      selectHfDataset: s.selectHfDataset,
      selectLocalDataset: s.selectLocalDataset,
      selectS3Source: s.selectS3Source,
      datasetFormat: s.datasetFormat,
      setDatasetFormat: s.setDatasetFormat,
      datasetSubset: s.datasetSubset,
      setDatasetSubset: s.setDatasetSubset,
      datasetSplit: s.datasetSplit,
      setDatasetSplit: s.setDatasetSplit,
      datasetEvalSplit: s.datasetEvalSplit,
      setDatasetEvalSplit: s.setDatasetEvalSplit,
      datasetStreaming: s.datasetStreaming,
      setDatasetStreaming: s.setDatasetStreaming,
      trainOnCompletions: s.trainOnCompletions,
      maxSteps: s.maxSteps,
      evalSteps: s.evalSteps,
      isVisionModel: s.isVisionModel,
      isAudioModel: s.isAudioModel,
      isEmbeddingModel: s.isEmbeddingModel,
      isDatasetImage: s.isDatasetImage,
      isDatasetAudio: s.isDatasetAudio,
      uploadedFile: s.uploadedFile,
      uploadedEvalFile: s.uploadedEvalFile,
      setUploadedEvalFile: s.setUploadedEvalFile,
      modelType: s.modelType,
      datasetSliceStart: s.datasetSliceStart,
      setDatasetSliceStart: s.setDatasetSliceStart,
      datasetSliceEnd: s.datasetSliceEnd,
      setDatasetSliceEnd: s.setDatasetSliceEnd,
    })),
  );

  const hfToken = useHfTokenStore((s) => s.token);
  const platformDeviceType = usePlatformStore((s) => s.deviceType);

  // Streaming is only supported for Hugging Face text datasets. Rather than
  // hiding the toggle when a constraint isn't met, keep it visible but disabled
  // and list the exact unmet requirement(s) in its tooltip — a control that
  // silently disappears is confusing. Downstream preprocessing
  // (convert_to_vlm_format, audio collators) needs random access and would
  // crash on an IterableDataset, hence the constraints below.
  const streamingBlockers: string[] = [];
  if (datasetSource !== "huggingface")
    streamingBlockers.push(
      "Use a Hugging Face dataset (not a local upload or S3 source).",
    );
  if (maxSteps <= 0)
    streamingBlockers.push(
      "Set Max Steps > 0 — streaming datasets have no known length.",
    );
  if (trainOnCompletions)
    streamingBlockers.push('Turn off "Assistant completions only".');
  if (
    !hasSeparateStreamingEvalSplit({
      evalSteps,
      datasetSplit,
      datasetEvalSplit,
    })
  )
    streamingBlockers.push(
      "Pick a separate eval split — evaluation is on but no distinct eval split is set.",
    );
  if (isVisionModel)
    streamingBlockers.push("Vision models don't support streaming.");
  if (isAudioModel)
    streamingBlockers.push("Audio models don't support streaming.");
  if (isEmbeddingModel)
    streamingBlockers.push(
      "Embedding models don't support streaming (training needs the full dataset).",
    );
  if (isDatasetImage)
    streamingBlockers.push(
      "This dataset looks like images, which can't stream.",
    );
  if (isDatasetAudio)
    streamingBlockers.push(
      "This dataset looks like audio, which can't stream.",
    );
  if (platformDeviceType === "mac")
    streamingBlockers.push(
      "Streaming isn't supported on Apple Silicon (MLX) yet.",
    );

  const isStreamingSupported = streamingBlockers.length === 0;

  // If streaming was previously enabled but the config became incompatible
  // (model switched to vision, dataset detected as image, etc.), clear it so
  // the backend never receives a stale flag.
  useEffect(() => {
    if (datasetStreaming && !isStreamingSupported) {
      setDatasetStreaming(false);
    }
  }, [datasetStreaming, isStreamingSupported, setDatasetStreaming]);

  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [localDatasets, setLocalDatasets] = useState<LocalDatasetInfo[]>([]);
  const [hasLoadedLocalDatasets, setHasLoadedLocalDatasets] = useState(false);
  const [localLoading, setLocalLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const openPreview = useDatasetPreviewDialogStore((s) => s.openPreview);

  const refreshLocalDatasets = useCallback(async () => {
    setLocalLoading(true);
    setLocalError(null);
    try {
      const response = await listLocalDatasets();
      setLocalDatasets(response.datasets ?? []);
    } catch (error) {
      setLocalError(
        error instanceof Error
          ? error.message
          : translate("studio.dataset.failedToLoadLocalDatasets"),
      );
    } finally {
      setHasLoadedLocalDatasets(true);
      setLocalLoading(false);
    }
  }, []);

  useEffect(() => {
    if (datasetSource !== "upload") return;
    const timer = window.setTimeout(() => {
      void refreshLocalDatasets();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [datasetSource, refreshLocalDatasets]);

  useEffect(() => {
    const handleRefresh = () => {
      if (document.hidden) return;
      if (datasetSource !== "upload") return;
      void refreshLocalDatasets();
    };

    window.addEventListener("focus", handleRefresh);
    document.addEventListener("visibilitychange", handleRefresh);
    return () => {
      window.removeEventListener("focus", handleRefresh);
      document.removeEventListener("visibilitychange", handleRefresh);
    };
  }, [datasetSource, refreshLocalDatasets]);

  const effectiveModelType = modelType ?? "text";
  const isMultimodalModel =
    effectiveModelType === "vision" ||
    effectiveModelType === "audio" ||
    isVisionModel ||
    isAudioModel;

  const selectedLocalDataset = useMemo(() => {
    if (!uploadedFile) return null;
    return localDatasets.find((item) => item.path === uploadedFile) ?? null;
  }, [localDatasets, uploadedFile]);

  useEffect(() => {
    if (!hasLoadedLocalDatasets) return;
    if (localLoading) return;
    if (localError) return;
    if (datasetSource !== "upload") return;
    if (!uploadedFile) return;
    if (selectedLocalDataset) return;
    if (looksLikeLocalPath(uploadedFile)) return;
    if (/\.(jsonl|json|csv|parquet|arrow)$/i.test(uploadedFile)) return;
    selectLocalDataset(null);
  }, [
    datasetSource,
    hasLoadedLocalDatasets,
    localError,
    localLoading,
    uploadedFile,
    selectedLocalDataset,
    selectLocalDataset,
  ]);

  useEffect(() => {
    if (datasetSource === "s3" && isMultimodalModel) {
      selectHfDataset(dataset);
    }
  }, [dataset, datasetSource, isMultimodalModel, selectHfDataset]);

  const activeSourceTab = datasetSource === "upload" ? "local" : "huggingface";
  const isHfDatasetSelected =
    datasetSource === "huggingface" &&
    !!dataset &&
    !isLikelyLocalDatasetRef(dataset);

  const selectedDatasetName =
    datasetSource === "upload" ? uploadedFile : dataset;
  const selectedLocalMetadata = selectedLocalDataset?.metadata ?? null;
  const selectedLocalColumns = selectedLocalMetadata?.columns ?? [];
  const selectedLocalRows =
    selectedLocalDataset?.rows ??
    selectedLocalMetadata?.actual_num_records ??
    null;
  const selectedLocalUpdatedAt = selectedLocalDataset?.updated_at ?? null;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const evalFileInputRef = useRef<HTMLInputElement>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [isDatasetDragOver, setIsDatasetDragOver] = useState(false);
  const [uploadLimitBytes, setUploadLimitBytes] = useState(
    getCachedUploadLimitBytes,
  );
  const [uploadLimitLabel, setUploadLimitLabel] = useState(
    getCachedUploadLimitLabel,
  );
  const [documentRedirectOpen, setDocumentRedirectOpen] = useState(false);
  const [redirectFileName, setRedirectFileName] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const applyLimit = (settings: {
      maxUploadSizeBytes: number;
      maxUploadSizeLabel: string;
    }) => {
      setUploadLimitBytes(settings.maxUploadSizeBytes);
      setUploadLimitLabel(settings.maxUploadSizeLabel);
    };
    const unsubscribe = subscribeUploadLimitSettings(applyLimit);
    void loadUploadLimitSettings()
      .then((settings) => {
        if (!cancelled) applyLimit(settings);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const handleUploadButtonClick = () => {
    fileInputRef.current?.click();
  };

  const getLatestUploadLimit = async () => {
    try {
      const settings = await loadUploadLimitSettings();
      setUploadLimitBytes(settings.maxUploadSizeBytes);
      setUploadLimitLabel(settings.maxUploadSizeLabel);
      return settings;
    } catch {
      return {
        maxUploadSizeBytes: uploadLimitBytes,
        maxUploadSizeLabel: uploadLimitLabel,
      };
    }
  };

  const handleFileUpload = async (
    file: File,
    onSuccess: (storedPath: string) => void,
    successMessage: string,
  ) => {
    const latestLimit = await getLatestUploadLimit();
    if (file.size > latestLimit.maxUploadSizeBytes) {
      toast.error("File too large", {
        description: `${file.name} is ${formatUploadSize(
          file.size,
        )}. Training uploads support up to ${latestLimit.maxUploadSizeLabel}.`,
      });
      return;
    }

    setIsUploading(true);
    try {
      const uploaded = await uploadTrainingDataset(file);
      bumpInventoryVersion();
      onSuccess(uploaded.stored_path);
      toast.success(successMessage, { description: uploaded.filename });
    } catch (error) {
      toast.error(t("studio.dataset.uploadFailed"), {
        description:
          error instanceof Error
            ? error.message
            : t("studio.dataset.unknownError"),
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDatasetFile = async (file: File) => {
    const extension = getFileExtension(file.name);
    if (!TRAINING_UPLOAD_EXTENSION_SET.has(extension)) {
      toast.error(t("studio.dataset.unsupportedFileType"), {
        description: t("studio.dataset.uploadOneFileType", {
          types: TRAINING_UPLOAD_LABEL,
        }),
      });
      return;
    }

    if (DOCUMENT_REDIRECT_EXTENSIONS.has(extension)) {
      setRedirectFileName(file.name);
      setDocumentRedirectOpen(true);
      return;
    }

    await handleFileUpload(
      file,
      selectLocalDataset,
      t("studio.dataset.datasetUploaded"),
    );
  };

  const handleDatasetFileChange = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    await handleDatasetFile(file);
  };

  const handleDatasetDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsDatasetDragOver(false);
    if (isUploading) return;

    const files = Array.from(event.dataTransfer.files);
    if (files.length === 0) return;

    if (files.length > 1) {
      toast.error(t("studio.dataset.uploadOneFileAtATime"), {
        description: t("studio.dataset.uploadSingleFileDescription"),
      });
      return;
    }

    void handleDatasetFile(files[0]);
  };

  const handleDatasetDragOver = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (isUploading) return;

    event.dataTransfer.dropEffect = "copy";
    setIsDatasetDragOver(true);
  };

  const handleDatasetDragLeave = () => {
    setIsDatasetDragOver(false);
  };

  const handleEvalFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    await handleFileUpload(
      file,
      setUploadedEvalFile,
      t("studio.dataset.evalDatasetUploaded"),
    );
  };

  const handleOpenLearningRecipes = useCallback(() => {
    sessionStorage.setItem(OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY, "1");
    setDocumentRedirectOpen(false);
    void navigate({ to: "/data-recipes" });
  }, [navigate]);

  function clearSelection() {
    if (activeSourceTab === "huggingface") {
      selectHfDataset(null);
      return;
    }
    selectLocalDataset(null);
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      {(() => {
        const sourceTabs: {
          value: "huggingface" | "upload" | "s3";
          label: string;
        }[] = [
          { value: "huggingface", label: t("studio.wizard.sourceBrowse") },
          ...(isMultimodalModel
            ? []
            : [{ value: "s3" as const, label: "Amazon S3" }]),
        ];
        if (sourceTabs.length < 2) return null;
        const activeIndex = Math.max(
          0,
          sourceTabs.findIndex((item) =>
            item.value === "huggingface"
              ? datasetSource !== "s3"
              : datasetSource === "s3",
          ),
        );
        return (
          <fieldset
            aria-label={t("studio.dataset.sourceAriaLabel")}
            className="hub-tab-toggle relative m-0 inline-flex h-9 min-w-0 w-full items-center rounded-full border-0 p-0"
          >
            <span
              aria-hidden="true"
              className="hub-tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 rounded-full transition-transform duration-200 ease-out"
              style={{
                width: `${100 / sourceTabs.length}%`,
                transform: `translateX(${activeIndex * 100}%)`,
              }}
            />
            {sourceTabs.map((item) => {
              const checked =
                item.value === "s3"
                  ? datasetSource === "s3"
                  : datasetSource !== "s3";
              return (
                <button
                  key={item.value}
                  type="button"
                  aria-pressed={checked}
                  onClick={() => {
                    if (item.value === "s3") {
                      if (isMultimodalModel || datasetSource === "s3") return;
                      selectS3Source();
                      return;
                    }
                    if (datasetSource !== "s3") return;
                    if (uploadedFile) {
                      selectLocalDataset(uploadedFile);
                    } else {
                      selectHfDataset(dataset);
                    }
                  }}
                  className={cn(
                    "relative z-10 inline-flex h-9 flex-1 cursor-pointer items-center justify-center rounded-full px-3 text-ui-12p5 font-medium transition-colors",
                    checked
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                </button>
              );
            })}
          </fieldset>
        );
      })()}

      {datasetSource === "s3" && <S3ConfigForm />}

      {datasetSource !== "s3" && (
        <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2 lg:gap-5">
          <div className="flex flex-col gap-2">
            <FieldLabel>{t("studio.wizard.hfDatasetLabel")}</FieldLabel>
            <DatasetSelector />
          </div>
          <div className="flex flex-col gap-2">
            <FieldLabel>{t("studio.wizard.uploadLocalLabel")}</FieldLabel>
            <button
              type="button"
              disabled={isUploading}
              onClick={handleUploadButtonClick}
              onDrop={handleDatasetDrop}
              onDragOver={handleDatasetDragOver}
              onDragLeave={handleDatasetDragLeave}
              className={cn(
                "group relative flex h-9 w-full select-none items-center justify-center gap-2 rounded-[12px] border border-dashed px-3 text-center transition-colors",
                "border-foreground/15 dark:border-white/15",
                "hover:border-foreground/30 hover:bg-foreground/[0.02] dark:hover:border-white/30 dark:hover:bg-white/[0.025]",
                isDatasetDragOver &&
                  "border-foreground/45 bg-foreground/[0.04] dark:border-white/40 dark:bg-white/[0.05]",
                isUploading && "cursor-progress opacity-80",
              )}
            >
              {isUploading ? (
                <Spinner className="size-3.5 text-muted-foreground" />
              ) : (
                <HugeiconsIcon
                  icon={CloudUploadIcon}
                  strokeWidth={1.5}
                  className="size-3.5 text-muted-foreground"
                />
              )}
              <span className="truncate text-ui-12p5 text-foreground/85">
                {isUploading
                  ? t("studio.dataset.uploading")
                  : isDatasetDragOver
                    ? t("studio.wizard.releaseToUpload")
                    : t("studio.dataset.dropFileOrClick")}
              </span>
            </button>
            <p className="truncate text-ui-10 text-muted-foreground">
              {TRAINING_DATASET_UPLOAD_LABEL} · up to {uploadLimitLabel};{" "}
              {DOCUMENT_REDIRECT_LABEL}
            </p>
          </div>
        </div>
      )}

      {datasetSource !== "s3" &&
        (isHfDatasetSelected ? (
          <HfDatasetSubsetSplitSelectors
            variant="studio"
            enabled={true}
            datasetName={dataset}
            accessToken={hfApiToken(hfToken)}
            datasetSubset={datasetSubset}
            setDatasetSubset={setDatasetSubset}
            datasetSplit={datasetSplit}
            setDatasetSplit={setDatasetSplit}
            datasetEvalSplit={datasetEvalSplit}
            setDatasetEvalSplit={setDatasetEvalSplit}
          />
        ) : selectedDatasetName ? (
          datasetSource === "upload" && selectedLocalDataset ? (
            <div className="rounded-lg border bg-muted/20 px-3.5 py-3">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">
                    {t("studio.dataset.localDatasetMetadata")}
                  </p>
                  <p className="text-ui-10 text-muted-foreground/80">
                    {t("studio.dataset.dataRecipeOutput")}
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  <MetadataRow
                    label={t("studio.dataset.rows")}
                    value={
                      typeof selectedLocalRows === "number"
                        ? selectedLocalRows.toLocaleString()
                        : "--"
                    }
                  />
                  <MetadataRow
                    label={t("studio.dataset.columns")}
                    value={
                      selectedLocalColumns.length > 0
                        ? String(selectedLocalColumns.length)
                        : "--"
                    }
                  />
                  <MetadataRow
                    label={t("studio.dataset.batches")}
                    value={
                      typeof selectedLocalMetadata?.num_completed_batches ===
                        "number" &&
                      typeof selectedLocalMetadata?.total_num_batches ===
                        "number"
                        ? `${selectedLocalMetadata.num_completed_batches}/${selectedLocalMetadata.total_num_batches}`
                        : "--"
                    }
                  />
                  <MetadataRow
                    label={t("studio.dataset.updated")}
                    value={formatUpdatedDate(selectedLocalUpdatedAt)}
                  />
                </div>
              </div>
            </div>
          ) : null
        ) : (
          <HfDatasetSubsetSplitSelectors
            variant="studio"
            enabled={false}
            datasetName={null}
            accessToken={hfApiToken(hfToken)}
            datasetSubset={datasetSubset}
            setDatasetSubset={setDatasetSubset}
            datasetSplit={datasetSplit}
            setDatasetSplit={setDatasetSplit}
            datasetEvalSplit={datasetEvalSplit}
            setDatasetEvalSplit={setDatasetEvalSplit}
          />
        ))}

      {datasetSource === "upload" && uploadedFile && (
        <div className="rounded-lg border bg-muted/20 px-3.5 py-3">
          <p className="mb-2 text-xs font-medium text-muted-foreground">
            {t("studio.dataset.evalDataset")}
          </p>
          {uploadedEvalFile ? (
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 overflow-hidden">
                <HugeiconsIcon
                  icon={FileAttachmentIcon}
                  className="size-3.5 shrink-0 text-muted-foreground"
                />
                <span className="truncate text-xs">
                  {deriveLocalDatasetName(uploadedEvalFile)}
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 shrink-0 cursor-pointer p-0"
                onClick={() => setUploadedEvalFile(null)}
              >
                <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
              </Button>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <Button
                variant="outline"
                size="sm"
                className="w-full cursor-pointer gap-1.5"
                disabled={isUploading}
                onClick={() => evalFileInputRef.current?.click()}
              >
                {isUploading ? (
                  <Spinner className="size-3.5" />
                ) : (
                  <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
                )}
                {isUploading
                  ? t("studio.dataset.uploading")
                  : t("studio.dataset.uploadEvalFile")}
              </Button>
              <p className="text-ui-10 text-muted-foreground/80">
                {t("studio.dataset.evalDatasetDescription")}
              </p>
            </div>
          )}
        </div>
      )}

      {datasetSource !== "s3" && selectedDatasetName && (
        <div className="flex items-center gap-3 rounded-lg border bg-muted/40 px-3.5 py-3">
          <div className="rounded-md bg-indigo-500/10 p-1.5">
            <HugeiconsIcon
              icon={FileAttachmentIcon}
              className="size-4 text-indigo-500"
            />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-sm font-medium">
              {datasetSource === "upload"
                ? (selectedLocalDataset?.label ??
                  deriveLocalDatasetName(selectedDatasetName))
                : selectedDatasetName}
            </p>
            <p className="text-ui-10 text-muted-foreground">
              {datasetSource === "upload" ? (
                uploadedFile ? (
                  <>
                    {t("studio.dataset.localDataset")}
                    {selectedLocalRows != null
                      ? t("studio.dataset.localDatasetRows", {
                          count: selectedLocalRows.toLocaleString(),
                        })
                      : ""}
                  </>
                ) : (
                  t("studio.dataset.localDataset")
                )
              ) : (
                <>
                  {t("studio.dataset.huggingFaceDataset")}
                  {datasetSubset && ` / ${datasetSubset}`}
                  {datasetSplit && ` / ${datasetSplit}`}
                </>
              )}
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="shrink-0 text-xs"
            onClick={() => openPreview()}
          >
            <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
            {t("studio.dataset.viewDataset")}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="shrink-0 text-xs"
            onClick={clearSelection}
          >
            {t("studio.dataset.clear")}
          </Button>
        </div>
      )}

      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={`size-3.5 transition-transform ${advancedOpen ? "rotate-180" : ""}`}
          />
          {t("studio.dataset.advanced")}
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                {t("studio.dataset.targetFormat")}
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
                    {t("studio.dataset.targetFormatTooltip")}{" "}
                    <a
                      href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline"
                    >
                      {t("studio.params.readMore")}
                    </a>
                  </TooltipContent>
                </Tooltip>
              </span>
              <Select
                value={datasetFormat}
                onValueChange={(v) =>
                  setDatasetFormat(v as typeof datasetFormat)
                }
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">
                    {t("studio.dataset.auto")}
                  </SelectItem>
                  <SelectItem value="alpaca">Alpaca</SelectItem>
                  <SelectItem value="chatml">ChatML</SelectItem>
                  <SelectItem value="sharegpt">ShareGPT</SelectItem>
                  <SelectItem value="raw">
                    {t("studio.dataset.rawText")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="datasetStreaming"
                checked={datasetStreaming}
                disabled={!isStreamingSupported}
                onCheckedChange={(v) => setDatasetStreaming(!!v)}
              />
              <label
                htmlFor="datasetStreaming"
                className={`text-xs text-muted-foreground ${
                  isStreamingSupported
                    ? "cursor-pointer"
                    : "cursor-not-allowed opacity-60"
                }`}
              >
                Enable streaming
              </label>
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
                  {isStreamingSupported ? (
                    <span>
                      Stream Hugging Face text datasets instead of downloading
                      them.
                    </span>
                  ) : (
                    <div className="max-w-xs">
                      <p className="font-medium">
                        Streaming unavailable. To enable:
                      </p>
                      <ul className="mt-1 list-disc space-y-0.5 pl-4">
                        {streamingBlockers.map((reason) => (
                          <li key={reason}>{reason}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  {t("studio.dataset.trainSplitStart")}
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
                      {t("studio.dataset.trainSplitStartTooltip")}
                    </TooltipContent>
                  </Tooltip>
                </span>
                <Input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  step={1}
                  placeholder="0"
                  value={datasetSliceStart ?? ""}
                  onChange={(e) =>
                    setDatasetSliceStart(normalizeSliceInput(e.target.value))
                  }
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  {t("studio.dataset.trainSplitEnd")}
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
                      {t("studio.dataset.trainSplitEndTooltip")}
                    </TooltipContent>
                  </Tooltip>
                </span>
                <Input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  step={1}
                  placeholder={t("studio.dataset.endPlaceholder")}
                  value={datasetSliceEnd ?? ""}
                  onChange={(e) =>
                    setDatasetSliceEnd(normalizeSliceInput(e.target.value))
                  }
                />
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      <input
        ref={fileInputRef}
        type="file"
        accept={TRAINING_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(event) => {
          void handleDatasetFileChange(event);
        }}
      />
      <input
        ref={evalFileInputRef}
        type="file"
        accept=".json,.jsonl,.csv,.parquet"
        className="hidden"
        onChange={(event) => {
          void handleEvalFileChange(event);
        }}
      />
      <DocumentUploadRedirectDialog
        open={documentRedirectOpen}
        onOpenChange={setDocumentRedirectOpen}
        fileName={redirectFileName}
        onOpenLearningRecipes={handleOpenLearningRecipes}
      />
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2 rounded-md bg-background/60 px-2 py-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </div>
  );
}
