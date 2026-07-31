// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_FOCUS_VISIBLE_CLASS } from "@/components/resource-picker/picker-focus";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { usePlatformStore } from "@/config/env";
import { DatasetSelector, datasetDisplayName } from "@/features/dataset-picker";
import {
  bumpInventoryVersion,
  hfApiToken,
  useDeviceInventorySources,
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
  cacheLocalPathMatchesSelection,
  isHuggingFaceDatasetSelected,
  uploadTrainingDataset,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  CloudUploadIcon,
  FileAttachmentIcon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import {
  DatasetAdvancedSettings,
  DatasetSourceToggle,
} from "./dataset-panel-controls";
import {
  formatUpdatedDate,
  getDatasetStreamingBlockers,
  getFileExtension,
} from "./dataset-panel-helpers";
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
const DOCUMENT_REDIRECT_EXTENSIONS = new Set([".pdf", ".docx", ".txt"]);

const OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY =
  "data-recipes:open-learning-recipes";

function FieldLabel({ children }: { children: ReactNode }) {
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
    restoreBrowseDatasetSource,
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
      restoreBrowseDatasetSource: s.restoreBrowseDatasetSource,
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
  const {
    localDatasets: localDatasetInventory,
    refresh: refreshLocalDatasets,
  } = useDeviceInventorySources(["localDatasets"], {
    enabled: datasetSource === "upload",
  });
  const wasUploadSource = useRef(false);
  const localDatasets = localDatasetInventory.rows;
  const platformDeviceType = usePlatformStore((s) => s.deviceType);
  const streamingBlockers = getDatasetStreamingBlockers({
    datasetEvalSplit,
    datasetSource,
    datasetSplit,
    evalSteps,
    isAppleSilicon: platformDeviceType === "mac",
    isAudioModel,
    isDatasetAudio,
    isDatasetImage,
    isEmbeddingModel,
    isVisionModel,
    maxSteps,
    trainOnCompletions,
  });

  const isStreamingSupported = streamingBlockers.length === 0;

  useEffect(() => {
    if (datasetStreaming && !isStreamingSupported) {
      setDatasetStreaming(false);
      if (isDatasetImage || isDatasetAudio) {
        toast.info(
          t(
            "studio.dataset.streaming.notifications.disabledForDetectedModality",
          ),
        );
      }
    }
  }, [
    datasetStreaming,
    isDatasetAudio,
    isDatasetImage,
    isStreamingSupported,
    setDatasetStreaming,
    t,
  ]);

  const openPreview = useDatasetPreviewDialogStore((s) => s.openPreview);

  useEffect(() => {
    const isUploadSource = datasetSource === "upload";
    if (
      isUploadSource &&
      !wasUploadSource.current &&
      localDatasetInventory.ready
    ) {
      void refreshLocalDatasets();
    }
    wasUploadSource.current = isUploadSource;
  }, [datasetSource, localDatasetInventory.ready, refreshLocalDatasets]);

  useEffect(() => {
    const handleRefresh = () => {
      if (document.hidden || datasetSource !== "upload") {
        return;
      }
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
    return (
      localDatasets.find((item) =>
        cacheLocalPathMatchesSelection(item.path, uploadedFile),
      ) ?? null
    );
  }, [localDatasets, uploadedFile]);

  useEffect(() => {
    if (datasetSource === "s3" && isMultimodalModel) {
      restoreBrowseDatasetSource();
    }
  }, [datasetSource, isMultimodalModel, restoreBrowseDatasetSource]);

  const activeSourceTab = datasetSource === "upload" ? "local" : "huggingface";
  const isHfDatasetSelected = isHuggingFaceDatasetSelected(
    datasetSource,
    dataset,
  );

  const selectedDatasetName =
    datasetSource === "upload" ? uploadedFile : dataset;
  const selectedLocalRecipe =
    selectedLocalDataset?.source === "recipe" ? selectedLocalDataset : null;
  const selectedLocalMetadata = selectedLocalRecipe?.metadata ?? null;
  const selectedLocalColumns = selectedLocalMetadata?.columns ?? [];
  const selectedLocalRows =
    selectedLocalRecipe?.rows ??
    selectedLocalMetadata?.actual_num_records ??
    null;
  const selectedLocalUpdatedAt = selectedLocalRecipe?.updated_at ?? null;

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
      toast.error(t("studio.dataset.fileTooLarge"), {
        description: t("studio.dataset.fileTooLargeDescription", {
          file: file.name,
          size: formatUploadSize(file.size),
          limit: latestLimit.maxUploadSizeLabel,
        }),
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
      <DatasetSourceToggle
        datasetSource={datasetSource}
        isMultimodalModel={isMultimodalModel}
        restoreBrowseDatasetSource={restoreBrowseDatasetSource}
        selectS3Source={selectS3Source}
      />

      {datasetSource === "s3" && <S3ConfigForm />}

      {datasetSource !== "s3" && (
        <div className="grid grid-cols-1 items-start gap-4 @xl/train-section:grid-cols-2 @xl/train-section:gap-5">
          <div className="flex flex-col gap-2">
            <FieldLabel>{t("studio.wizard.datasetLabel")}</FieldLabel>
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
                PICKER_FOCUS_VISIBLE_CLASS,
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
              {t("studio.dataset.uploadLimitsHint", {
                limit: uploadLimitLabel,
              })}
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
          datasetSource === "upload" && selectedLocalRecipe ? (
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
                  {datasetDisplayName(uploadedEvalFile)}
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                aria-label={`${t("studio.dataset.clear")} ${t(
                  "studio.dataset.evalDataset",
                )}`}
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
        <div className="flex flex-col items-stretch gap-3 rounded-lg border bg-muted/40 px-3.5 py-3 @md/train-section:flex-row @md/train-section:items-center">
          <div className="flex min-w-0 items-center gap-3">
            <div className="shrink-0 rounded-md bg-indigo-500/10 p-1.5">
              <HugeiconsIcon
                icon={FileAttachmentIcon}
                className="size-4 text-indigo-500"
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate font-mono text-sm font-medium">
                {datasetSource === "upload"
                  ? (selectedLocalDataset?.label ??
                    datasetDisplayName(selectedDatasetName))
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
          </div>
          <div className="flex flex-wrap items-center justify-end gap-1 @md/train-section:ml-auto @md/train-section:flex-nowrap">
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
        </div>
      )}

      <DatasetAdvancedSettings
        datasetFormat={datasetFormat}
        datasetSliceEnd={datasetSliceEnd}
        datasetSliceStart={datasetSliceStart}
        datasetStreaming={datasetStreaming}
        isStreamingSupported={isStreamingSupported}
        setDatasetFormat={setDatasetFormat}
        setDatasetSliceEnd={setDatasetSliceEnd}
        setDatasetSliceStart={setDatasetSliceStart}
        setDatasetStreaming={setDatasetStreaming}
        streamingBlockers={streamingBlockers}
      />

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
