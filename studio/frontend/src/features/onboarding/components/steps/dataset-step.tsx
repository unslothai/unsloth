// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
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
import { DatasetSelector, datasetDisplayName } from "@/features/dataset-picker";
import {
  bumpInventoryVersion,
  hfApiToken,
  useHfTokenStore,
  useOnlineStatus,
} from "@/features/hub";
import {
  formatUploadSize,
  getCachedUploadLimitBytes,
  getCachedUploadLimitLabel,
  loadUploadLimitSettings,
} from "@/features/settings";
import {
  HfDatasetSubsetSplitSelectors,
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
  uploadTrainingDataset,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import type { DatasetFormat } from "@/types/training";
import {
  InformationCircleIcon,
  SparklesIcon,
  Upload04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { HfTokenField } from "../hf-token-field";

const FORMAT_OPTIONS: { value: DatasetFormat; label: string }[] = [
  { value: "auto", label: "Auto Detect" },
  { value: "alpaca", label: "Alpaca" },
  { value: "chatml", label: "ChatML" },
  { value: "sharegpt", label: "ShareGPT" },
  { value: "raw", label: "Raw Text" },
];

export function DatasetStep() {
  const t = useT();
  const hfToken = useHfTokenStore((state) => state.token);
  const online = useOnlineStatus();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const {
    datasetSource,
    datasetFormat,
    setDatasetFormat,
    dataset,
    datasetKnownCached,
    datasetLocalPath,
    datasetStreaming,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
    datasetEvalSplit,
    setDatasetEvalSplit,
    setManualDatasetOptionsValid,
    markManualDatasetOptionsEdited,
    uploadedFile,
    selectLocalDataset,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      datasetSource: state.datasetSource,
      datasetFormat: state.datasetFormat,
      setDatasetFormat: state.setDatasetFormat,
      dataset: state.dataset,
      datasetKnownCached: state.datasetKnownCached,
      datasetLocalPath: state.datasetLocalPath,
      datasetStreaming: state.datasetStreaming,
      datasetSubset: state.datasetSubset,
      setDatasetSubset: state.setDatasetSubset,
      datasetSplit: state.datasetSplit,
      setDatasetSplit: state.setDatasetSplit,
      datasetEvalSplit: state.datasetEvalSplit,
      setDatasetEvalSplit: state.setDatasetEvalSplit,
      setManualDatasetOptionsValid: state.setManualDatasetOptionsValid,
      markManualDatasetOptionsEdited: state.markManualDatasetOptionsEdited,
      uploadedFile: state.uploadedFile,
      selectLocalDataset: state.selectLocalDataset,
    })),
  );

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    try {
      const uploadLimit = await loadUploadLimitSettings().catch(() => ({
        maxUploadSizeBytes: getCachedUploadLimitBytes(),
        maxUploadSizeLabel: getCachedUploadLimitLabel(),
      }));
      if (file.size > uploadLimit.maxUploadSizeBytes) {
        toast.error(t("studio.dataset.fileTooLarge"), {
          description: t("studio.dataset.fileTooLargeDescription", {
            file: file.name,
            size: formatUploadSize(file.size),
            limit: uploadLimit.maxUploadSizeLabel,
          }),
        });
        return;
      }
      const uploaded = await uploadTrainingDataset(file);
      bumpInventoryVersion();
      selectLocalDataset(uploaded.stored_path);
      toast.success(t("studio.dataset.datasetUploaded"), {
        description: uploaded.filename,
      });
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

  return (
    <FieldGroup>
      <HfTokenField />

      <Field>
        <FieldLabel>{t("studio.training.chooseDataset")}</FieldLabel>
        <FieldDescription>
          {t("studio.wizard.datasetPickerDescription")}
        </FieldDescription>
        <DatasetSelector />
      </Field>

      {datasetSource === "huggingface" && (
        <HfDatasetSubsetSplitSelectors
          variant="wizard"
          enabled={true}
          datasetName={dataset}
          accessToken={hfApiToken(hfToken)}
          localPath={datasetLocalPath}
          online={online}
          preferLocalCache={datasetKnownCached && !datasetStreaming}
          datasetSubset={datasetSubset}
          setDatasetSubset={setDatasetSubset}
          datasetSplit={datasetSplit}
          setDatasetSplit={setDatasetSplit}
          datasetEvalSplit={datasetEvalSplit}
          setDatasetEvalSplit={setDatasetEvalSplit}
          datasetStreaming={datasetStreaming}
          setManualDatasetOptionsValid={setManualDatasetOptionsValid}
          markManualDatasetOptionsEdited={markManualDatasetOptionsEdited}
        />
      )}

      <Field>
        <FieldLabel>{t("studio.wizard.uploadDataset")}</FieldLabel>
        <FieldDescription>
          {t("studio.wizard.uploadDatasetDescription")}
        </FieldDescription>
        <input
          ref={fileInputRef}
          type="file"
          accept={TRAINING_DATASET_UPLOAD_EXTENSIONS.join(",")}
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) {
              uploadFile(file).catch(() => undefined);
            }
          }}
        />
        <Button
          type="button"
          variant="outline"
          className="w-full"
          disabled={isUploading}
          onClick={() => fileInputRef.current?.click()}
        >
          {isUploading ? (
            <Spinner className="size-4" />
          ) : (
            <HugeiconsIcon icon={Upload04Icon} data-icon="inline-start" />
          )}
          {isUploading
            ? t("studio.dataset.uploading")
            : t("studio.wizard.chooseFile")}
        </Button>
        {datasetSource === "upload" && uploadedFile && (
          <Badge variant="secondary" className="w-fit max-w-full truncate">
            {datasetDisplayName(uploadedFile)}
          </Badge>
        )}
      </Field>

      <Field>
        <div className="flex items-center justify-between gap-4">
          <FieldLabel className="flex items-center gap-1.5">
            {t("studio.wizard.format")}
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  aria-label={t("studio.wizard.format")}
                  className="text-muted-foreground/50 hover:text-muted-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
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
          </FieldLabel>
          <Select
            value={datasetFormat}
            onValueChange={(value) => setDatasetFormat(value as DatasetFormat)}
          >
            <SelectTrigger className="w-44 shrink-0">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FORMAT_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.value === "auto" && (
                    <HugeiconsIcon
                      icon={SparklesIcon}
                      className="mr-1.5 inline size-3.5 align-text-bottom"
                    />
                  )}
                  {option.value === "auto"
                    ? t("studio.wizard.autoDetect")
                    : option.value === "raw"
                      ? t("studio.dataset.rawText")
                      : option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </Field>
    </FieldGroup>
  );
}
