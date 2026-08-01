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
} from "@/features/hub";
import {
  HfDatasetSubsetSplitSelectors,
  uploadTrainingDataset,
  useTrainingConfigStore,
} from "@/features/training";
import { TRAINING_DATASET_UPLOAD_EXTENSIONS } from "@/features/training/lib/native-dataset-drop";
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

const FORMAT_OPTIONS: { value: DatasetFormat; label: string }[] = [
  { value: "auto", label: "Auto Detect" },
  { value: "alpaca", label: "Alpaca" },
  { value: "chatml", label: "ChatML" },
  { value: "sharegpt", label: "ShareGPT" },
  { value: "raw", label: "Raw Text" },
];

export function DatasetStep() {
  const hfToken = useHfTokenStore((state) => state.token);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const {
    datasetSource,
    datasetFormat,
    setDatasetFormat,
    dataset,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
    datasetEvalSplit,
    setDatasetEvalSplit,
    uploadedFile,
    selectLocalDataset,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      datasetSource: state.datasetSource,
      datasetFormat: state.datasetFormat,
      setDatasetFormat: state.setDatasetFormat,
      dataset: state.dataset,
      datasetSubset: state.datasetSubset,
      setDatasetSubset: state.setDatasetSubset,
      datasetSplit: state.datasetSplit,
      setDatasetSplit: state.setDatasetSplit,
      datasetEvalSplit: state.datasetEvalSplit,
      setDatasetEvalSplit: state.setDatasetEvalSplit,
      uploadedFile: state.uploadedFile,
      selectLocalDataset: state.selectLocalDataset,
    })),
  );

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    try {
      const uploaded = await uploadTrainingDataset(file);
      bumpInventoryVersion();
      selectLocalDataset(uploaded.stored_path);
      toast.success("Dataset uploaded", { description: uploaded.filename });
    } catch (error) {
      toast.error("Upload failed", {
        description:
          error instanceof Error ? error.message : "Unknown upload error",
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <FieldGroup>
      <Field>
        <FieldLabel>Choose a dataset</FieldLabel>
        <FieldDescription>
          Search Hugging Face or choose a dataset already on this device.
        </FieldDescription>
        <DatasetSelector />
      </Field>

      {datasetSource === "huggingface" && (
        <HfDatasetSubsetSplitSelectors
          variant="wizard"
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
      )}

      <Field>
        <FieldLabel>Upload a dataset</FieldLabel>
        <FieldDescription>
          Supports CSV, JSONL, JSON, and Parquet.
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
          {isUploading ? "Uploading..." : "Choose a file"}
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
            Format
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-muted-foreground/50 hover:text-muted-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Auto detects and converts common dataset formats.{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  Read more
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
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </Field>
    </FieldGroup>
  );
}
