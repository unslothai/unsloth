// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "@/features/hub";
import { hasSeparateStreamingEvalSplit } from "@/features/training";
import type { DatasetSource } from "@/types/training";

const LOCAL_DATASET_FILE_PATTERN = /\.(jsonl|json|csv|parquet)$/i;
const LOCAL_DATASET_REFERENCE_PATTERN = /\.(jsonl|json|csv|parquet|arrow)$/i;
const LOCAL_DATASET_UUID_PREFIX_PATTERN = /^[a-f0-9]{32}_(.+)$/;
const POSITIVE_INTEGER_PATTERN = /^\d+$/;

export function getFileExtension(fileName: string): string {
  const extensionStart = fileName.lastIndexOf(".");
  return extensionStart >= 0
    ? fileName.slice(extensionStart).toLowerCase()
    : "";
}

export function isLikelyLocalDatasetRef(value: string): boolean {
  return (
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.includes("\\") ||
    LOCAL_DATASET_FILE_PATTERN.test(value)
  );
}

export function deriveLocalDatasetName(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  const parquetIndex = parts.lastIndexOf("parquet-files");
  if (parquetIndex > 0) {
    return parts[parquetIndex - 1];
  }
  const basename = parts[parts.length - 1] ?? path;
  const uuidPrefixMatch = basename.match(LOCAL_DATASET_UUID_PREFIX_PATTERN);
  if (uuidPrefixMatch) {
    return uuidPrefixMatch[1];
  }
  return basename;
}

export function formatUpdatedDate(timestamp: number | null): string {
  if (typeof timestamp !== "number") {
    return "--";
  }
  return new Date(timestamp * 1000).toLocaleDateString();
}

export function normalizeSliceInput(value: string): string | null {
  const trimmed = value.trim();
  return POSITIVE_INTEGER_PATTERN.test(trimmed) ? trimmed : null;
}

export function getDatasetStreamingBlockers({
  datasetEvalSplit,
  datasetSource,
  datasetSplit,
  evalSteps,
  isAppleSilicon,
  isAudioModel,
  isDatasetAudio,
  isDatasetImage,
  isEmbeddingModel,
  isVisionModel,
  maxSteps,
  trainOnCompletions,
}: {
  datasetEvalSplit: string | null;
  datasetSource: DatasetSource;
  datasetSplit: string | null;
  evalSteps: number;
  isAppleSilicon: boolean;
  isAudioModel: boolean;
  isDatasetAudio: boolean;
  isDatasetImage: boolean | null;
  isEmbeddingModel: boolean;
  isVisionModel: boolean;
  maxSteps: number;
  trainOnCompletions: boolean;
}): string[] {
  const blockers: string[] = [];
  if (datasetSource !== "huggingface") {
    blockers.push(
      "Use a Hugging Face dataset (not a local upload or S3 source).",
    );
  }
  if (maxSteps <= 0) {
    blockers.push(
      "Set Max Steps > 0 — streaming datasets have no known length.",
    );
  }
  if (trainOnCompletions) {
    blockers.push('Turn off "Assistant completions only".');
  }
  if (
    !hasSeparateStreamingEvalSplit({
      evalSteps,
      datasetSplit,
      datasetEvalSplit,
    })
  ) {
    blockers.push(
      "Pick a separate eval split — evaluation is on but no distinct eval split is set.",
    );
  }
  if (isVisionModel) {
    blockers.push("Vision models don't support streaming.");
  }
  if (isAudioModel) {
    blockers.push("Audio models don't support streaming.");
  }
  if (isEmbeddingModel) {
    blockers.push(
      "Embedding models don't support streaming (training needs the full dataset).",
    );
  }
  if (isDatasetImage) {
    blockers.push("This dataset looks like images, which can't stream.");
  }
  if (isDatasetAudio) {
    blockers.push("This dataset looks like audio, which can't stream.");
  }
  if (isAppleSilicon) {
    blockers.push("Streaming isn't supported on Apple Silicon (MLX) yet.");
  }
  return blockers;
}

export function shouldClearMissingLocalSelection({
  datasetSource,
  hasLoadedLocalDatasets,
  hasSelectedLocalDataset,
  localError,
  localLoading,
  uploadedFile,
}: {
  datasetSource: DatasetSource;
  hasLoadedLocalDatasets: boolean;
  hasSelectedLocalDataset: boolean;
  localError: string | null;
  localLoading: boolean;
  uploadedFile: string | null;
}): boolean {
  if (
    !hasLoadedLocalDatasets ||
    localLoading ||
    localError ||
    datasetSource !== "upload" ||
    !uploadedFile ||
    hasSelectedLocalDataset
  ) {
    return false;
  }
  return !(
    looksLikeLocalPath(uploadedFile) ||
    LOCAL_DATASET_REFERENCE_PATTERN.test(uploadedFile)
  );
}
