// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DatasetSource } from "@/types/training";
import { isValidHubResourceId } from "../../../components/resource-picker/hub-resource-id.ts";
import { looksLikeLocalPath } from "../../../lib/local-path.ts";
import { isTrainingDatasetUploadPath } from "./native-dataset-drop.ts";

export type DeletedLocalDatasetSelection = "huggingface" | "upload";

export function resolveDeletedLocalDatasetSelection({
  datasetName,
  source,
  dataset,
  uploadedFile,
}: {
  datasetName: string;
  source: DatasetSource;
  dataset: string | null;
  uploadedFile: string | null;
}): DeletedLocalDatasetSelection | null {
  if (!looksLikeLocalPath(datasetName)) {
    return null;
  }
  if (source === "upload" && uploadedFile === datasetName) {
    return "upload";
  }
  if (source === "huggingface" && dataset === datasetName) {
    return "huggingface";
  }
  return null;
}

export function isHuggingFaceDatasetSelected(
  source: DatasetSource,
  dataset: string | null,
): boolean {
  const selectedDataset = dataset?.trim();
  if (source !== "huggingface" || !selectedDataset) {
    return false;
  }
  return isValidHubResourceId(selectedDataset);
}

export function shouldClearMissingLocalDatasetSelection({
  source,
  selectedPath,
  inventorySettled,
  inventoryMatchFound,
}: {
  source: DatasetSource;
  selectedPath: string | null;
  inventorySettled: boolean;
  inventoryMatchFound: boolean;
}): boolean {
  return (
    source === "upload" &&
    Boolean(selectedPath) &&
    inventorySettled &&
    !inventoryMatchFound &&
    !isTrainingDatasetUploadPath(selectedPath ?? "")
  );
}
