// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DatasetSource } from "@/types/training";
import { isValidHubResourceId } from "../../../components/resource-picker/hub-resource-id.ts";

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
