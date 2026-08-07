// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { datasetDisplayName } from "@/components/resource-picker/dataset-display-name";
import { cacheLocalPathMatchesSelection } from "@/features/training";

export { datasetDisplayName };

export interface DatasetDisplayCandidate {
  path: string;
  title: string;
}

export function datasetSelectionDisplayName({
  source,
  dataset,
  uploadedFile,
  candidates,
}: {
  source: "huggingface" | "upload" | "s3";
  dataset: string | null;
  uploadedFile: string | null;
  candidates: readonly DatasetDisplayCandidate[];
}): string | null {
  if (source === "upload") {
    if (!uploadedFile) {
      return null;
    }
    const title = candidates.find((candidate) =>
      cacheLocalPathMatchesSelection(candidate.path, uploadedFile),
    )?.title;
    return title ?? datasetDisplayName(uploadedFile);
  }
  return dataset ? datasetDisplayName(dataset) : null;
}
