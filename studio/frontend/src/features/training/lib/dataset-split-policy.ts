// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingConfigState } from "../types/config";

type DatasetSplitState = Pick<
  TrainingConfigState,
  "datasetKnownCached" | "datasetSource" | "datasetSplit" | "datasetStreaming"
>;

export function requiresExplicitCachedDatasetSplit(
  state: DatasetSplitState,
): boolean {
  return (
    state.datasetSource === "huggingface" &&
    state.datasetKnownCached &&
    !state.datasetStreaming &&
    !state.datasetSplit?.trim()
  );
}
