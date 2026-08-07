


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
