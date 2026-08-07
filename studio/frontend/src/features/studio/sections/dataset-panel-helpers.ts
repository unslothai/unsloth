


import { hasSeparateStreamingEvalSplit } from "@/features/training";
import type { DatasetSource } from "@/types/training";

const POSITIVE_INTEGER_PATTERN = /^\d+$/;

export type DatasetStreamingBlocker =
  | "source"
  | "maxSteps"
  | "trainOnCompletions"
  | "evalSplit"
  | "visionModel"
  | "audioModel"
  | "embeddingModel"
  | "imageDataset"
  | "audioDataset"
  | "appleSilicon";

export function getFileExtension(fileName: string): string {
  const extensionStart = fileName.lastIndexOf(".");
  return extensionStart >= 0
    ? fileName.slice(extensionStart).toLowerCase()
    : "";
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
}): DatasetStreamingBlocker[] {
  const blockers: DatasetStreamingBlocker[] = [];
  if (datasetSource !== "huggingface") {
    blockers.push("source");
  }
  if (maxSteps <= 0) {
    blockers.push("maxSteps");
  }
  if (trainOnCompletions) {
    blockers.push("trainOnCompletions");
  }
  if (
    !hasSeparateStreamingEvalSplit({
      evalSteps,
      datasetSplit,
      datasetEvalSplit,
    })
  ) {
    blockers.push("evalSplit");
  }
  if (isVisionModel) {
    blockers.push("visionModel");
  }
  if (isAudioModel) {
    blockers.push("audioModel");
  }
  if (isEmbeddingModel) {
    blockers.push("embeddingModel");
  }
  if (isDatasetImage) {
    blockers.push("imageDataset");
  }
  if (isDatasetAudio) {
    blockers.push("audioDataset");
  }
  if (isAppleSilicon) {
    blockers.push("appleSilicon");
  }
  return blockers;
}
