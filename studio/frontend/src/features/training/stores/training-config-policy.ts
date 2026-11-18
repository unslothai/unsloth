// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_HYPERPARAMS } from "@/config/training";
import { getLocale, translate } from "@/i18n";
import type { DatasetFormat } from "@/types/training";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import type {
  BrowseDatasetSelection,
  DatasetCacheReferenceOptions,
  TrainingConfigState,
} from "../types/config";
export function emptyManualMapping(): TrainingConfigState["datasetManualMapping"] {
  return {};
}

export function createHfBrowseDatasetSelection(
  dataset: string | null,
  options?: DatasetCacheReferenceOptions,
): Extract<BrowseDatasetSelection, { source: "huggingface" }> {
  return {
    source: "huggingface",
    dataset,
    knownCached: dataset ? (options?.knownCached ?? false) : false,
    localPath: dataset ? (options?.localPath ?? null) : null,
  };
}

export function datasetSelectionStreamingPatch(
  selection: Extract<BrowseDatasetSelection, { source: "huggingface" }>,
  options?: DatasetCacheReferenceOptions,
): Partial<Pick<TrainingConfigState, "datasetStreaming">> {
  return selection.knownCached && options?.preferLocalCache
    ? { datasetStreaming: false }
    : {};
}

export function datasetSourceInvariantPatch(
  state: Pick<TrainingConfigState, "datasetSource" | "datasetStreaming">,
): Partial<Pick<TrainingConfigState, "datasetStreaming">> {
  return state.datasetSource !== "huggingface" && state.datasetStreaming
    ? { datasetStreaming: false }
    : {};
}

export function createUploadBrowseDatasetSelection(
  uploadedFile: string | null,
): Extract<BrowseDatasetSelection, { source: "upload" }> {
  return {
    source: "upload",
    uploadedFile,
  };
}

export const initialTrainingConfigState: TrainingConfigState = {
  userEditRevision: 0,
  modelType: null,
  selectedModel: null,
  modelKnownCached: false,
  modelLocalPath: null,
  modelFormat: null,
  modelSubfolder: null,
  decisionCheckpoints: null,
  trainAsDecision: false,
  decisionLayout: null,
  settingsBeforeDecision: null,
  projectName: "",
  trainingMethod: "qlora",
  trainingMethodProvenance: {
    learningRateManuallySet: false,
    modelAdapterLearningRate: null,
    datasetFormatBeforeCpt: null,
    targetModulesBeforeCpt: null,
    loraRankBeforeCpt: null,
    loraAlphaBeforeCpt: null,
    loraVariantBeforeCpt: null,
    trainOnCompletionsBeforeCpt: null,
  },
  datasetSource: "huggingface",
  browseDatasetSelection: createHfBrowseDatasetSelection(null),
  datasetFormat: "auto",
  dataset: null,
  datasetKnownCached: false,
  datasetLocalPath: null,
  datasetSubset: null,
  datasetSplit: null,
  datasetEvalSplit: null,
  datasetStreaming: false,
  manualDatasetOptionsValid: true,
  datasetManualMapping: emptyManualMapping(),
  datasetSystemPrompt: "",
  datasetLabelMapping: {},
  datasetAdvisorNotification: null,
  datasetSliceStart: null,
  datasetSliceEnd: null,
  uploadedFile: null,
  uploadedEvalFile: null,
  isCheckingVision: false,
  isVisionModel: false,
  isEmbeddingModel: false,
  isAudioModel: false,
  audioCapabilityUnknown: false,
  isLoadingModelDefaults: false,
  modelDefaultsError: null,
  modelDefaultsAppliedFor: null,
  advancedSettingsBaseline: null,
  trainOnCompletionsDefaultPendingFor: null,
  trainingToggleChoices: {},
  isCheckingDataset: false,
  isDatasetImage: null,
  isDatasetAudio: false,
  datasetCheckFailed: false,
  maxPositionEmbeddings: null,
  drafterCapability: null,
  ...DEFAULT_HYPERPARAMS,
};
export function hasSeparateStreamingEvalSplit(
  state: Pick<
    TrainingConfigState,
    "evalSteps" | "datasetSplit" | "datasetEvalSplit"
  >,
): boolean {
  if (state.evalSteps <= 0) {
    return true;
  }
  const trainSplit = state.datasetSplit || "train";
  return !!state.datasetEvalSplit && state.datasetEvalSplit !== trainSplit;
}

export function streamingCompatiblePatch(
  state: TrainingConfigState,
): Partial<TrainingConfigState> {
  const patch: Partial<TrainingConfigState> = {};

  if (state.datasetStreaming && state.maxSteps <= 0) {
    patch.datasetStreaming = false;
  }

  const willStream =
    patch.datasetStreaming !== undefined
      ? patch.datasetStreaming
      : state.datasetStreaming;

  if (willStream && state.trainOnCompletions) {
    patch.trainOnCompletions = false;
  }

  if (willStream && !hasSeparateStreamingEvalSplit(state)) {
    patch.evalSteps = 0;
  }

  return patch;
}

export function resolveDeferredTrainOnCompletionsDefault({
  currentValue,
  datasetFormat,
  datasetStreaming,
  isEmbeddingModel,
  modelDefault,
  trainingMethod,
}: {
  currentValue: boolean;
  datasetFormat: DatasetFormat;
  datasetStreaming: boolean;
  isEmbeddingModel: boolean;
  modelDefault: boolean | undefined;
  trainingMethod: TrainingConfigState["trainingMethod"];
}): boolean {
  if (
    datasetStreaming ||
    trainingMethod === "cpt" ||
    isRawTextDatasetFormat(datasetFormat) ||
    isEmbeddingModel
  ) {
    return false;
  }
  return modelDefault ?? currentValue;
}

export function formatStreamingDisabledOptions(
  trainOnCompletions: boolean,
  evaluation: boolean,
): string {
  const options: string[] = [];
  if (trainOnCompletions) {
    options.push(
      translate("studio.dataset.streaming.options.trainOnCompletions"),
    );
  }
  if (evaluation) {
    options.push(translate("studio.dataset.streaming.options.evaluation"));
  }
  if (typeof Intl.ListFormat !== "function") {
    return options.join(", ");
  }
  return new Intl.ListFormat(getLocale(), {
    style: "long",
    type: "conjunction",
  }).format(options);
}


/** Capabilities and training objectives decide these defaults, never a model-family whitelist. */
export function trainingToggleDefaults(state: Pick<TrainingConfigState,
  "isVisionModel" | "isAudioModel" | "isEmbeddingModel" | "isDatasetImage" |
  "isDatasetAudio" | "modelType" | "datasetStreaming" | "datasetFormat" | "trainingMethod"
>): { packing: boolean; trainOnCompletions: boolean } {
  const decisionOrEmbedding = state.isEmbeddingModel || state.modelType === "decision";
  const standaloneAudio = state.isAudioModel && !state.isVisionModel;
  return {
    // The image/audio collators consume examples directly; they do not use text sequence packing.
    packing: !decisionOrEmbedding && !standaloneAudio &&
      !(state.isVisionModel && (state.isDatasetImage === true || state.isDatasetAudio)),
    trainOnCompletions: !decisionOrEmbedding && !standaloneAudio &&
      !state.datasetStreaming && state.trainingMethod !== "cpt" &&
      !isRawTextDatasetFormat(state.datasetFormat),
  };
}

export function resolveTrainingToggleDefaults(state: Parameters<typeof trainingToggleDefaults>[0] &
  Pick<TrainingConfigState, "trainingToggleChoices">
): { packing: boolean; trainOnCompletions: boolean } {
  const defaults = trainingToggleDefaults(state);
  return {
    packing: defaults.packing && (state.trainingToggleChoices?.packing ?? true),
    trainOnCompletions: defaults.trainOnCompletions &&
      (state.trainingToggleChoices?.trainOnCompletions ?? true),
  };
}
