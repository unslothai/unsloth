// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelInventoryFormat } from "@/features/hub";
import type {
  DatasetFormat,
  DatasetSource,
  GradientCheckpointing,
  ModelType,
  S3Config,
  StepNumber,
  TrainingMethod,
} from "@/types/training";
import type { BackendModelConfig } from "../api/models-api";

export type LoraVariant = "lora" | "rslora" | "loftq" | "dora";

export interface ModelCacheReferenceOptions {
  knownCached?: boolean;
  localPath?: string | null;
  modelFormat?: ModelInventoryFormat | null;
}

export interface TrainingModelSelectionOptions
  extends ModelCacheReferenceOptions {
  isEmbedding?: boolean | null;
  isAudio?: boolean | null;
  isVision?: boolean | null;
}

export interface DatasetCacheReferenceOptions {
  knownCached?: boolean;
  localPath?: string | null;
  preferLocalCache?: boolean;
}

export type BrowseDatasetSelection =
  | {
      source: "huggingface";
      dataset: string | null;
      knownCached: boolean;
      localPath: string | null;
    }
  | {
      source: "upload";
      uploadedFile: string | null;
    };

export interface TrainingMethodProvenance {
  learningRateManuallySet: boolean;
  modelAdapterLearningRate: number | null;
  datasetFormatBeforeCpt: DatasetFormat | null;
}

/** Column-to-role mapping, e.g. { "problem": "user", "solution": "assistant", "context": "system" } */
export type DatasetManualMapping = Record<string, string>;

export interface TrainingConfigState {
  userEditRevision: number;
  currentStep: StepNumber;
  modelType: ModelType | null;
  selectedModel: string | null;
  modelKnownCached: boolean;
  modelLocalPath: string | null;
  modelFormat: ModelInventoryFormat | null;
  projectName: string;
  trainingMethod: TrainingMethod;
  trainingMethodProvenance: TrainingMethodProvenance;
  datasetSource: DatasetSource;
  browseDatasetSelection: BrowseDatasetSelection;
  datasetFormat: DatasetFormat;
  dataset: string | null;
  datasetKnownCached: boolean;
  datasetLocalPath: string | null;
  datasetSubset: string | null;
  datasetSplit: string | null;
  datasetEvalSplit: string | null;
  datasetStreaming: boolean;
  manualDatasetOptionsValid: boolean;
  datasetManualMapping: DatasetManualMapping;
  datasetSystemPrompt: string;
  datasetLabelMapping: Record<string, Record<string, string>>;
  datasetAdvisorNotification: string | null;
  datasetSliceStart: string | null;
  datasetSliceEnd: string | null;
  uploadedFile: string | null;
  uploadedEvalFile: string | null;
  epochs: number;
  contextLength: number;
  learningRate: number;
  embeddingLearningRate: number | null;
  optimizerType: string;
  lrSchedulerType: string;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  loraVariant: LoraVariant;
  batchSize: number;
  gradientAccumulation: number;
  weightDecay: number;
  warmupSteps: number;
  maxSteps: number;
  saveSteps: number;
  evalSteps: number;
  packing: boolean;
  trainOnCompletions: boolean;
  gradientCheckpointing: GradientCheckpointing;
  randomSeed: number;
  enableWandb: boolean;
  wandbToken: string;
  wandbProject: string;
  enableTensorboard: boolean;
  tensorboardDir: string;
  logFrequency: number;
  isCheckingVision: boolean;
  isVisionModel: boolean;
  isEmbeddingModel: boolean;
  isAudioModel: boolean;
  isLoadingModelDefaults: boolean;
  modelDefaultsError: string | null;
  modelDefaultsAppliedFor: string | null;
  advancedSettingsBaseline: AdvancedSettingsBaseline | null;
  trainOnCompletionsDefaultPendingFor: string | null;
  isCheckingDataset: boolean;
  isDatasetImage: boolean | null;
  isDatasetAudio: boolean;
  datasetCheckFailed: boolean;
  trustRemoteCode: boolean;
  approvedRemoteCodeFingerprint?: string | null;
  finetuneVisionLayers: boolean;
  finetuneLanguageLayers: boolean;
  finetuneAttentionModules: boolean;
  finetuneMLPModules: boolean;
  targetModules: string[];
  maxPositionEmbeddings: number | null;
  visionImageSize: number | null;
  s3Config: S3Config | null;
}

export type AdvancedSettingsBaseline = Partial<
  Pick<
    TrainingConfigState,
    | "optimizerType"
    | "lrSchedulerType"
    | "weightDecay"
    | "warmupSteps"
    | "saveSteps"
    | "evalSteps"
    | "randomSeed"
    | "packing"
    | "trainOnCompletions"
    | "gradientCheckpointing"
    | "visionImageSize"
    | "finetuneVisionLayers"
    | "finetuneLanguageLayers"
    | "finetuneAttentionModules"
    | "finetuneMLPModules"
    | "loraRank"
    | "loraAlpha"
    | "loraDropout"
    | "loraVariant"
    | "targetModules"
  >
>;

export interface TrainingConfigActions {
  setStep: (step: StepNumber) => void;
  nextStep: () => void;
  prevStep: () => void;
  setModelType: (type: ModelType) => void;
  setSelectedModel: (model: string | null) => void;
  selectTrainingModel: (
    model: string | null,
    modelType: ModelType | null,
    options?: TrainingModelSelectionOptions,
  ) => void;
  setSelectedModelCacheReference: (
    model: string,
    options: {
      localPath: string | null;
      modelFormat: ModelInventoryFormat | null;
    },
  ) => void;
  clearSelectedModelCacheReference: (
    model: string,
    localPath?: string | null,
  ) => void;
  clearSelectedDatasetCacheReference: (
    dataset: string,
    localPath?: string | null,
  ) => void;
  setSelectedDatasetCacheReference: (
    dataset: string,
    localPath: string | null,
  ) => void;
  setProjectName: (value: string) => void;
  ensureModelDefaultsLoaded: () => void;
  ensureDatasetChecked: () => void;
  setTrainingMethod: (method: TrainingMethod) => void;
  setDatasetSource: (source: DatasetSource) => void;
  selectHfDataset: (
    dataset: string | null,
    options?: DatasetCacheReferenceOptions,
  ) => void;
  selectLocalDataset: (file: string | null) => void;
  selectS3Source: () => void;
  restoreBrowseDatasetSource: () => void;
  setDatasetFormat: (format: DatasetFormat) => void;
  setDataset: (dataset: string | null) => void;
  setDatasetSubset: (subset: string | null) => void;
  setDatasetSplit: (split: string | null) => void;
  setDatasetEvalSplit: (split: string | null) => void;
  setDatasetStreaming: (value: boolean) => void;
  setManualDatasetOptionsValid: (value: boolean) => void;
  markManualDatasetOptionsEdited: (optionsValid: boolean) => void;
  setDatasetManualMapping: (mapping: DatasetManualMapping) => void;
  setDatasetAdvisorFields: (fields: {
    systemPrompt?: string;
    labelMapping?: Record<string, Record<string, string>>;
    notification?: string | null;
  }) => void;
  setDatasetSliceStart: (value: string | null) => void;
  setDatasetSliceEnd: (value: string | null) => void;
  setUploadedFile: (file: string | null) => void;
  setUploadedEvalFile: (file: string | null) => void;
  setEpochs: (epochs: number) => void;
  setContextLength: (length: number) => void;
  setVisionImageSize: (size: number | null) => void;
  setLearningRate: (rate: number) => void;
  setEmbeddingLearningRate: (rate: number | null) => void;
  setOptimizerType: (value: string) => void;
  setLrSchedulerType: (value: string) => void;
  setLoraRank: (rank: number) => void;
  setLoraAlpha: (alpha: number) => void;
  setLoraDropout: (dropout: number) => void;
  setLoraVariant: (variant: LoraVariant) => void;
  setBatchSize: (value: number) => void;
  setGradientAccumulation: (value: number) => void;
  setWeightDecay: (value: number) => void;
  setWarmupSteps: (value: number) => void;
  setMaxSteps: (value: number) => void;
  setSaveSteps: (value: number) => void;
  setEvalSteps: (value: number) => void;
  setPacking: (value: boolean) => void;
  setTrainOnCompletions: (value: boolean) => void;
  setGradientCheckpointing: (value: GradientCheckpointing) => void;
  setRandomSeed: (value: number) => void;
  setEnableWandb: (value: boolean) => void;
  setWandbToken: (value: string) => void;
  setWandbProject: (value: string) => void;
  setEnableTensorboard: (value: boolean) => void;
  setTensorboardDir: (value: string) => void;
  setLogFrequency: (value: number) => void;
  setFinetuneVisionLayers: (value: boolean) => void;
  setFinetuneLanguageLayers: (value: boolean) => void;
  setFinetuneAttentionModules: (value: boolean) => void;
  setFinetuneMLPModules: (value: boolean) => void;
  setTargetModules: (value: string[]) => void;
  setS3Config: (value: S3Config | null) => void;
  canProceed: () => boolean;
  reset: () => void;
  resetToModelDefaults: () => void;
  applyConfigPatch: (config: BackendModelConfig) => void;
}

export type TrainingConfigStore = TrainingConfigState & TrainingConfigActions;
