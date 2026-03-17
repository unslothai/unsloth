// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "vision" | "audio" | "embeddings" | "text";
export type TrainingMethod = "qlora" | "lora" | "full";

export function isAdapterMethod(method: TrainingMethod): boolean {
  return method === "lora" || method === "qlora";
}
export type StepNumber = 1 | 2 | 3 | 4 | 5;
export type DatasetSource = "huggingface" | "upload";
export type DatasetFormat = "auto" | "alpaca" | "chatml" | "sharegpt";
export type GradientCheckpointing = "none" | "true" | "unsloth";

export interface WizardState {
  currentStep: StepNumber;
  modelType: ModelType | null;
  selectedModel: string | null;
  trainingMethod: TrainingMethod;
  hfToken: string;
  datasetSource: DatasetSource;
  datasetFormat: DatasetFormat;
  dataset: string | null;
  datasetSubset: string | null;
  datasetSplit: string | null;
  uploadedFile: string | null;
  epochs: number;
  contextLength: number;
  learningRate: number;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  loraVariant: "lora" | "rslora" | "loftq";
  batchSize: number;
  gradientAccumulation: number;
  weightDecay: number;
  warmupSteps: number;
  maxSteps: number;
  saveSteps: number;
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
  finetuneVisionLayers: boolean;
  finetuneLanguageLayers: boolean;
  finetuneAttentionModules: boolean;
  finetuneMLPModules: boolean;
  targetModules: string[];
}

export interface WizardActions {
  setStep: (step: StepNumber) => void;
  nextStep: () => void;
  prevStep: () => void;
  setModelType: (type: ModelType) => void;
  setSelectedModel: (model: string | null) => void;
  setTrainingMethod: (method: TrainingMethod) => void;
  setHfToken: (token: string) => void;
  setDatasetSource: (source: DatasetSource) => void;
  setDatasetFormat: (format: DatasetFormat) => void;
  setDataset: (dataset: string | null) => void;
  setDatasetSubset: (subset: string | null) => void;
  setDatasetSplit: (split: string | null) => void;
  setUploadedFile: (file: string | null) => void;
  setEpochs: (epochs: number) => void;
  setContextLength: (length: number) => void;
  setLearningRate: (rate: number) => void;
  setLoraRank: (rank: number) => void;
  setLoraAlpha: (alpha: number) => void;
  setLoraDropout: (dropout: number) => void;
  setLoraVariant: (v: "lora" | "rslora" | "loftq") => void;
  setBatchSize: (v: number) => void;
  setGradientAccumulation: (v: number) => void;
  setWeightDecay: (v: number) => void;
  setWarmupSteps: (v: number) => void;
  setMaxSteps: (v: number) => void;
  setSaveSteps: (v: number) => void;
  setPacking: (v: boolean) => void;
  setTrainOnCompletions: (v: boolean) => void;
  setGradientCheckpointing: (v: GradientCheckpointing) => void;
  setRandomSeed: (v: number) => void;
  setEnableWandb: (v: boolean) => void;
  setWandbToken: (v: string) => void;
  setWandbProject: (v: string) => void;
  setEnableTensorboard: (v: boolean) => void;
  setTensorboardDir: (v: string) => void;
  setLogFrequency: (v: number) => void;
  setFinetuneVisionLayers: (v: boolean) => void;
  setFinetuneLanguageLayers: (v: boolean) => void;
  setFinetuneAttentionModules: (v: boolean) => void;
  setFinetuneMLPModules: (v: boolean) => void;
  setTargetModules: (v: string[]) => void;
  canProceed: () => boolean;
  reset: () => void;
}

export interface StepConfig {
  number: StepNumber;
  title: string;
  subtitle: string;
  description: string;
}
