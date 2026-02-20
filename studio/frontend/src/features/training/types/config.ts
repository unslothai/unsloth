import type {
  DatasetFormat,
  DatasetSource,
  GradientCheckpointing,
  ModelType,
  StepNumber,
  TrainingMethod,
} from "@/types/training";

export type LoraVariant = "lora" | "rslora" | "loftq";

export type DatasetManualMapping = {
  input: string | null;
  output: string | null;
};

export interface TrainingConfigState {
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
  datasetManualMapping: DatasetManualMapping;
  uploadedFile: string | null;
  epochs: number;
  contextLength: number;
  learningRate: number;
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
  isLoadingModelDefaults: boolean;
  modelDefaultsError: string | null;
  modelDefaultsAppliedFor: string | null;
  isCheckingDataset: boolean;
  isDatasetMultimodal: boolean | null;
  finetuneVisionLayers: boolean;
  finetuneLanguageLayers: boolean;
  finetuneAttentionModules: boolean;
  finetuneMLPModules: boolean;
  targetModules: string[];
}

export interface TrainingConfigActions {
  setStep: (step: StepNumber) => void;
  nextStep: () => void;
  prevStep: () => void;
  setModelType: (type: ModelType) => void;
  setSelectedModel: (model: string | null) => void;
  ensureModelDefaultsLoaded: () => void;
  ensureDatasetChecked: () => void;
  setTrainingMethod: (method: TrainingMethod) => void;
  setHfToken: (token: string) => void;
  setDatasetSource: (source: DatasetSource) => void;
  setDatasetFormat: (format: DatasetFormat) => void;
  setDataset: (dataset: string | null) => void;
  setDatasetSubset: (subset: string | null) => void;
  setDatasetSplit: (split: string | null) => void;
  setDatasetManualMapping: (mapping: DatasetManualMapping) => void;
  setUploadedFile: (file: string | null) => void;
  setEpochs: (epochs: number) => void;
  setContextLength: (length: number) => void;
  setLearningRate: (rate: number) => void;
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
  canProceed: () => boolean;
  reset: () => void;
}

export type TrainingConfigStore = TrainingConfigState & TrainingConfigActions;
