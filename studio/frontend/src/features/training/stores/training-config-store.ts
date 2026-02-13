import { DEFAULT_HYPERPARAMS, STEPS } from "@/config/training";
import type { StepNumber } from "@/types/training";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { TrainingConfigState, TrainingConfigStore } from "../types/config";

const MIN_STEP: StepNumber = 1;
const MAX_STEP: StepNumber = STEPS.length as StepNumber;

const initialState: TrainingConfigState = {
  currentStep: MIN_STEP,
  modelType: null,
  selectedModel: null,
  trainingMethod: "qlora",
  hfToken: "",
  datasetSource: "huggingface",
  datasetFormat: "auto",
  dataset: null,
  uploadedFile: null,
  ...DEFAULT_HYPERPARAMS,
};

function clampStep(step: number): StepNumber {
  return Math.min(MAX_STEP, Math.max(MIN_STEP, step)) as StepNumber;
}

function canProceedForStep(state: TrainingConfigState): boolean {
  switch (state.currentStep) {
    case 1:
      return state.modelType !== null;
    case 2:
      return state.selectedModel !== null;
    case 3:
      return state.datasetSource === "upload"
        ? state.uploadedFile !== null
        : state.dataset !== null;
    case 4:
    case 5:
      return true;
    default:
      return false;
  }
}

export const useTrainingConfigStore = create<TrainingConfigStore>()(
  persist(
    (set, get) => ({
      ...initialState,
      setStep: (step) => set({ currentStep: step }),
      nextStep: () => set({ currentStep: clampStep(get().currentStep + 1) }),
      prevStep: () => set({ currentStep: clampStep(get().currentStep - 1) }),
      setModelType: (modelType) => set({ modelType, selectedModel: null }),
      setSelectedModel: (selectedModel) => set({ selectedModel }),
      setTrainingMethod: (trainingMethod) => set({ trainingMethod }),
      setHfToken: (hfToken) => set({ hfToken }),
      setDatasetSource: (datasetSource) => set({ datasetSource }),
      setDatasetFormat: (datasetFormat) => set({ datasetFormat }),
      setDataset: (dataset) => set({ dataset }),
      setUploadedFile: (uploadedFile) => set({ uploadedFile }),
      setEpochs: (epochs) => set({ epochs }),
      setContextLength: (contextLength) => set({ contextLength }),
      setLearningRate: (learningRate) => set({ learningRate }),
      setLoraRank: (loraRank) => set({ loraRank }),
      setLoraAlpha: (loraAlpha) => set({ loraAlpha }),
      setLoraDropout: (loraDropout) => set({ loraDropout }),
      setLoraVariant: (loraVariant) => set({ loraVariant }),
      setBatchSize: (batchSize) => set({ batchSize }),
      setGradientAccumulation: (gradientAccumulation) =>
        set({ gradientAccumulation }),
      setWeightDecay: (weightDecay) => set({ weightDecay }),
      setWarmupSteps: (warmupSteps) => set({ warmupSteps }),
      setMaxSteps: (maxSteps) => set({ maxSteps }),
      setSaveSteps: (saveSteps) => set({ saveSteps }),
      setPacking: (packing) => set({ packing }),
      setTrainOnCompletions: (trainOnCompletions) =>
        set({ trainOnCompletions }),
      setGradientCheckpointing: (gradientCheckpointing) =>
        set({ gradientCheckpointing }),
      setRandomSeed: (randomSeed) => set({ randomSeed }),
      setEnableWandb: (enableWandb) => set({ enableWandb }),
      setWandbToken: (wandbToken) => set({ wandbToken }),
      setWandbProject: (wandbProject) => set({ wandbProject }),
      setEnableTensorboard: (enableTensorboard) => set({ enableTensorboard }),
      setTensorboardDir: (tensorboardDir) => set({ tensorboardDir }),
      setLogFrequency: (logFrequency) => set({ logFrequency }),
      setFinetuneVisionLayers: (finetuneVisionLayers) =>
        set({ finetuneVisionLayers }),
      setFinetuneLanguageLayers: (finetuneLanguageLayers) =>
        set({ finetuneLanguageLayers }),
      setFinetuneAttentionModules: (finetuneAttentionModules) =>
        set({ finetuneAttentionModules }),
      setFinetuneMLPModules: (finetuneMLPModules) => set({ finetuneMLPModules }),
      setTargetModules: (targetModules) => set({ targetModules }),
      canProceed: () => canProceedForStep(get()),
      reset: () => set(initialState),
    }),
    {
      name: "unsloth_training_config_v1",
      partialize: (state) => {
        const { modelType, ...rest } = state;
        return rest;
      },
    },
  ),
);
