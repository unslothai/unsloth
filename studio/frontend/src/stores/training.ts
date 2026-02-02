import { DEFAULT_HYPERPARAMS } from "@/config/training";
import type { StepNumber, WizardActions, WizardState } from "@/types/training";
import { create } from "zustand";

const MIN_STEP: StepNumber = 1;
const MAX_STEP: StepNumber = 5;

const initialState: WizardState = {
  isTraining: false,
  trainingMetrics: null,
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

function canProceedForStep(state: WizardState): boolean {
  switch (state.currentStep) {
    case 1:
      return state.modelType !== null;
    case 2:
      return state.selectedModel !== null;
    case 3: {
      if (state.datasetSource === "upload") {
        return state.uploadedFile !== null;
      }
      return state.dataset !== null;
    }
    case 4:
    case 5:
      return true;
    default:
      return false;
  }
}

export const useWizardStore = create<WizardState & WizardActions>(
  (set, get) => ({
    ...initialState,

    setStep: (step) => set({ currentStep: step }),

    nextStep: () => {
      const { currentStep } = get();
      set({ currentStep: clampStep(currentStep + 1) });
    },

    prevStep: () => {
      const { currentStep } = get();
      set({ currentStep: clampStep(currentStep - 1) });
    },

    setModelType: (type) => set({ modelType: type, selectedModel: null }),
    setSelectedModel: (model) => set({ selectedModel: model }),
    setTrainingMethod: (method) => set({ trainingMethod: method }),
    setHfToken: (token) => set({ hfToken: token }),
    setDatasetSource: (source) => set({ datasetSource: source }),
    setDatasetFormat: (format) => set({ datasetFormat: format }),
    setDataset: (dataset) => set({ dataset }),
    setUploadedFile: (file) => set({ uploadedFile: file }),
    setEpochs: (epochs) => set({ epochs }),
    setContextLength: (length) => set({ contextLength: length }),
    setLearningRate: (rate) => set({ learningRate: rate }),
    setLoraRank: (rank) => set({ loraRank: rank }),
    setLoraAlpha: (alpha) => set({ loraAlpha: alpha }),
    setLoraDropout: (dropout) => set({ loraDropout: dropout }),
    setLoraVariant: (v) => set({ loraVariant: v }),
    setBatchSize: (v) => set({ batchSize: v }),
    setGradientAccumulation: (v) => set({ gradientAccumulation: v }),
    setWeightDecay: (v) => set({ weightDecay: v }),
    setWarmupSteps: (v) => set({ warmupSteps: v }),
    setMaxSteps: (v) => set({ maxSteps: v }),
    setSaveSteps: (v) => set({ saveSteps: v }),
    setPacking: (v) => set({ packing: v }),
    setTrainOnCompletions: (v) => set({ trainOnCompletions: v }),
    setGradientCheckpointing: (v) => set({ gradientCheckpointing: v }),
    setRandomSeed: (v) => set({ randomSeed: v }),
    setEnableWandb: (v) => set({ enableWandb: v }),
    setWandbToken: (v) => set({ wandbToken: v }),
    setWandbProject: (v) => set({ wandbProject: v }),
    setEnableTensorboard: (v) => set({ enableTensorboard: v }),
    setTensorboardDir: (v) => set({ tensorboardDir: v }),
    setLogFrequency: (v) => set({ logFrequency: v }),
    setFinetuneVisionLayers: (v) => set({ finetuneVisionLayers: v }),
    setFinetuneLanguageLayers: (v) => set({ finetuneLanguageLayers: v }),
    setFinetuneAttentionModules: (v) => set({ finetuneAttentionModules: v }),
    setFinetuneMLPModules: (v) => set({ finetuneMLPModules: v }),
    setTargetModules: (v) => set({ targetModules: v }),
    setIsTraining: (v) => set({ isTraining: v }),
    setTrainingMetrics: (v) => set({ trainingMetrics: v }),

    canProceed: () => canProceedForStep(get()),

    reset: () => set(initialState),
  }),
);
