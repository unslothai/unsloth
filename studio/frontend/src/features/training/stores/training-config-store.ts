import { DEFAULT_HYPERPARAMS, STEPS } from "@/config/training";
import type { StepNumber } from "@/types/training";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { TrainingConfigState, TrainingConfigStore } from "../types/config";
import { checkVisionModel } from "../api/models-api";

const MIN_STEP: StepNumber = 1;
const MAX_STEP: StepNumber = STEPS.length as StepNumber;

function emptyManualMapping(): TrainingConfigState["datasetManualMapping"] {
  return { input: null, output: null };
}

const initialState: TrainingConfigState = {
  currentStep: MIN_STEP,
  modelType: null,
  selectedModel: null,
  trainingMethod: "qlora",
  hfToken: "",
  datasetSource: "huggingface",
  datasetFormat: "auto",
  dataset: null,
  datasetSubset: null,
  datasetSplit: null,
  datasetManualMapping: emptyManualMapping(),
  uploadedFile: null,
  isCheckingVision: false,
  isVisionModel: false,
  ...DEFAULT_HYPERPARAMS,
};

// AbortController for in-flight vision checks so rapid model changes
// cancel stale requests.
let _visionCheckController: AbortController | null = null;

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
      setSelectedModel: (selectedModel) => {
        set({ selectedModel });

        // Cancel any in-flight vision check
        _visionCheckController?.abort();
        _visionCheckController = null;

        if (!selectedModel) {
          set({ isCheckingVision: false });
          return;
        }

        // Fire async backend check to determine if model is vision
        const controller = new AbortController();
        _visionCheckController = controller;
        set({ isCheckingVision: true });

        checkVisionModel(selectedModel)
          .then((isVision) => {
            // Only apply if this is still the active check
            if (controller.signal.aborted) return;
            set({
              isVisionModel: isVision,
              isCheckingVision: false,
            });
          })
          .catch(() => {
            if (controller.signal.aborted) return;
            // On error, default to text and stop loading
            set({ isCheckingVision: false });
          });
      },
      setTrainingMethod: (trainingMethod) => set({ trainingMethod }),
      setHfToken: (hfToken) => set({ hfToken }),
      setDatasetSource: (datasetSource) => set({ datasetSource }),
      setDatasetFormat: (datasetFormat) => set({ datasetFormat }),
      setDataset: (dataset) =>
        set({
          dataset,
          datasetSubset: null,
          datasetSplit: null,
          datasetManualMapping: emptyManualMapping(),
        }),
      setDatasetSubset: (datasetSubset) =>
        set({
          datasetSubset,
          datasetSplit: null,
          datasetManualMapping: emptyManualMapping(),
        }),
      setDatasetSplit: (datasetSplit) =>
        set({ datasetSplit, datasetManualMapping: emptyManualMapping() }),
      setDatasetManualMapping: (datasetManualMapping) =>
        set({ datasetManualMapping }),
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
      version: 2,
      migrate: (persisted, version) => {
        const s = persisted as Record<string, unknown>;
        if (version >= 2) return s as unknown as TrainingConfigStore;
        if (s.datasetSubset == null && s.datasetConfig != null) {
          s.datasetSubset = s.datasetConfig;
        }
        delete s.datasetConfig;
        return s as unknown as TrainingConfigStore;
      },
      partialize: (state) => {
        const { modelType, isCheckingVision, isVisionModel, ...rest } = state;
        return rest;
      },
    },
  ),
);
