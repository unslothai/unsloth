import { DEFAULT_HYPERPARAMS, STEPS } from "@/config/training";
import type { StepNumber } from "@/types/training";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { checkDatasetFormat } from "../api/datasets-api";
import { checkVisionModel, getModelConfig } from "../api/models-api";
import { mapBackendModelConfigToTrainingPatch } from "../lib/model-defaults";
import type { TrainingConfigState, TrainingConfigStore } from "../types/config";

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
  isLoadingModelDefaults: false,
  modelDefaultsError: null,
  modelDefaultsAppliedFor: null,
  isCheckingDataset: false,
  isDatasetMultimodal: null,
  ...DEFAULT_HYPERPARAMS,
};

// AbortController for in-flight dataset multimodal checks.
let _datasetCheckController: AbortController | null = null;

// AbortController for in-flight model default loads.
let _modelConfigController: AbortController | null = null;

const NON_PERSISTED_STATE_KEYS: ReadonlySet<keyof TrainingConfigState> = new Set([
  "modelType",
  "isCheckingVision",
  "isVisionModel",
  "isLoadingModelDefaults",
  "modelDefaultsError",
  "isCheckingDataset",
  "isDatasetMultimodal",
]);

function partializePersistedState(
  state: TrainingConfigStore,
): Partial<TrainingConfigStore> {
  return Object.fromEntries(
    Object.entries(state).filter(([key]) => {
      const stateKey = key as keyof TrainingConfigState;
      return !NON_PERSISTED_STATE_KEYS.has(stateKey);
    }),
  ) as Partial<TrainingConfigStore>;
}

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
    (set, get) => {
      const loadAndApplyModelDefaults = (modelName: string) => {
        _modelConfigController?.abort();
        const controller = new AbortController();
        _modelConfigController = controller;
        set({
          isLoadingModelDefaults: true,
          isCheckingVision: true,
          modelDefaultsError: null,
        });

        void getModelConfig(modelName, controller.signal)
          .then((modelDetails) => {
            if (controller.signal.aborted) return;
            if (get().selectedModel !== modelName) return;

            set({
              ...mapBackendModelConfigToTrainingPatch(modelDetails.config),
              isVisionModel: modelDetails.is_vision,
              isLoadingModelDefaults: false,
              isCheckingVision: false,
              modelDefaultsError: null,
              modelDefaultsAppliedFor: modelName,
            });
          })
          .catch((error) => {
            if (controller.signal.aborted) return;
            if (get().selectedModel !== modelName) return;

            set({
              isLoadingModelDefaults: false,
              modelDefaultsError:
                error instanceof Error
                  ? error.message
                  : "Failed to load model defaults",
            });

            // Fallback vision check if config endpoint fails.
            void checkVisionModel(modelName)
              .then((isVision) => {
                if (get().selectedModel !== modelName) return;
                set({
                  isVisionModel: isVision,
                  isCheckingVision: false,
                });
              })
              .catch(() => {
                if (get().selectedModel !== modelName) return;
                set({ isCheckingVision: false });
              });
          });
      };

      return {
        ...initialState,
        setStep: (step) => set({ currentStep: step }),
        nextStep: () => set({ currentStep: clampStep(get().currentStep + 1) }),
        prevStep: () => set({ currentStep: clampStep(get().currentStep - 1) }),
        setModelType: (modelType) => {
          _modelConfigController?.abort();
          _modelConfigController = null;

          set({
            modelType,
            selectedModel: null,
            isCheckingVision: false,
            isVisionModel: false,
            isLoadingModelDefaults: false,
            modelDefaultsError: null,
            modelDefaultsAppliedFor: null,
          });
        },
        setSelectedModel: (selectedModel) => {
          const previousModel = get().selectedModel;
          set({ selectedModel, modelDefaultsError: null });

          if (!selectedModel) {
            _modelConfigController?.abort();
            _modelConfigController = null;
            set({
              isCheckingVision: false,
              isVisionModel: false,
              isLoadingModelDefaults: false,
              modelDefaultsError: null,
              modelDefaultsAppliedFor: null,
            });
            return;
          }

          const shouldLoadDefaults =
            selectedModel !== previousModel ||
            get().modelDefaultsAppliedFor !== selectedModel;
          if (shouldLoadDefaults) {
            void loadAndApplyModelDefaults(selectedModel);
          }
        },
        ensureModelDefaultsLoaded: () => {
          const state = get();
          if (!state.selectedModel) return;
          if (state.isLoadingModelDefaults) return;
          if (state.modelDefaultsAppliedFor === state.selectedModel) return;
          void loadAndApplyModelDefaults(state.selectedModel);
        },
        setTrainingMethod: (trainingMethod) => set({ trainingMethod }),
        setHfToken: (hfToken) => set({ hfToken }),
        setDatasetSource: (datasetSource) => set({ datasetSource }),
        setDatasetFormat: (datasetFormat) => set({ datasetFormat }),
        setDataset: (dataset) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          set({
            dataset,
            datasetSubset: null,
            datasetSplit: null,
            datasetManualMapping: emptyManualMapping(),
            isDatasetMultimodal: null,
            isCheckingDataset: false,
          });
        },
        setDatasetSubset: (datasetSubset) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          set({
            datasetSubset,
            datasetSplit: null,
            datasetManualMapping: emptyManualMapping(),
            isDatasetMultimodal: null,
            isCheckingDataset: false,
          });
        },
        setDatasetSplit: (datasetSplit) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          set({
            datasetSplit,
            datasetManualMapping: emptyManualMapping(),
            isDatasetMultimodal: null,
            isCheckingDataset: false,
          });

          const state = get();
          const datasetName =
            state.datasetSource === "huggingface"
              ? state.dataset
              : state.uploadedFile;
          if (!datasetName) return;

          const controller = new AbortController();
          _datasetCheckController = controller;
          set({ isCheckingDataset: true });

          checkDatasetFormat({
            datasetName,
            hfToken: state.hfToken.trim() || null,
            subset: state.datasetSubset,
            split: datasetSplit || "train",
          })
            .then((res) => {
              if (controller.signal.aborted) return;
              set({
                isDatasetMultimodal: !!res.is_multimodal,
                isCheckingDataset: false,
              });
            })
            .catch(() => {
              if (controller.signal.aborted) return;
              set({ isDatasetMultimodal: null, isCheckingDataset: false });
            });
        },
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
        setEvalSteps: (evalSteps) => set({ evalSteps }),
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
        setFinetuneMLPModules: (finetuneMLPModules) =>
          set({ finetuneMLPModules }),
        setTargetModules: (targetModules) => set({ targetModules }),
        canProceed: () => canProceedForStep(get()),
        reset: () => set(initialState),
      };
    },
    {
      name: "unsloth_training_config_v1",
      version: 3,
      migrate: (persisted, version) => {
        const s = persisted as Record<string, unknown>;
        if (version < 2 && s.datasetSubset == null && s.datasetConfig != null) {
          s.datasetSubset = s.datasetConfig;
        }
        delete s.datasetConfig;
        if (version < 3 && s.modelDefaultsAppliedFor == null) {
          s.modelDefaultsAppliedFor = null;
        }
        return s as unknown as TrainingConfigStore;
      },
      partialize: partializePersistedState,
    },
  ),
);
