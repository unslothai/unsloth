// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CPT_TARGET_MODULES, DEFAULT_HYPERPARAMS, LR_DEFAULT_CPT, LR_DEFAULT_FULL, LR_DEFAULT_LORA, STEPS, TARGET_MODULES } from "@/config/training";
import { authFetch } from "@/features/auth";
import { isAdapterMethod } from "@/types/training";
import type { DatasetFormat } from "@/types/training";
import type { ModelType, StepNumber, TrainingMethod } from "@/types/training";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { checkDatasetFormat } from "../api/datasets-api";
import { checkVisionModel, getModelConfig } from "../api/models-api";
import { mapBackendModelConfigToTrainingPatch } from "../lib/model-defaults";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState, TrainingConfigStore } from "../types/config";

const MIN_STEP: StepNumber = 1;
const MAX_STEP: StepNumber = STEPS.length as StepNumber;

/**
 * Auto-select LoRA (16-bit) vs QLoRA (4-bit) based on model size and GPU memory.
 *
 * Rule: if model_size_gb * 1.5 * context_scale fits in free VRAM, use "lora" (16-bit).
 * Otherwise use "qlora" (4-bit).
 *
 * Context scale: <=8192 = 1.0, >8192 = 1.7, >=16384 = 2.0, >=32768 = 4.0
 */
async function autoSelectTrainingMethod(
  modelSizeBytes: number,
  contextLength: number,
): Promise<TrainingMethod | null> {
  try {
    const res = await authFetch("/api/system/hardware");
    if (!res.ok) return null;
    const data = await res.json();
    const freeGb: number | null = data?.gpu?.vram_free_gb ?? null;
    if (freeGb == null) return null;

    const modelSizeGb = modelSizeBytes / (1024 ** 3);

    let contextScale = 1.0;
    if (contextLength >= 32768) contextScale = 4.0;
    else if (contextLength >= 16384) contextScale = 2.0;
    else if (contextLength > 8192) contextScale = 1.7;

    const estimatedUsage = modelSizeGb * 1.5 * contextScale;
    return estimatedUsage <= freeGb ? "lora" : "qlora";
  } catch {
    return null;
  }
}

function emptyManualMapping(): TrainingConfigState["datasetManualMapping"] {
  return {};
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
  datasetEvalSplit: null,
  datasetManualMapping: emptyManualMapping(),
  datasetSystemPrompt: "",
  datasetUserTemplate: "",
  datasetAssistantTemplate: "",
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
  isLoadingModelDefaults: false,
  modelDefaultsError: null,
  modelDefaultsAppliedFor: null,
  isCheckingDataset: false,
  isDatasetImage: null,
  isDatasetAudio: false,
  maxPositionEmbeddings: null,
  ...DEFAULT_HYPERPARAMS,
};

// AbortController for in-flight dataset multimodal checks.
let _datasetCheckController: AbortController | null = null;

// AbortController for in-flight model default loads.
let _modelConfigController: AbortController | null = null;

// Track whether the user has manually toggled trainOnCompletions
// since the last auto-set (model load or dataset change).
let _trainOnCompletionsManuallySet = false;

// Track whether the user has manually edited the learning rate
// since the last model load. When false, switching training method
// auto-sets LR to 2e-4 (LoRA/QLoRA) or 2e-5 (full fine-tune).
let _learningRateManuallySet = false;

// Stash the model-config-provided (YAML) learning rate so that
// setTrainingMethod can restore it when switching back from full to adapter.
let _yamlLearningRate: number | undefined = undefined;

// Track whether entering CPT auto-forced datasetFormat="raw" so that
// leaving CPT can restore the prior user-visible format.
let _datasetFormatBeforeCpt: DatasetFormat | null = null;
let _datasetFormatAutoForcedByCpt = false;

const NON_PERSISTED_STATE_KEYS: ReadonlySet<keyof TrainingConfigState> = new Set([
  "modelType",
  "isCheckingVision",
  "isEmbeddingModel",
  "isAudioModel",
  "isLoadingModelDefaults",
  "modelDefaultsError",
  "modelDefaultsAppliedFor",
  "isCheckingDataset",
  "isDatasetImage",
  "isDatasetAudio",
  "trainOnCompletions",
  "maxPositionEmbeddings",
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

type TrainingMethodStatePatch = Partial<
  Pick<
    TrainingConfigState,
    | "trainingMethod"
    | "learningRate"
    | "loraRank"
    | "loraAlpha"
    | "loraVariant"
    | "targetModules"
    | "datasetFormat"
    | "trainOnCompletions"
  >
>;

function getCptTrainingPatch(): TrainingMethodStatePatch {
  return {
    loraRank: 128,
    loraAlpha: 32,
    loraVariant: "rslora",
    targetModules: CPT_TARGET_MODULES,
    datasetFormat: "raw",
    trainOnCompletions: false,
  };
}

function getCptModelDefaultsPatch(): TrainingMethodStatePatch {
  return {
    ...getCptTrainingPatch(),
    learningRate: LR_DEFAULT_CPT,
  };
}

function getRestoreFromCptPatch(): TrainingMethodStatePatch {
  return {
    loraRank: DEFAULT_HYPERPARAMS.loraRank,
    loraAlpha: DEFAULT_HYPERPARAMS.loraAlpha,
    loraVariant: DEFAULT_HYPERPARAMS.loraVariant,
    targetModules: TARGET_MODULES,
  };
}

function clearCptDatasetFormatTracking(): void {
  _datasetFormatBeforeCpt = null;
  _datasetFormatAutoForcedByCpt = false;
}

function recordCptDatasetFormatOverride(currentDatasetFormat: DatasetFormat): void {
  if (isRawTextDatasetFormat(currentDatasetFormat)) {
    clearCptDatasetFormatTracking();
    return;
  }
  _datasetFormatBeforeCpt = currentDatasetFormat;
  _datasetFormatAutoForcedByCpt = true;
}

function getRestoreDatasetFormatFromCptPatch(): TrainingMethodStatePatch {
  if (!_datasetFormatAutoForcedByCpt || _datasetFormatBeforeCpt == null) {
    clearCptDatasetFormatTracking();
    return {};
  }

  const previousDatasetFormat = _datasetFormatBeforeCpt;
  clearCptDatasetFormatTracking();
  return { datasetFormat: previousDatasetFormat };
}

function resolveTrainingMethodLearningRate(
  prevMethod: TrainingMethod,
  nextMethod: TrainingMethod,
): number | undefined {
  if (_learningRateManuallySet) {
    return undefined;
  }

  const wasCpt = prevMethod === "cpt";
  const wasAdapter = isAdapterMethod(prevMethod);
  const nowAdapter = isAdapterMethod(nextMethod);

  if (nextMethod === "cpt") {
    return LR_DEFAULT_CPT;
  }
  if (wasCpt && nowAdapter) {
    return _yamlLearningRate ?? LR_DEFAULT_LORA;
  }
  if (wasAdapter && nowAdapter) {
    return undefined;
  }
  return nowAdapter ? _yamlLearningRate ?? LR_DEFAULT_LORA : LR_DEFAULT_FULL;
}

function buildTrainingMethodPatch(
  prevMethod: TrainingMethod,
  nextMethod: TrainingMethod,
  currentDatasetFormat: DatasetFormat,
): TrainingMethodStatePatch {
  const patch: TrainingMethodStatePatch = { trainingMethod: nextMethod };

  if (prevMethod !== "cpt" && nextMethod === "cpt") {
    recordCptDatasetFormatOverride(currentDatasetFormat);
    Object.assign(patch, getCptTrainingPatch());
  }
  if (prevMethod === "cpt" && nextMethod !== "cpt") {
    Object.assign(
      patch,
      getRestoreFromCptPatch(),
      getRestoreDatasetFormatFromCptPatch(),
    );
  }

  const learningRate = resolveTrainingMethodLearningRate(prevMethod, nextMethod);
  if (learningRate !== undefined) {
    patch.learningRate = learningRate;
  }

  return patch;
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

        void getModelConfig(modelName, controller.signal, get().hfToken || undefined)
          .then((modelDetails) => {
            if (controller.signal.aborted) return;
            if (get().selectedModel !== modelName) return;

            _trainOnCompletionsManuallySet = false;
            _learningRateManuallySet = false;
            _yamlLearningRate = undefined;
            const patch = mapBackendModelConfigToTrainingPatch(modelDetails.config);

            // If the model config provides a specific learning rate, treat
            // it as authoritative so the async auto-select does not overwrite it.
            const modelConfigHasLR = patch.learningRate !== undefined;
            _yamlLearningRate = patch.learningRate;

            // YAML learning rates are tuned for adapter methods (LoRA/QLoRA).
            // If the user is currently on full fine-tune, override with the
            // full-finetune default instead of applying the YAML adapter LR.
            if (modelConfigHasLR && !isAdapterMethod(get().trainingMethod)) {
              patch.learningRate = LR_DEFAULT_FULL;
            }

            // If vision model + image dataset already known, override
            // trainOnCompletions to false regardless of backend default.
            if (modelDetails.is_vision && get().isDatasetImage === true) {
              patch.trainOnCompletions = false;
            }

            const isAudio = !!modelDetails.is_audio;
            // Pure audio model -> always uncheck trainOnCompletions.
            if (isAudio && !modelDetails.is_vision) {
              patch.trainOnCompletions = false;
            }
            // Audio-capable vision model (e.g. gemma3n) + audio dataset -> uncheck.
            if (isAudio && modelDetails.is_vision && get().isDatasetAudio) {
              patch.trainOnCompletions = false;
            }

            // Use backend-provided model_type when available, otherwise
            // infer from capability flags.
            const isEmbedding = !!modelDetails.is_embedding;
            const inferredModelType: ModelType = modelDetails.model_type
              ?? (isEmbedding ? "embeddings" : modelDetails.is_vision ? "vision" : modelDetails.is_audio ? "audio" : "text");

            // Auto-select training method based on model size vs GPU memory.
            // If model_size * 1.5 * context_scale fits in free VRAM, use LoRA 16-bit.
            // Otherwise use QLoRA 4-bit.
            // Auto-select LoRA vs QLoRA based on GPU memory.
            // Skip if user has manually chosen CPT -- don't override it.
            const modelSizeBytes = modelDetails.model_size_bytes;
            if (modelSizeBytes && modelSizeBytes > 0 && get().trainingMethod !== "cpt") {
              void autoSelectTrainingMethod(modelSizeBytes, patch.contextLength ?? get().contextLength)
                .then((method) => {
                  if (get().selectedModel !== modelName) return;
                  if (get().trainingMethod === "cpt") return;
                  if (method) {
                    const lrPatch = !_learningRateManuallySet && !modelConfigHasLR
                      ? { learningRate: method === "full" ? LR_DEFAULT_FULL : LR_DEFAULT_LORA }
                      : {};
                    set({ trainingMethod: method, ...lrPatch });
                  }
                });
            }

            // Preserve CPT hyperparams: YAML adapter defaults (r/alpha/targets/LR)
            // are tuned for standard LoRA and would otherwise clobber CPT settings.
            const cptOverrides =
              get().trainingMethod === "cpt"
                ? getCptModelDefaultsPatch()
                : {};

            set({
              ...patch,
              ...cptOverrides,
              modelType: inferredModelType,
              isVisionModel: modelDetails.is_vision,
              isEmbeddingModel: isEmbedding,
              isAudioModel: isAudio,
              isLoadingModelDefaults: false,
              isCheckingVision: false,
              modelDefaultsError: null,
              modelDefaultsAppliedFor: modelName,
              maxPositionEmbeddings: modelDetails.max_position_embeddings ?? null,
            });
          })
          .catch((error) => {
            if (controller.signal.aborted) return;
            if (get().selectedModel !== modelName) return;

            set({
              isLoadingModelDefaults: false,
              isEmbeddingModel: false,
              isAudioModel: false,
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
                  modelType: isVision ? "vision" : "text",
                  isVisionModel: isVision,
                  isEmbeddingModel: false,
                  isAudioModel: false,
                  isCheckingVision: false,
                });
              })
              .catch(() => {
                if (get().selectedModel !== modelName) return;
                set({ isCheckingVision: false, isEmbeddingModel: false, isAudioModel: false });
              });
          });
      };

      const runDatasetCheck = (datasetName: string, split: string) => {
        _datasetCheckController?.abort();
        const controller = new AbortController();
        _datasetCheckController = controller;
        set({ isCheckingDataset: true });

        const state = get();
        checkDatasetFormat({
          datasetName,
          hfToken: state.hfToken.trim() || null,
          subset: state.datasetSubset,
          split,
          isVlm: state.isVisionModel,
        })
          .then((res) => {
            if (controller.signal.aborted) return;
            const isImage = !!res.is_image;
            const isAudio = !!res.is_audio;
            const updates: Record<string, unknown> = {
              isDatasetImage: isImage,
              isDatasetAudio: isAudio,
              isCheckingDataset: false,
            };
            if (!_trainOnCompletionsManuallySet) {
              const { isVisionModel, isAudioModel } = get();
              if (isVisionModel && isImage) {
                updates.trainOnCompletions = false;
              }
              // Pure audio model → always uncheck regardless of dataset.
              if (isAudioModel && !isVisionModel) {
                updates.trainOnCompletions = false;
              }
              // Audio-capable vision model (e.g. gemma3n) + audio dataset → uncheck.
              if (isAudioModel && isVisionModel && isAudio) {
                updates.trainOnCompletions = false;
              }
            }
            set(updates);
          })
          .catch(() => {
            if (controller.signal.aborted) return;
            set({ isDatasetImage: null, isCheckingDataset: false });
          });
      };

      const resetDatasetState = (): Partial<TrainingConfigStore> => ({
        datasetSubset: null,
        datasetSplit: null,
        datasetEvalSplit: null,
        datasetManualMapping: emptyManualMapping(),
        datasetSystemPrompt: "",
        datasetUserTemplate: "",
        datasetAssistantTemplate: "",
        datasetLabelMapping: {},
        datasetAdvisorNotification: null,
        datasetSliceStart: null,
        datasetSliceEnd: null,
        uploadedEvalFile: null,
        isDatasetImage: null,
        isDatasetAudio: false,
        isCheckingDataset: false,
      });

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
            isEmbeddingModel: false,
            isAudioModel: false,
            isDatasetAudio: false,
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
              isEmbeddingModel: false,
              isAudioModel: false,
              isDatasetAudio: false,
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
        setTrainingMethod: (trainingMethod) => {
          const state = get();
          set(
            buildTrainingMethodPatch(
              state.trainingMethod,
              trainingMethod,
              state.datasetFormat,
            ),
          );
        },
        setHfToken: (hfToken) =>
          set({ hfToken: hfToken.trim().replace(/^["']+|["']+$/g, "") }),
        setDatasetSource: (datasetSource) => set({ datasetSource }),
        selectHfDataset: (dataset) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set({
            datasetSource: "huggingface",
            dataset,
            uploadedFile: null,
            ...resetDatasetState(),
          });
        },
        selectLocalDataset: (uploadedFile) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set({
            datasetSource: "upload",
            dataset: null,
            uploadedFile,
            ...resetDatasetState(),
          });
          if (uploadedFile) {
            runDatasetCheck(uploadedFile, "train");
          }
        },
        setDatasetFormat: (datasetFormat) =>
          set((state) => {
            if (state.trainingMethod === "cpt") {
              if (isRawTextDatasetFormat(datasetFormat)) {
                clearCptDatasetFormatTracking();
              }
              return {
                datasetFormat: "raw",
                trainOnCompletions: false,
              };
            }

            return {
              datasetFormat,
              trainOnCompletions:
                isRawTextDatasetFormat(datasetFormat)
                  ? false
                  : state.trainOnCompletions,
            };
          }),
        setDataset: (dataset) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set({
            dataset,
            datasetSubset: null,
            datasetSplit: null,
            datasetEvalSplit: null,
            datasetManualMapping: emptyManualMapping(),
            datasetSliceStart: null,
            datasetSliceEnd: null,
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
          });
        },
        setDatasetSubset: (datasetSubset) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set({
            datasetSubset,
            datasetSplit: null,
            datasetEvalSplit: null,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
          });
        },
        setDatasetSplit: (datasetSplit) => {
          set({
            datasetSplit,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
          });

          const state = get();
          const datasetName =
            state.datasetSource === "huggingface"
              ? state.dataset
              : state.uploadedFile;
          if (!datasetName) return;

          runDatasetCheck(datasetName, datasetSplit || "train");
        },
        ensureDatasetChecked: () => {
          const state = get();
          if (state.isCheckingDataset) return;
          if (state.isDatasetImage !== null) return;

          const datasetName =
            state.datasetSource === "huggingface"
              ? state.dataset
              : state.uploadedFile;
          if (!datasetName) return;

          const split = state.datasetSplit || "train";
          runDatasetCheck(datasetName, split);
        },
        setDatasetEvalSplit: (datasetEvalSplit) => {
          set({
            datasetEvalSplit,
            evalSteps: datasetEvalSplit ? 0.1 : 0,
          });
        },
        setDatasetManualMapping: (datasetManualMapping) =>
          set({ datasetManualMapping }),
        setDatasetAdvisorFields: (fields) =>
          set({
            datasetSystemPrompt: fields.systemPrompt ?? get().datasetSystemPrompt,
            datasetUserTemplate: "",  // templates no longer used
            datasetAssistantTemplate: "",  // templates no longer used
            datasetLabelMapping: fields.labelMapping ?? get().datasetLabelMapping,
            datasetAdvisorNotification: fields.notification !== undefined ? fields.notification : get().datasetAdvisorNotification,
          }),
        clearDatasetAdvisorFields: () =>
          set({
            datasetSystemPrompt: "",
            datasetUserTemplate: "",
            datasetAssistantTemplate: "",
            datasetLabelMapping: {},
            datasetAdvisorNotification: null,
          }),
        setDatasetSliceStart: (datasetSliceStart) => set({ datasetSliceStart }),
        setDatasetSliceEnd: (datasetSliceEnd) => set({ datasetSliceEnd }),
        setUploadedFile: (uploadedFile) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set({
            uploadedFile,
            datasetSubset: null,
            datasetSplit: null,
            datasetEvalSplit: null,
            datasetManualMapping: emptyManualMapping(),
            datasetSliceStart: null,
            datasetSliceEnd: null,
            uploadedEvalFile: null,
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
          });
        },
        setUploadedEvalFile: (uploadedEvalFile) => set({
          uploadedEvalFile,
          evalSteps: uploadedEvalFile ? 0.1 : 0,
        }),
        setEpochs: (epochs) => set({ epochs }),
        setContextLength: (contextLength) => set({ contextLength }),
        setLearningRate: (learningRate) => {
          _learningRateManuallySet = true;
          set({ learningRate });
        },
        setEmbeddingLearningRate: (embeddingLearningRate) =>
          set({ embeddingLearningRate }),
        setOptimizerType: (optimizerType) => set({ optimizerType }),
        setLrSchedulerType: (lrSchedulerType) => set({ lrSchedulerType }),
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
        setTrainOnCompletions: (trainOnCompletions) => {
          _trainOnCompletionsManuallySet = true;
          set({ trainOnCompletions });
        },
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
        reset: () => {
          _trainOnCompletionsManuallySet = false;
          _learningRateManuallySet = false;
          _yamlLearningRate = undefined;
          clearCptDatasetFormatTracking();
          set(initialState);
        },
        resetToModelDefaults: () => {
          const { selectedModel } = get();
          if (!selectedModel) return;
          set({ modelDefaultsAppliedFor: null });
          loadAndApplyModelDefaults(selectedModel);
        },
        applyConfigPatch: (config: BackendModelConfig) => {
          const patch = mapBackendModelConfigToTrainingPatch(config);
          // Only clear the manual-edit flag when the config provides a LR,
          // so unrelated config patches don't silently disarm the guard.
          if (patch.learningRate !== undefined) {
            _learningRateManuallySet = false;
          }
          set(patch);
        },
      };
    },
    {
      name: "unsloth_training_config_v1",
      version: 10,
      migrate: (persisted, version) => {
        const s = persisted as Record<string, unknown>;
        if (version < 2 && s.datasetSubset == null && s.datasetConfig != null) {
          s.datasetSubset = s.datasetConfig;
        }
        delete s.datasetConfig;
        if (version < 3 && s.modelDefaultsAppliedFor == null) {
          s.modelDefaultsAppliedFor = null;
        }
        if (version < 4 && s.optimizerType == null) {
          s.optimizerType = DEFAULT_HYPERPARAMS.optimizerType;
        }
        if (version < 5 && s.lrSchedulerType == null) {
          s.lrSchedulerType = DEFAULT_HYPERPARAMS.lrSchedulerType;
        }
        if (version < 6 && s.datasetEvalSplit == null) {
          s.datasetEvalSplit = null;
        }
        if (version < 7) {
          s.datasetSliceStart ??= null;
          s.datasetSliceEnd ??= null;
        }
        if (version < 8) {
          s.datasetSystemPrompt ??= "";
          s.datasetUserTemplate ??= "";
          s.datasetAssistantTemplate ??= "";
          s.datasetLabelMapping ??= {};
          s.datasetAdvisorNotification ??= null;
        }
        if (version < 9) {
          // weight_decay default changed from 0.01 to 0.001.
          if (s.weightDecay === 0.01) {
            s.weightDecay = DEFAULT_HYPERPARAMS.weightDecay;
          }
        }
        if (version < 10 && s.trainingMethod === "cpt") {
          // Backfill CPT defaults for state persisted before they existed.
          s.loraRank = 128;
          s.loraAlpha = 32;
          s.loraVariant = "rslora";
          s.targetModules = CPT_TARGET_MODULES;
          s.datasetFormat = "raw";
          if (s.learningRate == null || s.learningRate === LR_DEFAULT_LORA) {
            s.learningRate = LR_DEFAULT_CPT;
          }
        }
        return s as unknown as TrainingConfigStore;
      },
      partialize: partializePersistedState,
    },
  ),
);
