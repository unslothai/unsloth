// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CPT_TARGET_MODULES,
  DEFAULT_HYPERPARAMS,
  LR_DEFAULT_CPT,
  LR_DEFAULT_FULL,
  LR_DEFAULT_LORA,
  STEPS,
  TARGET_MODULES,
} from "@/config/training";
import { authFetch } from "@/features/auth";
import type { ModelInventoryFormat } from "@/features/inventory";
import { createDebouncedJSONPersistStorage } from "@/lib/debounced-persist-storage";
import { getHfToken } from "@/stores/hf-token-store";
import { isAdapterMethod } from "@/types/training";
import type { DatasetFormat } from "@/types/training";
import type { ModelType, StepNumber, TrainingMethod } from "@/types/training";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { checkDatasetFormat } from "../api/datasets-api";
import { checkVisionModel } from "../api/models-api";
import type { BackendModelConfig } from "../api/models-api";
import { cacheReferenceMatchesSelection } from "../lib/cache-reference";
import { isMissingLocalDatasetCacheError } from "../lib/local-cache-errors";
import {
  fetchCachedModelConfig,
  modelConfigSelectionKey,
} from "../lib/model-config-fetch";
import { inferTrainingModelTypeFromFlags } from "../lib/model-type-inference";
import { mapBackendModelConfigToTrainingPatch } from "../lib/model-defaults";
import { migratePersistedResourceSelections } from "../lib/persisted-resource-selection";
import {
  createTrainingPersistedStateKeys,
  pickPersistedTrainingConfigState,
} from "../lib/training-config-persistence";
import { preserveTrainingDraftFromModelDefaults } from "../lib/training-draft-preservation";
import { isRawTextDatasetFormat } from "../lib/training-methods";
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

    const modelSizeGb = modelSizeBytes / 1024 ** 3;

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
  modelDefaultsAppliedKey: null,
  isCheckingDataset: false,
  isDatasetImage: null,
  isDatasetAudio: false,
  datasetCheckFailed: false,
  datasetMetadataStale: false,
  modelKnownCached: false,
  modelLocalPath: null,
  modelFormat: null,
  datasetKnownCached: false,
  datasetLocalPath: null,
  maxPositionEmbeddings: null,
  trainOnCompletionsManuallySet: false,
  learningRateManuallySet: false,
  trainingMethodManuallySet: false,
  yamlLearningRate: undefined,
  datasetFormatBeforeCpt: null,
  datasetFormatAutoForcedByCpt: false,
  ...DEFAULT_HYPERPARAMS,
};

const PERSISTED_STATE_KEYS = createTrainingPersistedStateKeys(initialState);

function pickPersistedState(
  state: Record<string, unknown>,
): Partial<TrainingConfigStore> {
  return pickPersistedTrainingConfigState(
    state,
    PERSISTED_STATE_KEYS,
  ) as Partial<TrainingConfigStore>;
}

function partializePersistedState(
  state: TrainingConfigStore,
): Partial<TrainingConfigStore> {
  return pickPersistedState(state as unknown as Record<string, unknown>);
}

function safePersistedState(state: unknown): Partial<TrainingConfigStore> {
  if (!state || typeof state !== "object") return {};
  return pickPersistedState(state as Record<string, unknown>);
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
    | "datasetFormatBeforeCpt"
    | "datasetFormatAutoForcedByCpt"
  >
>;

function resetModelManualStatePatch(): Partial<TrainingConfigState> {
  return {
    trainOnCompletionsManuallySet: false,
    learningRateManuallySet: false,
    trainingMethodManuallySet: false,
    yamlLearningRate: undefined,
    datasetFormatBeforeCpt: null,
    datasetFormatAutoForcedByCpt: false,
  };
}

function resetDatasetManualStatePatch(): Partial<TrainingConfigState> {
  return {
    trainOnCompletionsManuallySet: false,
    datasetFormatBeforeCpt: null,
    datasetFormatAutoForcedByCpt: false,
  };
}

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

function clearCptDatasetFormatTrackingPatch(): TrainingMethodStatePatch {
  return {
    datasetFormatBeforeCpt: null,
    datasetFormatAutoForcedByCpt: false,
  };
}

function recordCptDatasetFormatOverridePatch(
  currentDatasetFormat: DatasetFormat,
): TrainingMethodStatePatch {
  if (isRawTextDatasetFormat(currentDatasetFormat)) {
    return clearCptDatasetFormatTrackingPatch();
  }
  return {
    datasetFormatBeforeCpt: currentDatasetFormat,
    datasetFormatAutoForcedByCpt: true,
  };
}

function getRestoreDatasetFormatFromCptPatch(
  state: TrainingConfigState,
): TrainingMethodStatePatch {
  if (
    !state.datasetFormatAutoForcedByCpt ||
    state.datasetFormatBeforeCpt == null
  ) {
    return clearCptDatasetFormatTrackingPatch();
  }

  return {
    datasetFormat: state.datasetFormatBeforeCpt,
    ...clearCptDatasetFormatTrackingPatch(),
  };
}

function resolveTrainingMethodLearningRate(
  state: TrainingConfigState,
  nextMethod: TrainingMethod,
): number | undefined {
  if (state.learningRateManuallySet) {
    return undefined;
  }

  const prevMethod = state.trainingMethod;
  const wasCpt = prevMethod === "cpt";
  const wasAdapter = isAdapterMethod(prevMethod);
  const nowAdapter = isAdapterMethod(nextMethod);

  if (nextMethod === "cpt") {
    return LR_DEFAULT_CPT;
  }
  if (wasCpt && nowAdapter) {
    return state.yamlLearningRate ?? LR_DEFAULT_LORA;
  }
  if (wasAdapter && nowAdapter) {
    return undefined;
  }
  return nowAdapter
    ? (state.yamlLearningRate ?? LR_DEFAULT_LORA)
    : LR_DEFAULT_FULL;
}

function buildTrainingMethodPatch(
  state: TrainingConfigState,
  nextMethod: TrainingMethod,
): TrainingMethodStatePatch {
  const prevMethod = state.trainingMethod;
  const patch: TrainingMethodStatePatch = { trainingMethod: nextMethod };

  if (prevMethod !== "cpt" && nextMethod === "cpt") {
    Object.assign(
      patch,
      recordCptDatasetFormatOverridePatch(state.datasetFormat),
      getCptTrainingPatch(),
    );
  }
  if (prevMethod === "cpt" && nextMethod !== "cpt") {
    Object.assign(
      patch,
      getRestoreFromCptPatch(),
      getRestoreDatasetFormatFromCptPatch(state),
    );
  }

  const learningRate = resolveTrainingMethodLearningRate(state, nextMethod);
  if (learningRate !== undefined) {
    patch.learningRate = learningRate;
  }

  return patch;
}

function clearedModelMetadataPatch(): Partial<TrainingConfigState> {
  return {
    isCheckingVision: false,
    isVisionModel: false,
    isEmbeddingModel: false,
    isAudioModel: false,
    isDatasetImage: null,
    isDatasetAudio: false,
    datasetCheckFailed: false,
    datasetMetadataStale: false,
    isLoadingModelDefaults: false,
    modelDefaultsError: null,
    modelDefaultsAppliedFor: null,
    modelDefaultsAppliedKey: null,
    maxPositionEmbeddings: null,
  };
}

type ModelMetadataOptions = {
  knownCached?: boolean;
  localPath?: string | null;
  modelFormat?: ModelInventoryFormat | null;
  preserveTrainingDraft?: boolean;
};

function modelDefaultsRequestKey(
  modelName: string,
  options: ModelMetadataOptions,
  hfToken: string | null = getHfToken(),
): string {
  return modelConfigSelectionKey(modelName, hfToken, {
    preferLocalCache: options.knownCached,
    localPath: options.localPath,
    modelFormat: options.modelFormat,
  });
}

function pendingModelMetadataPatch(): Partial<TrainingConfigState> {
  return {
    ...clearedModelMetadataPatch(),
    isCheckingVision: true,
    isLoadingModelDefaults: true,
  };
}

type DatasetMetadataSnapshot = Pick<
  TrainingConfigState,
  "isDatasetImage" | "isDatasetAudio"
> | null;

export const useTrainingConfigStore = create<TrainingConfigStore>()(
  persist(
    (set, get) => {
      let _datasetCheckController: AbortController | null = null;
      let _modelConfigController: AbortController | null = null;
      let _datasetCheckRequestId = 0;
      let _modelConfigRequestId = 0;

      const cancelDatasetCheck = () => {
        _datasetCheckController?.abort();
        _datasetCheckController = null;
        _datasetCheckRequestId += 1;
      };

      const loadAndApplyModelDefaults = (
        modelName: string,
        options?: ModelMetadataOptions,
      ) => {
        _modelConfigController?.abort();
        const controller = new AbortController();
        _modelConfigController = controller;
        const requestId = ++_modelConfigRequestId;
        const isCurrentRequest = () =>
          !controller.signal.aborted &&
          _modelConfigController === controller &&
          _modelConfigRequestId === requestId &&
          get().selectedModel === modelName;
        const stateAtStart = get();
        const preserveTrainingDraft = options?.preserveTrainingDraft === true;
        const preferLocalCache =
          options?.knownCached ??
          (stateAtStart.selectedModel === modelName
            ? stateAtStart.modelKnownCached
            : false);
        const localPath =
          options?.localPath ??
          (stateAtStart.selectedModel === modelName
            ? stateAtStart.modelLocalPath
            : null);
        const modelFormat =
          options?.modelFormat ??
          (stateAtStart.selectedModel === modelName
            ? stateAtStart.modelFormat
            : null);
        const hfToken = getHfToken();
        const requestKey = modelDefaultsRequestKey(
          modelName,
          {
            knownCached: preferLocalCache,
            localPath,
            modelFormat,
          },
          hfToken,
        );
        set({
          isLoadingModelDefaults: true,
          isCheckingVision: true,
          modelDefaultsError: null,
        });

        void fetchCachedModelConfig(modelName, controller.signal, {
          preferLocalCache,
          localPath,
          modelFormat,
        })
          .then((modelDetails) => {
            if (!isCurrentRequest()) return;

            let patch = mapBackendModelConfigToTrainingPatch(
              modelDetails.config,
            );

            // If the model config provides a specific learning rate, treat
            // it as authoritative so the async auto-select does not overwrite it.
            const modelConfigHasLR = patch.learningRate !== undefined;
            const yamlLearningRate = patch.learningRate;
            const stateAtDefaults = get();

            // YAML learning rates are tuned for adapter methods (LoRA/QLoRA).
            // If the user is currently on full fine-tune, override with the
            // full-finetune default instead of applying the YAML adapter LR.
            if (modelConfigHasLR) {
              if (stateAtDefaults.learningRateManuallySet) {
                delete patch.learningRate;
              } else if (!isAdapterMethod(stateAtDefaults.trainingMethod)) {
                patch.learningRate = LR_DEFAULT_FULL;
              }
            }

            if (
              patch.trainOnCompletions !== undefined &&
              stateAtDefaults.trainOnCompletionsManuallySet
            ) {
              delete patch.trainOnCompletions;
            }

            // Use backend-provided model_type when available, otherwise
            // infer from capability flags.
            const backendIsEmbedding = !!modelDetails.is_embedding;
            const backendIsAudio = !!modelDetails.is_audio;
            const backendIsVision = !!modelDetails.is_vision;
            const backendModelType: ModelType =
              modelDetails.model_type ??
              inferTrainingModelTypeFromFlags({
                isEmbedding: backendIsEmbedding,
                isAudio: backendIsAudio,
                isVision: backendIsVision,
              });
            const selectedType = stateAtDefaults.modelType;
            const inferredModelType: ModelType =
              backendModelType === "text" &&
              (selectedType === "vision" ||
                selectedType === "audio" ||
                selectedType === "embeddings")
                ? selectedType
                : backendModelType;
            const isEmbedding =
              backendIsEmbedding || inferredModelType === "embeddings";
            const isAudio = backendIsAudio || inferredModelType === "audio";
            const isVision = backendIsVision || inferredModelType === "vision";

            // If vision model + image dataset already known, override
            // trainOnCompletions to false regardless of backend default.
            if (isVision && get().isDatasetImage === true) {
              patch.trainOnCompletions = false;
            }

            // Pure audio model -> always uncheck trainOnCompletions.
            if (isAudio && !isVision) {
              patch.trainOnCompletions = false;
            }
            // Audio-capable vision model (e.g. gemma3n) + audio dataset -> uncheck.
            if (isAudio && isVision && get().isDatasetAudio) {
              patch.trainOnCompletions = false;
            }
            if (preserveTrainingDraft) {
              patch = preserveTrainingDraftFromModelDefaults(patch);
            }

            // Auto-select training method based on model size vs GPU memory.
            // If model_size * 1.5 * context_scale fits in free VRAM, use LoRA 16-bit.
            // Otherwise use QLoRA 4-bit.
            // Auto-select LoRA vs QLoRA based on GPU memory.
            // Skip if user has manually chosen CPT -- don't override it.
            const modelSizeBytes = modelDetails.model_size_bytes;
            if (
              !preserveTrainingDraft &&
              modelSizeBytes &&
              modelSizeBytes > 0 &&
              get().trainingMethod !== "cpt"
            ) {
              void autoSelectTrainingMethod(
                modelSizeBytes,
                patch.contextLength ?? get().contextLength,
              ).then((method) => {
                if (!isCurrentRequest()) return;
                const current = get();
                if (current.trainingMethod === "cpt") return;
                if (current.trainingMethodManuallySet) return;
                if (method) {
                  const lrPatch =
                    !current.learningRateManuallySet && !modelConfigHasLR
                      ? {
                          learningRate:
                            method === "full"
                              ? LR_DEFAULT_FULL
                              : LR_DEFAULT_LORA,
                        }
                      : {};
                  set({ trainingMethod: method, ...lrPatch });
                }
              });
            }

            // Preserve CPT hyperparams: YAML adapter defaults (r/alpha/targets/LR)
            // are tuned for standard LoRA and would otherwise clobber CPT settings.
            const stateBeforeApply = get();
            const cptOverrides: TrainingMethodStatePatch =
              !preserveTrainingDraft && stateBeforeApply.trainingMethod === "cpt"
                ? getCptModelDefaultsPatch()
                : {};
            if (stateBeforeApply.learningRateManuallySet) {
              delete cptOverrides.learningRate;
            }
            if (!isCurrentRequest()) return;

            set({
              ...patch,
              ...cptOverrides,
              yamlLearningRate,
              modelType: inferredModelType,
              isVisionModel: isVision,
              isEmbeddingModel: isEmbedding,
              isAudioModel: isAudio,
              isLoadingModelDefaults: false,
              isCheckingVision: false,
              modelDefaultsError: null,
              modelDefaultsAppliedFor: modelName,
              modelDefaultsAppliedKey: requestKey,
              maxPositionEmbeddings:
                modelDetails.max_position_embeddings ?? null,
            });

            recheckSelectedDataset();
          })
          .catch((error) => {
            if (!isCurrentRequest()) return;

            set({
              isLoadingModelDefaults: false,
              modelDefaultsError:
                error instanceof Error
                  ? error.message
                  : "Failed to load model defaults",
            });

            // Fallback vision check if config endpoint fails; the probe only
            // resolves vision-vs-not, so a user-selected audio/embeddings
            // modality is preserved rather than flattened to text.
            void checkVisionModel(modelName, getHfToken() || undefined, {
              preferLocalCache,
              localPath,
            })
              .then((isVision) => {
                if (!isCurrentRequest()) return;
                const priorType = get().modelType;
                const fallbackType: ModelType = isVision
                  ? "vision"
                  : priorType === "audio" || priorType === "embeddings"
                    ? priorType
                    : "text";
                set({
                  modelType: fallbackType,
                  isVisionModel: isVision,
                  isEmbeddingModel: fallbackType === "embeddings",
                  isAudioModel: fallbackType === "audio",
                  isCheckingVision: false,
                });
                recheckSelectedDataset();
              })
              .catch(() => {
                if (!isCurrentRequest()) return;
                set({ isCheckingVision: false });
              });
          });
      };

      const runDatasetCheck = (
        datasetName: string,
        split: string,
        staleMetadata?: DatasetMetadataSnapshot,
      ) => {
        cancelDatasetCheck();
        const controller = new AbortController();
        _datasetCheckController = controller;
        const requestId = ++_datasetCheckRequestId;
        set({
          isCheckingDataset: true,
          datasetCheckFailed: false,
          datasetMetadataStale: false,
        });

        const state = get();
        const selectedModelAtStart = state.selectedModel;
        const isCurrentRequest = () =>
          !controller.signal.aborted &&
          _datasetCheckController === controller &&
          _datasetCheckRequestId === requestId &&
          get().selectedModel === selectedModelAtStart;
        checkDatasetFormat({
          datasetName,
          hfToken: getHfToken() || null,
          subset: state.datasetSubset,
          split,
          isVlm: state.isVisionModel,
          preferLocalCache: state.datasetKnownCached,
          localPath: state.datasetLocalPath,
          signal: controller.signal,
        })
          .then((res) => {
            if (!isCurrentRequest()) return;
            const isImage = !!res.is_image;
            const isAudio = !!res.is_audio;
            const updates: Record<string, unknown> = {
              isDatasetImage: isImage,
              isDatasetAudio: isAudio,
              isCheckingDataset: false,
              datasetCheckFailed: false,
              datasetMetadataStale: false,
            };
            if (!get().trainOnCompletionsManuallySet) {
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
          .catch((error) => {
            if (!isCurrentRequest()) return;
            const fallbackImage =
              staleMetadata?.isDatasetImage ?? state.isDatasetImage;
            const fallbackAudio =
              staleMetadata?.isDatasetAudio ?? state.isDatasetAudio;
            const hasRecoverablePrior =
              fallbackImage !== null &&
              !isMissingLocalDatasetCacheError(error) &&
              (state.datasetKnownCached || staleMetadata !== null);
            if (hasRecoverablePrior) {
              set({
                isDatasetImage: fallbackImage,
                isDatasetAudio: fallbackAudio,
                isCheckingDataset: false,
                datasetCheckFailed: false,
                datasetMetadataStale: true,
              });
              return;
            }
            set({
              isDatasetImage: null,
              isDatasetAudio: false,
              isCheckingDataset: false,
              datasetCheckFailed: true,
              datasetMetadataStale: false,
            });
          });
      };

      const recheckSelectedDataset = () => {
        const state = get();
        const datasetName =
          state.datasetSource === "huggingface"
            ? state.dataset
            : state.uploadedFile;
        if (!datasetName) return;
        runDatasetCheck(datasetName, state.datasetSplit || "train");
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
        datasetCheckFailed: false,
        datasetMetadataStale: false,
        datasetKnownCached: false,
        datasetLocalPath: null,
        ...resetDatasetManualStatePatch(),
      });

      return {
        ...initialState,
        setStep: (step) => set({ currentStep: step }),
        nextStep: () => set({ currentStep: clampStep(get().currentStep + 1) }),
        prevStep: () => set({ currentStep: clampStep(get().currentStep - 1) }),
        setModelType: (modelType) => {
          _modelConfigController?.abort();
          _modelConfigController = null;
          cancelDatasetCheck();

          set({
            modelType,
            selectedModel: null,
            modelKnownCached: false,
            modelLocalPath: null,
            modelFormat: null,
            ...clearedModelMetadataPatch(),
            ...resetModelManualStatePatch(),
          });
        },
        setSelectedModel: (selectedModel, options) => {
          const previousModel = get().selectedModel;
          const modelChanged = selectedModel !== previousModel;

          if (!selectedModel) {
            _modelConfigController?.abort();
            _modelConfigController = null;
            cancelDatasetCheck();
            set({
              selectedModel,
              modelKnownCached: false,
              modelLocalPath: null,
              modelFormat: null,
              ...clearedModelMetadataPatch(),
              ...resetModelManualStatePatch(),
            });
            return;
          }

          if (modelChanged) {
            _modelConfigController?.abort();
            _modelConfigController = null;
            cancelDatasetCheck();
          }
          const hasHintOptions = options !== undefined;
          const currentState = get();
          const knownCached = hasHintOptions
            ? Boolean(options?.knownCached)
            : modelChanged
              ? false
              : currentState.modelKnownCached;
          const localPath = hasHintOptions
            ? (options?.localPath ?? null)
            : modelChanged
              ? null
              : currentState.modelLocalPath;
          const modelFormat = hasHintOptions
            ? (options?.modelFormat ?? null)
            : modelChanged
              ? null
              : currentState.modelFormat;
          const defaultsKey = modelDefaultsRequestKey(selectedModel, {
            knownCached,
            localPath,
            modelFormat,
          });
          const shouldLoadDefaults =
            modelChanged ||
            currentState.modelDefaultsAppliedKey !== defaultsKey;
          set({
            selectedModel,
            modelKnownCached: knownCached,
            modelLocalPath: localPath,
            modelFormat,
            ...(modelChanged ? resetModelManualStatePatch() : {}),
            ...(shouldLoadDefaults
              ? pendingModelMetadataPatch()
              : { modelDefaultsError: null }),
          });
          if (shouldLoadDefaults) {
            void loadAndApplyModelDefaults(selectedModel, {
              knownCached,
              localPath,
              modelFormat,
            });
          }
        },
        selectTrainingModel: (selectedModel, modelType, options) => {
          const currentState = get();
          const previousModel = currentState.selectedModel;
          const modelChanged = selectedModel !== previousModel;
          const modelTypeChanged = currentState.modelType !== modelType;
          const preserveTrainingDraft =
            modelChanged && options?.preserveTrainingDraft === true;
          const hasHintOptions = options !== undefined;
          const knownCached = hasHintOptions
            ? Boolean(options?.knownCached)
            : modelChanged
              ? false
              : currentState.modelKnownCached;
          const localPath = hasHintOptions
            ? (options?.localPath ?? null)
            : modelChanged
              ? null
              : currentState.modelLocalPath;
          const modelFormat = hasHintOptions
            ? (options?.modelFormat ?? null)
            : modelChanged
              ? null
              : currentState.modelFormat;
          const defaultsKey = modelDefaultsRequestKey(selectedModel, {
            knownCached,
            localPath,
            modelFormat,
          });
          if (modelChanged) {
            _modelConfigController?.abort();
            _modelConfigController = null;
            cancelDatasetCheck();
          }
          const shouldLoadDefaults =
            modelChanged ||
            modelTypeChanged ||
            currentState.modelDefaultsAppliedKey !== defaultsKey;
          set({
            modelType,
            selectedModel,
            modelKnownCached: knownCached,
            modelLocalPath: localPath,
            modelFormat,
            ...(modelChanged && !preserveTrainingDraft
              ? resetModelManualStatePatch()
              : {}),
            ...(shouldLoadDefaults
              ? pendingModelMetadataPatch()
              : { modelDefaultsError: null }),
          });
          if (shouldLoadDefaults) {
            void loadAndApplyModelDefaults(selectedModel, {
              knownCached,
              localPath,
              modelFormat,
              preserveTrainingDraft,
            });
          }
        },
        ensureModelDefaultsLoaded: () => {
          const state = get();
          if (!state.selectedModel) return;
          if (state.isLoadingModelDefaults) return;
          const defaultsKey = modelDefaultsRequestKey(state.selectedModel, {
            knownCached: state.modelKnownCached,
            localPath: state.modelLocalPath,
            modelFormat: state.modelFormat,
          });
          if (state.modelDefaultsAppliedKey === defaultsKey) return;
          void loadAndApplyModelDefaults(state.selectedModel);
        },
        setTrainingMethod: (trainingMethod) => {
          const state = get();
          set({
            ...buildTrainingMethodPatch(state, trainingMethod),
            trainingMethodManuallySet: true,
          });
        },
        setDatasetSource: (datasetSource) => set({ datasetSource }),
        selectHfDataset: (dataset, options) => {
          cancelDatasetCheck();
          const previous = get();
          const sameDataset =
            Boolean(dataset) &&
            previous.datasetSource === "huggingface" &&
            previous.dataset === dataset;
          const hasHintOptions = options !== undefined;
          const staleMetadata = sameDataset
            ? {
                isDatasetImage: previous.isDatasetImage,
                isDatasetAudio: previous.isDatasetAudio,
              }
            : null;
          if (sameDataset && dataset) {
            set({
              datasetSource: "huggingface",
              uploadedFile: null,
              datasetKnownCached: hasHintOptions
                ? Boolean(options?.knownCached)
                : previous.datasetKnownCached,
              datasetLocalPath: hasHintOptions
                ? (options?.localPath ?? null)
                : previous.datasetLocalPath,
              isDatasetImage: null,
              isDatasetAudio: false,
              isCheckingDataset: false,
              datasetCheckFailed: false,
              datasetMetadataStale: false,
            });
            runDatasetCheck(
              dataset,
              previous.datasetSplit || "train",
              staleMetadata,
            );
            return;
          }
          set({
            datasetSource: "huggingface",
            dataset,
            uploadedFile: null,
            ...resetDatasetState(),
            datasetKnownCached: Boolean(dataset && options?.knownCached),
            datasetLocalPath: dataset ? (options?.localPath ?? null) : null,
          });
          if (dataset) {
            runDatasetCheck(dataset, "train", staleMetadata);
          }
        },
        selectLocalDataset: (uploadedFile) => {
          cancelDatasetCheck();
          const previous = get();
          const staleMetadata =
            previous.datasetSource === "upload" &&
            previous.uploadedFile === uploadedFile
              ? {
                  isDatasetImage: previous.isDatasetImage,
                  isDatasetAudio: previous.isDatasetAudio,
                }
              : null;
          set({
            datasetSource: "upload",
            dataset: null,
            uploadedFile,
            ...resetDatasetState(),
            datasetLocalPath: null,
          });
          if (uploadedFile) {
            runDatasetCheck(uploadedFile, "train", staleMetadata);
          }
        },
        setDatasetFormat: (datasetFormat) =>
          set((state) => {
            if (state.trainingMethod === "cpt") {
              const trackingPatch = isRawTextDatasetFormat(datasetFormat)
                ? clearCptDatasetFormatTrackingPatch()
                : {};
              return {
                datasetFormat: "raw",
                trainOnCompletions: false,
                ...trackingPatch,
              };
            }

            return {
              datasetFormat,
              trainOnCompletions: isRawTextDatasetFormat(datasetFormat)
                ? false
                : state.trainOnCompletions,
            };
          }),
        setDataset: (dataset) => {
          cancelDatasetCheck();
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
            datasetCheckFailed: false,
            datasetMetadataStale: false,
            datasetKnownCached: false,
            datasetLocalPath: null,
            ...resetDatasetManualStatePatch(),
          });
        },
        setDatasetSubset: (datasetSubset) => {
          cancelDatasetCheck();
          const previous = get();
          const staleMetadata = previous.datasetKnownCached
            ? {
                isDatasetImage: previous.isDatasetImage,
                isDatasetAudio: previous.isDatasetAudio,
              }
            : null;
          set({
            datasetSubset,
            datasetSplit: null,
            datasetEvalSplit: null,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
            datasetCheckFailed: false,
            datasetMetadataStale: false,
            ...resetDatasetManualStatePatch(),
          });

          const state = get();
          const datasetName =
            state.datasetSource === "huggingface"
              ? state.dataset
              : state.uploadedFile;
          if (!datasetName) return;

          runDatasetCheck(datasetName, "train", staleMetadata);
        },
        setDatasetSplit: (datasetSplit) => {
          const previous = get();
          const staleMetadata = previous.datasetKnownCached
            ? {
                isDatasetImage: previous.isDatasetImage,
                isDatasetAudio: previous.isDatasetAudio,
              }
            : null;
          set({
            datasetSplit,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
            datasetCheckFailed: false,
            datasetMetadataStale: false,
            ...resetDatasetManualStatePatch(),
          });

          const state = get();
          const datasetName =
            state.datasetSource === "huggingface"
              ? state.dataset
              : state.uploadedFile;
          if (!datasetName) return;

          runDatasetCheck(datasetName, datasetSplit || "train", staleMetadata);
        },
        ensureDatasetChecked: () => {
          const state = get();
          if (state.isLoadingModelDefaults || state.isCheckingVision) return;
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
            datasetSystemPrompt:
              fields.systemPrompt ?? get().datasetSystemPrompt,
            datasetUserTemplate: "", // templates no longer used
            datasetAssistantTemplate: "", // templates no longer used
            datasetLabelMapping:
              fields.labelMapping ?? get().datasetLabelMapping,
            datasetAdvisorNotification:
              fields.notification !== undefined
                ? fields.notification
                : get().datasetAdvisorNotification,
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
          cancelDatasetCheck();
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
            datasetCheckFailed: false,
            datasetMetadataStale: false,
            datasetKnownCached: false,
            datasetLocalPath: null,
            ...resetDatasetManualStatePatch(),
          });
        },
        setUploadedEvalFile: (uploadedEvalFile) =>
          set({
            uploadedEvalFile,
            evalSteps: uploadedEvalFile ? 0.1 : 0,
          }),
        setEpochs: (epochs) => set({ epochs }),
        setContextLength: (contextLength) => set({ contextLength }),
        setLearningRate: (learningRate) => {
          set({ learningRate, learningRateManuallySet: true });
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
          set({ trainOnCompletions, trainOnCompletionsManuallySet: true });
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
        clearSelectedDatasetCacheReference: (dataset, localPath) => {
          const state = get();
          if (
            state.datasetSource !== "huggingface" ||
            !cacheReferenceMatchesSelection({
              currentId: state.dataset,
              expectedId: dataset,
              knownCached: state.datasetKnownCached,
              currentLocalPath: state.datasetLocalPath,
              expectedLocalPath: localPath,
            })
          ) {
            return;
          }
          cancelDatasetCheck();
          set({
            datasetKnownCached: false,
            datasetLocalPath: null,
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
            datasetCheckFailed: false,
            datasetMetadataStale: false,
          });
          runDatasetCheck(dataset, state.datasetSplit || "train");
        },
        clearSelectedModelCacheReference: (model, localPath) => {
          const state = get();
          if (
            !cacheReferenceMatchesSelection({
              currentId: state.selectedModel,
              expectedId: model,
              knownCached: state.modelKnownCached,
              currentLocalPath: state.modelLocalPath,
              expectedLocalPath: localPath,
            })
          ) {
            return;
          }
          _modelConfigController?.abort();
          _modelConfigController = null;
          cancelDatasetCheck();
          set({
            modelKnownCached: false,
            modelLocalPath: null,
            modelFormat: null,
            ...pendingModelMetadataPatch(),
          });
          void loadAndApplyModelDefaults(model, {
            knownCached: false,
            localPath: null,
            modelFormat: null,
          });
        },
        setSelectedModelCacheReference: (model, options) => {
          const state = get();
          if (state.selectedModel !== model) {
            return;
          }
          const localPath = options.localPath ?? null;
          const modelFormat = options.modelFormat ?? null;
          const defaultsKey = modelDefaultsRequestKey(model, {
            knownCached: true,
            localPath,
            modelFormat,
          });
          const shouldLoadDefaults =
            state.modelDefaultsAppliedKey !== defaultsKey;
          set({
            modelKnownCached: true,
            modelLocalPath: localPath,
            modelFormat,
            ...(shouldLoadDefaults
              ? pendingModelMetadataPatch()
              : { modelDefaultsError: null }),
          });
          if (shouldLoadDefaults) {
            void loadAndApplyModelDefaults(model, {
              knownCached: true,
              localPath,
              modelFormat,
              preserveTrainingDraft: true,
            });
          }
        },
        canProceed: () => canProceedForStep(get()),
        reset: () => {
          _modelConfigController?.abort();
          _modelConfigController = null;
          cancelDatasetCheck();
          set(initialState);
        },
        resetToModelDefaults: () => {
          const { selectedModel } = get();
          if (!selectedModel) return;
          set({
            modelDefaultsAppliedFor: null,
            modelDefaultsAppliedKey: null,
            ...resetModelManualStatePatch(),
          });
          loadAndApplyModelDefaults(selectedModel);
        },
        applyConfigPatch: (config: BackendModelConfig) => {
          const patch = mapBackendModelConfigToTrainingPatch(config);
          // Only clear the manual-edit flag when the config provides a LR,
          // so unrelated config patches don't silently disarm the guard.
          if (patch.learningRate !== undefined) {
            set({
              ...patch,
              learningRateManuallySet: false,
              yamlLearningRate: patch.learningRate,
            });
            return;
          }
          set(patch);
        },
      };
    },
    {
      name: "unsloth_training_config_v1",
      version: 15,
      storage: createDebouncedJSONPersistStorage<Partial<TrainingConfigStore>>({
        delayMs: 250,
      }),
      // Keep legacy state migrations until at least 2026-11-24 for direct upgrades from older Studio builds.
      migrate: (persisted, version) => {
        const s = persisted as Record<string, unknown>;
        if (version < 2 && s.datasetSubset == null && s.datasetConfig != null) {
          s.datasetSubset = s.datasetConfig;
        }
        if (version < 2) {
          delete s.datasetConfig;
        }
        if (version < 12) {
          delete s.hfToken;
        }
        if (version < 13) {
          s.datasetLocalPath = null;
        }
        if (version < 14) {
          delete s.isDatasetImage;
          delete s.isDatasetAudio;
        }
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
        if (version < 15) {
          migratePersistedResourceSelections(s);
        }
        return s as unknown as TrainingConfigStore;
      },
      partialize: partializePersistedState,
      merge: (persisted, current) => ({
        ...current,
        ...safePersistedState(persisted),
      }),
    },
  ),
);
