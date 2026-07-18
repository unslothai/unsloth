// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CPT_TARGET_MODULES, DEFAULT_HYPERPARAMS, LR_DEFAULT_CPT, LR_DEFAULT_FULL, LR_DEFAULT_LORA, STEPS, TARGET_MODULES } from "@/config/training";
import { authFetch } from "@/features/auth";
import { getHfToken, mirrorHfTokenInto, useHfTokenStore } from "@/features/hub";
import { isAdapterMethod } from "@/types/training";
import type { DatasetFormat } from "@/types/training";
import type { ModelType, StepNumber, TrainingMethod } from "@/types/training";
import { toast } from "sonner";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { checkDatasetFormat } from "../api/datasets-api";
import { checkVisionModel, getModelConfig } from "../api/models-api";
import { mapBackendModelConfigToTrainingPatch } from "../lib/model-defaults";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import { validateS3Source } from "../lib/validation";
import type { BackendModelConfig } from "../api/models-api";
import type { TrainingConfigState, TrainingConfigStore } from "../types/config";

const MIN_STEP: StepNumber = 1;
const MAX_STEP: StepNumber = STEPS.length as StepNumber;

/**
 * Auto-select LoRA (16-bit) vs QLoRA (4-bit) by model size and GPU memory.
 * Use "lora" if model_size_gb * 1.5 * context_scale fits in free VRAM, else "qlora".
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
  projectName: "",
  trainingMethod: "qlora",
  hfToken: "",
  datasetSource: "huggingface",
  datasetFormat: "auto",
  dataset: null,
  datasetSubset: null,
  datasetSplit: null,
  datasetEvalSplit: null,
  datasetStreaming: false,
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
  contextLengthManuallySet: false,
  trainOnCompletionsManuallySet: false,
  learningRateManuallySet: false,
  trainingMethodManuallySet: false,
  ...DEFAULT_HYPERPARAMS,
};

// AbortController for in-flight dataset multimodal checks.
let _datasetCheckController: AbortController | null = null;

// AbortController for in-flight model default loads.
let _modelConfigController: AbortController | null = null;

// Stash the YAML learning rate so setTrainingMethod can restore it when
// switching back from full to adapter.
let _yamlLearningRate: number | undefined = undefined;

// Track whether entering CPT auto-forced datasetFormat="raw" so that
// leaving CPT can restore the prior user-visible format.
let _datasetFormatBeforeCpt: DatasetFormat | null = null;
let _datasetFormatAutoForcedByCpt = false;

// modelType / isVisionModel / isAudioModel persist so multimodal-only UI
// paints right on reload; the model-config fetch still re-derives them.
// hfToken mirrors the shared hf-token-store and is persisted there instead.
const NON_PERSISTED_STATE_KEYS: ReadonlySet<keyof TrainingConfigState> = new Set([
  "hfToken",
  "isCheckingVision",
  "isEmbeddingModel",
  "isLoadingModelDefaults",
  "modelDefaultsError",
  "modelDefaultsAppliedFor",
  "isCheckingDataset",
  "isDatasetImage",
  "isDatasetAudio",
  "maxPositionEmbeddings",
  "s3Config",
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
      if (state.datasetSource === "upload") {
        return state.uploadedFile !== null;
      }
      if (state.datasetSource === "s3") {
        return validateS3Source(state).ok;
      }
      return state.dataset !== null;
    case 4:
    case 5:
      return true;
    default:
      return false;
  }
}

// Single source of truth for the "streaming + eval needs a distinct split"
// rule. Shared between the store's compatibility patch and the UI gate
// (DatasetSection) so the two never drift apart.
export function hasSeparateStreamingEvalSplit(
  state: Pick<
    TrainingConfigState,
    "evalSteps" | "datasetSplit" | "datasetEvalSplit"
  >,
): boolean {
  if (state.evalSteps <= 0) return true;
  const trainSplit = state.datasetSplit || "train";
  return !!state.datasetEvalSplit && state.datasetEvalSplit !== trainSplit;
}

function streamingCompatiblePatch(
  state: TrainingConfigState,
): Partial<TrainingConfigState> {
  const patch: Partial<TrainingConfigState> = {};

  if (state.datasetStreaming && state.maxSteps <= 0) {
    patch.datasetStreaming = false;
  }

  // Evaluate the remaining streaming constraints against the *post-patch*
  // streaming value. If streaming is being turned off in this same patch
  // (e.g. maxSteps dropped to 0), its other constraints are moot and we must
  // NOT clobber unrelated user preferences like trainOnCompletions/evalSteps.
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

// streamingCompatiblePatch can silently flip streaming-coupled fields. Surface a
// toast when it does, so the indirect setters (split / eval-split / max-steps /
// eval-steps) match setDatasetStreaming's "tell the user what changed" behavior.
function notifyStreamingCompat(patch: Partial<TrainingConfigState>): void {
  if (patch.datasetStreaming === false) {
    toast.info("Streaming turned off: streaming needs a fixed Max Steps > 0.");
    return;
  }
  const disabled = [
    patch.trainOnCompletions === false && "assistant-completions-only",
    patch.evalSteps === 0 && "evaluation (needs a separate eval split)",
  ].filter(Boolean);
  if (disabled.length > 0) {
    toast.info(
      `Adjusted for streaming. Disabled incompatible options: ${disabled.join(", ")}.`,
    );
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

function getCptModelDefaultsPatch(learningRateManuallySet: boolean): TrainingMethodStatePatch {
  return {
    ...getCptTrainingPatch(),
    ...(learningRateManuallySet ? {} : { learningRate: LR_DEFAULT_CPT }),
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
  learningRateManuallySet: boolean,
): number | undefined {
  if (learningRateManuallySet) {
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
  learningRateManuallySet: boolean,
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

  const learningRate = resolveTrainingMethodLearningRate(prevMethod, nextMethod, learningRateManuallySet);
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
        _yamlLearningRate = undefined;
        set({
          isLoadingModelDefaults: true,
          isCheckingVision: true,
          modelDefaultsError: null,
        });

        void getModelConfig(modelName, controller.signal, get().hfToken || undefined)
          .then((modelDetails) => {
            if (controller.signal.aborted) return;
            if (get().selectedModel !== modelName) return;

            const patch = mapBackendModelConfigToTrainingPatch(modelDetails.config);
            const contextLengthManuallySet = get().contextLengthManuallySet;
            const trainOnCompletionsManuallySet = get().trainOnCompletionsManuallySet;
            const learningRateManuallySet = get().learningRateManuallySet;
            const effectiveContextLength = contextLengthManuallySet
              ? get().contextLength
              : patch.contextLength ?? get().contextLength;

            // User explicitly set context length: discard the model default so
            // the visible value is preserved. autoSelectTrainingMethod will
            // receive the preserved visible value via the explicit fallback.
            if (contextLengthManuallySet) {
              delete patch.contextLength;
            }
            if (learningRateManuallySet) {
              delete patch.learningRate;
            }
            if (trainOnCompletionsManuallySet) {
              delete patch.trainOnCompletions;
            }

            // Treat a model-config LR as authoritative so async auto-select
            // won't overwrite it.
            const modelConfigHasLR = patch.learningRate !== undefined;
            _yamlLearningRate = patch.learningRate;

            // YAML LRs are tuned for adapters (LoRA/QLoRA); on full fine-tune,
            // use the full-finetune default instead of the YAML adapter LR.
            if (!learningRateManuallySet && modelConfigHasLR && !isAdapterMethod(get().trainingMethod)) {
              patch.learningRate = LR_DEFAULT_FULL;
            }

            const isAudio = !!modelDetails.is_audio;
            // Vision model + known image dataset: force trainOnCompletions off.
            if (modelDetails.is_vision && get().isDatasetImage === true) {
              patch.trainOnCompletions = false;
            }
            // Pure audio model -> always uncheck trainOnCompletions.
            if (isAudio && !modelDetails.is_vision) {
              patch.trainOnCompletions = false;
            }
            // Audio-capable vision model (e.g. gemma3n) + audio dataset -> uncheck.
            if (isAudio && modelDetails.is_vision && get().isDatasetAudio) {
              patch.trainOnCompletions = false;
            }

            // Use backend model_type when available, else infer from flags.
            const isEmbedding = !!modelDetails.is_embedding;
            const inferredModelType: ModelType = modelDetails.model_type
              ?? (isEmbedding ? "embeddings" : modelDetails.is_vision ? "vision" : modelDetails.is_audio ? "audio" : "text");

            // Preserve CPT hyperparams: YAML adapter defaults (r/alpha/targets/LR)
            // are tuned for standard LoRA and would clobber CPT settings.
            const cptOverrides =
              get().trainingMethod === "cpt"
                ? getCptModelDefaultsPatch(learningRateManuallySet)
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

            // Auto-select LoRA vs QLoRA by model size vs GPU memory (see
            // autoSelectTrainingMethod). Skip if the user chose CPT or already
            // chose a method manually.
            const modelSizeBytes = modelDetails.model_size_bytes;
            if (
              modelSizeBytes &&
              modelSizeBytes > 0 &&
              !get().trainingMethodManuallySet &&
              get().trainingMethod !== "cpt"
            ) {
              void autoSelectTrainingMethod(modelSizeBytes, effectiveContextLength)
                .then((method) => {
                  if (get().selectedModel !== modelName) return;
                  if (get().trainingMethodManuallySet) return;
                  if (get().trainingMethod === "cpt") return;
                  if (method) {
                    const lrPatch = !get().learningRateManuallySet && !modelConfigHasLR
                      ? { learningRate: method === "full" ? LR_DEFAULT_FULL : LR_DEFAULT_LORA }
                      : {};
                    set({ trainingMethod: method, ...lrPatch });
                  }
                });
            }
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
              // Defaults load failed; reset so no prior model's value lingers.
              visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
            });

            // Fallback vision check; pass the token so a gated/private VLM classifies right.
            void checkVisionModel(modelName, get().hfToken || undefined)
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
            trainOnCompletionsManuallySet: false,
            contextLengthManuallySet: false,
            learningRateManuallySet: false,
            trainingMethodManuallySet: false,
          });
        },
        setSelectedModel: (selectedModel) => {
          const previousModel = get().selectedModel;
          // Reset vision_image_size on a true switch only; same-model reloads
          // go through the mapper, which preserves the user's choice.
          const patch: {
            selectedModel: string | null;
            modelDefaultsError: null;
            visionImageSize?: number | null;
            trustRemoteCode?: boolean;
            approvedRemoteCodeFingerprint?: string | null;
            isVisionModel?: boolean;
            isAudioModel?: boolean;
            isEmbeddingModel?: boolean;
            trainOnCompletionsManuallySet?: boolean;
            contextLengthManuallySet?: boolean;
            learningRateManuallySet?: boolean;
            trainingMethodManuallySet?: boolean;
          } = {
            selectedModel,
            modelDefaultsError: null,
          };
          if (selectedModel !== previousModel) {
            patch.visionImageSize = DEFAULT_HYPERPARAMS.visionImageSize;
            // Clear the prior model's approval so a clean model is not trained with a
            // stale trust_remote_code=true (disables fused CE). Its own YAML default is
            // re-applied below, and a custom-code model re-opens the dialog before start.
            patch.trustRemoteCode = false;
            patch.approvedRemoteCodeFingerprint = null;
            // Reset capability flags so a mid-fetch reload can't persist the
            // previous model's vision/audio flags against the new model.
            patch.isVisionModel = false;
            patch.isAudioModel = false;
            patch.isEmbeddingModel = false;
            patch.trainOnCompletionsManuallySet = false;
            patch.contextLengthManuallySet = false;
            patch.learningRateManuallySet = false;
            patch.trainingMethodManuallySet = false;
          }
          set(patch);

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
              trainOnCompletionsManuallySet: false,
              contextLengthManuallySet: false,
              learningRateManuallySet: false,
              trainingMethodManuallySet: false,
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
        setProjectName: (projectName) => set({ projectName }),
        setTrainingMethod: (trainingMethod) => {
          const state = get();
          set(
            {
              ...buildTrainingMethodPatch(
                state.trainingMethod,
                trainingMethod,
                state.datasetFormat,
                state.learningRateManuallySet,
              ),
              trainingMethodManuallySet: true,
            },
          );
        },
        setHfToken: (hfToken) => useHfTokenStore.getState().setToken(hfToken),
        setDatasetSource: (datasetSource) => set({ datasetSource }),
        selectHfDataset: (dataset) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          set({
            datasetSource: "huggingface",
            dataset,
            trainOnCompletionsManuallySet: false,
            uploadedFile: null,
            ...resetDatasetState(),
          });
        },
        selectLocalDataset: (uploadedFile) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          set({
            datasetSource: "upload",
            dataset: null,
            trainOnCompletionsManuallySet: false,
            uploadedFile,
            ...resetDatasetState(),
          });
          if (uploadedFile) {
            runDatasetCheck(uploadedFile, "train");
          }
        },
        selectS3Source: () => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          set({
            datasetSource: "s3",
            dataset: null,
            trainOnCompletionsManuallySet: false,
            uploadedFile: null,
            ...resetDatasetState(),
          });
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
          set({
            dataset,
            trainOnCompletionsManuallySet: false,
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
          set({
            datasetSubset,
            trainOnCompletionsManuallySet: false,
            datasetSplit: null,
            datasetEvalSplit: null,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
          });
        },
        setDatasetSplit: (datasetSplit) => {
          const state = get();
          const nextState = { ...state, datasetSplit };
          const streamingPatch = streamingCompatiblePatch(nextState);
          set({
            datasetSplit,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);

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
          const state = get();
          const evalSteps = datasetEvalSplit ? 0.1 : 0;
          const streamingPatch = streamingCompatiblePatch({
            ...state,
            datasetEvalSplit,
            evalSteps,
          });
          set({
            datasetEvalSplit,
            evalSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setDatasetStreaming: (datasetStreaming) => {
          if (!datasetStreaming) {
            set({ datasetStreaming: false });
            return;
          }

          const state = get();
          if (state.maxSteps <= 0) {
            set({ datasetStreaming: false });
            toast.warning(
              "Streaming needs a fixed Max Steps (streaming datasets have no known length). Set Max Steps > 0 first.",
            );
            return;
          }

          const dropsTrainOnCompletions = state.trainOnCompletions;
          const dropsEval = !hasSeparateStreamingEvalSplit(state);

          set({
            datasetStreaming: true,
            trainOnCompletions: false,
            evalSteps: dropsEval ? 0 : state.evalSteps,
          });

          if (dropsTrainOnCompletions || dropsEval) {
            const disabled = [
              dropsTrainOnCompletions && "assistant-completions-only",
              dropsEval && "evaluation (needs a separate eval split)",
            ].filter(Boolean);
            toast.info(
              `Streaming enabled. Disabled incompatible options: ${disabled.join(", ")}.`,
            );
          }
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
          set({
            uploadedFile,
            trainOnCompletionsManuallySet: false,
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
        setContextLength: (contextLength) =>
          set({ contextLength, contextLengthManuallySet: true }),
        setVisionImageSize: (visionImageSize) => set({ visionImageSize }),
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
        setMaxSteps: (maxSteps) => {
          const state = get();
          // streamingCompatiblePatch already turns streaming off when maxSteps<=0,
          // so no separate datasetStreaming reset is needed here.
          const streamingPatch = streamingCompatiblePatch({ ...state, maxSteps });
          set({
            maxSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setSaveSteps: (saveSteps) => set({ saveSteps }),
        setEvalSteps: (evalSteps) => {
          const state = get();
          const streamingPatch = streamingCompatiblePatch({ ...state, evalSteps });
          set({
            evalSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setPacking: (packing) => set({ packing }),
        setTrainOnCompletions: (trainOnCompletions) => {
          set({
            trainOnCompletions,
            trainOnCompletionsManuallySet: true,
            ...(trainOnCompletions ? { datasetStreaming: false } : {}),
          });
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
        setS3Config: (s3Config) => set({ s3Config }),
        canProceed: () => canProceedForStep(get()),
        reset: () => {
          _yamlLearningRate = undefined;
          clearCptDatasetFormatTracking();
          set({ ...initialState, hfToken: getHfToken() });
        },
        resetToModelDefaults: () => {
          const { selectedModel } = get();
          if (!selectedModel) return;
          set({
            contextLengthManuallySet: false,
            trainOnCompletionsManuallySet: false,
            learningRateManuallySet: false,
            trainingMethodManuallySet: false,
            modelDefaultsAppliedFor: null,
            visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
          });
          loadAndApplyModelDefaults(selectedModel);
        },
        applyConfigPatch: (config: BackendModelConfig) => {
          const patch = mapBackendModelConfigToTrainingPatch(config);
          // Only clear the manual-edit flag when the config provides a LR,
          // so unrelated config patches don't silently disarm the guard.
          set({
            ...patch,
            ...(patch.contextLength !== undefined
              ? { contextLengthManuallySet: true }
              : {}),
            ...(patch.trainOnCompletions !== undefined
              ? { trainOnCompletionsManuallySet: true }
              : {}),
            ...(patch.learningRate !== undefined
              ? { learningRateManuallySet: true }
              : {}),
          });
        },
      };
    },
    {
      name: "unsloth_training_config_v1",
      version: 14,
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
        if (version < 11) {
          // Standalone bump: users already on main's v10 (CPT) skipped the
          // streaming backfill when it was nested under v<10, so give it its
          // own version guard.
          s.datasetStreaming ??= false;
        }
        if (version < 12) {
          // hfToken moved to the shared hf-token-store; seed it once so an
          // existing Studio-only token isn't lost.
          const legacyToken = typeof s.hfToken === "string" ? s.hfToken.trim() : "";
          if (legacyToken && !getHfToken()) {
            useHfTokenStore.getState().setToken(legacyToken);
          }
          delete s.hfToken;
          s.contextLengthManuallySet = false;
        }
        if (version < 13) {
          s.trainOnCompletionsManuallySet = false;
          s.learningRateManuallySet = false;
          s.trainingMethodManuallySet = false;
        }
        if (version < 14) {
          s.trainOnCompletionsManuallySet = false;
        }
        return s as unknown as TrainingConfigStore;
      },
      partialize: partializePersistedState,
      onRehydrateStorage: () => (state) => {
        // datasetStreaming is persisted, but constraint-coupled fields like
        // trainOnCompletions / maxSteps / evalSteps are NON_PERSISTED and
        // rehydrate to defaults. That can resurrect an invalid combo (e.g.
        // streaming=true with a default trainOnCompletions) that the backend
        // rejects with 422. Reconcile immediately on load instead of relying
        // on a post-mount effect.
        if (!state) return;
        const patch = streamingCompatiblePatch(state);
        if (Object.keys(patch).length > 0) {
          // Sync localStorage hydration runs inside create(), before
          // useTrainingConfigStore is assigned (TDZ). Defer to a microtask so the
          // store exists when we reconcile the persisted streaming combo.
          queueMicrotask(() => useTrainingConfigStore.setState(patch));
        }
      },
    },
  ),
);

const unsubscribeHfTokenMirror = mirrorHfTokenInto(useTrainingConfigStore);
if (import.meta.hot) {
  import.meta.hot.dispose(unsubscribeHfTokenMirror);
}
