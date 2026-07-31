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
import { getHfToken, useHfTokenStore } from "@/features/hub";
import { getLocale, translate } from "@/i18n";
import { isAdapterMethod } from "@/types/training";
import type { DatasetFormat } from "@/types/training";
import type { ModelType, StepNumber, TrainingMethod } from "@/types/training";
import { toast } from "sonner";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { checkDatasetFormat } from "../api/datasets-api";
import { checkVisionModel, getModelConfig } from "../api/models-api";
import type { BackendModelConfig } from "../api/models-api";
import { cacheReferenceMatchesSelection } from "../lib/cache-reference";
import { isMissingLocalDatasetCacheError } from "../lib/local-cache-errors";
import { mapBackendModelConfigToTrainingPatch } from "../lib/model-defaults";
import { inferTrainingModelTypeFromFlags } from "../lib/model-type-inference";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import { validateS3Source } from "../lib/validation";
import type {
  BrowseDatasetSelection,
  DatasetCacheReferenceOptions,
  TrainingConfigState,
  TrainingConfigStore,
  TrainingModelSelectionOptions,
} from "../types/config";

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

function createHfBrowseDatasetSelection(
  dataset: string | null,
  options?: DatasetCacheReferenceOptions,
): Extract<BrowseDatasetSelection, { source: "huggingface" }> {
  return {
    source: "huggingface",
    dataset,
    knownCached: dataset ? (options?.knownCached ?? false) : false,
    localPath: dataset ? (options?.localPath ?? null) : null,
  };
}

function createUploadBrowseDatasetSelection(
  uploadedFile: string | null,
): Extract<BrowseDatasetSelection, { source: "upload" }> {
  return {
    source: "upload",
    uploadedFile,
  };
}

const initialState: TrainingConfigState = {
  currentStep: MIN_STEP,
  modelType: null,
  selectedModel: null,
  modelKnownCached: false,
  modelLocalPath: null,
  modelFormat: null,
  projectName: "",
  trainingMethod: "qlora",
  datasetSource: "huggingface",
  browseDatasetSelection: createHfBrowseDatasetSelection(null),
  datasetFormat: "auto",
  dataset: null,
  datasetKnownCached: false,
  datasetLocalPath: null,
  datasetSubset: null,
  datasetSplit: null,
  datasetEvalSplit: null,
  datasetStreaming: false,
  datasetManualMapping: emptyManualMapping(),
  datasetSystemPrompt: "",
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
  datasetCheckFailed: false,
  maxPositionEmbeddings: null,
  ...DEFAULT_HYPERPARAMS,
};

// AbortController for in-flight dataset multimodal checks.
let _datasetCheckController: AbortController | null = null;

// AbortController for in-flight model default loads.
let _modelConfigController: AbortController | null = null;

// Has the user manually toggled trainOnCompletions since the last auto-set
// (model load or dataset change)?
let _trainOnCompletionsManuallySet = false;

// Has the user manually edited the LR since the last model load? When false,
// switching method auto-sets LR to 2e-4 (LoRA/QLoRA) or 2e-5 (full fine-tune).
let _learningRateManuallySet = false;
let _trainingMethodEditGeneration = 0;

// Stash the YAML learning rate so setTrainingMethod can restore it when
// switching back from full to adapter.
let _yamlLearningRate: number | undefined;

// Track whether entering CPT auto-forced datasetFormat="raw" so that
// leaving CPT can restore the prior user-visible format.
let _datasetFormatBeforeCpt: DatasetFormat | null = null;
let _datasetFormatAutoForcedByCpt = false;

// Model capability flags persist so modality-specific UI paints correctly on
// reload; the model-config fetch still re-derives them.
const NON_PERSISTED_STATE_KEYS: ReadonlySet<keyof TrainingConfigState> =
  new Set([
    "isCheckingVision",
    "isLoadingModelDefaults",
    "modelDefaultsError",
    "modelDefaultsAppliedFor",
    "isCheckingDataset",
    "isDatasetImage",
    "isDatasetAudio",
    "datasetCheckFailed",
    "trainOnCompletions",
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

function formatStreamingDisabledOptions(
  trainOnCompletions: boolean,
  evaluation: boolean,
): string {
  const options: string[] = [];
  if (trainOnCompletions) {
    options.push(
      translate("studio.dataset.streaming.options.trainOnCompletions"),
    );
  }
  if (evaluation) {
    options.push(translate("studio.dataset.streaming.options.evaluation"));
  }
  if (typeof Intl.ListFormat !== "function") {
    return options.join(", ");
  }
  return new Intl.ListFormat(getLocale(), {
    style: "long",
    type: "conjunction",
  }).format(options);
}

// streamingCompatiblePatch can silently flip streaming-coupled fields. Surface a
// toast when it does, so the indirect setters (split / eval-split / max-steps /
// eval-steps) match setDatasetStreaming's "tell the user what changed" behavior.
function notifyStreamingCompat(patch: Partial<TrainingConfigState>): void {
  if (patch.datasetStreaming === false) {
    toast.info(
      translate("studio.dataset.streaming.notifications.turnedOffMaxSteps"),
    );
    return;
  }
  const options = formatStreamingDisabledOptions(
    patch.trainOnCompletions === false,
    patch.evalSteps === 0,
  );
  if (options) {
    toast.info(
      translate("studio.dataset.streaming.notifications.adjusted", {
        options,
      }),
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

function recordCptDatasetFormatOverride(
  currentDatasetFormat: DatasetFormat,
): void {
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
  return nowAdapter ? (_yamlLearningRate ?? LR_DEFAULT_LORA) : LR_DEFAULT_FULL;
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

  const learningRate = resolveTrainingMethodLearningRate(
    prevMethod,
    nextMethod,
  );
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
        const trainingMethodEditGeneration = _trainingMethodEditGeneration;
        const requestState = get();
        const requestedKnownCached =
          requestState.selectedModel === modelName &&
          requestState.modelKnownCached;
        const requestedLocalPath =
          requestState.selectedModel === modelName
            ? requestState.modelLocalPath
            : null;
        const preferLocalCache =
          requestedKnownCached && Boolean(requestedLocalPath?.trim());
        const requestMatchesSelection = () => {
          const state = get();
          return (
            state.selectedModel === modelName &&
            state.modelKnownCached === requestedKnownCached &&
            state.modelLocalPath === requestedLocalPath
          );
        };
        _modelConfigController = controller;
        set({
          isLoadingModelDefaults: true,
          isCheckingVision: true,
          modelDefaultsError: null,
        });

        void getModelConfig(
          modelName,
          controller.signal,
          getHfToken() || undefined,
          {
            preferLocalCache,
            localPath: preferLocalCache ? requestedLocalPath : null,
          },
        )
          .then((modelDetails) => {
            if (controller.signal.aborted) return;
            if (!requestMatchesSelection()) return;

            _trainOnCompletionsManuallySet = false;
            _learningRateManuallySet = false;
            _yamlLearningRate = undefined;

            if (modelDetails.is_lora) {
              set({
                modelType: null,
                modelFormat: "adapter",
                isVisionModel: false,
                isEmbeddingModel: false,
                isAudioModel: false,
                isLoadingModelDefaults: false,
                isCheckingVision: false,
                modelDefaultsError: null,
                modelDefaultsAppliedFor: modelName,
                maxPositionEmbeddings: null,
              });
              toast.error(translate("studio.modelPicker.cantUseModel"), {
                description: translate("studio.modelPicker.reasonAdapter"),
              });
              return;
            }

            const patch = mapBackendModelConfigToTrainingPatch(
              modelDetails.config,
            );

            // Treat a model-config LR as authoritative so async auto-select
            // won't overwrite it.
            const modelConfigHasLR = patch.learningRate !== undefined;
            _yamlLearningRate = patch.learningRate;

            // YAML LRs are tuned for adapters (LoRA/QLoRA); on full fine-tune,
            // use the full-finetune default instead of the YAML adapter LR.
            if (modelConfigHasLR && !isAdapterMethod(get().trainingMethod)) {
              patch.learningRate = LR_DEFAULT_FULL;
            }

            // Vision model + known image dataset: force trainOnCompletions off.
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

            // Use backend model_type when available, else infer from flags.
            const isEmbedding = !!modelDetails.is_embedding;
            const inferredModelType: ModelType =
              modelDetails.model_type ??
              (isEmbedding
                ? "embeddings"
                : modelDetails.is_vision
                  ? "vision"
                  : modelDetails.is_audio
                    ? "audio"
                    : "text");

            // Auto-select LoRA vs QLoRA by model size vs GPU memory (see
            // autoSelectTrainingMethod). Skip if the user chose CPT.
            const modelSizeBytes = modelDetails.model_size_bytes;
            if (
              modelSizeBytes &&
              modelSizeBytes > 0 &&
              get().trainingMethod !== "cpt"
            ) {
              void autoSelectTrainingMethod(
                modelSizeBytes,
                patch.contextLength ?? get().contextLength,
              ).then((method) => {
                if (!requestMatchesSelection()) return;
                if (get().trainingMethod === "cpt") return;
                if (
                  _trainingMethodEditGeneration !== trainingMethodEditGeneration
                ) {
                  return;
                }
                if (method) {
                  const lrPatch =
                    !_learningRateManuallySet && !modelConfigHasLR
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
            // are tuned for standard LoRA and would clobber CPT settings.
            const cptOverrides =
              get().trainingMethod === "cpt" ? getCptModelDefaultsPatch() : {};

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
              maxPositionEmbeddings:
                modelDetails.max_position_embeddings ?? null,
            });
          })
          .catch((error) => {
            if (controller.signal.aborted) return;
            if (!requestMatchesSelection()) return;

            set({
              isLoadingModelDefaults: false,
              modelDefaultsError:
                error instanceof Error
                  ? error.message
                  : "Failed to load model defaults",
              // Defaults load failed; reset so no prior model's value lingers.
              visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
            });

            if (preferLocalCache) {
              set({ isCheckingVision: false });
              return;
            }

            // Fallback vision check; pass the token so a gated/private VLM classifies right.
            void checkVisionModel(modelName, getHfToken() || undefined)
              .then((isVision) => {
                if (!requestMatchesSelection()) return;
                const state = get();
                set({
                  modelType: inferTrainingModelTypeFromFlags({
                    isEmbedding: state.isEmbeddingModel,
                    isAudio: state.isAudioModel,
                    isVision,
                  }),
                  isVisionModel: isVision,
                  isCheckingVision: false,
                });
              })
              .catch(() => {
                if (!requestMatchesSelection()) return;
                set({ isCheckingVision: false });
              });
          });
      };

      const runDatasetCheck = (
        datasetName: string,
        split: string,
        options?: { preferLocalCache?: boolean },
      ) => {
        _datasetCheckController?.abort();
        const controller = new AbortController();
        _datasetCheckController = controller;
        set({ isCheckingDataset: true });

        const state = get();
        const isHfSelection =
          state.datasetSource === "huggingface" &&
          state.dataset === datasetName;
        const requestedPreferLocalCache =
          options?.preferLocalCache ??
          (isHfSelection && state.datasetKnownCached);
        const preferLocalCache =
          requestedPreferLocalCache && !state.datasetStreaming;
        checkDatasetFormat({
          datasetName,
          hfToken: getHfToken() || null,
          subset: state.datasetSubset,
          split,
          isVlm: state.isVisionModel,
          preferLocalCache,
          localPath:
            isHfSelection && preferLocalCache ? state.datasetLocalPath : null,
        })
          .then((res) => {
            if (controller.signal.aborted) return;
            const isImage = !!res.is_image;
            const isAudio = !!res.is_audio;
            const updates: Record<string, unknown> = {
              isDatasetImage: isImage,
              isDatasetAudio: isAudio,
              isCheckingDataset: false,
              datasetCheckFailed: false,
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
          .catch((error) => {
            if (controller.signal.aborted) return;
            if (preferLocalCache && isMissingLocalDatasetCacheError(error)) {
              const current = get();
              if (
                current.datasetSource === "huggingface" &&
                current.dataset === datasetName
              ) {
                set({
                  datasetKnownCached: false,
                  datasetLocalPath: null,
                  browseDatasetSelection:
                    createHfBrowseDatasetSelection(datasetName),
                });
                runDatasetCheck(datasetName, split, {
                  preferLocalCache: false,
                });
                return;
              }
            }
            set({
              isDatasetImage: null,
              isCheckingDataset: false,
              datasetCheckFailed: true,
            });
          });
      };

      const recheckSelectedDatasetForStreamingMode = (
        datasetStreaming: boolean,
      ) => {
        const state = get();
        if (state.datasetSource !== "huggingface" || !state.dataset) {
          return;
        }
        runDatasetCheck(state.dataset, state.datasetSplit || "train", {
          preferLocalCache: !datasetStreaming && state.datasetKnownCached,
        });
      };

      const resetDatasetState = (): Partial<TrainingConfigStore> => ({
        datasetSubset: null,
        datasetSplit: null,
        datasetEvalSplit: null,
        datasetManualMapping: emptyManualMapping(),
        datasetSystemPrompt: "",
        datasetLabelMapping: {},
        datasetAdvisorNotification: null,
        datasetSliceStart: null,
        datasetSliceEnd: null,
        uploadedEvalFile: null,
        isDatasetImage: null,
        isDatasetAudio: false,
        isCheckingDataset: false,
        datasetCheckFailed: false,
      });

      const selectHfDatasetInternal = (
        dataset: string | null,
        options?: DatasetCacheReferenceOptions,
      ) => {
        const datasetId = dataset?.trim() || null;
        _datasetCheckController?.abort();
        _datasetCheckController = null;
        _trainOnCompletionsManuallySet = false;
        const browseDatasetSelection = createHfBrowseDatasetSelection(
          datasetId,
          options,
        );
        set({
          datasetSource: "huggingface",
          browseDatasetSelection,
          dataset: datasetId,
          uploadedFile: null,
          ...resetDatasetState(),
          datasetKnownCached: browseDatasetSelection.knownCached,
          datasetLocalPath: browseDatasetSelection.localPath,
        });
        if (datasetId) {
          runDatasetCheck(datasetId, "train");
        }
      };

      const selectLocalDatasetInternal = (uploadedFile: string | null) => {
        _datasetCheckController?.abort();
        _datasetCheckController = null;
        _trainOnCompletionsManuallySet = false;
        set({
          datasetSource: "upload",
          browseDatasetSelection:
            createUploadBrowseDatasetSelection(uploadedFile),
          dataset: null,
          uploadedFile,
          ...resetDatasetState(),
          datasetKnownCached: false,
          datasetLocalPath: null,
        });
        if (uploadedFile) {
          runDatasetCheck(uploadedFile, "train");
        }
      };

      const selectS3SourceInternal = () => {
        _datasetCheckController?.abort();
        _datasetCheckController = null;
        _trainOnCompletionsManuallySet = false;
        const state = get();
        const browseDatasetSelection =
          state.datasetSource === "s3"
            ? state.browseDatasetSelection
            : state.datasetSource === "upload"
              ? createUploadBrowseDatasetSelection(state.uploadedFile)
              : createHfBrowseDatasetSelection(state.dataset, {
                  knownCached: state.datasetKnownCached,
                  localPath: state.datasetLocalPath,
                });
        set({
          datasetSource: "s3",
          browseDatasetSelection,
          dataset: null,
          uploadedFile: null,
          ...resetDatasetState(),
          datasetKnownCached: false,
          datasetLocalPath: null,
        });
      };

      const restoreBrowseDatasetSourceInternal = () => {
        const selection = get().browseDatasetSelection;
        if (selection.source === "upload") {
          selectLocalDatasetInternal(selection.uploadedFile);
          return;
        }
        selectHfDatasetInternal(selection.dataset, {
          knownCached: selection.knownCached,
          localPath: selection.localPath,
        });
      };

      const selectModelInternal = (
        selectedModel: string | null,
        modelType: ModelType | null,
        options?: TrainingModelSelectionOptions,
      ) => {
        const currentState = get();
        const effectiveModelType = modelType ?? currentState.modelType;
        const previousModel = currentState.selectedModel;
        const nextKnownCached = selectedModel
          ? (options?.knownCached ?? false)
          : false;
        const nextLocalPath = selectedModel
          ? (options?.localPath ?? null)
          : null;
        const selectionChanged =
          selectedModel !== previousModel ||
          currentState.modelKnownCached !== nextKnownCached ||
          currentState.modelLocalPath !== nextLocalPath;
        const previousAdapterFormat =
          selectedModel === previousModel &&
          currentState.modelFormat === "adapter"
            ? currentState.modelFormat
            : null;
        const patch: {
          selectedModel: string | null;
          modelDefaultsError: null;
          modelKnownCached: boolean;
          modelLocalPath: string | null;
          modelFormat: TrainingConfigState["modelFormat"];
          modelType?: ModelType;
          visionImageSize?: number | null;
          trustRemoteCode?: boolean;
          approvedRemoteCodeFingerprint?: string | null;
          isVisionModel?: boolean;
          isAudioModel?: boolean;
          isEmbeddingModel?: boolean;
          modelDefaultsAppliedFor?: string | null;
        } = {
          selectedModel,
          modelDefaultsError: null,
          modelKnownCached: nextKnownCached,
          modelLocalPath: nextLocalPath,
          modelFormat: selectedModel
            ? (previousAdapterFormat ?? options?.modelFormat ?? null)
            : null,
        };
        if (effectiveModelType) {
          patch.modelType = effectiveModelType;
        }
        if (selectionChanged) {
          patch.visionImageSize = DEFAULT_HYPERPARAMS.visionImageSize;
          patch.trustRemoteCode = false;
          patch.approvedRemoteCodeFingerprint = null;
          patch.isVisionModel =
            options?.isVision ?? effectiveModelType === "vision";
          patch.isAudioModel =
            options?.isAudio ?? effectiveModelType === "audio";
          patch.isEmbeddingModel =
            options?.isEmbedding ?? effectiveModelType === "embeddings";
          patch.modelDefaultsAppliedFor = null;
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
          });
          return;
        }

        const shouldLoadDefaults =
          selectionChanged || get().modelDefaultsAppliedFor !== selectedModel;
        if (shouldLoadDefaults) {
          void loadAndApplyModelDefaults(selectedModel);
        }
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
            modelKnownCached: false,
            modelLocalPath: null,
            modelFormat: null,
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
          selectModelInternal(selectedModel, null);
        },
        selectTrainingModel: (selectedModel, modelType, options) => {
          selectModelInternal(selectedModel, modelType, options);
        },
        setSelectedModelCacheReference: (model, options) => {
          const state = get();
          if (state.selectedModel !== model) return;
          const cacheReferenceChanged =
            !state.modelKnownCached ||
            state.modelLocalPath !== options.localPath;
          set({
            modelKnownCached: true,
            modelLocalPath: options.localPath,
            modelFormat: options.modelFormat,
            ...(cacheReferenceChanged ? { modelDefaultsAppliedFor: null } : {}),
          });
          if (cacheReferenceChanged) {
            void loadAndApplyModelDefaults(model);
          }
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
          set({
            modelKnownCached: false,
            modelLocalPath: null,
            modelFormat: null,
            modelDefaultsAppliedFor: null,
          });
          void loadAndApplyModelDefaults(model);
        },
        clearSelectedDatasetCacheReference: (dataset, localPath) => {
          const state = get();
          if (state.datasetSource !== "huggingface") return;
          if (
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
          set({
            datasetKnownCached: false,
            datasetLocalPath: null,
            browseDatasetSelection: createHfBrowseDatasetSelection(dataset),
          });
          runDatasetCheck(dataset, get().datasetSplit || "train", {
            preferLocalCache: false,
          });
        },
        setSelectedDatasetCacheReference: (dataset, localPath) => {
          const state = get();
          if (
            state.datasetSource !== "huggingface" ||
            state.dataset !== dataset
          ) {
            return;
          }
          const cacheReferenceChanged =
            !state.datasetKnownCached || state.datasetLocalPath !== localPath;
          set({
            datasetKnownCached: true,
            datasetLocalPath: localPath,
            browseDatasetSelection: createHfBrowseDatasetSelection(dataset, {
              knownCached: true,
              localPath,
            }),
            ...(cacheReferenceChanged && !state.datasetStreaming
              ? {
                  isDatasetImage: null,
                  isDatasetAudio: false,
                  datasetCheckFailed: false,
                }
              : {}),
          });
          if (cacheReferenceChanged && !state.datasetStreaming) {
            recheckSelectedDatasetForStreamingMode(false);
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
          _trainingMethodEditGeneration += 1;
          const state = get();
          set(
            buildTrainingMethodPatch(
              state.trainingMethod,
              trainingMethod,
              state.datasetFormat,
            ),
          );
        },
        setDatasetSource: (datasetSource) => {
          const state = get();
          if (datasetSource === state.datasetSource) return;
          if (datasetSource === "s3") {
            selectS3SourceInternal();
            return;
          }
          if (
            state.datasetSource === "s3" &&
            state.browseDatasetSelection.source === datasetSource
          ) {
            restoreBrowseDatasetSourceInternal();
            return;
          }
          if (datasetSource === "upload") {
            selectLocalDatasetInternal(null);
            return;
          }
          selectHfDatasetInternal(null);
        },
        selectHfDataset: selectHfDatasetInternal,
        selectLocalDataset: selectLocalDatasetInternal,
        selectS3Source: selectS3SourceInternal,
        restoreBrowseDatasetSource: () => {
          if (get().datasetSource !== "s3") return;
          restoreBrowseDatasetSourceInternal();
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
              trainOnCompletions: isRawTextDatasetFormat(datasetFormat)
                ? false
                : state.trainOnCompletions,
            };
          }),
        setDataset: (dataset) => {
          const datasetId = dataset?.trim() || null;
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set((state) => ({
            dataset: datasetId,
            datasetKnownCached: false,
            datasetLocalPath: null,
            browseDatasetSelection:
              state.datasetSource === "huggingface"
                ? createHfBrowseDatasetSelection(datasetId)
                : state.browseDatasetSelection,
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
          }));
          if (datasetId) {
            runDatasetCheck(datasetId, "train");
          }
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
            const changed = get().datasetStreaming;
            set({
              datasetStreaming: false,
              ...(changed
                ? {
                    isDatasetImage: null,
                    isDatasetAudio: false,
                    datasetCheckFailed: false,
                  }
                : {}),
            });
            if (changed) {
              recheckSelectedDatasetForStreamingMode(false);
            }
            return;
          }

          const state = get();
          if (state.maxSteps <= 0) {
            set({ datasetStreaming: false });
            toast.warning(
              translate("studio.dataset.streaming.notifications.needsMaxSteps"),
            );
            return;
          }

          const dropsTrainOnCompletions = state.trainOnCompletions;
          const dropsEval = !hasSeparateStreamingEvalSplit(state);

          set({
            datasetStreaming: true,
            trainOnCompletions: false,
            evalSteps: dropsEval ? 0 : state.evalSteps,
            isDatasetImage: null,
            isDatasetAudio: false,
            datasetCheckFailed: false,
          });
          recheckSelectedDatasetForStreamingMode(true);

          if (dropsTrainOnCompletions || dropsEval) {
            const options = formatStreamingDisabledOptions(
              dropsTrainOnCompletions,
              dropsEval,
            );
            toast.info(
              translate(
                "studio.dataset.streaming.notifications.enabledAdjusted",
                { options },
              ),
            );
          }
        },
        setDatasetManualMapping: (datasetManualMapping) =>
          set({ datasetManualMapping }),
        setDatasetAdvisorFields: (fields) =>
          set({
            datasetSystemPrompt:
              fields.systemPrompt ?? get().datasetSystemPrompt,
            datasetLabelMapping:
              fields.labelMapping ?? get().datasetLabelMapping,
            datasetAdvisorNotification:
              fields.notification !== undefined
                ? fields.notification
                : get().datasetAdvisorNotification,
          }),
        setDatasetSliceStart: (datasetSliceStart) => set({ datasetSliceStart }),
        setDatasetSliceEnd: (datasetSliceEnd) => set({ datasetSliceEnd }),
        setUploadedFile: (uploadedFile) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          set((state) => ({
            uploadedFile,
            datasetKnownCached: false,
            datasetLocalPath: null,
            browseDatasetSelection:
              state.datasetSource === "upload"
                ? createUploadBrowseDatasetSelection(uploadedFile)
                : state.browseDatasetSelection,
            datasetCheckFailed: false,
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
          }));
        },
        setUploadedEvalFile: (uploadedEvalFile) =>
          set({
            uploadedEvalFile,
            evalSteps: uploadedEvalFile ? 0.1 : 0,
          }),
        setEpochs: (epochs) => set({ epochs }),
        setContextLength: (contextLength) => set({ contextLength }),
        setVisionImageSize: (visionImageSize) => set({ visionImageSize }),
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
        setMaxSteps: (maxSteps) => {
          const state = get();
          // streamingCompatiblePatch already turns streaming off when maxSteps<=0,
          // so no separate datasetStreaming reset is needed here.
          const streamingPatch = streamingCompatiblePatch({
            ...state,
            maxSteps,
          });
          set({
            maxSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setSaveSteps: (saveSteps) => set({ saveSteps }),
        setEvalSteps: (evalSteps) => {
          const state = get();
          const streamingPatch = streamingCompatiblePatch({
            ...state,
            evalSteps,
          });
          set({
            evalSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setPacking: (packing) => set({ packing }),
        setTrainOnCompletions: (trainOnCompletions) => {
          _trainOnCompletionsManuallySet = true;
          set({
            trainOnCompletions,
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
          _trainOnCompletionsManuallySet = false;
          _learningRateManuallySet = false;
          _yamlLearningRate = undefined;
          clearCptDatasetFormatTracking();
          set(initialState);
        },
        resetToModelDefaults: () => {
          const { selectedModel } = get();
          if (!selectedModel) return;
          set({
            modelDefaultsAppliedFor: null,
            visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
          });
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
      version: 16,
      migrate: (persisted, version) => {
        const s = persisted as Record<string, unknown>;
        if (version < 2 && s.datasetSubset == null && s.datasetConfig != null) {
          s.datasetSubset = s.datasetConfig;
        }
        s.datasetConfig = undefined;
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
          const legacyToken =
            typeof s.hfToken === "string" ? s.hfToken.trim() : "";
          if (legacyToken && !getHfToken()) {
            useHfTokenStore.getState().setToken(legacyToken);
          }
          delete s.hfToken;
        }
        if (version < 13) {
          s.modelKnownCached ??= false;
          s.modelLocalPath ??= null;
          s.modelFormat ??= null;
          s.datasetKnownCached ??= false;
          s.datasetLocalPath ??= null;
        }
        if (version < 14) {
          const dataset = typeof s.dataset === "string" ? s.dataset : null;
          const uploadedFile =
            typeof s.uploadedFile === "string" ? s.uploadedFile : null;
          s.browseDatasetSelection =
            s.datasetSource === "upload"
              ? createUploadBrowseDatasetSelection(uploadedFile)
              : createHfBrowseDatasetSelection(dataset, {
                  knownCached: s.datasetKnownCached === true,
                  localPath:
                    typeof s.datasetLocalPath === "string"
                      ? s.datasetLocalPath
                      : null,
                });
        }
        if (version < 15) {
          s.isEmbeddingModel = s.modelType === "embeddings";
        }
        if (version < 16) {
          delete s.datasetUserTemplate;
          delete s.datasetAssistantTemplate;
        }
        return s as unknown as TrainingConfigStore;
      },
      partialize: partializePersistedState,
      onRehydrateStorage: () => (state) => {
        // datasetStreaming, maxSteps, and evalSteps persist, while
        // trainOnCompletions rehydrates to its default. That can resurrect an
        // invalid combination that the backend rejects with 422. Reconcile
        // immediately on load instead of relying on a post-mount effect.
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
