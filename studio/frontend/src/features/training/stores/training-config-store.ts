// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_HYPERPARAMS,
  LR_DEFAULT_FULL,
  LR_DEFAULT_LORA,
} from "@/config/training";
import { getHfToken } from "@/features/hub";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { isAdapterMethod } from "@/types/training";
import type { ModelType } from "@/types/training";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { DatasetFormatError, checkDatasetFormat } from "../api/datasets-api";
import { checkVisionModel, getModelConfig } from "../api/models-api";
import type { BackendModelConfig } from "../api/models-api";
import { cacheReferenceMatchesSelection } from "../lib/cache-reference";
import {
  createDatasetCacheUsabilityIdentity,
  datasetCacheUsabilityIdentitiesEqual,
  trainingDatasetCacheRejections,
} from "../lib/dataset-cache-rejection";
import {
  claimDatasetCacheRecheck,
  datasetCacheRecheckKey,
} from "../lib/dataset-recheck-budget";
import { resolveDeletedLocalDatasetSelection } from "../lib/dataset-selection";
import { requiresExplicitCachedDatasetSplit } from "../lib/dataset-split-policy";
import { isMissingLocalDatasetCacheError } from "../lib/local-cache-errors";
import { mapBackendModelConfigToTrainingPatch } from "../lib/model-defaults";
import { trainingConfigPatchTouchesModelDefaults } from "../lib/model-defaults-edit-policy";
import {
  inferTrainingModelTypeFromFlags,
  resolveTrainingModelType,
} from "../lib/model-type-capabilities";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import type {
  DatasetCacheReferenceOptions,
  TrainingConfigState,
  TrainingConfigStore,
  TrainingModelSelectionOptions,
} from "../types/config";
import {
  TRAINING_CONFIG_PERSISTENCE_NAME,
  TRAINING_CONFIG_PERSISTENCE_VERSION,
  mergeTrainingConfig,
  migrateTrainingConfig,
  partializeTrainingConfig,
} from "./training-config-persistence";
import {
  canProceedForTrainingStep,
  clampTrainingStep,
  createHfBrowseDatasetSelection,
  createUploadBrowseDatasetSelection,
  datasetSelectionStreamingPatch,
  datasetSourceInvariantPatch,
  emptyManualMapping,
  formatStreamingDisabledOptions,
  hasSeparateStreamingEvalSplit,
  initialTrainingConfigState,
  resolveDeferredTrainOnCompletionsDefault,
  streamingCompatiblePatch,
} from "./training-config-policy";
import { selectTrainingMethodForHardware } from "./training-method-hardware-policy";
import {
  buildTrainingMethodPatch,
  getCptModelDefaultsPatch,
} from "./training-method-transition";

export { hasSeparateStreamingEvalSplit } from "./training-config-policy";

// AbortController for in-flight dataset multimodal checks.
let _datasetCheckController: AbortController | null = null;


// AbortController for in-flight model default loads.
let _modelConfigController: AbortController | null = null;

// Has the user manually toggled trainOnCompletions since the last auto-set?
let _trainOnCompletionsManuallySet = false;

let _trainingMethodEditGeneration = 0;
let _modelDefaultsEditGeneration = 0;
let _modelDefaultsEditBaseline: {
  modelName: string;
  editGeneration: number;
} | null = null;

function canReapplyModelDefaults(modelName: string): boolean {
  return (
    _modelDefaultsEditBaseline?.modelName === modelName &&
    _modelDefaultsEditBaseline.editGeneration === _modelDefaultsEditGeneration
  );
}

// streamingCompatiblePatch can silently flip streaming-coupled fields, so toast when it does,
// matching setDatasetStreaming's "tell the user what changed" behavior.
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

export const useTrainingConfigStore = create<TrainingConfigStore>()(
  persist(
    (set, get) => {
      const setUserEdit = (
        update:
          | Partial<TrainingConfigState>
          | ((state: TrainingConfigStore) => Partial<TrainingConfigState>),
      ) => {
        set((state) => {
          const patch = typeof update === "function" ? update(state) : update;
          const invariantPatch = datasetSourceInvariantPatch({
            datasetSource: patch.datasetSource ?? state.datasetSource,
            datasetStreaming:
              patch.datasetStreaming ?? state.datasetStreaming,
          });
          const normalizedPatch = { ...patch, ...invariantPatch };
          if (trainingConfigPatchTouchesModelDefaults(normalizedPatch)) {
            _modelDefaultsEditGeneration += 1;
          }
          return {
            ...normalizedPatch,
            userEditRevision: state.userEditRevision + 1,
          };
        });
      };

      const loadAndApplyModelDefaults = (
        modelName: string,
        options?: { applyTrainingDefaults?: boolean },
      ) => {
        const applyTrainingDefaults = options?.applyTrainingDefaults ?? true;
        _modelConfigController?.abort();
        const controller = new AbortController();
        const trainingMethodEditGeneration = _trainingMethodEditGeneration;
        const requestState = get();
        const requestedModelDefaultsEditGeneration =
          _modelDefaultsEditGeneration;
        if (applyTrainingDefaults) {
          _modelDefaultsEditBaseline = {
            modelName,
            editGeneration: requestedModelDefaultsEditGeneration,
          };
        }
        const requestedKnownCached =
          requestState.selectedModel === modelName &&
          requestState.modelKnownCached;
        const requestedLocalPath =
          requestState.selectedModel === modelName
            ? requestState.modelLocalPath
            : null;
        const canApplyTrainingDefaults = () =>
          applyTrainingDefaults &&
          _modelDefaultsEditGeneration === requestedModelDefaultsEditGeneration;
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

            const shouldApplyTrainingDefaults = canApplyTrainingDefaults();
            if (shouldApplyTrainingDefaults) {
              _trainOnCompletionsManuallySet = false;
            }

            if (modelDetails.is_lora) {
              set({
                ...(shouldApplyTrainingDefaults
                  ? {
                      trainingMethodProvenance: {
                        ...get().trainingMethodProvenance,
                        learningRateManuallySet: false,
                        modelAdapterLearningRate: null,
                      },
                    }
                  : {}),
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

            const modelDefaultsPatch = mapBackendModelConfigToTrainingPatch(
              modelDetails.config,
            );
            const patch = shouldApplyTrainingDefaults ? modelDefaultsPatch : {};

            // Treat a model-config LR as authoritative so async auto-select won't overwrite it.
            const modelConfigHasLR =
              modelDefaultsPatch.learningRate !== undefined;
            const modelAdapterLearningRate =
              modelDefaultsPatch.learningRate ?? null;

            // YAML LRs are tuned for adapters (LoRA/QLoRA); full fine-tune uses its own default.
            if (modelConfigHasLR && !isAdapterMethod(get().trainingMethod)) {
              modelDefaultsPatch.learningRate = LR_DEFAULT_FULL;
            }

            // Vision model + known image dataset: force trainOnCompletions off.
            if (modelDetails.is_vision && get().isDatasetImage === true) {
              modelDefaultsPatch.trainOnCompletions = false;
            }

            const isAudio = !!modelDetails.is_audio;
            // Pure audio model -> always uncheck trainOnCompletions.
            if (isAudio && !modelDetails.is_vision) {
              modelDefaultsPatch.trainOnCompletions = false;
            }
            // Audio-capable vision model (e.g. gemma3n) + audio dataset -> uncheck.
            if (isAudio && modelDetails.is_vision && get().isDatasetAudio) {
              modelDefaultsPatch.trainOnCompletions = false;
            }

            const isEmbedding = !!modelDetails.is_embedding;
            const inferredModelType = resolveTrainingModelType({
              modelType: modelDetails.model_type,
              isEmbedding,
              isVision: modelDetails.is_vision,
              isAudio: modelDetails.is_audio,
            });

            const modelSizeBytes = modelDetails.model_size_bytes;
            const autoSelectionPromise =
              shouldApplyTrainingDefaults &&
              typeof modelSizeBytes === "number" &&
              modelSizeBytes > 0 &&
              get().trainingMethod !== "cpt"
                ? selectTrainingMethodForHardware(
                    modelSizeBytes,
                    patch.contextLength ?? get().contextLength,
                    controller.signal,
                  )
                : null;

            // Preserve CPT hyperparams: YAML adapter defaults are tuned for standard LoRA.
            const cptOverrides =
              shouldApplyTrainingDefaults && get().trainingMethod === "cpt"
                ? getCptModelDefaultsPatch()
                : {};
            const modelDefaultsBaseline = {
              ...modelDefaultsPatch,
              ...(get().trainingMethod === "cpt"
                ? getCptModelDefaultsPatch()
                : {}),
            };
            const advancedSettingsBaseline =
              get().advancedSettingsBaseline ?? modelDefaultsBaseline;
            const deferredCompletionDefault =
              !shouldApplyTrainingDefaults &&
              get().trainOnCompletionsDefaultPendingFor === modelName
                ? {
                    trainOnCompletions:
                      resolveDeferredTrainOnCompletionsDefault({
                        currentValue: get().trainOnCompletions,
                        datasetFormat: get().datasetFormat,
                        datasetStreaming: get().datasetStreaming,
                        isEmbeddingModel: isEmbedding,
                        modelDefault: modelDefaultsPatch.trainOnCompletions,
                        trainingMethod: get().trainingMethod,
                      }),
                    trainOnCompletionsDefaultPendingFor: null,
                  }
                : {};

            set({
              ...patch,
              ...cptOverrides,
              ...deferredCompletionDefault,
              ...(shouldApplyTrainingDefaults
                ? {
                    trainingMethodProvenance: {
                      ...get().trainingMethodProvenance,
                      learningRateManuallySet: false,
                      modelAdapterLearningRate,
                    },
                  }
                : {}),
              advancedSettingsBaseline: shouldApplyTrainingDefaults
                ? modelDefaultsBaseline
                : advancedSettingsBaseline,
              modelType: inferredModelType,
              isVisionModel: modelDetails.is_vision,
              isEmbeddingModel: isEmbedding,
              isAudioModel: isAudio,
              isLoadingModelDefaults: autoSelectionPromise !== null,
              isCheckingVision: false,
              modelDefaultsError: null,
              modelDefaultsAppliedFor: modelName,
              ...(shouldApplyTrainingDefaults
                ? { trainOnCompletionsDefaultPendingFor: null }
                : {}),
              maxPositionEmbeddings:
                modelDetails.max_position_embeddings ?? null,
            });

            if (autoSelectionPromise) {
              void autoSelectionPromise.then((method) => {
                if (controller.signal.aborted || !requestMatchesSelection()) {
                  return;
                }
                const methodWasEdited =
                  _trainingMethodEditGeneration !==
                  trainingMethodEditGeneration;
                if (
                  !method ||
                  methodWasEdited ||
                  !canApplyTrainingDefaults() ||
                  get().trainingMethod === "cpt"
                ) {
                  set({ isLoadingModelDefaults: false });
                  return;
                }
                const lrPatch =
                  !get().trainingMethodProvenance.learningRateManuallySet &&
                  !modelConfigHasLR
                    ? {
                        learningRate:
                          method === "full" ? LR_DEFAULT_FULL : LR_DEFAULT_LORA,
                      }
                    : {};
                set({
                  trainingMethod: method,
                  ...lrPatch,
                  isLoadingModelDefaults: false,
                });
              });
            }
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
              ...(canApplyTrainingDefaults()
                ? { visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize }
                : {}),
            });

            if (preferLocalCache) {
              set({ isCheckingVision: false });
              return;
            }

            // Fallback vision check; pass the token so a gated/private VLM classifies right.
            void checkVisionModel(modelName, getHfToken() || undefined)
              .then((isVision) => {
                if (controller.signal.aborted || !requestMatchesSelection()) {
                  return;
                }
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
                if (controller.signal.aborted || !requestMatchesSelection()) {
                  return;
                }
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
        const requestedCacheIdentity =
          isHfSelection && preferLocalCache
            ? createDatasetCacheUsabilityIdentity({
                dataset: datasetName,
                cachePath: state.datasetLocalPath,
                subset: state.datasetSubset,
                split,
                streaming: state.datasetStreaming,
              })
            : null;
        const requestedCacheValidation = requestedCacheIdentity
          ? trainingDatasetCacheRejections.beginValidation(
              requestedCacheIdentity,
            )
          : null;
        checkDatasetFormat({
          datasetName,
          hfToken: getHfToken() || null,
          subset: state.datasetSubset,
          split,
          isVlm: state.isVisionModel,
          preferLocalCache,
          localPath:
            isHfSelection && preferLocalCache ? state.datasetLocalPath : null,
          signal: controller.signal,
        })
          .then((res) => {
            if (controller.signal.aborted) return;
            const isImage = !!res.is_image;
            const isAudio = !!res.is_audio;
            const current = get();
            const streamingDisabled =
              current.datasetStreaming && (isImage || isAudio);
            const updates: Partial<TrainingConfigState> = {
              isDatasetImage: isImage,
              isDatasetAudio: isAudio,
              isCheckingDataset: false,
              datasetCheckFailed: false,
              ...(streamingDisabled ? { datasetStreaming: false } : {}),
            };
            if (!_trainOnCompletionsManuallySet) {
              const { isVisionModel, isAudioModel } = current;
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
            if (streamingDisabled) {
              toast.info(
                translate(
                  "studio.dataset.streaming.notifications.disabledForDetectedModality",
                ),
              );
              recheckSelectedDatasetForStreamingMode(false);
            }
          })
          .catch((error) => {
            if (controller.signal.aborted) return;
            if (
              requestedCacheIdentity &&
              requestedCacheValidation &&
              isMissingLocalDatasetCacheError(error)
            ) {
              const current = get();
              const currentCacheIdentity =
                current.datasetSource === "huggingface" &&
                current.dataset === datasetName &&
                current.datasetKnownCached
                  ? createDatasetCacheUsabilityIdentity({
                      dataset: datasetName,
                      cachePath: current.datasetLocalPath,
                      subset: current.datasetSubset,
                      split: current.datasetSplit,
                      streaming: current.datasetStreaming,
                    })
                  : null;
              if (
                !currentCacheIdentity ||
                !datasetCacheUsabilityIdentitiesEqual(
                  currentCacheIdentity,
                  requestedCacheIdentity,
                )
              ) {
                return;
              }
              if (
                !trainingDatasetCacheRejections.rejectValidation(
                  requestedCacheValidation,
                )
              ) {
                if (
                  _datasetCheckController === controller &&
                  claimDatasetCacheRecheck(
                    datasetCacheRecheckKey({
                      dataset: datasetName,
                      subset: requestedCacheIdentity.subset,
                      split,
                      streaming: requestedCacheIdentity.streaming,
                    }),
                  )
                ) {
                  runDatasetCheck(datasetName, split, {
                    preferLocalCache: true,
                  });
                  return;
                }
                // Retry budget spent: resolve remotely rather than spin on a churning inventory cache.
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
            if (
              error instanceof DatasetFormatError &&
              error.status === 404 &&
              clearDeletedDataset(datasetName)
            ) {
              toast.error(error.message);
              return;
            }
            set({
              isDatasetImage: null,
              isDatasetAudio: false,
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
        if (requiresExplicitCachedDatasetSplit(state)) {
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
        manualDatasetOptionsValid: true,
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
        trainingDatasetCacheRejections.reset();
        _datasetCheckController?.abort();
        _datasetCheckController = null;
        _trainOnCompletionsManuallySet = false;
        const browseDatasetSelection = createHfBrowseDatasetSelection(
          datasetId,
          options,
        );
        setUserEdit({
          datasetSource: "huggingface",
          browseDatasetSelection,
          dataset: datasetId,
          uploadedFile: null,
          ...resetDatasetState(),
          datasetKnownCached: browseDatasetSelection.knownCached,
          datasetLocalPath: browseDatasetSelection.localPath,
          ...datasetSelectionStreamingPatch(browseDatasetSelection, options),
        });
        if (datasetId && !requiresExplicitCachedDatasetSplit(get())) {
          runDatasetCheck(datasetId, "train");
        }
      };

      const selectLocalDatasetInternal = (uploadedFile: string | null) => {
        trainingDatasetCacheRejections.reset();
        _datasetCheckController?.abort();
        _datasetCheckController = null;
        _trainOnCompletionsManuallySet = false;
        setUserEdit({
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
        trainingDatasetCacheRejections.reset();
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
        setUserEdit({
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
          !selectionChanged && currentState.modelFormat === "adapter"
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
          advancedSettingsBaseline?: null;
          trainOnCompletionsDefaultPendingFor?: null;
        } = {
          selectedModel,
          modelDefaultsError: null,
          modelKnownCached: nextKnownCached,
          modelLocalPath: nextLocalPath,
          modelFormat: selectedModel
            ? (options?.modelFormat ?? previousAdapterFormat)
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
          patch.advancedSettingsBaseline = null;
          patch.trainOnCompletionsDefaultPendingFor = null;
        }
        setUserEdit(patch);

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
            advancedSettingsBaseline: null,
            trainOnCompletionsDefaultPendingFor: null,
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
        ...initialTrainingConfigState,
        setStep: (step) => set({ currentStep: step }),
        nextStep: () =>
          set({ currentStep: clampTrainingStep(get().currentStep + 1) }),
        prevStep: () =>
          set({ currentStep: clampTrainingStep(get().currentStep - 1) }),
        setModelType: (modelType) => {
          _modelConfigController?.abort();
          _modelConfigController = null;

          setUserEdit({
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
            advancedSettingsBaseline: null,
            trainOnCompletionsDefaultPendingFor: null,
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
          });
          if (cacheReferenceChanged) {
            void loadAndApplyModelDefaults(model, {
              applyTrainingDefaults: canReapplyModelDefaults(model),
            });
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
          });
          void loadAndApplyModelDefaults(model, {
            applyTrainingDefaults: canReapplyModelDefaults(model),
          });
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
          const defaultsAlreadyApplied =
            state.modelDefaultsAppliedFor === state.selectedModel;
          void loadAndApplyModelDefaults(state.selectedModel, {
            applyTrainingDefaults:
              !defaultsAlreadyApplied ||
              canReapplyModelDefaults(state.selectedModel),
          });
        },
        setProjectName: (projectName) => setUserEdit({ projectName }),
        setTrainingMethod: (trainingMethod) => {
          _trainingMethodEditGeneration += 1;
          const state = get();
          const patch = buildTrainingMethodPatch(state, trainingMethod);
          setUserEdit({
            ...patch,
            ...(patch.trainOnCompletions !== undefined
              ? { trainOnCompletionsDefaultPendingFor: null }
              : {}),
          });
        },
        setDatasetSource: (datasetSource) => {
          const state = get();
          if (datasetSource === state.datasetSource) {
            const invariantPatch = datasetSourceInvariantPatch(state);
            if (invariantPatch.datasetStreaming !== undefined) {
              set(invariantPatch);
            }
            return;
          }
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
          setUserEdit((state) => {
            if (state.trainingMethod === "cpt") {
              if (isRawTextDatasetFormat(datasetFormat)) {
                return {
                  datasetFormat: "raw",
                  trainOnCompletions: false,
                  trainOnCompletionsDefaultPendingFor: null,
                  trainingMethodProvenance: {
                    ...state.trainingMethodProvenance,
                    datasetFormatBeforeCpt: null,
                  },
                };
              }
              return {
                datasetFormat: "raw",
                trainOnCompletions: false,
                trainOnCompletionsDefaultPendingFor: null,
              };
            }

            return {
              datasetFormat,
              trainOnCompletions: isRawTextDatasetFormat(datasetFormat)
                ? false
                : state.trainOnCompletions,
              ...(isRawTextDatasetFormat(datasetFormat)
                ? { trainOnCompletionsDefaultPendingFor: null }
                : {}),
            };
          }),
        setDataset: (dataset) => {
          const datasetId = dataset?.trim() || null;
          trainingDatasetCacheRejections.reset();
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          setUserEdit((state) => ({
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
            manualDatasetOptionsValid: true,
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
          const state = get();
          if (
            state.datasetSubset !== datasetSubset ||
            state.datasetSplit !== null
          ) {
            trainingDatasetCacheRejections.reset(state.dataset);
          }
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          setUserEdit({
            datasetSubset,
            datasetSplit: null,
            datasetEvalSplit: null,
            manualDatasetOptionsValid: true,
            datasetManualMapping: emptyManualMapping(),
            isDatasetImage: null,
            isDatasetAudio: false,
            isCheckingDataset: false,
          });
        },
        setDatasetSplit: (datasetSplit) => {
          const state = get();
          if (state.datasetSplit !== datasetSplit) {
            trainingDatasetCacheRejections.reset(state.dataset);
          }
          const nextState = { ...state, datasetSplit };
          const streamingPatch = streamingCompatiblePatch(nextState);
          setUserEdit({
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
          if (requiresExplicitCachedDatasetSplit({ ...state, datasetSplit })) {
            return;
          }

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
          if (requiresExplicitCachedDatasetSplit(state)) {
            return;
          }

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
          setUserEdit({
            datasetEvalSplit,
            evalSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setDatasetStreaming: (datasetStreaming) => {
          const state = get();
          if (datasetStreaming && state.datasetSource !== "huggingface") {
            if (state.datasetStreaming) {
              set({ datasetStreaming: false });
            }
            return;
          }
          const changed = state.datasetStreaming !== datasetStreaming;
          if (!datasetStreaming) {
            if (changed) {
              trainingDatasetCacheRejections.reset(state.dataset);
            }
            setUserEdit({
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

          if (state.maxSteps <= 0) {
            set({ datasetStreaming: false });
            toast.warning(
              translate("studio.dataset.streaming.notifications.needsMaxSteps"),
            );
            return;
          }

          if (changed) {
            trainingDatasetCacheRejections.reset(state.dataset);
          }

          const dropsTrainOnCompletions = state.trainOnCompletions;
          const dropsEval = !hasSeparateStreamingEvalSplit(state);

          setUserEdit({
            datasetStreaming: true,
            trainOnCompletions: false,
            trainOnCompletionsDefaultPendingFor: null,
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
        setManualDatasetOptionsValid: (manualDatasetOptionsValid) =>
          set((state) =>
            state.manualDatasetOptionsValid === manualDatasetOptionsValid
              ? state
              : { manualDatasetOptionsValid },
          ),
        markManualDatasetOptionsEdited: (manualDatasetOptionsValid) =>
          set((state) => ({
            manualDatasetOptionsValid,
            userEditRevision: state.userEditRevision + 1,
          })),
        setDatasetManualMapping: (datasetManualMapping) =>
          setUserEdit({ datasetManualMapping }),
        setDatasetAdvisorFields: (fields) =>
          setUserEdit({
            datasetSystemPrompt:
              fields.systemPrompt ?? get().datasetSystemPrompt,
            datasetLabelMapping:
              fields.labelMapping ?? get().datasetLabelMapping,
            datasetAdvisorNotification:
              fields.notification !== undefined
                ? fields.notification
                : get().datasetAdvisorNotification,
          }),
        setDatasetSliceStart: (datasetSliceStart) =>
          setUserEdit({ datasetSliceStart }),
        setDatasetSliceEnd: (datasetSliceEnd) =>
          setUserEdit({ datasetSliceEnd }),
        setUploadedFile: (uploadedFile) => {
          _datasetCheckController?.abort();
          _datasetCheckController = null;
          _trainOnCompletionsManuallySet = false;
          setUserEdit((state) => ({
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
            manualDatasetOptionsValid: true,
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
          setUserEdit({
            uploadedEvalFile,
            evalSteps: uploadedEvalFile ? 0.1 : 0,
          }),
        setEpochs: (epochs) => setUserEdit({ epochs }),
        setContextLength: (contextLength) => setUserEdit({ contextLength }),
        setVisionImageSize: (visionImageSize) =>
          setUserEdit({ visionImageSize }),
        setLearningRate: (learningRate) =>
          setUserEdit((state) => ({
            learningRate,
            trainingMethodProvenance: {
              ...state.trainingMethodProvenance,
              learningRateManuallySet: true,
            },
          })),
        setEmbeddingLearningRate: (embeddingLearningRate) =>
          setUserEdit({ embeddingLearningRate }),
        setOptimizerType: (optimizerType) => setUserEdit({ optimizerType }),
        setLrSchedulerType: (lrSchedulerType) =>
          setUserEdit({ lrSchedulerType }),
        setLoraRank: (loraRank) => setUserEdit({ loraRank }),
        setLoraAlpha: (loraAlpha) => setUserEdit({ loraAlpha }),
        setLoraDropout: (loraDropout) => setUserEdit({ loraDropout }),
        setLoraVariant: (loraVariant) => setUserEdit({ loraVariant }),
        setBatchSize: (batchSize) => setUserEdit({ batchSize }),
        setGradientAccumulation: (gradientAccumulation) =>
          setUserEdit({ gradientAccumulation }),
        setWeightDecay: (weightDecay) => setUserEdit({ weightDecay }),
        setWarmupSteps: (warmupSteps) => setUserEdit({ warmupSteps }),
        setMaxSteps: (maxSteps) => {
          const state = get();
          // streamingCompatiblePatch already turns streaming off when maxSteps<=0.
          const streamingPatch = streamingCompatiblePatch({
            ...state,
            maxSteps,
          });
          setUserEdit({
            maxSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setSaveSteps: (saveSteps) => setUserEdit({ saveSteps }),
        setEvalSteps: (evalSteps) => {
          const state = get();
          const streamingPatch = streamingCompatiblePatch({
            ...state,
            evalSteps,
          });
          setUserEdit({
            evalSteps,
            ...streamingPatch,
          });
          notifyStreamingCompat(streamingPatch);
        },
        setPacking: (packing) => setUserEdit({ packing }),
        setTrainOnCompletions: (trainOnCompletions) => {
          _trainOnCompletionsManuallySet = true;
          setUserEdit({
            trainOnCompletions,
            trainOnCompletionsDefaultPendingFor: null,
            ...(trainOnCompletions ? { datasetStreaming: false } : {}),
          });
        },
        setGradientCheckpointing: (gradientCheckpointing) =>
          setUserEdit({ gradientCheckpointing }),
        setRandomSeed: (randomSeed) => setUserEdit({ randomSeed }),
        setEnableWandb: (enableWandb) => setUserEdit({ enableWandb }),
        setWandbToken: (wandbToken) => setUserEdit({ wandbToken }),
        setWandbProject: (wandbProject) => setUserEdit({ wandbProject }),
        setEnableTensorboard: (enableTensorboard) =>
          setUserEdit({ enableTensorboard }),
        setTensorboardDir: (tensorboardDir) => setUserEdit({ tensorboardDir }),
        setLogFrequency: (logFrequency) => setUserEdit({ logFrequency }),
        setFinetuneVisionLayers: (finetuneVisionLayers) =>
          setUserEdit({ finetuneVisionLayers }),
        setFinetuneLanguageLayers: (finetuneLanguageLayers) =>
          setUserEdit({ finetuneLanguageLayers }),
        setFinetuneAttentionModules: (finetuneAttentionModules) =>
          setUserEdit({ finetuneAttentionModules }),
        setFinetuneMLPModules: (finetuneMLPModules) =>
          setUserEdit({ finetuneMLPModules }),
        setTargetModules: (targetModules) => setUserEdit({ targetModules }),
        setS3Config: (s3Config) => setUserEdit({ s3Config }),
        canProceed: () => canProceedForTrainingStep(get()),
        reset: () => {
          trainingDatasetCacheRejections.reset();
          _trainOnCompletionsManuallySet = false;
          _modelDefaultsEditBaseline = null;
          setUserEdit(initialTrainingConfigState);
        },
        resetToModelDefaults: () => {
          const { selectedModel } = get();
          if (!selectedModel) return;
          setUserEdit({
            modelDefaultsAppliedFor: null,
            advancedSettingsBaseline: null,
            visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
          });
          loadAndApplyModelDefaults(selectedModel);
        },
        applyConfigPatch: (config: BackendModelConfig) => {
          const patch = mapBackendModelConfigToTrainingPatch(config);
          setUserEdit((state) => ({
            ...patch,
            ...(patch.trainOnCompletions !== undefined
              ? { trainOnCompletionsDefaultPendingFor: null }
              : {}),
            ...(patch.learningRate !== undefined
              ? {
                  trainingMethodProvenance: {
                    ...state.trainingMethodProvenance,
                    learningRateManuallySet: false,
                  },
                }
              : {}),
          }));
        },
      };
    },
    {
      name: TRAINING_CONFIG_PERSISTENCE_NAME,
      version: TRAINING_CONFIG_PERSISTENCE_VERSION,
      migrate: migrateTrainingConfig,
      partialize: partializeTrainingConfig,
      merge: mergeTrainingConfig,
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        const sourcePatch = datasetSourceInvariantPatch(state);
        const patch = {
          ...streamingCompatiblePatch({ ...state, ...sourcePatch }),
          ...sourcePatch,
        };
        if (Object.keys(patch).length > 0) {
          // Sync localStorage hydration runs inside create(), before useTrainingConfigStore is assigned
          // (TDZ). Defer to a microtask so the store exists when the persisted combo is reconciled.
          queueMicrotask(() => useTrainingConfigStore.setState(patch));
        }
      },
    },
  ),
);

export function clearDeletedDataset(datasetName: string): boolean {
  const state = useTrainingConfigStore.getState();
  const selection = resolveDeletedLocalDatasetSelection({
    datasetName,
    source: state.datasetSource,
    dataset: state.dataset,
    uploadedFile: state.uploadedFile,
  });
  switch (selection) {
    case "upload":
      state.selectLocalDataset(null);
      return true;
    case "huggingface":
      state.setDataset(null);
      return true;
    default:
      return false;
  }
}
