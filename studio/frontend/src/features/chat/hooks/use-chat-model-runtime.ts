// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { consumeNativePathToken } from "@/features/native-intents";
import type { ModelInventoryFormat } from "@/features/inventory";
import { looksLikeLocalPath } from "@/lib/local-path";
import {
  notifyNative,
  primeNativeNotificationPermission,
  safeNotificationLabel,
  sanitizeNotificationBody,
} from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { getHfToken } from "@/stores/hf-token-store";
import { createElement, useCallback } from "react";
import {
  getDownloadProgress,
  getGgufDownloadProgress,
  getInferenceStatus,
  getLoadProgress,
  listLoras,
  listModels,
  loadModel,
  unloadModel,
  validateModel,
} from "../api/chat-api";
import { ModelLoadDescription } from "../components/model-load-status";
import { isExternalModelId } from "../external-providers";
import {
  ggufVariantsMatch,
  modelIdsMatch,
} from "../model-config/model-identity";
import type { PerModelConfig } from "../model-config/per-model-config";
import { applyPerModelConfigToRuntime } from "../model-runtime/apply-per-model-config";
import {
  clampLocalReasoningEffort,
  hydrateRuntimeFromLoadResponse,
  normalizeSpeculativeType,
} from "../model-runtime/load-hydration";
import {
  shouldLoadFromLocalFilesOnly,
  type LocalModelLoadSource,
} from "../model-runtime/local-files-only";
import {
  buildActiveModelLoadSource,
  resolveRollbackLoadOptions,
} from "../model-runtime/rollback-load-source";
import {
  mergeBackendRecommendedInference,
  resolveLoadMaxSeqLength,
} from "../presets/preset-policy";
import {
  CHAT_REASONING_ENABLED_KEY,
  type ChatModelLoadInfo,
  loadOptionalBool,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import { isMultimodalResponse } from "../types/api";
import type { ChatLoraSummary, ChatModelSummary } from "../types/runtime";
import { formatEta, formatRate } from "../utils/format-transfer";
import { smallQwenThinkingOffByDefault } from "../utils/qwen-params";

type SelectedModelInput = {
  id: string;
  isLora?: boolean;
  ggufVariant?: string;
  modelFormat?: ModelInventoryFormat | null;
  loadingDescription?: string;
  isDownloaded?: boolean;
  isPartial?: boolean;
  expectedBytes?: number;
  localPath?: string | null;
  preferLocalCache?: boolean;
  source?: LocalModelLoadSource;
  forceReload?: boolean;
  nativePathToken?: string;
  throwOnError?: boolean;
  // Per-model inference config to apply for this load. Passed explicitly (not
  // read back from a side-effecting global write) so the caller can't get the
  // order wrong; its presence is also what marks the load as config-driven.
  config?: PerModelConfig;
};

type RuntimePerModelConfigSnapshot = {
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  customContextLength: number | null;
  chatTemplateOverride: string | null;
  trustRemoteCode: boolean;
};

function getRuntimePerModelConfigSnapshot(): RuntimePerModelConfigSnapshot {
  const state = useChatRuntimeStore.getState();
  return {
    kvCacheDtype: state.kvCacheDtype,
    speculativeType: state.speculativeType,
    specDraftNMax: state.specDraftNMax,
    customContextLength: state.customContextLength,
    chatTemplateOverride: state.chatTemplateOverride,
    trustRemoteCode: state.params.trustRemoteCode ?? false,
  };
}

function restoreRuntimePerModelConfigSnapshot(
  snapshot: RuntimePerModelConfigSnapshot,
): void {
  useChatRuntimeStore.setState({
    kvCacheDtype: snapshot.kvCacheDtype,
    speculativeType: snapshot.speculativeType,
    specDraftNMax: snapshot.specDraftNMax,
    customContextLength: snapshot.customContextLength,
    chatTemplateOverride: snapshot.chatTemplateOverride,
  });
  const state = useChatRuntimeStore.getState();
  state.setParams({
    ...state.params,
    trustRemoteCode: snapshot.trustRemoteCode,
  });
}

function perModelConfigIdentity(config?: PerModelConfig): string {
  if (!config) return "";
  return JSON.stringify({
    chatTemplateOverride: config.chatTemplateOverride ?? null,
    customContextLength: config.customContextLength ?? null,
    kvCacheDtype: config.kvCacheDtype ?? null,
    specDraftNMax: config.specDraftNMax ?? null,
    speculativeType: config.speculativeType ?? null,
    trustRemoteCode: config.trustRemoteCode ?? false,
  });
}

const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "items-start gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

const ACTIVE_STREAM_LOAD_MESSAGE =
  "Stop the active response before loading a model.";

function hasActiveChatStream(): boolean {
  return Object.values(useChatRuntimeStore.getState().runningByThreadId).some(
    Boolean,
  );
}

let activeLoadAbort: AbortController | null = null;
let activeLoadToastId: string | number | null = null;
let activeLoadAttempt = 0;
let activeLoadToastDismissed = false;
let loadLifecycleGeneration = 0;
let cancelUnloadPending = false;
let cancelUnloadGeneration: number | null = null;

function getActiveLoadingModel() {
  return useChatRuntimeStore.getState().modelLoadInfo;
}

function loadingModelsMatch(
  candidate: ChatModelLoadInfo | null | undefined,
  expected: ChatModelLoadInfo,
): boolean {
  return (
    candidate != null &&
    modelIdsMatch(candidate.id, expected.id) &&
    ggufVariantsMatch(candidate.ggufVariant, expected.ggufVariant) &&
    (candidate.modelFormat ?? null) === (expected.modelFormat ?? null) &&
    (candidate.runtimeBackend ?? "unknown") ===
      (expected.runtimeBackend ?? "unknown") &&
    (candidate.localPath ?? null) === (expected.localPath ?? null) &&
    (candidate.nativePathToken ?? null) ===
      (expected.nativePathToken ?? null) &&
    Boolean(candidate.preferLocalCache) ===
      Boolean(expected.preferLocalCache) &&
    (candidate.maxSeqLength ?? null) === (expected.maxSeqLength ?? null) &&
    (candidate.configKey ?? "") === (expected.configKey ?? "")
  );
}

function waitForNextTask(): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, 0);
  });
}

const LORA_SUFFIX_RE = /_(\d{9,})$/;

function parseTrailingEpoch(input: string): number | undefined {
  const match = input.match(LORA_SUFFIX_RE);
  if (!match) {
    return undefined;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function stripTrailingEpoch(input: string): string {
  const cleaned = input
    .replace(LORA_SUFFIX_RE, "")
    .replace(/[_-]+$/, "")
    .trim();
  return cleaned || input;
}

function describeModel(model: {
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_audio?: boolean;
  has_audio_input?: boolean;
}): string | undefined {
  const tags: string[] = [];
  if (model.is_gguf) tags.push("GGUF");
  if (model.is_lora) tags.push("LoRA");
  if (model.is_vision) tags.push("Vision");
  if (model.is_audio) tags.push("Audio");
  if (model.has_audio_input) tags.push("Audio Input");
  if (
    !model.is_lora &&
    !model.is_vision &&
    !model.is_gguf &&
    !model.is_audio &&
    !model.has_audio_input
  )
    tags.push("Base");
  return tags.join(" · ");
}

function toChatModelSummary(model: {
  id: string;
  name?: string | null;
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
}): ChatModelSummary {
  return {
    id: model.id,
    name: model.name || model.id,
    description: describeModel(model),
    isLora: Boolean(model.is_lora),
    isVision: Boolean(model.is_vision),
    isGguf: Boolean(model.is_gguf),
    isAudio: Boolean(model.is_audio),
    audioType: model.audio_type ?? null,
    hasAudioInput: Boolean(model.has_audio_input),
  };
}

function toLoraSummary(lora: {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
  source?: "training" | "exported" | null;
  export_type?: "lora" | "merged" | "gguf" | null;
  run_display_name?: string | null;
  training_method?: string | null;
}): ChatLoraSummary {
  const idTail = lora.adapter_path.split("/").filter(Boolean).at(-1) ?? "";
  const updatedAt =
    parseTrailingEpoch(lora.display_name) ?? parseTrailingEpoch(idTail);
  const runDisplayName = lora.run_display_name?.trim() || undefined;

  return {
    id: lora.adapter_path,
    name: stripTrailingEpoch(lora.display_name),
    baseModel: lora.base_model || "Unknown base model",
    updatedAt,
    source: lora.source ?? undefined,
    exportType: lora.export_type ?? undefined,
    runDisplayName,
    trainingMethod: lora.training_method?.trim() || undefined,
  };
}

function getTrustRemoteCodeRequiredMessage(modelName: string): string {
  return `${modelName} needs custom code enabled to load. Turn on "Enable custom code" in Chat Settings, then try again.`;
}

export function useChatModelRuntime() {
  const params = useChatRuntimeStore((state) => state.params);
  const models = useChatRuntimeStore((state) => state.models);
  const loras = useChatRuntimeStore((state) => state.loras);
  const setModels = useChatRuntimeStore((state) => state.setModels);
  const setLoras = useChatRuntimeStore((state) => state.setLoras);
  const setParams = useChatRuntimeStore((state) => state.setParams);
  const setModelsError = useChatRuntimeStore((state) => state.setModelsError);
  const setCheckpoint = useChatRuntimeStore((state) => state.setCheckpoint);
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);
  const loadingModel = useChatRuntimeStore((state) => state.modelLoadInfo);
  const loadProgress = useChatRuntimeStore((state) => state.modelLoadProgress);
  const loadToastDismissed = useChatRuntimeStore(
    (state) => state.modelLoadToastDismissed,
  );
  const setLoadingModel = useChatRuntimeStore(
    (state) => state.setModelLoadInfo,
  );
  const setLoadProgress = useChatRuntimeStore(
    (state) => state.setModelLoadProgress,
  );
  const setLoadToastDismissed = useChatRuntimeStore(
    (state) => state.setModelLoadToastDismissed,
  );

  const setLoadToastDismissedState = useCallback(
    (dismissed: boolean) => {
      activeLoadToastDismissed = dismissed;
      setLoadToastDismissed(dismissed);
    },
    [setLoadToastDismissed],
  );

  const resetLoadingUi = useCallback(() => {
    setLoadingModel(null);
    setLoadProgress(null);
    activeLoadAbort = null;
    activeLoadToastId = null;
    setLoadToastDismissedState(false);
    if (!cancelUnloadPending) {
      useChatRuntimeStore.getState().setModelLoading(false);
    }
  }, [setLoadProgress, setLoadToastDismissedState, setLoadingModel]);

  const renderLoadDescription = useCallback(
    (
      title: string,
      message: string,
      progressPercent?: number | null,
      progressLabel?: string | null,
      onStop?: () => void,
    ) =>
      createElement(ModelLoadDescription, {
        title,
        message,
        progressPercent,
        progressLabel,
        onStop,
      }),
    [],
  );

  const refresh = useCallback(async () => {
    setModelsError(null);
    try {
      const [listRes, statusRes, lorasRes] = await Promise.all([
        listModels(),
        getInferenceStatus(),
        listLoras(),
      ]);

      setModels(listRes.models.map(toChatModelSummary));
      setLoras(lorasRes.loras.map(toLoraSummary));

      const selectedCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const isExternalSelectionActive = isExternalModelId(selectedCheckpoint);
      if (statusRes.active_model && !isExternalSelectionActive) {
        setCheckpoint(statusRes.active_model, statusRes.gguf_variant);

        // Apply inference defaults on reconnect (page refresh with model already loaded)
        if (statusRes.inference) {
          const currentParams = useChatRuntimeStore.getState().params;
          setParams(
            mergeBackendRecommendedInference({
              current: currentParams,
              response: statusRes,
              modelId: statusRes.active_model,
              presetSource: useChatRuntimeStore.getState().activePresetSource,
            }),
          );
        }

        // Restore reasoning/tools support flags and context length
        const hydratingExistingModel =
          !modelIdsMatch(selectedCheckpoint, statusRes.active_model) ||
          !ggufVariantsMatch(
            useChatRuntimeStore.getState().activeGgufVariant,
            statusRes.gguf_variant,
          );
        const supportsReasoning = statusRes.supports_reasoning ?? false;
        const reasoningAlwaysOn = statusRes.reasoning_always_on ?? false;
        const reasoningStyle = statusRes.reasoning_style ?? "enable_thinking";
        const reasoningEffortLevels =
          reasoningStyle === "reasoning_effort"
            ? (["low", "medium", "high"] as const)
            : (["low", "medium", "high"] as const);
        const supportsPreserveThinking =
          statusRes.supports_preserve_thinking ?? false;
        const supportsTools = statusRes.supports_tools ?? false;
        const storedReasoningEnabled = loadOptionalBool(
          CHAT_REASONING_ENABLED_KEY,
        );
        const currentGgufContextLength = statusRes.is_gguf
          ? (statusRes.context_length ?? null)
          : null;
        const ggufMaxContextLength = statusRes.is_gguf
          ? (statusRes.max_context_length ?? null)
          : null;
        const ggufNativeContextLength = statusRes.is_gguf
          ? (statusRes.native_context_length ?? null)
          : null;
        const loadedCustomContextLength =
          statusRes.is_gguf &&
          currentGgufContextLength != null &&
          (ggufNativeContextLength == null ||
            currentGgufContextLength !== ggufNativeContextLength)
            ? currentGgufContextLength
            : null;
        const currentSpecType = normalizeSpeculativeType(
          statusRes.speculative_type,
        );
        // Refresh runs both on F5 (fresh store needs hydration) AND right
        // after a fresh load (store was already set by the load path). For
        // the user-configurable model params we only hydrate when the shadow
        // `loaded*` field is still null -- that signals "not yet hydrated".
        // Otherwise we'd clobber the values the load path just applied and
        // the UI would appear to revert the user's changes.
        const prevState = useChatRuntimeStore.getState();
        const clampedReasoningEffort = clampLocalReasoningEffort(
          prevState.reasoningEffort,
        );
        const nextDefaultChatTemplate =
          statusRes.chat_template === undefined
            ? prevState.defaultChatTemplate
            : statusRes.chat_template;
        useChatRuntimeStore.setState({
          supportsReasoning,
          reasoningAlwaysOn,
          reasoningStyle,
          supportsReasoningOff: reasoningStyle !== "reasoning_effort",
          reasoningEffortLevels,
          reasoningEffort: clampedReasoningEffort,
          supportsPreserveThinking,
          supportsTools,
          // Reset per-turn reasoning flag so:
          //   1. models that do not support reasoning do not inherit a stale
          //      off state from a prior model, and
          //   2. local reasoning-effort models (where the composer hides
          //      the Off option via supportsReasoningOff=false) cannot end
          //      up with reasoningEnabled=false carried over from an
          //      external model where Off was selected — the composer would
          //      keep showing "Think: <level>" via effectiveReasoningEnabled,
          //      but the chat-adapter would omit the kwarg and the Harmony
          //      template would fall back to its own default effort.
          reasoningEnabled: supportsReasoning
            ? reasoningStyle === "reasoning_effort"
              ? true
              : useChatRuntimeStore.getState().reasoningEnabled
            : true,
          ggufContextLength: currentGgufContextLength,
          ggufMaxContextLength,
          ggufNativeContextLength,
          customContextLength: loadedCustomContextLength,
          loadedCustomContextLength,
          modelRequiresTrustRemoteCode:
            statusRes.requires_trust_remote_code ?? false,
          defaultChatTemplate: nextDefaultChatTemplate,
          loadedIsMultimodal: isMultimodalResponse(statusRes),
          ...(prevState.loadedSpeculativeType === null && {
            speculativeType: currentSpecType,
            loadedSpeculativeType: currentSpecType,
          }),
          ...(statusRes.spec_draft_n_max !== undefined &&
            prevState.loadedSpecDraftNMax === null &&
            prevState.specDraftNMax === null && {
              specDraftNMax: statusRes.spec_draft_n_max ?? null,
              loadedSpecDraftNMax: statusRes.spec_draft_n_max ?? null,
            }),
          ...(statusRes.cache_type_kv !== undefined &&
            prevState.loadedKvCacheDtype === null && {
              kvCacheDtype: statusRes.cache_type_kv,
              loadedKvCacheDtype: statusRes.cache_type_kv,
            }),
          ...(statusRes.chat_template_override !== undefined &&
            prevState.loadedChatTemplateOverride === null &&
            prevState.chatTemplateOverride === null && {
              chatTemplateOverride: statusRes.chat_template_override,
              loadedChatTemplateOverride: statusRes.chat_template_override,
            }),
        });

        if (
          supportsReasoning &&
          hydratingExistingModel &&
          storedReasoningEnabled === null
        ) {
          useChatRuntimeStore.setState({
            reasoningEnabled: !smallQwenThinkingOffByDefault(
              statusRes.active_model,
            ),
          });
        }
      } else if (!statusRes.active_model && !isExternalSelectionActive) {
        useChatRuntimeStore.setState({
          modelRequiresTrustRemoteCode: false,
          loadedIsMultimodal: false,
        });
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load models";
      setModelsError(message);
      toast.error("Failed to refresh models", {
        description: message,
      });
    }
  }, [setCheckpoint, setLoras, setModels, setModelsError, setParams]);

  const cancelLoading = useCallback(() => {
    const model = getActiveLoadingModel();
    if (!model) return;
    const cancelGeneration = ++loadLifecycleGeneration;
    activeLoadAbort?.abort();
    activeLoadAbort = null;
    const tid = activeLoadToastId;
    activeLoadToastId = null;
    setLoadingModel(null);
    setLoadProgress(null);
    setLoadToastDismissedState(false);
    clearCheckpoint();
    if (tid != null) toast.dismiss(tid);
    const isCachedOrLocal = model.isDownloaded || model.isCachedLora;
    toast.info("Stopped loading model", {
      description: isCachedOrLocal
        ? undefined
        : "The current download may still finish in the background.",
    });
    cancelUnloadPending = true;
    cancelUnloadGeneration = cancelGeneration;
    useChatRuntimeStore.getState().setModelLoading(true);
    void (async () => {
      try {
        await waitForNextTask();
        const activeModel = getActiveLoadingModel();
        if (
          loadLifecycleGeneration === cancelGeneration &&
          (!activeModel || loadingModelsMatch(activeModel, model))
        ) {
          await unloadModel({ model_path: model.id }).catch(() => {});
        }
      } finally {
        if (cancelUnloadGeneration === cancelGeneration) {
          cancelUnloadPending = false;
          cancelUnloadGeneration = null;
        }
        if (
          loadLifecycleGeneration === cancelGeneration &&
          !getActiveLoadingModel()
        ) {
          useChatRuntimeStore.getState().setModelLoading(false);
        }
      }
    })();
  }, [
    clearCheckpoint,
    setLoadProgress,
    setLoadToastDismissedState,
    setLoadingModel,
  ]);

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      const ggufVariant =
        typeof selection === "string" ? undefined : selection.ggufVariant;
      const modelFormat =
        typeof selection === "string" ? null : (selection.modelFormat ?? null);
      const forceReload =
        typeof selection === "string"
          ? false
          : (selection.forceReload ?? false);
      const nativePathToken =
        typeof selection === "string" ? undefined : selection.nativePathToken;
      const localPath =
        typeof selection === "string" ? null : (selection.localPath ?? null);
      const throwOnError =
        typeof selection === "string"
          ? false
          : (selection.throwOnError ?? false);
      const perModelConfig =
        typeof selection === "string" ? undefined : selection.config;
      const hasPerModelConfig = perModelConfig != null;
      const configKey = perModelConfigIdentity(perModelConfig);
      const preferLocalCache =
        typeof selection === "string"
          ? false
          : Boolean(selection.preferLocalCache);
      const runtimeState = useChatRuntimeStore.getState();
      const currentVariant = runtimeState.activeGgufVariant;
      const currentCheckpoint = runtimeState.params.checkpoint;
      const requestedMaxSeqLength = runtimeState.params.maxSeqLength;
      const runtimeBackend: NonNullable<ChatModelLoadInfo["runtimeBackend"]> =
        modelFormat === "gguf" || ggufVariant != null
          ? "llama_cpp"
          : modelFormat === "adapter"
            ? "adapter"
            : modelFormat === "safetensors" || modelFormat === "checkpoint"
              ? "transformers"
              : "unknown";
      if (
        !forceReload &&
        (!modelId ||
          (modelIdsMatch(currentCheckpoint, modelId) &&
            ggufVariantsMatch(ggufVariant, currentVariant)))
      ) {
        return;
      }
      // Prevent duplicate loads if already loading this model
      const activeLoadingModel = getActiveLoadingModel();
      if (
        modelIdsMatch(activeLoadingModel?.id, modelId) &&
        ggufVariantsMatch(activeLoadingModel?.ggufVariant, ggufVariant) &&
        (activeLoadingModel?.modelFormat ?? null) === (modelFormat ?? null) &&
        (activeLoadingModel?.runtimeBackend ?? "unknown") === runtimeBackend &&
        (activeLoadingModel?.localPath ?? null) === (localPath ?? null) &&
        Boolean(activeLoadingModel?.preferLocalCache) === preferLocalCache &&
        (activeLoadingModel?.maxSeqLength ?? null) === requestedMaxSeqLength &&
        (activeLoadingModel?.configKey ?? "") === configKey &&
        (activeLoadingModel?.nativePathToken ?? null) ===
          (nativePathToken ?? null)
      )
        return;
      if (hasActiveChatStream()) {
        toast.warning("Stop the active response first", {
          description: "Model changes are disabled while a reply is streaming.",
        });
        if (throwOnError) {
          throw new Error(ACTIVE_STREAM_LOAD_MESSAGE);
        }
        return;
      }
      loadLifecycleGeneration += 1;
      const previousToastId = activeLoadToastId;
      activeLoadAbort?.abort();
      activeLoadToastId = null;
      if (previousToastId != null) {
        toast.dismiss(previousToastId);
      }
      const previousRuntimeConfigSnapshot = getRuntimePerModelConfigSnapshot();
      // Apply the picker's config only after both early returns above. Applying
      // it earlier would mutate the store even when this call is a no-op (e.g. a
      // concurrent same-model load already in flight): the in-flight load never
      // reloads with the new settings and clobbers them on completion, silently
      // dropping the just-confirmed KV dtype / spec mode / chat template /
      // context. Applied here it still precedes the load's store reads, so its
      // presence marks this as a config-driven load.
      if (perModelConfig) {
        applyPerModelConfigToRuntime(perModelConfig);
      }

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const extraLoadingDescription =
        typeof selection === "string"
          ? undefined
          : selection.loadingDescription;
      const isDownloaded =
        typeof selection === "string"
          ? false
          : (selection.isDownloaded ?? false);
      const isPartial =
        typeof selection === "string" ? false : (selection.isPartial ?? false);
      const model = models.find((entry) => modelIdsMatch(entry.id, modelId));
      const lora = loras.find((entry) => modelIdsMatch(entry.id, modelId));
      const loraIsAdapter = lora?.exportType === "lora";
      const isLora = explicitIsLora ?? model?.isLora ?? loraIsAdapter ?? false;
      const displayName = model?.name || lora?.name || modelId;
      const loadAttemptId = ++activeLoadAttempt;
      primeNativeNotificationPermission().catch(() => undefined);
      const notificationModelKey = `${modelId}:${ggufVariant ?? ""}:${loadAttemptId}`;
      const safeModelName = safeNotificationLabel(displayName, "The model");
      const previousCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const previousVariant =
        useChatRuntimeStore.getState().activeGgufVariant ?? null;
      const reloadingSameModel =
        modelIdsMatch(previousCheckpoint, modelId) &&
        ggufVariantsMatch(ggufVariant, previousVariant);
      const previousModel = previousCheckpoint
        ? models.find((entry) => modelIdsMatch(entry.id, previousCheckpoint))
        : undefined;
      const previousLora = previousCheckpoint
        ? loras.find((entry) => modelIdsMatch(entry.id, previousCheckpoint))
        : undefined;
      const previousIsLora =
        previousModel?.isLora ?? previousLora?.exportType === "lora";
      const isLocal = looksLikeLocalPath(modelId);
      const isCachedLora = isLora && isLocal;
      const localFilesOnly = shouldLoadFromLocalFilesOnly({
        modelId,
        nativePathToken,
        isCachedLora,
        selection: typeof selection === "string" ? undefined : selection,
      });
      const resolvedLocalPath = localFilesOnly ? localPath : null;
      const activeModelLoadSource =
        typeof selection === "string"
          ? null
          : buildActiveModelLoadSource({
              source: selection.source,
              isDownloaded,
              isPartial,
              preferLocalCache,
              localPath,
              modelFormat,
              ggufVariant: ggufVariant ?? null,
              nativePathToken: nativePathToken ?? null,
            });
      const loadingDescription = [
        currentCheckpoint ? "Switching models." : null,
        extraLoadingDescription ?? null,
        isDownloaded ? "Loading cached model into memory." : null,
        !isDownloaded && isCachedLora
          ? "Loading trained model into memory."
          : null,
      ]
        .filter(Boolean)
        .join(" ");
      setModelsError(null);
      setLoadToastDismissedState(false);
      const loadInfo = {
        id: modelId,
        displayName,
        isDownloaded,
        isCachedLora,
        nativePathToken: nativePathToken ?? null,
        modelFormat,
        runtimeBackend,
        ggufVariant: ggufVariant ?? null,
        localPath,
        preferLocalCache,
        maxSeqLength: requestedMaxSeqLength,
        configKey,
      };
      setLoadingModel(loadInfo);
      useChatRuntimeStore.getState().setModelLoading(true);
      setLoadProgress(
        isDownloaded || isCachedLora
          ? { percent: null, label: null, phase: "starting" }
          : { percent: 0, label: "Preparing download", phase: "downloading" },
      );
      const abortCtrl = new AbortController();
      activeLoadAbort = abortCtrl;
      const isCurrentLoad = () => activeLoadAbort === abortCtrl;
      try {
        async function performLoad(): Promise<void> {
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          let previousWasUnloaded = false;
          const currentCheckpoint =
            useChatRuntimeStore.getState().params.checkpoint;
          const stateBeforeUnload = useChatRuntimeStore.getState();
          const shouldResetRuntimeConfig =
            !hasPerModelConfig && !reloadingSameModel;
          const trustRemoteCode = shouldResetRuntimeConfig
            ? false
            : (stateBeforeUnload.params.trustRemoteCode ?? false);
          const maxSeqLength = stateBeforeUnload.params.maxSeqLength;
          const previousCheckpointLower = previousCheckpoint?.toLowerCase();
          const previousIsGguf =
            previousModel?.isGguf === true ||
            previousVariant != null ||
            (previousCheckpointLower?.endsWith(".gguf") ?? false) ||
            (previousCheckpointLower?.startsWith("ollama-manifest:") ?? false);
          const rollbackMaxSeqLength = previousIsGguf
            ? (stateBeforeUnload.ggufContextLength ?? 0)
            : maxSeqLength;
          const hfToken = getHfToken() || null;
          const previousModelRequiresTrustRemoteCode =
            stateBeforeUnload.modelRequiresTrustRemoteCode;
          const previousActiveNativePathToken =
            stateBeforeUnload.activeNativePathToken;
          const previousLoadSource = stateBeforeUnload.activeModelLoadSource;
          try {
            // Lightweight pre-flight validation: avoid unloading a working model
            // if the new identifier is clearly invalid (e.g. bad HF id / path).
            const validateNativePathLease = nativePathToken
              ? (
                  await consumeNativePathToken(
                    nativePathToken,
                    "validate-model",
                  )
                ).nativePathLease
              : undefined;
            const validation = await validateModel({
              model_path: modelId,
              nativePathLease: validateNativePathLease,
              hf_token: hfToken,
              max_seq_length: maxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              model_format: modelFormat,
              local_files_only: localFilesOnly,
              local_path: resolvedLocalPath,
            });
            if (validation.requires_trust_remote_code && !trustRemoteCode) {
              throw new Error(getTrustRemoteCodeRequiredMessage(displayName));
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            const loadNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "load-model"))
                  .nativePathLease
              : undefined;

            if (currentCheckpoint) {
              await unloadModel({ model_path: currentCheckpoint });
              previousWasUnloaded = true;
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            if (
              currentCheckpoint &&
              !modelIdsMatch(currentCheckpoint, modelId)
            ) {
              useChatRuntimeStore.setState({
                loadedSpeculativeType: null,
                loadedSpecDraftNMax: null,
              });
            }
            // Per-model inference settings must not leak across models. An
            // applied picker config is authoritative and a same-model reload
            // keeps its current settings; every other load (native drop,
            // programmatic reload of a different model) starts from defaults
            // so model B never inherits model A's chat template, KV dtype,
            // custom context, or spec-decoding choice.
            if (shouldResetRuntimeConfig) {
              const store = useChatRuntimeStore.getState();
              useChatRuntimeStore.setState({
                speculativeType: null,
                specDraftNMax: null,
                kvCacheDtype: null,
                customContextLength: null,
                chatTemplateOverride: null,
              });
              store.setParams({
                ...store.params,
                trustRemoteCode: false,
              });
            }

            const {
              chatTemplateOverride,
              kvCacheDtype,
              customContextLength,
              ggufContextLength,
              speculativeType,
              specDraftNMax,
              activePresetSource,
              activeGgufVariant,
            } = useChatRuntimeStore.getState();
            const effectiveMaxSeqLength = resolveLoadMaxSeqLength({
              modelId,
              ggufVariant,
              modelFormat,
              customContextLength,
              ggufContextLength,
              currentCheckpoint,
              activeGgufVariant,
              maxSeqLength,
              presetSource: activePresetSource,
            });
            const effectiveChatTemplateOverride = chatTemplateOverride?.trim()
              ? chatTemplateOverride
              : null;
            const loadResponse = await loadModel({
              model_path: modelId,
              nativePathLease: loadNativePathLease,
              hf_token: hfToken,
              max_seq_length: effectiveMaxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              model_format: modelFormat,
              local_files_only: localFilesOnly,
              local_path: resolvedLocalPath,
              trust_remote_code: trustRemoteCode,
              chat_template_override: effectiveChatTemplateOverride,
              cache_type_kv: kvCacheDtype,
              speculative_type: speculativeType,
              spec_draft_n_max: specDraftNMax,
            });

            // If cancelled while loading, don't update UI to show
            // the model as active -- it's being unloaded.
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            const hydration = hydrateRuntimeFromLoadResponse({
              response: loadResponse,
              modelId,
              ggufVariant: ggufVariant ?? null,
              requestedConfig: {
                chatTemplateOverride: effectiveChatTemplateOverride,
                customContextLength,
              },
              stateBeforeLoad: stateBeforeUnload,
              reloadingSameModel,
              nativePathToken: nativePathToken ?? null,
              activeModelLoadSource,
            });
            // Qwen3/3.5/3.6: apply thinking-mode-specific params after load
            if (
              modelId.toLowerCase().includes("qwen3") &&
              hydration.supportsReasoning
            ) {
              const store = useChatRuntimeStore.getState();
              if (store.activePresetSource === "builtin-default") {
                const mid = modelId.toLowerCase();
                const needsPresencePenalty =
                  mid.includes("qwen3.5") || mid.includes("qwen3.6");
                const p = hydration.nextReasoningEnabled
                  ? {
                      temperature: 0.6,
                      topP: 0.95,
                      topK: 20,
                      minP: 0.0,
                      ...(needsPresencePenalty ? { presencePenalty: 1.5 } : {}),
                    }
                  : {
                      temperature: 0.7,
                      topP: 0.8,
                      topK: 20,
                      minP: 0.0,
                      ...(needsPresencePenalty ? { presencePenalty: 1.5 } : {}),
                    };
                store.setParams({ ...store.params, ...p });
              }
            }
            await refresh();
          } catch (error) {
            // Skip rollback if user cancelled -- model is already being unloaded.
            if (abortCtrl.signal.aborted) throw error;
            // If we unloaded a previous model and the new load failed, attempt a rollback.
            if (previousWasUnloaded && previousCheckpoint) {
              let rollbackNativePathLease: string | undefined;
              if (previousActiveNativePathToken) {
                try {
                  rollbackNativePathLease = (
                    await consumeNativePathToken(
                      previousActiveNativePathToken,
                      "load-model",
                    )
                  ).nativePathLease;
                } catch {
                  throw new Error(
                    "Could not reload the previous local model: please re-select the file.",
                  );
                }
              }
              try {
                const rollbackOptions = resolveRollbackLoadOptions({
                  previousCheckpoint,
                  previousVariant,
                  previousIsLora,
                  previousActiveNativePathToken,
                  previousLoadSource,
                });
                await loadModel({
                  model_path: previousCheckpoint,
                  nativePathLease: rollbackNativePathLease,
                  hf_token: hfToken,
                  max_seq_length: rollbackMaxSeqLength,
                  load_in_4bit: true,
                  is_lora: previousIsLora,
                  gguf_variant: previousVariant,
                  model_format: rollbackOptions.modelFormat,
                  local_files_only: rollbackOptions.localFilesOnly,
                  local_path: rollbackOptions.localPath,
                  trust_remote_code:
                    previousModelRequiresTrustRemoteCode ||
                    previousRuntimeConfigSnapshot.trustRemoteCode,
                  chat_template_override:
                    previousRuntimeConfigSnapshot.chatTemplateOverride?.trim()
                      ? previousRuntimeConfigSnapshot.chatTemplateOverride
                      : null,
                  cache_type_kv: previousRuntimeConfigSnapshot.kvCacheDtype,
                  speculative_type:
                    previousRuntimeConfigSnapshot.speculativeType,
                  spec_draft_n_max: previousRuntimeConfigSnapshot.specDraftNMax,
                });
                useChatRuntimeStore.setState({
                  activeNativePathToken: previousActiveNativePathToken ?? null,
                  activeModelLoadSource: previousLoadSource ?? null,
                });
                await refresh();
              } catch {
                // Rollback also failed; surface the original load error below.
              }
            }
            if (isCurrentLoad()) {
              restoreRuntimePerModelConfigSnapshot(
                previousRuntimeConfigSnapshot,
              );
            }
            throw error;
          }
        }

        const isCachedLoad = isDownloaded || isCachedLora;
        const toastTitle = isCachedLoad
          ? "Starting model…"
          : "Downloading model…";
        const toastId = toast(null, {
          description: renderLoadDescription(
            toastTitle,
            loadingDescription,
            isCachedLoad ? null : 0,
            isCachedLoad ? null : "Preparing download",
            cancelLoading,
          ),
          duration: Number.POSITIVE_INFINITY,
          classNames: MODEL_LOAD_TOAST_CLASSNAMES,
          onDismiss: (dismissedToast) => {
            if (activeLoadToastId !== dismissedToast.id) {
              return;
            }
            setLoadToastDismissedState(true);
          },
        });
        activeLoadToastId = toastId;

        // Poll download progress for non-cached models (GGUF and non-GGUF).
        // Then, once the download wraps (or for already-cached models),
        // poll the llama-server mmap phase so "Starting model..." no
        // longer looks frozen for several minutes on large MoE models.
        let progressInterval: ReturnType<typeof setInterval> | null = null;
        const expectedBytes =
          typeof selection !== "string" ? (selection.expectedBytes ?? 0) : 0;

        // Rolling window of byte samples for rate / ETA estimation.
        // Shared across download + mmap phases so the estimator doesn't
        // reset when the phase flips.
        type Sample = { t: number; b: number };
        const minSamples = 3;
        const minWindow = 3_000; // ms
        const maxWindow = 15_000; // ms
        const dlSamples: Sample[] = [];
        const mmapSamples: Sample[] = [];

        function estimate(
          samples: Sample[],
          bytes: number,
          total: number,
        ): { rate: number; eta: number; stable: boolean } {
          const now = Date.now();
          // Drop samples if the counter reset (e.g. phase flipped).
          if (samples.length > 0 && bytes < samples[samples.length - 1].b) {
            samples.length = 0;
          }
          samples.push({ t: now, b: bytes });
          const cutoff = now - maxWindow;
          while (samples.length > 2 && samples[0].t < cutoff) {
            samples.shift();
          }
          if (samples.length < minSamples) {
            return { rate: 0, eta: 0, stable: false };
          }
          const first = samples[0];
          const last = samples[samples.length - 1];
          const dt = (last.t - first.t) / 1000;
          const db = last.b - first.b;
          if (dt * 1000 < minWindow || db <= 0) {
            return { rate: 0, eta: 0, stable: false };
          }
          const rate = db / dt;
          const eta =
            total > 0 && bytes < total && rate > 0 ? (total - bytes) / rate : 0;
          return { rate, eta, stable: true };
        }

        function composeProgressLabel(
          dlGb: number,
          totalGb: number,
          bytes: number,
          total: number,
          samples: Sample[],
        ): string {
          const base =
            totalGb > 0
              ? `${dlGb.toFixed(1)} of ${totalGb.toFixed(1)} GB`
              : `${dlGb.toFixed(1)} GB downloaded`;
          const est = estimate(samples, bytes, total);
          if (!est.stable) return base;
          const rateStr = formatRate(est.rate);
          const etaStr = total > 0 ? formatEta(est.eta) : "";
          return etaStr && etaStr !== "--"
            ? `${base} • ${rateStr} • ${etaStr} left`
            : `${base} • ${rateStr}`;
        }

        let downloadComplete = isDownloaded || isCachedLora;

        const pollDownload = async () => {
          if (abortCtrl.signal.aborted || !getActiveLoadingModel()) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const progressHfToken = getHfToken() || null;
            const prog =
              ggufVariant && expectedBytes > 0
                ? await getGgufDownloadProgress(modelId, {
                    variant: ggufVariant,
                    expectedBytes,
                    hfToken: progressHfToken,
                    signal: abortCtrl.signal,
                  })
                : await getDownloadProgress(modelId, {
                    hfToken: progressHfToken,
                    signal: abortCtrl.signal,
                  });
            if (!getActiveLoadingModel()) return;

            if (prog.progress > 0 && prog.progress < 1) {
              hasShownProgress = true;
              const dlGb = prog.downloaded_bytes / 1024 ** 3;
              const totalGb = prog.expected_bytes / 1024 ** 3;
              const pct = Math.round(prog.progress * 100);
              const progressLabel = composeProgressLabel(
                dlGb,
                totalGb,
                prog.downloaded_bytes,
                prog.expected_bytes,
                dlSamples,
              );
              setLoadProgress({
                percent: pct,
                label: progressLabel,
                phase: "downloading",
              });
              if (activeLoadToastDismissed) return;
              toast(null, {
                id: toastId,
                description: renderLoadDescription(
                  "Downloading model…",
                  loadingDescription,
                  pct,
                  progressLabel,
                  cancelLoading,
                ),
                duration: Number.POSITIVE_INFINITY,
                classNames: MODEL_LOAD_TOAST_CLASSNAMES,
                onDismiss: (dismissedToast) => {
                  if (activeLoadToastId !== dismissedToast.id) return;
                  setLoadToastDismissedState(true);
                },
              });
            } else if (
              prog.downloaded_bytes > 0 &&
              prog.expected_bytes === 0 &&
              prog.progress === 0
            ) {
              hasShownProgress = true;
              const dlGb = prog.downloaded_bytes / 1024 ** 3;
              const est = estimate(dlSamples, prog.downloaded_bytes, 0);
              const rateSuffix = est.stable ? ` • ${formatRate(est.rate)}` : "";
              setLoadProgress({
                percent: null,
                label: `${dlGb.toFixed(1)} GB downloaded${rateSuffix}`,
                phase: "downloading",
              });
            } else if (prog.progress >= 1 && hasShownProgress) {
              downloadComplete = true;
              setLoadProgress({
                percent: 100,
                label: "Download complete",
                phase: "starting",
              });
              if (!activeLoadToastDismissed) {
                toast(null, {
                  id: toastId,
                  description: renderLoadDescription(
                    "Starting model…",
                    "Download complete. Loading the model into memory.",
                    100,
                    "Download complete",
                    cancelLoading,
                  ),
                  duration: Number.POSITIVE_INFINITY,
                  classNames: MODEL_LOAD_TOAST_CLASSNAMES,
                  onDismiss: (dismissedToast) => {
                    if (activeLoadToastId !== dismissedToast.id) return;
                    setLoadToastDismissedState(true);
                  },
                });
              }
              notifyNative({
                key: `model-downloaded:${notificationModelKey}`,
                title: "Model downloaded",
                body: `${safeModelName} finished downloading and is loading into memory.`,
                requestPermission: false,
              }).catch(() => undefined);
              // Keep polling: the mmap branch below takes over from here.
            }
          } catch {
            // Ignore polling errors; keep polling.
          }
        };

        const pollLoad = async () => {
          if (abortCtrl.signal.aborted || !getActiveLoadingModel()) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog = await getLoadProgress();
            if (!getActiveLoadingModel()) return;
            if (!prog || prog.phase == null) return;
            if (prog.phase === "ready") {
              if (progressInterval) clearInterval(progressInterval);
              return;
            }
            if (prog.bytes_total <= 0) return; // nothing useful to render
            const loadedGb = prog.bytes_loaded / 1024 ** 3;
            const totalGb = prog.bytes_total / 1024 ** 3;
            const pct = Math.min(99, Math.round(prog.fraction * 100));
            const est = estimate(
              mmapSamples,
              prog.bytes_loaded,
              prog.bytes_total,
            );
            const base = `${loadedGb.toFixed(1)} of ${totalGb.toFixed(1)} GB in memory`;
            const label = est.stable
              ? `${base} • ${formatRate(est.rate)}${
                  formatEta(est.eta) !== "--"
                    ? ` • ${formatEta(est.eta)} left`
                    : ""
                }`
              : base;
            setLoadProgress({
              percent: pct,
              label,
              phase: "starting",
            });
            if (activeLoadToastDismissed) return;
            toast(null, {
              id: toastId,
              description: renderLoadDescription(
                "Starting model…",
                "Paging weights into memory.",
                pct,
                label,
                cancelLoading,
              ),
              duration: Number.POSITIVE_INFINITY,
              classNames: MODEL_LOAD_TOAST_CLASSNAMES,
              onDismiss: (dismissedToast) => {
                if (activeLoadToastId !== dismissedToast.id) return;
                setLoadToastDismissedState(true);
              },
            });
          } catch {
            // Ignore polling errors.
          }
        };

        const pollProgress = async () => {
          if (downloadComplete) {
            await pollLoad();
          } else {
            await pollDownload();
          }
        };

        let hasShownProgress = false;
        setTimeout(pollProgress, 500);
        progressInterval = setInterval(pollProgress, 2000);

        try {
          await performLoad();
          if (activeLoadToastDismissed) {
            toast.success(`${displayName} loaded`);
          } else {
            toast.success(`${displayName} loaded`, {
              id: toastId,
              description: undefined,
              duration: 8000,
            });
          }
          notifyNative({
            key: `model-loaded:${notificationModelKey}`,
            title: "Model ready",
            body: `${safeModelName} is loaded and ready to chat.`,
            requestPermission: false,
          }).catch(() => undefined);
        } catch (err) {
          if (!abortCtrl.signal.aborted) {
            const message =
              err instanceof Error ? err.message : "Failed to load model";
            if (activeLoadToastDismissed) {
              toast.error(message);
            } else {
              toast.error(message, {
                id: toastId,
                description: undefined,
                duration: 8000,
              });
            }
            notifyNative({
              key: `model-load-failed:${notificationModelKey}`,
              title: "Model failed to load",
              body: sanitizeNotificationBody(
                message,
                "The model failed to load.",
              ),
              requestPermission: false,
            }).catch(() => undefined);
          }
          throw err;
        } finally {
          if (progressInterval) clearInterval(progressInterval);
          if (isCurrentLoad()) resetLoadingUi();
        }
      } catch (error) {
        if (abortCtrl.signal.aborted) return; // User cancelled, nothing to report
        if (isCurrentLoad()) resetLoadingUi();
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
        if (throwOnError) {
          throw error instanceof Error ? error : new Error(message);
        }
      }
    },
    [
      cancelLoading,
      loras,
      models,
      refresh,
      renderLoadDescription,
      resetLoadingUi,
      setLoadProgress,
      setLoadToastDismissedState,
      setLoadingModel,
      setModelsError,
    ],
  );

  const ejectModel = useCallback(async () => {
    if (!params.checkpoint) {
      return;
    }
    setModelsError(null);
    if (isExternalModelId(params.checkpoint)) {
      clearCheckpoint();
      await refresh();
      return;
    }
    try {
      async function performUnload(): Promise<void> {
        await unloadModel({ model_path: params.checkpoint });
        clearCheckpoint();
        await refresh();
      }

      await toast.promise(performUnload(), {
        loading: "Unloading model",
        success: { message: "Model unloaded", duration: 1200 },
        error: (err) =>
          err instanceof Error ? err.message : "Failed to unload model",
        description: "Releases VRAM and resets inference state.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to unload model";
      setModelsError(message);
    }
  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);

  return {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  };
}
