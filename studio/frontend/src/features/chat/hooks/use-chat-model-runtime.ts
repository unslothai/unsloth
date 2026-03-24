// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createElement, useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { ModelLoadDescription } from "../components/model-load-status";
import {
  getDownloadProgress,
  getGgufDownloadProgress,
  getInferenceStatus,
  listLoras,
  listModels,
  loadModel,
  unloadModel,
  validateModel,
} from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { LoadModelResponse } from "../types/api";
import type {
  ChatLoraSummary,
  ChatModelSummary,
  InferenceParams,
} from "../types/runtime";

type SelectedModelInput = {
  id: string;
  isLora?: boolean;
  ggufVariant?: string;
  loadingDescription?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  forceReload?: boolean;
};

const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "items-start gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

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
  const cleaned = input.replace(LORA_SUFFIX_RE, "").replace(/[_-]+$/, "").trim();
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
  if (!model.is_lora && !model.is_vision && !model.is_gguf && !model.is_audio && !model.has_audio_input)
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
}): ChatLoraSummary {
  const idTail = lora.adapter_path.split("/").filter(Boolean).at(-1) ?? "";
  const updatedAt =
    parseTrailingEpoch(lora.display_name) ?? parseTrailingEpoch(idTail);

  return {
    id: lora.adapter_path,
    name: stripTrailingEpoch(lora.display_name),
    baseModel: lora.base_model || "Unknown base model",
    updatedAt,
    source: lora.source ?? undefined,
    exportType: lora.export_type ?? undefined,
  };
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

function mergeRecommendedInference(
  current: InferenceParams,
  response: LoadModelResponse,
  modelId: string,
): InferenceParams {
  const inference = response.inference;
  // GGUF: use actual context length from GGUF metadata, fallback to 131072
  // Non-GGUF: 4096
  const defaultMaxTokens = response.is_gguf
    ? (response.context_length ?? 131072)
    : 4096;
  return {
    ...current,
    checkpoint: modelId,
    maxTokens: defaultMaxTokens,
    temperature:
      toFiniteNumber(inference?.temperature) ?? current.temperature,
    topP: toFiniteNumber(inference?.top_p) ?? current.topP,
    topK: toFiniteNumber(inference?.top_k) ?? current.topK,
    minP: toFiniteNumber(inference?.min_p) ?? current.minP,
    presencePenalty:
      toFiniteNumber(inference?.presence_penalty) ?? current.presencePenalty,
    trustRemoteCode:
      typeof inference?.trust_remote_code === "boolean"
        ? inference.trust_remote_code
        : current.trustRemoteCode,
  };
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

  const [loadingModel, setLoadingModel] = useState<{
    id: string;
    displayName: string;
    isDownloaded?: boolean;
  } | null>(null);
  const [loadToastDismissed, setLoadToastDismissed] = useState(false);
  const [loadProgress, setLoadProgress] = useState<{
    percent: number | null;
    label: string | null;
    phase: "downloading" | "starting";
  } | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  const loadingModelRef = useRef<typeof loadingModel>(null);
  const loadToastIdRef = useRef<string | number | null>(null);
  const loadToastDismissedRef = useRef(false);

  const setLoadToastDismissedState = useCallback((dismissed: boolean) => {
    loadToastDismissedRef.current = dismissed;
    setLoadToastDismissed(dismissed);
  }, []);

  const resetLoadingUi = useCallback(() => {
    setLoadingModel(null);
    setLoadProgress(null);
    loadingModelRef.current = null;
    loadAbortRef.current = null;
    loadToastIdRef.current = null;
    setLoadToastDismissedState(false);
    useChatRuntimeStore.getState().setModelLoading(false);
  }, [setLoadToastDismissedState]);

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

      if (statusRes.active_model) {
        setCheckpoint(statusRes.active_model, statusRes.gguf_variant);

        // Apply inference defaults on reconnect (page refresh with model already loaded)
        if (statusRes.inference) {
          const currentParams = useChatRuntimeStore.getState().params;
          setParams(
            mergeRecommendedInference(currentParams, statusRes as any, statusRes.active_model),
          );
        }

        // Restore reasoning/tools support flags and context length
        const supportsReasoning = statusRes.supports_reasoning ?? false;
        const supportsTools = statusRes.supports_tools ?? false;
        useChatRuntimeStore.setState({
          supportsReasoning,
          supportsTools,
          ggufContextLength: statusRes.is_gguf ? (statusRes.context_length ?? null) : null,
        });

        // Set reasoning default for Qwen3.5 small models
        if (supportsReasoning) {
          let reasoningDefault = true;
          const mid = statusRes.active_model.toLowerCase();
          if (mid.includes("qwen3.5")) {
            const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
            if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
              reasoningDefault = false;
            }
          }
          useChatRuntimeStore.getState().setReasoningEnabled(reasoningDefault);
        }
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
    const model = loadingModelRef.current;
    if (!model) return;
    loadAbortRef.current?.abort();
    loadAbortRef.current = null;
    loadingModelRef.current = null;
    const tid = loadToastIdRef.current;
    loadToastIdRef.current = null;
    setLoadingModel(null);
    setLoadProgress(null);
    setLoadToastDismissedState(false);
    clearCheckpoint();
    if (tid != null) toast.dismiss(tid);
    toast.info("Stopped loading model", {
      description: "The current download may still finish in the background.",
    });
    // Fire-and-forget: tell backend to stop, don't block UI
    unloadModel({ model_path: model.id }).catch(() => {});
  }, [clearCheckpoint, setLoadToastDismissedState]);

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      const ggufVariant =
        typeof selection === "string" ? undefined : selection.ggufVariant;
      const forceReload =
        typeof selection === "string" ? false : selection.forceReload ?? false;
      const currentVariant = useChatRuntimeStore.getState().activeGgufVariant;
      if (!forceReload && (!modelId || (params.checkpoint === modelId && (ggufVariant ?? null) === (currentVariant ?? null)))) {
        return;
      }
      // Prevent duplicate loads if already loading this model
      if (loadingModelRef.current?.id === modelId) return;

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const extraLoadingDescription =
        typeof selection === "string" ? undefined : selection.loadingDescription;
      const isDownloaded =
        typeof selection === "string" ? false : selection.isDownloaded ?? false;
      const model = models.find((entry) => entry.id === modelId);
      const lora = loras.find((entry) => entry.id === modelId);
      const isLora =
        explicitIsLora ?? model?.isLora ?? (lora ? true : false);
      const displayName = model?.name || lora?.name || modelId;
      const currentCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const previousCheckpoint = currentCheckpoint;
      const previousVariant =
        useChatRuntimeStore.getState().activeGgufVariant ?? null;
      const previousModel = previousCheckpoint
        ? models.find((entry) => entry.id === previousCheckpoint)
        : undefined;
      const previousLora = previousCheckpoint
        ? loras.find((entry) => entry.id === previousCheckpoint)
        : undefined;
      const previousIsLora =
        previousModel?.isLora ?? (previousLora ? true : false);
      const loadingDescription = [
        currentCheckpoint ? "Switching models." : null,
        extraLoadingDescription ?? null,
        isDownloaded ? "Loading cached model into memory." : null,
      ]
        .filter(Boolean)
        .join(" ");
      setModelsError(null);
      setLoadToastDismissedState(false);
      const loadInfo = { id: modelId, displayName, isDownloaded };
      setLoadingModel(loadInfo);
      useChatRuntimeStore.getState().setModelLoading(true);
      setLoadProgress(
        isDownloaded
          ? { percent: null, label: null, phase: "starting" }
          : { percent: 0, label: "Preparing download", phase: "downloading" },
      );
      loadingModelRef.current = loadInfo;
      const abortCtrl = new AbortController();
      loadAbortRef.current = abortCtrl;
      try {
        async function performLoad(): Promise<void> {
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          let previousWasUnloaded = false;
          const currentCheckpoint =
            useChatRuntimeStore.getState().params.checkpoint;
          const paramsBeforeLoad = useChatRuntimeStore.getState().params;
          const maxSeqLength = paramsBeforeLoad.maxSeqLength;
          try {
            // Lightweight pre-flight validation: avoid unloading a working model
            // if the new identifier is clearly invalid (e.g. bad HF id / path).
            await validateModel({
              model_path: modelId,
              hf_token: null,
              max_seq_length: maxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
            });

            if (currentCheckpoint) {
              await unloadModel({ model_path: currentCheckpoint });
              previousWasUnloaded = true;
            }

            const { chatTemplateOverride, kvCacheDtype } = useChatRuntimeStore.getState();
            const loadResponse = await loadModel({
              model_path: modelId,
              hf_token: null,
              max_seq_length: maxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              trust_remote_code: paramsBeforeLoad.trustRemoteCode ?? false,
              chat_template_override: chatTemplateOverride,
              cache_type_kv: kvCacheDtype,
            });

            // If cancelled while loading, don't update UI to show
            // the model as active -- it's being unloaded.
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            const currentParams = useChatRuntimeStore.getState().params;
            setParams(
              mergeRecommendedInference(currentParams, loadResponse, modelId),
            );
            // Qwen3.5 small models (0.8B, 2B, 4B, 9B) disable thinking by default
            let reasoningDefault = loadResponse.supports_reasoning ?? false;
            if (reasoningDefault) {
              const mid = modelId.toLowerCase();
              if (mid.includes("qwen3.5")) {
                const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
                if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
                  reasoningDefault = false;
                }
              }
            }
            useChatRuntimeStore.setState({
              ggufContextLength: loadResponse.is_gguf
                ? (loadResponse.context_length ?? 131072)
                : null,
              supportsReasoning: loadResponse.supports_reasoning ?? false,
              reasoningEnabled: reasoningDefault,
              supportsTools: loadResponse.supports_tools ?? false,
              toolsEnabled: false,
              kvCacheDtype: loadResponse.cache_type_kv ?? null,
              defaultChatTemplate: loadResponse.chat_template ?? null,
              chatTemplateOverride: null,
            });
            // Qwen3/3.5: apply thinking-mode-specific params after load
            if (modelId.toLowerCase().includes("qwen3") && (loadResponse.supports_reasoning ?? false)) {
              const store = useChatRuntimeStore.getState();
              const p = reasoningDefault
                ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
                : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
              store.setParams({ ...store.params, ...p });
            }
            await refresh();
          } catch (error) {
            // Skip rollback if user cancelled -- model is already being unloaded.
            if (abortCtrl.signal.aborted) throw error;
            // If we unloaded a previous model and the new load failed, attempt a rollback.
            if (previousWasUnloaded && previousCheckpoint) {
              try {
                await loadModel({
                  model_path: previousCheckpoint,
                  hf_token: null,
                  max_seq_length: maxSeqLength,
                  load_in_4bit: true,
                  is_lora: previousIsLora,
                  gguf_variant: previousVariant,
                });
                await refresh();
              } catch {
                // If rollback also fails, surface the original error.
              }
            }
            throw error;
          }
        }

        const toastTitle = isDownloaded ? "Starting model…" : "Downloading model…";
        const toastId = toast(
          null,
          {
            description: renderLoadDescription(
              toastTitle,
              loadingDescription,
              isDownloaded ? null : 0,
              isDownloaded ? null : "Preparing download",
              cancelLoading,
            ),
            duration: Infinity,
            closeButton: false,
            classNames: MODEL_LOAD_TOAST_CLASSNAMES,
            onDismiss: (dismissedToast) => {
              if (loadToastIdRef.current !== dismissedToast.id) {
                return;
              }
              setLoadToastDismissedState(true);
            },
          },
        );
        loadToastIdRef.current = toastId;

        // Poll download progress for non-cached models (GGUF and non-GGUF)
        let progressInterval: ReturnType<typeof setInterval> | null = null;
        if (!isDownloaded) {
          const expectedBytes =
            typeof selection !== "string" ? selection.expectedBytes ?? 0 : 0;
          let hasShownProgress = false;

          const pollProgress = async () => {
            if (abortCtrl.signal.aborted || !loadingModelRef.current) {
              if (progressInterval) clearInterval(progressInterval);
              return;
            }
            try {
              const prog = ggufVariant && expectedBytes > 0
                ? await getGgufDownloadProgress(modelId, ggufVariant, expectedBytes)
                : await getDownloadProgress(modelId);

              if (!loadingModelRef.current) return;

              if (prog.progress > 0 && prog.progress < 1) {
                hasShownProgress = true;
                const dlGb = prog.downloaded_bytes / (1024 ** 3);
                const totalGb = prog.expected_bytes / (1024 ** 3);
                const pct = Math.round(prog.progress * 100);
                const progressLabel = totalGb > 0
                  ? `${dlGb.toFixed(1)} of ${totalGb.toFixed(1)} GB`
                  : `${dlGb.toFixed(1)} GB downloaded`;
                setLoadProgress({
                  percent: pct,
                  label: progressLabel,
                  phase: "downloading",
                });
                if (loadToastDismissedRef.current) return;
                toast(
                  null,
                  {
                    id: toastId,
                    description: renderLoadDescription(
                      "Downloading model…",
                      loadingDescription,
                      pct,
                      progressLabel,
                      cancelLoading,
                    ),
                    duration: Infinity,
                    closeButton: false,
                    classNames: MODEL_LOAD_TOAST_CLASSNAMES,
                    onDismiss: (dismissedToast) => {
                      if (loadToastIdRef.current !== dismissedToast.id) return;
                      setLoadToastDismissedState(true);
                    },
                  },
                );
              } else if (prog.downloaded_bytes > 0 && prog.expected_bytes === 0 && prog.progress === 0) {
                hasShownProgress = true;
                const dlGb = prog.downloaded_bytes / (1024 ** 3);
                setLoadProgress({
                  percent: null,
                  label: `${dlGb.toFixed(1)} GB downloaded`,
                  phase: "downloading",
                });
              } else if (prog.progress >= 1 && hasShownProgress) {
                setLoadProgress({
                  percent: 100,
                  label: "Download complete",
                  phase: "starting",
                });
                if (loadToastDismissedRef.current) {
                  if (progressInterval) clearInterval(progressInterval);
                  return;
                }
                toast(null, {
                  id: toastId,
                  description: renderLoadDescription(
                    "Starting model…",
                    "Download complete. Loading the model into memory.",
                    100,
                    "Download complete",
                    cancelLoading,
                  ),
                  duration: Infinity,
                  closeButton: false,
                  classNames: MODEL_LOAD_TOAST_CLASSNAMES,
                  onDismiss: (dismissedToast) => {
                    if (loadToastIdRef.current !== dismissedToast.id) return;
                    setLoadToastDismissedState(true);
                  },
                });
                if (progressInterval) clearInterval(progressInterval);
              }
            } catch {
              // Ignore polling errors
            }
          };

          setTimeout(pollProgress, 500);
          progressInterval = setInterval(pollProgress, 2000);
        }

        try {
          await performLoad();
          if (loadToastDismissedRef.current) {
            toast.success(`${displayName} loaded`);
          } else {
            toast.success(`${displayName} loaded`, {
              id: toastId,
              description: undefined,
              closeButton: false,
              duration: 2000,
            });
          }
        } catch (err) {
          if (!abortCtrl.signal.aborted) {
            const message =
              err instanceof Error ? err.message : "Failed to load model";
            if (loadToastDismissedRef.current) {
              toast.error(message);
            } else {
              toast.error(message, {
                id: toastId,
                description: undefined,
                closeButton: false,
                duration: 5000,
              });
            }
          }
          throw err;
        } finally {
          if (progressInterval) clearInterval(progressInterval);
          resetLoadingUi();
        }
      } catch (error) {
        if (abortCtrl.signal.aborted) return; // User cancelled, nothing to report
        resetLoadingUi();
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
      }
    },
    [
      cancelLoading,
      loras,
      models,
      params.checkpoint,
      refresh,
      renderLoadDescription,
      resetLoadingUi,
      setLoadToastDismissedState,
      setModelsError,
      setParams,
    ],
  );

  const ejectModel = useCallback(async () => {
    if (!params.checkpoint) {
      return;
    }
    setModelsError(null);
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
