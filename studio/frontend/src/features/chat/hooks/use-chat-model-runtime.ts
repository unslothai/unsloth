// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";
import { toast } from "sonner";
import {
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
};

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
  return {
    ...current,
    checkpoint: modelId,
    temperature:
      toFiniteNumber(inference?.temperature) ?? current.temperature,
    topP: toFiniteNumber(inference?.top_p) ?? current.topP,
    topK: toFiniteNumber(inference?.top_k) ?? current.topK,
    minP: toFiniteNumber(inference?.min_p) ?? current.minP,
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
  } | null>(null);

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
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load models";
      setModelsError(message);
      toast.error("Failed to refresh models", {
        description: message,
      });
    }
  }, [setCheckpoint, setLoras, setModels, setModelsError]);

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      const ggufVariant =
        typeof selection === "string" ? undefined : selection.ggufVariant;
      const currentVariant = useChatRuntimeStore.getState().activeGgufVariant;
      if (!modelId || (params.checkpoint === modelId && (ggufVariant ?? null) === (currentVariant ?? null))) {
        return;
      }

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const extraLoadingDescription =
        typeof selection === "string" ? undefined : selection.loadingDescription;
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
        currentCheckpoint ? "Unloading previous model first." : null,
        extraLoadingDescription ?? null,
        "This may include downloading. Large models can take a while.",
      ]
        .filter(Boolean)
        .join(" ");

      setModelsError(null);
      setLoadingModel({ id: modelId, displayName });
      try {
        async function performLoad(): Promise<void> {
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

            const loadResponse = await loadModel({
              model_path: modelId,
              hf_token: null,
              max_seq_length: maxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              trust_remote_code: paramsBeforeLoad.trustRemoteCode ?? false,
            });

            const currentParams = useChatRuntimeStore.getState().params;
            setParams(
              mergeRecommendedInference(currentParams, loadResponse, modelId),
            );
            await refresh();
          } catch (error) {
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

        const loadPromise = performLoad().finally(() => {
          setLoadingModel(null);
        });

        await toast.promise(loadPromise, {
          loading: "Loading model…",
          success: `${displayName} loaded`,
          error: (err) =>
            err instanceof Error ? err.message : "Failed to load model",
          description: loadingDescription,
        });
      } catch (error) {
        setLoadingModel(null);
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
      }
    },
    [loras, models, params.checkpoint, refresh, setModelsError, setParams],
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
    loadingModel,
  };
}
