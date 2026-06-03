// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getInferenceStatus } from "../api/chat-api";
import { mergeBackendRecommendedInference } from "../presets/preset-policy";
import {
  CHAT_REASONING_ENABLED_KEY,
  loadOptionalBool,
  type ReasoningEffort,
  resolveToolsEnabledOnLoad,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import { isMultimodalResponse, type InferenceStatusResponse } from "../types/api";
import type { ChatModelSummary } from "../types/runtime";

type LocalReasoningEffort = Extract<ReasoningEffort, "low" | "medium" | "high">;

// Canonicalises backend / persisted speculative mode values onto the UI modes.
export function normalizeSpeculativeType(
  v: string | null | undefined,
): string | null {
  if (v == null) return null;
  const s = String(v).trim().toLowerCase();
  if (!s) return null;
  if (s === "auto" || s === "default") return "auto";
  if (s === "off") return "off";
  if (s === "ngram-simple") return "ngram-simple";
  if (s === "mtp" || s === "draft-mtp") return "mtp";
  if (s === "ngram" || s === "ngram-mod") return "ngram";
  if (s === "mtp+ngram") return "mtp+ngram";
  const parts = s.split(",").map((p) => p.trim()).filter(Boolean);
  const hasMtp = parts.some((p) => p === "mtp" || p === "draft-mtp");
  const hasNgram = parts.some((p) => p === "ngram" || p === "ngram-mod");
  if (hasMtp && hasNgram) return "mtp+ngram";
  if (hasMtp) return "mtp";
  if (hasNgram) return "ngram";
  return "auto";
}

export function clampLocalReasoningEffort(
  value: ReasoningEffort,
): LocalReasoningEffort {
  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }
  return "low";
}

export function resolveInferenceCheckpointId(
  status: InferenceStatusResponse,
): string | null {
  if (!status.active_model) return null;
  return status.model_identifier ?? status.active_model;
}

function ensureActiveModelInStoreList(
  status: InferenceStatusResponse,
  checkpointId: string,
): void {
  const store = useChatRuntimeStore.getState();
  if (store.models.some((model) => model.id === checkpointId)) {
    return;
  }
  const summary: ChatModelSummary = {
    id: checkpointId,
    name: status.active_model ?? checkpointId,
    isVision: status.is_vision ?? false,
    isLora: false,
    isGguf: status.is_gguf ?? false,
    isAudio: status.is_audio ?? false,
    audioType: status.audio_type ?? null,
    hasAudioInput: status.has_audio_input ?? false,
  };
  store.setModels([...store.models, summary]);
}

export type ApplyInferenceStatusOptions = {
  previousCheckpoint?: string;
};

/** Mirror refresh() hydration so adopted CLI models get reasoning/tools flags. */
export function applyActiveModelStatusToStore(
  status: InferenceStatusResponse,
  options: ApplyInferenceStatusOptions = {},
): void {
  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) return;

  const store = useChatRuntimeStore.getState();
  const previousCheckpoint =
    options.previousCheckpoint ?? store.params.checkpoint;

  if (status.inference) {
    store.setParams(
      mergeBackendRecommendedInference({
        current: store.params,
        response: status,
        modelId: checkpointId,
        presetSource: store.activePresetSource,
      }),
    );
  }

  const hydratingExistingModel =
    previousCheckpoint !== checkpointId ||
    store.activeGgufVariant !== (status.gguf_variant ?? null);
  const supportsReasoning = status.supports_reasoning ?? false;
  const reasoningAlwaysOn = status.reasoning_always_on ?? false;
  const reasoningStyle = status.reasoning_style ?? "enable_thinking";
  const reasoningEffortLevels =
    reasoningStyle === "reasoning_effort"
      ? (["low", "medium", "high"] as const)
      : (["low", "medium", "high"] as const);
  const supportsPreserveThinking = status.supports_preserve_thinking ?? false;
  const supportsTools = status.supports_tools ?? false;
  const storedReasoningEnabled = loadOptionalBool(CHAT_REASONING_ENABLED_KEY);
  const currentGgufContextLength = status.is_gguf
    ? (status.context_length ?? null)
    : null;
  const ggufMaxContextLength = status.is_gguf
    ? (status.max_context_length ?? null)
    : null;
  const ggufNativeContextLength = status.is_gguf
    ? (status.native_context_length ?? null)
    : null;
  const currentSpecType = normalizeSpeculativeType(status.speculative_type);
  const prevState = useChatRuntimeStore.getState();
  const clampedReasoningEffort = clampLocalReasoningEffort(
    prevState.reasoningEffort,
  );
  const nextDefaultChatTemplate =
    status.chat_template === undefined
      ? prevState.defaultChatTemplate
      : status.chat_template;

  useChatRuntimeStore.setState({
    supportsReasoning,
    reasoningAlwaysOn,
    reasoningStyle,
    supportsReasoningOff: reasoningStyle !== "reasoning_effort",
    reasoningEffortLevels,
    reasoningEffort: clampedReasoningEffort,
    supportsPreserveThinking,
    supportsTools,
    ...resolveToolsEnabledOnLoad(supportsTools),
    reasoningEnabled: supportsReasoning
      ? reasoningStyle === "reasoning_effort"
        ? true
        : useChatRuntimeStore.getState().reasoningEnabled
      : true,
    ggufContextLength: currentGgufContextLength,
    ggufMaxContextLength,
    ggufNativeContextLength,
    modelRequiresTrustRemoteCode: status.requires_trust_remote_code ?? false,
    defaultChatTemplate: nextDefaultChatTemplate,
    loadedIsMultimodal: isMultimodalResponse(status),
    ...(prevState.loadedSpeculativeType === null && {
      speculativeType: currentSpecType,
      loadedSpeculativeType: currentSpecType,
    }),
    ...(status.spec_draft_n_max !== undefined &&
      prevState.loadedSpecDraftNMax === null &&
      prevState.specDraftNMax === null && {
        specDraftNMax: status.spec_draft_n_max ?? null,
        loadedSpecDraftNMax: status.spec_draft_n_max ?? null,
      }),
    ...(status.cache_type_kv !== undefined &&
      prevState.loadedKvCacheDtype === null && {
        kvCacheDtype: status.cache_type_kv,
        loadedKvCacheDtype: status.cache_type_kv,
      }),
    ...(status.chat_template_override !== undefined &&
      prevState.loadedChatTemplateOverride === null &&
      prevState.chatTemplateOverride === null && {
        chatTemplateOverride: status.chat_template_override,
        loadedChatTemplateOverride: status.chat_template_override,
      }),
  });

  ensureActiveModelInStoreList(status, checkpointId);

  if (
    supportsReasoning &&
    hydratingExistingModel &&
    storedReasoningEnabled === null
  ) {
    let reasoningDefault = true;
    const mid = checkpointId.toLowerCase();
    if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
      const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
      if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
        reasoningDefault = false;
      }
    }
    useChatRuntimeStore.setState({ reasoningEnabled: reasoningDefault });
  }
}

/**
 * Adopt the model already loaded on the inference server (e.g. via
 * ``unsloth studio run -m``) into the chat UI checkpoint without
 * triggering a new /api/inference/load.
 */
export async function tryAdoptServerActiveModel(): Promise<boolean> {
  const store = useChatRuntimeStore.getState();
  if (store.params.checkpoint) {
    return true;
  }

  const status = await getInferenceStatus();
  if (!status.active_model) {
    return false;
  }

  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) {
    return false;
  }

  const previousCheckpoint = store.params.checkpoint;
  store.setCheckpoint(checkpointId, status.gguf_variant);
  applyActiveModelStatusToStore(status, { previousCheckpoint });
  return true;
}
