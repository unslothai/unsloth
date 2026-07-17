// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getInferenceStatus } from "../api/chat-api";
import { mergeBackendRecommendedInference } from "../presets/preset-policy";
import { clampReasoningEffortToLevels } from "../provider-capabilities";
import {
  CHAT_REASONING_ENABLED_KEY,
  type ReasoningEffort,
  type ReasoningStyle,
  loadOptionalBool,
  resolveToolsEnabledOnLoad,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import {
  type InferenceStatusResponse,
  isMultimodalResponse,
} from "../types/api";
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
  if (s === "mtp" || s === "draft-mtp") return "mtp";
  if (s === "ngram" || s === "ngram-mod" || s === "ngram-simple") {
    return "ngram";
  }
  if (s === "mtp+ngram") return "mtp+ngram";
  const parts = s
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  const hasMtp = parts.some((p) => p === "mtp" || p === "draft-mtp");
  const hasNgram = parts.some(
    (p) => p === "ngram" || p === "ngram-mod" || p === "ngram-simple",
  );
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

/**
 * Reasoning capability fields derived from a model load/status response.
 *
 * Centralises the effort-levels + can-disable derivation so every load path
 * (main load, status sync, shared/Compare composer, first-chat auto-load) agrees:
 * a hybrid GLM-style `enable_thinking_effort` model keeps its high|max|Off
 * controls no matter which path loaded it, instead of falling back to the
 * default low|medium|high and losing Max/Off.
 */
export function reasoningCapsFromLoad(resp: {
  reasoning_style?: ReasoningStyle | null;
  reasoning_effort_levels?: string[] | null;
}): {
  reasoningStyle: ReasoningStyle;
  reasoningEffortLevels: readonly ReasoningEffort[];
  supportsReasoningOff: boolean;
} {
  const reasoningStyle: ReasoningStyle =
    resp.reasoning_style ?? "enable_thinking";
  const reasoningEffortLevels: readonly ReasoningEffort[] =
    resp.reasoning_effort_levels && resp.reasoning_effort_levels.length > 0
      ? (resp.reasoning_effort_levels as ReasoningEffort[])
      : (["low", "medium", "high"] as const);
  // enable_thinking and enable_thinking_effort can both be turned off; only the
  // pure gpt-oss-style reasoning_effort is always-on.
  return {
    reasoningStyle,
    reasoningEffortLevels,
    supportsReasoningOff: reasoningStyle !== "reasoning_effort",
  };
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
  // GLM-5.2-style models report their own effort levels (e.g. high|max);
  // everything else keeps the default low/medium/high.
  const reasoningEffortLevels =
    status.reasoning_effort_levels && status.reasoning_effort_levels.length > 0
      ? (status.reasoning_effort_levels as ReasoningEffort[])
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
  const clampedReasoningEffort =
    reasoningStyle === "enable_thinking_effort" ||
    reasoningStyle === "reasoning_effort"
      ? clampReasoningEffortToLevels(
          prevState.reasoningEffort,
          reasoningEffortLevels,
        )
      : clampLocalReasoningEffort(prevState.reasoningEffort);
  const nextDefaultChatTemplate =
    status.chat_template === undefined
      ? prevState.defaultChatTemplate
      : status.chat_template;
  // While a load is in flight, performLoad owns the load params. Seeding them
  // from a stale poll here would clobber the values the load dialog just set.
  const seedLoadParams =
    !prevState.modelLoading && prevState.pendingSelection === null;
  const syncVisionProjectorDraft =
    hydratingExistingModel ||
    prevState.loadedVisionProjectorEnabled === null ||
    prevState.visionProjectorEnabled === prevState.loadedVisionProjectorEnabled;

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
    // A non-GGUF status must also drop a stale native-path token: without this the
    // isGguf OR (activeGgufVariant || activeNativePathToken || ggufContextLength)
    // stays true after switching from a native GGUF to a transformers model, so a
    // Codex-only detection would auto-select for a model its preflight rejects. A real
    // GGUF load reports is_gguf: true, so its token is preserved (the load path owns it).
    ...(status.is_gguf ? {} : { activeNativePathToken: null }),
    modelRequiresTrustRemoteCode: status.requires_trust_remote_code ?? false,
    defaultChatTemplate: nextDefaultChatTemplate,
    loadedIsMultimodal: isMultimodalResponse(status),
    loadedIsDiffusion: status.is_diffusion ?? false,
    specFallbackReason: status.spec_fallback_reason ?? null,
    ...(seedLoadParams &&
      prevState.loadedSpeculativeType === null && {
        speculativeType: currentSpecType,
        loadedSpeculativeType: currentSpecType,
      }),
    ...(seedLoadParams &&
      status.spec_draft_n_max !== undefined &&
      prevState.loadedSpecDraftNMax === null &&
      prevState.specDraftNMax === null && {
        specDraftNMax: status.spec_draft_n_max ?? null,
        loadedSpecDraftNMax: status.spec_draft_n_max ?? null,
      }),
    ...(seedLoadParams &&
      status.cache_type_kv !== undefined &&
      prevState.loadedKvCacheDtype === null && {
        kvCacheDtype: status.cache_type_kv,
        loadedKvCacheDtype: status.cache_type_kv,
      }),
    ...(seedLoadParams &&
      status.tensor_parallel !== undefined &&
      prevState.loadedTensorParallel === null && {
        tensorParallel: status.tensor_parallel,
        loadedTensorParallel: status.tensor_parallel,
      }),
    ...(seedLoadParams &&
      status.load_mmproj !== undefined && {
        loadedVisionProjectorEnabled: status.load_mmproj,
        ...(syncVisionProjectorDraft && {
          visionProjectorEnabled: status.load_mmproj,
        }),
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
      if (sizeMatch && Number.parseFloat(sizeMatch[1]) < 9) {
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

  let status: InferenceStatusResponse;
  try {
    status = await getInferenceStatus();
  } catch {
    // Status endpoint unavailable: fall back to the normal auto-load path.
    return false;
  }
  if (!status.active_model) {
    return false;
  }

  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) {
    return false;
  }

  // Re-check after the await: keep a checkpoint the user picked meanwhile.
  const previousCheckpoint = useChatRuntimeStore.getState().params.checkpoint;
  if (previousCheckpoint) {
    return true;
  }
  store.setCheckpoint(checkpointId, status.gguf_variant);
  applyActiveModelStatusToStore(status, { previousCheckpoint });
  return true;
}
