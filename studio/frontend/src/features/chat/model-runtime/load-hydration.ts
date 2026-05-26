// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { mergeBackendRecommendedInference } from "../presets/preset-policy";
import {
  type ReasoningEffort,
  type ReasoningStyle,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import type { ActiveModelLoadSource } from "./rollback-load-source";
import { normalizeSpeculativeType } from "./speculative-type.ts";
import { isMultimodalResponse, type LoadModelResponse } from "../types/api";
import { smallQwenThinkingOffByDefault } from "../utils/qwen-params";

export { normalizeSpeculativeType } from "./speculative-type.ts";

export type RuntimeLoadRequestConfig = {
  chatTemplateOverride: string | null;
  customContextLength: number | null;
};

export type RuntimeLoadStateSnapshot = Pick<
  ReturnType<typeof useChatRuntimeStore.getState>,
  | "reasoningEnabled"
  | "toolsEnabled"
  | "codeToolsEnabled"
  | "reasoningEffort"
>;

export type RuntimeLoadHydrationResult = {
  supportsReasoning: boolean;
  nextReasoningEnabled: boolean;
};

type LocalReasoningEffort = Extract<ReasoningEffort, "low" | "medium" | "high">;

export function clampLocalReasoningEffort(
  value: ReasoningEffort,
): LocalReasoningEffort {
  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }
  return "low";
}

export function hydrateRuntimeFromLoadResponse({
  response,
  modelId,
  ggufVariant,
  requestedConfig,
  stateBeforeLoad,
  reloadingSameModel,
  nativePathToken,
  activeModelLoadSource,
}: {
  response: LoadModelResponse;
  modelId: string;
  ggufVariant?: string | null;
  requestedConfig: RuntimeLoadRequestConfig;
  stateBeforeLoad?: RuntimeLoadStateSnapshot;
  reloadingSameModel: boolean;
  nativePathToken?: string | null;
  activeModelLoadSource?: ActiveModelLoadSource | null;
}): RuntimeLoadHydrationResult {
  const storeBeforeParams = useChatRuntimeStore.getState();
  const previous = stateBeforeLoad ?? storeBeforeParams;
  storeBeforeParams.setParams(
    mergeBackendRecommendedInference({
      current: storeBeforeParams.params,
      response,
      modelId,
      presetSource: storeBeforeParams.activePresetSource,
    }),
  );

  const reasoningDefault =
    (response.supports_reasoning ?? false) &&
    !smallQwenThinkingOffByDefault(modelId);
  const loadedKv = response.cache_type_kv ?? null;
  const loadedSpec = normalizeSpeculativeType(response.speculative_type);
  const nativeCtx = response.is_gguf ? (response.context_length ?? 131072) : null;
  const ggufMaxContextLength = response.is_gguf
    ? (response.max_context_length ?? null)
    : null;
  const ggufNativeContextLength = response.is_gguf
    ? (response.native_context_length ?? null)
    : null;
  const loadedCustomContextLength =
    response.is_gguf &&
    requestedConfig.customContextLength != null &&
    (ggufNativeContextLength == null || nativeCtx !== ggufNativeContextLength)
      ? nativeCtx
      : null;
  const reasoningAlwaysOn = response.reasoning_always_on ?? false;
  const reasoningStyle: ReasoningStyle =
    response.reasoning_style ?? "enable_thinking";
  const supportsReasoning = response.supports_reasoning ?? false;
  const supportsTools = response.supports_tools ?? false;
  const appliedChatTemplateOverride =
    response.chat_template_override !== undefined
      ? (response.chat_template_override ?? null)
      : requestedConfig.chatTemplateOverride;
  const reasoningEffortLevels =
    reasoningStyle === "reasoning_effort"
      ? (["low", "medium", "high"] as const)
      : (["low", "medium", "high"] as const);
  const clampedReasoningEffort = clampLocalReasoningEffort(
    previous.reasoningEffort,
  );
  const nextReasoningEnabled = reasoningAlwaysOn
    ? true
    : reloadingSameModel && supportsReasoning
      ? previous.reasoningEnabled
      : reasoningDefault;

  useChatRuntimeStore.setState({
    activeGgufVariant: response.is_gguf ? (ggufVariant ?? null) : null,
    ggufContextLength: nativeCtx,
    ggufMaxContextLength,
    ggufNativeContextLength,
    modelRequiresTrustRemoteCode:
      response.requires_trust_remote_code ?? false,
    supportsReasoning,
    reasoningAlwaysOn,
    reasoningEnabled: nextReasoningEnabled,
    reasoningStyle,
    supportsReasoningOff: reasoningStyle !== "reasoning_effort",
    reasoningEffortLevels,
    reasoningEffort: clampedReasoningEffort,
    supportsPreserveThinking: response.supports_preserve_thinking ?? false,
    supportsTools,
    toolsEnabled:
      reloadingSameModel && supportsTools ? previous.toolsEnabled : supportsTools,
    codeToolsEnabled:
      reloadingSameModel && supportsTools
        ? previous.codeToolsEnabled
        : supportsTools,
    kvCacheDtype: loadedKv,
    loadedKvCacheDtype: loadedKv,
    speculativeType: loadedSpec,
    loadedSpeculativeType: loadedSpec,
    specDraftNMax: response.spec_draft_n_max ?? null,
    loadedSpecDraftNMax: response.spec_draft_n_max ?? null,
    customContextLength: loadedCustomContextLength,
    loadedCustomContextLength,
    defaultChatTemplate: response.chat_template ?? null,
    chatTemplateOverride: appliedChatTemplateOverride,
    loadedChatTemplateOverride: appliedChatTemplateOverride,
    loadedIsMultimodal: isMultimodalResponse(response),
    activeNativePathToken: nativePathToken ?? null,
    activeModelLoadSource: activeModelLoadSource ?? null,
  });

  return { supportsReasoning, nextReasoningEnabled };
}
