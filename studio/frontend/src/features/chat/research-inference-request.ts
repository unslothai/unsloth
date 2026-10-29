// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max";


export interface ResearchInferenceRequest {
  model: string;
  providerId?: string;
  providerType?: string;
  externalModel?: string;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  maxOutputTokens?: number;
  maxOutputTokensFromSavedCap?: boolean;
  maxOutputTokensPublished?: number;
  enableThinking?: boolean;
  reasoningEffort?: string;
  supportsReasoning?: boolean;
  supportsReasoningOff?: boolean;
}

export function buildResearchInferenceRequest(input: {
  checkpoint: string;
  external?: {
    providerId: string;
    providerType: string;
    modelId: string;
    maxOutputTokens: number | null;
    maxOutputTokensFromSavedCap: boolean;
    maxOutputTokensPublished: number | null;
    supportsReasoning?: boolean;
    supportsReasoningOff?: boolean;
  };
  temperature: number;
  topP: number;
  maxTokens: number;
  reasoningRequested: boolean;
  reasoningStyle: string;
  reasoningEffort: ReasoningEffort;
  reasoningEffortLevels: readonly ReasoningEffort[];
  clampReasoningEffort: (
    effort: ReasoningEffort,
    levels: readonly ReasoningEffort[],
  ) => ReasoningEffort;
}): ResearchInferenceRequest {
  const model = input.external?.modelId ?? input.checkpoint;
  const request: ResearchInferenceRequest = {
    model,
    ...(input.external
      ? {
          providerId: input.external.providerId,
          providerType: input.external.providerType,
          externalModel: input.external.modelId,
          ...(typeof input.external.supportsReasoning === "boolean"
            ? { supportsReasoning: input.external.supportsReasoning }
            : {}),
          ...(typeof input.external.supportsReasoningOff === "boolean"
            ? { supportsReasoningOff: input.external.supportsReasoningOff }
            : {}),
          ...(input.external.maxOutputTokens != null &&
          Number.isFinite(input.external.maxOutputTokens) &&
          input.external.maxOutputTokens > 0
            ? {
                maxOutputTokens: Math.floor(input.external.maxOutputTokens),
                maxOutputTokensFromSavedCap: input.external.maxOutputTokensFromSavedCap,
                // The ceiling includes the override, so it cannot say whether the model stops there.
                ...(input.external.maxOutputTokensPublished != null &&
                Number.isFinite(input.external.maxOutputTokensPublished) &&
                input.external.maxOutputTokensPublished > 0
                  ? {
                      maxOutputTokensPublished: Math.floor(
                        input.external.maxOutputTokensPublished,
                      ),
                    }
                  : {}),
              }
            : {}),
        }
      : {}),
  };
  if (Number.isFinite(input.temperature) && input.temperature >= 0 && input.temperature <= 2) {
    request.temperature = input.temperature;
  }
  // Connections omit Off (1), as chat requests do.
  if (
    Number.isFinite(input.topP) &&
    input.topP > 0 &&
    (input.external ? input.topP < 1 : input.topP <= 1)
  ) {
    request.topP = input.topP;
  }
  if (Number.isFinite(input.maxTokens) && input.maxTokens > 0) {
    request.maxTokens = Math.min(8192, Math.floor(input.maxTokens));
  }
  if (
    input.reasoningStyle === "enable_thinking" ||
    input.reasoningStyle === "enable_thinking_effort"
  ) {
    request.enableThinking = input.reasoningRequested;
  }
  if (
    input.reasoningRequested &&
    (input.reasoningStyle === "reasoning_effort" ||
      input.reasoningStyle === "enable_thinking_effort")
  ) {
    request.reasoningEffort = input.clampReasoningEffort(
      input.reasoningEffort,
      input.reasoningEffortLevels,
    );
  } else if (
    input.reasoningStyle === "reasoning_effort" &&
    input.external?.supportsReasoningOff
  ) {
    // Ollama thinks when no control arrives.
    request.reasoningEffort = "none";
  }
  return request;
}
