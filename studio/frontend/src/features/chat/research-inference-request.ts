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
  enableThinking?: boolean;
  reasoningEffort?: string;
}

export function buildResearchInferenceRequest(input: {
  checkpoint: string;
  external?: { providerId: string; providerType: string; modelId: string };
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
        }
      : {}),
  };
  if (Number.isFinite(input.temperature) && input.temperature >= 0 && input.temperature <= 2) {
    request.temperature = input.temperature;
  }
  if (Number.isFinite(input.topP) && input.topP > 0 && input.topP <= 1) {
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
  }
  return request;
}
