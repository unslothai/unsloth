// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ServiceTier =
  | "auto"
  | "default"
  | "flex"
  | "priority"
  | "scale"
  | "standard_only";

// All `number | null` / `boolean | null` fields below follow the same
// convention: `null` = field omitted from the wire request (provider
// uses its own default). Per-provider capability gating lives in
// provider-capabilities.ts; the chat-adapter forwards only when the
// active provider's bucket has the matching flag set true.
export interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  presencePenalty: number;
  /** OpenAI Chat only; rejected by Responses + Anthropic. */
  frequencyPenalty: number;
  /** Determinism seed. OpenAI Chat + most OAI-compat backends only. */
  seed: number | null;
  /** OAI Chat `stop` / Anthropic `stop_sequences`. OAI caps at 4. */
  stop: string[];
  /** Per-provider enum via `getServiceTierOptions`. `null` = provider default. */
  serviceTier: ServiceTier | null;
  /** Anthropic inverts to `disable_parallel_tool_use`. */
  parallelToolCalls: boolean;
  /** llama.cpp `typ_p`. 1.0 disables. */
  typicalP: number | null;
  /** llama.cpp `top_n_sigma`. -1 disables. */
  topNSigma: number | null;
  /** llama.cpp `repeat_last_n`. 0 disables, -1 = ctx-size. */
  repeatLastN: number | null;
  /** llama.cpp `dynatemp_range`. 0 disables. */
  dynatempRange: number | null;
  /** llama.cpp `dynatemp_exponent`. Pairs with dynatempRange. */
  dynatempExponent: number | null;
  /** llama.cpp `mirostat` (0/1/2). 0 disables. */
  mirostat: number | null;
  mirostatTau: number | null;
  mirostatEta: number | null;
  /** OpenRouter `top_a`. Range [0, 1]. */
  topA: number | null;
  /**
   * llama.cpp DRY sampler — multiplier is the master switch (0 disables
   * the 4-field chain). See llama.cpp/tools/server/README.md.
   */
  dryMultiplier: number | null;
  /** Default 1.75. */
  dryBase: number | null;
  /** Default 2. */
  dryAllowedLength: number | null;
  /** 0 disables, -1 = ctx-size. */
  dryPenaltyLastN: number | null;
  /** llama.cpp XTC — probability is the master switch (0 disables). */
  xtcProbability: number | null;
  /** Default 0.1. */
  xtcThreshold: number | null;
  /** llama.cpp `min_keep` — min tokens past all filters. 0 disables. */
  minKeep: number | null;
  /** Continue past EOS. llama.cpp + vLLM. */
  ignoreEos: boolean | null;
  /** Min tokens before stop / EOS can fire. llama.cpp + vLLM. */
  minTokens: number | null;
  /** vLLM only. Default true; forward only when false. */
  skipSpecialTokens: boolean | null;
  /** vLLM only. Default true; forward only when false. */
  spacesBetweenSpecialTokens: boolean | null;
  /** vLLM only. Useful for agentic tools needing the matched stop string echoed. */
  includeStopStrInOutput: boolean | null;
  /** vLLM only. Left-truncate the prompt. */
  truncatePromptTokens: number | null;
  /** llama.cpp `n_keep`. 0 disables, -1 = keep all. */
  nKeep: number | null;
  /** llama.cpp `n_probs` — top-N token probabilities per token. */
  nProbs: number | null;
  /** llama.cpp `cache_prompt`. Default true; forward only when false. */
  cachePrompt: boolean | null;
  /** llama.cpp `return_tokens` (debug). */
  returnTokens: boolean | null;
  /** llama.cpp `timings_per_token` (perf debug). */
  timingsPerToken: boolean | null;
  /** llama.cpp `post_sampling_probs` (sampler debug). */
  postSamplingProbs: boolean | null;
  maxSeqLength: number;
  maxTokens: number;
  systemPrompt: string;
  checkpoint: string;
  /** Trust custom model code (e.g. NVIDIA Nemotron). Only for trusted repos. */
  trustRemoteCode?: boolean;
  /** Anthropic Opus 4.6 / 4.7 only. 6x pricing for higher OTPS. */
  fastMode?: boolean;
}

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1.0,
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  seed: null,
  stop: [],
  serviceTier: null,
  parallelToolCalls: true,
  typicalP: null,
  topNSigma: null,
  repeatLastN: null,
  dynatempRange: null,
  dynatempExponent: null,
  mirostat: null,
  mirostatTau: null,
  mirostatEta: null,
  topA: null,
  dryMultiplier: null,
  dryBase: null,
  dryAllowedLength: null,
  dryPenaltyLastN: null,
  xtcProbability: null,
  xtcThreshold: null,
  minKeep: null,
  ignoreEos: null,
  minTokens: null,
  skipSpecialTokens: null,
  spacesBetweenSpecialTokens: null,
  includeStopStrInOutput: null,
  truncatePromptTokens: null,
  nKeep: null,
  nProbs: null,
  cachePrompt: null,
  returnTokens: null,
  timingsPerToken: null,
  postSamplingProbs: null,
  maxSeqLength: 4096,
  maxTokens: 8192,
  systemPrompt: "",
  checkpoint: "",
  trustRemoteCode: false,
  fastMode: false,
};

export interface ChatModelSummary {
  id: string;
  name: string;
  description?: string;
  isVision: boolean;
  isLora: boolean;
  isGguf?: boolean;
  isAudio?: boolean;
  audioType?: string | null;
  hasAudioInput?: boolean;
}

export interface ChatLoraSummary {
  id: string;
  name: string;
  baseModel: string;
  updatedAt?: number;
  source?: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
}
