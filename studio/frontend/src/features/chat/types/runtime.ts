// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ServiceTier =
  | "auto"
  | "default"
  | "flex"
  | "priority"
  | "scale"
  | "standard_only";

export interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  presencePenalty: number;
  /** OpenAI Chat Completions only; rejected by Responses + Anthropic. */
  frequencyPenalty: number;
  /**
   * Best-effort determinism seed. OpenAI Chat Completions only; the
   * Responses family and Anthropic reject it (silently dropped server-side).
   * `null` = unset (no `seed` field on the wire).
   */
  seed: number | null;
  /**
   * Custom stop sequences. Maps to `stop` on OpenAI Chat Completions and
   * `stop_sequences` on Anthropic Messages. OpenAI caps the array at 4
   * entries; backend truncates with a warning. Empty array = unset.
   */
  stop: string[];
  /**
   * Provider service tier. Each provider accepts a different enum set;
   * `getServiceTierOptions(providerType)` resolves the legal values. `null`
   * means "let the provider pick its default" and is the safe choice on
   * provider switch.
   */
  serviceTier: ServiceTier | null;
  /**
   * Whether the provider may dispatch tool calls in parallel. Maps to
   * `parallel_tool_calls` on both OpenAI APIs and is inverted into
   * `disable_parallel_tool_use` for Anthropic. Default true matches the
   * upstream defaults across all three.
   */
  parallelToolCalls: boolean;
  /**
   * Locally typical sampling (llama.cpp `typ_p`). Local llama-server
   * only — no SaaS provider currently accepts this. 1.0 disables (and
   * is the llama-server default). `null` = unset (not forwarded).
   */
  typicalP: number | null;
  /** llama.cpp `top_n_sigma`. -1 disables. `null` = unset. */
  topNSigma: number | null;
  /** llama.cpp `repeat_last_n`. 0 disables, -1 = ctx-size. `null` = unset. */
  repeatLastN: number | null;
  /** llama.cpp `dynatemp_range`. 0.0 disables. `null` = unset. */
  dynatempRange: number | null;
  /** llama.cpp `dynatemp_exponent`. `null` = unset. */
  dynatempExponent: number | null;
  /** llama.cpp `mirostat` mode (0/1/2). 0 disables. `null` = unset. */
  mirostat: number | null;
  /** llama.cpp `mirostat_tau` target entropy. `null` = unset. */
  mirostatTau: number | null;
  /** llama.cpp `mirostat_eta` learning rate. `null` = unset. */
  mirostatEta: number | null;
  /**
   * OpenRouter `top_a` alternate dynamic-top-P. OpenRouter-only.
   * Range [0, 1]. `null` = unset.
   */
  topA: number | null;
  /**
   * llama.cpp DRY (Don't Repeat Yourself) penalty multiplier.
   * 0.0 disables (server default). `null` = unset.
   * https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
   */
  dryMultiplier: number | null;
  /** llama.cpp DRY base value (exponential growth base). Default 1.75. `null` = unset. */
  dryBase: number | null;
  /** llama.cpp DRY allowed token-extension threshold. Default 2. `null` = unset. */
  dryAllowedLength: number | null;
  /** llama.cpp DRY penalty scan window. 0 disables, -1 = ctx-size. `null` = unset. */
  dryPenaltyLastN: number | null;
  /** llama.cpp XTC sampler probability. 0.0 disables. `null` = unset. */
  xtcProbability: number | null;
  /** llama.cpp XTC sampler threshold. Default 0.1. `null` = unset. */
  xtcThreshold: number | null;
  /** llama.cpp `min_keep` (force min N tokens past filters). 0 disables. `null` = unset. */
  minKeep: number | null;
  /**
   * Force generation past the EOS token. llama.cpp + vLLM accept this.
   * `null` = unset; `false` matches upstream default.
   */
  ignoreEos: boolean | null;
  /**
   * Minimum output tokens before stop sequences / EOS can fire.
   * vLLM + llama.cpp accept this. 0 disables. `null` = unset.
   */
  minTokens: number | null;
  maxSeqLength: number;
  maxTokens: number;
  systemPrompt: string;
  checkpoint: string;
  /** Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust. */
  trustRemoteCode?: boolean;
  /**
   * Anthropic fast-mode toggle. Opus 4.6 / 4.7 only; higher OTPS at
   * 6x standard Opus pricing. Default false.
   * https://platform.claude.com/docs/en/build-with-claude/fast-mode
   */
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
