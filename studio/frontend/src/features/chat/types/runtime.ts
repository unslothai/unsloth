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
