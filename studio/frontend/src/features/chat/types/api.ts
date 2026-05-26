// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface BackendModelDetails {
  id: string;
  name?: string | null;
  is_vision?: boolean;
  is_lora?: boolean;
  is_gguf?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
}

export interface ListModelsResponse {
  models: BackendModelDetails[];
  default_models: string[];
}

export interface BackendLoraInfo {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
  source?: "training" | "exported" | null;
  export_type?: "lora" | "merged" | "gguf" | null;
}

export interface ListLorasResponse {
  loras: BackendLoraInfo[];
  outputs_dir: string;
}

export interface LoadModelRequest {
  model_path: string;
  nativePathLease?: string | null;
  hf_token: string | null;
  max_seq_length: number;
  load_in_4bit: boolean;
  is_lora: boolean;
  gguf_variant?: string | null;
  /** Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust. */
  trust_remote_code?: boolean;
  chat_template_override?: string | null;
  cache_type_kv?: string | null;
  /**
   * Speculative decoding mode for GGUF models. Canonical values:
   * "auto" (platform-aware: MTP on MTP GGUFs, ngram-mod fallback for
   * sub-3B), "mtp" (force draft-mtp only on both GPU and CPU), "ngram"
   * (force ngram-mod only), "mtp+ngram" (force ngram-mod + draft-mtp
   * chain on both platforms), or "off". Legacy values "default" /
   * "draft-mtp" / "ngram-mod" / "ngram-simple" are still accepted by
   * the backend.
   */
  speculative_type?: string | null;
  /**
   * Override --spec-draft-n-max for MTP speculative decoding. Only
   * applied when speculative_type resolves to "mtp" or "mtp+ngram".
   */
  spec_draft_n_max?: number | null;
}

export interface ValidateModelResponse {
  valid: boolean;
  message: string;
  identifier?: string | null;
  display_name?: string | null;
  is_gguf?: boolean;
  is_lora?: boolean;
  is_vision?: boolean;
  requires_trust_remote_code?: boolean;
}

export interface GgufVariantDetail {
  filename: string;
  quant: string;
  size_bytes: number;
  downloaded?: boolean;
}

export interface GgufVariantsResponse {
  repo_id: string;
  variants: GgufVariantDetail[];
  has_vision: boolean;
  default_variant: string | null;
}

export function isMultimodalResponse(
  response:
    | {
        is_vision?: boolean;
        is_audio?: boolean;
        audio_type?: string | null;
        has_audio_input?: boolean;
      }
    | null
    | undefined,
): boolean {
  return (
    Boolean(response?.is_vision) ||
    Boolean(response?.is_audio) ||
    Boolean(response?.has_audio_input) ||
    response?.audio_type === "audio_vlm"
  );
}

export interface LoadModelResponse {
  status: string;
  model: string;
  display_name: string;
  is_vision: boolean;
  is_lora: boolean;
  is_gguf?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  inference?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    presence_penalty?: number;
    trust_remote_code?: boolean;
  };
  requires_trust_remote_code?: boolean;
  context_length?: number | null;
  max_context_length?: number | null;
  native_context_length?: number | null;
  supports_reasoning?: boolean;
  reasoning_style?: "enable_thinking" | "reasoning_effort";
  reasoning_always_on?: boolean;
  supports_preserve_thinking?: boolean;
  supports_tools?: boolean;
  cache_type_kv?: string | null;
  chat_template?: string | null;
  /** Canonical UI-facing mode the load request resolved to. See LoadModelRequest. */
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
}

export interface UnloadModelRequest {
  model_path: string;
}

export interface InferenceStatusResponse {
  active_model: string | null;
  is_vision: boolean;
  is_gguf?: boolean;
  gguf_variant?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  loading: string[];
  loaded: string[];
  inference?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    presence_penalty?: number;
    trust_remote_code?: boolean;
  };
  requires_trust_remote_code?: boolean;
  supports_reasoning?: boolean;
  reasoning_style?: "enable_thinking" | "reasoning_effort";
  reasoning_always_on?: boolean;
  supports_preserve_thinking?: boolean;
  supports_tools?: boolean;
  chat_template?: string | null;
  context_length?: number | null;
  max_context_length?: number | null;
  native_context_length?: number | null;
  cache_type_kv?: string | null;
  chat_template_override?: string | null;
  /** Canonical UI-facing mode currently active. See LoadModelRequest. */
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
}

export interface AudioGenerationResponse {
  id: string;
  object: string;
  model: string;
  audio: {
    data: string;
    format: string;
    sample_rate: number;
  };
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string;
  }>;
}

export type OpenAIMessageContent =
  | string
  | Array<
      | { type: "text"; text: string }
      | { type: "image_url"; image_url: { url: string } }
    >;

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant";
  content: OpenAIMessageContent;
}

export interface OpenAIChatCompletionsRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream: boolean;
  /** Reasoning-class OpenAI models reject these — caller may omit. */
  temperature?: number;
  top_p?: number;
  max_tokens: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  image_base64?: string;
  audio_base64?: string;
  use_adapter?: boolean | string | null;
  enable_thinking?: boolean | null;
  reasoning_effort?:
    | "none"
    | "minimal"
    | "low"
    | "medium"
    | "high"
    | "max"
    | "xhigh"
    | null;
  preserve_thinking?: boolean | null;
  enable_tools?: boolean | null;
  enabled_tools?: string[];
  auto_heal_tool_calls?: boolean;
  max_tool_calls_per_message?: number;
  tool_call_timeout?: number;
  session_id?: string;
  cancel_id?: string;
  provider_id?: string;
  provider_type?: string;
  external_model?: string;
  encrypted_api_key?: string;
  provider_base_url?: string | null;
  enable_prompt_caching?: boolean | null;
  /**
   * OpenAI shell-tool container id captured from the prior response in
   * this chat thread. When set and the Code pill is on, the backend
   * routes the next /v1/responses call with
   * `environment.type="container_reference"` so filesystem state
   * persists across turns. Unset → backend uses
   * `environment.type="container_auto"` and OpenAI creates a fresh
   * container. Only meaningful for OpenAI cloud + gpt-5.5 family.
   */
  openai_code_exec_container_id?: string | null;
  /**
   * Anthropic code_execution container id captured from the prior
   * response in this chat thread. When set and the Code pill is on,
   * the backend forwards a top-level `container` field on
   * /v1/messages so filesystem state persists across turns. Unset →
   * Anthropic auto-creates a fresh container. Only meaningful for
   * the Anthropic provider with `code_execution` in `enabled_tools`.
   */
  anthropic_code_exec_container_id?: string | null;
  /**
   * OpenAI Chat Completions only; rejected by the Responses family and
   * silently dropped by Anthropic. Range -2.0 .. 2.0.
   */
  frequency_penalty?: number;
  /**
   * Best-effort determinism seed. OpenAI Chat / OpenAI-compat backends
   * forward it; Responses + Anthropic drop it server-side.
   */
  seed?: number;
  /**
   * Custom stop sequences. Backend translates to `stop_sequences` for
   * Anthropic; OpenAI Chat caps at 4 entries (server-side truncates
   * with a warning). Empty arrays are omitted.
   */
  stop?: string[];
  /**
   * Provider service tier. Anthropic accepts `auto|standard_only`;
   * OpenAI Chat + Responses both accept
   * `auto|default|flex|scale|priority` per the live `openai-python`
   * SDK (`src/openai/types/responses/response_create_params.py`
   * declares `Optional[Literal["auto", "default", "flex", "scale",
   * "priority"]]`). The wire-side helper in
   * `studio/backend/core/inference/external_provider.py` drops values
   * that a given provider does not accept; this union stays permissive
   * so the request-builder typechecks against
   * `InferenceParams.serviceTier` without per-provider narrowing.
   */
  service_tier?:
    | "auto"
    | "default"
    | "flex"
    | "priority"
    | "scale"
    | "standard_only";
  /**
   * Whether the provider may dispatch tool calls in parallel.
   * OpenAI: forwarded as `parallel_tool_calls`. Anthropic: inverted
   * into `disable_parallel_tool_use` server-side. Default `undefined`
   * keeps each provider's upstream default.
   */
  parallel_tool_calls?: boolean;
  /**
   * Anthropic fast-mode toggle. Opus 4.6 / 4.7 only; backend drops
   * silently on every other model + provider. See
   * https://platform.claude.com/docs/en/build-with-claude/fast-mode
   */
  fast_mode?: boolean | null;
}

export interface OpenAIChatDelta {
  role?: string;
  content?: string;
}

export interface OpenAIChatChunkChoice {
  delta?: OpenAIChatDelta;
  finish_reason?: string | null;
}

export interface OpenAIChatChunk {
  choices?: OpenAIChatChunkChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  timings?: Record<string, number>;
}
