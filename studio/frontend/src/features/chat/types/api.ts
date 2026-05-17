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
  speculative_type?: string | null;
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
  speculative_type?: string | null;
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
  speculative_type?: string | null;
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
