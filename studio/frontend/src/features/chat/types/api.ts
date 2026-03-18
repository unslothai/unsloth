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
  hf_token: string | null;
  max_seq_length: number;
  load_in_4bit: boolean;
  is_lora: boolean;
  gguf_variant?: string | null;
  /** Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust. */
  trust_remote_code?: boolean;
  chat_template_override?: string | null;
  cache_type_kv?: string | null;
}

export interface ValidateModelResponse {
  valid: boolean;
  message: string;
  identifier?: string | null;
  display_name?: string | null;
  is_gguf?: boolean;
  is_lora?: boolean;
  is_vision?: boolean;
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
  context_length?: number | null;
  supports_reasoning?: boolean;
  supports_tools?: boolean;
  cache_type_kv?: string | null;
  chat_template?: string | null;
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
  supports_reasoning?: boolean;
  supports_tools?: boolean;
  context_length?: number | null;
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

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface OpenAIChatCompletionsRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream: boolean;
  temperature: number;
  top_p: number;
  max_tokens: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  presence_penalty: number;
  image_base64?: string;
  audio_base64?: string;
  use_adapter?: boolean | string | null;
  enable_thinking?: boolean | null;
  enable_tools?: boolean | null;
  enabled_tools?: string[];
  auto_heal_tool_calls?: boolean;
  max_tool_calls_per_message?: number;
  tool_call_timeout?: number;
  session_id?: string;
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
}
