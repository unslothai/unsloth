// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeInfo } from "@/features/transformers-upgrade";

export interface BackendModelDetails {
  id: string;
  name?: string | null;
  is_vision?: boolean;
  is_lora?: boolean;
  is_gguf?: boolean;
  is_mlx?: boolean;
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
  /** sha256 fingerprint pinning user approval of this exact custom-code version. */
  approved_remote_code_fingerprint?: string | null;
  chat_template_override?: string | null;
  cache_type_kv?: string | null;
  /**
   * Speculative decoding mode for GGUF models. Canonical values: "auto"
   * (platform-aware: MTP on MTP GGUFs, ngram-mod fallback for sub-3B), "mtp"
   * (force draft-mtp), "ngram" (force ngram-mod), "mtp+ngram" (ngram-mod +
   * draft-mtp chain), "off". Legacy "default"/"draft-mtp"/"ngram-mod"/
   * "ngram-simple" are still accepted by the backend.
   */
  speculative_type?: string | null;
  /**
   * Override --spec-draft-n-max for MTP speculative decoding. Applied only
   * when speculative_type resolves to "mtp" or "mtp+ngram".
   */
  spec_draft_n_max?: number | null;
  /**
   * Split the model across GPUs by tensor (--split-mode tensor) instead
   * of by layer for GGUF models. Multi-GPU only; no effect on a single GPU.
   */
  tensor_parallel?: boolean | null;
  /** Load the GGUF mmproj vision projector for image input. */
  load_mmproj?: boolean | null;
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
  // HF flagged unsafe files, so the load is hard-blocked pending dialog review.
  requires_security_review?: boolean;
  /** Native context length from the local GGUF header; null until downloaded. */
  context_length?: number | null;
  /** Architecture only shipped by a newer transformers; UI pauses on the upgrade dialog. */
  requires_transformers_upgrade?: boolean;
  /** Set only when requires_transformers_upgrade. */
  transformers_upgrade?: TransformersUpgradeInfo | null;
}

export interface GgufVariantDetail {
  filename: string;
  quant: string;
  size_bytes: number;
  download_size_bytes?: number;
  downloaded?: boolean;
  update_available?: boolean;
}

export interface GgufVariantsResponse {
  repo_id: string;
  variants: GgufVariantDetail[];
  has_vision: boolean;
  default_variant: string | null;
  /** Native max context from GGUF metadata; present once a variant is downloaded. */
  context_length?: number | null;
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
  is_diffusion?: boolean;
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
  reasoning_style?: "enable_thinking" | "reasoning_effort" | "enable_thinking_effort";
  reasoning_effort_levels?: string[];
  reasoning_always_on?: boolean;
  supports_preserve_thinking?: boolean;
  supports_tools?: boolean;
  cache_type_kv?: string | null;
  chat_template?: string | null;
  /** Canonical UI-facing mode the load request resolved to. See LoadModelRequest. */
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  /** Whether tensor-parallel split (--split-mode tensor) is active. */
  tensor_parallel?: boolean;
  /** Whether the current GGUF load requested mmproj vision projector support. */
  load_mmproj?: boolean;
}

export interface UnloadModelRequest {
  model_path: string;
}

export interface InferenceStatusResponse {
  active_model: string | null;
  model_identifier?: string | null;
  is_vision: boolean;
  is_gguf?: boolean;
  is_diffusion?: boolean;
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
  } | null;
  requires_trust_remote_code?: boolean;
  supports_reasoning?: boolean;
  reasoning_style?: "enable_thinking" | "reasoning_effort" | "enable_thinking_effort";
  reasoning_effort_levels?: string[];
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
  /** Whether tensor-parallel split (--split-mode tensor) is active. */
  tensor_parallel?: boolean;
  /** Whether the current GGUF load requested mmproj vision projector support. */
  load_mmproj?: boolean;
  /**
   * Why MTP was disabled on the loaded model despite being requested.
   * "binary_no_mtp" / "binary_outdated" -> updating llama.cpp would re-enable
   * it; "runtime_error" -> the current build could not run it;
   * "mla_mtp_disabled" -> an Auto-mode policy downgrade for MLA models
   * (GLM-5.2 et al.) whose llama.cpp MTP path is slower than no speculation
   * (updating won't help; choose MTP in Settings to force it). Null otherwise.
   */
  spec_fallback_reason?: string | null;
}

export interface ApiMonitorEntry {
  id: string;
  endpoint: string;
  method: string;
  model: string;
  prompt?: string;
  reply?: string;
  prompt_preview: string;
  reply_preview: string;
  prompt_truncated: boolean;
  reply_truncated: boolean;
  status: "running" | "completed" | "cancelled" | "error";
  started_at: number;
  updated_at: number;
  finished_at?: number | null;
  duration_ms?: number | null;
  context_length?: number | null;
  context_usage?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  error?: string | null;
}

export interface ApiMonitorResponse {
  status: "idle" | "ready" | "generating";
  active_model?: string | null;
  context_length?: number | null;
  active_requests: number;
  entries: ApiMonitorEntry[];
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

export type OpenAIReasoningSummaryPart = {
  type: "summary_text";
  text: string;
};

export type OpenAIReasoningContentPart = {
  type: "reasoning";
  id: string;
  summary: OpenAIReasoningSummaryPart[];
  status?: "in_progress" | "completed" | "incomplete";
};

export type OpenAIImageGenerationCallContentPart = {
  type: "image_generation_call";
  id: string;
  response_id?: string;
};

export type OpenAIMessageContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | OpenAIReasoningContentPart
  | OpenAIImageGenerationCallContentPart;

export type OpenAIMessageContent = string | OpenAIMessageContentPart[];

/**
 * OpenAI Chat Completions tool_call shape. Assistant turns echo function calls
 * as `tool_calls`; the matching result rides on a separate `role="tool"`
 * message keyed by `tool_call_id`. `extra_content.google.thought_signature` is
 * the Gemini round-trip field the backend translator emits (on `delta.
 * tool_calls`) and consumes (when rebuilding the functionCall part next turn).
 */
export interface OpenAIToolCallPart {
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
  extra_content?: unknown;
}

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: OpenAIMessageContent | null;
  /** Assistant tool-call deltas, when the turn invoked a function tool. */
  tool_calls?: OpenAIToolCallPart[];
  /** `role="tool"` only: id matching `assistant.tool_calls[].id`. */
  tool_call_id?: string;
  /** `role="tool"` only: name of the function that produced the result. */
  name?: string;
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
  thinking?: {type: "disabled" | "enabled";} | null;
  enable_tools?: boolean | null;
  enabled_tools?: string[];
  /** Local models + enable_tools only. */
  mcp_enabled?: boolean;
  /** Local models + enable_tools only. */
  confirm_tool_calls?: boolean;
  /**
   * Local models + enable_tools only. Gate level for local tool calls: "ask"
   * prompts on every call, "auto" prompts only on calls flagged unsafe, "off"
   * never prompts, "full" never prompts and drops the sandbox. Unset behaves
   * as "ask".
   */
  permission_mode?: "ask" | "auto" | "off" | "full";
  /** Local models + enable_tools only. Full-access escape hatch. */
  bypass_permissions?: boolean;
  /** `kb_id` is exclusive; otherwise project and thread scopes may combine. */
  rag_scope?: {
    kb_id?: string;
    project_id?: string;
    thread_id?: string;
    default_top_k: number;
    mode: "hybrid" | "lexical" | "dense";
    autoinject?: boolean;
    autoinject_min_score?: number;

    whole_doc?: boolean;
    context_length?: number;
  };
  auto_heal_tool_calls?: boolean;
  nudge_tool_calls?: boolean;
  max_tool_calls_per_message?: number;
  tool_call_timeout?: number;
  session_id?: string;
  cancel_id?: string;
  provider_id?: string;
  provider_type?: string;
  external_model?: string;
  encrypted_api_key?: string;
  provider_base_url?: string | null;
  /**
   * Boolean toggle for OpenAI/Anthropic ephemeral cache_control. For Gemini the
   * backend also accepts a cached-content resource name (`cachedContents/...`)
   * string, forwarded as `generationConfig.cachedContent`.
   */
  enable_prompt_caching?: boolean | string | null;
  /**
   * OpenAI shell-tool container id from the prior response in this thread. When
   * set and the Code pill is on, the backend routes the next /v1/responses with
   * `environment.type="container_reference"` so filesystem state persists; unset
   * → `container_auto` (fresh container). OpenAI cloud + gpt-5.5 family only.
   */
  openai_code_exec_container_id?: string | null;
  /**
   * Anthropic code_execution container id from the prior response in this
   * thread. When set and the Code pill is on, the backend forwards a top-level
   * `container` on /v1/messages so filesystem state persists; unset →
   * auto-created. Anthropic provider with `code_execution` in `enabled_tools`.
   */
  anthropic_code_exec_container_id?: string | null;
  /**
   * Anthropic fast-mode toggle. Opus 4.6 / 4.7 only; dropped silently elsewhere.
   * See https://platform.claude.com/docs/en/build-with-claude/fast-mode
   */
  fast_mode?: boolean | null;
  /**
   * Opt into the OpenAI-standard trailing usage chunk on streams
   * (`choices: []` with `usage` + llama-server `timings` populated). The
   * backend only emits it when `include_usage` is set; the local chat UI
   * sends it so the context-usage bar and tok/s readout populate.
   */
  stream_options?: { include_usage?: boolean } | null;
}

export interface OpenAIChatDelta {
  role?: string;
  content?: string | null;
  /**
   * Streamed assistant tool calls. The Gemini and OpenAI Responses translators
   * emit incremental deltas (function name + arguments fragments) so the
   * chat-adapter can render tool cards as they arrive.
   */
  tool_calls?: OpenAIToolCallPart[];
  /**
   * Provider-specific passthrough. Gemini ships `thoughtSignature`, citations,
   * `native_part`, etc., here so the round-trip can replay them on follow-up
   * turns without bleeding into other providers.
   */
  extra_content?: Record<string, unknown>;
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
