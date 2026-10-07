// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeInfo } from "@/features/transformers-upgrade";
import type { CustomReasoningConfig } from "../custom-reasoning";

export type CpuFallbackReason = "vulkan_startup_crash";

export type MmprojFallbackReason =
  | "cpu_offload"
  | "projector_incompatible"
  | "projector_startup_failure";

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
  has_video_input?: boolean;
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
  size_bytes?: number | null;
  audio_type?: string | null;
}

export interface ListLorasResponse {
  loras: BackendLoraInfo[];
  outputs_dir: string;
}

export interface LoadModelRequest {
  engine_parallelism?: "tensor" | "pipeline" | "data";
  engine_precision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  model_path: string;
  load_request_id?: string | null;

  force_reload?: boolean;
  alongside?: boolean;
  /** Set only after the user confirms: the load replaces the shared llama-server. */
  force_cancel_active?: boolean;
  nativePathLease?: string | null;
  hf_token: string | null;
  max_seq_length: number;
  max_seq_length_auto_derived?: boolean;
  load_in_4bit: boolean;
  is_lora: boolean;
  gguf_variant?: string | null;
  /** Only enable for repos you trust. */
  trust_remote_code?: boolean;
  approved_remote_code_fingerprint?: string | null;
  chat_template_override?: string | null;
  cache_type_kv?: string | null;
  mlx_kv_quant?: string | null;
  mlx_int8_prefill?: boolean;
  /** "auto", "mtp", "dspark", "dflash", "ngram", "mtp+ngram", "off"; legacy spellings accepted. */
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  /** 1..64; the VRAM fitter may launch fewer to stay on GPU. */
  n_parallel?: number | null;
  /** -1 unrestricted, 0 end immediately, >0 token cap. */
  reasoning_budget?: number;
  reasoning_budget_message?: string;
  n_batch?: number | null;
  n_ubatch?: number | null;
  load_mode?: string | null;
  spec_draft_cache_type?: string | null;
  ctx_checkpoints?: number | null;
  /** MiB; null = default 8192, 0 disables, -1 unlimited */
  cache_ram?: number | null;
  /** Appended after Unsloth's flags (last wins); managed flags get a 4xx. Null inherits, [] none. */
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[] | null;
  tensor_parallel?: boolean | null;
  disable_vision?: boolean | null;
  /** "manual": gpu_layers -1 hands sizing to llama.cpp --fit, >= 0 pins layers/n_cpu_moe. */
  gpu_memory_mode?: "auto" | "manual";
  /** -1 = Auto (--fit). */
  gpu_layers?: number;
  cpu_fallback?: boolean;
  n_cpu_moe?: number;
  tensor_split?: number[] | null;
  gpu_ids?: number[];
  audio_device?: "auto" | "cpu" | "gpu";
}

export interface ValidateModelResponse {
  valid: boolean;
  message: string;
  identifier?: string | null;
  resident?: boolean;
  display_name?: string | null;
  is_gguf?: boolean;
  is_diffusion?: boolean;
  /** Check was inconclusive: `is_diffusion: false` means not known, not known-false. */
  diffusion_unknown?: boolean;
  is_lora?: boolean;
  is_vision?: boolean;
  requires_trust_remote_code?: boolean;
  requires_security_review?: boolean;
  context_length?: number | null;
  /** gpu-layers ceiling is this + 1, since llama.cpp counts the output layer. */
  layer_count?: number | null;
  moe_layer_count?: number | null;
  chat_template?: string | null;
  requires_transformers_upgrade?: boolean;
  transformers_upgrade?: TransformersUpgradeInfo | null;
  mlx_loads_base_model?: string | null;
}

export interface GgufVariantDetail {
  context_length?: number | null;
  cache_path?: string | null;
  /** Stand-in for a redacted `cache_path`; API-key callers use it to target one copy. */
  cache_ref?: string | null;
  filename: string;
  quant: string;
  display_label?: string | null;
  size_bytes: number;
  download_size_bytes?: number;
  pending_drafter_filename?: string | null;
  pending_drafter_size_bytes?: number;
  downloaded?: boolean;
  update_available?: boolean;
  partial?: boolean;
  /** Not repo-wide: one repo can hold several GGUF families. Null means the repo is one group. */
  dependency_key?: string | null;
}

export interface GgufVariantsResponse {
  repo_id: string;
  variants: GgufVariantDetail[];
  has_vision: boolean;
  default_variant: string | null;
  dependencies_resolved?: boolean;
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
  engine_parallelism?: "tensor" | "pipeline" | "data";
  engine_precision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  is_mlx?: boolean;
  evicted?: string[];
  is_npu?: boolean;
  status: string;
  model: string;
  display_name: string;
  is_vision: boolean;
  is_lora: boolean;
  is_gguf?: boolean;
  is_local_model?: boolean;
  /** Unknown-typed so an older backend cannot render "undefined GB"; see parseCarveoutAdvice. */
  carveout_advice?: unknown;
  memory_warning?: string | null;
  is_diffusion?: boolean;
  /** Requested ngl when it differs from applied (a shim without --ngl reports -1). */
  diffusion_requested_ngl?: number | null;
  is_audio?: boolean;
  audio_type?: string | null;
  audio_workflows?: string[] | null;
  has_audio_input?: boolean;
  has_video_input?: boolean;
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
  context_length_enforced?: boolean | null;
  context_unbounded_when_batched?: boolean;
  supports_reasoning?: boolean;
  reasoning_style?:
    | "enable_thinking"
    | "reasoning_effort"
    | "enable_thinking_effort";
  reasoning_effort_levels?: string[];
  reasoning_always_on?: boolean;
  supports_preserve_thinking?: boolean;
  preserve_thinking_default?: boolean;
  supports_tools?: boolean;
  cache_type_kv?: string | null;
  mlx_kv_quant?: string | null;
  mlx_kv_quant_requested?: string | null;
  mlx_kv_quant_eligibility?: string | null;
  mlx_kv_quant_reason?: string | null;
  chat_template_override_reason?: string | null;
  mlx_kv_quant_note?: string | null;
  mlx_int8_prefill?: boolean | null;
  mlx_int8_prefill_requested?: boolean | null;
  mlx_int8_prefill_reason?: string | null;
  chat_template?: string | null;
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  tensor_parallel?: boolean;
  /** Echoes the request, unlike vision_disabled_by_user below. */
  disable_vision?: boolean;
  vision_disabled_by_user?: boolean;
  gpu_memory_mode?: "auto" | "manual";
  gpu_layers?: number;
  offloaded_layers?: number | null;
  offload_total_layers?: number | null;
  offload_overridden?: boolean | null;
  gpu_backend_unavailable?: boolean | null;
  cpu_fallback_reason?: CpuFallbackReason | null;
  mmproj_fallback_reason?: MmprojFallbackReason | null;
  n_cpu_moe?: number;
  tensor_split?: number[] | null;
  n_layers?: number | null;
  n_moe_layers?: number;
  gpu_ids?: number[] | null;
  requested_gpu_ids?: number[] | null;
  requested_parallel_slots?: number | null;
  reasoning_budget?: number;
  reasoning_budget_message?: string;
  /** Requested value before LLAMA_ARG_THINK_BUDGET*, which a client can resend. */
  // biome-ignore lint/style/useNamingConvention: API schema
  requested_reasoning_budget?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  requested_reasoning_budget_message?: string;
  parallel_slots?: number | null;
  requested_n_batch?: number | null;
  requested_n_ubatch?: number | null;
  requested_load_mode?: string | null;
  requested_spec_draft_cache_type?: string | null;
  requested_ctx_checkpoints?: number | null;
  requested_cache_ram?: number | null;
  requested_llama_extra_args?: string[] | null;
}

export interface UnloadModelRequest {
  model_path: string;
  cancel_load_request_id?: string | null;
  /** Set only after confirmation: unloading takes down the shared llama-server. */
  force_cancel_active?: boolean;
}

export interface InferenceStatusResponse {
  engine_parallelism?: "tensor" | "pipeline" | "data";
  engine_precision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  is_mlx?: boolean;
  is_npu?: boolean;
  active_model: string | null;
  model_identifier?: string | null;
  is_vision: boolean;
  is_gguf?: boolean;
  is_local_model?: boolean;
  is_diffusion?: boolean;
  /** Requested ngl when it differs from applied (a shim without --ngl reports -1). */
  diffusion_requested_ngl?: number | null;
  gguf_variant?: string | null;
  memory_warning?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
  audio_family?: string | null;
  /** Unknown-typed on purpose: the Audio page validates it with parseAudioOptions. */
  audio_options?: unknown;
  audio_workflows?: string[] | null;
  audio_reference_text?: "required" | "optional" | "unused" | null;
  audio_options_by_workflow?: Record<string, unknown> | null;
  /** e.g. {"clone": "clon"}; a task other than audio_server_task reloads. */
  audio_workflow_tasks?: Record<string, string> | null;
  audio_server_task?: string | null;
  audio_convert_route?: string | null;
  audio_convert?: AudioConvertCaps | null;
  audio_required_inputs?: string[] | null;
  /** Unknown-shaped on purpose: validated by parseMusicCapabilities. */
  audio_music?: unknown;
  has_audio_input?: boolean;
  has_video_input?: boolean;
  loading: string[];
  loaded: string[];
  serving?: string[];
  serving_checkpoints?: string[];
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
  reasoning_style?:
    | "enable_thinking"
    | "reasoning_effort"
    | "enable_thinking_effort";
  reasoning_effort_levels?: string[];
  reasoning_always_on?: boolean;
  supports_preserve_thinking?: boolean;
  preserve_thinking_default?: boolean;
  supports_tools?: boolean;
  chat_template?: string | null;
  context_length?: number | null;
  max_context_length?: number | null;
  native_context_length?: number | null;
  context_length_enforced?: boolean | null;
  context_unbounded_when_batched?: boolean;
  cache_type_kv?: string | null;
  mlx_kv_quant?: string | null;
  mlx_kv_quant_requested?: string | null;
  mlx_kv_quant_eligibility?: string | null;
  mlx_kv_quant_reason?: string | null;
  chat_template_override_reason?: string | null;
  mlx_kv_quant_note?: string | null;
  mlx_int8_prefill?: boolean | null;
  mlx_int8_prefill_requested?: boolean | null;
  mlx_int8_prefill_reason?: string | null;
  chat_template_override?: string | null;
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  tensor_parallel?: boolean;
  disable_vision?: boolean;
  vision_disabled_by_user?: boolean;
  gpu_memory_mode?: "auto" | "manual";
  gpu_layers?: number;
  offloaded_layers?: number | null;
  offload_total_layers?: number | null;
  offload_overridden?: boolean | null;
  gpu_backend_unavailable?: boolean | null;
  cpu_fallback_reason?: CpuFallbackReason | null;
  mmproj_fallback_reason?: MmprojFallbackReason | null;
  n_cpu_moe?: number;
  tensor_split?: number[] | null;
  /** 0 = backend chooses; re-seeds the pin on hydration. */
  requested_context_length?: number | null;
  gpu_ids?: number[] | null;
  requested_gpu_ids?: number[] | null;
  requested_parallel_slots?: number | null;
  reasoning_budget?: number;
  reasoning_budget_message?: string;
  /** Requested value before LLAMA_ARG_THINK_BUDGET*, which a client can resend. */
  // biome-ignore lint/style/useNamingConvention: API schema
  requested_reasoning_budget?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  requested_reasoning_budget_message?: string;
  parallel_slots?: number | null;
  requested_n_batch?: number | null;
  requested_n_ubatch?: number | null;
  requested_load_mode?: string | null;
  requested_spec_draft_cache_type?: string | null;
  requested_ctx_checkpoints?: number | null;
  requested_cache_ram?: number | null;
  requested_llama_extra_args?: string[] | null;
  n_layers?: number | null;
  n_moe_layers?: number;
  /** Why a requested drafter was disabled; binary_* reasons mean updating llama.cpp re-enables it. */
  /** "mtp", "dspark" or "dflash"; needed because speculative_type may still read "auto". */
  spec_drafter_kind?: string | null;
  spec_fallback_reason?: string | null;
  spec_fallback_binary_changed?: boolean | null;
  spec_probe_retry_pending?: boolean | null;
  spec_dflash_retry_pending?: boolean | null;
  spec_dspark_sidecar_absent?: boolean | null;
  tensor_parallel_dropped_by_arch_gate?: boolean | null;
  /** Virtualised Metal device: every GGUF request is rewritten to the CPU pin. */
  gpu_placement_paravirtual?: boolean | null;
  audio_probe_pending?: boolean | null;
  diffusion_split_supported?: boolean | null;
}

export interface ApiMonitorEntry {
  id: string;
  endpoint: string;
  method: string;
  model: string;
  prompt?: string;
  reply?: string;
  // True for API-key callers, not UI sessions: the panel auto-opens off this.
  via_api_key: boolean;
  prompt_preview: string;
  reply_preview: string;
  prompt_truncated: boolean;
  reply_truncated: boolean;
  status: "running" | "completed" | "cancelled" | "error";
  started_at: number;
  updated_at: number;
  finished_at?: number | null;
  duration_ms?: number | null;
  // duration_ms includes queue and prefill; decode_ms is only generation and may be absent.
  decode_ms?: number | null;
  context_length?: number | null;
  context_usage?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  error?: string | null;
  kind?: "request" | "lifecycle";
  event?: "load" | "unload" | "download" | null;
  reason?: "manual" | "idle" | "api" | null;
  progress?: number | null;
  running_phase?: "prompt_processing" | "token_generation" | null;
  prompt_progress?: {
    total: number | null;
    processed: number | null;
    cached: number | null;
    time_ms: number | null;
    percent: number | null;
  } | null;
  ttft_ms?: number | null;
  tok_per_sec?: number | null;
  prompt_tok_per_sec?: number | null;
  stop_reason?: string | null;
}

export interface ApiMonitorQueue {
  capacity: number;
  active: number;
  queued: number;
  free: number;
}

export interface ApiMonitorResponse {
  status: "idle" | "ready" | "generating";
  // Server wall clock, so started_at can be dated without trusting the browser clock.
  server_time?: number;
  active_model?: string | null;
  context_length?: number | null;
  active_requests: number;
  queue?: ApiMonitorQueue | null;
  /** Absent on older backends: treat only an explicit `false` as disabled. */
  logging_enabled?: boolean;
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

export type ProviderCompactionContentPart = {
  type: "compaction";
  content?: string;
  encrypted_content?: string;
};

export type OpenAIMessageContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | OpenAIReasoningContentPart
  | OpenAIImageGenerationCallContentPart
  | ProviderCompactionContentPart;

export type OpenAIMessageContent = string | OpenAIMessageContentPart[];

/** OpenAI tool_calls pair by tool_call_id; Gemini uses extra_content.google.thought_signature */
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
  tool_calls?: OpenAIToolCallPart[];
  tool_call_id?: string;
  name?: string;
}

export interface OpenAIChatCompletionsRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream: boolean;
  /** Reasoning-class OpenAI models reject these; caller may omit. */
  temperature?: number;
  top_p?: number;
  max_tokens: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  image_base64?: string;
  audio_base64?: string;
  extra_audio_base64?: string[];
  video_base64?: string;
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
  /** Local models only: the external-provider proxy forwards an explicit field list. */
  continue_final_message?: boolean;
  thinking?: { type: "disabled" | "enabled" } | null;
  enable_tools?: boolean | null;
  enabled_tools?: string[];
  mcp_enabled?: boolean;
  mcp_image?: string;
  studio_tool_history?: boolean;
  confirm_tool_calls?: boolean;
  /** "ask" every call, "auto" unsafe only, "off" never, "full" never and no sandbox. Unset = ask. */
  permission_mode?: "ask" | "auto" | "off" | "full";
  /** "high" (default) adds the OS sandbox when it works; "low" runs Python/Terminal on software
   *  safeguards only. Full access overrides both. */
  sandbox_level?: "high" | "low";
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
  run_tools_locally?: boolean;
  nudge_tool_calls?: boolean;
  context_overflow?: "error" | "truncate_middle" | "truncate_oldest";
  context_policy?: "checkpoint" | "rolling";
  compaction_headroom_ratio?: number;
  max_tool_calls_per_message?: number;
  tool_call_timeout?: number;
  session_id?: string;
  cancel_id?: string;
  provider_id?: string;
  provider_type?: string;
  external_model?: string;
  encrypted_api_key?: string;
  provider_base_url?: string | null;
  provider_api_type?: "chat_completions" | "responses";
  provider_reasoning_config?: CustomReasoningConfig;
  /** Boolean toggle for OpenAI/Anthropic ephemeral cache_control. For Gemini the backend also accepts
   *  a cached-content resource name, forwarded as `generationConfig.cachedContent`. */
  enable_prompt_caching?: boolean | string | null;
  /** OpenAI cloud and gpt-5.5 family only; unset means a fresh container. */
  openai_code_exec_container_id?: string | null;
  anthropic_code_exec_container_id?: string | null;
  /** Opus 4.6 / 4.7 only; dropped silently elsewhere. */
  fast_mode?: boolean | null;
  /** The backend only emits the usage chunk when include_usage is set. */
  stream_options?: { include_usage?: boolean } | null;
}

export interface OpenAIChatDelta {
  role?: string;
  /** Magistral streams structured content parts: read through extractDeltaText. */
  content?: string | unknown[] | null;
  tool_calls?: OpenAIToolCallPart[];
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
  quote_cut?: boolean;
  context_truncated?: {
    dropped_messages: number;
    prompt_tokens_before?: number;
    prompt_tokens_after?: number;
    context_length?: number;
    fits: boolean;
    // Counts only, never message text: this rides an SSE chunk to the client.
    archived_messages?: number;
    recalled_chunks?: number;
    // Only when `fits` is false: the irreducible floor, to tell history from the new message.
    irreducible_tokens?: number;
    latest_turn_tokens?: number;
    // true when latest_turn_tokens is counted rather than estimated at four characters per token
    latest_turn_exact?: boolean;
    // subtract message-free prompt cost so tools are not charged to the turn
    shared_prompt_tokens?: number;
    // absolute request boundary prevents repeated refits from advancing past evicted turns
    boundary_messages?: number;
    // true when this fit started a checkpoint, including inside the current tool loop
    checkpoint_started?: boolean;
    // true when the provider summarized earlier turns instead of Unsloth dropping them
    summarized?: boolean;
    // boundary text lets the count be re-derived after deleting an already evicted prompt
    boundary_anchor?: string;
    // discard a replayed boundary when its trim used more headroom than the current request
    boundary_headroom_ratio?: number;
    // the latest tool-loop message may be a tool result rather than a user message
    latest_turn_role?: string;
    // prompt share of context_length after the reply reserve, calculated by the fit
    prompt_target?: number;
  };
}

export interface AudioConvertCaps {
  modes: ("speech" | "singing")[];
  target: "audio" | "builtin";
  builtin_voices: { id: string; label: string }[];
  pitch: Partial<
    Record<"speech" | "singing", { auto: boolean; shift_with_auto?: boolean }>
  >;
  style: boolean;
  route_reloads: boolean;
  source_max_seconds: number;
}
