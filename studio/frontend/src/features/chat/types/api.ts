// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeInfo } from "@/features/transformers-upgrade";

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
  /** Codec of the checkpoint's base model when it fine-tunes an audio model, else null. */
  audio_type?: string | null;
}

export interface ListLorasResponse {
  loras: BackendLoraInfo[];
  outputs_dir: string;
}

export interface LoadModelRequest {
  model_path: string;
  /** Opaque client attempt ID used to cancel only this in-flight load. */
  load_request_id?: string | null;

  /** Start a fresh runtime even when the active settings already match. */
  force_reload?: boolean;
  /** Stop any chats still generating instead of getting a 409: a load replaces the single
   *  llama-server they all decode on. Set only after the user confirms. */
  force_cancel_active?: boolean;
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
  mlx_kv_bits?: number | null;
  /** Speculative decoding mode for GGUF models: "auto" (platform-aware DSpark/DFlash when the model
   *  ships that sidecar, else MTP on MTP GGUFs, ngram-mod for sub-3B), "mtp", "dspark",
   *  "dflash", "ngram", "mtp+ngram", "off". The legacy spellings are still accepted. */
  speculative_type?: string | null;
  /** Override --spec-draft-n-max for drafter speculative decoding. Applied only when speculative_type
   *  resolves to "mtp", "mtp+ngram", "dspark" or "dflash". */
  spec_draft_n_max?: number | null;
  /** Parallel decode slots for llama-server (--parallel), 1..64. Omit/null = the launch default. The
   *  VRAM fitter may launch fewer to stay on GPU. */
  n_parallel?: number | null;
  /** prompt batch size (--batch-size), 1..65536; omit/null = llama.cpp default 2048, gguf only */
  n_batch?: number | null;
  /** prompt micro-batch size (--ubatch-size), 1..65536; omit/null = llama.cpp default 512, capped at the batch size */
  n_ubatch?: number | null;
  /** Weight loading mode (--load-mode): auto/none/mmap/mlock/mmap+mlock/dio. Omit/null = llama.cpp's
   *  own `auto`. Settings -> Model Memory overrides it. */
  load_mode?: string | null;
  /** KV cache dtype for the DRAFT model's context (--spec-draft-type-k/-v); omit/null = f16. Only
   *  reaches the command line when the load attaches a separate draft model. */
  spec_draft_cache_type?: string | null;
  /** context checkpoints per slot (--ctx-checkpoints); omit/null = default 32, 0 disables */
  ctx_checkpoints?: number | null;
  /** host prompt cache size in MiB (--cache-ram); omit/null = default 8192, 0 disables, -1 unlimited */
  cache_ram?: number | null;
  /** Pass-through llama-server args, one argv token per entry, appended after Unsloth's own flags so
   *  llama.cpp's last-wins parser takes these. Managed flags are refused with a 4xx naming the
   *  flag. Omit/null inherits the stored per-model value; [] launches with none. GGUF only. */
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[] | null;
  /** Split the model across GPUs by tensor (--split-mode tensor) instead of by layer for GGUF models.
   *  Multi-GPU only. */
  tensor_parallel?: boolean | null;
  /** Load a vision-capable GGUF without its mmproj, freeing the VRAM the projector would occupy.
   *  Image input is unavailable for the session; text generation is unaffected. */
  disable_vision?: boolean | null;
  /** GPU memory strategy for GGUF models. "auto" (default): Unsloth selects GPUs and caps context to
   *  fit VRAM. "manual": you own the offload, with gpu_layers -1 handing sizing to llama.cpp's
   *  --fit and >= 0 pinning layers/n_cpu_moe. */
  gpu_memory_mode?: "auto" | "manual";
  /** Manual mode: layers to offload to GPU (--gpu-layers, --fit off); -1 = Auto (--fit). */
  gpu_layers?: number;
  /** Restore a previous automatic Vulkan CPU recovery after a failed model switch. */
  cpu_fallback?: boolean;
  /** Manual mode: MoE expert layers to keep on CPU (--n-cpu-moe); 0 = none. */
  n_cpu_moe?: number;
  /** Manual mode: relative model share per GPU (--tensor-split), in GPU order. */
  tensor_split?: number[] | null;
  /** Picked CUDA/ROCm physical IDs or Vulkan ordinals (omit/empty = automatic). */
  gpu_ids?: number[];
  /** Native audio (TTS / music) only: "cpu" holds the weights in system RAM
   *  rather than the GPU. Ignored for every non-audio model. */
  audio_device?: "auto" | "cpu" | "gpu";
}

export interface ValidateModelResponse {
  valid: boolean;
  message: string;
  identifier?: string | null;
  display_name?: string | null;
  is_gguf?: boolean;
  is_diffusion?: boolean;
  /** The diffusion check was inconclusive, so `is_diffusion: false` above means "not known to be
   *  diffusion", not "known to be ordinary". */
  diffusion_unknown?: boolean;
  is_lora?: boolean;
  is_vision?: boolean;
  requires_trust_remote_code?: boolean;
  // HF flagged unsafe files, so the load is hard-blocked pending dialog review.
  requires_security_review?: boolean;
  /** Native context length from the local GGUF header; null until downloaded. */
  context_length?: number | null;
  /** Total layer count (GGUF block_count); the manual gpu-layers ceiling is this + 1, since llama.cpp
   *  counts the output layer as offloadable. Null until downloaded. */
  layer_count?: number | null;
  /** MoE expert-layer count from the GGUF header (manual --n-cpu-moe ceiling); 0 for dense models,
   *  null until downloaded. */
  moe_layer_count?: number | null;
  /** Embedded GGUF chat template, returned when include_chat_template is set; null for non-GGUF,
   *  over-cap, or not read. */
  chat_template?: string | null;
  /** Architecture only shipped by a newer transformers; UI pauses on the upgrade dialog. */
  requires_transformers_upgrade?: boolean;
  /** Set only when requires_transformers_upgrade. */
  transformers_upgrade?: TransformersUpgradeInfo | null;
}

export interface GgufVariantDetail {
  filename: string;
  /** Selection identity. Path-qualified when a repo holds several checkpoints at one quant. */
  quant: string;
  /** What to SHOW for `quant` ("Q6_K · distilled"); absent when the key already reads as a label. */
  display_label?: string | null;
  size_bytes: number;
  download_size_bytes?: number;
  shard_count?: number;
  downloaded?: boolean;
  update_available?: boolean;
  /** An interrupted download: some shards are missing, so it cannot load yet. */
  partial?: boolean;
  /** Variants sharing this key share one companion download footprint. The set is not repo-wide: one
   *  repo can hold GGUFs of different families, and FLUX.2-klein picks its text encoder per
   *  checkpoint size. Null/absent means unknown, so the repo is one group. */
  dependency_key?: string | null;
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
  is_mlx?: boolean;
  status: string;
  model: string;
  display_name: string;
  is_vision: boolean;
  is_lora: boolean;
  is_gguf?: boolean;
  is_local_model?: boolean;
  is_diffusion?: boolean;
  /** GPU-layer count the diffusion runner was ASKED for, when it differs from what it applied: a shim
   *  without --ngl runs Auto, so gpu_layers reports -1 while this carries the request. */
  diffusion_requested_ngl?: number | null;
  is_audio?: boolean;
  audio_type?: string | null;
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
  mlx_kv_bits?: number | null;
  mlx_kv_bits_requested?: number | null;
  mlx_kv_quant_eligibility?: string | null;
  mlx_kv_quant_reason?: string | null;
  chat_template_override_reason?: string | null;
  mlx_kv_quant_note?: string | null;
  chat_template?: string | null;
  /** Canonical UI-facing mode the load request resolved to. See LoadModelRequest. */
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  /** Whether tensor-parallel split (--split-mode tensor) is active. */
  tensor_parallel?: boolean;
  /** The load ran with the vision projector deliberately left unloaded. Echoes the request, so it
   *  round-trips the Advanced Settings switch even on a GGUF that never had a projector, unlike
   *  vision_disabled_by_user below. */
  disable_vision?: boolean;
  /** Image input is off because the user asked, not because the mmproj is missing. */
  vision_disabled_by_user?: boolean;
  gpu_memory_mode?: "auto" | "manual";
  gpu_layers?: number;
  /** Set when an automatic Vulkan startup crash was recovered by loading on CPU. */
  cpu_fallback_reason?: CpuFallbackReason | null;
  /** How Unsloth recovered after a multimodal projector failed at startup. */
  mmproj_fallback_reason?: MmprojFallbackReason | null;
  n_cpu_moe?: number;
  tensor_split?: number[] | null;
  n_layers?: number | null;
  /** Model's MoE expert-layer count (the n_cpu_moe ceiling); 0 if not MoE. */
  n_moe_layers?: number;
  /** Effective GPU placement after fit-time narrowing. */
  gpu_ids?: number[] | null;
  /** User-requested GPU placement pool before fit-time narrowing. */
  requested_gpu_ids?: number[] | null;
  /** Slots the load was invoked with (else the --parallel default). Null for non-GGUF loads. */
  requested_parallel_slots?: number | null;
  /** Slots llama-server actually runs, after any fit-time reduction. Null for non-GGUF loads. */
  parallel_slots?: number | null;
  /** batch size (--batch-size) the load was invoked with; null = default */
  requested_n_batch?: number | null;
  /** micro-batch size (--ubatch-size) the load was invoked with; null = default */
  requested_n_ubatch?: number | null;
  /** load mode (--load-mode) the load was invoked with; null = default */
  requested_load_mode?: string | null;
  /** draft KV cache dtype the load was invoked with; null = default */
  requested_spec_draft_cache_type?: string | null;
  /** checkpoints (--ctx-checkpoints) the load was invoked with; null = default */
  requested_ctx_checkpoints?: number | null;
  /** host prompt cache (--cache-ram) the load was invoked with; null = default */
  requested_cache_ram?: number | null;
  /** Pass-through llama-server arguments the running load was invoked with. */
  requested_llama_extra_args?: string[] | null;
}

export interface UnloadModelRequest {
  model_path: string;
  /** Cancel this exact in-flight load; never unload an already-resident model. */
  cancel_load_request_id?: string | null;
  /** Stop any chats still generating instead of getting a 409: the unload takes down the llama-server
   *  they all decode on. */
  force_cancel_active?: boolean;
}

export interface InferenceStatusResponse {
  is_mlx?: boolean;
  active_model: string | null;
  model_identifier?: string | null;
  is_vision: boolean;
  is_gguf?: boolean;
  is_local_model?: boolean;
  is_diffusion?: boolean;
  /** GPU-layer count the diffusion runner was ASKED for, when it differs from what it applied: a shim
   *  without --ngl runs Auto, so gpu_layers reports -1 while this carries the request. */
  diffusion_requested_ngl?: number | null;
  gguf_variant?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  has_video_input?: boolean;
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
  cache_type_kv?: string | null;
  mlx_kv_bits?: number | null;
  mlx_kv_bits_requested?: number | null;
  mlx_kv_quant_eligibility?: string | null;
  mlx_kv_quant_reason?: string | null;
  chat_template_override_reason?: string | null;
  mlx_kv_quant_note?: string | null;
  chat_template_override?: string | null;
  /** Canonical UI-facing mode currently active. See LoadModelRequest. */
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
  /** Whether tensor-parallel split (--split-mode tensor) is active. */
  tensor_parallel?: boolean;
  /** The load ran with the vision projector deliberately left unloaded. Echoes the request, so it
   *  round-trips the Advanced Settings switch even on a GGUF that never had a projector. */
  disable_vision?: boolean;
  /** Image input is off because the user asked, not because the mmproj is missing. */
  vision_disabled_by_user?: boolean;
  gpu_memory_mode?: "auto" | "manual";
  gpu_layers?: number;
  /** Set while the active model is a recovered CPU-only Vulkan load. */
  cpu_fallback_reason?: CpuFallbackReason | null;
  /** How the active GGUF recovered after a multimodal projector startup failure. */
  mmproj_fallback_reason?: MmprojFallbackReason | null;
  n_cpu_moe?: number;
  tensor_split?: number[] | null;
  /** The context the active load was invoked with (0 = let the backend choose); re-seeds
   * the pin on hydration. Null where the serving backend records no request. */
  requested_context_length?: number | null;
  /** Effective GPU placement after fit-time narrowing. */
  gpu_ids?: number[] | null;
  /** User-requested GPU placement pool before fit-time narrowing. */
  requested_gpu_ids?: number[] | null;
  /** Slots the active load was invoked with (else the --parallel default). Null when no GGUF model is loaded. */
  requested_parallel_slots?: number | null;
  /** Slots llama-server actually runs, after any fit-time reduction. Null when no GGUF model is loaded. */
  parallel_slots?: number | null;
  /** batch size (--batch-size) the active load was invoked with; null = default */
  requested_n_batch?: number | null;
  /** micro-batch size (--ubatch-size) the active load was invoked with; null = default */
  requested_n_ubatch?: number | null;
  /** load mode (--load-mode) the active load was invoked with; null = default */
  requested_load_mode?: string | null;
  /** draft KV cache dtype the active load was invoked with; null = default */
  requested_spec_draft_cache_type?: string | null;
  /** checkpoints (--ctx-checkpoints) the active load was invoked with; null = default */
  requested_ctx_checkpoints?: number | null;
  /** host prompt cache (--cache-ram) the active load was invoked with; null = default */
  requested_cache_ram?: number | null;
  /** Pass-through llama-server arguments the running load was invoked with. */
  requested_llama_extra_args?: string[] | null;
  n_layers?: number | null;
  /** Model's MoE expert-layer count (the n_cpu_moe ceiling); 0 if not MoE. */
  n_moe_layers?: number;
  /** Why a speculative drafter was disabled despite being requested. "binary_no_mtp"/
   *  "binary_outdated": updating llama.cpp would re-enable it. "runtime_error": the build could
   *  not run it. "drafter_not_found": its sidecar was unavailable. "drafter_no_vram": an
   *  Auto-mode fit downgrade. "mla_mtp_disabled" and "mtp_partial_offload": Auto-mode policy
   *  downgrades where MTP costs more than it wins. Null otherwise. */
  /** Which drafter the resolution was about: "mtp", "dspark" or "dflash". Auto resolves the kind
   *  itself, so speculative_type still reads "auto" and a fallback leaves the engaged type at
   *  "default": neither names the file to fix. */
  spec_drafter_kind?: string | null;
  spec_fallback_reason?: string | null;
  /** Only for a binary stand-down: whether a different llama-server is installed now. */
  spec_fallback_binary_changed?: boolean | null;
  /** The capability probe has started answering since a launch it degraded. */
  spec_probe_retry_pending?: boolean | null;
  /** A DFlash sidecar fetch failed retryably, which records no fallback reason. */
  spec_dflash_retry_pending?: boolean | null;
  /** The DSpark drafter is absent for good, not transiently unfetchable. */
  spec_dspark_sidecar_absent?: boolean | null;
  /** The architecture gate normalized a tensor-parallel request to layer mode. */
  tensor_parallel_dropped_by_arch_gate?: boolean | null;
  /** A virtualised Metal device: every GGUF request is rewritten to the CPU pin. */
  gpu_placement_paravirtual?: boolean | null;
  /** The post-launch audio probe did not finish; only a load retries it. */
  audio_probe_pending?: boolean | null;
  /** A diffusion launch right now would honour --ngl. */
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
  // duration_ms covers the whole request, queue wait and prefill included. decode_ms is only the
  // generating span, and is absent unless the engine reported it.
  decode_ms?: number | null;
  context_length?: number | null;
  context_usage?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  error?: string | null;
  // "lifecycle" is a model load/unload/download: event/reason instead of a prompt.
  kind?: "request" | "lifecycle";
  event?: "load" | "unload" | "download" | null;
  reason?: "manual" | "idle" | "api" | null;
  // 0-100 while a download row is running.
  progress?: number | null;
  // Server-side time to first token (measured, else engine prefill).
  ttft_ms?: number | null;
  tok_per_sec?: number | null;
  /** Final request-specific prompt rate from engine timings. */
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
  // Server wall clock (seconds) at snapshot, so started_at can be dated without trusting the
  // browser's clock. Absent on an older backend.
  server_time?: number;
  active_model?: string | null;
  context_length?: number | null;
  active_requests: number;
  /** Live slot/queue occupancy; null when no llama model is loaded. */
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

export type OpenAIMessageContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | OpenAIReasoningContentPart
  | OpenAIImageGenerationCallContentPart;

export type OpenAIMessageContent = string | OpenAIMessageContentPart[];

/** OpenAI Chat Completions tool_call shape. Assistant turns echo function calls as `tool_calls`;
 *  the matching result rides on a separate `role="tool"` message keyed by `tool_call_id`.
 *  `extra_content.google.thought_signature` is the Gemini round-trip field. */
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
  /** Reasoning-class OpenAI models reject these; caller may omit. */
  temperature?: number;
  top_p?: number;
  max_tokens: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  /** Omitted when unset, which leaves the server to draw its own seed. */
  seed?: number;
  image_base64?: string;
  audio_base64?: string;
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
  /** Resume the trailing assistant turn rather than opening a new one: the rendered prompt ends inside
   *  the partial answer, so the model emits its next token. Local models only, since the
   *  external-provider proxy forwards an explicit field list. */
  continue_final_message?: boolean;
  thinking?: { type: "disabled" | "enabled" } | null;
  enable_tools?: boolean | null;
  enabled_tools?: string[];
  /** Local models + enable_tools only. */
  mcp_enabled?: boolean;
  /** The replayed tool calls came from Studio's own local tool loop. */
  studio_tool_history?: boolean;
  /** Local models + enable_tools only. */
  confirm_tool_calls?: boolean;
  /** Local models plus enable_tools only. Gate level for local tool calls: "ask" prompts on every
   *  call, "auto" only on calls flagged unsafe, "off" never, "full" never and drops the
   *  sandbox. Unset behaves as "ask". */
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
  /** Run the selected tools here rather than as the provider's hosted builtins. */
  run_tools_locally?: boolean;
  nudge_tool_calls?: boolean;
  /** Local GGUF overflow policy. Rolling mode preserves the transcript but omits oldest turns. */
  context_overflow?: "error" | "truncate_middle" | "truncate_oldest";
  /** Override UNSLOTH_CONTEXT_POLICY for this local GGUF request. */
  context_policy?: "checkpoint" | "rolling";
  /** Extra share of the prompt budget to drop when a rolling compaction fires. */
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
  /** Boolean toggle for OpenAI/Anthropic ephemeral cache_control. For Gemini the backend also accepts
   *  a cached-content resource name, forwarded as `generationConfig.cachedContent`. */
  enable_prompt_caching?: boolean | string | null;
  /** OpenAI shell-tool container id from the prior response in this thread. When set and the Code
   *  pill is on, the backend routes the next /v1/responses with
   *  `environment.type="container_reference"` so filesystem state persists; unset means a fresh
   *  container. OpenAI cloud and the gpt-5.5 family only. */
  openai_code_exec_container_id?: string | null;
  /** Anthropic code_execution container id from the prior response in this thread. When set and the
   *  Code pill is on, the backend forwards a top-level `container` on /v1/messages so filesystem
   *  state persists; unset is auto-created. */
  anthropic_code_exec_container_id?: string | null;
  /** Anthropic fast-mode toggle. Opus 4.6 / 4.7 only; dropped silently elsewhere. */
  fast_mode?: boolean | null;
  /** Opt into the OpenAI-standard trailing usage chunk on streams. The backend only emits it when
   *  `include_usage` is set; the local chat UI sends it so the context-usage bar and tok/s
   *  readout populate. */
  stream_options?: { include_usage?: boolean } | null;
}

export interface OpenAIChatDelta {
  role?: string;
  content?: string | null;
  /** Streamed assistant tool calls. The Gemini and OpenAI Responses translators emit incremental
   *  deltas so the chat-adapter can render tool cards as they arrive. */
  tool_calls?: OpenAIToolCallPart[];
  /** Provider-specific passthrough. Gemini ships `thoughtSignature`, citations, `native_part` and the
   *  like here so the round-trip can replay them without bleeding into other providers. */
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
  context_truncated?: {
    dropped_messages: number;
    prompt_tokens_before?: number;
    prompt_tokens_after?: number;
    context_length?: number;
    fits: boolean;
    // Present when the evicted turns were archived and searched. Counts only, never message text: this
    // rides an SSE chunk that reaches the client.
    archived_messages?: number;
    recalled_chunks?: number;
    // Present only when `fits` is false: the floor the conversation cannot go below, and how much of
    // it is the message just sent. Together they say whether the history or that one message is
    // the problem.
    irreducible_tokens?: number;
    latest_turn_tokens?: number;
    // Whether `latest_turn_tokens` is a real count or the four-characters-a-token estimate the fit
    // falls back to. Only the counted one may be quoted as the turn's size.
    latest_turn_exact?: boolean;
    // The floor both counts above carry: what a rendered prompt costs with no messages, which on a
    // tool-enabled request is the whole tool catalogue. Subtract it before comparing them, or the
    // catalogue is blamed on the turn.
    shared_prompt_tokens?: number;
    // Where the compaction boundary sits in the messages THIS request was sent with. Absolute, unlike
    // dropped_messages, so re-sending it after a turn that refit several times cannot advance the
    // boundary past the turns actually evicted.
    boundary_messages?: number;
    // The text the boundary landed ON, so the count can be re-derived by position: a count is only
    // valid against the transcript it was counted on, and deleting an already evicted prompt
    // shortens that transcript.
    boundary_anchor?: string;
    // How much extra trim the fit that set the boundary used. Replayed against the request's own
    // ratio, so a boundary cut under more headroom than the caller now asks for is discarded.
    boundary_headroom_ratio?: number;
    // Whose message that is: in a tool loop the last one is often a tool result rather than anything the user typed.
    latest_turn_role?: string;
    // The prompt's share of the window (context_length minus the reply reserve), which is what one turn
    // must fit inside. Not re-derived here: the formula lives in the fit.
    prompt_target?: number;
  };
}
