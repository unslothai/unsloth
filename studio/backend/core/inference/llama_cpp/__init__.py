# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions through its
OpenAI-compatible /v1/chat/completions endpoint.

This package replaced a single 43,000-line module. Every name that module defined is
re-exported here, so ``from core.inference.llama_cpp import X`` and ``llama_cpp.X`` keep
working, and tests that patch a name on this module still reach the code that reads it:
``__setattr__`` below rebinds the same object in every submodule that imported it.
"""

import sys as _sys
import types as _types

# The imports the old module made, kept so ``llama_cpp.subprocess`` and the names tests
# patch by string (``patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback")``)
# remain attributes of this module.
import ast
import atexit
import bisect
import weakref
import contextlib
import ctypes
import errno
import functools
import hashlib
import json
import logging
import math
import os
import re
import struct
from loggers import get_logger
from utils.gpu_memory_events import invalidate_gpu_memory as _invalidate_gpu_memory
from utils import host_memory as _host_memory
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Union,
)

import httpx

from core.inference.chat_template_helpers import build_dac_tts_prompt
from core.inference.context_window import (
    _COMPACTION_HEADROOM_RATIO,
    clamp_compaction_headroom_ratio,
    prompt_budget,
    compact_completed_tool_arguments,
    compact_executed_call_arguments,
    compact_refused_tool_arguments,
    estimate_message_tokens_without_unpriced_media,
    estimate_messages_tokens_without_unpriced_media,
    _reply_floor,
    estimate_messages_tokens_conservative,
    estimate_messages_tokens_dense,
    evicted_messages,
    fit_rolling_context,
    messages_without_unpriced_media,
    retrieval_budget,
    fit_oversized_attachments,
    tool_result_budget,
    turn_is_servable,
)
from core.inference.llama_tool_schema import llama_grammar_tools
from core.inference.stream_errors import stream_error_from_chunk
from hub.utils.hf_errors import modelscope_missing
from hub.utils.hf_tokens import (
    call_hub_with_anonymous_retry,
    hf_token_rejected_hint,
    is_rejected_credential_error,
)
from core.inference.llama_server_args import (
    _CACHE_RAM_FLAGS,
    _CACHE_TYPE_K_FLAGS,
    _CACHE_TYPE_V_FLAGS,
    _CTX_CHECKPOINTS_FLAGS,
    _DEVICE_FLAGS,
    _GPU_LAYER_FLAGS,
    _LAYER_OFFLOAD_FLAGS,
    _MOE_OFFLOAD_FLAGS,
    _PARALLEL_FLAGS,
    _SPLIT_MODE_FLAGS,
    _TENSOR_SPLIT_FLAGS,
    _effective_tensor_parallel,
    _flag_name,
    _last_flag_value,
    _tensor_parallel_matches_loaded,
    LLAMA_CTX_CHECKPOINTS_DEFAULT,
    apply_load_mode_policy,
    apply_model_memory_policy,
    ctx_checkpoints_within_host_budget,
    effective_ctx_checkpoints,
    parse_ctx_checkpoints_override,
    resolve_ctx_checkpoints,
    extra_args_disable_mmproj,
    extra_args_image_max_tokens,
    extra_args_mmproj_auto,
    extra_args_select_load_mode,
    fit_is_enabled_in,
    force_pageable_load,
    memory_env_selects_load_mode,
    memory_state_satisfies_settings,
    fit_is_effectively_on,
    fit_target_margin_in,
    MANAGED_DIO_FLAGS,
    no_reserve_requires_dio,
    resolve_launch_load_mode,
    resolve_effective_load_state,
    resolve_effective_memory_state,
    scrub_denied_env,
    scrub_memory_env,
    parse_cache_override,
    parse_cache_override_per_axis,
    parse_ctx_override,
    parse_gpu_layers_override,
    parse_reasoning_budget_message_override,
    parse_reasoning_budget_override,
    parse_split_mode_override,
    resolve_requested_ctx,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
    resolve_reasoning_budget_message_with_env,
    resolve_reasoning_budget_with_env,
    split_policy_starves_devices,
    strip_context_only,
    strip_shadowing_flags,
    strip_split_mode_only,
)


# Share strip / signal constants with the multi-format parser so BUFFERING also
# catches Llama-3 / Mistral / Gemma 4 (legacy helper only knew <tool_call> / <function=).
from core.inference.tool_call_parser import (
    _GEMMA_BARE_TC_PREFIX_RE,
    _balanced_brace_end,
    blocked_bare_json_chain_may_continue,
    blocked_gemma_chain_may_continue,
    held_bare_gemma_tail_len,
    blocked_markerless_prefix_end,
    leading_bare_gemma_call_is_promotable,
    promotable_gemma_call_pos,
    TOOL_XML_SIGNALS as _SHARED_TOOL_XML_SIGNALS,
    StreamingMarkupStripper as _StreamingMarkupStripper,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    RAG_SEARCH_TOOLS,
    parse_tool_calls_from_text as _shared_parse_tool_calls_from_text,
    strip_leading_bare_json_call,
    strip_llama3_leading_sentinels,
    strip_tool_markup as _shared_strip_tool_markup,
)
from core.tool_healing import (
    _markerless_promotable,
)

from utils.native_path_leases import child_env_without_native_path_secret
from utils.child_stdio import utf8_child_env
from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
from utils.log_retention import prune_log_dir
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)
from utils.process_lifetime import child_popen_kwargs as _child_popen_kwargs
from utils.process_lifetime import is_signalable_pid as _is_signalable_pid
from utils.process_lifetime import spawn_on_lifetime_thread as _spawn_on_lifetime_thread
from core.inference.tool_call_parser import (
    BUDGET_EXHAUSTED_NUDGE,
    MAX_ACT_REPROMPTS as _MAX_REPROMPTS,
    NUDGE_TOOL_CALLS_STATUS as _NUDGE_TOOL_CALLS_STATUS,
    REPROMPT_MAX_CHARS as _REPROMPT_MAX_CHARS,
    is_reprompt_repeat as _is_reprompt_repeat,
    is_reprompt_restatement as _is_reprompt_restatement,
    is_short_intent_without_action as _is_short_intent_without_action,
    continue_after_length_message as _continue_after_length_message,
    repeated_result_message as _repeated_result_message,
    starved_result_message as _starved_result_message,
    reprompt_to_act_message as _reprompt_to_act_message,
    reasoning_cap_spent_message as _reasoning_cap_spent_message,
    thinking_exhausted_message as _thinking_exhausted_message,
    unfinished_thought_progress as _unfinished_thought_progress,
)
from core.inference.passthrough_healing import nudge_enabled as _nudge_enabled
from core.inference.repetition_guard import is_repetition_dominated
from core.inference.mcp_images import (
    DETACHED_IMAGE_TURN_TEXT as MCP_DETACHED_IMAGE_TURN_TEXT,
    IMAGE_TURN_TEXT as MCP_IMAGE_TURN_TEXT,
    MAX_MODEL_IMAGES as MCP_MAX_MODEL_IMAGES,
    MAX_TOTAL_MODEL_IMAGES as MCP_MAX_TOTAL_MODEL_IMAGES,
    append_image_turn as append_mcp_image_turn,
    is_image_tool as mcp_is_image_tool,
    mentions_images as mcp_images_mentioned_in,
)
from core.inference.tool_loop_controller import (
    _WORKSPACE_READ_TOOLS,
    BROWSER_TOOL_NAMES,
    ToolLoopController,
    _reruns_after_new_work,
    append_deferred_nudges,
    awaiting_approval_status,
    deferred_nudge_text,
    provisional_tool_provenance,
    restore_replayed_tool_results,
    with_appended_text,
    tool_call_limit_nudge,
)
from state.tool_approvals import (
    DECISION_EXPIRED,
    TOOL_APPROVAL_EXPIRED_MESSAGE,
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    decision_reason,
    wait_tool_decision,
)
from utils.paths.path_utils import _is_wsl, is_appledouble_metadata
from utils.code_integrity import code_integrity_block_reason, code_integrity_user_message
from utils.first_token_timeout import (
    finish_first_token_wait,
    socket_timeout,
    start_first_token_wait,
)

# The leaf module, not utils.models: importing anything from that package runs its __init__,
# which pulls in model_config and therefore PyYAML. This is the chat backend, imported wherever
# Unsloth's Python is, so it must not make the whole models package a hard import dependency.
from utils.gguf_archs import (
    SPEECH_GGUF_ARCHS as _SPEECH_GGUF_ARCHS,
    is_audio_cpp_gguf_architecture,
    is_speech_gguf_architecture,
)

from . import (
    compaction as _compaction,
    log as _log,
    core_types as _core_types,
    server_log as _server_log,
    binary_runtime as _binary_runtime,
    answer_artifacts as _answer_artifacts,
    tool_loop_text as _tool_loop_text,
    gguf_cache as _gguf_cache,
    swa_cache as _swa_cache,
    hf_offline as _hf_offline,
    reasoning as _reasoning,
    drafters as _drafters,
    load_markers as _load_markers,
    memory_policy as _memory_policy,
    launch_flags as _launch_flags,
    placement_args as _placement_args,
    draft_args as _draft_args,
    batch_args as _batch_args,
    moe_cache as _moe_cache,
    spec_args as _spec_args,
    timings as _timings,
    launch_support as _launch_support,
    backend_properties as _backend_properties,
    backend_binary as _backend_binary,
    backend_capabilities as _backend_capabilities,
    backend_gpu_devices as _backend_gpu_devices,
    backend_gpu_memory as _backend_gpu_memory,
    backend_gpu_fit as _backend_gpu_fit,
    backend_kv_cache as _backend_kv_cache,
    backend_download as _backend_download,
    backend_process_io as _backend_process_io,
    backend_diffusion as _backend_diffusion,
    backend_preflight as _backend_preflight,
    backend_start_failures as _backend_start_failures,
    backend_launch_retries as _backend_launch_retries,
    backend_load as _backend_load,
    backend_shutdown as _backend_shutdown,
    backend_spill as _backend_spill,
    backend_slots as _backend_slots,
    backend_health as _backend_health,
    backend_generation as _backend_generation,
    backend_tool_loop as _backend_tool_loop,
    backend_tokens as _backend_tokens,
    backend as _backend,
)

from .compaction import (
    _backend_supports_tools,
    _memory_tool_withheld,
    _can_reset_epoch,
    _archive_is_degraded,
    _compaction_fit_kwargs,
    _request_uses_checkpoint,
    _fit_context,
    _fit_rolling_remembered,
    _fit_with_instruction_pins,
    _DEFAULT_MAX_TOKENS_FLOOR,
    _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
    _conversation_recall_reserve,
    _keeps_compaction_boundary,
    _archive_branch_transcript,
    _branch_boundary,
    _records_boundary,
    _effective_headroom_ratio,
    _boundary_metadata,
    _branch_non_system,
    _branch_boundary_anchor,
    _ANCHOR_ARGS_CHARS,
    _ANCHOR_TEXT_CHARS,
    _anchor_text,
    _archive_message_text,
    _archive_content_on_branch,
    _row_replayed,
    _row_ended_early,
    _archive_branch_chain,
    _archive_as_wire,
    _CompactionBranchState,
    _reply_text,
    _row_truncation,
    _RESUME_HOPS,
    _RESUME_SIBLINGS,
    _max_tokens_cut,
    _row_key,
    _resumable_cuts,
    _resumed_reply,
    _compaction_branch_states,
    _sticky_compaction_state,
    _sticky_compaction_details,
    _sticky_compaction_manual,
    _sticky_compaction_boundary,
    _prefix_user_text,
    _retrieval_budget,
    _recall_top_k,
    _archive_and_recall,
    _inject_recall,
)
from .log import logger
from .core_types import (
    _GGUF_AUDIO_READ_TIMEOUT,
    LlamaServerNotFoundError,
    _gpu_short_message,
    GpuMemoryShortError,
    _gpu_plan_mib,
    _net_of_held_vram,
    _serving_backends,
    _serving_backends_lock,
    register_serving_backend,
    unregister_serving_backend,
    _serving_backends_snapshot,
    GgufDownloadCancelled,
    _LlamaStreamCancelled,
    _GPU_DEVICE_LOST_MARKERS,
    _DEVICE_LOST_LOG_GRACE_S,
    _is_gpu_device_lost,
    _CombinedCancelEvent,
    GgufLoadIntent,
    _CpuFallbackRuntime,
    LLAMA_SERVER_NOT_FOUND_DETAIL,
    _SPEC_ENV_VARS,
    _PERMANENT_HUB_ERRORS,
    _UNRECOVERABLE_LOCAL_ERRNOS,
    _SPEC_KIND_CAPABILITY,
    _LLAMA_RANDOM_SEED,
    _EMPTY_THINK_PROBE_LOCK,
    _apply_seeded_llama_request,
    _SLOT_SAVE_MAX_BYTES,
    _SLOT_SAVE_HTTP_TIMEOUT,
)
from .server_log import (
    _PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR,
    _VULKAN_DIFFUSION_GPU_IDS_ERROR,
    _GPU_OFFLOAD_MARKERS,
    _OFFLOADED_LAYERS_RE,
    _LLAMA_TRACE_VERBOSITY,
    _TRACE_VERBOSITY_HELP_RE,
    _LOG_VERBOSITY_FLAGS,
    _LLAMA_LOG_FULL_BYTES,
    _ANSI,
    _LOG_LEVEL_TOKEN_RE,
    _STDOUT_TRIM_AT,
    _STDOUT_TAIL_KEEP,
    _STDOUT_HEAD_MAX,
    _llama_line_is_warning_or_error,
    _trace_verbosity_args,
    _GPU_MODEL_BUFFER_RE,
    _DEVICE_ROW_RE,
    _GPU_DEVICE_PREFIXES,
    parse_gpu_offload_counts,
    llama_saw_gpu_device,
    classify_gpu_offload_lines,
)
from .binary_runtime import (
    _wsl_system_rocm_lib_dirs,
    _bundled_hip_present,
    _native_linux_system_rocm_lib_dirs,
    _IGPU_HOST_RESERVE_MIB,
    _HOST_RAM_HEADROOM_MIB,
    _CARVEOUT_ADVICE_MIN_HOST_GB,
    _CARVEOUT_ADVICE_HOST_FRACTION,
    _CARVEOUT_NOMINAL_SLACK_GB,
    _PAGEABLE_OVERRIDE_NOTE,
    _PAGEABLE_OVERRIDE_TEXT_ONLY_NOTICE,
    _CGROUP_ROOT,
    _PROC_SELF_CGROUP,
    _apply_igpu_host_reserve_mib,
    _resolve_llama_binary,
    _llama_lib_dir,
    _INSTALLER_WRAPPER_RE,
    _is_installer_entrypoint,
    _loader_path_var,
    _prepend_loader_dir,
    _CPU_RUNTIME_OWNER_FILE,
    _LOAD_PROBE_STATE,
    _load_probe_memo_armed,
    _load_probe_memo_get,
    _load_probe_memo_put,
    _arm_load_probe_memo,
    _load_probe_memo_scope,
    _MISSING,
    _SELF_EVIDENTLY_DISCRETE,
    _NEEDS_ROCM_CLASSIFICATION,
    _DEVICE_VISIBILITY_ENV,
    _LISTED_DEVICE_RE,
    _LIST_DEVICES_HEADER,
    _parse_listed_devices,
    _GGML_GPU_BACKEND_RE,
    _cpu_runtime_owner_alive,
    _sweep_abandoned_cpu_runtimes,
    _lib_dir_has_ggml_backend,
    _is_external_link,
)
from .answer_artifacts import (
    _CLOSED_CODE_FENCE,
    _CLOSED_MARKUP_ARTIFACT,
    _HAS_ANSWER_ARTIFACT,
    _FENCE_RUN_RE,
    _BLOCKQUOTE_PREFIX,
    _FENCE_INFO_STRING_RE,
    _LIST_MARKER_ONLY,
    _ONE_QUOTE_MARKER,
    _has_unclosed_code_fence,
    _has_unclosed_markup_block,
    _EMPTY_MARKUP_SKELETON,
    _DOCTYPE_PREFIX,
    _is_empty_markup_skeleton,
    _is_blank_fence,
    _first_real_artifact,
    _strip_markup_outside_fences,
    _THINK_OPEN_RE,
    _THINKING_OPEN_RE,
    _BRACKET_THINK_OPEN_RE,
    _artifact_spans,
    _find_outside_artifacts,
    _text_outside_think,
    _has_answer_artifact,
)
from .tool_loop_text import (
    _RESPAWN_REAP_GRACE_S,
    _finalize_reasoning_only_cumulative,
    _ends_inside_quote,
    _quote_cut_event,
    _PROVISIONAL_ARGS_MIN_CHARS,
    _DEFAULT_STREAM_STALL_TIMEOUT_S,
    _MAX_TOOL_CALLS_PER_TURN,
    _MAX_LENGTH_CONTINUATIONS,
    _CONTINUE_AFTER_LENGTH_STATUS,
    _CONTINUE_TRUNCATED_ANSWER_STATUS,
    _MIN_USEFUL_RESULT_TOKENS,
    _FALLBACK_IMAGE_TOKENS,
    _mcp_image_tokens,
    _MAX_IDENTICAL_TOOL_RESULTS,
    _UNSET,
    _WINDOW_NOTICE_MARKER,
    _ZERO_ROOM_MARKER,
    _is_window_notice,
    _note_before_envelopes,
    _FAILURE_SCAN_TAIL_CHARS,
    _FAILURE_QUOTE_CHARS,
    _FORCED_PLAN_INTENT,
    _FINAL_ANSWER_SIGNAL,
    _ANSWER_PIVOT,
    _gguf_active_tool_names,
    _GGUF_REHEARSAL_ARGS_RE,
    _gguf_rehearsal_signal_pos,
    _gguf_has_genuine_tool_signal,
    _TEXT_TOOL_NAME_RE,
    _TEXT_TOOL_GEMMA_RE,
    _TEXT_TOOL_REHEARSAL_RE,
    _sniff_text_tool_name,
    _is_rehearsal_prefix,
    _held_rehearsal_tail_len,
    _should_suppress_forced_no_tool_output,
)
from .gguf_cache import (
    _SHARD_FULL_RE,
    _SHARD_RE,
    _gguf_snapshot_files,
    _cached_hf_snapshot_file,
    _snapshot_has_all_shards,
    _resolve_repo_id_casing,
    _cached_gguf_snapshots,
    _cached_colocated_split_main,
    _cached_variant_resolution,
    _cached_variant_candidates,
    _cached_variant_sizes,
    _cached_candidate_matches_revision_size,
    _cached_complete_candidate,
    cached_gguf_for_load,
    _snapshot_dir_of,
    _hub_cache_dir_for_snapshot_path,
    _companion_snapshot_sibling,
    _pick_dspark,
    _MTP_SHARD_SUFFIX_RE,
    _is_published_mtp_drafter_name,
    _mtp_head_borrows,
    _pick_mtp,
    _pick_mtp_root_only,
    _mtp_drafter_loads_standalone,
    _pick_mmproj,
    _hub_download_in_flight,
    _hub_download_blocks_gguf_load,
    _drafter_set_bytes,
    _gguf_extra_shards,
    _quant_label_for_endian,
    _gguf_files_for_variant,
    _resolve_variant_gguf_files,
)
from .swa_cache import (
    _BOOTSTRAP_SWA_DEFAULTS,
    _SWA_CACHE_LOCK,
    _swa_cache_path,
    _load_swa_cache,
    _save_swa_cache,
    _period_from_layer_types,
    _swa_entry_from_layer_types,
    _SWA_CONFIRMED_MISS,
    _SWA_MISSED_REPOS_KEY,
    _swa_missed_repos,
    _remember_swa_repo_miss,
    _fetch_swa_entry_from_hf,
    _arch_aliases,
    _swa_entry_from_config_obj,
    _SWA_PATTERN_SOURCE_RE,
    _resolve_swa_entry_from_transformers,
    _resolve_swa_pattern,
)
from .hf_offline import (
    _hf_endpoint_host,
    _probe_dns_dead,
    _hf_env_offline,
    _hf_unreachable,
    _MISS,
    _GGUF_UINT_WIDTHS,
    _hf_offline_if_unreachable,
    _hf_offline_if_unreachable_for,
    _hf_repo_from_url,
    _extract_model_size_b,
)
from .reasoning import (
    _REASONING_EFFORT_SCALE,
    _QWEN38_MODEL_RE,
    _extract_reasoning_effort_levels,
    _REASONING_EFFORT_CLAMPS_LOGGED,
    _clamp_reasoning_effort_to_levels,
    _GGUF_METADATA_LOGGED,
    _GGUF_METADATA_LOGGED_MAX,
    _GGUF_METADATA_LOGGED_LOCK,
    _note_gguf_metadata_read,
    detect_reasoning_flags,
    _INKLING_REASONING_EFFORT,
    _MUSE_SPARK_REASONING_EFFORT,
    _coerce_reasoning_effort,
)
from .drafters import (
    _GEMMA_MTP_FAMILY_RE,
    _is_gemma_mtp_family,
    _is_gemma_mtp_name,
    _is_mtp_model_name,
    _DRAFTER_KINDS,
    _DRAFTER_DIR_KINDS,
    _DRAFTER_DISPLAY_LABELS,
    _drafter_path_kind,
    _IMATRIX_TOKEN_RE,
    _is_companion_gguf_path,
    _is_mtp_only_drafter_path,
    _is_dspark_drafter_path,
    _is_dflash_drafter_path,
    _is_root_dflash_drafter_path,
    _BIG_ENDIAN_GGUF_FILENAME_RE,
    _GGUF_KNOWN_QUANT_RE,
    _DSPARK_BEFORE_QUANT_RE,
    _is_big_endian_gguf_path,
)
from .load_markers import (
    _LOADS_IN_FLIGHT,
    _LOADS_IN_FLIGHT_LOCK,
    gguf_load_in_flight,
    chat_load_in_flight,
    chat_load_active,
    zero_vram_chat_load,
    hf_gguf_load_in_flight,
    _with_gguf_load_marker,
)
from .memory_policy import (
    _MTP_MIN_SIZE_B,
    _CTX_FIT_VRAM_FRACTION,
    _APPLE_UNIFIED_MEMORY_FRACTION,
    _APPLE_WIRED_CEILING_FRACTION,
    _FIT_MIN_CTX,
    _FIT_FLOOR_MIN_CTX,
    _LLAMA_FIT_MIN_CTX,
    _TP_UNMEASURED_CTX,
    _AUTO_OFFLOAD_CTX,
    _AMD_SMI_TOTAL_TOLERANCE,
    _MTP_VRAM_RESERVE_FRAC,
    _VRAM_FLOOR_RESERVE_MIB,
    _LLAMA_FIT_TARGET_DEFAULT_MIB,
    _WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB,
    _WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION,
    _vram_reserve_floor_mib,
    _vram_usable_mib,
    _model_memory_settings_or_none,
    _active_vram_fraction,
    _kv_bytes_per_elem,
    _GPU_UNACCELERATED_CACHE_TYPES,
    _kv_cache_gpu_fallback_warning,
    _ASSUMED_MAX_VOCAB,
    _ASSUMED_MAX_ACTIVATION_WIDTH,
    _pad_kv_cells,
    _kv_cache_cell_layout,
    _env_main_cache_type_for_budget,
    _extra_args_main_cache_type_for_budget,
    _effective_main_cache_types,
    _planned_main_cache_types,
    _VALID_CACHE_TYPES,
    _planned_scratch_cache_type,
    _auto_mode_drops_mtp,
    _MLA_MTP_FAST_ARCHS,
    _FAST_MTP_MIN_CTX,
    _MTP_SLOT_CTX_TRIM,
    _arch_has_fast_mla_mtp,
    _arch_mtp_skips_target_kv_copy,
    _mla_mtp_auto_enabled,
    _extra_args_set_spec_type,
    _GPU_OFFLOAD_OVERRIDE_FLAGS,
    _CPU_MOE_COUNT_FLAGS,
    _CPU_FFN_COUNT_FLAGS,
    _FIT_CTX_FLAGS,
    _FIT_CONTROL_FLAGS,
    _FIT_CONTROL_ENV_VARS,
    _CPU_PLACEMENT_PRESENCE_FLAGS,
    _CPU_PLACEMENT_FLAGS,
    _VRAM_FIT_VOIDING_FLAGS,
    _VRAM_FIT_VOIDING_ENV_VARS,
    _THREAD_OVERRIDE_FLAGS,
    _GENERATION_CPU_AFFINITY_FLAGS,
    _TENSOR_SPLIT_FLAGS,
    _extra_args_set_any_flag,
)
from .launch_flags import (
    _linux_math_core_count,
    _spilled_decode_threads,
    _strip_flag_pairs,
    _DEFAULT_LLAMA_N_BATCH,
    _DEFAULT_LLAMA_N_UBATCH,
    _MMPROJ_NON_CAUSAL_IMAGE_TOKENS,
    _MMPROJ_UNKNOWN_UBATCH,
    _GEMMA4V_CAUSAL_TEXT_N_EMBD,
    _LLAMA_ARG_TRUE_VALUES,
    _LLAMA_ARG_FALSE_VALUES,
    _LLAMA_ARG_AUTO_VALUES,
    _LLAMA_ARG_TRUE_OR_AUTO_VALUES,
    _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES,
    _diffusion_manual_ngl,
    _scale_diffusion_required_gb,
    _DIFFUSION_RUNNER_OVERHEAD_MIB,
    _DIFFUSION_FIT_CTX,
    _diffusion_auto_ngl,
    _source_declares_option,
    _shim_supports_ngl,
    _visual_server_splits_gpus,
    _binary_has_marker,
    _swa_full_from_args_or_env,
    _reasoning_from_env,
    _reasoning_effort_from_env,
    _env_asks_for_the_native_context,
    _env_ctx_checkpoints_override,
    _kv_offload_from_args,
    _is_positive_int,
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
    _env_fixes_gpu_layers,
    _pipeline_parallel_disabled_by_args,
    _kv_unified_from_args,
    _FLASH_ATTN_INCOMPATIBLE_ARCHITECTURES,
    _architecture_forces_flash_attn_off,
    _planned_flash_attn_state,
    _user_fit_disabled,
    _placement_is_fitter_proof,
    _reserved_flash_attn_state,
    _asked_for_auto_flash_attn,
    _flash_attn_enabled_from_args,
    _effective_spec_type,
    _child_spec_env,
    _TARGET_ROLLBACK_SPEC_TYPES,
    _SSM_EVERY_LAYER_ARCHS,
    _SSM_FFN_FREE_LAYER_ARCHS,
    _TARGET_KV_EXCLUDES_NEXTN_ARCHS,
    _accumulated_spec_types,
    _extra_args_requests_mtp,
    _extra_args_requests_dspark,
    _dflash_drafter_breaks_on_images,
    _extra_args_requests_dflash,
)
from .placement_args import (
    _metal_device_is_paravirtual,
    _paravirtual_probe_answered,
    _paravirtual_draft_ngl_flag,
    _argv_keeps_projector_on_gpu,
    _paravirtual_mmproj_pinnable,
    ctx_checkpoints_allocated,
    ctx_checkpoints_default_for_caps,
    effective_ctx_checkpoints_for_caps,
    _idle_slot_clearing_active,
    _mmproj_env_is_audio_only,
    _MMPROJ_OFFLOAD_FLAGS,
    _extra_args_set_mmproj_offload,
    _extra_args_mmproj_offload_value,
    _MMPROJ_OFFLOAD_ENV_VAR,
    _MMPROJ_OFFLOAD_NEG_ENV_VAR,
    _LLAMA_VIDEO_FPS,
    _VIDEO_FPS_ENV_VAR,
    _video_fps_flags,
    requested_video_fps,
    _env_sets_mmproj_offload,
    _env_mmproj_offload_value,
    _resolved_mmproj_offload,
    _OVERRIDE_TENSOR_FLAGS,
    _paravirtual_strip_gpu_overrides,
    ParavirtualPlacement,
    paravirtual_normalized_request,
    _paravirtual_split_mode_pin,
)
from .draft_args import (
    _extra_args_requests_separate_draft,
    _extra_args_spec_draft_n_max,
    _extra_args_mtp_draft_path,
    _HF_DRAFT_FLAGS,
    _LOCAL_DRAFT_FLAGS,
    _extra_args_mtp_draft_source,
    _normalized_load_mode,
    _BREAKDOWN_DEVICE_ROW_RE,
    _BREAKDOWN_HOST_ROW_RE,
    _parse_metal_memory_breakdown,
    _VALID_KV_CACHE_TYPES,
    _extra_args_draft_cache_types,
    _DRAFT_GPU_LAYER_FLAGS,
    _extra_args_draft_gpu_layers,
    _draft_is_split_across_host,
    _SIDECAR_ADAPTER_FLAGS,
    _sidecar_adapter_paths,
    _sidecar_adapter_bytes,
    _extra_args_draft_offloaded_to_cpu,
    _extra_args_device,
    _extra_args_main_device,
    _without_subsequence,
    _count_subsequence,
    _subsequence_index,
    _CPU_DEVICE_VALUES,
    _pass_through_overrides_placement,
    _device_selection_is_cpu,
    _split_mode_confines_to_one_device,
    _extra_args_draft_device,
    _GPU_DEVICE_TOKEN_RE,
    _COMPANION_DEVICE_FLAG_GROUPS,
    _widen_pin_ids_for_companion_devices,
    _extra_args_draft_device_pin,
)
from .batch_args import (
    _emitted_n_batch,
    _with_parallel_slots,
    _extra_args_split_mode,
    _extra_args_tensor_split,
    _extra_args_n_parallel,
    _named_batch_sizes,
    _extra_args_n_ubatch,
    _read_gguf_embedding_length,
    _unknown_projector_ubatch,
    _mmproj_required_ubatch,
    _launch_required_ubatch,
    _batch_ubatch_for_mmproj,
    _embedding_batch_ubatch,
    _MOE_SPILL_N_UBATCH,
    _moe_spill_batch_ubatch,
    _override_targets_host,
    _expert_spill_places_tensors_on_cpu,
)
from .moe_cache import (
    _MOE_CACHE_AUTO_TOKENS,
    _MOE_CACHE_MIN_RESIDENT_SHARE,
    _MOE_CACHE_MAX_RESIDENT_SHARE,
    _MOE_CACHE_WITH_DRAFTER,
    _MOE_SPILL_UB_MAX_RESIDENT_LOSS,
    _MOE_CACHE_ENV,
    _MOE_SPILL_UBATCH_ENV,
    _MOE_SWITCH_ON,
    _MOE_SWITCH_OFF,
    _moe_switch,
    _moe_resident_expert_share,
    _moe_cache_share_admits,
    _moe_spill_raise_keeps_experts,
    _MOE_CACHE_USER_OWNED_FLAGS,
    _MOE_CACHE_USER_OWNED_ENV,
    _MOE_CACHE_ERROR_RE,
    _LAZY_READ_TENSORS,
    _LAZY_READ_AUTO_MIN_BYTES,
    _NEXTN_SKIPPED_WITHOUT_MTP_ARCHS,
    _LAZY_MODE_SPELLINGS,
    _LLAMA_CACHE_RAM_DEFAULT_MIB,
    _prompt_cache_host_bytes,
    _argv_loads_pinned,
    _effective_lazy_mode,
    _lazy_read_bytes,
)
from .spec_args import (
    _build_ngram_mod_flags,
    _build_launch_reasoning_args,
    _build_reasoning_budget_flags,
    _CANONICAL_SPEC_MODES,
    _LEGACY_SPEC_MODE_MAP,
    _trained_drafter_sidecar,
    _canonicalize_spec_mode,
)
from .timings import (
    _backfill_usage_from_timings,
    _context_usage_event,
    _llama_chunk_has_generated_output,
    _perf_callback_wants_timings,
    _report_live_llama_timings,
)
from .launch_support import (
    CountAborted,
    _PROC_ROOT,
    _metal_capable_host,
    _expanded_user_path,
    _llama_server_api_key_enabled,
    _write_direct_stream_key,
    _llama_server_key_via_env,
    _llama_server_key_launch,
    _extra_args_have_tensor_split,
    _local_ssl_context,
    _SIGKILL_REAP_TIMEOUT_S,
    _QUIT_SIGNAL_EXIT_GRACE_S,
)
from .backend import LlamaCppBackend

_SUBMODULES = (
    _compaction,
    _log,
    _core_types,
    _server_log,
    _binary_runtime,
    _answer_artifacts,
    _tool_loop_text,
    _gguf_cache,
    _swa_cache,
    _hf_offline,
    _reasoning,
    _drafters,
    _load_markers,
    _memory_policy,
    _launch_flags,
    _placement_args,
    _draft_args,
    _batch_args,
    _moe_cache,
    _spec_args,
    _timings,
    _launch_support,
    _backend_properties,
    _backend_binary,
    _backend_capabilities,
    _backend_gpu_devices,
    _backend_gpu_memory,
    _backend_gpu_fit,
    _backend_kv_cache,
    _backend_download,
    _backend_process_io,
    _backend_diffusion,
    _backend_preflight,
    _backend_start_failures,
    _backend_launch_retries,
    _backend_load,
    _backend_shutdown,
    _backend_spill,
    _backend_slots,
    _backend_health,
    _backend_generation,
    _backend_tool_loop,
    _backend_tokens,
    _backend,
)

# Submodules that call ``LlamaCppBackend.<staticmethod>`` by name. The class is composed last, so
# it is bound into them here, as the one module's global used to be.
for _module in (
    _backend_binary,
    _backend_generation,
    _backend_gpu_devices,
    _backend_gpu_fit,
    _backend_gpu_memory,
    _backend_health,
    _backend_kv_cache,
    _backend_launch_retries,
    _backend_load,
    _backend_process_io,
    _backend_properties,
    _backend_shutdown,
    _backend_start_failures,
    _backend_tokens,
    _load_markers,
):
    _module.LlamaCppBackend = LlamaCppBackend
del _module

# Module globals a function rebinds with ``global``: read live from the owning module rather
# than copied here, so ``llama_cpp._SWA_CACHE`` is the dict the cache loader installed.
_LIVE = {
    "_CHAT_LOADS_IN_FLIGHT": _load_markers,
    "_SWA_CACHE": _swa_cache,
}

_UNBOUND = object()


class _Facade(_types.ModuleType):
    """The package module: attribute writes reach every submodule alias of the old object."""

    def __getattr__(self, name):
        owner = _LIVE.get(name)
        if owner is None:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        return getattr(owner, name)

    def __setattr__(self, name, value):
        owner = _LIVE.get(name)
        if owner is not None:
            setattr(owner, name, value)
            return
        old = self.__dict__.get(name, _UNBOUND)
        super().__setattr__(name, value)
        if old is _UNBOUND:
            return
        for module in _SUBMODULES:
            if module.__dict__.get(name, _UNBOUND) is old:
                module.__dict__[name] = value

    def __delattr__(self, name):
        owner = _LIVE.get(name)
        if owner is not None:
            delattr(owner, name)
            return
        super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(_LIVE))


_sys.modules[__name__].__class__ = _Facade
