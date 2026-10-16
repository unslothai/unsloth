# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, load_model and load intent matching."""

import contextlib
import functools
import os
import re
import subprocess
import sys
import threading
import time

from dataclasses import replace
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Union,
)
from core.inference.llama_server_args import (
    _CACHE_TYPE_K_FLAGS,
    _CACHE_TYPE_V_FLAGS,
    _CTX_CHECKPOINTS_FLAGS,
    _effective_tensor_parallel,
    _flag_name,
    _GPU_LAYER_FLAGS,
    _last_flag_value,
    apply_load_mode_policy,
    apply_model_memory_policy,
    extra_args_disable_mmproj,
    extra_args_mmproj_auto,
    extra_args_select_load_mode,
    fit_is_effectively_on,
    fit_is_enabled_in,
    fit_target_margin_in,
    MANAGED_DIO_FLAGS,
    memory_env_selects_load_mode,
    memory_state_satisfies_settings,
    no_reserve_requires_dio,
    parse_cache_override,
    parse_cache_override_per_axis,
    parse_ctx_override,
    parse_split_mode_override,
    resolve_ctx_checkpoints,
    resolve_launch_load_mode,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
    resolve_reasoning_budget_message_with_env,
    resolve_reasoning_budget_with_env,
    resolve_requested_ctx,
    scrub_denied_env,
    scrub_memory_env,
    strip_context_only,
    strip_shadowing_flags,
    strip_split_mode_only,
)
from utils.gpu_memory_events import (
    invalidate_gpu_memory as _invalidate_gpu_memory,
    invalidates_gpu_memory as _invalidates_gpu_memory,
)
from utils.log_retention import prune_log_dir
from utils.process_lifetime import (
    child_popen_kwargs as _child_popen_kwargs,
    spawn_on_lifetime_thread as _spawn_on_lifetime_thread,
)
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .batch_args import (
    _batch_ubatch_for_mmproj,
    _embedding_batch_ubatch,
    _emitted_n_batch,
    _expert_spill_places_tensors_on_cpu,
    _extra_args_n_parallel,
    _extra_args_n_ubatch,
    _launch_required_ubatch,
    _moe_spill_batch_ubatch,
    _named_batch_sizes,
    _read_gguf_embedding_length,
    _with_parallel_slots,
)
from .binary_runtime import (
    _DEVICE_VISIBILITY_ENV,
    _arm_load_probe_memo,
)
from .core_types import (
    GgufLoadIntent,
    GpuMemoryShortError,
    LLAMA_SERVER_NOT_FOUND_DETAIL,
    LlamaServerNotFoundError,
    _CombinedCancelEvent,
    _SPEC_ENV_VARS,
    _gpu_plan_mib,
    _gpu_short_message,
    _net_of_held_vram,
)
from .draft_args import (
    _COMPANION_DEVICE_FLAG_GROUPS,
    _LOCAL_DRAFT_FLAGS,
    _VALID_KV_CACHE_TYPES,
    _device_selection_is_cpu,
    _draft_is_split_across_host,
    _extra_args_device,
    _extra_args_draft_cache_types,
    _extra_args_draft_offloaded_to_cpu,
    _extra_args_main_device,
    _extra_args_mtp_draft_path,
    _extra_args_requests_separate_draft,
    _extra_args_spec_draft_n_max,
    _pass_through_overrides_placement,
    _sidecar_adapter_bytes,
    _subsequence_index,
    _widen_pin_ids_for_companion_devices,
    _without_subsequence,
)
from .drafters import (
    _DRAFTER_DISPLAY_LABELS,
    _is_gemma_mtp_name,
    _is_mtp_model_name,
)
from .gguf_cache import (
    _mtp_drafter_loads_standalone,
    _resolve_repo_id_casing,
)
from .hf_offline import (
    _extract_model_size_b,
    _hf_offline_if_unreachable,
)
from .launch_flags import (
    _DEFAULT_LLAMA_N_UBATCH,
    _TARGET_ROLLBACK_SPEC_TYPES,
    _accumulated_spec_types,
    _architecture_forces_flash_attn_off,
    _args_place_tensors_on_cpu,
    _child_spec_env,
    _dflash_drafter_breaks_on_images,
    _env_asks_for_the_native_context,
    _env_ctx_checkpoints_override,
    _env_fixes_gpu_layers,
    _env_places_tensors_on_cpu,
    _extra_args_requests_dflash,
    _extra_args_requests_dspark,
    _extra_args_requests_mtp,
    _flash_attn_enabled_from_args,
    _is_positive_int,
    _kv_offload_from_args,
    _kv_unified_from_args,
    _pipeline_parallel_disabled_by_args,
    _placement_is_fitter_proof,
    _planned_flash_attn_state,
    _reasoning_effort_from_env,
    _reasoning_from_env,
    _reserved_flash_attn_state,
    _swa_full_from_args_or_env,
    _user_fit_disabled,
)
from .launch_support import (
    _extra_args_have_tensor_split,
    _llama_server_api_key_enabled,
    _llama_server_key_launch,
    _metal_capable_host,
)
from .load_markers import _with_gguf_load_marker
from .log import logger
from .memory_policy import (
    _AUTO_OFFLOAD_CTX,
    _CTX_FIT_VRAM_FRACTION,
    _FAST_MTP_MIN_CTX,
    _FIT_CTX_FLAGS,
    _FIT_FLOOR_MIN_CTX,
    _FIT_MIN_CTX,
    _GPU_OFFLOAD_OVERRIDE_FLAGS,
    _GPU_UNACCELERATED_CACHE_TYPES,
    _LLAMA_FIT_MIN_CTX,
    _LLAMA_FIT_TARGET_DEFAULT_MIB,
    _MTP_MIN_SIZE_B,
    _MTP_SLOT_CTX_TRIM,
    _MTP_VRAM_RESERVE_FRAC,
    _THREAD_OVERRIDE_FLAGS,
    _VALID_CACHE_TYPES,
    _VRAM_FIT_VOIDING_ENV_VARS,
    _VRAM_FIT_VOIDING_FLAGS,
    _active_vram_fraction,
    _arch_has_fast_mla_mtp,
    _effective_main_cache_types,
    _env_main_cache_type_for_budget,
    _extra_args_main_cache_type_for_budget,
    _extra_args_set_any_flag,
    _extra_args_set_spec_type,
    _kv_cache_gpu_fallback_warning,
    _mla_mtp_auto_enabled,
    _model_memory_settings_or_none,
    _planned_main_cache_types,
    _planned_scratch_cache_type,
    _vram_usable_mib,
)
from .moe_cache import (
    _MOE_CACHE_AUTO_TOKENS,
    _MOE_CACHE_ENV,
    _MOE_CACHE_ERROR_RE,
    _MOE_CACHE_WITH_DRAFTER,
    _MOE_SPILL_UBATCH_ENV,
    _argv_loads_pinned,
    _moe_cache_share_admits,
    _moe_resident_expert_share,
    _moe_spill_raise_keeps_experts,
    _moe_switch,
    _prompt_cache_host_bytes,
)
from .placement_args import (
    _argv_keeps_projector_on_gpu,
    _env_sets_mmproj_offload,
    _extra_args_set_mmproj_offload,
    _idle_slot_clearing_active,
    _metal_device_is_paravirtual,
    _mmproj_env_is_audio_only,
    _paravirtual_draft_ngl_flag,
    _paravirtual_mmproj_pinnable,
    _paravirtual_probe_answered,
    _paravirtual_split_mode_pin,
    _paravirtual_strip_gpu_overrides,
    _resolved_mmproj_offload,
    _video_fps_flags,
    effective_ctx_checkpoints_for_caps,
    paravirtual_normalized_request,
)
from .reasoning import detect_reasoning_flags
from .server_log import (
    _PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR,
    _VULKAN_DIFFUSION_GPU_IDS_ERROR,
    _trace_verbosity_args,
    classify_gpu_offload_lines,
    llama_saw_gpu_device,
    parse_gpu_offload_counts,
)
from .spec_args import (
    _build_launch_reasoning_args,
    _build_ngram_mod_flags,
    _build_reasoning_budget_flags,
    _canonicalize_spec_mode,
    _trained_drafter_sidecar,
)
from .swa_cache import _swa_cache_path

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _LoadMixin:
    @_invalidates_gpu_memory("llama.cpp load")
    @_with_gguf_load_marker
    def load_model(
        self,
        intent: GgufLoadIntent,
        load_cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Start llama-server from one immutable load intent."""

        def _load_cancelled() -> bool:
            return self._cancel_event.is_set() or bool(
                load_cancel_event is not None and load_cancel_event.is_set()
            )

        download_cancel_event = (
            _CombinedCancelEvent(self._cancel_event, load_cancel_event)
            if load_cancel_event is not None
            else self._cancel_event
        )

        gguf_path = intent.gguf_path
        mmproj_path = intent.mmproj_path
        mtp_draft_path = intent.mtp_draft_path
        dspark_draft_path = intent.dspark_draft_path
        dflash_draft_path = intent.dflash_draft_path
        hf_repo = intent.hf_repo
        hf_variant = intent.hf_variant
        hf_token = intent.hf_token
        model_identifier = intent.model_identifier
        is_vision = intent.is_vision
        disable_vision = intent.disable_vision
        n_ctx = intent.n_ctx
        _replayed_ctx_refit = False
        chat_template_override = intent.chat_template_override
        cache_type_kv = intent.cache_type_kv
        speculative_type = intent.speculative_type
        spec_draft_n_max = intent.spec_draft_n_max
        tensor_parallel = intent.tensor_parallel
        gpu_memory_mode = intent.gpu_memory_mode
        gpu_layers = intent.gpu_layers
        n_cpu_moe = intent.n_cpu_moe
        tensor_split = list(intent.tensor_split) if intent.tensor_split is not None else None
        gpu_ids = list(intent.gpu_ids) if intent.gpu_ids is not None else None
        gpu_ids_are_vulkan_ordinals = intent.gpu_ids_are_vulkan_ordinals
        n_threads = intent.n_threads
        n_parallel = intent.n_parallel
        n_batch = intent.n_batch
        n_ubatch = intent.n_ubatch
        load_mode = intent.load_mode
        spec_draft_cache_type = intent.spec_draft_cache_type
        ctx_checkpoints = intent.ctx_checkpoints
        cache_ram = intent.cache_ram
        extra_args = list(intent.extra_args) if intent.extra_args is not None else None
        custom_mmproj = _extra_args_device(extra_args, {"--mmproj", "-mm"})
        if custom_mmproj is not None:
            mmproj_path = self._resolve_launch_mmproj_path(
                model_path = gguf_path or "", mmproj_path = mmproj_path, extra_args = extra_args
            )
            is_vision = True
        preserve_multi_gpu_on_layer = intent.preserve_multi_gpu_on_layer
        reasoning_budget = intent.reasoning_budget
        reasoning_budget_message = intent.reasoning_budget_message
        # Before the serial scope: a queued load still belongs to the lifecycle it
        # was requested in.
        # The process-wide equivalent, for the same reason. Only _begin_server_lifecycle
        # advances the per-instance one, and a helper load owns a backend that never
        # gets it, so an embedded second session would otherwise release this load.
        # Serialise the whole load so concurrent /load calls never leave two
        # llama-server processes alive (#5401 / #5161). Doesn't block /unload.
        self._codec_failure.message = None
        with self._serial_load_scope():
            # Here, not at the spawn: a lock gives a waiter no priority, and the
            # duplicate-adoption phase below kills whatever is loaded -- after a
            # restart, the new lifecycle's model.
            with self._spawn_lock:
                _stale_load = self._spawn_is_stale()
            if _stale_load:
                logger.info("dropping a load left over from the previous server lifecycle")
                return False
            # In-app update swapping binaries: refuse fast (set under this lock,
            # so any in-flight load has drained) instead of using a half-swapped one.
            if getattr(self, "_llama_update_in_progress", False):
                raise RuntimeError("llama.cpp is updating; try again in a moment.")

            intent = self._preserve_cpu_fallback_intent(intent)
            tensor_parallel = intent.tensor_parallel
            gpu_memory_mode = intent.gpu_memory_mode
            gpu_layers = intent.gpu_layers
            n_cpu_moe = intent.n_cpu_moe
            tensor_split = list(intent.tensor_split) if intent.tensor_split is not None else None
            gpu_ids = list(intent.gpu_ids) if intent.gpu_ids is not None else None

            # Read before the pin below rewrites every placement, Auto included, to
            # manual/0. Owning the memory budget is a property of what was asked for,
            # and the three Metal context guards further down skip themselves on it.
            _caller_owns_budget = gpu_memory_mode == "manual" and gpu_layers >= 0

            # A virtualised Apple GPU corrupts every offloaded layer, so pin GGUF
            # inference to CPU there; physical Apple Silicon is untouched. Settled
            # ABOVE the duplicate-load check (and the KV estimates) so the compared
            # placement is the one that launches: the launch records the normalized
            # values, so comparing the raw request would miss the fast path and tear
            # down a healthy CPU server on every repeat Auto /load.
            # Keyed on the hardware, not the request, and applied whole rather than
            # only when the request asks for offload: Manual + 0 layers already places
            # the main model on CPU, but the companions below (mmproj, a separate
            # drafter) do not read --gpu-layers, and an `-ot ".*=Metal"` the route never
            # strips would put weights straight back on the corrupt device.
            # paravirtual_normalized_request is the one definition of that rewrite,
            # shared with both duplicate-load comparators.
            _paravirtual_cpu_forced = _metal_device_is_paravirtual()
            if _paravirtual_cpu_forced:
                _pv_placement = paravirtual_normalized_request(
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    tensor_parallel = tensor_parallel,
                    tensor_split = tensor_split,
                    n_cpu_moe = n_cpu_moe,
                    extra_args = extra_args,
                )
                if (gpu_memory_mode, gpu_layers) != (
                    _pv_placement.gpu_memory_mode,
                    _pv_placement.gpu_layers,
                ):
                    logger.warning(
                        "Forcing gpu_layers=0 for %s: this Mac's Metal device is "
                        "virtualised and offloaded layers produce corrupt output.",
                        model_identifier or gguf_path,
                    )
                if extra_args and _pv_placement.extra_args != list(extra_args):
                    logger.warning(
                        "Dropping offload flags from extra args on a "
                        "virtualised Metal device: %s -> %s",
                        list(extra_args),
                        _pv_placement.extra_args,
                    )
                if tensor_parallel or tensor_split:
                    logger.info(
                        "Virtualised Metal device: dropping tensor split/parallel "
                        "flags (nothing to split on the GPU)"
                    )
                gpu_memory_mode = _pv_placement.gpu_memory_mode
                gpu_layers = _pv_placement.gpu_layers
                n_cpu_moe = _pv_placement.n_cpu_moe
                extra_args = _pv_placement.extra_args
                # Spelled out (not read off _pv_placement, which returns exactly these)
                # so tests/test_tp_vision_regression.py's TP-drop allowlist sees this site.
                tensor_parallel = False
                tensor_split = None
                # The intent is what the duplicate-load check compares and what a retry
                # replays, so carry the rewrite into it rather than the raw request.
                intent = replace(
                    intent,
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    n_cpu_moe = n_cpu_moe,
                    extra_args = tuple(extra_args) if extra_args is not None else None,
                    tensor_parallel = False,
                    tensor_split = None,
                )

            # Fraction of each card this load may claim, resolved ONCE so ranking
            # and fitting price the same number (mixed totals mis-order otherwise);
            # unset means _CTX_FIT_VRAM_FRACTION, the historical behaviour. Under
            # the lock and ahead of the duplicate check, or a queued load would plan
            # with a stale fraction while the check reads the live one, and could
            # evict the resident child for a budget it then fails to apply.
            _vram_frac = _active_vram_fraction()
            # How far this budget moves llama.cpp's own fitter margin, set once the
            # devices are known. Signed: lowering holds more back, raising packs
            # closer to the card. Zero at the default.
            _fit_target_delta_mib = 0.0
            # Set only if that margin actually reached the command line, which is
            # what makes the child budget-priced in a mode the planner skipped.
            _fit_target_priced = False

            def _budget_priced_placement() -> Optional[float]:
                """The fraction this child was sized against, or None if it wasn't.

                Read at call time, because both names it closes over are rebound
                during the load. Every consumer of the fraction is gated on a
                non-empty ``gpus``, so manual mode and a host with no discrete GPU
                plan without it; and an auto Vulkan crash that recovers on CPU
                rewrites the intent to CPU placement while leaving ``gpus``
                populated from the attempt that failed. Recording a budget for
                either would make a later change to the setting evict a healthy
                child for a reload that cannot move anything.
                """
                if intent.cpu_fallback:
                    return None
                if gpus:
                    return _vram_frac
                # Manual + Auto empties gpus to bypass the planner, but --fit-target
                # still carries the budget to llama.cpp's fitter, so that child IS
                # priced. Without this a later change reports no reload needed and
                # the duplicate check adopts a child on the old margin.
                return _vram_frac if _fit_target_priced else None

            # Armed before the duplicate check, not merely cleared: on an inactive
            # backend a save landing in that gap saw no pending value and no live
            # process, so it was told no reload was needed while this load went on to
            # launch against the captured fraction. The load scope hands it back on
            # every exit, the fast path below included.
            self._vram_fraction_pending = _vram_frac
            # _vram_fraction_launched is committed at the launch site instead: the
            # duplicate-load fast path, a cancellation and every failed preflight
            # return with the previous child still serving, so setting it here would
            # claim that stale child carries the new budget and hide its reload.

            # Duplicate /load that raced past the route check: do nothing if the
            # live server already satisfies this request.
            if self.adopt_load_intent_if_matched(intent):
                logger.info(
                    f"load_model: backend already in target state for "
                    f"'{model_identifier}', skipping reload"
                )
                # Retry probe only if a prior attempt didn't finish.
                if not self._audio_probed:
                    try:
                        detected = self._detect_audio_type_strict()
                        self._audio_probed = True
                    except Exception as exc:
                        logger.debug("Fast-path audio probe failed: %s", exc)
                        detected = None
                    applied_audio = (
                        self._apply_detected_audio(detected)
                        if intent.audio_codec_path is None
                        else self._apply_detected_audio(detected, intent.audio_codec_path)
                    )
                    if not applied_audio:
                        return False
                if not self._healthy:
                    return False
                return True

            # Stays armed through the download and planning, which can take minutes.
            # Placement is undecided until the spawn site narrows it: a needless
            # reload prompt beats sizing a child against a replaced budget.
            self._cancel_event.clear()
            # Belongs to this attempt only; a guard that runs before this load's first
            # health wait would otherwise read the previous request's cancellation.
            self._health_wait_cancelled = False
            if _load_cancelled():
                logger.info("Load cancelled before GGUF resolution")
                return False

            # Resolve llama-server now but defer a not-found error: a block-diffusion
            # GGUF uses the diffusion runner, and its arch is only known after the header.
            # Serialize this snapshot with settings writes. If a save wins the
            # guard, this load sees the new folder; if the load wins, the save
            # response compares against this pending old revision and asks for a
            # reload even before Popen makes is_active true.
            from utils.llama_cpp_path_settings import llama_cpp_path_selection_guard

            with llama_cpp_path_selection_guard():
                binary = self._find_llama_server_binary()
                # macOS: launch the resolved server, never a shell entrypoint. SIP
                # purges DYLD_* while starting the protected /bin/sh a wrapper
                # entrypoint runs under, so the loader path built for the child
                # would not survive the wrapper's own exec (#8566). The installer
                # writes a wrapper only when it cannot symlink, and that wrapper
                # does nothing but exec the target with "$@", so this is the same
                # launch minus the shell hop. The resolved path stays inside the
                # managed install, so provenance-aware advice is unaffected.
                # Resolved here rather than at the later reset so every consumer of
                # `binary` below (capability probe, revision stamp) shares one path
                # space; `_binary_changed_since_launch` compares against this one.
                binary = self._exec_path_for_launch(binary)
                self._binary_revision_pending = self._binary_revision(binary)

            # Every capability answer shaping this launch, latching whether any was a
            # guess. Accumulated, not sampled: the decisions straddle the whole load (slot
            # clamp before an HF download, command build after), and with a 30s window a
            # later success would otherwise erase an earlier one that already degraded it.
            _launch_probe_inconclusive = False
            # Set where flash attention ends up off with the argv unable to say
            # so: a build that has no --flash-attn at all, and the crash-recovery
            # retry on a build whose flag takes no value, which disables FA by
            # dropping it. Either way the default below has to carry the answer.
            _flash_attn_known_off = False

            def _launch_caps(bin_path):
                nonlocal _launch_probe_inconclusive
                caps = self.probe_server_capabilities(bin_path)
                if caps.get("mtp_probe_inconclusive"):
                    _launch_probe_inconclusive = True
                return caps

            _reasoning_requested = any(
                self.reasoning_budget_settings_requested(
                    extra_args = extra_args,
                    reasoning_budget = reasoning_budget,
                    reasoning_budget_message = reasoning_budget_message,
                )
            )
            if (
                _reasoning_requested
                and gguf_path
                and Path(gguf_path).is_file()
                and self._gguf_path_is_diffusion(gguf_path, model_identifier)
            ):
                raise ValueError(
                    "Reasoning Budget settings are not supported for DiffusionGemma models."
                )
            # Nothing configured means nothing to reject; _launch_caps probes the binary anyway.
            if _reasoning_requested:
                self.validate_reasoning_budget_capabilities(
                    binary,
                    extra_args = extra_args,
                    reasoning_budget = reasoning_budget,
                    reasoning_budget_message = reasoning_budget_message,
                )
            is_vulkan_backend = self._is_vulkan_backend(binary)
            # Once, and reused everywhere the strip is asked about: the reload
            # comparator has to reach the same answer as the argv it compares against.
            _gpu_ids_own_device_flags = self._gpu_ids_own_placement(gpu_ids)
            _vulkan_ordinal_pin = (
                is_vulkan_backend and bool(gpu_ids) and gpu_ids_are_vulkan_ordinals is not False
            )
            _cpu_only_pin = (
                bool(gpu_ids) and not is_vulkan_backend and self._backend_lacks_gpu_lib(binary)
            )

            # Without --kv-unified an explicit --parallel N splits -c into windows of -c/N, so on a
            # build lacking the flag the default of 4 would quarter every context window for a
            # feature it cannot serve: fall back to one slot. Ahead of the KV estimates so the
            # fit matches what launches.
            if n_parallel > 1 and binary and not _launch_caps(binary).get("supports_kv_unified"):
                logger.warning(
                    "llama-server at %s has no --kv-unified, so %d parallel slots would "
                    "split the context window %d ways. Using 1 slot instead; update "
                    "llama.cpp to run chats in parallel.",
                    binary,
                    n_parallel,
                    n_parallel,
                )
                n_parallel = 1  # allow-slot-clamp: the build cannot serve more

            # MTP does not clamp the slots. #7717 did, for a draft-acceptance collapse, but on
            # b10310 four MTP slots hit 1.97x the batch throughput of one, beating four without.
            # llama.cpp#26031 reports concurrent MTP garbling on hybrid archs; it did not
            # reproduce here (qwen35 and qwen35moe on CUDA, 24 staggered uneven requests).

            # ── Vulkan-ordinal preflight (BEFORE the Phase 1 kill) ────────
            # An explicit Vulkan pin the ggml probe never enumerated cannot be honored.
            # Validate it ABOVE the kill so an invalid selection leaves the live model
            # untouched: CUDA ids are range-checked at the route, but Vulkan ordinals are
            # not, so a stale gpu_ids=[99] used to kill the server then 400, leaving
            # nothing running (#7239). _get_gpu_memory needs only the binary (safe pre-
            # download) and reuses the later fit's issubset logic. Guarded on a found
            # Vulkan build + a pin so a deferred not-found stays deferred for diffusion,
            # and on the ordinals being unconfirmed: a caller-confirmed physical CUDA pin
            # is not in this namespace and must not be range-checked against it.
            if _vulkan_ordinal_pin and binary:
                _pf_wanted = {int(x) for x in gpu_ids}
                _pf_probed = {g[0] for g in self._get_gpu_memory(binary)}
                if not _pf_wanted.issubset(_pf_probed):
                    raise ValueError(
                        f"Requested Vulkan GPU ordinal(s) {sorted(_pf_wanted)} not "
                        f"present. Available Vulkan devices: {sorted(_pf_probed)}."
                    )

            # Classify before killing the healthy server (#7205); Phase 2 reuses this
            # path. Diffusion changes the meaning or validity of requested placement,
            # so both the remote and the local branch settle it above the teardown.
            # A zero-layer diffusion pin this shim cannot apply is refused; probe
            # once here so the HF case can be settled above the teardown as well.
            _pv_diffusion_unpinnable = bool(
                _paravirtual_cpu_forced and not self.diffusion_split_supported()
            )
            _preflight_model_path = None
            self._pending_variant_fallback = None
            if hf_repo and (_vulkan_ordinal_pin or _cpu_only_pin or _pv_diffusion_unpinnable):
                _resolved_repo = _resolve_repo_id_casing(hf_repo)
                if _resolved_repo != hf_repo:
                    logger.info(
                        "Using cached repo_id casing '%s' for requested '%s'",
                        _resolved_repo,
                        hf_repo,
                    )
                    hf_repo = _resolved_repo
                with _hf_offline_if_unreachable():
                    _preflight_model_path = self._download_gguf(
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        hf_token = hf_token,
                        cancel_event = download_cancel_event,
                    )
                _preflight_is_diffusion = self._gguf_path_is_diffusion(
                    _preflight_model_path, model_identifier
                )
                if _preflight_is_diffusion:
                    if _vulkan_ordinal_pin:
                        raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)
                    if _pv_diffusion_unpinnable:
                        raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)
                elif _cpu_only_pin:
                    raise ValueError(
                        f"Requested gpu_ids {list(gpu_ids)} but the llama.cpp build has "
                        "no GPU backend (CPU-only build); it would ignore the pin and run "
                        "on CPU. Omit gpu_ids to run on CPU."
                    )
            elif is_vulkan_backend and gpu_ids and gguf_path and not hf_repo:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

            # Only an ordinal pin is ambiguous for the diffusion runner; a
            # caller-confirmed physical pin keeps its CUDA placement path.
            if _vulkan_ordinal_pin and gguf_path and not hf_repo:
                self._reject_vulkan_diffusion_gpu_ids_before_teardown(
                    gguf_path,
                    model_identifier,
                )
            # Same for the unpinnable diffusion split: _start_diffusion_server enforces
            # it everywhere, but only after the teardown, so settle the local case here
            # and leave the running server alone.
            if (
                _pv_diffusion_unpinnable
                and gguf_path
                and not hf_repo
                and self._gguf_path_is_diffusion(gguf_path, model_identifier)
            ):
                raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)

            # An image / video GGUF already on disk is refused HERE, before the teardown:
            # the header answers this without llama-server, so there is no reason to make
            # the user pay their resident chat model for it.
            if gguf_path and not hf_repo and Path(gguf_path).is_file():
                _early_non_chat = self._non_chat_gguf_refusal_for_path(gguf_path, model_identifier)
                if _early_non_chat:
                    logger.error(
                        "Refusing non-chat GGUF before teardown: %s (%s)",
                        _early_non_chat,
                        gguf_path,
                    )
                    raise ValueError(_early_non_chat)

            # The same refusal for a REPO load, which is what the Model Hub actually sends:
            # the route resolves every Hub model to hf_repo, so leaving this to the
            # post-download check killed the live model first in the reported case. The file
            # does not exist yet, but a header-sized byte range decides it (see
            # _remote_non_chat_gguf_refusal). When a preflight above already fetched the
            # file, judge that instead -- same verdict, no request. Fails open into the
            # post-download check below, which stays as the backstop.
            if hf_repo:
                # Same offline guard the download below uses: the probe asks the Hub twice
                # before reading any local header (the file listing, then the cached
                # candidate's revision sizes), and each would wait out its retry backoff,
                # putting two timeouts in front of a cached load that needs no network. The
                # guard is memoised and only bites when the Hub really is unreachable.
                with _hf_offline_if_unreachable():
                    _early_non_chat = (
                        self._non_chat_gguf_refusal_for_path(
                            _preflight_model_path, model_identifier
                        )
                        if _preflight_model_path
                        else self._remote_non_chat_gguf_refusal(
                            hf_repo = hf_repo,
                            hf_variant = hf_variant,
                            hf_token = hf_token,
                            model_identifier = model_identifier,
                        )
                    )
                if _early_non_chat:
                    logger.error(
                        "Refusing non-chat GGUF before teardown: %s (%s)",
                        _early_non_chat,
                        hf_repo,
                    )
                    raise ValueError(_early_non_chat)

            # The probe can take a moment on a slow Hub, so re-read the cancel flag rather
            # than tearing down a healthy server for a load that was called off.
            if _load_cancelled():
                logger.info("Load cancelled before teardown")
                return False

            # Fail fast before killing the live server; an explicit reload retries (freed memory
            # can make the same load fit) and clears the entry.
            _abort_memo_model = repr(
                (
                    replace(intent, hf_token = None, force_reload = False, verified_gguf = None),
                    _vram_frac,
                    _model_memory_settings_or_none(),
                )
            )
            if intent.force_reload:
                LlamaCppBackend._forget_sched_reserve_abort(binary, _abort_memo_model)
            elif LlamaCppBackend._sched_reserve_aborts(binary, _abort_memo_model):
                logger.warning(
                    "Skipping reload of '%s': it already aborted in the llama.cpp graph "
                    "scheduler this session.",
                    model_identifier,
                )
                raise RuntimeError(self._sched_reserve_abort_message())

            # ── Phase 1: kill old process (under lock, fast) ──────────
            # The previous load's advisory is dropped HERE, at the one point this call
            # commits to replacing the resident server, rather than on the first line of
            # load_model. Everything above can stand down with that server still serving
            # (the update-in-progress refusal, the duplicate-load fast path, an
            # unenumerated Vulkan pin, a diffusion or non-chat refusal, a cancel), and
            # clearing on the way past retired the advisory of a child that keeps
            # running -- so the next `already_loaded` answered memory_warning: null for
            # an oversized model that is still paging from disk. Placing the reset at
            # the teardown rather than restoring it on each rejected path is what keeps
            # that true for early returns added later: this is the line that separates
            # "the old child survives" from "the old child is gone", so a new stand-down
            # above it is covered by construction and needs no bookkeeping of its own.
            # Nothing between the top of load_model and this point records a warning.
            _replaying_cpu_fallback = intent.cpu_fallback
            with self._lock:
                self._begin_load_warnings()
                self._kill_process()
                if not _replaying_cpu_fallback:
                    self._cpu_fallback_reason = None
                    self._cleanup_cpu_fallback_runtime()

                self._mmproj_fallback_reason = None
                self._variant_fallback_warning = None
            # Both describe the process just killed, so they are reset HERE rather than
            # where the binary is resolved: everything above can bail with the old server
            # still running (a stand-down, a non-chat refusal, a cancel), and resetting on
            # the way past would tell the next Apply that the resident process launched from
            # the binary installed now and is owed no DFlash retry -- so an Apply after
            # `unsloth studio update` or a retryable sidecar failure would dedupe against it
            # instead of relaunching. Nothing between the two points reads either field.
            #
            # Kept so a stand-down blamed on the binary can later tell a real update
            # from the same file still being installed.
            self._launch_binary_revision = self._binary_revision(binary)
            # Cleared per load, not only where the fetch runs: a local-file load never
            # reaches the DFlash fetch, and a verdict left over from the previous model
            # would make every Apply for this one reload.
            self._dflash_retry_needed = False

            # ── Phase 2: download (NO lock held, so cancel can proceed) ──
            # mtp_draft_path arrives set for local Gemma loads (detected
            # sibling); for -hf loads it's None here and resolved just below.
            # The canonical mode drives which drafter is downloaded, sized and
            # launched, so resolve it once before either branch can use it.
            _spec_canon = _canonicalize_spec_mode(speculative_type) or "auto"
            # #11308: DFlash2 + --split-mode tensor aborts at startup (ggml-org/llama.cpp#27819) and the
            # route falls back to layer split, so Auto prefers a loadable MTP sidecar on tensor instead.
            # _estimate_gguf_required_gb mirrors this.
            _auto_tensor_split = _spec_canon == "auto" and _effective_tensor_parallel(
                extra_args, tensor_parallel
            )

            def _auto_dflash_blocked_by_tensor() -> bool:
                if (
                    _auto_tensor_split
                    and mtp_draft_path
                    and not _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                ):
                    try:
                        from utils.models.gguf_metadata import read_gguf_nextn_predict_layers
                        return bool(
                            _launch_caps(binary).get("mtp_token")
                            and not (read_gguf_nextn_predict_layers(model_path) or 0) > 0
                            and _mtp_drafter_loads_standalone(mtp_draft_path)
                        )
                    except Exception:
                        return False
                return False

            def _dflash_breaks_on_images() -> bool:
                if not dflash_draft_path or disable_vision or extra_args_disable_mmproj(extra_args):
                    return False
                try:
                    projector = self._resolve_launch_mmproj_path(
                        model_path = model_path, mmproj_path = mmproj_path, extra_args = extra_args
                    )
                except Exception:
                    return False
                return _dflash_drafter_breaks_on_images(model_path, dflash_draft_path, projector)

            _unloadable_mtp_draft_path: Optional[str] = None
            # Scope HF_HUB_OFFLINE to the download block only when DNS is
            # dead; cleanup runs even on exception so a transient hiccup
            # can't quarantine future loads.
            if hf_repo:
                # Resolve the requested repo id to its cached canonical casing once,
                # up front, so the main GGUF and its companions (mmproj / MTP drafter)
                # all resolve from the same cache entry. Otherwise a case-variant
                # request resolves the main file from the canonical cache dir while the
                # companions keep the requested casing and miss the cached files.
                _resolved_repo = _resolve_repo_id_casing(hf_repo)
                if _resolved_repo != hf_repo:
                    logger.info(
                        "Using cached repo_id casing '%s' for requested '%s'",
                        _resolved_repo,
                        hf_repo,
                    )
                    hf_repo = _resolved_repo
                with _hf_offline_if_unreachable():
                    model_path = _preflight_model_path
                    if model_path is None:
                        # Revalidate the carried file before skipping download resolution.
                        model_path = self._verified_cached_gguf(intent, hf_repo, hf_variant)
                        if model_path:
                            logger.info("Reusing verified cached GGUF: %s", model_path)
                    if model_path is None:
                        model_path = self._download_gguf(
                            hf_repo = hf_repo,
                            hf_variant = hf_variant,
                            hf_token = hf_token,
                            cancel_event = download_cancel_event,
                        )
                    if self._pending_variant_fallback:
                        hf_variant, self._variant_fallback_warning = self._pending_variant_fallback
                        # Replayed by the respawn and a variant-less Apply, which would re-download.
                        intent = replace(intent, hf_variant = hf_variant)
                    # Auto-download mmproj for vision models unless opted out. Vision
                    # switched off does NOT opt out: remote discovery calls any mmproj
                    # filename vision, and only the file's metadata separates an image
                    # tower from an audio encoder (ultravox, Voxtral, Qwen3-ASR).
                    # Skipping the fetch would take audio away from an audio-only model
                    # to save one small companion file. Suppression happens at the
                    # resolve, where the metadata is readable.
                    if is_vision and not mmproj_path and not extra_args_disable_mmproj(extra_args):
                        mmproj_path = self._download_mmproj(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            cancel_event = download_cancel_event,
                            near_path = model_path,
                        )
                    # Auto-download the separate MTP drafter (e.g. Gemma) when
                    # the requested spec mode can use it. The size gate stays out
                    # of it: a separate drafter speeds up even sub-3B (Gemma E2B),
                    # and the resolver below decides the final emission. Skipped
                    # only when the user disabled MTP or drives --spec-type
                    # manually.
                    #
                    # The MTP/ fallback is qwen4exp only, whose published GGUFs
                    # carry no head at all; every other arch keeps its root-mirror
                    # behaviour. Qwen3.8-27B bakes the head into 20 of 24 quants,
                    # and llama.cpp prefers a -md drafter over an embedded one, so
                    # on UD-Q4_K_XL the two accept identically (143 of 223) for
                    # byte-identical output: 1.37 GB to displace what is loaded.
                    # Read from the file, not self, whose metadata is set below.
                    if (
                        not mtp_draft_path
                        and _spec_canon in ("auto", "mtp", "mtp+ngram")
                        and not _extra_args_set_spec_type(extra_args)
                    ):
                        mtp_draft_path = self._download_mtp(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            cancel_event = download_cancel_event,
                            near_path = model_path,
                            allow_nested = self._gguf_path_wants_nested_mtp(model_path),
                        )
                    # "auto" is included: DSpark is the default whenever the repo
                    # ships a sidecar. Repos without one no-op, exactly like the
                    # MTP fetch above, so this costs one listing and no download.
                    if (
                        not dspark_draft_path
                        and _spec_canon in ("auto", "dspark")
                        and not _extra_args_set_spec_type(extra_args)
                    ):
                        dspark_draft_path = self._download_dspark(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            cancel_event = download_cancel_event,
                            near_path = model_path,
                            binary = binary,
                            caps_probe = _launch_caps,
                        )
                    # DFlash: same shape, and Auto is included for the same
                    # reason, but it is not gated behind an opt-in the way the
                    # ~11 GB DSpark sidecar is. The published sidecar is ~1.5 GiB
                    # and ships in the model's own GGUF repo, so under Auto it
                    # costs about what the MTP drafter costs. Repos without one
                    # no-op after a single listing.
                    # Under Auto, DSpark takes first refusal in the promotion
                    # below, so a repo that ships both kinds would never launch
                    # the DFlash sidecar. Fetching it anyway costs ~1.5 GiB of
                    # bandwidth and cache for a file that cannot be used, so the
                    # Auto fetch stands down once DSpark has resolved. It has to
                    # be the same "resolved" the promotion means, capability and
                    # all: a DSpark path this binary cannot launch loses the
                    # promotion, so standing down on it would leave a
                    # DFlash-capable binary with no drafter at all. An explicit
                    # "dflash" request still fetches.
                    if (
                        not dflash_draft_path
                        and _spec_canon in ("auto", "dflash")
                        and not _auto_dflash_blocked_by_tensor()
                        and not self._dspark_wins_auto(
                            binary = binary,
                            dspark_draft_path = dspark_draft_path,
                            spec_canon = _spec_canon,
                            extra_args = extra_args,
                        )
                        and not _extra_args_set_spec_type(extra_args)
                    ):
                        dflash_draft_path = self._download_dflash(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            cancel_event = download_cancel_event,
                            near_path = model_path,
                            binary = binary,
                        )
            elif gguf_path:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
                model_path = gguf_path
            else:
                raise ValueError("Either gguf_path or hf_repo must be provided")

            # Set identifier early so _read_gguf_metadata can use it (DeepSeek).
            self._model_identifier = model_identifier

            # Auto resolves to DSpark whenever a sidecar is available and this
            # binary can run it: 1.84x decode on 4x B200 and 1.91x on one, against
            # the ngram-mod fallback this architecture would otherwise get. The
            # capability gate lives in _dspark_wins_auto, shared with the DFlash
            # fetch above so the fetch and the promotion cannot disagree about
            # whether Auto is going to end up on DSpark.
            if self._dspark_wins_auto(
                binary = binary,
                dspark_draft_path = dspark_draft_path,
                spec_canon = _spec_canon,
                extra_args = extra_args,
                caps = _launch_caps,
            ):
                _spec_canon = "dspark"
                logger.info("Auto: DSpark sidecar available, using draft-dspark.")

            # DFlash is the other Auto promotion, on the same capability gate.
            # DSpark keeps first refusal (llama.cpp's own downloader ranks it
            # ahead too), so a repo shipping both is unchanged; in practice a
            # repo ships one kind or neither.
            if (
                _spec_canon == "auto"
                and dflash_draft_path
                and not _extra_args_set_spec_type(extra_args)
            ):
                try:
                    if not _launch_caps(binary).get("supports_dflash"):
                        pass
                    elif _auto_dflash_blocked_by_tensor():
                        logger.info(
                            "Auto: DFlash sidecar available but tensor split is on; "
                            "keeping tensor split with the MTP drafter "
                            "(llama.cpp cannot run DFlash2 with --split-mode tensor yet)."
                        )
                    elif _dflash_breaks_on_images():
                        logger.info(
                            "Auto: DFlash sidecar %s has no M-RoPE sections, so llama.cpp "
                            "cannot carry it past an image on this vision model; keeping "
                            "vision and skipping DFlash. Choose DFlash in the Speculative "
                            "Decoding dropdown to force it, or turn Vision off.",
                            dflash_draft_path,
                        )
                    else:
                        _spec_canon = "dflash"
                        logger.info("Auto: DFlash sidecar available, using draft-dflash.")
                except Exception as exc:
                    logger.debug("DFlash capability probe failed during Auto: %s", exc)
            elif (
                _spec_canon == "dflash"
                and not _extra_args_set_spec_type(extra_args)
                and _dflash_breaks_on_images()
            ):
                logger.warning(
                    "DFlash forced with sidecar %s, which has no M-RoPE sections: llama.cpp "
                    "will fail the first image sent after text in a chat. Use Auto or turn "
                    "Vision off to keep images working.",
                    dflash_draft_path,
                )

            # MTP and DSpark are mutually exclusive, and the drafter sizing and
            # lifecycle path below is architecture agnostic, so it carries the
            # one drafter the mode selected. From here mtp_draft_path means
            # "the drafter this load will launch", whichever kind that is;
            # _spec_canon is the only thing that says which.
            if _spec_canon == "dspark":
                mtp_draft_path = dspark_draft_path
            elif _spec_canon == "dflash":
                mtp_draft_path = dflash_draft_path

            # Read GGUF metadata (context_length, chat_template); header-only.
            self._read_gguf_metadata(model_path)

            if (
                self._nextn_predict_layers
                and mtp_draft_path
                and _spec_canon not in ("dspark", "dflash")
            ):
                # A root mtp-*.gguf may mirror the embedded head; -md would replace it.
                logger.info("Main GGUF contains an embedded MTP head; ignoring separate drafter.")
                mtp_draft_path = None

            # Before the fit prices it, or a drafter that never launches pushes layers
            # off the GPU. DSpark/DFlash borrow token_embd from the target by design.
            #
            # Not when the extras name their own drafter: draft flags are last-wins, so
            # that file is the one llama-server opens and the sibling is never touched.
            # Dropping it anyway reads as drafterless, and the fallback emits ngram-mod
            # or --spec-default BEFORE the extras are appended, so the override stops
            # running as MTP. Same helper and child-env rule as the paravirtual drop
            # below: inherited LLAMA_ARG_SPEC_DRAFT_* reaches the child only when the
            # extras own --spec-type.
            if (
                mtp_draft_path
                and _spec_canon not in ("dspark", "dflash")
                and not _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                and not _mtp_drafter_loads_standalone(mtp_draft_path)
            ):
                logger.warning(
                    "Dropping MTP drafter %s: it carries neither token_embd.weight nor "
                    "nextn_shared_target_tensors, so llama-server cannot load it as "
                    "--model-draft; loading without it.",
                    mtp_draft_path,
                )
                _unloadable_mtp_draft_path = mtp_draft_path
                mtp_draft_path = None

            if _load_cancelled():
                logger.info("Load cancelled after download phase")
                return False

            # Whether the user set the batch pair, read before the embedding and projector
            # raises below overwrite n_batch / n_ubatch. Gates the expert-spill raise.
            _user_named_batch = any(_named_batch_sizes(extra_args, None, n_batch, n_ubatch)[2:])

            # MEAN/CLS inputs must fit one micro-batch (LAST splits). Before the projector raise,
            # which would otherwise read as a user-set micro-batch.
            if self._pooling_type in (1, 2):
                n_batch, n_ubatch = _embedding_batch_ubatch(
                    resolve_requested_ctx(extra_args, n_ctx) or self._context_length or 0,
                    n_batch,
                    n_ubatch,
                    extra_args,
                )

            # Decide after downloading the projector and before pricing the load.
            n_batch, n_ubatch = _batch_ubatch_for_mmproj(
                _launch_required_ubatch(
                    None
                    if (disable_vision or not is_vision)
                    else self._resolve_launch_mmproj_path(
                        model_path = model_path,
                        mmproj_path = mmproj_path,
                        extra_args = extra_args,
                    ),
                    # Use the same order-independent metadata read as the estimators.
                    _read_gguf_embedding_length(model_path),
                    extra_args,
                    is_vision = is_vision,
                    vision_off = disable_vision,
                ),
                n_batch,
                n_ubatch,
                extra_args,
            )

            # Backstop for everything the pre-teardown probes fail open on: refuse from the
            # header rather than watching llama-server die as "failed to start".
            non_chat = self._non_chat_gguf_refusal(model_path)
            if non_chat:
                logger.error("Refusing non-chat GGUF: %s (%s)", non_chat, model_path)
                raise ValueError(non_chat)

            # Block-diffusion GGUFs (DiffusionGemma) cannot run on llama-server;
            # serve them with the diffusion runner (same OpenAI-compat interface).
            if self._is_diffusion:
                # Final defense: route and pre-teardown preflights reject before Phase 1.
                # The diffusion runner cannot translate Vulkan ordinals to CUDA IDs, so
                # this fires unless the caller confirmed the pin is physical CUDA IDs.
                if is_vulkan_backend and gpu_ids and gpu_ids_are_vulkan_ordinals is not False:
                    raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)
                # Not a tensor/layer GGUF: clear any preserved-fallback flag from a
                # prior load (this path skips the command builder that clears it).
                self._layer_preserves_tensor_intent = False
                # Same reason: a diffusion server never carries a drafter, only unload
                # clears these, and the dedupe compares the pair, so a stale one
                # reloads a healthy diffusion server on every Apply.
                self._mtp_draft_path = None
                self._mtp_draft_suppressed_path = None
                self._mtp_draft_suppressed_reason = None
                # And the verdict on why the last load had no drafter, for the same
                # reason once more: _build_speculative_flags clears it at the top of
                # every load that reaches it, and this path never does. Both retry
                # rules in the dedupe read it -- the drafter one and the binary one --
                # and a stale answer would relaunch this server on every Apply for a
                # drafter it was never going to carry.
                self._spec_fallback_reason = None
                self._spec_drafter_kind = None
                # Discovery runs before the metadata read that says diffusion, so a
                # transient sidecar failure can set this for a drafterless server.
                self._dflash_retry_needed = False
                # Same reason: this path returns before the launch block, and the
                # diffusion runner ignores the budget, so a previous llama-server's
                # fraction would relaunch a healthy diffusion server on every Apply.
                self._vram_fraction_launched = None
                # Outside the lock, as for llama-server: the Auto layer fit reads free VRAM,
                # which a just-killed runner still holds until the driver reclaims it.
                self._wait_for_vram_settle(since_kill = self._last_kill_monotonic)
                with self._lock:
                    if _load_cancelled():
                        logger.info("Load cancelled before diffusion server start")
                        return False
                    return self._start_diffusion_server(
                        model_path = model_path,
                        gguf_path = gguf_path,
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        model_identifier = model_identifier,
                        n_ctx = n_ctx,
                        extra_args = extra_args,
                        gpu_ids = gpu_ids,
                        gpu_memory_mode = gpu_memory_mode,
                        gpu_layers = gpu_layers,
                        cancelled = _load_cancelled,
                    )

            if not binary:
                # distinguish a transiently locked binary (antivirus / in-flight
                # install) from a missing one so the user retries, not reinstalls
                locked = self._find_llama_server_binary(include_denied = True)
                if locked:
                    raise RuntimeError(
                        f"llama-server at {locked} is temporarily unavailable "
                        "(access-denied; antivirus or an in-flight install). "
                        "Retry the load once it is released."
                    )
                # Reached only after the diffusion early-return above, so this is a
                # genuine llama-server-backed GGUF with no runtime. Raise the typed
                # error so /load returns the actionable 400 (not a generic 500), the
                # same message remote validation already shows.
                raise LlamaServerNotFoundError(LLAMA_SERVER_NOT_FOUND_DETAIL)

            server_caps = _launch_caps(binary)
            self._reserves_micro_batch_outputs = self.reserves_micro_batch_outputs(binary)
            self._resumes_thoughts = self.resumes_thoughts(binary)

            # Outside ``self._lock`` so /unload, /cancel, /status aren't
            # blocked. ``unload_model`` also records the kill, so the
            # frontend /unload+/load Apply path engages the wait here even
            # without an in-process kill.
            self._wait_for_vram_settle(since_kill = self._last_kill_monotonic)

            # ── Phase 3: start llama-server (under lock) ──────────────
            with self._lock:
                # Re-check cancel inside lock
                if _load_cancelled():
                    logger.info("Load cancelled before server start")
                    return False

                self._port = self._find_free_port()

                # Select GPU(s) from model size + estimated KV cache. Seed
                # safe defaults before probing so the except path has valid
                # state to publish.
                ctx_override = parse_ctx_override(extra_args)
                requested_ctx = resolve_requested_ctx(extra_args, n_ctx)
                swa_full = _swa_full_from_args_or_env(extra_args)

                # The emitted batch, not the requested one: the floor raise below is what
                # llama.cpp caps the micro-batch against, so the fit, the slot search and
                # every compute-buffer reserve must be sized from the value that launches.
                # Slot-dependent, since the floor is max(slots, 2): the slot search below
                # reduces the count, and each candidate launches at its own batch.
                def _ubatch_for_slots(slots: int) -> Optional[int]:
                    return _extra_args_n_ubatch(
                        extra_args,
                        n_ctx = (requested_ctx if requested_ctx > 0 else self._context_length),
                        n_batch = _emitted_n_batch(n_batch, slots),
                        n_ubatch = n_ubatch,
                    )

                _effective_ubatch = _ubatch_for_slots(n_parallel)

                # The budget is per slot and the fit can still cut n_parallel, so both
                # counts are taken after it settles, as _effective_ubatch is below.
                def _ctx_checkpoints_for_final_slots() -> int:
                    return effective_ctx_checkpoints_for_caps(
                        server_caps,
                        extra_args,
                        ctx_checkpoints,
                        per_checkpoint_bytes = self._ctx_checkpoint_bytes(
                            cache_type_kv, swa_full = swa_full, flash_attn = planned_flash_attn
                        ),
                        n_parallel = n_parallel,
                        total_host_bytes = ((self._host_memory_capacity_mib() or 0) * 1024 * 1024)
                        or None,
                    )

                _requested_ctx_checkpoints = (
                    resolve_ctx_checkpoints(extra_args, ctx_checkpoints)
                    if server_caps.get("ctx_checkpoints_flag")
                    else 0
                )
                # Snapshots are host RAM (copied device-to-host): only a host-RAM pool pays.
                _fit_ctx_checkpoints = (
                    0 if self._rollback_state_bytes(1) > 0 else _requested_ctx_checkpoints
                )
                # --embedding forces n_batch = n_ubatch, which aborts a load below the slots.
                if (
                    self.is_embedding_gguf
                    and _effective_ubatch is not None
                    and _effective_ubatch < n_parallel
                ):
                    # max(): a degenerate "-b 0" extra resolves to 0, and --parallel 0 is
                    # rejected at arg parse.
                    _embedding_slots = max(1, _effective_ubatch)
                    logger.warning(
                        "Reducing serving slots from %d to %d: llama-server caps the batch "
                        "at the %d-token micro-batch under --embedding and aborts when that "
                        "is below the slot count.",
                        n_parallel,
                        _embedding_slots,
                        _effective_ubatch,
                    )
                    n_parallel = _embedding_slots  # allow-slot-clamp: llama-server would abort
                    _effective_ubatch = _ubatch_for_slots(n_parallel)
                planned_kv_unified = _kv_unified_from_args(
                    extra_args,
                    default = n_parallel > 1 and server_caps.get("supports_kv_unified", False),
                )
                _pipeline_parallel_off = _pipeline_parallel_disabled_by_args(
                    extra_args, n_layers = self._n_layers
                )
                cache_override = parse_cache_override(extra_args)
                # Budget the heavier of asymmetric --cache-type-k/-v extras (they
                # win per axis at launch, appended last); resolve_cache_type_kv only
                # returns the last-wins type, which under-reserves the heavier axis.
                # The user's extras still set the real (possibly asymmetric) child
                # cache, so this only affects the reserve, not the emitted command.
                _extras_cache = _extra_args_main_cache_type_for_budget(extra_args)
                cache_type_kv = _extras_cache if _extras_cache is not None else cache_type_kv
                _cache_type_from_env = False
                # An unknown type is never emitted (see _VALID_CACHE_TYPES below), so the
                # child falls through to whatever LLAMA_ARG_CACHE_TYPE_K/_V it inherits.
                # Budgeting the unemittable string would price it at _kv_bytes_per_elem's
                # 2.0 unknown default while the child allocates the inherited type: a 2x
                # under-reservation against an inherited f32, in the mode with no --fit
                # valve. Treat it as unset so the env adoption below sees it.
                if cache_type_kv is not None and (
                    cache_type_kv.strip().lower() not in _VALID_CACHE_TYPES
                ):
                    cache_type_kv = None
                if cache_type_kv is None:
                    # Param/extras set nothing, so the child inherits
                    # LLAMA_ARG_CACHE_TYPE_K/_V. Adopt an env type that budgets
                    # differently from f16 (f32, quantized) for the reserve only;
                    # the launch does NOT re-emit it (that would rewrite an
                    # asymmetric K=f32,V=f16 env into symmetric flags), so
                    # _cache_type_from_env keeps it out of the emitted flags.
                    cache_type_kv = _env_main_cache_type_for_budget()
                    _cache_type_from_env = cache_type_kv is not None
                # The heavier axis above budgets KV BYTES; the dequant scratch keys off
                # the LIGHTER axis instead. Since ggml-org/llama.cpp#23792 an asymmetric
                # -ctk q4_0 -ctv f16 reaches the child, where the heavier scalar (f16)
                # would price the scratch as if nothing were quantized. Same precedence.
                _scratch_cache_type_kv = _planned_scratch_cache_type(
                    None if _cache_type_from_env else cache_type_kv,
                    extra_args,
                )
                # The (K, V) pair this launch will run, for the tensor-abort latch:
                # since #23792 the type reaches the child, so an abort can be specific
                # to it and the latch must not generalise across types.
                _planned_cache_pair = _planned_main_cache_types(
                    None if _cache_type_from_env else cache_type_kv,
                    extra_args,
                )
                # Auto leaves this negative, which is llama.cpp's default and no override.
                _manual_gpu_layers = gpu_layers if gpu_memory_mode == "manual" else None
                # After the cache pair, because a quantized V forces flash attention on. NOT
                # pinned False for the no-flash recovery: --fit re-places that respawn.
                planned_flash_attn = _reserved_flash_attn_state(
                    _planned_flash_attn_state(
                        extra_args,
                        planned_cache_types = _planned_cache_pair,
                        supports_flash_attn = bool(server_caps.get("supports_flash_attn", True)),
                        tensor_parallel = tensor_parallel,
                        architecture = self._architecture,
                    ),
                    extra_args,
                    tensor_parallel = tensor_parallel,
                    gpu_layers = _manual_gpu_layers,
                )

                def _replan_env(_current_tp: bool) -> "Optional[Mapping[str, str]]":
                    """The environment the CHILD will see: the launch clears a non-layer
                    inherited LLAMA_ARG_SPLIT_MODE, which otherwise turns tensor mode back
                    on inside the helper."""
                    if _current_tp:
                        return None
                    inherited = (os.environ.get("LLAMA_ARG_SPLIT_MODE") or "").strip().lower()
                    if not inherited or inherited == "layer":
                        return None
                    return {
                        key: value
                        for key, value in os.environ.items()
                        if key not in ("LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_TENSOR_SPLIT")
                    }

                def _replanned_flash_attn(_current_tp: bool) -> bool:
                    """Re-resolve the planned attention after a tensor-mode downgrade: AUTO
                    plans it ON under tensor, so a stale plan budgets a V layer never gets."""
                    _env = _replan_env(_current_tp)
                    return _reserved_flash_attn_state(
                        _planned_flash_attn_state(
                            extra_args,
                            planned_cache_types = _planned_cache_pair,
                            supports_flash_attn = bool(server_caps.get("supports_flash_attn", True)),
                            tensor_parallel = _current_tp,
                            architecture = self._architecture,
                            env = _env,
                        ),
                        extra_args,
                        tensor_parallel = _current_tp,
                        gpu_layers = _manual_gpu_layers,
                        env = _env,
                    )

                # A user --split-mode in extras last-wins-overrides the toggle, and
                # an inherited tensor LLAMA_ARG_SPLIT_MODE flips it on (the child
                # would run tensor unbudgeted otherwise). The duplicate-load matchers
                # use the same helper so a healthy env-driven tensor server matches.
                split_mode_override = parse_split_mode_override(extra_args)
                tensor_parallel = _effective_tensor_parallel(extra_args, tensor_parallel)
                # gpu_layers=0 leaves nothing to split, yet --split-mode tensor or
                # a per-GPU ratio still launches tensor mode -- and under the
                # CPU-only mask below (no visible devices) that aborts the server
                # instead of loading on CPU. Drop both here (nothing to split).
                if gpu_memory_mode == "manual" and gpu_layers == 0:
                    if tensor_parallel or tensor_split:
                        logger.info(
                            "Manual gpu_layers=0: dropping tensor split/parallel "
                            "flags (nothing to split on the GPU)"
                        )
                    tensor_parallel = False
                    tensor_split = None
                    # The plan was made under tensor, the one state forcing AUTO on. Same
                    # for every later downgrade.
                    planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                # Record the requested strategy for /status and the load
                # response. 'manual' has no fallback, so the request value is the
                # value actually applied.
                self._gpu_memory_mode = gpu_memory_mode
                # The layer/MoE/split knobs apply only with an explicit offload
                # (manual + gpu_layers >= 0); else record defaults so /status and
                # /load don't report knobs the server never applied. Reset before the
                # branch so an earlier launch's recorded drop cannot excuse this one.
                self._arch_gate_dropped_tensor_split = None
                self._arch_gate_dropped_tensor_parallel = False
                # Same reason, same place: the auto ratio is recorded only on the
                # auto tensor branch, so without this a manual load would carry a
                # previous auto load's ratio forward into the next comparison.
                self._auto_tensor_split = None
                self._auto_tensor_split_emitted = None
                if gpu_memory_mode == "manual" and gpu_layers >= 0:
                    self._gpu_layers = gpu_layers
                    self._n_cpu_moe = n_cpu_moe
                    self._tensor_split = tensor_split
                else:
                    self._gpu_layers = -1
                    self._n_cpu_moe = 0
                    self._tensor_split = None
                    self._auto_tensor_split = None
                    self._auto_tensor_split_emitted = None
                self._requested_gpu_ids = [int(i) for i in gpu_ids] if gpu_ids else None
                self._gpu_ids = list(self._requested_gpu_ids) if self._requested_gpu_ids else None
                # Manual offload skips the TP planner but still emits --split-mode
                # tensor at launch; drop it when fewer than 2 GPUs are in use --
                # tensor split is a no-op there and aborts on some architectures.
                if (
                    tensor_parallel
                    and gpu_memory_mode == "manual"
                    and gpu_layers >= 0
                    and self._effective_gpu_count(sorted(gpu_ids) if gpu_ids else None) < 2
                ):
                    logger.info(
                        "Tensor parallelism requested in manual mode but fewer "
                        "than 2 GPUs are in use; ignoring (needs >= 2)."
                    )
                    tensor_parallel = False
                    planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                if tensor_parallel and gpu_memory_mode == "manual" and gpu_layers < 0:
                    logger.info(
                        "Manual mode with Auto layers hands memory management to "
                        "llama.cpp --fit, which is incompatible with tensor "
                        "parallelism; ignoring the tensor split."
                    )
                    tensor_parallel = False
                    planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                if ctx_override is not None and ctx_override > 0:
                    logger.info(f"User --ctx-size {ctx_override} honored; skipping auto-reduce")
                if cache_override is not None:
                    _ck, _cv = parse_cache_override_per_axis(extra_args)
                    logger.info(
                        f"User --cache-type-k/-v (k={_ck}, v={_cv}) honored; "
                        "KV estimate budgets the heavier axis"
                    )
                if split_mode_override is not None:
                    logger.info(
                        f"User --split-mode {split_mode_override} honored; "
                        "reconciled into tensor_parallel state"
                    )
                effective_ctx = requested_ctx if requested_ctx > 0 else (self._context_length or 0)
                max_available_ctx = self._context_length or effective_ctx
                gpus: list[tuple[int, int]] = []
                # Keep fit-budget and launch-flag mmproj resolution in sync.
                launch_mmproj_path = None
                # Set only where the SWITCH is what dropped a usable image projector.
                # A None launch path does not mean that on its own: the resolve also
                # returns None for a missing file or a family mismatch, and reporting
                # the switch there tells the user to turn Vision back on when the same
                # projector will just be rejected again.
                _dv_dropped_image_projector = False
                if not extra_args_disable_mmproj(extra_args):
                    launch_mmproj_path = self._resolve_launch_mmproj_path(
                        model_path = model_path,
                        mmproj_path = mmproj_path,
                        extra_args = extra_args,
                    )
                    # The switch turns VISION off, and a projector is not always a
                    # vision tower: ultravox, Voxtral and Qwen3-ASR declare an audio
                    # encoder and no vision at all, so suppressing one takes the
                    # model's audio input away and gives no image VRAM back. Nothing
                    # there for the switch to turn off, so keep it. Resolution is a
                    # disk and family check with no download, so asking first is free.
                    if disable_vision and launch_mmproj_path:
                        _dv_has_audio, _dv_accepts_image = False, True
                        try:
                            from utils.models.gguf_metadata import mmproj_capabilities
                            _dv_has_audio, _dv_accepts_image = mmproj_capabilities(
                                launch_mmproj_path
                            )
                        except Exception as e:
                            # Unknown reads image-capable upstream, so honor the switch
                            # rather than quietly keep a projector the user turned off.
                            logger.debug(f"mmproj capability read failed: {e}")
                        if _dv_accepts_image:
                            if _dv_has_audio:
                                # One projector serves both modalities (Qwen2.5-Omni)
                                # and llama.cpp cannot load half of it, so audio input
                                # goes with vision. Say so: the switch only named images.
                                logger.warning(
                                    "Vision is off for this load, so this model's audio "
                                    "input goes too: its projector serves both and "
                                    "llama.cpp cannot load one modality without the "
                                    "other. Turn Vision back on to send audio."
                                )
                            launch_mmproj_path = None
                            _dv_dropped_image_projector = True
                        else:
                            logger.info(
                                "Vision is off for this load, but this model's projector "
                                "is audio-only (no vision tower), so it stays loaded: "
                                "dropping it would remove audio input and free no image "
                                "VRAM."
                            )
                # A virtualised Metal device corrupts the vision encoder too, and
                # --no-mmproj-offload is the only way to keep it off there (clip.cpp
                # never reads --gpu-layers). Without that flag the choice is a projector
                # on the corrupt device or no projector, so drop it: silently encoding
                # every image from garbage is the failure this fallback exists to stop.
                # Only when the probe answered: --no-mmproj-offload is from b5178 and
                # the base argv already needs b6325, so a build that can start here
                # always has it, and a false capability means the --help probe failed
                # (see _paravirtual_probe_answered). Dropping vision on that would be a
                # self-inflicted outage; the pin below covers the unanswered case.
                _pv_mmproj_unpinnable = bool(
                    launch_mmproj_path
                    and _paravirtual_cpu_forced
                    and _paravirtual_probe_answered(server_caps)
                    and not server_caps.get("supports_no_mmproj_offload")
                )
                if _pv_mmproj_unpinnable:
                    logger.warning(
                        "Disabling image/audio input for this session: this Mac's "
                        "Metal device is virtualised and this llama-server build has "
                        "no --no-mmproj-offload, so the vision projector would keep "
                        "running on the device whose output is corrupt. Run 'unsloth "
                        "studio update' to install a build that supports it."
                    )
                    launch_mmproj_path = None
                # Need both a resolved mmproj AND the config vision flag; a stray
                # mmproj passing the family-name heuristic must not flip a non-VLM
                # GGUF into vision mode.
                effective_is_vision = bool(launch_mmproj_path) and bool(is_vision)
                if is_vision and not effective_is_vision and not _pv_mmproj_unpinnable:
                    logger.warning(
                        "Vision-capable GGUF loaded without a usable mmproj; "
                        "image input will be disabled for this session"
                    )
                # Seed before the try: the except (GPU-selection failure ->
                # --fit on) falls through to the launch which reads this, and the
                # probe that assigns it may throw first. Captured before manual
                # empty `gpus` so the speculative defaults stay GPU-aware and the
                # CPU-fallback check still knows GPUs were present.
                _detected_gpus: list[tuple[int, int]] = []
                _shared_gpu_ids: set[int] = set()
                _placement_ctx_checkpoints = 0
                # Set when the arch gate emptied a non-empty GPU pool, so the env
                # block below masks the child onto the CPU. Bound before the try for
                # the same reason as _detected_gpus: the except path (--fit on) falls
                # through to the launch, which reads it.
                _arch_gate_forced_cpu = False
                # The GPUs THIS launch pinned the child to, or None when it pinned
                # nothing. Set where the pin is emitted, and read by the pageable
                # override so it prices against VRAM the child can actually spend.
                # Not derived from env: an inherited mask is already applied by the
                # probe, so re-reading it there would subtract the same pin twice.
                _launch_pinned_ids: Optional[list] = None
                # The pageable rewrite a CPU replay applied, kept across attempts: the
                # env is shared, so only the first replay to need it produces a note.
                _replay_pageable_override: Optional[str] = None
                # Set by the restored cpu_fallback path, consumed once the launch is
                # healthy: str carries that replay's override note, True means none.
                _restored_cpu_reprice: Union[str, bool, None] = None
                # Whether Unsloth's placement planner actually RAN and came back
                # unable to fit the model. `use_fit` alone cannot answer that: it
                # starts True at its declaration below and is also what the except
                # path restores, so an unfitted `--fit on` is indistinguishable
                # from the default unless the verdict is recorded on its own.
                # Bound before the try for the same reason as _detected_gpus.
                _placement_verdict_partial = False
                _ctx_capped_for_vram = False
                # A path that never prices the launch must not commit the previous one's plan.
                self._pending_plan_mib = {}
                # Set by the expert-spill raise after the fit.
                self._moe_spill_batch_restore: Optional[tuple[Optional[int], Optional[int]]] = None
                _moe_cache_candidate = False
                _moe_cache_ram_ok = False
                _shared_gpus = frozenset()
                # Sized inputs for the tensor-spill planner, None when the fit never
                # priced them. Bound before the try like _placement_verdict_partial:
                # the except arm restores use_fit=True without rebinding
                # kv_cache_bytes, so reading it from the spill arm would be an
                # UnboundLocalError on the path that reaches it most often.
                _spill_inputs: Optional[dict] = None
                # Whether the tensor-parallel planner ran to completion. Bound before
                # the try for the same reason as the two above, and read AFTER it: the
                # auto ``--tensor-split`` fallback needs the planner's own working set
                # (``tp_gpus``, ``gpu_indices``, the MTP terms), and every one of those
                # is bound inside the try. The except arm restores ``tp_tensor_split =
                # None``, which is precisely the state the fallback fires on, so
                # without this it would read locals a throw left unbound -- or, worse,
                # a ``gpu_indices`` the arm rebuilt from a WIDER device set than the
                # planner's reserve filter admitted, and index a card that is not in it.
                _tp_planned = False
                # Held rather than recorded where it is decided: _record_load_warning is
                # first-notice-wins, and both arms that raise an unmeasured ceiling run
                # long before the host-RAM and offload advisories, so recording there took
                # the slot from a memory warning on the one path that also raises the
                # launched context. Flushed past those two, appending if one spoke.
                _unmeasured_ctx_notice: Optional[str] = None
                _cuda_ctx_notice: Optional[str] = None
                _ctx_cap_fits = False
                _vram_fit_ctx: Optional[int] = None
                _flat_mtp_engages = False
                total_by_idx: dict[int, int] = {}
                _gpu_mem: list[tuple[int, int, int]] = []
                model_size = None  # set in the fit try; used by the APU RAM guard
                _mtp_will_engage = False
                _separate_draft_launches = False  # a sidecar displaces an embedded head
                # "none" once the fit proves the load needs no demand paging, else None
                # for llama.cpp's own default. Bound before the try like the verdict
                # flags above: the except path falls through to the launch, which reads it.
                _fit_load_mode: Optional[str] = None
                # Layer-fallback min GPUs; raised below on a tensor downgrade. Bound
                # before the try so the --fit-on except path still has it (no UnboundLocal).
                _layer_min_gpus = 1
                # An explicit Vulkan ordinal absent from the ggml probe cannot be
                # honored; flag it in the fit and reject after the try (raising inside
                # would be swallowed into the --fit-on fallback). Bound before the try.
                _vulkan_explicit_unmatched = False
                _vulkan_requested_ids: list[int] = []
                _vulkan_available_ordinals: list[int] = []
                # Auto dropped the drafter because only the target fit. Bound before the
                # try: the launch below reads it either way.
                _spec_dropped_no_vram = False
                _mtp_slots_asked: Optional[int] = None
                _mtp_slots_target_ctx: Optional[int] = None
                # The projector runs on the CPU and so allocates no VRAM:
                # --no-mmproj-offload clears mmproj_use_gpu and clip.cpp gates its whole
                # GPU backend on that, leaving no partial offload. Only the paravirtual
                # reason is known this early; the fit below may add the automatic one.
                # Bound before the try, like _spec_dropped_no_vram.
                _mmproj_cpu_pinned = bool(
                    launch_mmproj_path
                    and _paravirtual_cpu_forced
                    and _paravirtual_mmproj_pinnable(server_caps)
                )
                # What the pin took out of model_size, for the one consumer that must
                # have it back: _apu_ram_shortfall_message prices SYSTEM RAM, and a
                # CPU-pinned projector is still in system RAM.
                _mmproj_pinned_bytes = 0
                # Metal unified-memory refusal for a hand-set context, raised AFTER this
                # block: the `except Exception` below would swallow the raise into the
                # placement fallback and restore the original request, which is the very
                # over-commit being refused. Carrying the message out survives that arm,
                # and the ceiling it names was measured before the throw. (Worded around
                # the handler's own log line on purpose: test_tp_vision_regression
                # string-searches this function's source for it to check ordering.)
                _metal_ctx_refusal: Optional[str] = None
                _metal_ctx_warning: Optional[str] = None
                # A discounted Metal ceiling must launch the measured full-offload placement.
                _metal_full_offload_pinned = False
                # --fit-target floor from the Metal wired headroom; 0 off Apple Silicon.
                _metal_fit_target_floor_mib = 0.0
                # The fit and launch must use the same Model Memory snapshot.
                from utils.model_memory_settings import capture_model_memory_settings

                _mem_settings = capture_model_memory_settings(
                    lambda pair: setattr(self, "_memory_pending_launch", pair)
                )
                try:
                    gguf_size = self._get_gguf_size_bytes(model_path)
                    # Include GPU-loaded mmproj in the fit budget (#5825). GPU-loaded is
                    # the operative word: --no-mmproj-offload already put the projector in
                    # host RAM, so its bytes are not on the card. Charging them anyway
                    # shrinks the context (measured 9984 -> 4096) or spills layers to make
                    # room for VRAM nothing occupies. The RESOLVED value decides, not just
                    # ownership: arg.cpp reads the env before argv, so the environment
                    # places the projector just as surely as the flag.
                    _user_mmproj_offload = _resolved_mmproj_offload(extra_args)
                    mmproj_size = (
                        self._mmproj_vram_bytes(launch_mmproj_path)
                        if (effective_is_vision and _user_mmproj_offload is not False)
                        else 0
                    )
                    if _user_mmproj_offload is False and launch_mmproj_path:
                        # Still in system RAM, which the APU shortfall guard prices.
                        _mmproj_pinned_bytes = self._mmproj_vram_bytes(launch_mmproj_path)
                    model_size = gguf_size + mmproj_size
                    # 2-tuple gpus for existing logic + a total map for the absolute
                    # per-GPU headroom (correct when the GPU is already partly used).
                    # Pass binary so a Vulkan build probes ggml's Vulkan ordinals.
                    # AUTOMATIC placement only. Gating an explicit pin would leave
                    # `gpus` empty, skipping the fit and layer plan and forcing
                    # `--fit on`, yet `gpu_indices` is restored from `gpu_ids` below
                    # so the pin happens anyway. A pinned uncovered GPU is the user's
                    # call and already reports "device kernel image is invalid".
                    _gpu_mem = self._get_gpu_memory(binary, for_llama_server = not gpu_ids)
                    _held_by_others = self._other_planned_vram_mib()
                    _shared_gpus = frozenset(_held_by_others)
                    _gpu_mem = _net_of_held_vram(_gpu_mem, _held_by_others)
                    # Every present device gated out (#7624). Left alone the launch
                    # takes the `--fit on` arm with `gpu_indices` still None, so no
                    # mask is written, the child enumerates every unsupported card and
                    # dies, and the reactive retry cannot help (its guard needs a
                    # truthy `gpu_indices`). CPU is the only placement that runs here.
                    # ROCm auto only; the guard spares CUDA and CPU-only hosts, where
                    # an empty probe just means "no GPU".
                    if (
                        not gpu_ids
                        and not _gpu_mem
                        and not is_vulkan_backend
                        and self._host_torch_is_rocm()
                    ):
                        _ungated = self._get_gpu_memory(binary)
                        # Re-derive the verdict rather than infer it from the empty
                        # result: _get_gpu_memory turns any probe error into [], so
                        # "gated empty, ungated not" also describes a one-shot failure
                        # of the FIRST probe on a host the gate never filters, and
                        # masking that onto CPU is a silent, permanent cliff. Unknown
                        # coverage or an unreadable arch keeps every device.
                        _coverage = self._installed_llama_gfx_archs(binary)
                        _arch_by_id = self._rocm_arch_by_physical_id() if _ungated else {}
                        _all_uncovered = (
                            bool(_ungated)
                            and _coverage is not None
                            and all(
                                (_arch_by_id.get(idx) or "") not in ("", *_coverage)
                                for idx, _free, _total in _ungated
                            )
                        )
                        if _all_uncovered:
                            _arch_gate_forced_cpu = True
                            _present = ", ".join(
                                f"{idx} ({_arch_by_id.get(idx) or 'unknown arch'})"
                                for idx, _free, _total in _ungated
                            )
                            logger.warning(
                                "The installed llama.cpp build has no kernels for "
                                "any GPU on this host (%s), so inference falls back "
                                "to CPU. Reinstall llama.cpp for these devices to "
                                "get GPU inference back.",
                                _present,
                            )
                    if is_vulkan_backend and not gpu_ids and gpu_memory_mode != "manual":
                        _gpu_mem = self._vulkan_auto_gpu_memory(_gpu_mem)
                    gpus = [(idx, free) for idx, free, _t in _gpu_mem]
                    # Restrict the fit (and thus the layer plan + pin env) to the
                    # selected GPUs; fail-open if none match so a stale UI choice
                    # can't strand the load on CPU (issue #7164).
                    if gpu_ids:
                        # A Vulkan build indexes by ggml ordinal. An explicit ordinal
                        # absent from the probe can't be pinned, so reject after the try
                        # rather than fail-open onto a device the user didn't pick.
                        _wanted_ids = {int(x) for x in gpu_ids}
                        # Reject if ANY requested ordinal is absent, not only when none
                        # match: [0, 99] against {0, 1} silently drops 99. Comparing the
                        # full requested set (before filter narrows) still lets the fitter
                        # pick a valid subset later -- that is narrowing, not absence.
                        _probed_ordinals = {g[0] for g in gpus}
                        if is_vulkan_backend and not _wanted_ids.issubset(_probed_ordinals):
                            _vulkan_explicit_unmatched = True
                            _vulkan_requested_ids = sorted(_wanted_ids)
                            _vulkan_available_ordinals = sorted(_probed_ordinals)
                        # Restrict the probed pool to the selection; fail-open (keep the
                        # full pool) if none match so a stale UI choice can't strand the
                        # load on CPU (issue #7164).
                        _sel_gpus = [g for g in gpus if g[0] in _wanted_ids]
                        gpus = _sel_gpus if _sel_gpus else gpus
                    total_by_idx = {idx: total for idx, _f, total in _gpu_mem}
                    # GPU picker: restrict every mode to the chosen devices, so
                    # auto selection only considers them and manual mask to
                    # them (the env block below pins CUDA/HIP_VISIBLE_DEVICES).
                    if gpu_ids:
                        _picked = set(gpu_ids)
                        gpus = [g for g in gpus if g[0] in _picked]

                    # GPUs the model will run on -- captured before manual
                    # empty `gpus` to bypass the planner. bool() drives the
                    # GPU-aware speculative defaults; the list feeds the
                    # CPU-fallback check.
                    _detected_gpus = list(gpus)
                    # Vulkan reports total 0 only for integrated GPUs. Their
                    # free "VRAM" is the same host pool the RAM guard prices.
                    _shared_gpu_ids = (
                        {idx for idx, _free in _detected_gpus if total_by_idx.get(idx, 1) <= 0}
                        if is_vulkan_backend
                        else set()
                    )
                    # Checkpoints are host RAM: a discrete card never holds them, but a
                    # unified-memory GPU's "VRAM" is that same pool, so it still pays. ANY
                    # shared candidate keeps the charge: Auto may later pick that one alone.
                    _gpu_ids_now = [idx for idx, _free in gpus]
                    _placement_ctx_checkpoints = (
                        _fit_ctx_checkpoints
                        if _fit_ctx_checkpoints
                        and _gpu_ids_now
                        and (
                            bool(_shared_gpu_ids)
                            or (
                                not is_vulkan_backend
                                and (
                                    self._amd_apu_wants_unified_memory(_gpu_ids_now)
                                    or (
                                        self._integrated_cuda_probe_is_free()
                                        and self._integrated_cuda_unified_memory(_gpu_ids_now)
                                    )
                                )
                            )
                        )
                        else 0
                    )
                    # The --fit fallback is llama.cpp's own fitter, which knows nothing
                    # about this budget: it keeps its own margin and packs the rest on,
                    # so the slider never reached the path that runs when the fit is
                    # tightest. Hand it the difference, both directions, measured from
                    # the default so an untouched slider leaves the command alone.
                    # One broadcast value: --fit-target does take a per-device list, but
                    # in llama.cpp's enumeration order, which the visible-device pin and
                    # the ROCr/Vulkan ordinal quirks make not ours to assume, and a
                    # misaligned list hands a card the wrong margin. So the direction
                    # picks the card: lowering scales by the largest, so no device ends
                    # up under the margin it asked for, raising by the smallest, so none
                    # is taken below its own. Mixed cards gain a little less, and hold
                    # back a little more, than a per-device list would.
                    if _vram_frac != _CTX_FIT_VRAM_FRACTION and gpus:
                        # Free stands in for an unreported total, the same scale
                        # _vram_usable_mib falls back to: MIG/vGPU and the two-column
                        # probe report no total, and deriving the whole adjustment
                        # from a zero would silently drop the budget on exactly the
                        # devices whose headroom is hardest to see.
                        _scales = [total_by_idx.get(_idx) or _free for _idx, _free in gpus]
                        _scale = (
                            max(_scales) if _vram_frac < _CTX_FIT_VRAM_FRACTION else min(_scales)
                        )
                        _fit_target_delta_mib = (_CTX_FIT_VRAM_FRACTION - _vram_frac) * _scale

                    _sysmem_fallback = self._sysmem_fallback_risk(binary)

                    def _gpu_usable(g, frac = _vram_frac):
                        # Callers pass the ACTIVE fraction so the ranking matches the
                        # budget the fit then tests, else mixed totals mis-order.
                        idx, free = g
                        return _vram_usable_mib(
                            free,
                            total_by_idx.get(idx, 0),
                            frac,
                            sysmem_fallback = _sysmem_fallback,
                        )

                    def _pool_budget_mib(subset, frac):
                        # Sum each GPU's own usable budget. Pooling free and total
                        # separately would let an unknown-total GPU (MIG/vGPU/N/A)
                        # add full free with no cushion among known-total GPUs.
                        return sum(max(0.0, _gpu_usable(g, frac)) for g in subset)

                    # Resolve effective context: 0 means let llama-server use
                    # the model's native length. Only expand to a known native
                    # length if metadata exists; else keep 0 as a sentinel.
                    if requested_ctx > 0:
                        effective_ctx = requested_ctx
                    elif self._context_length is not None:
                        effective_ctx = self._context_length
                    else:
                        effective_ctx = 0
                    original_ctx = effective_ctx
                    # Default UI ceiling to the native context length;
                    # GPU/VRAM-fit logic below may shrink it on limited HW.
                    max_available_ctx = self._context_length or effective_ctx

                    # Manual + Auto layers (the Manual default): hand memory
                    # management to llama.cpp's --fit. Emptying the probed GPU set
                    # no-ops the selection/TP planning below, leaving gpu_indices
                    # None (an explicit gpu_ids pick still pins below) and use_fit
                    # True. An explicit context is honored (--fit optimizes around
                    # it); 0 lets --fit size it.
                    if gpu_memory_mode == "manual" and gpu_layers < 0:
                        gpus = []
                        effective_ctx = requested_ctx if requested_ctx > 0 else 0
                        original_ctx = effective_ctx
                        # --fit aborts under --split-mode tensor; a raw extras
                        # --split-mode/--tensor-split (appended last) would
                        # otherwise reach llama-server. Strip it like the TP
                        # downgrade does.
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                    elif gpu_memory_mode == "manual":
                        # Manual offload (--gpu-layers + --fit off): no automatic
                        # device masking (a gpu_ids pick still pins below) or
                        # context cap -- the user owns both. tensor_parallel is
                        # honored but skips the memory-based planner (gpus = []);
                        # the toggle just emits --split-mode tensor (split by free
                        # VRAM, or by the Split ratio if set).
                        gpus = []
                        effective_ctx = (
                            requested_ctx if requested_ctx > 0 else (self._context_length or 0)
                        )
                        original_ctx = effective_ctx
                        # Strip the user --split-mode when the toggle owns the split
                        # (TP engaged -> Unsloth emits --split-mode tensor) or when the
                        # user asked for tensor (which aborts on a single GPU even if
                        # the manual <2-GPU guard downgraded TP). Otherwise keep their
                        # non-tensor mode (row/none/layer) -- the toggle can't express
                        # those.
                        if tensor_parallel or split_mode_override == "tensor":
                            extra_args = strip_split_mode_only(extra_args)
                            planned_flash_attn = _replanned_flash_attn(tensor_parallel)

                    # Will MTP engage? If so, auto-fit reserves draft-model VRAM.
                    # Mirrors _build_speculative_flags: forced mtp/mtp+ngram always
                    # engage; auto only on an MTP model >= 3B; ngram/off never. A
                    # separate drafter (Gemma) counts as an MTP model.
                    # _spec_canon, not the raw request: Auto has already resolved to
                    # dspark or dflash above when a sidecar is available, and the
                    # reserve below differs by kind (no duplicated target-KV copy,
                    # and DSpark uses depth 3).
                    _mtp_effective = _spec_canon
                    _mtp_size_for_fit = _extract_model_size_b(model_identifier)
                    # Sub-3B drops MTP only for an embedded head; a separate
                    # drafter (Gemma) engages and needs its VRAM reserved.
                    _mtp_sub_3b_for_fit = (
                        _mtp_size_for_fit is not None
                        and _mtp_size_for_fit < _MTP_MIN_SIZE_B
                        and not bool(mtp_draft_path)
                    )
                    # The inherited spec env reaches the child only when the extras own
                    # --spec-type; the launch scrubs it otherwise. Reserve against the
                    # env in that case only, where flags and env accumulate and launch.
                    _spec_env: Mapping[str, str] = _child_spec_env(extra_args)
                    # Extras can run MTP even when Unsloth suppresses its own emission.
                    _user_mtp_via_extras = _extra_args_requests_mtp(extra_args, env = _spec_env)
                    # A non-MTP model-based draft mode (draft-simple/draft-eagle3) in
                    # extras also loads a separate draft model that needs reserving;
                    # engage only when extras actually name a drafter for it.
                    _user_draft_via_extras = _extra_args_requests_separate_draft(
                        extra_args, env = _spec_env
                    ) and bool(_extra_args_mtp_draft_path(extra_args))
                    # Mirror _build_speculative_flags: reserve only for MTP the launch
                    # resolver will actually emit (needs a head/drafter and a binary
                    # that supports --spec-type mtp).
                    _mtp_model_for_fit = bool(
                        self._nextn_predict_layers
                        or _is_mtp_model_name(model_identifier, model_path)
                        or bool(mtp_draft_path)
                    ) and not (
                        # Drafterless Gemma falls back to ngram-mod; reserve no
                        # drafter VRAM for it (mirrors the launch resolver).
                        _is_gemma_mtp_name(model_identifier, model_path)
                        and not mtp_draft_path
                        and not self._nextn_predict_layers
                    )
                    _mtp_binary_ok = True
                    _mtp_probe_raised = False
                    if not _user_mtp_via_extras:
                        try:
                            _fit_caps = _launch_caps(binary) or {}
                            _sidecar_cap = {
                                "dspark": "supports_dspark",
                                "dflash": "supports_dflash",
                            }.get(_mtp_effective)
                            _mtp_binary_ok = bool(
                                _fit_caps.get(_sidecar_cap)
                                if _sidecar_cap
                                else _fit_caps.get("mtp_token")
                            )
                        except Exception:
                            _mtp_binary_ok = False
                            _mtp_probe_raised = True
                    _auto_studio_mtp = (
                        not _extra_args_set_spec_type(extra_args)
                        and _mtp_model_for_fit
                        and (
                            _mtp_effective in ("mtp", "mtp+ngram", "dspark", "dflash")
                            or (_mtp_effective == "auto" and not _mtp_sub_3b_for_fit)
                        )
                        and (
                            _mtp_binary_ok
                            # Reserve on a raised (uncached) probe too: it re-probes in
                            # _build_speculative_flags and may still engage MTP (embedded
                            # head or separate drafter -- _mtp_model_for_fit covers both).
                            or _mtp_probe_raised
                        )
                    )
                    _mtp_will_engage = bool(
                        _user_mtp_via_extras or _user_draft_via_extras or _auto_studio_mtp
                    )
                    # The duplicated full target-KV copy (ctx_tgt) is an MTP-only
                    # cost: the MTP head runs a second context over the target
                    # model's own KV geometry. Every separate-drafter mode loads a
                    # distinct drafter with its own KV and keeps no such copy, so
                    # only charge it when the engaged mode is truly MTP. Those modes
                    # arrive by two routes, hence the two conditions: draft-simple
                    # and draft-eagle3 via _user_draft_via_extras (already excluded,
                    # they never set _user_mtp_via_extras), and DSpark/DFlash via
                    # Unsloth's own resolution, which does set _auto_studio_mtp.
                    _engaged_is_mtp = bool(
                        _user_mtp_via_extras
                        or (_auto_studio_mtp and _mtp_effective not in ("dspark", "dflash"))
                    )
                    # Without draft-mtp the embedded head is never loaded, so its blocks
                    # leave the weights the fit prices (about 3 GB on Nemotron 3.5 Super).
                    if not _engaged_is_mtp:
                        _nextn_unloaded = self._unloaded_nextn_bytes(model_path)
                        gguf_size -= _nextn_unloaded
                        model_size -= _nextn_unloaded
                    # The recurrent rollback copies are a WIDER set than that copy
                    # (_TARGET_ROLLBACK_SPEC_TYPES): DSpark and DFlash pay them too,
                    # and draft-simple is the one engaged draft mode that does not,
                    # which is why the pass-through arm reads the types rather than
                    # assuming every extras-owned drafter qualifies.
                    _target_rollback = bool(
                        _auto_studio_mtp
                        or _user_mtp_via_extras
                        or (
                            _user_draft_via_extras
                            and _accumulated_spec_types(extra_args, env = _spec_env)
                            & _TARGET_ROLLBACK_SPEC_TYPES
                        )
                    )

                    # Effective draft depth, by what the CHILD will run on rather than
                    # what was asked for: an extras depth wins (last-wins at launch),
                    # else the field, else the platform default (2 GPU / 3 CPU).
                    _extra_n_max = _extra_args_spec_draft_n_max(extra_args)
                    _mtp_eff_n_max = _extra_n_max
                    if _mtp_eff_n_max is None and _extra_args_set_spec_type(extra_args):
                        # The extras own the spec block, so _build_speculative_flags
                        # returns without emitting a depth at all: neither the request
                        # field nor the platform default below reaches the child. What
                        # does: the inherited env, which survives exactly this case,
                        # then the build's own default. The rollback reserve scales by
                        # this number, so carrying a request field of 2 into a build
                        # that drafts 16 under-reserves by a factor of eight.
                        try:
                            _depth_caps = _launch_caps(binary) or {}
                        except Exception:
                            _depth_caps = {}
                        # The env twin of whichever depth flag this build carries, and
                        # only that one: a legacy build spells the pair --draft-max /
                        # LLAMA_ARG_DRAFT_MAX, and a post-rename build ignores the
                        # legacy name, so a stale value there must not price its
                        # rollback. Both names only where the probe named no flag.
                        _depth_flag = _depth_caps.get("spec_draft_n_max_flag")
                        if _depth_flag == "--draft-max":
                            _env_names = ("LLAMA_ARG_DRAFT_MAX",)
                        elif _depth_flag:
                            _env_names = ("LLAMA_ARG_SPEC_DRAFT_N_MAX",)
                        else:
                            _env_names = ("LLAMA_ARG_SPEC_DRAFT_N_MAX", "LLAMA_ARG_DRAFT_MAX")
                        _env_n_max = next(
                            (v for v in map(_spec_env.get, _env_names) if _is_positive_int(v)),
                            None,
                        )
                        if _env_n_max is not None:
                            _mtp_eff_n_max = int(str(_env_n_max).strip())
                        else:
                            # Never fall through to Unsloth's platform default here:
                            # the child is drafting at the build's number, and a
                            # probe that timed out or printed no default still leaves
                            # it drafting. Assume the deepest llama.cpp has shipped.
                            _mtp_eff_n_max = (
                                _depth_caps.get("spec_draft_n_max_default")
                                or self._UNKNOWN_SPEC_DRAFT_N_MAX
                            )
                    elif _mtp_eff_n_max is None:
                        _mtp_eff_n_max = spec_draft_n_max
                    if _mtp_eff_n_max is None:
                        # _detected_gpus (not gpus) so manual -- which empty
                        # gpus to bypass the planner -- keep the GPU draft depth the
                        # launch flags also use, instead of the CPU default.
                        _mtp_eff_n_max = (
                            3 if _mtp_effective == "dspark" else (2 if _detected_gpus else 3)
                        )
                    # Separate-drafter weights live on GPU (an embedded head is
                    # already in model_size). Size the drafter the launch loads, by
                    # precedence: extras --model-draft (last-wins), else Unsloth's
                    # emitted mtp_draft_path, else the env drafter. Sizing the wrong
                    # one would under-reserve and OOM.
                    _cli_draft_for_budget = _extra_args_mtp_draft_path(extra_args, env = {})
                    _studio_draft_for_budget = (
                        mtp_draft_path
                        if (
                            _mtp_will_engage
                            and mtp_draft_path
                            and not _extra_args_set_spec_type(extra_args)
                        )
                        else None
                    )
                    # Same scrubbed view: reserving for an inherited drafter the child
                    # never loads shrinks the context or rejects a placement that fits.
                    _env_draft_for_budget = _extra_args_mtp_draft_path([], env = _spec_env)
                    _mtp_draft_for_budget = (
                        _cli_draft_for_budget or _studio_draft_for_budget or _env_draft_for_budget
                    )
                    # Will a SEPARATE drafter be emitted at all, taken before the CPU
                    # nulling below. Not mtp_draft_path: extras owning --spec-type return
                    # before Unsloth's sidecar becomes --model-draft, which
                    # _studio_draft_for_budget already encodes.
                    _separate_draft_launches = bool(_mtp_draft_for_budget)
                    # Drafter offloaded to CPU keeps its weights+KV off the GPU, so
                    # drop it from the budget (an embedded head stays in the model).
                    # Consult the env too: the child honors LLAMA_ARG_N_GPU_LAYERS_DRAFT.
                    _draft_on_cpu = _extra_args_draft_offloaded_to_cpu(extra_args, env = os.environ)
                    # Kept for the load-mode fit, which the nulling below would leave
                    # short a whole draft GGUF: dropping the drafter from the VRAM budget
                    # moves those bytes to host RAM rather than deleting them. See
                    # _cpu_resident_draft_bytes.
                    _cpu_draft_path = _mtp_draft_for_budget if _draft_on_cpu else None
                    if _draft_on_cpu:
                        _mtp_draft_for_budget = None
                    _mtp_draft_weights = 0
                    if _mtp_draft_for_budget:
                        try:
                            _mtp_draft_weights = self._get_gguf_size_bytes(_mtp_draft_for_budget)
                        except Exception:
                            _mtp_draft_weights = 0
                    # Draft K/V types (f16 by default; independent extras overrides).
                    _mtp_draft_ck, _mtp_draft_cv = _extra_args_draft_cache_types(extra_args)
                    # The control underneath them: emitted before the extras, so an
                    # extra still last-wins and the reserve prices what will run.
                    _budget_draft_cache = (
                        spec_draft_cache_type.strip().lower() if spec_draft_cache_type else None
                    )
                    if _budget_draft_cache in _VALID_KV_CACHE_TYPES:
                        _mtp_draft_ck = _mtp_draft_ck or _budget_draft_cache
                        _mtp_draft_cv = _mtp_draft_cv or _budget_draft_cache

                    # Byte-accurate reserve when dims allow, else None -> flat fallback.
                    mtp_overhead_fn: Optional[Callable[[int], int]] = None
                    # True when the byte reserve is the drafter weights ONLY because
                    # its KV couldn't be sized; the flat fraction must then stay on
                    # as the cushion for that unsized draft KV (it is not covered by
                    # the weights-only mtp_overhead_fn).
                    _mtp_kv_unsized = False
                    if _mtp_will_engage:
                        _probe_ctx = self._context_length or (
                            effective_ctx if effective_ctx > 0 else 4096
                        )
                        _draft_kv_probe = self._mtp_draft_kv_bytes(
                            _probe_ctx,
                            drafter_path = _mtp_draft_for_budget,
                            draft_cache_type_k = _mtp_draft_ck,
                            draft_cache_type_v = _mtp_draft_cv,
                            n_parallel = n_parallel,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            flash_attn = planned_flash_attn,
                        )
                        if (
                            self._estimate_mtp_overhead_bytes(
                                _probe_ctx,
                                spec_draft_n_max = _mtp_eff_n_max,
                                draft_cache_type_k = _mtp_draft_ck,
                                draft_cache_type_v = _mtp_draft_cv,
                                drafter_path = _mtp_draft_for_budget,
                                draft_weights_bytes = _mtp_draft_weights,
                                n_parallel = n_parallel,
                                mtp_keeps_target_ctx = _engaged_is_mtp,
                                target_rollback = _target_rollback,
                                swa_full = swa_full,
                                kv_unified = planned_kv_unified,
                                n_ubatch = _effective_ubatch,
                                flash_attn = planned_flash_attn,
                            )
                            is not None
                        ):
                            # Reserve is weights-only when the draft KV is unsizable.
                            _mtp_kv_unsized = _draft_kv_probe is None

                            # Closure binding this load's draft params; ctx varies.
                            def mtp_overhead_fn(
                                ctx: int,
                                _n: int = _mtp_eff_n_max,
                                _ck: Optional[str] = _mtp_draft_ck,
                                _cv: Optional[str] = _mtp_draft_cv,
                                _dp: Optional[str] = _mtp_draft_for_budget,
                                _w: int = _mtp_draft_weights,
                                _np: int = n_parallel,
                                _mtp: bool = _engaged_is_mtp,
                                _rollback: bool = _target_rollback,
                                _swa_full: bool = swa_full,
                                _kv_unified: bool = planned_kv_unified,
                                _n_ubatch: Optional[int] = _effective_ubatch,
                            ) -> int:
                                v = self._estimate_mtp_overhead_bytes(
                                    ctx,
                                    spec_draft_n_max = _n,
                                    draft_cache_type_k = _ck,
                                    draft_cache_type_v = _cv,
                                    drafter_path = _dp,
                                    draft_weights_bytes = _w,
                                    n_parallel = _np,
                                    mtp_keeps_target_ctx = _mtp,
                                    target_rollback = _rollback,
                                    swa_full = _swa_full,
                                    kv_unified = _kv_unified,
                                    n_ubatch = _n_ubatch,
                                    # Read at call time, like the KV and compute terms
                                    # beside it: a downgrade re-plans the attention, and a
                                    # default argument would hold the tensor plan's True
                                    # while those price the layer split's padded V, which
                                    # short-changes the draft reserve 4.5x on a hybrid SWA
                                    # draft head. The other captures settle with the load.
                                    flash_attn = planned_flash_attn,
                                )
                                return v if v is not None else 0

                    def _mtp_bytes(
                        ctx: int,
                        slots: Optional[int] = None,
                        ubatch: Optional[int] = None,
                    ) -> int:
                        # slots/ubatch override the closure's requested pair: the reserve
                        # scales with both for compact SWA and MLA+recurrent, so the slot
                        # fit re-prices it per candidate.
                        if mtp_overhead_fn is None:
                            return 0
                        if slots is None:
                            if ubatch:
                                return mtp_overhead_fn(ctx, _n_ubatch = ubatch)
                            return mtp_overhead_fn(ctx)
                        return mtp_overhead_fn(
                            ctx, _np = slots, _n_ubatch = ubatch if ubatch else _effective_ubatch
                        )

                    def _kv_bytes(ctx: int, ctx_checkpoints: int = 0) -> int:
                        # Checkpoints default OFF: placement prices them itself, and only
                        # on unified memory. The load-mode host figure passes a count.
                        return self._estimate_kv_cache_bytes(
                            ctx,
                            cache_type_kv,
                            n_parallel = n_parallel,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            ctx_checkpoints = ctx_checkpoints,
                            flash_attn = planned_flash_attn,
                        )

                    def _cc_bytes(
                        ctx: int,
                        n_gpus: int = 1,
                        slots: int = 0,
                        cache_type: Optional[str] = None,
                    ) -> int:
                        # Context-linear compute-buffer growth (flash-attn KQ mask +
                        # attention scratch); the flat _compute_buffer_pipeline folded
                        # into model_size_fit only covers ctx -> 0. Charged per
                        # candidate context so the fit can't over-pin and spill. The
                        # rate depends on the KV cache type (quantized adds a dequant
                        # scratch), so pass it through -- the LIGHTER axis, since any
                        # quantized axis allocates it and the heavier scalar hides a
                        # paired one. In a layer split this buffer is
                        # replicated on EVERY device (measured ~equal per GPU), so scale
                        # by the device count; a large model at high context otherwise
                        # under-reserves ~(n-1)x it (e.g. Qwen3.5-397B on 3 GPUs). The
                        # per-device rate also steps up once split, unless llama.cpp
                        # declines the pipeline parallelism that causes the step.
                        # A nonzero ``slots`` prices that candidate count and its micro-batch.
                        # Read at call time: a tensor-mode downgrade re-plans it.
                        return max(1, n_gpus) * self._compute_buffer_ctx_bytes(
                            ctx,
                            _ubatch_for_slots(slots) if slots else _effective_ubatch,
                            cache_type or _scratch_cache_type_kv,
                            layer_split = n_gpus > 1 and not _pipeline_parallel_off,
                            flash_attn = planned_flash_attn,
                            n_parallel = slots or n_parallel,
                        )

                    def _cc_split_extra(
                        ctx: int,
                        slots: int = 0,
                        cache_type: Optional[str] = None,
                    ) -> int:
                        # Per-device step from the single-device rate to the split one,
                        # for the paths that must select GPUs before they know the
                        # count. 0 when llama.cpp declines pipeline parallelism.
                        return max(
                            0,
                            _cc_bytes(ctx, 2, slots, cache_type) // 2
                            - _cc_bytes(ctx, 1, slots, cache_type),
                        )

                    def _layer_compute_buffer(slots: int) -> int:
                        cb = self._estimate_compute_buffer_bytes(
                            n_ubatch = _ubatch_for_slots(slots),
                            n_parallel = slots,
                            per_device_tensor = False,
                        )
                        return (
                            cb if cb > 0 else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # Layer-split compute buffer (one lump; tensor mode reserves it
                    # per device in _plan_tensor_parallel). Context-independent, so
                    # fold it into the model footprint for the branches below. Falls
                    # back to the flat reserve when dims are missing (returns 0), a
                    # safe upper bound since the tensor buffer >= the layer one.
                    _compute_buffer_pipeline = self._estimate_compute_buffer_bytes(
                        n_ubatch = _effective_ubatch,
                        n_parallel = n_parallel,
                        per_device_tensor = False,
                    )
                    if _compute_buffer_pipeline <= 0:
                        _compute_buffer_pipeline = (
                            self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # ...and the tensor-mode rate, reserved on EVERY selected device:
                    # each holds the whole buffer rather than a share of it, so the layer
                    # lump above is one device of what a tensor split allocates. Same flat
                    # fallback when the dims are missing.
                    _compute_buffer_tensor = self._estimate_compute_buffer_bytes(
                        n_ubatch = _effective_ubatch,
                        n_parallel = n_parallel,
                        per_device_tensor = True,
                    )
                    if _compute_buffer_tensor <= 0:
                        _compute_buffer_tensor = (
                            self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # Layer split adds a fixed per-device overhead on every GPU. The
                    # folded buffer covers one device; reserve the extra devices'
                    # share so a k-GPU split can't pin a context that OOMs a device
                    # (k=1 adds nothing).
                    _pipeline_overhead_bytes = self._PIPELINE_PER_DEVICE_OVERHEAD_MIB * 1024 * 1024

                    # Auto-cap context to fit VRAM and select GPUs. Explicit n_ctx:
                    # honor it, cap only if it fits no combination. Auto (native):
                    # prefer fewer GPUs with reduced context (multi-GPU is slower).
                    gpu_indices, use_fit = None, True
                    # Per-GPU weight proportions for tensor mode (None lets
                    # llama.cpp split by free VRAM).
                    tp_tensor_split: Optional[list[int]] = None
                    # Whether the tensor-parallel branch below really planned this load.
                    # Not readable from tp_tensor_split, which stays None on a tensor
                    # split llama.cpp is left to size itself. Re-set here, not only
                    # above the try, so a retry through this scope starts clean.
                    _tp_planned = False
                    explicit_ctx = requested_ctx > 0
                    # Nothing speculative on the GPU: the drafter that launches is the
                    # separate CPU-offloaded one, and a sidecar wins over an embedded head
                    # (llama.cpp loads the draft model on has_dft()). A head with no
                    # sidecar is still GPU-resident and still reserved.
                    _draft_cpu_no_embedded = _draft_on_cpu and (
                        _separate_draft_launches or not self._nextn_predict_layers
                    )
                    # A replay was fitted without the drafter: re-fit it like a fresh MTP load, before any explicit_ctx reader.
                    if (
                        explicit_ctx
                        and ctx_override is None
                        and intent.max_seq_length_auto_derived
                        # CPU-pinned drafters too: the target still pays rollback state for them.
                        and _mtp_will_engage
                        # Forced = anything that bypasses the Auto drop probe, advanced arguments included.
                        and (
                            (_canonicalize_spec_mode(speculative_type) or "auto") != "auto"
                            or _user_mtp_via_extras
                            or _user_draft_via_extras
                            or _extra_args_set_spec_type(extra_args)
                            or _extra_args_mtp_draft_path(extra_args, env = _spec_env)
                        )
                    ):
                        explicit_ctx = False
                        _replayed_ctx_refit = True

                    # The two tensor -> layer downgrades that need nothing the probe
                    # decides run BEFORE it: the probe is gated on `not tensor_parallel`
                    # and must answer for any load that ends up layer-split.

                    # The route fallback retry is tensor-off; keep it multi-GPU.
                    if preserve_multi_gpu_on_layer:
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))

                    if tensor_parallel and self._tensor_quant_kv_unsupported_binary(
                        binary, _planned_cache_pair
                    ):
                        # This build already refused a quantized KV cache under a tensor
                        # split; go straight to layer rather than reload the model to be
                        # told again. Keeps the quantized cache, which layer supports.
                        logger.info(
                            "Tensor parallelism skipped: this llama.cpp build cannot "
                            "use a quantized KV cache with --split-mode tensor "
                            "(needs b9455+); loading with a layer split."
                        )
                        tensor_parallel = False
                        # Like the split-axis skip below: a missing capability says
                        # nothing about capacity, so keep the multi-GPU request rather
                        # than let the layer planner settle on the first card it fits.
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                    if tensor_parallel and self._tensor_split_aborts(
                        binary, model_identifier, _planned_cache_pair
                    ):
                        # Aborted on tensor for this model this session (#6415); skip
                        # tensor upfront, layer split serves it.
                        logger.info(
                            "Tensor parallelism skipped: this llama.cpp build aborted "
                            "on --split-mode tensor for this model earlier this "
                            "session; using layer split across %d GPU(s).",
                            len(gpus),
                        )
                        tensor_parallel = False
                        # The abort says nothing about capacity, so keep multi-GPU.
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        # An extras --split-mode tensor would re-engage tensor mode.
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)

                    # Tensor mode replicates a compute buffer on every GPU, so drop
                    # GPUs below that reserve from the set up front (gpu_indices
                    # becomes the CUDA_VISIBLE_DEVICES mask, fully excluding them).
                    tp_gpus = gpus
                    # Manual mode owns the layer count and context, so it skips
                    # the memory-based planner; its toggle still emits
                    # --split-mode tensor below (split by free VRAM, or by the
                    # Split ratio if set). auto plans here.
                    plan_tp = tensor_parallel and gpu_memory_mode != "manual"
                    if plan_tp:
                        # Deterministic per-device compute buffer (replicated on
                        # every device in tensor mode); flat fallback when dims
                        # are unavailable. _plan_tensor_parallel uses the same.
                        _tp_reserve_bytes = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = True,
                        )
                        reserve_mib = (
                            _tp_reserve_bytes // (1024 * 1024)
                            if _tp_reserve_bytes > 0
                            else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
                        )
                        # Admit by usable budget (free - (1-frac)*total), not raw
                        # free: a partly-used big card can clear the reserve on raw
                        # free yet have no budget left.
                        tp_gpus = [g for g in gpus if _gpu_usable(g) >= reserve_mib]

                    if plan_tp and len(tp_gpus) < 2:
                        # Tensor parallelism needs >= 2 usable GPUs. On a single
                        # GPU --split-mode tensor is a no-op; with 0 GPUs (CPU-only
                        # or probe failed) it must not reach llama-server; and a
                        # GPU below the buffer reserve can't participate. Drop the
                        # flag and fall through to normal layer/CPU allocation.
                        logger.info(
                            "Tensor parallelism requested but only %d of %d GPU(s) "
                            "have enough free VRAM for the compute buffer; "
                            "ignoring (needs >= 2).",
                            len(tp_gpus),
                            len(gpus),
                        )
                        tensor_parallel = False
                        # GPUs below tensor's compute-buffer reserve can still do layer
                        # split, so keep multi-GPU (mirrors the budget/geometry drops);
                        # _select_gpus caps unusable cards.
                        if len(gpus) >= 2:
                            _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)

                    # Normalize the speculative reserve BEFORE anything prices it.
                    # _mtp_bytes reads mtp_overhead_fn at call time, so leaving this
                    # until after the probes below charged the projector probe the full
                    # GPU-drafter footprint for a drafter pinned to the CPU (-ngld 0),
                    # which allocates none of it: the probe then answers "does not fit",
                    # pins the projector, and costs ~8.8x on every image encode for
                    # memory nothing was holding. _draft_cpu_no_embedded is settled well
                    # above, and every input this needs with it, so there is nothing to
                    # wait for.
                    if _draft_cpu_no_embedded and mtp_overhead_fn is not None:
                        # Nothing speculative is GPU-resident: the drafter that launches
                        # is the separate CPU-offloaded one, and any embedded head it
                        # displaced does not run. The flat fraction below is gated on
                        # this; the byte-accurate callback was not, so the fit went on
                        # charging VRAM no drafter allocates.
                        # Two terms survive: a recurrent target's rollback snapshots and
                        # the verification output rows sit in the TARGET context, so
                        # pinning the drafter to CPU does not move them. Charge those
                        # alone. Flat in ctx (both are per-slot), hence the same
                        # _np/_n_ubatch keywords the replaced callback takes, so
                        # _mtp_bytes can still re-price slots.
                        def _cpu_draft_target_state(
                            _ctx: int,
                            _np: int = n_parallel,
                            _n_ubatch: Optional[int] = _effective_ubatch,
                            _n: int = _mtp_eff_n_max,
                            _rollback: bool = _target_rollback,
                        ) -> int:
                            if _n <= 0:
                                return 0
                            rollback = self._rollback_state_bytes(_np) * _n if _rollback else 0
                            return rollback + self._spec_verify_rows_bytes(_np, _n)

                        mtp_overhead_fn = (
                            _cpu_draft_target_state
                            if _cpu_draft_target_state(effective_ctx) > 0
                            else None
                        )
                        _mtp_kv_unsized = False

                    # Projector placement, the FIRST thing given up when the load does
                    # not fit: the encoder runs once per image, layers once per token,
                    # so host RAM is its better home when it is what costs residency.
                    # Ahead of the drafter (a bounded per-image cost is cheaper to give
                    # up than a per-token speedup); never a reason to drop vision, which
                    # would answer a pasted screenshot text-only with no signal.
                    #
                    # Does the pin free anything? Only on a device with a budget of its
                    # own. Apple unified memory and AMD APUs / iGPUs report free SYSTEM
                    # RAM as free "VRAM", so the encoder just moves inside one pool.
                    # Their marker is a total of 0, which also covers a total the probe
                    # could not read (MIG/vGPU) -- same answer either way.
                    #
                    # Asked of the budgeted devices, not all of them: a laptop pairing a
                    # discrete card with an APU enumerates both, and demanding every one
                    # be discrete refused the pin there, keeping a projector the discrete
                    # card could have been rid of.
                    # Applied here, or after the pooled weight-budget check gives
                    # tensor mode up. One body, so the two sites cannot drift.
                    _mm_pin_deferred = False

                    def _apply_mmproj_cpu_pin(floor_ctx: int) -> None:
                        nonlocal _mmproj_cpu_pinned, _mmproj_pinned_bytes
                        nonlocal mmproj_size, model_size
                        _mmproj_cpu_pinned = True
                        _mmproj_pinned_bytes = mmproj_size
                        # Everything downstream prices the projector off these two,
                        # including the drop probe below: it sees the lighter footprint
                        # and gives the drafter up only if the pin was not enough.
                        mmproj_size = 0
                        model_size = gguf_size
                        # What the startup-recovery pin reports, so a CPU-resident
                        # projector is announced whether it was predicted or salvaged;
                        # otherwise image encoding just gets mysteriously slower. After
                        # the per-load reset, before the recovery paths, which override
                        # it with their more specific reason.
                        self._mmproj_fallback_reason = "cpu_offload"
                        logger.info(
                            "Auto: running the vision projector on the CPU "
                            "(--no-mmproj-offload); it does not fit in VRAM "
                            "alongside the model at %d context. Image encoding "
                            "gets slower, text generation is unaffected. Pass "
                            "--mmproj-offload in the advanced arguments to keep it "
                            "on the GPU.",
                            floor_ctx,
                        )

                    _mm_budgeted_gpus = [
                        (_idx, _free) for _idx, _free in gpus if total_by_idx.get(_idx, 0) > 0
                    ]
                    _discrete_vram = bool(_mm_budgeted_gpus)
                    if (
                        effective_is_vision
                        and mmproj_size > 0
                        and _discrete_vram
                        and not _mmproj_cpu_pinned
                        and gpu_memory_mode != "manual"
                        # Not gated on tensor_parallel: these are layer-split numbers,
                        # so the ANSWER is still never applied to a tensor load, but the
                        # pooled weight-budget check below can end one, and it prices
                        # weights + projector -- it downgrades exactly the loads moving
                        # the encoder would have rescued. Asking here and applying after
                        # that downgrade is the non-circular order; making the check
                        # pin-aware would decide TP from a pin decided from TP.
                        and _paravirtual_mmproj_pinnable(server_caps)
                        # Last-wins on the placement pair, so whoever named either
                        # spelling owns it. The environment counts: arg.cpp applies
                        # set_env before argv, so --no-mmproj-offload here would
                        # OVERWRITE LLAMA_ARG_MMPROJ_OFFLOAD=1, not lose to it.
                        and not _extra_args_set_mmproj_offload(extra_args)
                        and not _env_sets_mmproj_offload()
                    ):
                        # One question, asked once: at the context this load will
                        # actually run at, with speculative decoding STILL LIVE, does
                        # model + drafter + projector stay resident in VRAM? Which
                        # context that is, is the whole question.
                        #
                        # AUTO gets the FLOOR, which makes this a residency question and
                        # not a context one. Auto shrinks the CONTEXT before it spills
                        # layers, reaching `--fit on` only once even _FIT_MIN_CTX will
                        # not place -- and then RAISES the context to _AUTO_OFFLOAD_CTX,
                        # since offload is already unavoidable by that point. So this
                        # floor is deliberately the residency floor and not the context
                        # the load ends up running at; the projector's bytes are charged
                        # again at the real context through _subset_model_size below.
                        # Pricing at the native length would answer "does not fit" for a
                        # load merely heading for a smaller context and pin a projector
                        # that was resident all along -- 8.8x on image encode for nothing.
                        #
                        # An EXPLICIT context gets its requested length, because there
                        # the reasoning points the other way: the loop honors a pinned
                        # context verbatim, so the only give left is `--fit on`, which
                        # spills MODEL LAYERS and collapses decode ~3x. Asking at the
                        # floor instead would answer "fits", keep the projector, and pay
                        # in offloaded weights -- this probe's trade at its worst rate.
                        #
                        # The drafter IS charged: this runs before the drop probe, so
                        # speculative decoding is still in the footprint. Priced exactly
                        # where the fit prices it below, term for term.
                        _mm_mtp_on_gpu = _mtp_will_engage and not _draft_cpu_no_embedded
                        _mm_frac = _vram_frac - (
                            _MTP_VRAM_RESERVE_FRAC
                            if (_mm_mtp_on_gpu and (mtp_overhead_fn is None or _mtp_kv_unsized))
                            else 0.0
                        )
                        _mm_floor_ctx = (
                            effective_ctx
                            if explicit_ctx
                            else min(self._MMPROJ_FIT_FLOOR_CTX, effective_ctx)
                        )
                        _mm_need = (
                            gguf_size
                            + mmproj_size
                            + _compute_buffer_pipeline
                            + self._CUDA_CONTEXT_RESERVE_BYTES
                            + int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                            + (self._MTP_DRAFT_COMPUTE_BYTES if _mm_mtp_on_gpu else 0)
                        )
                        if _mm_floor_ctx > 0 and self._can_estimate_kv():
                            _mm_need += (
                                _kv_bytes(_mm_floor_ctx)
                                + _cc_bytes(_mm_floor_ctx)
                                + _mtp_bytes(_mm_floor_ctx)
                            )
                        else:
                            # No KV estimate: the fit builds one flat footprint against
                            # _mtp_bytes at the native length and no context it hands
                            # out lowers it, so charging the reserve at a floor the fit
                            # never applies would promise room the fit never sees.
                            # Rebound rather than used inline so the log below names the
                            # length this actually priced against.
                            _mm_floor_ctx = self._context_length or effective_ctx or 4096
                            _mm_need += _mtp_bytes(_mm_floor_ctx)
                        # Run through whichever selector the branch below will use, at
                        # the same fraction and per-device overhead, so "does not fit"
                        # here is a placement that branch could not have chosen either.
                        # A probe more optimistic than the placement it gates is the
                        # failure mode that matters: projector kept, layers spilled.
                        #
                        # Split-aware, because a layer split replicates a BIGGER
                        # context-compute buffer per device (_CTX_COMPUTE_SPLIT_MULT):
                        # at 65536 on two cards, 96 MiB charged against 768 MiB spent, a
                        # gap wider than the projector surcharge being decided. Mirrored
                        # term for term, including the single-to-split step.
                        #
                        # Plain only where the placement itself is plain: with no KV
                        # estimate it falls to the file-size-only path, which is
                        # _select_gpus at the flat overhead. Same condition that decided
                        # whether _mm_need carries a context-compute term, so the two
                        # cannot drift apart.
                        _mm_split_aware = self._can_estimate_kv()
                        _, _mm_needs_fit = (
                            self._select_gpus_split_aware(
                                _mm_need,
                                _mm_budgeted_gpus,
                                usable_fraction = _mm_frac,
                                total_by_idx = total_by_idx,
                                per_device_overhead_bytes = _pipeline_overhead_bytes
                                + _cc_bytes(_mm_floor_ctx),
                                min_gpus = _layer_min_gpus,
                                split_extra_bytes = _cc_split_extra(_mm_floor_ctx),
                            )
                            if _mm_split_aware
                            else self._select_gpus(
                                _mm_need,
                                _mm_budgeted_gpus,
                                usable_fraction = _mm_frac,
                                total_by_idx = total_by_idx,
                                per_device_overhead_bytes = _pipeline_overhead_bytes,
                                min_gpus = _layer_min_gpus,
                            )
                        )
                        if _mm_needs_fit:
                            if tensor_parallel:
                                # Held, not dropped. Applying it here would price a
                                # tensor load off layer-split numbers; discarding it
                                # would lose the verdict the downgrade below needs.
                                _mm_pin_deferred = True
                            else:
                                _apply_mmproj_cpu_pin(_mm_floor_ctx)

                    # Target pins but the drafter's reserve tips it over: Auto drops the
                    # drafter rather than pay for speed with context (or a --fit offload,
                    # where decode collapses ~3x). Priced over the SAME ranked subsets the
                    # placement loop walks, at the context the target alone would get, so
                    # it drops only when no placement could have held both. Carried to the
                    # launch as drafter_no_vram. Skipped under tensor parallelism, whose
                    # per-device buffer has its own geometry these numbers do not model;
                    # `tensor_parallel` already reflects the downgrades hoisted above, so
                    # a request that ends up layer-split IS probed. The pooled tensor
                    # weight-budget check below still escapes, since it prices the reserve
                    # this probe may clear and running it first would be circular. The
                    # gate reads the user's choice, not _mtp_effective, which by here is
                    # the kind Auto resolved and can no longer tell the two apart.
                    if (
                        _mtp_will_engage
                        and (_canonicalize_spec_mode(speculative_type) or "auto") == "auto"
                        and not _user_mtp_via_extras
                        and not _user_draft_via_extras
                        and not _extra_args_set_spec_type(extra_args)
                        # A bare --model-draft / --spec-draft-hf sets neither flag above
                        # (both key off --spec-type), yet llama.cpp loads whatever it
                        # names anyway: load_model gates the draft on has_dft(). Dropping
                        # here would free the reserve for a drafter that still loads.
                        and not _extra_args_mtp_draft_path(extra_args, env = _spec_env)
                        and not _draft_cpu_no_embedded
                        and not tensor_parallel
                        and gpus
                        and effective_ctx > 0
                        and self._can_estimate_kv()
                    ):

                        def _probe_frac(drafter: bool) -> float:
                            # The flat fraction is the reserve whenever _mtp_bytes is 0.
                            # Same budget the fit uses, else the probe answers for a
                            # card size the fit never tests.
                            return _vram_frac - (
                                _MTP_VRAM_RESERVE_FRAC
                                if (drafter and (mtp_overhead_fn is None or _mtp_kv_unsized))
                                else 0.0
                            )

                        def _probe_base(
                            drafter: bool,
                            n: int,
                            slots: int = 0,
                        ) -> int:
                            # model_size_fit's terms, before it is built below.
                            _soft = self._CUDA_CONTEXT_RESERVE_BYTES
                            if effective_is_vision and mmproj_size > 0:
                                _soft += int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                            if drafter:
                                _soft += self._MTP_DRAFT_COMPUTE_BYTES
                            return (
                                model_size
                                + (
                                    _layer_compute_buffer(slots)
                                    if slots
                                    else _compute_buffer_pipeline
                                )
                                + _soft
                                + max(0, n - 1) * _pipeline_overhead_bytes
                            )

                        # Fewest GPUs first, ranked by usable VRAM: the same order and
                        # the same budget the auto placement loop uses, so the answer is
                        # about placements that loop could actually choose.
                        def _probe_rank(drafter: bool) -> list:
                            return sorted(
                                gpus,
                                key = lambda g: _gpu_usable(g, _probe_frac(drafter)),
                                reverse = True,
                            )

                        _probe_overhead_mib = _pipeline_overhead_bytes / (1024 * 1024)

                        def _probe_floor(ranked: list, drafter: bool) -> int:
                            # Same floor as the placement loop's _auto_min_gpus: a tensor
                            # downgrade raises _layer_min_gpus to keep the request
                            # multi-GPU, and a one-GPU placement the loop is forbidden to
                            # choose must not be the one that saves the drafter.
                            return max(
                                1,
                                min(
                                    _layer_min_gpus,
                                    sum(
                                        1
                                        for g in ranked
                                        if _gpu_usable(g, _probe_frac(drafter))
                                        > _probe_overhead_mib
                                    )
                                    or 1,
                                ),
                            )

                        # Both rankings: an unsized drafter costs five points of pin
                        # fraction, which can reorder heterogeneous cards (a busy 80 GB
                        # can lead at 0.97, an idle 24 GB at 0.92). The placement loop
                        # sorts under the fraction that load ends up with, so a prefix
                        # only the drafter's ranking produces is still one it can choose.
                        # Identical whenever the orders agree, i.e. every uniform machine.
                        _probe_orders = [_probe_rank(False)]
                        _ranked_w = _probe_rank(True)
                        if [id(g) for g in _ranked_w] != [id(g) for g in _probe_orders[0]]:
                            _probe_orders.append(_ranked_w)
                        _target_fits_somewhere = False
                        _both_fit_somewhere = False
                        _probe_ctx = 0
                        _probe_need = _probe_have = 0.0
                        # (GPU count, target-only ctx, budget with drafter) per drafter-ranked subset.
                        _target_placements: list[tuple[int, int, float]] = []
                        _drafter_order = _probe_orders[-1]
                        for _probe_ranked in _probe_orders:
                            if _both_fit_somewhere:
                                break
                            _probe_min_gpus = _probe_floor(_probe_ranked, False)
                            for _n in range(_probe_min_gpus, len(_probe_ranked) + 1):
                                _subset = _probe_ranked[:_n]
                                _cc_n = lambda c, _k = _n: _cc_bytes(c, _k)
                                _base_wo = _probe_base(False, _n)
                                _budget_wo = _pool_budget_mib(_subset, _probe_frac(False))
                                # Explicit context is honored verbatim, so that is what the
                                # drafter has to fit alongside; Auto gets its own best cap.
                                _ctx_wo = (
                                    effective_ctx
                                    if explicit_ctx
                                    else self._fit_context_to_vram(
                                        effective_ctx,
                                        _budget_wo,
                                        _base_wo,
                                        cache_type_kv,
                                        swa_full = swa_full,
                                        n_parallel = n_parallel,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = _effective_ubatch,
                                        flash_attn = planned_flash_attn,
                                        mtp_engaged = False,
                                        mtp_overhead_fn = None,
                                        compute_ctx_bytes_fn = _cc_n,
                                        budget_frac = 1.0,
                                        pooled = True,
                                        total_mib = None,
                                        ctx_checkpoints = _placement_ctx_checkpoints,
                                    )
                                )
                                if _ctx_wo <= 0:
                                    continue
                                _shared = _kv_bytes(_ctx_wo) + _cc_n(_ctx_wo)
                                _foot_wo = (_base_wo + _shared) / (1024 * 1024)
                                if _foot_wo > _budget_wo:
                                    continue
                                # The pooled figure hides the compute buffer every device
                                # replicates, so the weakest card may not hold the context
                                # the pool priced and the placement loop caps to what it
                                # does. Charging the drafter at the uncapped context
                                # condemns it at a context this load never reaches. Same
                                # reserve and cap the auto-context loop below applies, so
                                # the two cannot disagree. Auto only: an explicit context
                                # is honored verbatim and overflows to --fit.
                                _usable_wo = [_gpu_usable(g, _probe_frac(False)) for g in _subset]
                                _probe_reserve_at = lambda c, _k = _n: (
                                    (_pipeline_overhead_bytes if _k > 1 else 0)
                                    + _cc_bytes(c, _k) // _k
                                )
                                if not self._every_gpu_holds_reserve(
                                    _usable_wo, _probe_reserve_at(_ctx_wo)
                                ):
                                    if explicit_ctx:
                                        # Nothing to cap: this subset cannot hold it, and
                                        # _select_gpus_split_aware will go to --fit too.
                                        # Calling it a fit would drop the drafter and then
                                        # report a pin that never happened.
                                        continue
                                    _ctx_wo = self._cap_ctx_to_per_device_reserve(
                                        _ctx_wo, _usable_wo, _probe_reserve_at
                                    )
                                    if _ctx_wo <= 0:
                                        continue
                                    # Every pooled term shrinks with the context, so this
                                    # cannot newly fail; re-price rather than lean on it.
                                    _shared = _kv_bytes(_ctx_wo) + _cc_n(_ctx_wo)
                                    _foot_wo = (_base_wo + _shared) / (1024 * 1024)
                                    if _foot_wo > _budget_wo:
                                        continue
                                _foot_w = (
                                    _probe_base(True, _n) + _shared + _mtp_bytes(_ctx_wo)
                                ) / (1024 * 1024)
                                _budget_w = _pool_budget_mib(_subset, _probe_frac(True))
                                if not _target_fits_somewhere:
                                    _target_fits_somewhere = True
                                    _probe_ctx, _probe_need, _probe_have = (
                                        _ctx_wo,
                                        _foot_w,
                                        _budget_w,
                                    )
                                if _probe_ranked is _drafter_order:
                                    _target_placements.append((_n, _ctx_wo, _budget_w))
                                if _foot_w <= _budget_w:
                                    _both_fit_somewhere = True
                                    break
                                # Both do not fit at the target's own context. The loop does
                                # not move on: it re-caps the context WITH the drafter and
                                # takes this subset at whatever that leaves. So a smaller
                                # context holding both here IS the placement the load takes,
                                # paying for the drafter in context, which is the trade being
                                # refused. Only if no context fits both does it widen.
                                _ctx_w = (
                                    0
                                    if explicit_ctx
                                    else self._fit_context_to_vram(
                                        _ctx_wo,
                                        _budget_w,
                                        _probe_base(True, _n),
                                        cache_type_kv,
                                        swa_full = swa_full,
                                        n_parallel = n_parallel,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = _effective_ubatch,
                                        flash_attn = planned_flash_attn,
                                        mtp_engaged = True,
                                        mtp_overhead_fn = mtp_overhead_fn,
                                        compute_ctx_bytes_fn = _cc_n,
                                        budget_frac = 1.0,
                                        pooled = True,
                                        total_mib = None,
                                        ctx_checkpoints = _placement_ctx_checkpoints,
                                    )
                                )
                                if (
                                    _ctx_w > 0
                                    and (
                                        _probe_base(True, _n)
                                        + _kv_bytes(_ctx_w)
                                        + _cc_n(_ctx_w)
                                        + _mtp_bytes(_ctx_w)
                                    )
                                    / (1024 * 1024)
                                    <= _budget_w
                                ):
                                    # A NextN-only head is the main decode speedup: pay context.
                                    if (
                                        _engaged_is_mtp
                                        and _arch_has_fast_mla_mtp(
                                            getattr(self, "_architecture", None)
                                        )
                                        and _ctx_w >= min(_ctx_wo, _FAST_MTP_MIN_CTX)
                                    ):
                                        _both_fit_somewhere = True
                                        logger.info(
                                            "Auto: keeping the embedded MTP head; context %d -> %d "
                                            "so it fits (%.1f GB budget).",
                                            _ctx_wo,
                                            _ctx_w,
                                            _budget_w / 1024,
                                        )
                                    break
                        # Fewer slots before no drafter. The walk takes the first subset holding the
                        # drafter, so it must keep its target-only ctx (less the trim at one slot).
                        # An extras --parallel wins at launch, so it is kept; the env twin is scrubbed.
                        if (
                            _target_fits_somewhere
                            and not _both_fit_somewhere
                            and n_parallel > 1
                            and intent.n_parallel_from_default
                            and _extra_args_n_parallel(extra_args, env = {}) is None
                        ):

                            def _mtp_foot_mib(n: int, ctx: int, s: int) -> float:
                                ub = _ubatch_for_slots(s)
                                return (
                                    _probe_base(True, n, s)
                                    + self._estimate_kv_cache_bytes(
                                        ctx,
                                        cache_type_kv,
                                        n_parallel = s,
                                        swa_full = swa_full,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = ub,
                                        flash_attn = planned_flash_attn,
                                    )
                                    + _cc_bytes(ctx, n, s)
                                    + _mtp_bytes(ctx, s, ub)
                                ) / (1024 * 1024)

                            def _ctx_with_drafter(n: int, ctx: int, budget: float, s: int) -> int:
                                if explicit_ctx:
                                    return ctx if _mtp_foot_mib(n, ctx, s) <= budget else 0
                                ub = _ubatch_for_slots(s)
                                fitted = self._fit_context_to_vram(
                                    ctx,
                                    budget,
                                    _probe_base(True, n, s),
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = s,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = ub,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = True,
                                    mtp_overhead_fn = (
                                        functools.partial(mtp_overhead_fn, _np = s, _n_ubatch = ub)
                                        if mtp_overhead_fn is not None
                                        else None
                                    ),
                                    compute_ctx_bytes_fn = lambda c: _cc_bytes(c, n, s),
                                    budget_frac = 1.0,
                                    pooled = True,
                                    total_mib = None,
                                    ctx_checkpoints = _placement_ctx_checkpoints,
                                )
                                return (
                                    fitted
                                    if fitted > 0 and _mtp_foot_mib(n, fitted, s) <= budget
                                    else 0
                                )

                            _mtp_slots = _mtp_slots_ctx = 0
                            for _s in range(n_parallel - 1, 0, -1):
                                _keep = 1 - (
                                    _MTP_SLOT_CTX_TRIM if _s == 1 and not explicit_ctx else 0
                                )
                                for _n, _ctx_c, _budget_c in _target_placements:
                                    _c = _ctx_with_drafter(_n, _ctx_c, _budget_c, _s)
                                    if _c <= 0:
                                        continue
                                    if _c >= _ctx_c * _keep:
                                        _mtp_slots, _mtp_slots_ctx = _s, _c
                                        _mtp_slots_target_ctx = _ctx_c
                                    break
                                if _mtp_slots:
                                    break
                            if _mtp_slots:
                                logger.info(
                                    "Auto: serving slots reduced %d -> %d so the drafter fits "
                                    "in VRAM at context %d (target alone: %d). Set Parallel "
                                    "Slots to keep %d without the drafter.",
                                    n_parallel,
                                    _mtp_slots,
                                    _mtp_slots_ctx,
                                    _probe_ctx,
                                    n_parallel,
                                )
                                _mtp_slots_asked = n_parallel
                                n_parallel = _mtp_slots  # allow-slot-clamp: the drafter fits in VRAM only at fewer slots
                                _effective_ubatch = _ubatch_for_slots(n_parallel)
                                _compute_buffer_pipeline = _layer_compute_buffer(n_parallel)
                                if mtp_overhead_fn is not None:
                                    # Its defaults were bound at the requested count.
                                    mtp_overhead_fn = functools.partial(
                                        mtp_overhead_fn,
                                        _np = n_parallel,
                                        _n_ubatch = _effective_ubatch,
                                    )
                                _both_fit_somewhere = True
                        if _target_fits_somewhere and not _both_fit_somewhere:
                            _spec_dropped_no_vram = True
                            _mtp_will_engage = False
                            # Clearing the flag alone leaves the reserve: the _mtp_bytes
                            # sites are unconditional and _fit_context_to_vram invokes any
                            # non-None mtp_overhead_fn whatever mtp_engaged says, so the
                            # fit would still shrink the context for a drafter that no
                            # longer launches. _mtp_kv_unsized goes with it, or
                            # _flat_mtp_engages swaps the flat fraction in as replacement.
                            mtp_overhead_fn = None
                            _mtp_kv_unsized = False
                            logger.warning(
                                "Speculative decoding disabled for this load: the model "
                                "fits in VRAM at context %d but its drafter does not "
                                "(needs %.1f GB of a %.1f GB budget, on any GPU subset). "
                                "Auto keeps the context rather than shrink it for a speed "
                                "option. Select the drafter in Settings to force it.",
                                _probe_ctx,
                                _probe_need / 1024,
                                _probe_have / 1024,
                            )

                    # Flat MTP reserve fraction: used only as the fallback when the
                    # byte-accurate mtp_overhead_fn can't size the draft KV (dims
                    # unavailable, or _mtp_kv_unsized = weights-only). A separate
                    # drafter on CPU uses no GPU (no reserve); an embedded head is on
                    # GPU regardless of draft-offload flags (keep its reserve).
                    _flat_mtp_engages = _mtp_will_engage and (
                        mtp_overhead_fn is None or _mtp_kv_unsized
                    )
                    # MTP reserves GPU VRAM unless its only drafter is a separate
                    # CPU-offloaded one (an embedded head stays on GPU). The tensor
                    # path reserves like the layer path; gate both on this.
                    _mtp_reserves_gpu = _mtp_will_engage and not _draft_cpu_no_embedded
                    # Narrower than that: whether ANY byte reserve survives on the GPU.
                    # A CPU-pinned drafter leaves the target's own rollback state behind
                    # (the block above keeps it), and the tensor floor has to hold it,
                    # tensor mode having no --fit valve and no amount of context
                    # shrinking freeing a per-slot allocation. It stays out of
                    # _mtp_reserves_gpu because that also charges the draft decode
                    # graph, which follows the drafter onto the CPU.
                    _mtp_gpu_bytes_remain = _mtp_reserves_gpu or mtp_overhead_fn is not None
                    _flat_mtp_reserve = (
                        _MTP_VRAM_RESERVE_FRAC
                        if (_flat_mtp_engages and not _draft_cpu_no_embedded)
                        else 0.0
                    )
                    _pin_fraction = _vram_frac - _flat_mtp_reserve

                    # Charge the soft overhead _CTX_FIT_VRAM_FRACTION under-covers on tight
                    # tiers, gated so plain dense loads (#5106) only pay the CUDA-ctx base.
                    # CUDA/cuBLAS context is discrete-GPU only (not Metal); the mmproj and
                    # MTP draft-graph buffers exist on every backend.
                    _soft_overhead = self._CUDA_CONTEXT_RESERVE_BYTES if gpus else 0
                    if effective_is_vision and mmproj_size > 0:
                        _soft_overhead += int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                    if _mtp_reserves_gpu:
                        _soft_overhead += self._MTP_DRAFT_COMPUTE_BYTES
                    # A CPU-pinned projector left model_size, which is right only where
                    # the bytes it gave up landed somewhere else. Shared memory -- an APU /
                    # iGPU pool, Apple unified memory, or a total the probe could not read
                    # -- has nowhere else, so the encoder sits in the very memory this fit
                    # measures and the context can spend those bytes twice.
                    #
                    # Near _discrete_vram's question but not the same one: that drops
                    # shared devices to ask whether the pin frees anything ANYWHERE, and
                    # one discrete card settles that. The fit asks what a candidate subset
                    # draws on, and the selection walks prefixes of `gpus`, so a shared
                    # device ranked first is a candidate alone. Measured: 9000 MiB of
                    # shared pool beside a nearly full discrete card fitted a 6 GiB model
                    # to 32000 tokens with a 1 GiB projector pinned into that same pool.
                    # So the charge follows the PRESENCE of shared memory, not its absence.
                    # On a mixed host that ends up purely discrete this over-charges by the
                    # projector, costing context rather than correctness.
                    _shared_pool_mmproj = (
                        _mmproj_pinned_bytes
                        if (not gpus or len(_mm_budgeted_gpus) < len(gpus))
                        else 0
                    )
                    model_size_fit = (
                        model_size + _compute_buffer_pipeline + _soft_overhead + _shared_pool_mmproj
                    )

                    def _subset_model_size(n_gpus: int) -> int:
                        return model_size_fit + max(0, n_gpus - 1) * _pipeline_overhead_bytes

                    # Unified-memory budget (0 off Apple Silicon) for the no-GPU Metal cap below.
                    _apple_budget_mib = self._apple_metal_memory_budget_bytes() // (1024 * 1024)

                    if tensor_parallel and tp_gpus:
                        # Pooled usable budget (after each device's compute buffer)
                        # must hold the non-shrinkable footprint: weights + the MTP
                        # reserve. The planner can shrink ctx/KV, not these.
                        _tp_weight_budget_mib = (
                            sum(_gpu_usable(g) for g in tp_gpus) - len(tp_gpus) * reserve_mib
                        )
                        _tp_flat_mtp = 2 * 1024**3  # flat reserve when dims unavailable
                        if not _mtp_gpu_bytes_remain:
                            # No MTP, or its only drafter is CPU-offloaded and the
                            # target keeps no state of its own (no GPU bytes at all).
                            _tp_mtp_floor = 0
                        elif mtp_overhead_fn is not None and not _mtp_kv_unsized:
                            _tp_mtp_floor = _mtp_bytes(
                                min(2048, effective_ctx) if effective_ctx > 0 else 2048
                            )
                        else:
                            # Dims unavailable / weights-only: tensor mode has no
                            # --fit valve, so keep the flat reserve as the unsized-KV
                            # cushion, never below the known byte reserve.
                            _tp_mtp_floor = max(
                                _tp_flat_mtp,
                                _mtp_bytes(min(2048, effective_ctx) if effective_ctx > 0 else 2048),
                            )
                        _tp_required_mib = (model_size + _tp_mtp_floor + _soft_overhead) / (
                            1024 * 1024
                        )
                        if _tp_weight_budget_mib <= _tp_required_mib:
                            logger.info(
                                "Tensor parallelism requested but the pooled VRAM "
                                "budget cannot hold the weights, MTP reserve, and "
                                "per-device compute buffers; falling back to layer split."
                            )
                            tensor_parallel = False
                            # Weights needed >1 card, so keep multi-GPU across the
                            # usable tensor GPUs.
                            if len(tp_gpus) >= 2:
                                _layer_min_gpus = max(_layer_min_gpus, len(tp_gpus))
                            extra_args = strip_split_mode_only(extra_args)
                            planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                            # Layer split now, so the withheld verdict applies -- but not
                            # when a GPU drafter is still in play. The drafter-drop probe
                            # is gated off under tensor parallelism, so it has not run;
                            # the documented order is projector first and drafter second,
                            # and pinning here without being able to re-ask the second
                            # half pays the encoder cost AND still reaches --fit on.
                            # Re-running that probe after this downgrade is the complete
                            # fix.
                            if _mm_pin_deferred and not _mtp_reserves_gpu:
                                _apply_mmproj_cpu_pin(_mm_floor_ctx)
                                _mm_pin_deferred = False
                                # Rebuild what was derived from model_size before the
                                # pin. _subset_model_size closes over model_size_fit by
                                # name, so rebinding it carries to the fit below.
                                _soft_overhead = self._CUDA_CONTEXT_RESERVE_BYTES if gpus else 0
                                if _mtp_reserves_gpu:
                                    _soft_overhead += self._MTP_DRAFT_COMPUTE_BYTES
                                _shared_pool_mmproj = (
                                    _mmproj_pinned_bytes
                                    if (not gpus or len(_mm_budgeted_gpus) < len(gpus))
                                    else 0
                                )
                                model_size_fit = (
                                    model_size
                                    + _compute_buffer_pipeline
                                    + _soft_overhead
                                    + _shared_pool_mmproj
                                )

                    if tensor_parallel and tp_gpus:
                        # Tensor-parallel allocation; see _plan_tensor_parallel.
                        target_ctx = (
                            effective_ctx
                            if explicit_ctx
                            else (self._context_length or effective_ctx)
                        )
                        # When the draft KV couldn't be sized (weights-only reserve),
                        # the planner's mtp_overhead_fn is non-None but covers only
                        # weights, so pass the flat cushion for the unsized KV (else
                        # the binary search spends it on context).
                        _tp_unsized_mtp_reserve = (
                            2 * 1024**3 if (_mtp_reserves_gpu and _mtp_kv_unsized) else 0
                        )
                        (
                            effective_ctx,
                            max_available_ctx,
                            gpu_indices,
                            tp_tensor_split,
                        ) = self._plan_tensor_parallel(
                            tp_gpus,
                            model_size,
                            target_ctx,
                            cache_type_kv = cache_type_kv,
                            scratch_cache_type_kv = _scratch_cache_type_kv,
                            n_parallel = n_parallel,
                            mtp_engaged = _mtp_reserves_gpu,
                            mtp_overhead_fn = mtp_overhead_fn,
                            mtp_flat_reserve_bytes = _tp_unsized_mtp_reserve,
                            # Report the UI ceiling from native ctx, not the
                            # explicit small request.
                            max_target_ctx = self._context_length or target_ctx,
                            total_by_idx = total_by_idx,
                            n_ubatch = _effective_ubatch,
                            soft_overhead_bytes = _soft_overhead,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            flash_attn = planned_flash_attn,
                            # Already resolved, not a second read: a mid-load save
                            # would otherwise size placement against a fraction the
                            # ranking above never used.
                            vram_fraction = _vram_frac,
                            explicit_ctx = explicit_ctx,
                        )
                        use_fit = False
                        _tp_planned = True
                        if explicit_ctx and not self._can_estimate_kv():
                            # Tensor mode emits --fit off, so nothing downstream catches it.
                            _unmeasured_ctx_notice = self._unmeasured_context_notice(
                                effective_ctx, cache_type_kv
                            )
                    elif gpus and self._can_estimate_kv() and effective_ctx > 0:
                        # Compute the largest hardware-aware cap from the model's
                        # native context across all usable GPU subsets (for UI
                        # bounds), independent of the currently requested context.
                        native_ctx_for_cap = self._context_length or effective_ctx
                        if native_ctx_for_cap > 0:
                            ranked_for_cap = sorted(
                                gpus,
                                key = lambda g: _gpu_usable(g, _vram_frac - _flat_mtp_reserve),
                                reverse = True,
                            )
                            best_cap = 0
                            _cap_fraction = _vram_frac - _flat_mtp_reserve
                            for n_gpus in range(1, len(ranked_for_cap) + 1):
                                subset = ranked_for_cap[:n_gpus]
                                # Per-GPU-consistent pool budget (fixes mixed
                                # known/unknown totals); pass it as an absolute
                                # budget so the fit and the check below agree.
                                pool_budget = _pool_budget_mib(subset, _cap_fraction)
                                _ms = _subset_model_size(n_gpus)
                                # Compute buffer is replicated per device in a layer
                                # split, so scale the context term by the subset size.
                                _cc_sub = lambda c, n = n_gpus: _cc_bytes(c, n)
                                capped = self._fit_context_to_vram(
                                    native_ctx_for_cap,
                                    pool_budget,
                                    _ms,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = _mtp_reserves_gpu,
                                    mtp_overhead_fn = mtp_overhead_fn,
                                    compute_ctx_bytes_fn = _cc_sub,
                                    budget_frac = 1.0,
                                    pooled = True,
                                    total_mib = None,
                                    ctx_checkpoints = _placement_ctx_checkpoints,
                                )
                                kv = _kv_bytes(capped)
                                footprint_mib = (
                                    _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                ) / (1024 * 1024)
                                if footprint_mib <= pool_budget:
                                    best_cap = max(best_cap, capped)
                            if best_cap > 0:
                                max_available_ctx = best_cap
                                _ctx_cap_fits = True
                                # A forced GPU floor plans subsets this sweep did not hold to it.
                                _vram_fit_ctx = best_cap if _layer_min_gpus <= 1 else None
                            else:
                                # Weights exceed 90% of every GPU subset, so no
                                # context fits. Anchor the UI "safe zone" at the
                                # Auto offload fallback so the slider warns ABOVE
                                # what Auto itself selects: anchoring below it
                                # makes every Auto load in this branch exceed its
                                # own published ceiling, and the chat sheet reads
                                # that as "context exceeds the estimated VRAM
                                # capacity" while advising the user to leave it
                                # on Auto.
                                max_available_ctx = min(_AUTO_OFFLOAD_CTX, native_ctx_for_cap)

                        if explicit_ctx:
                            # Honor the requested context verbatim. If it fits,
                            # pin GPUs and skip --fit; else ship -c <ctx> --fit
                            # on and let llama-server flex -ngl (CPU offload).
                            requested_total = (
                                model_size_fit
                                + _kv_bytes(effective_ctx)
                                + _mtp_bytes(effective_ctx)
                                + _cc_bytes(effective_ctx)
                            )
                            # The compute buffer is replicated on every device in a
                            # layer split; fold it into the per-device reserve so a
                            # multi-GPU pin sizes each card for its own copy. Priced
                            # single-device (the count comes out of this figure), then
                            # re-checked at the split rate, falling back to --fit on.
                            gpu_indices, use_fit = self._select_gpus_split_aware(
                                requested_total,
                                gpus,
                                usable_fraction = _pin_fraction,
                                total_by_idx = total_by_idx,
                                per_device_overhead_bytes = _pipeline_overhead_bytes
                                + _cc_bytes(effective_ctx),
                                min_gpus = _layer_min_gpus,
                                split_extra_bytes = _cc_split_extra(effective_ctx),
                                shared = _shared_gpus,
                            )
                            # No silent shrink: effective_ctx stays == requested_ctx.
                            # use_fit = the pin failed and --fit on will offload; say why.
                            # Else max_available_ctx is the Auto anchor, which does not fit either.
                            # -nkvo keeps the cache on the host, so the priced overflow is not real.
                            if (
                                use_fit
                                and _ctx_cap_fits
                                and not _cuda_ctx_notice
                                and _kv_offload_from_args(extra_args)
                                and not _args_place_tensors_on_cpu(extra_args)
                                and not _env_places_tensors_on_cpu()
                                and not _device_selection_is_cpu(extra_args, os.environ)
                                and not _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
                                and not _env_fixes_gpu_layers()
                                # A trailing user --fit off disables the fitter that would offload.
                                and (
                                    fit_is_enabled_in(extra_args)
                                    or not _extra_args_set_any_flag(extra_args, {"-fit", "--fit"})
                                )
                                # At the native ceiling the true max above it was never searched.
                                and max_available_ctx < native_ctx_for_cap
                            ):
                                _q8_fits = False
                                if planned_flash_attn and (
                                    cache_type_kv or "f16"
                                ).strip().lower() in (
                                    "f16",
                                    "fp16",
                                    "",
                                ):
                                    _q8_kv = self._estimate_kv_cache_bytes(
                                        effective_ctx,
                                        "q8_0",
                                        n_parallel = n_parallel,
                                        swa_full = swa_full,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = _effective_ubatch,
                                        flash_attn = planned_flash_attn,
                                    )
                                    if _q8_kv > 0:
                                        _, _q8_use_fit = self._select_gpus_split_aware(
                                            model_size_fit
                                            + _q8_kv
                                            + _mtp_bytes(effective_ctx)
                                            + _cc_bytes(effective_ctx, cache_type = "q8_0"),
                                            gpus,
                                            usable_fraction = _pin_fraction,
                                            total_by_idx = total_by_idx,
                                            per_device_overhead_bytes = (
                                                _pipeline_overhead_bytes
                                                + _cc_bytes(effective_ctx, cache_type = "q8_0")
                                            ),
                                            min_gpus = _layer_min_gpus,
                                            split_extra_bytes = _cc_split_extra(
                                                effective_ctx, cache_type = "q8_0"
                                            ),
                                        )
                                        _q8_fits = not _q8_use_fit
                                _cuda_ctx_notice = self._cuda_context_overcommit_notice(
                                    effective_ctx,
                                    max_available_ctx,
                                    cache_type_kv,
                                    quantised_ctx_fits = _q8_fits,
                                )
                        else:
                            # Auto context: prefer fewer GPUs, cap to fit. Same
                            # headroom threshold as _select_gpus (#5106). Rank by the
                            # active pin fraction so the order matches the fit budget.
                            pin_fraction = _pin_fraction
                            ranked = sorted(
                                gpus, key = lambda g: _gpu_usable(g, pin_fraction), reverse = True
                            )
                            # Skips _select_gpus, so apply its cap: count only cards
                            # whose usable VRAM clears the per-device layer overhead.
                            _pipeline_overhead_mib = _pipeline_overhead_bytes / (1024 * 1024)
                            _auto_min_gpus = max(
                                1,
                                min(
                                    _layer_min_gpus,
                                    sum(
                                        1
                                        for g in ranked
                                        if _gpu_usable(g, pin_fraction) > _pipeline_overhead_mib
                                    )
                                    or 1,
                                ),
                            )
                            _ctx_before_fit = effective_ctx
                            if _shared_gpus and _auto_min_gpus == 1 and effective_ctx > 0:
                                # A card no other model runs on, when it holds this one at the full
                                # context: sharing a card splits its compute between the models.
                                _alone_mib = (
                                    _subset_model_size(1)
                                    + _kv_bytes(effective_ctx)
                                    + _mtp_bytes(effective_ctx)
                                    + _cc_bytes(effective_ctx, 1)
                                ) / (1024 * 1024)
                                _unshared = next(
                                    (
                                        g
                                        for g in ranked
                                        if g[0] not in _shared_gpus
                                        and _gpu_usable(g, pin_fraction) >= _alone_mib
                                    ),
                                    None,
                                )
                                if _unshared is not None:
                                    ranked.remove(_unshared)
                                    ranked.insert(0, _unshared)
                            for n_gpus in range(_auto_min_gpus, len(ranked) + 1):
                                subset = ranked[:n_gpus]
                                pool_budget = _pool_budget_mib(subset, pin_fraction)
                                _ms = _subset_model_size(n_gpus)
                                # Compute buffer is replicated per device in a layer
                                # split, so scale the context term by the subset size.
                                _cc_sub = lambda c, n = n_gpus: _cc_bytes(c, n)
                                capped = self._fit_context_to_vram(
                                    effective_ctx,
                                    pool_budget,
                                    _ms,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = _mtp_reserves_gpu,
                                    mtp_overhead_fn = mtp_overhead_fn,
                                    compute_ctx_bytes_fn = _cc_sub,
                                    budget_frac = 1.0,
                                    pooled = True,
                                    total_mib = None,
                                    ctx_checkpoints = _placement_ctx_checkpoints,
                                )
                                kv = _kv_bytes(capped)
                                footprint_mib = (
                                    _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                ) / (1024 * 1024)
                                if footprint_mib > pool_budget:
                                    continue
                                # Pooling admits a subset whose per-device reserve
                                # does not fit the smallest card, which then OOMs at
                                # launch; only knowable once `capped` is chosen.
                                _usable_mib = [_gpu_usable(g, pin_fraction) for g in subset]
                                _reserve_at = lambda c, n = n_gpus: (
                                    (_pipeline_overhead_bytes if n > 1 else 0)
                                    + _cc_bytes(c, n) // n
                                )
                                if not self._every_gpu_holds_reserve(
                                    _usable_mib, _reserve_at(capped)
                                ):
                                    # Keep the largest context the smallest card holds
                                    # rather than lose the subset to the 4096 drop below.
                                    # Costs no context: subsets are prefixes of a
                                    # usable-descending ranking and the reserve is the
                                    # same function of context for every n > 1, so any
                                    # later subset clears the gate no higher than this.
                                    capped = self._cap_ctx_to_per_device_reserve(
                                        capped, _usable_mib, _reserve_at
                                    )
                                    if capped <= 0:
                                        continue
                                    # Pooled terms only shrink with the context, so this
                                    # cannot newly fail; re-price rather than lean on that
                                    # across the KV and MTP branches.
                                    kv = _kv_bytes(capped)
                                    footprint_mib = (
                                        _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                    ) / (1024 * 1024)
                                    if footprint_mib > pool_budget:
                                        continue
                                effective_ctx = capped
                                _ctx_capped_for_vram = 0 < capped < _ctx_before_fit
                                gpu_indices = sorted(idx for idx, _ in subset)
                                use_fit = False
                                break
                            else:
                                # No discrete-GPU subset holds the model at a fitted
                                # context. Prefer a useful chat context before
                                # handing placement to --fit on and host offload.
                                effective_ctx = min(_AUTO_OFFLOAD_CTX, effective_ctx)
                                if effective_ctx > 0:
                                    for n_gpus in range(_auto_min_gpus, len(ranked) + 1):
                                        subset = ranked[:n_gpus]
                                        kv = _kv_bytes(effective_ctx)
                                        footprint_mib = (
                                            _subset_model_size(n_gpus)
                                            + kv
                                            + _mtp_bytes(effective_ctx)
                                            + _cc_bytes(effective_ctx, n_gpus)
                                        ) / (1024 * 1024)
                                        if footprint_mib > _pool_budget_mib(subset, pin_fraction):
                                            continue
                                        # Pooling hides the per-device reserve here too.
                                        if not self._every_gpu_holds_reserve(
                                            (_gpu_usable(g, pin_fraction) for g in subset),
                                            (_pipeline_overhead_bytes if n_gpus > 1 else 0)
                                            + _cc_bytes(effective_ctx, n_gpus) // n_gpus,
                                        ):
                                            continue
                                        gpu_indices = sorted(idx for idx, _ in subset)
                                        use_fit = False
                                        break

                    elif gpus:
                        # Can't estimate KV -- file-size-only check; keep the
                        # ceiling at native context (already the default).
                        logger.debug(
                            "Falling back to file-size-only GPU selection",
                            model_size_gb = round(model_size / (1024**3), 2),
                        )
                        # Add the byte-accurate MTP reserve here too when it is
                        # available; otherwise _pin_fraction carries the flat
                        # fallback (the two are mutually exclusive by design).
                        _fs_total = model_size_fit + _mtp_bytes(
                            self._context_length or effective_ctx or 4096
                        )
                        gpu_indices, use_fit = self._select_gpus(
                            _fs_total,
                            gpus,
                            usable_fraction = _pin_fraction,
                            total_by_idx = total_by_idx,
                            per_device_overhead_bytes = _pipeline_overhead_bytes,
                            min_gpus = _layer_min_gpus,
                            shared = _shared_gpus,
                        )
                        if use_fit and not explicit_ctx:
                            # Without KV metadata, llama.cpp owns the fit. Keep the
                            # same useful Auto default as the measured offload path.
                            effective_ctx = (
                                min(_AUTO_OFFLOAD_CTX, effective_ctx)
                                if effective_ctx > 0
                                else _AUTO_OFFLOAD_CTX
                            )

                    elif _apple_budget_mib > 0 and effective_ctx > 0:
                        # No GPU on Metal: the branches above are skipped and the context
                        # stays at native, over-committing unified memory (#5118, #6529).
                        # Not "--fit on" alone: llama.cpp sizes from ggml-metal's free-memory report, blind to Unsloth's own footprint
                        # and the wired limit. See _metal_context_overcommit_message.
                        native_ctx_for_cap = self._context_length or effective_ctx
                        # This arm's floor, which is _FIT_MIN_CTX only while the child's
                        # fitter can come down from it. With --fit off (argv or
                        # LLAMA_ARG_FIT) nothing reduces what we emit, so hand over the
                        # number llama.cpp's own fitter would have reduced to instead.
                        # _metal_zero_ctx_floor draws the same distinction but never sees
                        # these paths: they assign a POSITIVE context, and its first
                        # condition returns 0 for any positive effective_ctx.
                        _metal_floor_ctx = (
                            _FIT_MIN_CTX
                            if fit_is_effectively_on(extra_args, os.environ)
                            else _LLAMA_FIT_MIN_CTX
                        )
                        # Only a ceiling backed by a real KV estimate may refuse; the 4096
                        # fallback below is a guess and would block working loads.
                        _apple_measured_ceiling: Optional[int] = None
                        # The other measured verdict: nothing fits, so there is no ceiling
                        # to name and every explicit request over-commits.
                        _apple_nothing_fits = False
                        # No measurement: the floor below is a guess, so it may neither
                        # refuse nor be published as a ceiling (#9653).
                        _apple_ctx_unmeasured = False
                        # Reserve the flat MTP fraction up front like the discrete
                        # _pin_fraction, so an unsized MTP draft (e.g. Qwen3.6-MTP, #6529)
                        # can't over-commit. No-op when MTP is off; exclusive with the
                        # byte-accurate _mtp_bytes reserve.
                        _apple_fit_budget_mib = int(
                            _apple_budget_mib * max(0.0, 1.0 - _flat_mtp_reserve)
                        )
                        # A CPU-pinned projector is already carried by model_size_fit here:
                        # this arm is reached only with no GPU enumerated, so no device has
                        # a budget of its own and _shared_pool_mmproj charged those bytes.
                        # Adding them again would price the encoder twice.
                        # Exclude pageable input embeddings that the probe found outside Metal.
                        _apple_host_embd = self._metal_demand_paged_embedding_bytes(
                            model_path,
                            extra_args,
                            binary = binary,
                            requested_load_mode = load_mode,
                            supports_load_mode = bool(server_caps.get("supports_load_mode")),
                            settings = _mem_settings,
                            layers_fixed = _caller_owns_budget or _paravirtual_cpu_forced,
                            mtp_may_engage = _mtp_will_engage,
                        )
                        _apple_model_size_fit = model_size_fit - _apple_host_embd
                        if _apple_host_embd:
                            logger.info(
                                "Metal fit: %.2f GB of input embeddings stay in the CPU "
                                "file mapping; charging %.2f GB of weights to unified memory.",
                                _apple_host_embd / (1024**3),
                                _apple_model_size_fit / (1024**3),
                            )

                        def _apple_ctx_fit(
                            target: int,
                            min_ctx: int,
                            budget_mib: Optional[int] = None,
                        ) -> int:
                            return self._fit_context_to_vram(
                                target,
                                _apple_fit_budget_mib if budget_mib is None else budget_mib,
                                _apple_model_size_fit,
                                cache_type_kv,
                                min_ctx = min_ctx,
                                swa_full = swa_full,
                                n_parallel = n_parallel,
                                kv_unified = planned_kv_unified,
                                n_ubatch = _effective_ubatch,
                                flash_attn = planned_flash_attn,
                                mtp_engaged = _mtp_reserves_gpu,
                                mtp_overhead_fn = mtp_overhead_fn,
                                compute_ctx_bytes_fn = _cc_bytes,
                                budget_frac = 1.0,
                                pooled = True,
                                total_mib = None,
                                ctx_checkpoints = _fit_ctx_checkpoints,
                            )

                        def _apple_footprint_mib(ctx: int, ctx_checkpoints: int = 0) -> float:
                            return (
                                _apple_model_size_fit
                                + _kv_bytes(ctx, ctx_checkpoints)
                                + _mtp_bytes(ctx)
                                + _cc_bytes(ctx)
                            ) / (1024 * 1024)

                        if self._can_estimate_kv():
                            cap = _apple_ctx_fit(native_ctx_for_cap, _FIT_MIN_CTX)
                            _cap_footprint_mib = _apple_footprint_mib(cap)
                            # Fit returns the request unchanged when it fits OR weights
                            # exceed budget; only the latter over-commits, so floor to
                            # _FIT_MIN_CTX.
                            if _cap_footprint_mib <= _apple_fit_budget_mib:
                                max_available_ctx = cap
                                _apple_measured_ceiling = cap
                            else:
                                # Two states land here, only one unmeasurable. Weights
                                # alone over budget: the fit returned the request
                                # untouched, priced nothing, and no context rescues it.
                                # Or the weights fit and it is the helper's own
                                # _FIT_MIN_CTX floor that does not -- a floor, not a
                                # measurement, so re-price under it before concluding
                                # nothing was measured. Without that pass a Mac with room
                                # for no tested context skipped the refusal and reached
                                # llama-server, whose own --fit stops at LLAMA.CPP's
                                # fit_params_min_ctx (4096, a different number from ours
                                # since this arm's floor became _FIT_MIN_CTX) and so
                                # cannot rescue it either.
                                _floor_cap = _apple_ctx_fit(native_ctx_for_cap, _FIT_FLOOR_MIN_CTX)
                                # Ask the budget directly rather than inferring the
                                # weights-only state from the two fits disagreeing: both
                                # are bounded by the same target, so on a model whose
                                # native length is at or under the floor they tie for a
                                # reason unrelated to the weights, which read as "priced
                                # nothing" and refused nothing. Reachable at native == 256,
                                # and whenever the GGUF carries no context length and the
                                # request itself is that small.
                                _weights_over_budget = (
                                    _apple_model_size_fit / (1024 * 1024)
                                ) > _apple_fit_budget_mib
                                if _weights_over_budget:
                                    # The fit priced nothing: no context rescues a load
                                    # whose weights alone miss the budget, and that is the
                                    # host-RAM guard's failure. Floor for the UI, but do
                                    # not refuse against a number the fit never vouched for.
                                    max_available_ctx = min(_metal_floor_ctx, native_ctx_for_cap)
                                elif _apple_footprint_mib(_floor_cap) <= _apple_fit_budget_mib:
                                    max_available_ctx = _floor_cap
                                    _apple_measured_ceiling = _floor_cap
                                else:
                                    # It shrank, so the weights fit and even the smallest
                                    # context the search prices does not. A measurement
                                    # too, and the strongest available: nothing fits, so
                                    # there is no number to lower to and every explicit
                                    # request over-commits. Auto is untouched as everywhere
                                    # else here -- it keeps this arm's floor, which is now
                                    # _FIT_MIN_CTX like every other arm, rather than the
                                    # separate 4096 Metal used to hold.
                                    max_available_ctx = min(_metal_floor_ctx, native_ctx_for_cap)
                                    _apple_nothing_fits = True
                        else:
                            # No KV estimate: mirror the discrete file-size-only fallback
                            # and floor to _FIT_MIN_CTX rather than launch at native and
                            # over-commit.
                            max_available_ctx = min(_metal_floor_ctx, native_ctx_for_cap)
                            _apple_ctx_unmeasured = True
                        if not explicit_ctx:
                            effective_ctx = max_available_ctx
                        elif _apple_ctx_unmeasured:
                            # This arm never overruled the request but published the floor:
                            # max_context_length 4,096 for a load launched at 262,144 (#9653).
                            max_available_ctx = max(max_available_ctx, effective_ctx)
                            _unmeasured_ctx_notice = self._unmeasured_context_notice(
                                effective_ctx, cache_type_kv
                            )
                        elif (
                            self._can_estimate_kv()
                            and not _caller_owns_budget
                            and not _paravirtual_cpu_forced
                        ):
                            # Exempt for the reason the other two Metal guards are: a
                            # manual load with a fixed layer count is the user taking the
                            # memory budget over, read off the request so the paravirtual
                            # CPU pin (which rewrites Auto to manual/0) cannot make a plain
                            # Auto load look caller-owned.
                            #
                            # The virtualised device is exempt for the opposite reason:
                            # that pin puts the whole load on CPU behind --device none, so
                            # it allocates no Metal memory and this budget is the wrong
                            # yardstick. Refusing would break Mac VM loads that work today,
                            # and the message would describe a GPU the launch never
                            # touches. Its real risk is host RAM, which
                            # _host_offload_shortfall_message already prices.
                            #
                            # The fit above is sized through the model's native length, so
                            # on its own it caps the ceiling there and refuses every
                            # request past it -- including ones this Mac has the memory
                            # for. Nothing clamps a request to native on the way in (the
                            # Extra Arguments box takes a raw "--ctx-size" and even
                            # suggests "--rope-scaling yarn"), and llama.cpp builds the
                            # context at the full -c, capping only the per-slot value
                            # afterwards, so the request is what gets allocated. Re-price
                            # through it, adopting the answer only when the budget
                            # vouches for it.
                            if _apple_measured_ceiling is not None and (
                                effective_ctx > native_ctx_for_cap
                            ):
                                _extended_ceiling = _apple_ctx_fit(effective_ctx, _FIT_MIN_CTX)
                                if _apple_footprint_mib(_extended_ceiling) > _apple_fit_budget_mib:
                                    # _FIT_MIN_CTX is the helper's floor, not a
                                    # measurement, and the cap above re-prices under it
                                    # for the same reason.
                                    # It bites here whenever native is below the floor: the
                                    # request is above native and the floor above what
                                    # fits, so the probe hands back a non-fitting floor, the
                                    # check discards it, and the refusal keeps naming the
                                    # native-sized cap -- "the largest that fits is 2,048"
                                    # on a 2048-native model whose machine launches 3,072
                                    # when asked for it directly.
                                    _extended_ceiling = _apple_ctx_fit(
                                        effective_ctx, _FIT_FLOOR_MIN_CTX
                                    )
                                if (
                                    _extended_ceiling > _apple_measured_ceiling
                                    and _apple_footprint_mib(_extended_ceiling)
                                    <= _apple_fit_budget_mib
                                ):
                                    _apple_measured_ceiling = _extended_ceiling
                                    # Publish it too: max_available_ctx becomes
                                    # max_context_length, which the UI reads as the
                                    # largest context that fits and both amber warnings
                                    # compare against. Left at native, a load this branch
                                    # just measured and allowed would arrive with a
                                    # warning saying it does not fit. The fit is bounded
                                    # by the request, so the bound rises to the context
                                    # that loaded and no further.
                                    max_available_ctx = max(
                                        max_available_ctx, _apple_measured_ceiling
                                    )
                            # Past free memory macOS swaps; past the wired limit the machine panics.
                            _apple_wired_mib = int(
                                self._apple_metal_wired_ceiling_bytes()
                                // (1024 * 1024)
                                * max(0.0, 1.0 - _flat_mtp_reserve)
                            )
                            if (
                                _apple_wired_mib <= 0
                                or _apple_model_size_fit / (1024 * 1024) > _apple_wired_mib
                            ):
                                _metal_ctx_refusal = self._metal_context_overcommit_message(
                                    effective_ctx,
                                    _apple_measured_ceiling or 0,
                                    cache_type_kv,
                                    nothing_fits = _apple_nothing_fits,
                                )
                            else:
                                # Priced like the fit (SWA checkpoints included) so admission matches the refusal ceiling.
                                _requested_mib = _apple_footprint_mib(
                                    effective_ctx, _fit_ctx_checkpoints
                                )
                                if _requested_mib > _apple_wired_mib:
                                    _wired_ctx = _apple_ctx_fit(
                                        effective_ctx, _FIT_FLOOR_MIN_CTX, _apple_wired_mib
                                    )
                                    _wired_fits = (
                                        _apple_footprint_mib(_wired_ctx, _fit_ctx_checkpoints)
                                        <= _apple_wired_mib
                                    )
                                    _metal_ctx_refusal = self._metal_context_overcommit_message(
                                        effective_ctx,
                                        _wired_ctx if _wired_fits else 0,
                                        cache_type_kv,
                                        nothing_fits = not _wired_fits,
                                        wired_limit = True,
                                    )
                                elif _requested_mib > _apple_fit_budget_mib:
                                    _metal_ctx_warning = self._metal_context_pressure_message(
                                        effective_ctx, _requested_mib, _apple_fit_budget_mib
                                    )
                        # The paravirtual pin loads on CPU, so no Metal budget held anything.
                        _vram_fit_ctx = None if _paravirtual_cpu_forced else _apple_measured_ceiling
                        # The measured ceiling was priced without SWA checkpoints; re-fit with them.
                        if _vram_fit_ctx and (
                            _apple_footprint_mib(_vram_fit_ctx, _fit_ctx_checkpoints)
                            > _apple_fit_budget_mib
                        ):
                            _ckpt_cap = _apple_ctx_fit(_vram_fit_ctx, _FIT_FLOOR_MIN_CTX)
                            _vram_fit_ctx = (
                                _ckpt_cap
                                if _apple_footprint_mib(_ckpt_cap, _fit_ctx_checkpoints)
                                <= _apple_fit_budget_mib
                                else None
                            )
                        # Past the wired limit Metal refuses, so it is the harder ceiling when readable.
                        _fit_wired_mib = (
                            int(
                                self._apple_metal_wired_ceiling_bytes()
                                // (1024 * 1024)
                                * max(0.0, 1.0 - _flat_mtp_reserve)
                            )
                            if _vram_fit_ctx
                            else 0
                        )
                        if (
                            _fit_wired_mib > 0
                            and _apple_footprint_mib(_vram_fit_ctx, _fit_ctx_checkpoints)
                            > _fit_wired_mib
                        ):
                            _wired_cap = _apple_ctx_fit(
                                _vram_fit_ctx, _FIT_FLOOR_MIN_CTX, _fit_wired_mib
                            )
                            _vram_fit_ctx = (
                                _wired_cap
                                if _apple_footprint_mib(_wired_cap, _fit_ctx_checkpoints)
                                <= _fit_wired_mib
                                else None
                            )
                            # Auto launches a fixed -c, so llama.cpp's fitter could only move
                            # layers to CPU; give up context first, down to the Auto floor,
                            # so every layer stays on Metal.
                            if (
                                not explicit_ctx
                                and _vram_fit_ctx
                                and min(_FIT_MIN_CTX, effective_ctx)
                                <= _vram_fit_ctx
                                < effective_ctx
                            ):
                                effective_ctx = max_available_ctx = _vram_fit_ctx
                        # Unmeasured floors still rely on llama.cpp's fitter.
                        _metal_full_offload_pinned = bool(
                            _apple_host_embd and _apple_measured_ceiling is not None
                        )
                        # Weights over the budget launch here too and the fitter places
                        # them, so it must see the same wired limit the refusals do.
                        if not _caller_owns_budget and not _paravirtual_cpu_forced:
                            _metal_fit_target_floor_mib = self._metal_fit_target_floor_mib()

                    # Prefer fewer serving slots on GPU over --fit on offload: when the extra
                    # --parallel slots push the footprint past the pin budget, llama-server
                    # offloads layers to host and decode collapses ~3x (#6718). Retry the fit
                    # at fewer slots, keeping the largest count that stays fully on GPU and the
                    # chosen context. Skips tensor mode / Metal / KV-inestimable paths.
                    if (
                        use_fit
                        and n_parallel > 1
                        and gpus
                        and self._can_estimate_kv()
                        and effective_ctx > 0
                    ):
                        # The PROBE context for the search below, not the context the load
                        # runs at: the re-fit further down awards that, from the final slot
                        # count. So price the search at the fit search floor, never at
                        # _AUTO_OFFLOAD_CTX.
                        #
                        # _AUTO_OFFLOAD_CTX is what Auto settles for once offload really is
                        # unavoidable. Asking this search at that value asks the wrong
                        # question, because this block exists to OVERTURN "offload is
                        # unavoidable" by re-asking at fewer slots -- a higher probe can only
                        # make it fail, and its failure is all-or-nothing: no reduction and
                        # --fit on, so the load keeps the slot count it could not afford and
                        # offloads anyway. Measured on a 12 GiB card, probing at 8192 instead
                        # of the floor moves the hybrid 10,900-11,100 MiB and dense
                        # 9,800-10,500 MiB bands from fully GPU-resident to host offload, the
                        # ~3x decode collapse (#6718) this whole block was written to avoid.
                        #
                        # An explicit context is the exception: it launches verbatim, so the
                        # search has to be priced at the context that will actually run or the
                        # slot count it picks is one the card cannot hold.
                        #
                        # _FIT_MIN_CTX rather than a literal: _largest_ctx below starts its
                        # binary search at the same floor, and the two must agree or the
                        # search can pick a slot count the re-fit then refuses to confirm.
                        _reduce_ctx = (
                            effective_ctx if explicit_ctx else min(effective_ctx, _FIT_MIN_CTX)
                        )
                        # Slot-independent footprint (folded compute buffer and the MTP
                        # reserve swapped out so the helper re-prices both per candidate).
                        _base_footprint = (
                            model_size_fit - _compute_buffer_pipeline + _cc_bytes(_reduce_ctx)
                        )
                        _gi_slots, _uf_slots, _slots = self._slots_that_fit_on_gpu(
                            n_parallel,
                            _reduce_ctx,
                            gpus,
                            total_by_idx,
                            _base_footprint,
                            cache_type_kv,
                            _pin_fraction,
                            _pipeline_overhead_bytes + _cc_bytes(_reduce_ctx),
                            _layer_min_gpus,
                            _effective_ubatch,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            flash_attn = planned_flash_attn,
                            split_extra_for_slots = lambda s: _cc_split_extra(_reduce_ctx, s),
                            ubatch_for_slots = _ubatch_for_slots,
                            mtp_bytes_for_slots = lambda s, ub: _mtp_bytes(_reduce_ctx, s, ub),
                            ctx_checkpoints = _placement_ctx_checkpoints,
                        )
                        if not _uf_slots:
                            _slots_asked = n_parallel
                            gpu_indices, use_fit, n_parallel = _gi_slots, False, _slots
                            # Fewer slots means a lower batch floor, so the launch runs a
                            # smaller micro-batch than the one this was sized at. Re-derive
                            # it, or the recorded _n_ubatch describes a graph never built.
                            _effective_ubatch = _ubatch_for_slots(n_parallel)
                            # Re-fit at the final slot count; the KV and compute closures
                            # read the values just rebound above.

                            def _slots_hold(
                                ctx: int, devices: list[tuple[int, int]]
                            ) -> Optional[list[int]]:
                                """Return the GPU subset if the final slot count fits."""
                                _gi, _uf, _got = self._slots_that_fit_on_gpu(
                                    n_parallel,
                                    ctx,
                                    devices,
                                    total_by_idx,
                                    model_size_fit - _compute_buffer_pipeline + _cc_bytes(ctx),
                                    cache_type_kv,
                                    _pin_fraction,
                                    _pipeline_overhead_bytes + _cc_bytes(ctx),
                                    _layer_min_gpus,
                                    _effective_ubatch,
                                    swa_full = swa_full,
                                    kv_unified = planned_kv_unified,
                                    flash_attn = planned_flash_attn,
                                    split_extra_bytes = _cc_split_extra(ctx),
                                    ubatch_for_slots = _ubatch_for_slots,
                                    mtp_bytes_for_slots = (lambda s, ub, c = ctx: _mtp_bytes(c, s, ub)),
                                    include_requested = True,
                                    exact = True,  # only n_parallel is accepted below
                                    ctx_checkpoints = _placement_ctx_checkpoints,
                                )
                                return _gi if not _uf and _got == n_parallel else None

                            def _largest_ctx(
                                devices: list[tuple[int, int]],
                            ) -> Optional[tuple[int, list[int]]]:
                                """Largest 256-aligned context these cards hold; the
                                footprint grows monotonically with context.

                                Bound here, not by the arm above: the arm that reaches this
                                is not always the one that binds it, and a GGUF with no
                                native length would arrive with the 0 sentinel and search
                                nothing."""
                                _native = self._context_length or _reduce_ctx
                                _lo = max(1, min(_FIT_MIN_CTX, _native) // 256)
                                _hi = _native // 256
                                _best = None
                                while _lo <= _hi:
                                    _mid = (_lo + _hi) // 2
                                    _held = _slots_hold(_mid * 256, devices)
                                    if _held is None:
                                        _hi = _mid - 1
                                    else:
                                        _best = (_mid * 256, _held)
                                        _lo = _mid + 1
                                return _best

                            # Only the cards the reduction chose. Searching every GPU
                            # again buys context by pulling in another device -- the layer
                            # split this path avoids, and one a direct request never makes.
                            _kept = set(_gi_slots or ())
                            _plan_gpus = [g for g in gpus if g[0] in _kept]
                            _refit = _largest_ctx(_plan_gpus)
                            # Explicit contexts are never rewritten. Otherwise the re-fit is
                            # the answer, in both directions: it is the largest context the
                            # FINAL slot count holds on the cards the reduction chose, so
                            # taking anything else launches a context that was never priced.
                            # Not gated on it being an increase over effective_ctx -- that
                            # gate was safe only while the search and effective_ctx shared a
                            # value, and _AUTO_OFFLOAD_CTX above the probe floor is exactly
                            # the case where the re-fit legitimately comes back lower.
                            if _refit is not None and not explicit_ctx:
                                if _refit[0] != effective_ctx:
                                    logger.info(
                                        "Context re-fitted %d -> %d for %d serving slot(s).",
                                        effective_ctx,
                                        _refit[0],
                                        n_parallel,
                                    )
                                effective_ctx, gpu_indices = _refit
                            # After the re-fit, so it names the context that launches
                            # rather than one sized for slots this load dropped.
                            logger.info(
                                "Serving slots reduced %d -> %d to keep the model on GPU "
                                "(avoid --fit offload) at context %d.",
                                _slots_asked,
                                n_parallel,
                                effective_ctx,
                            )
                            # Across every card at the final slot count, matching the
                            # best_cap sweep above; an explicit request that large does
                            # re-select GPUs and load. Not a hardware maximum -- a larger
                            # request can reduce slots further and free KV for more.
                            _ceiling = (
                                _refit if len(_plan_gpus) == len(gpus) else _largest_ctx(gpus)
                            )
                            if _ceiling is not None:
                                # Replaces the anchor rather than raising it. Reaching here
                                # means the offload fallback set max_available_ctx for a plan
                                # this block just superseded, and max() would keep publishing
                                # that plan's number: with _AUTO_OFFLOAD_CTX above what the
                                # final slot count holds, the sheet would advertise a context
                                # the launched plan cannot serve, which is the same ceiling /
                                # selection inversion #9492 removed, pointing the other way.
                                max_available_ctx = _ceiling[0]
                                _vram_fit_ctx = _ceiling[0]

                    # Single-GPU VRAM shortfall vs free less --fit-target (None = unpriceable).
                    _model_size_fit_cb = _compute_buffer_pipeline

                    def _fit_ctx_overridden(
                        child_extras: Optional[Iterable[str]], child_env: Mapping[str, str]
                    ) -> bool:
                        """A --fit-ctx / -fitc (argv, else LLAMA_ARG_FIT_CTX) the child
                        runs with that is not the value _ctx_integrity_flags emits."""
                        _studio_fitc: Optional[int] = None
                        if (
                            use_fit
                            and not (gpu_memory_mode == "manual" and gpu_layers >= 0)
                            and server_caps.get("supports_fit_ctx")
                        ):
                            if requested_ctx > 0 and effective_ctx > 0:
                                _studio_fitc = int(effective_ctx)
                            elif gpu_memory_mode == "manual" and gpu_layers < 0:
                                _studio_fitc = 8192
                        try:
                            _raw = _last_flag_value(child_extras, _FIT_CTX_FLAGS)
                        except ValueError:
                            return True
                        # The emitted flag beats the env; argv after it beats both.
                        if _raw is None and _studio_fitc is None:
                            _raw = child_env.get("LLAMA_ARG_FIT_CTX")
                        if _raw is None or not str(_raw).strip():
                            return False
                        try:
                            return int(str(_raw).strip()) != _studio_fitc
                        except ValueError:
                            return True

                    def _fitter_vram_shortfall() -> Optional[float]:
                        if not (gpus and effective_ctx > 0 and self._can_estimate_kv()):
                            return None
                        # Smart offload may still pin experts to CPU against Unsloth's budget.
                        from core.inference.offload_planner import smart_offload_enabled

                        if smart_offload_enabled(os.environ):
                            return None
                        # A drafter priced with a flat cushion has no draft-KV term here.
                        if _flat_mtp_engages:
                            return None
                        _on = [g for g in gpus if gpu_indices is None or g[0] in gpu_indices]
                        # Per-device placement (common/fit.cpp) can spill despite a pooled fit.
                        if len(_on) != 1:
                            return None
                        _child_extras = (
                            self._strip_device_extra_args(extra_args)
                            if _gpu_ids_own_device_flags
                            else extra_args
                        )
                        _child_env = dict(os.environ)
                        if _gpu_ids_own_device_flags:
                            self._clear_device_placement_env(_child_env)
                        if _pass_through_overrides_placement(
                            _child_extras,
                            _child_env,
                            len(_on),
                            is_vulkan_backend = is_vulkan_backend,
                            gpu_indices = gpu_indices,
                        ):
                            return None
                        # Priced at Unsloth's --parallel; any other pass-through one wins.
                        _np_override = _extra_args_n_parallel(_child_extras, {})
                        if _np_override is not None and _np_override != n_parallel:
                            return None
                        # A later pass-through -c 0 lets llama.cpp pick the context.
                        if parse_ctx_override(_child_extras) == 0:
                            return None
                        # A --fit-ctx Studio did not emit moves llama.cpp's fit.
                        if _fit_ctx_overridden(_child_extras, _child_env):
                            return None
                        # A GPU projector Studio never sized; Vision off keeps only audio-only.
                        _env_mmproj = _child_env.get("LLAMA_ARG_MMPROJ")
                        # The emitted --mmproj replaces an inherited path (argv after env).
                        if effective_is_vision:
                            _env_mmproj = None
                        # --mmproj-device / MTMD_BACKEND_DEVICE re-enable the GPU projector.
                        _mmproj_dev = _extra_args_device(
                            _child_extras, {"--mmproj-device", "-mmdev"}
                        )
                        if _mmproj_dev is None:
                            _mmproj_dev = _child_env.get("MTMD_BACKEND_DEVICE")
                        _mmproj_dev_gpu = bool(
                            _mmproj_dev is not None and str(_mmproj_dev).strip().lower() != "none"
                        )
                        if _resolved_mmproj_offload(_child_extras, _child_env) is False:
                            if _mmproj_dev_gpu:
                                return None
                        elif disable_vision:
                            if _mmproj_env_is_audio_only(_env_mmproj):
                                return None
                        elif (
                            _env_mmproj
                            or _child_env.get("LLAMA_ARG_MMPROJ_URL")
                            or (
                                not launch_mmproj_path
                                and extra_args_mmproj_auto(_child_extras, _child_env)
                            )
                        ):
                            return None
                        # --fit-target precedence: argv, Studio's emitted value, env, default.
                        _supports_fit_target = bool(server_caps.get("supports_fit_target"))
                        _auto_fit = gpu_memory_mode == "manual" and gpu_layers < 0
                        _managed_mib = self._fit_target_margin_mib(
                            auto_fit = _auto_fit,
                            fit_target_delta_mib = _fit_target_delta_mib,
                            supports_fit_target = _supports_fit_target,
                        )
                        # A per-device list reads as its largest entry (the larger shortfall).
                        _margin_mib = fit_target_margin_in(_child_extras or (), None)
                        if _margin_mib is None:
                            _emits = _supports_fit_target and (
                                _auto_fit or _managed_mib != _LLAMA_FIT_TARGET_DEFAULT_MIB
                            )
                            _env_mib = None if _emits else fit_target_margin_in((), _child_env)
                            # An inherited 0 is a valid margin, not "unset".
                            _margin_mib = _managed_mib if _env_mib is None else _env_mib
                        # A lazily read table stays mmap'd on the host, never on the GPU.
                        _lazy_host = self._lazy_read_host_bytes(
                            model_path,
                            caps = server_caps,
                            extra_args = _child_extras,
                            env = _child_env,
                            auto_resolves_off = self._lazy_auto_resolves_off(
                                gpu_indices = gpu_indices,
                                detected_gpus = _detected_gpus,
                                shared_gpu_ids = _shared_gpu_ids,
                                is_vulkan_backend = is_vulkan_backend,
                            ),
                        )
                        _need = (
                            model_size_fit
                            - _lazy_host
                            - _model_size_fit_cb
                            + _compute_buffer_pipeline
                            + (
                                _kv_bytes(effective_ctx)
                                if _kv_offload_from_args(_child_extras, _child_env)
                                else 0
                            )
                            + (
                                _mtp_bytes(effective_ctx, n_parallel, _effective_ubatch)
                                if _mtp_will_engage
                                else 0
                            )
                            + _cc_bytes(effective_ctx, len(_on))
                        )
                        _free = sum(max(0.0, free - _margin_mib) for _idx, free in _on)
                        return _need - _free * 1024 * 1024

                    def _moe_resident_share_now() -> Optional[float]:
                        if self.n_moe_layers <= 0:
                            return None
                        try:
                            _short = _fitter_vram_shortfall()
                            if _short is None:
                                return None
                            _scan = self._gguf_tensor_scan(model_path) or (None, {}, 0)
                            return _moe_resident_expert_share(_short, _scan[2])
                        except Exception as e:
                            logger.debug(f"MoE resident share unpriceable ({e})")
                            return None

                    # MoE experts in host RAM: a larger micro-batch amortises the expert
                    # copies of prompt processing (see _MOE_SPILL_N_UBATCH). The placement
                    # verdict above was priced at the default micro-batch, so a model that
                    # fits keeps today's launch. This runs before every term that reads the
                    # micro-batch (MTP reserve, KV, compute buffer, spill planner, load mode),
                    # so the larger buffer is priced: the spill planner keeps fewer experts on
                    # the GPU, and llama.cpp's fitter measures a --fit on launch at the emitted
                    # value. Manual pinned layers (--fit off) are the user's placement, where an
                    # unpriced larger buffer could OOM, so they are left alone. A GPU-only -ot
                    # does not count for the same reason. A user --device none / cpu runs on
                    # the CPU with nothing to stream, like the CPU fallback.
                    _moe_experts_on_host = bool(
                        not (gpu_memory_mode == "manual" and gpu_layers >= 0)
                        and not intent.cpu_fallback
                        and not _device_selection_is_cpu(extra_args, os.environ)
                        and (use_fit or _expert_spill_places_tensors_on_cpu(extra_args, os.environ))
                    )
                    _moe_spill_discrete_gpu = bool(
                        _moe_experts_on_host
                        and self._discrete_gpu_for_expert_spill(
                            gpu_indices, _detected_gpus, _shared_gpu_ids
                        )
                    )
                    _spill_n_batch, _spill_n_ubatch = _moe_spill_batch_ubatch(
                        n_batch,
                        n_ubatch,
                        n_moe_layers = self.n_moe_layers,
                        experts_on_host = _moe_experts_on_host,
                        discrete_gpu = _moe_spill_discrete_gpu,
                        user_named_batch = _user_named_batch,
                    )
                    _moe_ub_switch = _moe_switch(_MOE_SPILL_UBATCH_ENV)
                    if _spill_n_ubatch != n_ubatch and _moe_ub_switch != "off":
                        _share_before = (
                            _moe_resident_share_now() if _moe_ub_switch == "auto" else None
                        )
                        _pre_raise = (
                            n_batch,
                            n_ubatch,
                            _effective_ubatch,
                            _compute_buffer_pipeline,
                            _compute_buffer_tensor,
                        )
                        n_batch, n_ubatch = _spill_n_batch, _spill_n_ubatch
                        _effective_ubatch = _ubatch_for_slots(n_parallel)
                        # Same calls and fallbacks as above, at the new micro-batch.
                        _compute_buffer_pipeline = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = False,
                        ) or (self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024)
                        _compute_buffer_tensor = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = True,
                        ) or (self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024)
                        _share_after = (
                            _moe_resident_share_now()
                            if _moe_ub_switch == "auto" and _share_before is not None
                            else None
                        )
                        if _moe_spill_raise_keeps_experts(
                            _share_before, _share_after, switch = _moe_ub_switch
                        ):
                            logger.info(
                                "MoE experts are offloaded to system RAM; raising "
                                "--ubatch-size %s -> %s for faster prompt processing.",
                                (
                                    _pre_raise[1]
                                    if _pre_raise[1] is not None
                                    else _DEFAULT_LLAMA_N_UBATCH
                                ),
                                _spill_n_ubatch,
                            )
                            self._moe_spill_batch_restore = (_pre_raise[0], _pre_raise[1])
                        else:
                            (
                                n_batch,
                                n_ubatch,
                                _effective_ubatch,
                                _compute_buffer_pipeline,
                                _compute_buffer_tensor,
                            ) = _pre_raise
                            logger.info(
                                "MoE experts: keeping the default --ubatch-size; %s would "
                                "move the GPU's resident share of the routed experts "
                                "%.0f%% -> %.0f%%.",
                                _spill_n_ubatch,
                                100 * (_share_before or 0.0),
                                100 * (_share_after or 0.0),
                            )

                    _moe_cache_candidate = self._moe_cache_auto_eligible(
                        caps = server_caps,
                        gpu_memory_mode = gpu_memory_mode,
                        use_fit = use_fit,
                        experts_on_host = _moe_experts_on_host,
                        discrete_gpu = _moe_spill_discrete_gpu,
                        gpu_indices = gpu_indices,
                        detected_gpus = _detected_gpus,
                        is_vulkan_backend = is_vulkan_backend,
                        tensor_parallel = tensor_parallel,
                        extra_args = extra_args,
                        env = os.environ,
                    )

                    # Pass the final slot and micro-batch values instead of the defaults
                    # captured before slot reduction.
                    _mtp_reserve_bytes = (
                        _mtp_bytes(effective_ctx, n_parallel, _effective_ubatch)
                        if _mtp_will_engage
                        else 0
                    )
                    if _mtp_will_engage:
                        _mtp_note = self._mtp_reserve_note(
                            _mtp_reserve_bytes,
                            n_ctx = effective_ctx,
                            n_parallel = n_parallel or 1,
                            n_ubatch = _effective_ubatch,
                            n_max = _mtp_eff_n_max,
                            target_rollback = _target_rollback,
                            flat_fallback = mtp_overhead_fn is None,
                            reprice = lambda slots, ub: _mtp_bytes(effective_ctx, slots, ub),
                        )
                    else:
                        _mtp_note = ""

                    if effective_ctx < original_ctx:
                        kv_est = _kv_bytes(effective_ctx)
                        logger.info(
                            f"Context auto-reduced: {original_ctx} -> {effective_ctx} "
                            f"(model: {model_size / (1024**3):.1f} GB, "
                            f"est. KV cache: {kv_est / (1024**3):.1f} GB, "
                            f"{_mtp_note}".rstrip(", ")
                            + ")"
                        )

                    kv_cache_bytes = _kv_bytes(effective_ctx)
                    # The Manual launch branch below emits --fit off; decide it before the
                    # decision line so the log matches the argv (#10821).
                    if gpu_memory_mode == "manual" and gpu_layers >= 0:
                        use_fit = False
                    # Everything the spill planner needs, snapshotted as plain ints
                    # where it is already evaluated.
                    _spill_inputs = {
                        "model_size": model_size,
                        "kv_cache_bytes": kv_cache_bytes,
                        "gpus": list(gpus or ()),
                        # The budget the fit above tested, per device: _gpu_usable
                        # applies the VRAM-budget fraction and the reserve floor.
                        # Raw free_mib is 3.5 GiB per card larger at 0.80 on a
                        # 24 GiB card, so planning on it under-spills and then pins
                        # the result with --fit off.
                        #
                        # _pin_fraction, not the bare _vram_frac: when the drafter
                        # engages but its KV cannot be sized, the placement paths
                        # spend _MTP_VRAM_RESERVE_FRAC less of every card and
                        # _mtp_reserve_bytes carries no replacement for the unsized
                        # part, so crediting the full fraction here would hand the
                        # planner 5% of VRAM the fit deliberately kept free. Equal to
                        # _vram_frac on every other load (_flat_mtp_reserve is 0).
                        "gpu_usable_mib": {
                            _idx: max(0.0, _gpu_usable((_idx, _free), _pin_fraction))
                            for _idx, _free in (gpus or ())
                        },
                        "gpu_indices": gpu_indices,
                        "soft_overhead": _soft_overhead,
                        # GPU-resident bytes in model_size_fit the planner cannot
                        # see: it rebuilds from the target GGUF's tensor table, so
                        # the projector and MTP reserve are absent from its deficit.
                        "extra_gpu_bytes": (mmproj_size + _shared_pool_mmproj + _mtp_reserve_bytes),
                        # The compute buffer, charged on the same terms as the fit
                        # that produced the use_fit verdict: model_size_fit carries
                        # the flat _compute_buffer_pipeline IN ADDITION to
                        # _soft_overhead, and the fit adds _cc_bytes(ctx) per device
                        # on top, so there is no double charge. Omitting them is not
                        # conservative -- _cc_bytes is context-linear (tens of GiB
                        # on deepseek4 at 1M, ~2.4 GiB on a quantised cache at 128K)
                        # and the plan is pinned with --fit off, so the shortfall is
                        # a graph_reserve OOM with nothing left to catch it.
                        "compute_buffer_flat": int(_compute_buffer_pipeline),
                        "ctx_compute_per_device": (
                            _cc_bytes(effective_ctx, max(1, len(gpus or ())))
                            // max(1, len(gpus or ()))
                        ),
                        # An engaging draft turns the trailing nextn/MTP blocks from
                        # skipped bytes into resident ones in the TARGET's load, so
                        # charge them back. Any engaged draft counts: load_mtp comes
                        # from the spec type (common/common.cpp:1713), and a
                        # separate drafter that is itself an MTP head selects that
                        # same type (common/speculative.cpp:2257-2261).
                        # Over-charging only spills more.
                        "mtp_will_engage": bool(_mtp_will_engage),
                        # A SEPARATE drafter that stays on the GPU: a second model,
                        # whose weights and KV llama.cpp distributes like any other.
                        # None once the drafter is CPU-offloaded (_mtp_draft_for_budget
                        # is nulled there), which is the case that really is off-GPU.
                        "separate_draft_on_gpu": bool(_mtp_will_engage and _mtp_draft_for_budget),
                        # The planner reads the real tensor table, not a bucket
                        # total, so it can spill the MINIMUM set of blocks.
                        "model_path": model_path,
                        "n_ctx": effective_ctx,
                        "n_threads": n_threads,
                        # The slot count the KV and recurrent state were PRICED at.
                        # A pass-through --parallel is appended after Unsloth's own
                        # and wins, and both caches scale with it.
                        "n_parallel": int(n_parallel or 1),
                        # Relative cache size per layer: only the ratios are used
                        # (the planner scales them to kv_cache_bytes), but they say
                        # WHERE the big caches land, which the total cannot. Same
                        # launch settings kv_cache_bytes was priced at, so the
                        # vector describes the same cache geometry.
                        "kv_layer_weights": self._kv_layer_weights(
                            effective_ctx,
                            swa_full = swa_full,
                            n_parallel = n_parallel,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            flash_attn = planned_flash_attn,
                            cache_type_kv = cache_type_kv,
                        ),
                        # An iGPU or APU reports host RAM as VRAM: crediting it and
                        # then "spilling" into that pool counts one memory twice.
                        "shared_gpu_ids": set(_shared_gpu_ids or ()),
                    }
                    mmproj_note = (
                        f"mmproj: {mmproj_size / (1024**3):.1f} GB, " if mmproj_size else ""
                    )
                    logger.info(
                        f"GGUF size: {gguf_size / (1024**3):.1f} GB, "
                        f"{mmproj_note}"
                        f"est. KV cache: {kv_cache_bytes / (1024**3):.1f} GB, "
                        f"{_mtp_note}"
                        f"context: {effective_ctx}, "
                        # --fit flag state, not "does it fit": off means this subset provably fits.
                        # Manual modes empty gpus on purpose; say so, not a failed probe.
                        f"GPUs free: {gpus}"
                        f"{' (manual mode)' if not gpus and gpu_memory_mode == 'manual' else ''}, "
                        f"selected: {gpu_indices}, --fit: {'on' if use_fit else 'off'}"
                    )
                    _on = list(gpu_indices or [idx for idx, _ in gpus])
                    if _on:
                        self._pending_plan_mib = _gpu_plan_mib(
                            gguf_size + mmproj_size + kv_cache_bytes, _on, gpus
                        )
                    if (
                        not gpus
                        and not _detected_gpus
                        and not gpu_ids
                        and not _arch_gate_forced_cpu
                        and not _metal_capable_host()
                    ):
                        # Claims no placement: llama.cpp enumerates devices itself.
                        # _detected_gpus, not gpus: Manual modes empty gpus on purpose.
                        # gpu_ids is pinned for the child below; the arch gate warns better.
                        logger.warning(
                            "Unsloth could not enumerate any GPU, so this load was planned "
                            "without device information: %s",
                            LlamaCppBackend._explain_empty_gpu_probe(binary),
                        )
                    # The planner ran to completion over a real device list and
                    # left --fit on, i.e. it could not prove the model fits. THAT
                    # is a partial-placement verdict. Set last, so any early
                    # return or raise above leaves it False.
                    _placement_verdict_partial = bool(_detected_gpus) and bool(use_fit)

                    # Load mode, from the same numbers the placement just used: mmap
                    # exists to demand-page a footprint the machine cannot hold, so a
                    # load that fits wants "none" and anything else keeps "auto". See
                    # _fit_derived_load_mode. `gpus`, not `_detected_gpus`: the arch gate
                    # empties that one on "--device none", and crediting VRAM the child
                    # never reaches would claim a fit that has to come from RAM.
                    _fit_devices = max(1, len(gpu_indices or ()) or len(gpus))
                    # A CPU-pinned drafter is out of the VRAM budget above but still
                    # resident in host RAM, so the footprint carries it. None means it
                    # cannot be priced -> abstain, like the other unreadable terms.
                    _cpu_draft_fit_bytes = self._cpu_resident_draft_bytes(
                        effective_ctx,
                        drafter_path = _cpu_draft_path,
                        draft_cache_type_k = _mtp_draft_ck,
                        draft_cache_type_v = _mtp_draft_cv,
                        n_parallel = n_parallel,
                        swa_full = swa_full,
                        kv_unified = planned_kv_unified,
                        n_ubatch = _effective_ubatch,
                        flash_attn = planned_flash_attn,
                    )
                    # The extras and env as the child really gets them: a gpu_ids pin
                    # drops the device flags AND their env twins, so classifying on
                    # either would void the VRAM term for a "--device none" the launch
                    # strips. Unpinned, the child inherits LLAMA_ARG_DEVICE verbatim.
                    _fit_extras = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args
                    )
                    _fit_env = dict(os.environ)
                    if gpu_memory_mode == "manual":
                        self._clear_manual_placement_env(_fit_env)
                    if _gpu_ids_own_device_flags:
                        self._clear_device_placement_env(_fit_env)
                    # What the placement branch below will emit, so the effective
                    # (last-wins) fitter state is readable before the final argv exists.
                    # Every reachable branch is fixed by use_fit.
                    _fitter_runs = fit_is_effectively_on(
                        ["--fit", "on" if use_fit else "off", *(_fit_extras or [])],
                        _fit_env,
                    )
                    # _ctx_integrity_flags emits --fit-target only while use_fit, so a
                    # fitter the EXTRAS turned back on keeps llama.cpp's own 1024 MiB
                    # default; a pass-through --fit-target wins over either (below).
                    # A drafter the extras SPLIT between card and host with a fixed
                    # positive -ngld: _cpu_draft_path is None there (that predicate is
                    # the whole-drafter question), so mtp_bytes carries the drafter WHOLE
                    # while llama.cpp keeps the layers past the count, and their KV, in
                    # host RAM. Nothing here says which slice stays on the card, so this
                    # is the "engaged but unsized" case the fit abstains on.
                    _draft_split_across_host = bool(
                        _mtp_draft_for_budget
                        and _draft_is_split_across_host(
                            _fit_extras,
                            _fit_env,
                            n_draft_layers = getattr(
                                self._draft_backend_for(_mtp_draft_for_budget),
                                "_n_layers",
                                None,
                            ),
                        )
                    )
                    _fit_margin_mib = (
                        max(
                            self._fit_target_margin_mib(
                                auto_fit = (gpu_memory_mode == "manual" and gpu_layers < 0),
                                fit_target_delta_mib = _fit_target_delta_mib if use_fit else 0.0,
                                supports_fit_target = use_fit
                                and bool(server_caps.get("supports_fit_target")),
                                floor_mib = _metal_fit_target_floor_mib,
                            ),
                            fit_target_margin_in(_fit_extras, _fit_env) or 0.0,
                        )
                        if _fitter_runs
                        else 0.0
                    )
                    # A projector Unsloth never resolved but the child loads anyway:
                    # LLAMA_ARG_MMPROJ / _URL are read before argv and only the vision
                    # switch and the paravirtual pin clear them, while model_size carries
                    # none of those bytes. A URL names a download that has not happened
                    # (and outranks even the --mmproj this launch emits), and an
                    # unreadable path cannot be sized, so both abstain.
                    _fit_env_mmproj_scrubbed = bool(
                        _pv_mmproj_unpinnable or _paravirtual_cpu_forced
                    )
                    _fit_env_mmproj = (
                        ""
                        if (
                            launch_mmproj_path
                            or _fit_env_mmproj_scrubbed
                            or (
                                disable_vision
                                # The scrub's own audio-only reprieve: that projector
                                # survives the switch, so it is loaded.
                                and not _mmproj_env_is_audio_only(_fit_env.get("LLAMA_ARG_MMPROJ"))
                            )
                        )
                        else (_fit_env.get("LLAMA_ARG_MMPROJ") or "").strip()
                    )
                    _fit_env_mmproj_url = (
                        ""
                        if (_fit_env_mmproj_scrubbed or disable_vision)
                        else (_fit_env.get("LLAMA_ARG_MMPROJ_URL") or "").strip()
                    )
                    _fit_env_mmproj_bytes = self._mmproj_vram_bytes(_fit_env_mmproj or None)
                    _fit_env_mmproj_unsized = bool(_fit_env_mmproj_url) or (
                        bool(_fit_env_mmproj) and _fit_env_mmproj_bytes <= 0
                    )
                    # Host-resident when the resolved offload says CPU, the same split
                    # the resolved projector gets.
                    _fit_env_mmproj_on_host = (
                        _resolved_mmproj_offload(_fit_extras, _fit_env) is False
                    )
                    _fit_model_size = model_size
                    if _fit_env_mmproj_unsized:
                        # Same abstain the other unreadable terms take.
                        _fit_model_size = None
                    elif _fit_model_size and not _fit_env_mmproj_on_host:
                        _fit_model_size += _fit_env_mmproj_bytes
                    _fit_soft_overhead = _soft_overhead + self._inherited_mmproj_soft_overhead(
                        _fit_env_mmproj_bytes,
                        on_host = _fit_env_mmproj_on_host,
                    )
                    # The same projector, handed to the spill planner. Its
                    # extra_gpu_bytes carries the RESOLVED one only, so without this
                    # an inherited LLAMA_ARG_MMPROJ is VRAM no term in the plan has
                    # paid for -- and a plan is pinned with --fit off, so llama.cpp's
                    # fitter never gets to absorb it. Weights plus the runtime
                    # allowance, exactly what _fit_model_size and _fit_soft_overhead
                    # just charged, and the same abstain when it cannot be sized.
                    _spill_inputs["env_mmproj_unsized"] = _fit_env_mmproj_unsized
                    _spill_inputs["env_mmproj_bytes"] = (
                        0
                        if _fit_env_mmproj_on_host
                        else _fit_env_mmproj_bytes
                        + self._inherited_mmproj_soft_overhead(_fit_env_mmproj_bytes, on_host = False)
                    )
                    # Derive the host-only share from the same estimator as the total.
                    _effective_ctx_checkpoints = _ctx_checkpoints_for_final_slots()
                    _ckpt_host_bytes = max(
                        0,
                        _kv_bytes(effective_ctx, _effective_ctx_checkpoints)
                        - _kv_bytes(effective_ctx, 0),
                    )
                    # A lazily read table stays mmap'd under every load mode: not pinned RAM.
                    _fit_lazy_bytes = self._lazy_read_host_bytes(
                        model_path,
                        caps = server_caps,
                        extra_args = _fit_extras,
                        env = _fit_env,
                        auto_resolves_off = self._lazy_auto_resolves_off(
                            gpu_indices = gpu_indices,
                            detected_gpus = _detected_gpus,
                            shared_gpu_ids = _shared_gpu_ids,
                            is_vulkan_backend = is_vulkan_backend,
                        ),
                    )
                    _fit_load_mode_kwargs = dict(
                        model_size = _fit_model_size,
                        mmproj_pinned_bytes = _mmproj_pinned_bytes
                        + (_fit_env_mmproj_bytes if _fit_env_mmproj_on_host else 0),
                        # Checkpoints are split into host_only_bytes below.
                        kv_cache_bytes = _kv_bytes(effective_ctx, 0),
                        kv_sized = self._can_estimate_kv(),
                        # mtp_overhead_fn bound the pre-raise micro-batch; charge the raised one.
                        mtp_bytes = _mtp_bytes(
                            effective_ctx,
                            ubatch = (
                                _effective_ubatch
                                if self._moe_spill_batch_restore is not None
                                else None
                            ),
                        ),
                        # _flat_mtp_engages whole: its other arm, _mtp_kv_unsized,
                        # prices weights but NO draft KV, and the placement covers that
                        # gap with a flat cushion this footprint has no term for --
                        # the "engaged but unsized" case the predicate abstains on.
                        mtp_unsized = bool(
                            _flat_mtp_engages
                            or _cpu_draft_fit_bytes is None
                            or _draft_split_across_host
                        ),
                        # Neither a CPU drafter nor checkpoints can be covered by free VRAM.
                        host_only_bytes = (_cpu_draft_fit_bytes or 0) + _ckpt_host_bytes,
                        # One lump on the layer path, where the graph buffer is
                        # allocated once. A tensor split replicates it on every selected
                        # device, so pricing one LAYER-mode buffer there understates a
                        # multi-GPU load by GiBs. pipeline_overhead_bytes below is a flat
                        # per-device constant and cannot cover it.
                        compute_buffer_flat = (
                            _fit_devices * _compute_buffer_tensor
                            if _tp_planned
                            else _compute_buffer_pipeline
                        ),
                        compute_buffer_ctx = _cc_bytes(effective_ctx, _fit_devices),
                        # Ungated, like _subset_model_size: the per-device CUDA context
                        # and scratch are there whether or not llama.cpp keeps the
                        # pipeline (_cc_bytes is the term that gates on it).
                        pipeline_overhead_bytes = (
                            max(0, _fit_devices - 1) * _pipeline_overhead_bytes
                        ),
                        soft_overhead = _fit_soft_overhead,
                        gpus = gpus,
                        gpu_indices = gpu_indices,
                        shared_gpu_ids = _shared_gpu_ids,
                        # A running fitter keeps --fit-target free per device, so that
                        # margin is not VRAM this fit may spend. Zero only when the
                        # fitter is provably off; see _fitter_runs above.
                        fit_margin_mib = _fit_margin_mib,
                        is_vulkan_backend = is_vulkan_backend,
                        # Never stripped, and appended last, so they beat this
                        # launch's own -ngl.
                        extra_args = _fit_extras,
                        env = _fit_env,
                        lazy_read_bytes = _fit_lazy_bytes,
                    )
                    _fit_load_mode = self._fit_derived_load_mode(**_fit_load_mode_kwargs)
                    if _moe_cache_candidate:
                        _moe_cache_switch = _moe_switch(_MOE_CACHE_ENV)
                        _moe_share = (
                            _moe_resident_share_now() if _moe_cache_switch == "auto" else None
                        )
                        if not _moe_cache_share_admits(
                            _moe_share, drafter = _mtp_will_engage, switch = _moe_cache_switch
                        ):
                            _moe_cache_candidate = False
                            if _moe_cache_switch == "off":
                                _why = f"{_MOE_CACHE_ENV} is off"
                            elif _moe_share is None:
                                _why = "the share of routed experts left on the GPU is unknown"
                            elif _mtp_will_engage and not _MOE_CACHE_WITH_DRAFTER:
                                _why = "a drafter is engaged"
                            else:
                                _why = (
                                    f"an estimated {100 * _moe_share:.0f}% of the routed "
                                    "experts stay on the GPU"
                                )
                            logger.info(
                                "MoE cache: %s; keeping the --fit split without "
                                "--moe-cache-mib.",
                                _why,
                            )
                    if _moe_cache_candidate:
                        # The cache frees no host RAM: every routed expert plus the prompt
                        # cache must fit pinned (moved out of model_size, not counted twice).
                        _moe_expert_bytes = (self._gguf_tensor_scan(model_path) or (None, {}, 0))[2]
                        _moe_prompt_cache_bytes = _prompt_cache_host_bytes(
                            cache_ram,
                            _fit_extras,
                            _fit_env,
                            supports_cache_ram = bool(server_caps.get("supports_cache_ram")),
                        )
                        # An unknown size (unsized inherited projector) admits nothing.
                        _moe_cache_ram_ok = bool(
                            _fit_load_mode_kwargs["model_size"]
                            and _moe_expert_bytes > 0
                            and _moe_prompt_cache_bytes is not None
                            and self._fit_derived_load_mode(
                                **{
                                    **_fit_load_mode_kwargs,
                                    "model_size": max(
                                        1, _fit_load_mode_kwargs["model_size"] - _moe_expert_bytes
                                    ),
                                    "host_only_bytes": _fit_load_mode_kwargs["host_only_bytes"]
                                    + _moe_expert_bytes
                                    + _moe_prompt_cache_bytes,
                                    "announce": False,
                                }
                            )
                            == self._FIT_LOAD_MODE
                        )
                        if not _moe_cache_ram_ok:
                            logger.info(
                                "MoE cache: system RAM cannot hold the %.1f GB of routed "
                                "experts pinned; launching without --moe-cache-mib.",
                                _moe_expert_bytes / (1024**3),
                            )
                except Exception as e:
                    logger.warning(f"GPU selection failed ({e}), using --fit on")
                    gpu_indices, use_fit = None, True
                    _metal_full_offload_pinned = False
                    # Not a verdict: this arm never priced anything. Explicit, so
                    # the flag cannot survive from a previous load.
                    _placement_verdict_partial = False
                    # Same reasoning: an unpriced load keeps llama.cpp's default.
                    _fit_load_mode = None
                    _moe_cache_candidate = _moe_cache_ram_ok = False
                    # And a half-built snapshot is not a price either: the throw can
                    # land after the dict exists but before the later terms are added
                    # to it, which would plan against a footprint missing them.
                    _spill_inputs = None
                    # Nor is a ceiling the fallback's own fitter may spill past.
                    _vram_fit_ctx = None
                    tp_tensor_split = None
                    effective_ctx = requested_ctx  # fall back to original
                    # Nor is a slot cut for the drafter: requested count, no drafter, as before.
                    if _mtp_slots_asked is not None:
                        n_parallel = _mtp_slots_asked
                        _mtp_slots_asked = _mtp_slots_target_ctx = None
                        _effective_ubatch = _ubatch_for_slots(n_parallel)
                        _compute_buffer_pipeline = _layer_compute_buffer(n_parallel)
                        _spec_dropped_no_vram = True
                        _mtp_will_engage = False
                        mtp_overhead_fn = None
                        _mtp_kv_unsized = False

                # A hand-set context unified memory cannot hold. Raised here so the handler
                # above cannot turn it back into the launch it refuses, and ahead of the
                # Vulkan and APU checks, which describe hardware this branch has ruled out.
                if _metal_ctx_refusal:
                    raise RuntimeError(_metal_ctx_refusal)
                self._record_load_warning(_metal_ctx_warning)

                if (
                    intent.refuse_partial_gpu_fit
                    and (_placement_verdict_partial or _ctx_capped_for_vram)
                    and gpu_memory_mode != "manual"
                ):
                    capped_only = _ctx_capped_for_vram and not _placement_verdict_partial
                    asked_ctx = _ctx_before_fit if capped_only else effective_ctx
                    need_mib = (gguf_size + mmproj_size + _kv_bytes(asked_ctx)) // (1024 * 1024)
                    free_mib = sum(free for _, free in gpus)
                    need_gb, free_gb = need_mib / 1024, free_mib / 1024
                    raise GpuMemoryShortError(
                        _gpu_short_message(need_gb, free_gb, asked_ctx, effective_ctx, capped_only),
                        capped = capped_only,
                        short_mib = max(0, need_mib - free_mib),
                        gpu_indices = tuple(idx for idx, _ in gpus),
                    )

                # An unenumerated explicit Vulkan ordinal can't be pinned; fail loudly
                # instead of fitting onto an unselected device. Clear the raw selection
                # the early state-publish recorded so it never leaks into gpu_ids (#7239).
                if _vulkan_explicit_unmatched:
                    self._gpu_ids = None
                    self._requested_gpu_ids = None
                    raise ValueError(
                        f"Requested Vulkan GPU ordinal(s) {_vulkan_requested_ids} not "
                        f"present. Available Vulkan devices: {_vulkan_available_ordinals}."
                    )

                # GPU picker: when no narrower subset was chosen (manual, or
                # a failed/file-size selection), pin the whole picked set so the
                # model can't spill onto an unpicked GPU.
                if gpu_ids:
                    _picked_order = [int(i) for i in gpu_ids]
                    if gpu_indices is None:
                        gpu_indices = _picked_order
                    else:
                        # A fit narrowed the pool. Only the survivors remain, but their
                        # relative order is still the one the user dragged them into.
                        _kept = {int(i) for i in gpu_indices}
                        _before_reorder = [int(i) for i in gpu_indices]
                        _reordered = [i for i in _picked_order if i in _kept] + [
                            int(i) for i in gpu_indices if int(i) not in set(_picked_order)
                        ]
                        # _plan_tensor_parallel returned the shares positional over the
                        # ascending gpu_indices, so reordering the devices without moving
                        # the shares hands the roomier card's share to the smaller one --
                        # exactly when a weighted split was needed. Veto rather than
                        # half-apply, as _repoint_emitted_tensor_split does for the mask.
                        if tp_tensor_split and len(tp_tensor_split) == len(_before_reorder):
                            _share_by_id = dict(zip(_before_reorder, tp_tensor_split))
                            if all(i in _share_by_id for i in _reordered):
                                tp_tensor_split = [_share_by_id[i] for i in _reordered]
                                gpu_indices = _reordered
                        elif not tp_tensor_split:
                            gpu_indices = _reordered
                # Auto Vulkan fit prefers discrete GPUs and keeps that pool pinned.
                elif (
                    is_vulkan_backend
                    and gpu_memory_mode != "manual"
                    and use_fit
                    and gpu_indices is None
                    and _detected_gpus
                ):
                    gpu_indices = sorted(idx for idx, _free in _detected_gpus)

                # Status reports the explicit subset the launch actually uses,
                # while reload dedupe compares requested_gpu_ids so repeating a
                # wider request that fit narrowed does not restart the server.
                # A forced-CPU launch carries --device none, so it pins no GPU at all:
                # echoing the request as effective would report devices the runtime never
                # touched, and clearing an ineffective pick would then reload a CPU server
                # that is already in the target state. requested_gpu_ids still holds the
                # pick, so repeating it keeps matching.
                if gpu_ids and not _paravirtual_cpu_forced:
                    effective_pin = gpu_indices if gpu_indices is not None else gpu_ids
                    # Survivor order, not sorted: the block above put gpu_indices in the
                    # order the user picked, and _runtime_matches_intent compares this
                    # value positionally. Sorting here reports an effective pin that a
                    # client round-tripping it would match against the opposite order.
                    self._gpu_ids = [int(idx) for idx in effective_pin]
                else:
                    self._gpu_ids = None

                # Did the APU preflight below find a shortfall. Carried to the launch
                # site, which is where `cmd` and `env` first exist and so the only place
                # the pageable-load override can be applied. Without it an oversized
                # unmapped load on an APU is never remapped: _shared_gpu_ids is
                # populated for Vulkan alone, so the discrete guard downstream credits
                # the APU's reported GPU pool against the weights, finds no spill and
                # abstains, while the RAM this launch actually loads into cannot hold
                # them. "none" and "mlock" do not mmap, so that is an OOM kill.
                _apu_ram_oversized = False
                # model_size less a lazily read table; the text-only reprice charges the same.
                _apu_resident_model_size = model_size
                # The two shortfall notices this load recorded, kept verbatim (and
                # un-amended) so the text-only fallback further down can re-price them
                # once it drops the CPU-pinned projector whose bytes they charged. Both
                # are re-stated by the arch-crash retry, which starts its own warning
                # scope, so what these hold is always the pair behind the notice the
                # route would hand back right now.
                _apu_ram_msg: Optional[str] = None
                _host_ram_msg: Optional[str] = None
                # The RAM figure those notices were priced against, kept with them.
                _apu_avail_mib: Optional[int] = None
                # ...and which part they describe, for the same reason.
                _apu_ram_part = "APU"

                # Unified-memory APUs load weights into system RAM (under WSL the VM
                # cap, not the ROCm-reported VRAM, is the real ceiling); refuse an
                # oversize load the OS would otherwise kill mid-flight. Base model
                # only: an optional MTP drafter is dropped by the MTP-drop fallback.
                # CUDA/ROCm ids only; a Vulkan build's gpu_indices are ggml ordinals.
                if (
                    model_size is not None
                    and not is_vulkan_backend
                    # An integrated CUDA SoC loads into system RAM like an APU, and
                    # _shared_gpu_ids is Vulkan-only, so without this its pool is
                    # credited as dedicated VRAM and the spill prices out at zero.
                    and (
                        self._amd_apu_wants_unified_memory(gpu_indices)
                        or (
                            self._integrated_cuda_probe_is_free()
                            and self._integrated_cuda_selection_is_all_shared(gpu_indices)
                        )
                    )
                ):
                    # Read ONCE and kept, because the text-only fallback re-prices this
                    # same decision much later, with the weights already resident. A
                    # second live reading there would be the pool MINUS the model the
                    # reprice is asking about, which double-charges it.
                    _apu_avail_mib = self._available_system_memory_mib()
                    # Kept beside the notice: the text-only fallback rebuilds it later
                    # and would otherwise turn a Spark's into an APU's, .wslconfig hint
                    # and all.
                    _apu_ram_part = (
                        "APU" if self._amd_apu_wants_unified_memory(gpu_indices) else "SoC"
                    )
                    _apu_resident_model_size = max(
                        1,
                        model_size
                        - self._lazy_read_host_bytes(
                            model_path,
                            caps = server_caps,
                            extra_args = extra_args,
                            env = os.environ,
                            auto_resolves_off = self._lazy_auto_resolves_off(
                                gpu_indices = gpu_indices,
                                detected_gpus = _detected_gpus,
                                shared_gpu_ids = _shared_gpu_ids,
                                is_vulkan_backend = False,
                                probe_integrated_cuda = False,
                            ),
                        ),
                    )
                    _ram_msg = self._apu_ram_shortfall_message(
                        # A pinned projector left model_size but not system RAM, and
                        # this guard exists to stop an oversize load being OOM-killed
                        # mid-read, so it has to weigh the projector either way.
                        _apu_resident_model_size + _mmproj_pinned_bytes,
                        _apu_avail_mib,
                        part = _apu_ram_part,
                    )
                    # gpu_indices is None for a launch nothing pinned, so the guard
                    # priced every visible card, including an APU the gate is about to
                    # drop and whose shared pool the narrowed child never touches.
                    # Re-ask for the set the child will actually see. Warned branch
                    # only, so a passing launch pays for no probe; an empty answer
                    # keeps the warning.
                    if _ram_msg and gpu_indices is None and not gpu_ids:
                        _gated_pin = self._arch_gate_survivors(binary)
                        if _gated_pin and not self._amd_apu_wants_unified_memory(_gated_pin):
                            logger.info(
                                "Skipping the APU RAM warning: the arch gate pins "
                                "this launch to discrete GPU(s) %s.",
                                _gated_pin,
                            )
                            _ram_msg = None
                    # The condition, not the message: read after the arch-gate skip
                    # above, so a launch the gate moves onto discrete cards is not
                    # remapped for a shortfall it no longer has. Independent of
                    # UNSLOTH_ALLOW_HOST_OFFLOAD, which this preflight never consulted.
                    _apu_ram_oversized = bool(_ram_msg)
                    # ...and the MESSAGE is the one thing the deprecated opt-out may
                    # take away. _apu_ram_shortfall_message prices RAM and never reads
                    # it, so without this UNSLOTH_ALLOW_HOST_OFFLOAD silenced the
                    # discrete guard next door and left the APU advisory in
                    # memory_warning, against a contract that is only ever "silences
                    # the warning". Read AFTER the verdict above, so the rewrite the
                    # verdict drives is untouched.
                    if _ram_msg and self._host_offload_warning_opted_out():
                        logger.info(
                            "UNSLOTH_ALLOW_HOST_OFFLOAD set: not recording the APU RAM advisory."
                        )
                        _ram_msg = None
                    _apu_ram_msg = _ram_msg
                    self._record_load_warning(_ram_msg)

                # Audio input straight from the mmproj (clip.has_audio_encoder),
                # independent of token names. A projector passed only via
                # LLAMA_ARG_MMPROJ misses launch_mmproj_path, so probe that too,
                # but never one the unpinnable guard just dropped.
                self._mmproj_has_audio = False
                self._mmproj_accepts_image = True
                # ...and never one the vision switch is about to scrub out of the
                # child environment either. Reading it would record an audio encoder
                # the launched server does not have, and _apply_detected_audio turns
                # that into has_audio_input, so the composer would offer attachments
                # the server cannot process.
                #
                # Unless the switch is going to keep it: an inherited audio-only
                # encoder survives the scrub below, so the server really does have it
                # and the same reasoning says to record it. One predicate for both, off
                # the same inherited value, so the probe cannot describe a child the
                # scrub did not build.
                # ...and not on a virtualised Metal device, whose own scrub below
                # takes BOTH projector vars unconditionally. It runs after the switch's
                # block, so a file kept there is gone by launch, and a probe that still
                # described it would report an audio encoder the child does not have.
                _dv_env_mmproj_kept = (
                    disable_vision
                    and not _paravirtual_cpu_forced
                    and _mmproj_env_is_audio_only(os.environ.get("LLAMA_ARG_MMPROJ"))
                )
                _mmproj_probe = launch_mmproj_path or (
                    ""
                    if (_pv_mmproj_unpinnable or (disable_vision and not _dv_env_mmproj_kept))
                    else (os.environ.get("LLAMA_ARG_MMPROJ") or "")
                )
                self._mmproj_projector_type = None
                if launch_mmproj_path or os.path.isfile(_mmproj_probe):
                    try:
                        from utils.models.gguf_metadata import (
                            mmproj_capabilities,
                            read_mmproj_projector_type,
                        )

                        has_audio, accepts_image = mmproj_capabilities(_mmproj_probe)
                        self._mmproj_has_audio = has_audio
                        self._mmproj_accepts_image = accepts_image
                        # The family name, so KV admission can look up how many tokens
                        # one image may become instead of assuming a ceiling.
                        self._mmproj_projector_type = read_mmproj_projector_type(_mmproj_probe)
                    except Exception as e:
                        logger.debug(f"mmproj capability read failed: {e}")

                auto_fit = gpu_memory_mode == "manual" and gpu_layers < 0
                # What the user asked for, kept whole. The strip below is a launch
                # decision, but _requested_extra_args is the comparator a later Apply
                # is matched against: storing the stripped list there made an
                # unchanged request compare unequal and reload the model.
                _extra_args_as_requested = list(extra_args) if extra_args is not None else None
                if self._metal_drops_zero_ctx_override(ctx_override, _caller_owns_budget):
                    extra_args = strip_context_only(extra_args)
                    logger.warning(
                        "Dropping the pass-through zero context: on Metal it pins the "
                        "model's native length and disables --fit."
                    )
                _metal_floor = self._metal_zero_ctx_floor(
                    effective_ctx,
                    auto_fit,
                    _caller_owns_budget,
                    self._context_length,
                    max_available_ctx,
                    # Whether the child's own fitter runs at all, which decides both the
                    # Auto-layers exemption and how far the floor may reach. Every branch
                    # that can reach the floor emits "--fit on" or nothing (llama.cpp
                    # defaults it on); the one branch that emits "--fit off" is manual
                    # with a fixed layer count, which _caller_owns_budget exempts. So the
                    # only ways off here are the user's own, and both are read: extras
                    # land after Unsloth's flag and win, and LLAMA_ARG_FIT is applied
                    # before argv when argv sets nothing.
                    fitter_runs = fit_is_effectively_on(extra_args, os.environ),
                )
                if _metal_floor:
                    effective_ctx = _metal_floor
                    # Replace, do not max(). On the exception path max_available_ctx
                    # still holds the native length its initialiser put there, which
                    # no cap ever said fits. Keeping the larger publishes native as
                    # max_context_length, which the UI reads as the largest context
                    # that fits, advertising this very failure as safe. The floor is
                    # already bounded by any real ceiling the cap did compute.
                    max_available_ctx = _metal_floor
                    logger.warning(
                        "No GPU is enumerated on Metal and the context cap did not "
                        f"run, so starting at {effective_ctx} rather than the model's "
                        "native length."
                    )

                # Gated like every other optional flag, but failing OPEN: these
                # are emitted on every launch, so an unreadable --help keeps
                # today's command and only a build that positively lacks one
                # drops it. Harmless for the pinned prebuilt, which has all
                # three; the case this covers is a stale or user-supplied
                # LLAMA_SERVER_PATH, where an unknown argument is an immediate
                # exit rather than a degraded launch.
                _caps = _launch_caps(binary)
                cmd = [
                    binary,
                    "-m",
                    model_path,
                    "--port",
                    str(self._port),
                    "--parallel",
                    str(n_parallel),
                ]
                # Force flash attention for speed. Two separate answers: whether
                # the build has the flag at all (predates flash attention, or a
                # wrapper that never exposed it), and whether its declaration
                # takes a value -- older builds take -fa as a bare boolean and
                # read a following "on" as a stray positional.
                if _caps.get("supports_flash_attn", True):
                    if not _extra_args_set_any_flag(extra_args, {"-fa", "--flash-attn"}):
                        cmd.append("--flash-attn")
                        if _caps.get("flash_attn_takes_value", True):
                            cmd.append("on")
                else:
                    # Only ever reached on an authoritative --help that has no
                    # flash attention to lose, so there is no speed to give up.
                    _flash_attn_known_off = True
                # Error out at n_ctx instead of silently rotating the KV cache; frontend catches it and points the user at "Context Length".
                if _caps.get("supports_no_context_shift", True):
                    if not _extra_args_set_any_flag(
                        extra_args, {"--context-shift", "--no-context-shift"}
                    ):
                        cmd.append("--no-context-shift")
                # A positive context is always passed (in auto-fit, --fit then
                # optimizes the gpu-layer offload around it). When auto-fit has
                # no explicit context, omit -c so --fit sizes it to fit VRAM:
                # "-c 0" would instead pin the FULL native context (llama.cpp's
                # -c handler sets fit_params_min_ctx = UINT32_MAX on value 0,
                # disabling --fit's reduction). See gpu_memory_mode.
                if effective_ctx > 0:
                    cmd.extend(["-c", str(effective_ctx)])
                elif not auto_fit:
                    cmd.extend(["-c", "0"])

                # llama-server 501s on /v1/embeddings without this; pooling stays model-default.
                if self.is_embedding_gguf:
                    cmd.append("--embedding")

                # emitted before user extras so a pass-through -b / -ub still last-wins-overrides
                if n_batch is not None:
                    # Two separate aborts, both reachable from the picker since the input
                    # clamps into BATCH_MIN = 1 rather than rejecting:
                    #   -b < --parallel  -> GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)
                    #   -b 1 (any slots) -> GGML_ASSERT(n_tokens_all <= cparams.n_batch)
                    # Measured: b1/p1 and b1/p2 abort, b2/p1 and b2/p2 load, b4/p8 aborts,
                    # b8/p8 loads. So the floor is max(slots, 2), not slots alone. The
                    # bounds are per-field and cannot express it, so raise here instead of
                    # failing the load.
                    _emit_batch = _emitted_n_batch(n_batch, n_parallel)
                    if _emit_batch != n_batch:
                        logger.warning(
                            "Raising --batch-size from %s to %s: llama-server aborts below "
                            "2 and below the %s parallel slot(s) it serves.",
                            n_batch,
                            _emit_batch,
                            n_parallel,
                        )
                    cmd.extend(["--batch-size", str(_emit_batch)])
                if n_ubatch is not None:
                    cmd.extend(["--ubatch-size", str(n_ubatch)])
                # The expert-spill raise, as emitted and as it was before, so a CPU
                # replay (no GPU, nothing to stream) can hand back today's pair.
                self._moe_spill_batch_tokens = None
                if self._moe_spill_batch_restore is not None:
                    _orig_batch, _orig_ubatch = self._moe_spill_batch_restore
                    _raised = ["--ubatch-size", str(n_ubatch)]
                    _before = [] if _orig_ubatch is None else ["--ubatch-size", str(_orig_ubatch)]
                    if n_batch is not None and n_batch != _orig_batch:
                        _raised = ["--batch-size", str(_emit_batch), *_raised]
                        _before = [
                            *(
                                []
                                if _orig_batch is None
                                else [
                                    "--batch-size",
                                    str(_emitted_n_batch(_orig_batch, n_parallel)),
                                ]
                            ),
                            *_before,
                        ]
                    self._moe_spill_batch_tokens = (_raised, _before)

                server_caps = _launch_caps(binary)

                # Before the extras, like the batch pair: a hand-typed flag still
                # last-wins over the control. Each is gated on the capability
                # probe, because a build that predates the flag exits on it.
                if ctx_checkpoints is not None:
                    _ctxcp_flag = server_caps.get("ctx_checkpoints_flag")
                    if _ctxcp_flag:
                        cmd.extend([str(_ctxcp_flag), str(int(ctx_checkpoints))])
                    else:
                        logger.info(
                            "llama-server has no --ctx-checkpoints; skipping the requested %s.",
                            ctx_checkpoints,
                        )
                if cache_ram is not None:
                    if server_caps.get("supports_cache_ram"):
                        cmd.extend(["--cache-ram", str(int(cache_ram))])
                    else:
                        logger.info(
                            "llama-server has no --cache-ram; skipping the requested %s MiB.",
                            cache_ram,
                        )
                # Read early so the effective budget can fold in an inherited
                # LLAMA_ARG_THINK_BUDGET*; the launch env is rebuilt below before spawn.
                env = self._llama_server_env_for_binary(binary)
                # On every launch, not only where Studio emits --mmproj: a projector
                # from the extras or LLAMA_ARG_MMPROJ takes video too.
                cmd.extend(_video_fps_flags(server_caps))

                # Report a clean public model id (matching GET /v1/models) rather
                # than the raw -m path in llama-server's own /v1/models and the
                # "model" field of its chat/completions responses.
                from core.inference.model_ids import public_model_id

                _alias = public_model_id(self._model_identifier or model_path)
                if _alias:
                    cmd.extend(["--alias", _alias])

                fully_gpu_offloaded = False
                # Set when a positional --tensor-split is emitted, so the env block
                # can pin CUDA to PCI order even without a GPU subset (see below).
                manual_tensor_split_emitted = False
                # The MoE cache's "auto" sizing runs only under llama.cpp's fitter.
                _placed_by_fitter = False
                if gpu_memory_mode == "manual" and gpu_layers >= 0:
                    # Pin the user's layer count and disable auto-fit. --fit off
                    # also means _ctx_integrity_flags must not add --fit-ctx.
                    use_fit = False
                    cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])
                    # Keep the first n_cpu_moe MoE layers' experts on CPU.
                    moe_flag = self._resolve_cpu_moe_flag(
                        n_cpu_moe,
                        self.n_moe_layers,
                        self._leading_dense_block_count or 0,
                    )
                    if moe_flag is not None:
                        cmd.extend(["--n-cpu-moe", str(moe_flag)])
                    elif n_cpu_moe:
                        # Requested on a dense model: nothing was emitted, so
                        # don't report a count llama-server never received.
                        self._n_cpu_moe = 0
                    # Distribute the model across GPUs by the user's per-GPU shares
                    # (--tensor-split). Works with layer split and tensor
                    # parallelism; --fit off means no fit/tensor abort. Only emit
                    # when >1 GPU is in use AND the list length matches that count:
                    # the field is hidden (not cleared) when the picker narrows to
                    # one, and a direct caller can send a stale ratio for a different
                    # GPU set. Unsloth drops any mismatch to the free-VRAM default
                    # (llama.cpp would silently zero-pad a short list, or abort past
                    # its 16-device cap).
                    _split_gpus = self._effective_gpu_count(gpu_indices)
                    if tensor_split and _split_gpus > 1:
                        # An all-zero/non-positive sanitized split assigns nothing
                        # anywhere, so fall through to the free-VRAM default in
                        # that case.
                        _sanitized_split = self._sanitize_tensor_split(tensor_split)
                        _split_total = sum(_sanitized_split)
                        if len(_sanitized_split) == _split_gpus and _split_total > 0:
                            cmd.extend(
                                ["--tensor-split", ",".join(f"{x:g}" for x in _sanitized_split)]
                            )
                            self._tensor_split = _sanitized_split
                            manual_tensor_split_emitted = True
                        else:
                            logger.warning(
                                "Dropping manual --tensor-split (%d entries for "
                                "%d GPUs, sanitized total %s); llama.cpp's "
                                "free-VRAM split applies instead",
                                len(tensor_split),
                                _split_gpus,
                                _split_total,
                            )
                            self._tensor_split = None
                    elif tensor_split:
                        # Single effective GPU: the split is never emitted, so
                        # don't report it as active via /status and /load.
                        self._tensor_split = None
                elif use_fit and _metal_full_offload_pinned:
                    # Keep the launch on the full-offload mapping the discount measured.
                    cmd.extend(["-ngl", "-1", "--fit", "off"])
                    # Same claim as the pinned-GPU arm below, so it owes the same
                    # recovery: the --fit on retry after a startup crash is gated on
                    # this flag, and without it an optimistic discount would skip
                    # straight to the terminal fallbacks. -1 is llama.cpp's own
                    # default (llama_model_default_params), so the retry's fitter
                    # does not hit its "n_gpu_layers already set by user" abort
                    # (common/fit.cpp:462) and can still move layers off Metal.
                    fully_gpu_offloaded = True
                elif use_fit:
                    # Unsloth could not prove a fit, so llama.cpp's fitter takes the
                    # placement. Its dense path fills "back to front with dense
                    # layers" (common/fit.cpp:402), i.e. WHOLE layers, and a layer
                    # takes its KV cache with it because the cache buffer comes
                    # from model.dev_layer(il) (llama-kv-cache.cpp:215). Spilling
                    # weight TENSORS instead leaves the cache resident: measured on
                    # a 27B at 128K, 13.63 t/s resident against 1.03 t/s evicted.
                    _spill = self._planned_tensor_spill(
                        _spill_inputs,
                        extra_args = extra_args,
                        env = os.environ,
                    )
                    # NOT `if _spill:` -- Plan is a dataclass, so every instance is
                    # truthy, including ones saying it could not place this at all.
                    _spill_flags = self._spill_plan_flags_for(_spill)
                    if _spill_flags:
                        self._spill_plan_flags = _spill_flags
                        cmd.extend(self._spill_plan_flags)
                        logger.info(
                            "Tensor spill: %s (resident %.1f GB, host %.1f GB)",
                            " ".join(_spill.ot_patterns),
                            _spill.vram_bytes / (1024**3),
                            _spill.host_bytes / (1024**3),
                        )
                    else:
                        cmd.extend(["--fit", "on"])
                        _placed_by_fitter = True
                elif gpu_indices is not None:
                    # Fits on selected GPU(s) -- force all layers on GPU. --fit off is
                    # required: without it llama.cpp's default --fit on second-guesses
                    # and offloads ~1 GB at --parallel 4 even though the model fits.
                    cmd.extend(["-ngl", "-1", "--fit", "off"])
                    fully_gpu_offloaded = True

                # Expose Prometheus /metrics for the engine-stats logger, only
                # when the binary advertises it (older/custom binaries may not).
                if server_caps.get("supports_metrics"):
                    if not _extra_args_set_any_flag(extra_args, {"--metrics", "--no-metrics"}):
                        cmd.append("--metrics")
                self._slot_save_dir = None
                self._slot_save_binary = None
                self._prompt_cache_disabled = False
                if server_caps.get("supports_slot_save"):
                    try:
                        from utils.paths.storage_roots import (  # noqa: WPS433
                            llama_slot_cache_root,
                        )

                        slot_dir = llama_slot_cache_root()
                        slot_dir.mkdir(parents = True, exist_ok = True)
                        # Saved KV encodes chat content; keep it from other local users.
                        with contextlib.suppress(OSError):
                            os.chmod(slot_dir, 0o700)
                        slot_path = str(slot_dir)
                        # llama.cpp's fs_is_directory() constructs a narrow
                        # std::filesystem::path on Windows, so an existing UTF-8
                        # path such as C:\Users\Егор is rejected as missing.
                        # Slot persistence is optional; keep model loading available
                        # until the managed runtime carries the upstream UTF-8 fix.
                        if sys.platform == "win32" and not slot_path.isascii():
                            logger.warning(
                                "Disabling llama.cpp slot persistence because its "
                                "Windows path contains non-ASCII characters: %s",
                                slot_path,
                            )
                        else:
                            cmd.extend(["--slot-save-path", slot_path])
                            self._slot_save_dir = slot_path
                            self._slot_save_binary = (binary, Path(binary).stat().st_mtime_ns)
                    except OSError:
                        self._slot_save_dir = None
                        self._slot_save_binary = None
                _integrity_args = (
                    n_parallel,
                    use_fit,
                    auto_fit,
                    requested_ctx,
                    effective_ctx,
                    server_caps,
                )
                _integrity_flags = self._ctx_integrity_flags(
                    *_integrity_args,
                    fit_target_delta_mib = _fit_target_delta_mib,
                )
                # From the flags, not the inputs: --fit off, an older server without
                # the capability and a budget at the default all leave the child
                # unpriced, and each is decided inside that call.
                _fit_target_priced = "--fit-target" in _integrity_flags
                if _metal_fit_target_floor_mib:
                    # Rebuilt after the price is read: the floor is the machine's
                    # margin, not the budget slider's.
                    _integrity_flags = self._ctx_integrity_flags(
                        *_integrity_args,
                        fit_target_delta_mib = _fit_target_delta_mib,
                        fit_target_floor_mib = _metal_fit_target_floor_mib,
                    )
                cmd.extend(_integrity_flags)
                offload_overridden = _extra_args_set_any_flag(
                    extra_args, _GPU_OFFLOAD_OVERRIDE_FLAGS
                )
                threads_overridden = _extra_args_set_any_flag(extra_args, _THREAD_OVERRIDE_FLAGS)
                full_offload_tuning_active = fully_gpu_offloaded and not offload_overridden
                # The CPU replay must undo what this launch pins for an absent GPU;
                # a user's own --threads still wins.
                _drop_full_offload_threads = (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and not threads_overridden
                )

                # Thread count: an unset --threads makes llama.cpp pick physical
                # cores (common_cpu_get_num_math), but an explicit --threads -1
                # resolves to hardware_concurrency() (every hyperthread), which
                # contends on the memory bus and slows CPU / hybrid decode. So
                # omit the flag when unset and only pin it for an explicit
                # override or the Windows full-offload OpenMP cap. Pass-through
                # thread flags in extra_args still win (appended last). #5692
                if (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and not threads_overridden
                ):
                    cmd.extend(["--threads", "2"])
                elif n_threads is not None and n_threads > 0:
                    cmd.extend(["--threads", str(n_threads)])

                # Enable Jinja chat template rendering
                if _caps.get("supports_jinja", True):
                    if not _extra_args_set_any_flag(extra_args, {"--jinja", "--no-jinja"}):
                        cmd.extend(["--jinja"])
                cmd.extend(_trace_verbosity_args(_caps, extra_args, os.environ))

                # KV cache data type
                _valid_cache_types = _VALID_CACHE_TYPES
                # Normalize like the budget does (_planned_main_cache_types): a
                # case-sensitive match drops "Q8_0", emitting no flag, so llama.cpp
                # runs f16 while the estimate priced q8_0. Emit the normalized
                # spelling; kv_cache_type_from_str is case-sensitive.
                cache_type_kv = cache_type_kv.strip().lower() if cache_type_kv else cache_type_kv
                if (
                    cache_type_kv
                    and cache_type_kv in _valid_cache_types
                    and not _cache_type_from_env
                ):
                    gpu_fallback_warning = (
                        None
                        if intent.cpu_fallback
                        or _arch_gate_forced_cpu
                        or not _detected_gpus
                        or _extras_cache is not None
                        else _kv_cache_gpu_fallback_warning(
                            cache_type_kv,
                            gpu_memory_mode,
                            gpu_layers,
                            lambda: self._installed_ggml_backends(binary),
                        )
                    )
                    if gpu_fallback_warning:
                        logger.warning(gpu_fallback_warning)
                        cache_type_kv = _GPU_UNACCELERATED_CACHE_TYPES[cache_type_kv]
                    if not _extra_args_set_any_flag(extra_args, _CACHE_TYPE_K_FLAGS):
                        cmd.extend(["--cache-type-k", cache_type_kv])
                    if not _extra_args_set_any_flag(extra_args, _CACHE_TYPE_V_FLAGS):
                        cmd.extend(["--cache-type-v", cache_type_kv])
                    self._cache_type_kv = cache_type_kv
                    logger.info(f"KV cache type: {cache_type_kv}")
                else:
                    # An env-only type is left inherited (untouched) so an
                    # asymmetric K/V env reaches the child as set.
                    self._cache_type_kv = None

                # Tensor parallelism: split the model across GPUs by tensor
                # rather than by layer. The UI only offers it on multi-GPU; a
                # direct single-GPU caller is redundant (supported archs no-op,
                # unsupported ones abort and the /load path retries layer split).
                # Default (layer split) is left implicit by omitting the flag.
                # See llama.cpp --split-mode.
                if tensor_parallel:
                    cmd.extend(["--split-mode", "tensor"])
                    _emitted_tensor_split: Optional[str] = None
                    if tp_tensor_split and len(tp_tensor_split) > 1:
                        _emitted_tensor_split = ",".join(str(int(x)) for x in tp_tensor_split)
                    elif (
                        gpu_memory_mode != "manual" and tensor_split and _tp_planned and gpu_indices
                    ):
                        # Auto tensor planning may decide an even split is safe and
                        # return None, but a user who set a per-GPU ratio still expects
                        # it to reach llama-server. Fall back to the ratio only when
                        # the planner emitted nothing and the ratio actually fits the
                        # per-GPU budget the planner used.
                        #
                        # Gated on _tp_planned, not on tp_tensor_split alone: the fit
                        # block's except arm ALSO leaves tp_tensor_split None, with
                        # gpu_indices either None or rebuilt from every device rather
                        # than the ones the reserve filter admitted. That arm is a
                        # designed degradation -- it launches under `--fit on` -- and it
                        # must not be turned into a KeyError by pricing a ratio against
                        # a survey that was never taken.
                        _split_gpus = self._effective_gpu_count(gpu_indices)
                        _sanitized_split = self._sanitize_tensor_split(tensor_split)
                        if (
                            len(_sanitized_split) == _split_gpus
                            and sum(_sanitized_split) > 0
                            and self._tensor_split_fits_budget(
                                _sanitized_split,
                                tp_gpus,
                                gpu_indices,
                                model_size,
                                effective_ctx,
                                cache_type_kv = cache_type_kv,
                                n_parallel = n_parallel,
                                n_ubatch = _effective_ubatch,
                                kv_unified = planned_kv_unified,
                                flash_attn = planned_flash_attn,
                                swa_full = swa_full,
                                mtp_engaged = _mtp_reserves_gpu,
                                mtp_overhead_fn = mtp_overhead_fn,
                                mtp_flat_reserve_bytes = (
                                    2 * 1024**3 if (_mtp_reserves_gpu and _mtp_kv_unsized) else 0
                                ),
                                total_by_idx = total_by_idx,
                                vram_fraction = _vram_frac,
                                # The planner's own operands, so "the same budget"
                                # in the docstring is true. Both are bound by the
                                # time _tp_planned is set.
                                soft_overhead_bytes = _soft_overhead,
                                scratch_cache_type_kv = _scratch_cache_type_kv,
                            )
                        ):
                            _emitted_tensor_split = self._format_tensor_split(_sanitized_split)
                    _emitted_values: Optional[list[float]] = None
                    if _emitted_tensor_split is not None:
                        cmd.extend(["--tensor-split", _emitted_tensor_split])
                        _emitted_values = [float(x) for x in _emitted_tensor_split.split(",")]
                    self._tensor_parallel = True
                    self._layer_preserves_tensor_intent = False
                    # Record the ratio this auto load was ASKED for, so a later
                    # request with a different ratio is not incorrectly reused.
                    # Not the emitted list: see _auto_split_fingerprint for why
                    # recording that instead reloads on every identical repeat.
                    if gpu_memory_mode != "manual":
                        self._auto_tensor_split = self._auto_split_fingerprint(tensor_split)
                        self._auto_tensor_split_emitted = self._auto_split_fingerprint(
                            _emitted_values
                        )
                    logger.info(
                        "Tensor parallelism: --split-mode tensor, --tensor-split %s",
                        _emitted_tensor_split,
                    )
                else:
                    self._tensor_parallel = False
                    # > 1 only when a tensor request was downgraded but kept multi-GPU.
                    self._layer_preserves_tensor_intent = _layer_min_gpus > 1

                # Speculative decoding. See _build_speculative_flags for the
                # mode resolution, benchmarks, and llama.cpp references.
                _vulkan_pin_ids = gpu_indices if gpu_indices is not None else (gpu_ids or None)
                _draft_device = (
                    _extra_args_main_device(extra_args) if not _gpu_ids_own_device_flags else None
                )
                if _draft_device is None and is_vulkan_backend and _vulkan_pin_ids:
                    _draft_device = ",".join(f"Vulkan{i}" for i in _vulkan_pin_ids)
                launch_mtp_draft_path = self._resolve_launch_mtp_path(
                    mtp_draft_path = mtp_draft_path,
                    drafter_label = _DRAFTER_DISPLAY_LABELS.get(_spec_canon, "MTP"),
                )
                _suppressed_draft_path: Optional[str] = _unloadable_mtp_draft_path
                _suppressed_draft_reason: Optional[str] = (
                    "unloadable" if _unloadable_mtp_draft_path else None
                )
                _pv_suppressed_spec_extra_args: Optional[List[str]] = None
                # Same shape as the projector drop above: the CPU pin below needs a
                # draft-layer flag from the probe, and without one the drafter keeps its
                # automatic placement on the virtualised device. Unlike the projector
                # this is throughput, not correctness -- a token is emitted only when the
                # CPU-side target samples it itself (common_sampler_sample_and_accept_n
                # pushes the target's own draw and stops at the first mismatch), so a
                # corrupt drafter can only get its proposals rejected, costing a forward
                # pass per step for nothing. So drop it, from both sources: Unsloth's
                # resolved drafter and a user/env one in the extras. The rest of the spec
                # group goes with it, since a --spec-type needing a draft model
                # (draft-simple, eagle3) aborts llama-server once the model is gone;
                # _build_speculative_flags re-derives the drafter-free modes below. A
                # user who already pinned the drafter (-ngld 0 / --spec-draft-device cpu)
                # keeps it; the probe only decides whether Unsloth can emit the flag.
                # And as with the projector, --gpu-layers-draft has existed since 2023,
                # so a missing flag means the probe failed: on an unanswered probe pin
                # with the legacy spelling rather than drop.
                # Keyed on the drafter the launch really loads, not one that merely
                # exists: a user-owned --spec-type returns early from
                # _build_speculative_flags, so Unsloth's resolved sibling never becomes
                # --model-draft and keying on it would strip a drafter-free mode
                # (--spec-type ngram-mod and its knobs) for a drafter that never
                # launches. The inherited env counts only where it survives to the child
                # (the launch scrubs LLAMA_ARG_SPEC_DRAFT_* unless the extras own
                # --spec-type), or os.environ would drop a drafter that never loads and
                # take the caller's spec tuning with it.
                _pv_draft_unpinnable = bool(
                    _paravirtual_cpu_forced
                    and not _paravirtual_draft_ngl_flag(server_caps)
                    and (
                        _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                        or (launch_mtp_draft_path and not _extra_args_set_spec_type(extra_args))
                    )
                    and not _extra_args_draft_offloaded_to_cpu(extra_args)
                )
                if _pv_draft_unpinnable:
                    logger.warning(
                        "Dropping the speculative drafter for this session: this "
                        "Mac's Metal device is virtualised and this llama-server "
                        "build advertises no draft-layer flag, so the drafter would "
                        "keep running on the device whose output is corrupt. Output "
                        "stays correct either way; this only costs speed. Run "
                        "'unsloth studio update' to install a build that supports it."
                    )
                    # Remember what was suppressed: the drafter stays on disk, so caller
                    # and route keep detecting it, and comparing that against the
                    # launched None would reload a healthy server on every repeat Apply.
                    # Only what this drop removed: extras trigger it with none of ours
                    # launched, and None would clear a record an earlier drop made.
                    if launch_mtp_draft_path:
                        _suppressed_draft_path = launch_mtp_draft_path
                        # A property of the build, not the file, so a repair must not reload.
                        _suppressed_draft_reason = "paravirtual"
                    launch_mtp_draft_path = None
                    if extra_args:
                        # Same reason: record the extras as REQUESTED next to the
                        # launched ones. Both comparators key on the caller's list,
                        # which keeps arriving unchanged, so comparing it against this
                        # rewrite would reload a healthy server on every repeat Apply.
                        _pv_suppressed_spec_extra_args = list(extra_args)
                        extra_args = strip_shadowing_flags(
                            extra_args,
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                # The same view of argv and env the child will get, built with the
                # helpers the launch itself uses so the two cannot drift: manual mode
                # drops the placement vars, and a gpu_ids pin drops the device flags
                # from both. Classifying on the raw extras would read a stale
                # --device none as CPU-only for a load that does offload.
                _spec_placement_env = dict(os.environ)
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(_spec_placement_env)
                _spec_placement_extras = extra_args
                if _gpu_ids_own_device_flags:
                    _spec_placement_extras = self._strip_device_extra_args(extra_args)
                    self._clear_device_placement_env(_spec_placement_env)
                # A device the launch really carries, and not the CPU spelling of it.
                # Only consulted where the probe found nothing: it is the one remaining
                # way the child can still be pointed at a GPU.
                _spec_device_pick = _extra_args_main_device(
                    [*cmd, *(_spec_placement_extras or [])]
                ) or _spec_placement_env.get("LLAMA_ARG_DEVICE")
                _spec_extras_pick_a_gpu = bool(_spec_device_pick) and not _device_selection_is_cpu(
                    [*cmd, *(_spec_placement_extras or [])], _spec_placement_env
                )
                spec_flags = self._build_speculative_flags(
                    speculative_type = speculative_type,
                    spec_draft_n_max = spec_draft_n_max,
                    extra_args = extra_args,
                    model_identifier = model_identifier,
                    model_path = model_path,
                    gpus = bool(_detected_gpus),
                    binary = binary,
                    mtp_draft_path = (
                        None if _spec_canon in ("dspark", "dflash") else launch_mtp_draft_path
                    ),
                    dspark_draft_path = (launch_mtp_draft_path if _spec_canon == "dspark" else None),
                    dspark_fit_sized = not use_fit,
                    dflash_draft_path = (launch_mtp_draft_path if _spec_canon == "dflash" else None),
                    dflash_fit_sized = not use_fit,
                    # So the fallback says unopenable, not missing. DSpark/DFlash never
                    # reach the drop.
                    mtp_drafter_unloadable = bool(_unloadable_mtp_draft_path),
                    drafter_no_vram = _spec_dropped_no_vram,
                    embedded_mtp_partial_offload = bool(
                        _spec_canon == "auto"
                        and self._nextn_predict_layers
                        and self._ssm_inner_size is not None
                        and self._full_attention_interval is not None
                        and not launch_mtp_draft_path
                        and not _extra_args_set_spec_type(extra_args)
                        # No GPU, nothing to partially offload TO. `--fit on` is also
                        # what a CPU-only box and a Metal Mac emit (use_fit starts True
                        # and only the planner lowers it), and there the rollback copies
                        # cost no VRAM at all -- same reason the helper treats -ngl 0 as
                        # CPU-only rather than partial. Those keep the CPU MTP policy.
                        # gpu_indices counts too: an explicit pick the VRAM probe could
                        # not enumerate still pins the child to those devices, so the
                        # launch offloads to them whatever the probe returned. So does
                        # a device named in the extras, which is the same flag
                        # _device_selection_is_cpu reads for the CPU answer: a failed
                        # probe is not evidence of no GPU when the child is pointed at
                        # one by hand.
                        and (_detected_gpus or gpu_indices or _spec_extras_pick_a_gpu)
                        and self._partially_offloads_layers(
                            [*cmd, *(_spec_placement_extras or [])],
                            _spec_placement_env,
                            # Manual mode skips Unsloth's placement planner (gpus is
                            # emptied above), so its --fit on is the default this
                            # function starts at, not a finding that the model does
                            # not fit. Only Auto's fitter carries that evidence --
                            # and only when it actually ran and actually failed to
                            # fit, which is what _placement_verdict_partial records.
                            # Auto alone is not enough: an empty VRAM probe with a
                            # hand-pinned device (Metal0, Vulkan0) satisfies the GPU
                            # guard above while every planner branch stayed gated on
                            # the empty `gpus`, and the except path restores --fit on
                            # having priced nothing. A concrete --gpu-layers count is
                            # unaffected; it is independent evidence inside the helper.
                            fit_implies_partial = (
                                gpu_memory_mode != "manual" and _placement_verdict_partial
                            ),
                        )
                    ),
                    draft_device = _draft_device,
                )
                # _build_speculative_flags judged the stripped list, so a user
                # --spec-type the drop removed left the requested mode reading "auto"
                # while the caller's extras still mean "the user owns it" (None).
                # Restore the requested view, or the spec-mode compare never matches.
                if _extra_args_set_spec_type(_pv_suppressed_spec_extra_args):
                    self._requested_spec_mode = None
                # The draft KV cache dtype sizes the DRAFT context, so it means
                # nothing without --model-draft. Judged on the flags actually built,
                # not the requested mode: a mode that fell back to no drafter emits
                # nothing, and an MTP load that did attach one is covered.
                # Normalized and allow-listed like the main cache dtype above, and
                # for the same reason: llama-server exits during argument parsing on
                # a value it cannot map, and by then the resident model is gone. An
                # unknown value emits nothing instead, so the load still comes up.
                _draft_cache_type = (
                    spec_draft_cache_type.strip().lower() if spec_draft_cache_type else None
                )
                if _draft_cache_type and _draft_cache_type not in _VALID_KV_CACHE_TYPES:
                    logger.warning(
                        "Ignoring unsupported draft KV cache type %r", spec_draft_cache_type
                    )
                    _draft_cache_type = None
                if _draft_cache_type and any(
                    _flag_name(tok) in _LOCAL_DRAFT_FLAGS for tok in spec_flags
                ):
                    _draft_k_flag = server_caps.get("spec_draft_cache_k_flag")
                    _draft_v_flag = server_caps.get("spec_draft_cache_v_flag")
                    if _draft_k_flag and _draft_v_flag:
                        # Both axes together: llama.cpp refuses K != V on an MLA
                        # draft context, and the control is one dtype for the pair.
                        spec_flags.extend(
                            [
                                str(_draft_k_flag),
                                _draft_cache_type,
                                str(_draft_v_flag),
                                _draft_cache_type,
                            ]
                        )
                        logger.info("Draft KV cache dtype: %s", _draft_cache_type)
                    else:
                        logger.info(
                            "llama-server has no draft KV cache flags; skipping the requested %s.",
                            _draft_cache_type,
                        )
                # Remember where the spec block sits so a drafter-load failure
                # can be retried with these flags swapped out (see below).
                _spec_start = len(cmd)
                cmd.extend(spec_flags)

                # A *separate* drafter does not inherit the main --gpu-layers:
                # common_base_params_to_speculative overwrites n_gpu_layers with
                # params.speculative.draft.n_gpu_layers, whose default -1 means auto, so
                # the drafter would run the corrupt path the CPU pin exists to avoid. An
                # embedded MTP head needs nothing here: with no draft model that override
                # is skipped (has_draft is false) and the head follows the main
                # placement. Probe-gated like the other optional flags so the name
                # matches the build (-ngld predates --spec-draft-ngl) and an unknown
                # argument never refuses the start; a build advertising none of them had
                # its drafter dropped above. extra_args is inspected alongside spec_flags
                # because a user-owned --spec-type makes _build_speculative_flags return
                # nothing, leaving their --model-draft unpinned. Emitted after the
                # pass-through extras below, not here: the parser is last-wins and the
                # paravirtual strip covers only the *main* offload flags, so a user
                # -ngld/--spec-draft-ngl would otherwise undo this.
                _pv_draft_cpu_pin: List[str] = []
                _pv_mmproj_cpu_pin: List[str] = []
                _pv_split_mode_pin: List[str] = []
                # --gpu-layers 0 alone does not keep work off the device: with op_offload
                # on (the default) ggml_backend_sched still sends ops whose weights sit
                # in host buffers to a higher-priority backend, and Metal accepts them at
                # batch >= 32. --device none drops the device from the list outright.
                _pv_device_pin: List[str] = ["--device", "none"] if _paravirtual_cpu_forced else []
                if (
                    _paravirtual_cpu_forced
                    # Same env the child will see: when the extras own --spec-type the
                    # launch keeps LLAMA_ARG_SPEC_DRAFT_*, so a drafter named only there
                    # still loads and would take the corrupt device unpinned.
                    and _extra_args_mtp_draft_path(
                        [*spec_flags, *(extra_args or [])], env = _child_spec_env(extra_args)
                    )
                    is not None
                    and _paravirtual_draft_ngl_flag(server_caps)
                ):
                    # The layer count alone is not enough, for the same reason
                    # --gpu-layers 0 was not: common_base_params_to_speculative replaces
                    # the draft context's device list with the draft one, so the main
                    # --device none never reaches it and an empty draft list leaves every
                    # device visible. --device-draft is spelled the same across the
                    # whole supported range.
                    _pv_draft_cpu_pin = [
                        str(_paravirtual_draft_ngl_flag(server_caps)),
                        "0",
                        "--device-draft",
                        "none",
                    ]
                    logger.warning(
                        "Pinning the speculative drafter to CPU: this Mac's Metal "
                        "device is virtualised."
                    )

                # Emitted after the pass-through extras too, same last-wins reason.
                if _paravirtual_cpu_forced:
                    _pv_split_mode_pin = _paravirtual_split_mode_pin(extra_args)

                self._chat_template_override = chat_template_override
                self._empty_chat_render_refused = False
                _effective_template = self._effective_chat_template(chat_template_override)
                if _effective_template:
                    import tempfile

                    flags = detect_reasoning_flags(
                        _effective_template,
                        self._model_identifier,
                        log_source = (
                            "GGUF chat template override"
                            if chat_template_override
                            else "repaired GGUF chat template"
                        ),
                    )
                    self._supports_reasoning = flags["supports_reasoning"]
                    self._reasoning_style = flags["reasoning_style"]
                    self._reasoning_effort_levels = flags.get("reasoning_effort_levels", [])
                    self._reasoning_always_on = flags["reasoning_always_on"]
                    self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                    self._preserve_thinking_default = flags["preserve_thinking_default"]
                    self._supports_tools = flags["supports_tools"]

                    self._chat_template_file = tempfile.NamedTemporaryFile(
                        mode = "w",
                        encoding = "utf-8",
                        suffix = ".jinja",
                        delete = False,
                        prefix = "unsloth_chat_template_",
                    )
                    self._chat_template_file.write(_effective_template)
                    self._chat_template_file.close()
                    cmd.extend(["--chat-template-file", self._chat_template_file.name])
                    logger.info(f"Using custom chat template file: {self._chat_template_file.name}")

                # Default thinking mode for reasoning models. The Qwen3.5 Small tier
                # (0.8B, 2B, 4B, 9B) ships with reasoning off by default and the rest of
                # the family has it on; the templates are byte-identical across sizes, so
                # this tier split only exists as a launch flag. Always-on templates ignore
                # the kwarg, so skip.
                if self._supports_reasoning and not self._reasoning_always_on:
                    thinking_default = True
                    mid = (model_identifier or "").lower()
                    if "qwen3.5" in mid or "qwen3.6" in mid:
                        # extract_model_size_b prefers MoE active params, reading
                        # 35B-A3B as 3B. Match total params instead: scan segments
                        # right to left so the size nearest the leaf beats a size-like
                        # parent dir, and keep its \s* / M-suffix spellings.
                        size_re = r"(?:^|[-_.])(\d+\.?\d*)\s*([bm])(?:$|[-_.])"
                        size_match = None
                        for seg in reversed(mid.replace("\\", "/").split("/")):
                            size_match = re.search(size_re, seg)
                            if size_match:
                                break
                        if size_match:
                            size_b = float(size_match.group(1))
                            if size_match.group(2) == "m":
                                size_b /= 1000.0
                            # <= 9, not < 9: 9B is the top of the Small tier, so it is off too.
                            if size_b <= 9:
                                thinking_default = False
                    # argv beats the env `unsloth start` pins these through, so repeat it.
                    _env_reasoning = _reasoning_from_env(env)
                    if _env_reasoning is not None:
                        thinking_default = _env_reasoning
                    reasoning_kw = self._reasoning_kwargs(
                        thinking_default, _reasoning_effort_from_env(env)
                    )
                    # An effort ladder thinks at every level but "none" (Inkling: 0), low included.
                    self._reasoning_default = reasoning_kw.get(
                        "enable_thinking", reasoning_kw.get("reasoning_effort") not in ("none", 0)
                    )
                    # preserve_thinking is independent of the thinking gate.
                    # Qwen3.8 defaults it on; Qwen3.6, Gemma 4, and every other
                    # supporting family keep the existing off default. The
                    # frontend receives the same resolved value via load/status.
                    if self._supports_preserve_thinking:
                        reasoning_kw["preserve_thinking"] = self._preserve_thinking_default
                    _reasoning_args = _build_launch_reasoning_args(server_caps, reasoning_kw)
                    cmd.extend(_reasoning_args)
                    logger.info(
                        f"Reasoning model: {reasoning_kw} by default "
                        f"(launched as {' '.join(_reasoning_args)})"
                    )

                reasoning_budget = resolve_reasoning_budget(extra_args, reasoning_budget)
                reasoning_budget_message = resolve_reasoning_budget_message(
                    extra_args, reasoning_budget_message
                )
                cmd.extend(
                    _build_reasoning_budget_flags(
                        server_caps, reasoning_budget, reasoning_budget_message
                    )
                )
                # What was asked for, pass-through flags folded in but not the environment: the
                # value a client can resend, and the one the reuse check compares against.
                self._requested_reasoning_budget = reasoning_budget
                self._requested_reasoning_budget_message = reasoning_budget_message
                # Only a CONCLUSIVE "no such flag" means the child ignores the environment. A probe
                # that timed out says nothing, and the child still applies LLAMA_ARG_THINK_BUDGET*,
                # so recording the bare request there would misreport the running server.
                _budget_env_applies = server_caps.get(
                    "supports_reasoning_budget"
                ) or server_caps.get("reasoning_budget_probe_inconclusive")
                reasoning_budget = (
                    resolve_reasoning_budget_with_env(extra_args, reasoning_budget, env)
                    if _budget_env_applies
                    else reasoning_budget
                )
                reasoning_budget_message = (
                    resolve_reasoning_budget_message_with_env(
                        extra_args, reasoning_budget_message, env
                    )
                    if server_caps.get("supports_reasoning_budget_message")
                    or server_caps.get("reasoning_budget_probe_inconclusive")
                    else reasoning_budget_message
                )
                self._reasoning_budget = reasoning_budget
                self._reasoning_budget_message = reasoning_budget_message

                if launch_mmproj_path and effective_is_vision:
                    cmd.extend(["--mmproj", launch_mmproj_path])
                    logger.info(f"Using mmproj for vision: {launch_mmproj_path}")
                    # The projector never reads --gpu-layers: clip.cpp picks a GPU
                    # backend from the boolean mmproj_use_gpu, which defaults true, so
                    # the vision encoder would keep running the corrupt path the CPU pin
                    # exists to avoid and encode every image from garbage. A build that
                    # conclusively lacks the flag never gets here (the projector is
                    # dropped up front); an inconclusive probe still pins, since every
                    # build that can start here has had the flag since b5178. Deferred
                    # past the pass-through extras like the drafter pin, since
                    # --mmproj-offload is a real flag a user can pass and llama.cpp is
                    # last-wins.
                    # One flag whichever reason pinned it: the corrupt paravirtual Metal
                    # device, or the fit deciding the projector does not fit in VRAM.
                    # Both already checked the capability.
                    if _mmproj_cpu_pinned:
                        _pv_mmproj_cpu_pin = ["--no-mmproj-offload"]
                        if _paravirtual_cpu_forced:
                            logger.warning(
                                "Disabling mmproj GPU offload: this Mac's Metal "
                                "device is virtualised."
                            )

                # llama-server sends permissive CORS, so without a key any open web page could drive it.
                import secrets as _secrets

                _key_on = _llama_server_api_key_enabled()
                if _key_on and not server_caps.get("supports_api_key_file", True):
                    # A custom build predating --api-key-file would exit on the flag; run it keyless as before.
                    logger.warning(
                        "This llama-server build has no --api-key-file; starting it without an API key."
                    )
                    _key_on = False
                _key_env: dict[str, str] = {}
                if _key_on:
                    self._api_key = _secrets.token_urlsafe(32)
                    # Through a file or the child env, not argv: a command line is readable by every process of this Unix user, and the auth directory is not.
                    _key_argv, _key_env, self._api_key_file = _llama_server_key_launch(
                        self._api_key, self._api_key_file
                    )
                    cmd.extend(_key_argv)
                else:
                    self._api_key = None

                # Windows + full offload: drop the host-RAM KV checkpoints that cause
                # WDDM/PCI-E overhead, but keep prompt caching (in-VRAM prefix reuse) so
                # a repeated prompt is not re-prefilled on every request. #5692.
                # ... unless the offload target IS system RAM. #5692 is a discrete card,
                # whose host-RAM checkpoints cross PCI-E under WDDM, so dropping them is
                # a saving. An iGPU has one pool and no bus, so the flags only remove the
                # prompt cache and every chance of reusing a prefix; --cache-ram 0 takes
                # --cache-idle-slots down with it. Measured on a Strix Halo: a 48.8 h
                # session spent 44.3 h re-ingesting, 3.38 M prompt against 72 k generated.
                # Behind the platform gate on purpose: the helper reads torch, and a
                # launch that never reaches this block would spend a device probe on a
                # question with no consumer.
                #
                # gpu_indices is the picker's answer, not always the child's: with no
                # gpu_ids a user --device survives in the extras and wins last-wins over
                # the generated pin. Decline rather than re-derive the target from argv --
                # the failure that matters is --cache-ram 0 against a shared pool, and
                # keeping the prompt cache on a discrete card only forgoes a tuning.
                # The exact tokens the tuning appended, so the arch-crash respawn can
                # take them back off when it lands on a different device class.
                _cache_flags_emitted: list[str] = []
                # llama.cpp reads the variable before argv, so a 0 appended below would
                # overrule the operator while every estimate still priced their value.
                _ctx_checkpoints_owned = (
                    ctx_checkpoints
                    if ctx_checkpoints is not None
                    else _env_ctx_checkpoints_override()
                )
                # extra_args, not the memory policy's list: that one is built later, and
                # a gpu_ids pin is the case _strip_device_extra_args removes the flag in.
                # The env twin counts too: only an explicit gpu_ids clears
                # LLAMA_ARG_DEVICE, and llama.cpp reads it before argv, so an automatic
                # load can place against a target the generated pin never names.
                _cache_target_unknown = self._cache_tuning_target_unknown(
                    extra_args, _gpu_ids_own_device_flags, os.environ
                )
                _shared_memory_offload = (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and (
                        _cache_target_unknown
                        or self._offload_target_shares_system_memory(
                            is_vulkan_backend = is_vulkan_backend,
                            shared_gpu_ids = _shared_gpu_ids,
                            detected_gpus = _detected_gpus,
                            gpu_indices = gpu_indices,
                        )
                    )
                )
                if _cache_target_unknown and sys.platform == "win32" and full_offload_tuning_active:
                    logger.info(
                        "Keeping the llama-server prompt cache: a user --device in the extra "
                        "arguments overrides the automatic placement, so the offload target "
                        "this tuning would be chosen against is not the one the child gets."
                    )
                elif _shared_memory_offload:
                    logger.info(
                        "Keeping the llama-server prompt cache: this load offloads to a GPU "
                        "that shares system memory, so --cache-ram 0 and --ctx-checkpoints 0 "
                        "would remove prefix reuse without saving a transfer."
                    )
                elif sys.platform == "win32" and full_offload_tuning_active:
                    unsupported_cache_flags: list[str] = []
                    # An explicit control wins over the platform tuning: llama.cpp
                    # is last-wins, so a 0 emitted after it would overrule the panel
                    # while the panel still showed the value that was typed.
                    if cache_ram is not None:
                        pass
                    elif server_caps.get("supports_cache_ram"):
                        _cache_flags_emitted.extend(["--cache-ram", "0"])
                    else:
                        unsupported_cache_flags.append("--cache-ram")
                    if _ctx_checkpoints_owned is not None:
                        pass
                    elif server_caps.get("ctx_checkpoints_flag"):
                        _cache_flags_emitted.extend([str(server_caps["ctx_checkpoints_flag"]), "0"])
                    else:
                        unsupported_cache_flags.append("--ctx-checkpoints")
                    cmd.extend(_cache_flags_emitted)
                    if unsupported_cache_flags:
                        logger.info(
                            "Skipping unsupported Windows cache flags for llama-server: %s",
                            ", ".join(unsupported_cache_flags),
                        )

                def _decide_auto_ctx_checkpoints() -> Optional[int]:
                    """The cap for the slots n_parallel currently names, or None to stand down."""
                    return (
                        self._bounded_ctx_checkpoints(
                            n_parallel,
                            server_caps,
                            extra_args,
                            cache_type_kv = cache_type_kv,
                            swa_full = swa_full,
                            flash_attn = planned_flash_attn,
                        )
                        if ctx_checkpoints is None
                        else None
                    )

                _auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()
                # Tracked apart from _cache_flags_emitted, which means "the Windows tuning
                # ran", so the arch-crash respawn can re-decide each one on its own.
                _auto_ckpt_emitted: list[str] = []

                def _emit_auto_ctx_checkpoints(command: list[str]) -> list[str]:
                    """Append the automatic cap unless the Windows tuning already zeroed it."""
                    if _auto_ctx_checkpoints is None:
                        return []
                    if any(
                        _flag_name(str(token)) in _CTX_CHECKPOINTS_FLAGS
                        for token in _cache_flags_emitted
                    ):
                        return []
                    pair = [
                        str(server_caps["ctx_checkpoints_flag"]),
                        str(_auto_ctx_checkpoints),
                    ]
                    command.extend(pair)
                    return pair

                _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(cmd)

                # Record the pin actually applied (fit-narrowed gpu_indices, else the raw
                # request) for the keep-warm loop, dedupe, and /status, so an explicit
                # [0, 1] narrowed to [0] records [0] and /status never echoes an ordinal
                # the child never saw. Auto selection (no gpu_ids) stays None (#7239).
                if _paravirtual_cpu_forced:
                    # --device none pins nothing, so record no effective pin here either.
                    # Ahead of the backend branches: this is the authoritative record and
                    # would otherwise re-echo the device the child never saw.
                    self._gpu_ids = None
                elif is_vulkan_backend:
                    # Only record an EXPLICIT Vulkan pin: an auto pick still narrows +
                    # pins below, but recording it would misreport an explicit pin and
                    # make dedupe miss the loaded server; mirrors the CUDA/ROCm branch.
                    self._gpu_ids = (
                        [int(x) for x in _vulkan_pin_ids] if (gpu_ids and _vulkan_pin_ids) else None
                    )
                elif gpu_ids:
                    # Physical pin: the fit-selected subset when the fit ran, else the raw
                    # user selection so an explicit choice is honoured even when the fit
                    # could not size the model.
                    _effective_pin_ids = (
                        [int(x) for x in gpu_indices]
                        if gpu_indices is not None
                        else [int(x) for x in gpu_ids]
                    )
                    # Survivor order, not sorted. This is the authoritative record and it
                    # overwrites the one above, so sorting here put the order back and
                    # /status reported a pin a client could round-trip into a match
                    # against the opposite order.
                    self._gpu_ids = (
                        [int(x) for x in _effective_pin_ids] if _effective_pin_ids else None
                    )
                else:
                    self._gpu_ids = None

                # Also record the RAW requested pin (before the fit narrowed it). Load
                # dedupe compares this so a [0, 1] narrowed to [0] and re-sent as [0, 1]
                # still matches, while /status keeps echoing the effective pin (#7239).
                self._requested_gpu_ids = [int(x) for x in gpu_ids] if gpu_ids else None

                if is_vulkan_backend and _vulkan_pin_ids is not None:
                    cmd += LlamaCppBackend._vulkan_pin_args(_vulkan_pin_ids)

                # Model Memory settings. Managed flag lands before the extras, and
                # vetoed ones are dropped from a launch-only copy: extra_args keeps
                # None-vs-[] (inherit vs clear) for the commit block below, and the
                # user's saved flags survive turning the toggle back off.
                # mlock pins a mapping in HOST RAM, so it only implements
                # residency where the weights actually live there: unified
                # memory, or any load that is not fully offloaded. On a discrete
                # GPU with full offload it would hold a second copy of the model
                # in system RAM and do nothing for VRAM, so it is not emitted
                # and the idle-unload veto carries residency by itself.
                # fully_gpu_offloaded is only set by the auto branch. Manual mode
                # and a user -ngl reach the same placement by their own routes,
                # so derive those too or they would still be pinned. The CPU
                # placement guard applies to BOTH: an auto fit that offloads
                # every layer still leaves weights in RAM if an extra pins them
                # there, so it cannot short-circuit past the check.
                # Asked on EVERY launch, not just when a lock is on the table:
                # _memory_mlock_applicable is recorded from it, and a LATER
                # "keep resident" save is compared against that. Assuming host
                # residency by default made enabling the toggle demand a reload
                # and reject the duplicate-load fast path, tearing down a healthy
                # server to relaunch identical argv. Only the Vulkan probe stays
                # gated, so the default path still spawns nothing.
                # The same view of the environment the child will get: manual
                # mode drops the placement vars below, so reading os.environ
                # would pin for a CPU-MoE setting the child never sees.
                _mem_env = dict(os.environ)
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(_mem_env)
                # A gpu_ids pin drops the device flags from argv and env below,
                # so classifying on them would pin for a --device cpu the child
                # never sees. Same helpers the launch uses, so they cannot drift.
                _mem_extra_args = extra_args
                if _gpu_ids_own_device_flags:
                    _mem_extra_args = self._strip_device_extra_args(extra_args)
                    self._clear_device_placement_env(_mem_env)
                _mem_keep_resident, _mem_no_reserve = _mem_settings
                _mem_should_mlock = _mem_keep_resident and not _mem_no_reserve
                # Armed HERE, not at the load call: that also spans the Hub download, and
                # rows from before it price the fit against VRAM since allocated.
                _arm_load_probe_memo()
                _mem_dio_possible = no_reserve_requires_dio(
                    supports_load_mode = bool(server_caps.get("supports_load_mode")),
                    gpu_offload_confirmed = True,
                )
                # NOT gated on the toggle: the answer is this launch's recorded placement
                # and a LATER save is compared against it, so gating made the record mean
                # "we did not look". Costs one subprocess, on Windows Vulkan builds only.
                _mem_probe_for_dio = _mem_dio_possible
                # The ordinals the CHILD will use, resolved once for every consumer: a
                # pass-through `--device` wins, so an override onto a unified-memory APU
                # was otherwise priced against the discrete card it replaced. Physical for
                # the classifier, compact for the build's device list. One selection.
                _mem_physical_indices, _mem_compact_indices = self._effective_gpu_selection(
                    binary,
                    _mem_env,
                    gpu_indices,
                    _mem_extra_args,
                    _mem_dio_possible,
                    is_vulkan_backend,
                )
                _mem_host_resident = self._weights_in_host_memory(
                    fully_gpu_offloaded = fully_gpu_offloaded,
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    extra_args = _mem_extra_args,
                    gpu_indices = _mem_physical_indices,
                    is_vulkan_backend = is_vulkan_backend,
                    binary = binary,
                    env = _mem_env,
                    # _mem_should_mlock is always False under no-reserve, so gating on it
                    # alone made the DirectIO branch unreachable on the Vulkan build.
                    probe_vulkan = _mem_should_mlock or _mem_probe_for_dio,
                    # Over the built cmd AND the extras, so Unsloth's own --fit
                    # counts and a later user --fit still wins by last-arg.
                    fit_active = fit_is_effectively_on([*cmd, *(_mem_extra_args or [])], _mem_env),
                )
                # The gate masks every device away, so the child runs from host RAM
                # whatever was asked, while _weights_in_host_memory answered for the
                # ORIGINAL placement. Without this the page-lock is skipped and
                # _memory_mlock_applicable records that as deliberate, which no
                # relaunch undoes.
                if _arch_gate_forced_cpu:
                    _mem_host_resident = True
                # Moved up: the managed DirectIO defers to an inherited mode and runs first.
                _fit_load_mode_env_view = dict(_mem_env)
                scrub_memory_env(_fit_load_mode_env_view, _mem_settings)
                # The view as it would be under no-reserve: the scrub is what the SETTINGS
                # do, so asking against the live view answered for the wrong toggle.
                _mem_env_view_no_reserve = dict(_mem_env)
                scrub_memory_env(_mem_env_view_no_reserve, (_mem_keep_resident, True))
                # POSITIVE, not "not host-resident": that predicate only gates skipping a
                # page-lock, so it errs True for an unprobed device.
                # NOT gated on where the projector lands: mtmd_context_params has no
                # use_mmap or load_mode, and clip.cpp reads the mmproj through its own
                # ifstream, so gating on it only withheld dio from the weights that respond.
                _mem_gpu_offload_confirmed = self._gpu_offload_confirmed(
                    binary,
                    _mem_env,
                    _mem_compact_indices,
                    _mem_host_resident,
                    _mem_dio_possible,
                    _mem_extra_args,
                )
                _mem_managed, _mem_extras = apply_model_memory_policy(
                    extra_args,
                    supports_load_mode = bool(server_caps.get("supports_load_mode")),
                    weights_in_host_memory = _mem_host_resident,
                    gpu_offload_confirmed = _mem_gpu_offload_confirmed,
                    env = _fit_load_mode_env_view,
                    settings = _mem_settings,
                )
                # What the user wrote, so the strip can tell their pair from ours.
                self._memory_dio_user_tokens = list(_mem_extras or [])
                self._memory_dio_flags = (
                    list(_mem_managed) if tuple(_mem_managed) == MANAGED_DIO_FLAGS else []
                )
                # After Model Memory and on the extras it returned, because it
                # defers to those settings: while either one owns host placement
                # the per-model pick emits nothing at all.
                #
                # The fit's "none" only fills in for a model with no pick of its own,
                # and sits here rather than inside the policy so the whole precedence
                # chain stays in one place: the per-model mode wins over the fit, Model
                # Memory wins over both, and a hand-typed flag still wins by last-arg.
                #
                # An INHERITED loader mode is an explicit choice too: llama.cpp reads
                # LLAMA_ARG_LOAD_MODE (and the deprecated twins) before argv and both
                # assign the same params.load_mode, so the managed block below would
                # beat one silently. Only the FIT's mode stands aside, since a per-model
                # pick is a choice made for THIS model. Asked AFTER the same scrub the
                # child gets, so a var a Model Memory toggle drops vetoes nothing.
                if (
                    _fit_load_mode
                    and not load_mode
                    and memory_env_selects_load_mode(_fit_load_mode_env_view)
                ):
                    logger.info(
                        "Load mode: the environment already selects a loader mode; "
                        "leaving the fit's --load-mode %s off this launch.",
                        _fit_load_mode,
                    )
                    _fit_load_mode = None
                # And a mode the EXTRAS pick. Their flag already wins by last-arg, but
                # the retry rungs below strip the pair by VALUE and take the FIRST match
                # (_without_subsequence), so a user who typed the same "--load-mode none"
                # leaves two identical pairs: one rung removes the fit's and records the
                # stripped argv, and the next, respawning from it, removes theirs.
                # Standing aside costs nothing and leaves every strip below the no-op
                # its docstring already claims to be.
                if _fit_load_mode and not load_mode and extra_args_select_load_mode(_mem_extras):
                    logger.info(
                        "Load mode: the extra arguments already select a loader mode; "
                        "leaving the fit's --load-mode %s off this launch.",
                        _fit_load_mode,
                    )
                    _fit_load_mode = None
                _resolved_load_mode = load_mode or _fit_load_mode
                _load_mode_managed, _mem_extras = apply_load_mode_policy(
                    _mem_extras,
                    supports_load_mode = bool(server_caps.get("supports_load_mode")),
                    weights_in_host_memory = _mem_host_resident,
                    requested_load_mode = _resolved_load_mode,
                    settings = _mem_settings,
                )

                # Whether a relaunch under no-reserve would really stream. Placement alone
                # is not enough: a per-model mmap, an extra or an inherited choice lands
                # after the managed pair and wins by last-arg, so it demanded a reload no
                # relaunch could satisfy. Toggle-independent, so a LATER save is compared
                # against this launch instead of reading as satisfied.
                def _ask(pair, env_view):
                    return resolve_launch_load_mode(
                        extra_args,
                        supports_load_mode = bool(server_caps.get("supports_load_mode")),
                        weights_in_host_memory = _mem_host_resident,
                        gpu_offload_confirmed = _mem_gpu_offload_confirmed,
                        requested_load_mode = _resolved_load_mode,
                        env = env_view,
                        settings = pair,
                    )

                # Would a relaunch ADD the pair. The POLICY emitting it is the question,
                # not the effective mode: a user's own "dio" already streams, and a reload
                # to reach the mode the child is in is a notice that never clears.
                _hypo_emitted, _hypo_effective = _ask(
                    (_mem_keep_resident, True), _mem_env_view_no_reserve
                )
                self._memory_dio_applicable = _hypo_emitted and _hypo_effective

                # The child with the policy off entirely, which is what activity asks.
                _off_view = dict(_mem_env)
                scrub_memory_env(_off_view, (False, False))

                def _mem_env_for(child_env):
                    """`_mem_env`'s placement scrubbing with the CHILD's current
                    visibility. `_mem_env` predates the gate, so probing with it reused
                    the unnarrowed set's verdict."""
                    if child_env is None:
                        return _mem_env
                    view = dict(_mem_env)
                    for _name in _DEVICE_VISIBILITY_ENV:
                        if _name in child_env:
                            view[_name] = child_env[_name]
                        else:
                            view.pop(_name, None)
                    return view

                def _dio_decision_for(
                    devices,
                    *,
                    fully_offloaded,
                    child_env = None,
                ):
                    """``(pair, applicable, active)`` for a CHANGED device set.

                    Everything the launch above asks, asked again for the devices a
                    rung has just narrowed to: the host-residency verdict, the
                    backend and probe confirmation, the live and forced-on answers,
                    and the toggles-off comparison that decides activity. The rungs
                    used to ask a subset each, and every missing piece became its own
                    review round: a partial offload getting the pair, a narrowed set
                    never gaining it, a redundant pair recorded as activity.
                    """
                    _rung_env = _mem_env_for(child_env)
                    _rung_physical, _rung_compact = self._effective_gpu_selection(
                        binary,
                        _rung_env,
                        devices,
                        _mem_extra_args,
                        _mem_dio_possible,
                        is_vulkan_backend,
                    )
                    host_resident = self._weights_in_host_memory(
                        fully_gpu_offloaded = fully_offloaded,
                        gpu_memory_mode = gpu_memory_mode,
                        gpu_layers = gpu_layers,
                        extra_args = _mem_extra_args,
                        gpu_indices = _rung_physical,
                        is_vulkan_backend = is_vulkan_backend,
                        binary = binary,
                        env = _rung_env,
                        probe_vulkan = _mem_probe_for_dio or _mem_should_mlock,
                        fit_active = fit_is_effectively_on(
                            [*cmd, *(_mem_extra_args or [])], _rung_env
                        ),
                    )
                    confirmed = self._gpu_offload_confirmed(
                        binary,
                        _rung_env,
                        _rung_compact,
                        host_resident,
                        _mem_dio_possible,
                        _mem_extra_args,
                    )

                    def _for(pair, env_view):
                        return resolve_launch_load_mode(
                            extra_args,
                            supports_load_mode = bool(server_caps.get("supports_load_mode")),
                            weights_in_host_memory = host_resident,
                            gpu_offload_confirmed = confirmed,
                            requested_load_mode = _resolved_load_mode,
                            env = env_view,
                            settings = pair,
                        )

                    live_emitted, live_effective = _for(_mem_settings, _fit_load_mode_env_view)
                    hypo_emitted, hypo_effective = _for(
                        (_mem_keep_resident, True), _mem_env_view_no_reserve
                    )
                    pair = list(MANAGED_DIO_FLAGS) if (live_emitted and live_effective) else []
                    # Redundant with the user's own loader: nothing a relaunch could undo.
                    active = bool(pair) and live_effective != _for((False, False), _off_view)[1]
                    return pair, (hypo_emitted and hypo_effective), active

                # Only when the FIT chose it: a user's own pick survives every fallback
                # below, but a conclusion about a placement has to go when that
                # placement does.
                self._fit_load_mode_flags = (
                    list(_load_mode_managed) if (_fit_load_mode and not load_mode) else []
                )
                # Remembered so the reload hint and the duplicate-load comparator do
                # not demand an mlock this launch deliberately skipped.
                # A probe forced for the DirectIO decision must not make the page-lock
                # bookkeeping LESS conservative: an unreadable Vulkan probe answers "not an
                # iGPU", which recorded the lock inapplicable on maybe-host-backed weights.
                self._memory_mlock_applicable = _mem_host_resident or (
                    _mem_probe_for_dio
                    and not _mem_should_mlock
                    and is_vulkan_backend
                    and not self._vulkan_probe_answered(binary)
                )
                if _mem_should_mlock and not _mem_host_resident:
                    logger.info(
                        "Model Memory: skipping page-lock, the weights are fully "
                        "offloaded to a discrete GPU; residency is kept by not "
                        "unloading the model."
                    )
                if _mem_managed:
                    cmd.extend(_mem_managed)
                    logger.info(
                        "Model Memory: applying loading policy (%s)",
                        " ".join(_mem_managed),
                    )
                if _load_mode_managed:
                    cmd.extend(_load_mode_managed)
                    # Protected like the extras: the per-model selection is the user's too,
                    # so a rung that stripped a COPY left it behind for the next strip.
                    self._memory_dio_user_tokens = [
                        *self._memory_dio_user_tokens,
                        *_load_mode_managed,
                    ]
                    logger.info("Load mode: %s", " ".join(_load_mode_managed))

                # User pass-through args go last. Placement flags are removed
                # below when the Unsloth picker owns the GPU selection.
                if _mem_extras:
                    _emit_extra_args = self._strip_mmproj_args(list(_mem_extras))
                    if _gpu_ids_own_device_flags:
                        # gpu_ids owns placement, so remove competing device flags.
                        _before_device_strip = list(_emit_extra_args)
                        _emit_extra_args = self._strip_device_extra_args(_emit_extra_args)
                        if _emit_extra_args != _before_device_strip:
                            logger.info(
                                "Dropped user device placement flags from extra args: "
                                "explicit gpu_ids owns placement."
                            )
                    cmd.extend(str(a) for a in _emit_extra_args)
                    logger.info(
                        f"Appending user extra args to llama-server: {list(_emit_extra_args)}"
                    )
                # Last so it wins: the drafter CPU pin on a virtualised Metal device
                # is a correctness fix, not a preference, and llama.cpp is last-wins.
                if _pv_draft_cpu_pin:
                    cmd.extend(_pv_draft_cpu_pin)
                if _pv_mmproj_cpu_pin:
                    cmd.extend(_pv_mmproj_cpu_pin)
                # Also last, and for the same last-wins reason: suppressing Unsloth's own
                # --mmproj and scrubbing the env vars still leaves a remembered
                # --mmproj-auto in the extras above, and that flag asks llama-server to
                # rediscover the adjacent projector by itself. Vision would come back on
                # a load that reports it off and whose fit budget never charged the
                # projector's VRAM. Only when the projector really was suppressed. Both
                # ways of keeping an audio-only one are excluded: the resolved path keeps
                # launch_mmproj_path set, and an inherited LLAMA_ARG_MMPROJ keeps neither
                # that nor anything else this gate can see, so it needs the same predicate
                # the env scrub used. --no-mmproj-auto does not actually unload either of
                # them -- server-context.cpp gates the load on a non-empty mmproj.path and
                # never reads no_mmproj, which only suppresses the HF auto-download and
                # the router's capability report -- but advertising a model as text-only
                # while loading its encoder is its own bug, and one flag away from a real
                # one if that gate ever starts reading no_mmproj.
                if disable_vision and not launch_mmproj_path and not _dv_env_mmproj_kept:
                    cmd.append("--no-mmproj-auto")
                # Also last: _zero_offload_keeps_gpu_visible reads the finished cmd, so
                # the override must be in it before the env block below judges this a
                # zero-VRAM server.
                if _pv_split_mode_pin:
                    cmd.extend(_pv_split_mode_pin)
                if _pv_device_pin:
                    cmd.extend(_pv_device_pin)
                # Pinned loads only: under mmap the cache measured slower than none.
                self._moe_cache_flags = []
                if (
                    _moe_cache_candidate
                    and _moe_cache_ram_ok
                    and _placed_by_fitter
                    and _argv_loads_pinned(cmd, _fit_load_mode_env_view)
                ):
                    self._moe_cache_flags = list(_MOE_CACHE_AUTO_TOKENS)
                    cmd.extend(self._moe_cache_flags)
                    logger.info(
                        "MoE experts spill to pinned system RAM on one GPU; adding "
                        "--moe-cache-mib auto so llama.cpp caches hot experts in VRAM."
                    )

                # A build with no --flash-attn cannot run a quantized V cache
                # either: llama.cpp aborts init with "V cache quantization
                # requires flash_attn", and the KV type is emitted from the
                # user's setting with no flash-attn coupling. The crash-recovery
                # rung resets V for exactly that abort, but it only fires when
                # the argv had a flag to turn off, so on a build that never had
                # one nothing would catch it. Before the log below, so what is
                # logged is what launches; length is preserved, so _spec_start
                # still indexes the same tokens.
                if _flash_attn_known_off:
                    cmd = self._reset_quantized_v_cache(
                        cmd,
                        "this build has no --flash-attn",
                        mla = self._target_kv_symmetry(),
                        # No env argument: the child environment is not built until
                        # below, and reading it here is an UnboundLocalError. The
                        # inherited os.environ is the right source anyway, since
                        # Unsloth never rewrites LLAMA_ARG_SPEC_DRAFT_MODEL.
                        draft_mla = self._draft_kv_symmetry(cmd),
                    )

                kv_cache_unified = _kv_unified_from_args(cmd)

                logger.info(f"Starting llama-server: {' '.join(self._redacted_cmd_for_log(cmd))}")

                # Library paths so llama-server finds its shared libs and CUDA DLLs.
                env = self._llama_server_env_for_binary(binary)
                # The GGUF header controls pooling; ignore inherited defaults.
                for _pooling_var in (
                    "LLAMA_ARG_POOLING",
                    "LLAMA_ARG_RERANKING",
                    "LLAMA_ARG_EMBEDDINGS",
                ):
                    env.pop(_pooling_var, None)
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(env)
                # llama.cpp reads LLAMA_ARG_MLOCK / _MMAP / _LOAD_MODE before argv,
                # so stripping the tokens alone would leave an inherited value in
                # force. Only when a toggle is on, so existing setups are untouched.
                # _mem_settings, not the live toggles: a save landing in between scrubbed
                # the env under the new pair while the argv came from the old one.
                _mem_scrubbed = scrub_memory_env(env, _mem_settings)
                if _mem_scrubbed:
                    logger.info(
                        "Model Memory owns placement; dropped inherited %s",
                        ", ".join(_mem_scrubbed),
                    )
                # Same reasoning one level up: a flag validate_extra_args refuses has
                # an env twin llama.cpp reads before argv, so denying the token alone
                # would leave the capability reachable and unrecorded.
                _denied_scrubbed = scrub_denied_env(env)
                if _denied_scrubbed:
                    logger.info(
                        "dropped inherited %s: managed by Unsloth Studio",
                        ", ".join(_denied_scrubbed),
                    )
                # After the scrub, which drops an inherited LLAMA_API_KEY.
                env.update(_key_env)
                # Record what the child will ACTUALLY run with (env defaults plus
                # last-wins argv), not just what Unsloth emitted, so the reload
                # hint also catches a user-supplied --mlock / --no-mmap.
                #
                # In argv order, including the load-mode pair: "none" holds a full
                # anonymous host copy (llama-model-loader.cpp sets use_mmap only for
                # mmap/mmap+mlock/auto), so leaving it out records the launch as
                # non-reserving and a later "Don't reserve system RAM" is judged already
                # satisfied, keeping the reservation instead of relaunching with mmap.
                self._record_memory_state(
                    list(_mem_managed) + list(_load_mode_managed) + list(_mem_extras), env
                )
                # Did the policy change this launch at all: emitted a flag,
                # suppressed a requested one, or scrubbed an inherited env var.
                # Turning both toggles off must relaunch to undo any of those,
                # but must not fight a launch it never touched.
                # Split out: the fit-off retry can drop _mem_managed, and what is
                # left decides whether that child still differs from an unmanaged one.
                _mem_policy_touched_extras = bool(_mem_scrubbed) or _mem_extras != list(
                    extra_args or []
                )
                # Activity is "would turning the toggles off change this child".
                # bool(_mem_managed) is not: a later mmap shadows the pair and a user's own
                # "dio" makes it redundant, both leaving the child equal to an unmanaged
                # one, so counting either made turning no-reserve OFF demand an inert
                # reload. The keep-resident block cannot be shadowed.
                _mem_managed_is_effective = bool(_mem_managed) and (
                    tuple(_mem_managed) != MANAGED_DIO_FLAGS
                    or self._memory_direct_io != _ask((False, False), _off_view)[1]
                )
                self._memory_policy_active = _mem_managed_is_effective or _mem_policy_touched_extras
                self._memory_policy_extras_touched = _mem_policy_touched_extras

                # What `cmd` itself means, snapshotted before any respawn edits it.
                # The dio TOKENS, not just applicability: the fit-off retry records a pair
                # `cmd` never carried, and a fallback respawning `cmd` read that as
                # "already has it". Copied, so a later strip cannot reach back in.
                # _spawn_and_wait's --fit retries append a page-lock to THEIR argv
                # and write the policy back; the arch-crash retry (#7624) respawns
                # `cmd`, which never carried that lock, so it restores these.
                def _snapshot_policy_for_cmd():
                    """What `cmd` means RIGHT NOW, for the arch-crash retry to restore.

                    A function because every site that mutates `cmd`'s managed pair has
                    to retake it, and the proactive gate did not: the retry restored a
                    pre-gate snapshot over a post-gate command, appending a second pair
                    after an addition or believing an absent one was still there."""
                    return (
                        _mem_host_resident,
                        self._memory_state,
                        self._memory_direct_io,
                        self._memory_dio_applicable,
                        list(self._memory_dio_flags),
                        self._memory_policy_active,
                        self._memory_mlock_applicable,
                    )

                _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                # Omitting --threads relies on llama.cpp's physical-core default, so
                # drop an inherited LLAMA_ARG_THREADS that would otherwise feed the
                # arg handler and silently force hardware_concurrency(). #5692
                if "--threads" not in cmd:
                    env.pop("LLAMA_ARG_THREADS", None)
                # Same shape, for the context. Auto-layers omits -c so --fit can size
                # it, and with no -c to win the env-then-argv race an inherited
                # LLAMA_ARG_CTX_SIZE=0 pins the native length and the fit never runs.
                # Only a zero, and only where a pass-through zero would go too, so a
                # positive inherited context stays the legitimate way to set one.
                if (
                    not any(token in cmd for token in ("-c", "--ctx-size"))
                    and _env_asks_for_the_native_context(env)
                    and self._metal_drops_zero_ctx_override(0, _caller_owns_budget)
                ):
                    env.pop("LLAMA_ARG_CTX_SIZE", None)
                    logger.warning(
                        "Dropping the inherited LLAMA_ARG_CTX_SIZE=0: on Metal it pins "
                        "the model's native length and disables --fit."
                    )
                # A build whose help has no --flash-attn never registers the option,
                # so it never reads LLAMA_ARG_FLASH_ATTN either (llama.cpp resolves
                # each env var through the arg that declares it). Leaving an
                # inherited one in place would only fool the accounting below, which
                # reads env plus argv and would then record flash attention as on for
                # a server running without it, under-sizing the padded V cache.
                if _flash_attn_known_off and self._drop_env_flash_attn(env):
                    logger.info(
                        "Dropped inherited LLAMA_ARG_FLASH_ATTN: this build has no --flash-attn."
                    )
                # Same for an env-only quantized V cache, which the argv reset
                # above cannot reach.
                if _flash_attn_known_off and self._drop_env_quantized_v_cache(
                    env,
                    mla = self._target_kv_symmetry(),
                    draft_mla = self._draft_kv_symmetry(cmd, env),
                ):
                    logger.info(
                        "Dropped inherited quantized V-cache env: this build has no --flash-attn."
                    )

                # An inherited type kv_cache_type_from_str cannot parse aborts the child
                # at argument parsing, and the layer retry inherits the same env, so
                # BOTH attempts fail and the user is left with no server. The managed
                # path has long been normalised and allow-listed (_VALID_CACHE_TYPES);
                # the env path only stopped being scrubbed once the tensor gate went
                # away, so give it the same two protections. Split-mode independent,
                # since llama.cpp parses it identically either way.
                for _ct_var in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"):
                    # Compared against the ORIGINAL, not a stripped copy:
                    # common_arg::get_value_from_env hands the raw getenv string to
                    # kv_cache_type_from_str, which does an exact ggml_type_name(t) == s
                    # compare and throws otherwise. So " q8_0 " is as fatal as a typo,
                    # and normalising against an already-stripped copy would compare
                    # equal, write nothing back, and leave the whitespace on the child.
                    _ct_orig = env.get(_ct_var) or ""
                    _ct_norm = _ct_orig.strip().lower()
                    if not _ct_norm:
                        continue
                    if _ct_norm not in _VALID_CACHE_TYPES:
                        env.pop(_ct_var, None)
                        logger.info(
                            "Dropped inherited %s=%s: llama.cpp does not accept that "
                            "KV cache type, and it would abort the load.",
                            _ct_var,
                            _ct_orig,
                        )
                    elif _ct_norm != _ct_orig:
                        # Case and surrounding whitespace both matter to that
                        # compare, so "Q8_0" and " q8_0 " are rewritten alike.
                        env[_ct_var] = _ct_norm

                # Reconcile the inherited LLAMA_ARG_* env with Unsloth's final
                # decision: stripping CLI extras on a tensor->layer downgrade
                # can't remove env vars, so the child could run a mode Unsloth
                # didn't budget.
                if not tensor_parallel:
                    # Layer split: clear a non-layer inherited split mode (and any
                    # paired tensor-split) so the child can't override the layer plan.
                    _inherited_sm = (env.get("LLAMA_ARG_SPLIT_MODE") or "").strip().lower()
                    if _inherited_sm and _inherited_sm != "layer":
                        env.pop("LLAMA_ARG_SPLIT_MODE", None)
                        env.pop("LLAMA_ARG_TENSOR_SPLIT", None)
                else:
                    # Unsloth owns the tensor split: it emits --tensor-split when it
                    # picks an uneven one (CLI wins) and nothing when an even split
                    # is safe. Clear any inherited LLAMA_ARG_TENSOR_SPLIT so the even
                    # case can't be overridden by a stale env (the layer branch above
                    # clears it too).
                    env.pop("LLAMA_ARG_TENSOR_SPLIT", None)

                # A gpu_ids pin also owns inherited device placement.
                if _gpu_ids_own_device_flags:
                    self._clear_device_placement_env(env)

                # After the scrubs above, so a placement this launch already dropped
                # cannot disqualify the replay of a child that never saw it.
                _cpu_fallback_eligible = self._auto_vulkan_cpu_fallback_eligible(
                    binary,
                    intent,
                    extra_args,
                    env,
                )

                # The unpinnable-drafter drop above rewrites argv, which cannot reach the
                # drafter the child picks up from its own env. The spec type goes too,
                # for the same reason the CLI one did: llama.cpp appends types rather
                # than replacing them, so an inherited draft-simple would outlive the
                # model it needs. Unsloth owns the spec block whenever the extras do not
                # name a --spec-type, and nothing _build_speculative_flags emits can undo
                # an inherited LLAMA_ARG_SPEC_TYPE (env applied first, then appended), so
                # a managed non-MTP launch would still start MTP, a crash-recovery replay
                # could not drop it, and the fit never budgeted the drafter the env adds.
                # Clearing it makes the emitted flags final. A user-owned --spec-type is
                # left alone: their flags and the env accumulate, which is what the slot
                # clamp above reads.
                if _pv_draft_unpinnable or not _extra_args_set_spec_type(extra_args):
                    for _pv_spec_var in _SPEC_ENV_VARS:
                        env.pop(_pv_spec_var, None)

                # The projector guard has a worse blind spot: it only clears Unsloth's
                # resolved path, so an inherited LLAMA_ARG_MMPROJ loads a projector the
                # guard believes it dropped, unpinned and independent of --gpu-layers 0.
                # LLAMA_ARG_MMPROJ_URL goes too: its download overwrites mmproj.path, so
                # it outranks even the --mmproj Unsloth emits. Unsloth always passes its
                # own projector on the command line (with --no-mmproj-offload), so an
                # inherited one is only ever the corrupt path.
                # Turning vision off has the same blind spot, and it is the whole point
                # of the switch: suppressing Unsloth's own --mmproj leaves an inherited
                # LLAMA_ARG_MMPROJ / LLAMA_ARG_MMPROJ_URL untouched, so llama-server
                # loads a projector anyway (arg.cpp sets params.mmproj.path / .url
                # straight from those vars). The load would then report text-only over a
                # server that is still multimodal, and the fit budget that gave the
                # projector's bytes to context is short by exactly those bytes.
                if disable_vision:
                    # With the same audio-only exception the resolved path gets, and for
                    # the same reason: a projector is not always a vision tower, so
                    # clearing an inherited ultravox / Voxtral / Qwen3-ASR encoder takes
                    # the model's audio input away and gives no image VRAM back. Asked of
                    # the file rather than assumed, and unknown still reads image-capable,
                    # so the switch is honored wherever it might mean anything.
                    _dv_env_mmproj = (env.get("LLAMA_ARG_MMPROJ") or "").strip()
                    if not _mmproj_env_is_audio_only(_dv_env_mmproj):
                        # Same signal the resolved path records, for the same reason:
                        # this one IS restored by turning Vision back on, so the composer
                        # must point at the switch rather than at a missing mmproj. Only
                        # for a file that exists -- a stale path drops nothing, and
                        # blaming the switch there sends the user to a control that
                        # cannot help. Unknown-but-present reads image-capable, matching
                        # the rule the scrub itself just used.
                        if _dv_env_mmproj and Path(_dv_env_mmproj).is_file():
                            _dv_dropped_image_projector = True
                        env.pop("LLAMA_ARG_MMPROJ", None)
                    # No such reprieve for the URL: it names a download that has not
                    # happened, so there is no file to classify and nothing to do but
                    # honor the switch.
                    env.pop("LLAMA_ARG_MMPROJ_URL", None)
                if _paravirtual_cpu_forced:
                    for _pv_mmproj_var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
                        env.pop(_pv_mmproj_var, None)
                    # LLAMA_ARG_OVERRIDE_TENSOR is the one placement var --gpu-layers 0
                    # cannot answer: applied during weight-buft selection, before any
                    # layer is assigned, so it puts weights back on the corrupt Metal
                    # buffer on its own. Same CPU-target rule as the argv strip, so an
                    # inherited exps=CPU override keeps working and only GPU-bound goes.
                    _pv_env_ot = env.get("LLAMA_ARG_OVERRIDE_TENSOR")
                    if _pv_env_ot and not _paravirtual_strip_gpu_overrides(
                        ["-ot", str(_pv_env_ot)]
                    ):
                        env.pop("LLAMA_ARG_OVERRIDE_TENSOR", None)

                # Windows + full offload: PASSIVE OMP + 2 threads stop
                # spin-wait burning CPU. CPU/partial offload keeps default
                # OMP parallelism. #5692.
                if sys.platform == "win32" and full_offload_tuning_active:
                    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
                    if not threads_overridden:
                        env.setdefault("OMP_NUM_THREADS", "2")

                # Managed allocations trade ROCm APU carve-out capacity for host RAM.
                # Vulkan indices are not ROCm ids. Track ownership so later retries
                # preserve inherited values.
                _unified_env_applied = False
                _unified_opt_out = self._unified_memory_opted_out(env)
                _unified_opt_in = self._unified_memory_opted_in(env)
                # The arch gate and arch-crash retry re-select without rebinding gpu_indices.
                _unified_gpu_indices = gpu_indices

                def _unified_need_now(argv = None, mtp_engages = None):
                    """Bytes a forced full offload puts on the device, or None."""
                    run_argv = cmd if argv is None else argv
                    # The trailing blocks load only when the EMBEDDED head drafts: a
                    # sidecar wins over it (llama.cpp has_dft()), wherever it lands.
                    if mtp_engages is None:
                        engages = _mtp_will_engage and not _separate_draft_launches
                    else:
                        engages = mtp_engages
                    if model_size is None or not self._launch_forces_full_offload(run_argv, env):
                        return None
                    need = int(model_size)
                    if mmproj_size and not _argv_keeps_projector_on_gpu(run_argv, env):
                        need -= int(mmproj_size)
                    # dev_input is CPU-pinned even at full offload, so token_embd never
                    # reaches the device; a tied file duplicates the matrix onto it instead.
                    layout = self._tensor_spill_layout(model_path, all_shards = True)
                    if layout is None or not getattr(layout, "complete", True):
                        return None
                    if int(getattr(layout, "lm_head_bytes", 0) or 0):
                        need -= int(getattr(layout, "token_embd_bytes", 0) or 0)
                    if not engages and self._nextn_predict_layers:
                        need -= int(getattr(layout, "excluded_block_bytes", 0) or 0)
                    return need

                def _unified_withdraw_if_unneeded(
                    run_cmd,
                    *,
                    mtp_engages = None,
                    why = "",
                ):
                    """Drop the variable THIS launch set once a respawn stops needing it."""
                    nonlocal _unified_env_applied
                    # Before the withdrawal test and outside it: every caller is a retry
                    # whose argv differs from what the advice was priced against, and
                    # dropping a projector or the MTP blocks can take the footprint back
                    # under the carve-out. Re-priced rather than cleared, so a spill that
                    # still stands is still reported. Computed here rather than in the
                    # argument list, which is evaluated OUTSIDE the recorder's try.
                    try:
                        _carveout_need = _unified_need_now(argv = run_cmd, mtp_engages = mtp_engages)
                    except Exception:
                        _carveout_need = None
                    self._record_carveout_advice(
                        _unified_gpu_indices,
                        _carveout_need,
                        is_vulkan_backend = is_vulkan_backend,
                        shared_gpu_ids = _shared_gpu_ids,
                        detected_gpus = _detected_gpus,
                        target_unknown = _cache_target_unknown,
                        forced_cpu = _arch_gate_forced_cpu,
                    )
                    if not _unified_env_applied:
                        return
                    if self._unified_memory_for_launch(
                        _unified_gpu_indices,
                        _unified_need_now(argv = run_cmd, mtp_engages = mtp_engages),
                        opted_in = _unified_opt_in,
                    ):
                        return
                    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
                    _unified_env_applied = False
                    logger.info(
                        "%s no longer needs managed memory; dropped "
                        "GGML_CUDA_ENABLE_UNIFIED_MEMORY.",
                        why or "The retry",
                    )

                _unified_need = _unified_need_now()
                if _unified_opt_out:
                    # ggml tests presence, so passing a user's "0" through would
                    # ENABLE what they turned off (#8651). Only absence is off.
                    if env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None) is not None:
                        logger.info(
                            "Unified memory opted out: unset GGML_CUDA_ENABLE_UNIFIED_MEMORY"
                        )
                elif not is_vulkan_backend and self._unified_memory_for_launch(
                    gpu_indices, _unified_need, opted_in = _unified_opt_in
                ):
                    _unified_env_applied = "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
                    env.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")
                    logger.info(
                        "AMD unified-memory APU: set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 (%s)",
                        "UNSLOTH_ENABLE_UNIFIED_MEMORY=1"
                        if _unified_opt_in
                        else "the weights outgrow the carve-out and host RAM is the larger pool",
                    )

                # Whether the user could stop the spill existing at all, independent of
                # the managed-memory decision above, which only copes with one that is.
                # The placement facts go with it: the index space gpu_indices is in, and
                # whether a user --device makes the child's target unknowable.
                self._record_carveout_advice(
                    gpu_indices,
                    _unified_need,
                    is_vulkan_backend = is_vulkan_backend,
                    shared_gpu_ids = _shared_gpu_ids,
                    detected_gpus = _detected_gpus,
                    target_unknown = _cache_target_unknown,
                    forced_cpu = _arch_gate_forced_cpu,
                )

                # An inherited GGML_CUDA_P2P=0 means OFF to the user and ON to
                # llama.cpp, which tests presence and not value (#10613).
                # _llama_server_env_for_binary already stripped it for every backend;
                # this covers an env assembled another way.
                self._sanitize_p2p_env(env)

                # A multi-GPU box that is NOT a datacenter part never reaches the
                # block below, so warn here or not at all: the 2x RTX 3090 case,
                # whose own truthy GGML_CUDA_P2P rides through to the child.
                # Computed here because a deliberate zero-offload load is masked to
                # CUDA_VISIBLE_DEVICES=-1 and moves no tensors between GPUs, so
                # probing or warning it is noise about traffic it never generates.
                _cpu_only_zero_offload = (
                    gpu_memory_mode == "manual"
                    and gpu_layers == 0
                    and not is_vulkan_backend
                    and not self._zero_offload_keeps_gpu_visible(cmd, env)
                )

                # Whether the child will resolve gpu_indices as the nvidia-smi ids
                # the P2P gate verifies. The launch below pins PCI_BUS_ID for an
                # explicit pick, and for an auto-fit selection only when NO mask was
                # inherited (nothing to reinterpret); anything else leaves CUDA on
                # FASTEST_FIRST, where those ids may name other cards.
                # That auto-fit arm also requires the ids to BE PCI indices: when the
                # nvidia-smi query fails _get_gpu_memory falls back to torch CUDA
                # ordinals, and pinning PCI order re-emits those numbers as a
                # different set of cards. An explicit pick is unaffected, its ids
                # come from the UI's own PCI-ordered picker.
                _p2p_launch_order_pinned = bool(gpu_ids) or (
                    os.environ.get("CUDA_VISIBLE_DEVICES") is None
                    and LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is True
                )
                # Whether THESE ids live in the matrix's index space. An explicit pick
                # is not evidence: with nvidia-smi gone the picker lists torch ordinals.
                # Ask instead: nvidia-smi produced the ids, or PCI order is pinned.
                _p2p_ids_are_pci = (
                    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is True
                    or os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
                )

                # Only when the fabric is NOT confirmed: on a verified NV# pair the
                # flag is the benchmarked configuration. Datacenter boxes are
                # excluded because _apply_datacenter_env warns about the same
                # variable from its own veto branch.
                if (
                    env.get("GGML_CUDA_P2P")
                    and not _cpu_only_zero_offload
                    and not is_vulkan_backend
                    and not self._is_datacenter_gpu(gpu_indices)
                    and self._effective_gpu_count(gpu_indices) > 1
                    and not LlamaCppBackend._warned_no_nvlink
                ):
                    _p2p_veto = self._p2p_veto_reason(
                        gpu_indices, _p2p_launch_order_pinned, _p2p_ids_are_pci
                    )
                    if _p2p_veto is not None:
                        LlamaCppBackend._warned_no_nvlink = True
                        logger.warning(
                            "GGML_CUDA_P2P=%s is set in the environment but this "
                            "selection has no confirmed NVLink fabric: %s. Peer copies "
                            "over PCIe can be discarded silently while CUDA still "
                            "reports success, which surfaces as garbled output rather "
                            "than an error. Verify with scripts/p2p_integrity_probe.py "
                            "and unset the variable entirely (not =0, which reads as ON "
                            "upstream) if it fails (#10613).",
                            env["GGML_CUDA_P2P"],
                            _p2p_veto,
                        )

                # DC NVIDIA GPUs: FP32 accum and launch queues, plus P2P once a
                # multi-GPU selection has a CONFIRMED NVLink fabric (#10613). Opt
                # out with UNSLOTH_DISABLE_DC_TUNING=1, or UNSLOTH_DISABLE_DC_P2P=1
                # for the peer flag alone.
                if not (is_vulkan_backend or _cpu_only_zero_offload):
                    self._apply_datacenter_env(
                        env,
                        gpu_indices,
                        p2p_opted_out = self._p2p_user_opted_out(),
                        launch_order_pinned = _p2p_launch_order_pinned,
                        ids_are_pci_indices = _p2p_ids_are_pci,
                    )

                # Pin to selected GPU(s) (issue #7164; resolved above into gpu_indices).
                # On ROCm, narrowing only CUDA_VISIBLE_DEVICES leaves the AMD child
                # seeing the full set, so set HIP_VISIBLE_DEVICES too. Vulkan is pinned
                # via --device (above), not here.
                # A deliberate zero-offload load with no GPU companions runs entirely
                # on CPU, yet a visible CUDA device still costs the child ~0.5 GB
                # (context + compute scratch) that the CPU-only classification below
                # reports as free. Hide the GPUs so the load is exactly what it
                # claims. Companion loads keep the normal masking, and a user device
                # pin (extras or an inherited LLAMA_ARG_DEVICE) keeps control of its
                # own devices, the child aborting on a pin it cannot see. The
                # draft-device forms count too: parsed with no drafter loaded.
                _child_gpu_physical_ids: Optional[tuple[int, ...]] = None
                # companion flags fitted to the child's mask, refitted by the arch-crash retry
                _companion_fit_mask: Optional[list[int]] = None
                _companion_added_device: Optional[list[str]] = None
                if not is_vulkan_backend and _gpu_mem:
                    _child_gpu_physical_ids = self._unmasked_child_gpu_physical_ids()

                # The CUDA SM gate goes here, where the child's GPU visibility is
                # finally known: both arms below write CUDA_VISIBLE_DEVICES=-1, so no
                # kernel image loads and the abort this pre-empts cannot happen.
                # Deciding it from the request over-refused -- speculative "auto", a
                # --no-mmproj-offload projector and a --spec-draft-ngl 0 drafter all
                # read as GPU-bearing there, only the resolved argv knows. Still
                # before every spawn below and outside the fit's try/except, so the
                # refusal cannot be swallowed into the --fit on fallback.
                # Pinned ids are checked against nvidia-smi's caps only in its index space
                # (torch-fallback ordinals are not); otherwise "any visible", as before.
                _sm_gate_ids_are_pci = not is_vulkan_backend and _p2p_ids_are_pci
                if not (_cpu_only_zero_offload or _arch_gate_forced_cpu):
                    # Pinned launches check the pinned cards, not "any visible".
                    _sm_gate = self._cuda_sm_gate_error(
                        binary, gpu_indices if _sm_gate_ids_are_pci else None
                    )
                    if _sm_gate:
                        raise RuntimeError(_sm_gate)
                # The arch gate emptying the pool lands here for the same reason
                # (#7624): no device set this binary can launch on, so the child must
                # not see the cards it would enumerate and abort on. gpu_indices is
                # None there, so this branch writes the mask.
                if _cpu_only_zero_offload or _arch_gate_forced_cpu:
                    self._emit_child_gpu_visibility(env, "-1")
                    _child_gpu_physical_ids = ()
                    # Manual mode admits tensor parallelism on the FULL device count,
                    # which still counts the dropped cards, so --split-mode tensor can
                    # reach a child that sees none. llama_prepare_model_devices then
                    # fails with "LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices" and
                    # llama-server exits 1 rather than falling back, so normalise to
                    # layer/CPU. Gate-scoped: the zero-offload arm already dropped
                    # Unsloth's flags, and a user --split-mode there is overridden.
                    _cpu_cmd = (
                        self._without_flags(cmd, ("--split-mode", "-sm", "--tensor-split", "-ts"))
                        if _arch_gate_forced_cpu
                        else None
                    )
                    if _cpu_cmd is not None:
                        logger.info(
                            "Forced-CPU launch: dropped the split-mode/tensor-split "
                            "flags, which abort a server with no visible device."
                        )
                        cmd = _cpu_cmd
                        # Recorded, or the next Apply reads the normalized server as a
                        # different one and rebuilds a multi-GB model every time.
                        if self._tensor_parallel:
                            self._arch_gate_dropped_tensor_parallel = True
                        if self._tensor_split:
                            self._arch_gate_dropped_tensor_split = tuple(self._tensor_split)
                        self._tensor_parallel = False
                        self._tensor_split = None
                    if _arch_gate_forced_cpu:
                        # Not gated on the argv strip: llama.cpp reads the LLAMA_ARG_*
                        # twins whether or not we emitted the flags, so an inherited
                        # tensor mode aborts a child with no visible device on its own.
                        self._clear_split_placement_env(env)
                elif gpu_indices is not None and not is_vulkan_backend:
                    # When the user picked GPUs by index, align CUDA's ordering with
                    # the PCI-bus order the picker enumerated, so "GPU 1" in the UI is
                    # GPU 1 to llama.cpp, not CUDA's FASTEST_FIRST default (#5025).
                    # Extended to an auto-fit selection ONLY when the inherited mask
                    # is ABSENT: a scheduler's CUDA_VISIBLE_DEVICES=0,1 meaning
                    # physical 2,0 would be re-read as physical 0,1 once the order
                    # changes, running on a device hidden on purpose. With no mask
                    # there is nothing to preserve, and the one emitted below is our
                    # own nvidia-smi indices, which pinning is what makes true. Where
                    # we cannot pin, _p2p_veto_reason withholds the flag.
                    if _p2p_launch_order_pinned:
                        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    # The set is ours to pin, but the ORDER is the user's to choose,
                    # and gpu_indices is ascending at every producer. Keep a mask that
                    # deliberately reorders the same cards.
                    _pin_ids = list(gpu_indices)
                    # A dragged picker order is the more explicit and more recent of
                    # the two, so it wins over whatever the environment was masked to.
                    _inherited_order = (
                        None if gpu_ids else self._inherited_child_gpu_order(_pin_ids)
                    )
                    if _inherited_order is not None:
                        # A user --tensor-split is positional over the order they
                        # expected, so reordering under it re-weights the wrong cards.
                        # Theirs to own: decline instead of rewriting it.
                        if _extra_args_have_tensor_split(extra_args, env):
                            logger.info(
                                "Keeping ascending GPU order: a pass-through "
                                "--tensor-split is positional over it."
                            )
                        elif self._repoint_emitted_tensor_split(cmd, _pin_ids, _inherited_order):
                            logger.info(
                                "Pinning the child's GPU order to the inherited "
                                "visibility mask: %s (ascending would be %s).",
                                _inherited_order,
                                _pin_ids,
                            )
                            _pin_ids = _inherited_order
                            # The emitted fingerprint was recorded from the pre-rewrite
                            # value, so /status would pair each share with the wrong
                            # device and the repeat check would compare against a split
                            # the child never ran.
                            if "--tensor-split" in cmd and gpu_memory_mode != "manual":
                                _rewritten = cmd[cmd.index("--tensor-split") + 1]
                                try:
                                    self._auto_tensor_split_emitted = self._auto_split_fingerprint(
                                        [float(x) for x in str(_rewritten).split(",")]
                                    )
                                except (TypeError, ValueError):
                                    self._auto_tensor_split_emitted = None
                    # Mask on AMD at the ROCr/HSA layer: HIP-only masking still
                    # enumerates every agent first, which segfaults on a deselected
                    # unsupported GPU (e.g. gfx1036 iGPU under a gfx103X prebuilt).
                    # Companion flags use the unpinned numbering (#11810). Skipped for a
                    # uuid/MIG mask (CUDA<n> is ambiguous) and an inherited LLAMA_ARG_DEVICE
                    # (the appended --device would override it).
                    _companion_widen = ""
                    if not self._visibility_mask_is_unmappable() and not (
                        _extra_args_main_device(cmd) is None
                        and str(env.get("LLAMA_ARG_DEVICE", "")).strip()
                    ):
                        _had_main_device = _extra_args_main_device(cmd) is not None
                        _pin_ids, _companion_widen = _widen_pin_ids_for_companion_devices(
                            cmd,
                            _pin_ids,
                            self._resolve_visible_physical_ids(),
                            allowed_ids = gpu_ids or None,
                        )
                        if _companion_widen:
                            _companion_fit_mask = list(_pin_ids)
                            if not _had_main_device and _extra_args_main_device(cmd) is not None:
                                _companion_added_device = list(cmd[-2:])
                            if gpu_ids:
                                self._adopt_widened_pin(_pin_ids)
                    if _companion_widen:
                        logger.info(
                            "Companion device flags name GPUs by their unpinned "
                            "numbering; fitted the pinned mask to them: %s",
                            _companion_widen,
                        )
                        # The widened mask can add a card the gate above never saw.
                        if _sm_gate_ids_are_pci:
                            _sm_gate = self._cuda_sm_gate_error(binary, _pin_ids)
                            if _sm_gate:
                                raise RuntimeError(_sm_gate)
                    self._emit_child_gpu_visibility(
                        env,
                        LlamaCppBackend._child_visibility_for(_pin_ids),
                        prefer_rocr = True,
                    )
                    _child_gpu_physical_ids = tuple(int(i) for i in _pin_ids)
                    _launch_pinned_ids = list(_pin_ids)
                elif not is_vulkan_backend and not gpu_ids:
                    # Nothing above pinned the child. Two shapes land here and the gate
                    # breaks both the same way, so they share one probe: `--fit on`
                    # owning placement, and a manual per-GPU ratio across ALL GPUs.
                    # Either way gpu_indices is None, the reactive retry's guard cannot
                    # fire, and an uncovered card reaches the child. gpu_ids is falsy
                    # by construction, an explicit pick taking the arm above.
                    _survivors = self._arch_gate_survivors(binary)
                    if _survivors:
                        # The manual ratio was validated against the FULL visible
                        # count and masking re-indexes the survivors under it -- the
                        # positional mismatch _without_tensor_split exists for. Drop it
                        # and let llama.cpp split the survivors by free VRAM.
                        _gated_cmd = self._without_tensor_split(cmd)
                        if _gated_cmd is not None:
                            logger.warning(
                                "Dropping the manual --tensor-split: it was sized "
                                "for every visible GPU, and the installed build "
                                "has no kernels for some of them."
                            )
                            cmd = _gated_cmd
                            # The argv lost the ratio but the request still carries
                            # it and the next Apply re-sends it verbatim. Record it, or
                            # the duplicate-load check reads the normalized server as a
                            # different one and rebuilds it on every Apply.
                            self._arch_gate_dropped_tensor_split = (
                                tuple(self._tensor_split) if self._tensor_split else None
                            )
                            self._tensor_split = None
                        # Narrowing to one survivor leaves --split-mode tensor a no-op
                        # still REPORTED as active (tensor_parallel drives the UI and
                        # arms the MTP crash watchdog), manual mode having admitted it
                        # on the FULL device count. _without_tensor_split takes only the
                        # ratio, so drop the mode too, as the arch-crash retry does.
                        if len(_survivors) < 2 and self._tensor_parallel:
                            _no_tensor_mode = self._without_flags(cmd, ("--split-mode", "-sm"))
                            if _no_tensor_mode is not None:
                                cmd = _no_tensor_mode
                            self._arch_gate_dropped_tensor_parallel = True
                            self._tensor_parallel = False
                            logger.info(
                                "The arch gate leaves one GPU; dropped tensor "
                                "parallelism, which needs at least two."
                            )
                        logger.info(
                            "The installed llama.cpp build has no kernels for GPU(s) "
                            "left visible; pinning the child to %s.",
                            _survivors,
                        )
                        # The argv strips above cannot reach an inherited split
                        # mode or ratio, and the mask re-indexes the survivors
                        # under both.
                        self._clear_split_placement_env(env)
                        # The setting is process-wide, so recompute it after changing
                        # the selected devices. Ownership protects inherited values.
                        _unified_gpu_indices = _survivors
                        _survivors_gain_unified = self._unified_memory_for_launch(
                            _survivors, _unified_need_now(), opted_in = _unified_opt_in
                        )
                        if _unified_env_applied and not _survivors_gain_unified:
                            env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
                            _unified_env_applied = False
                            logger.info(
                                "Arch gate narrowed the launch onto GPU(s) the "
                                "unified-memory swap cannot help; dropped "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY."
                            )
                        elif (
                            _survivors_gain_unified
                            and not _unified_opt_out
                            and "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
                        ):
                            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
                            _unified_env_applied = True
                            logger.info(
                                "Arch gate narrowed the launch onto a unified-memory "
                                "APU the weights outgrow; set "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1."
                            )
                        self._emit_child_gpu_visibility(
                            env, ",".join(str(i) for i in _survivors), prefer_rocr = True
                        )
                        _child_gpu_physical_ids = tuple(int(i) for i in _survivors)
                        # Narrower than any pin above, so it replaces it.
                        _launch_pinned_ids = list(_survivors)
                        # Taken against the UNNARROWED set, where an unsupported APU
                        # answered host-resident for the whole launch; this gate removes
                        # it. fully_offloaded False: the fitter still owns placement.
                        _gate_dio, _gate_applicable, _gate_active = _dio_decision_for(
                            _survivors, fully_offloaded = False, child_env = env
                        )
                        self._memory_dio_applicable = _gate_applicable
                        # BOTH directions: narrowing can also take the offload away, and
                        # the pair then reads CPU-resident layers into allocated buffers.
                        if self._memory_dio_flags and not _gate_dio:
                            cmd = self._drop_managed_dio(
                                cmd,
                                "the arch gate's surviving GPU(s) no longer confirm a "
                                "full offload",
                            )
                            self._memory_dio_applicable = _gate_applicable
                            self._record_memory_state(cmd, env)
                            _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                        elif _gate_dio and not self._memory_dio_flags:
                            cmd = [*cmd, *_gate_dio]
                            self._memory_dio_flags = list(_gate_dio)
                            self._memory_policy_active = _gate_active or self._memory_policy_active
                            self._record_memory_state(cmd, env)
                            _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                            logger.info(
                                "Model Memory: applying %s; the arch gate pins this "
                                "launch to discrete GPU(s) %s.",
                                " ".join(_gate_dio),
                                _survivors,
                            )
                        # And the carve-out advice with it: upstream priced the
                        # UNNARROWED set, which _rocm_selected_pool_mib declines on a
                        # mixed host, so a model outgrowing the surviving APU's
                        # carve-out was never advised about.
                        try:
                            _gated_carveout_need = _unified_need_now(argv = cmd)
                        except Exception:
                            _gated_carveout_need = None
                        self._record_carveout_advice(
                            _survivors,
                            _gated_carveout_need,
                            is_vulkan_backend = is_vulkan_backend,
                            shared_gpu_ids = _shared_gpu_ids,
                            detected_gpus = _detected_gpus,
                            target_unknown = _cache_target_unknown,
                            forced_cpu = _arch_gate_forced_cpu,
                        )
                    elif manual_tensor_split_emitted:
                        # A manual per-GPU ratio across ALL GPUs (no explicit pick, so
                        # no mask above): the UI built --tensor-split in ascending
                        # physical/PCI order, so pin the child's enumeration to match.
                        # The whole visible set stays in use, only its order is fixed.
                        _child_gpu_physical_ids = self._pin_visible_gpu_order_for_split(env)

                _child_has_no_gpu = (
                    # each names a device present but unusable, so it owns the empty pool
                    _arch_gate_forced_cpu
                    or _paravirtual_cpu_forced
                    # neither says a device exists, so an empty pool stays unreadable
                    or (
                        bool(_detected_gpus)
                        and (
                            _cpu_only_zero_offload or self._binary_ships_no_gpu_backend(binary, env)
                        )
                    )
                )
                # The host pool as it stands before ANYTHING spawns, kept for the Vulkan
                # CPU replay to re-price against. That replay only runs once a CPU-only
                # child is healthy and holding the whole model in system RAM, where a
                # live reading is the pool minus the very model being priced. Taken
                # here, and handed to the guard below as well, so the two verdicts
                # differ in exactly one term -- the VRAM credit the replay loses -- and
                # never in the pool they were read against. Only on the path that can
                # use it: a load with no CPU replay available pays for no probe, and
                # None restores the guard's own live reading.
                _preflight_avail_mib = (
                    self._available_system_memory_mib() if _cpu_fallback_eligible else None
                )
                # Only the cards this child can reach. VRAM on a card this launch
                # pinned away is not VRAM it can spend: crediting it prices the spill
                # too low and leaves an unmapped oversized load unrewritten, which is
                # an OOM rather than the slow load the override exists to produce.
                _reachable_gpus = self._rows_the_child_can_reach(_detected_gpus, _launch_pinned_ids)
                # reads the argv the child gets, not the mid-fit state that produced it
                _offload_msg = self._launch_host_shortfall_message(
                    cmd,
                    _reachable_gpus,
                    env,
                    child_has_no_gpu = _child_has_no_gpu,
                    avail_mib = _preflight_avail_mib,
                    shared_gpu_ids = _shared_gpu_ids,
                )
                # Snapshotted BEFORE the override note is appended below, so a re-price
                # that promotes this notice appends that note once rather than twice.
                _host_ram_msg = _offload_msg

                def _launch_is_oversized() -> bool:
                    """Whether SOMETHING about to launch does not fit, warning aside.

                    Three readings, because one alone misses a shape the override has
                    to cover. The APU preflight, whose shortfall this guard cannot see:
                    _shared_gpu_ids covers Vulkan only, so a ROCm APU's reported pool is
                    credited here as if it were dedicated VRAM and the spill prices out
                    at zero. The warning just produced, which is the discrete case. And,
                    only when the deprecated opt-out silenced that warning, the same
                    pricing with the opt-out ignored: the variable's contract is that it
                    hides a message, and hiding a message must not withdraw the rewrite
                    that makes the load possible.
                    """
                    if _apu_ram_oversized or _offload_msg:
                        return True
                    if not self._host_offload_warning_opted_out():
                        return False
                    return bool(
                        self._launch_host_shortfall_message(
                            cmd,
                            _reachable_gpus,
                            env,
                            child_has_no_gpu = _child_has_no_gpu,
                            shared_gpu_ids = _shared_gpu_ids,
                            honor_opt_out = False,
                        )
                    )

                cmd, _pageable_note = self._page_an_oversized_unmapped_load(
                    cmd, env, _launch_is_oversized
                )
                if _pageable_note:
                    # Only onto a warning that is actually emitted; when the opt-out
                    # suppressed it the override still went to the log, which is what
                    # keeps a silenced load traceable.
                    if _offload_msg:
                        _offload_msg += _pageable_note
                    # And onto whatever is already recorded, which is not the same
                    # thing. First notice wins, so when the APU preflight also warned
                    # its text is what the route hands back and the _offload_msg
                    # amended just above is dropped by _record_load_warning below --
                    # leaving memory_warning silent about an override that did change
                    # the child's argv. A no-op when nothing was recorded, including a
                    # load the opt-out silenced, and it cannot double-append: the note
                    # reaches at most one of the two strings.
                    self._amend_load_warning(_pageable_note)
                    # The recorded state has to describe the argv that launches, or the
                    # reload comparator judges this child against a mode it no longer
                    # runs. The snapshot too: the arch-crash retry restores it over
                    # `cmd`, which now carries the override.
                    self._record_memory_state(cmd, env)
                    _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                self._record_load_warning(_offload_msg)

                # Captured before any text-only fallback strips it from cmd.
                launched_with_mmproj = "--mmproj" in cmd

                # One-shot --fit off retry: recent llama.cpp runs a "fitting
                # params to device memory" step by default (--fit defaults to
                # 'on') even when -ngl is explicit. That step has aborted on
                # some ROCm hosts (ggml-cuda.cu ROCm error during worst-case
                # estimation, e.g. MTP + mmproj models on gfx1151). When
                # Unsloth's own VRAM math already placed the model
                # (use_fit=False), the step is redundant second-guessing --
                # retry once with --fit off before declaring the load failed.
                # Never retry when fit was requested (use_fit) or the caller
                # passed an explicit fit flag via extra args.
                # Argv actually launched (post --fit off / MTP); text-only retry strips this.
                _last_spawn_cmd = list(cmd)
                _spawn_cwd: Optional[str] = None
                # Load-scoped, not per call: the correction edits the shared env,
                # so it outlives this spawn, and the launch that finally comes up
                # healthy is often a later one (drafterless, no-flash, arch
                # fallback). A per-call flag left those successes unrecorded.
                _did_rocm_retry = False

                def _enable_managed_fit_for_no_flash(fa_cmd: list) -> list:
                    """Unsloth's own ``--fit off`` back to ``on`` for a --flash-attn off respawn.

                    The reserve is only safe because the respawn gets re-placed, and it
                    inherits a managed ``--fit off`` that `_fit_off_retry_eligible` will not
                    override, so without this it OOMs on the smaller-cache placement. Skipped
                    where the flip buys nothing: a user's own ``--fit``, or a fixed count.
                    """
                    if "--fit" not in fa_cmd:
                        return fa_cmd
                    if any(
                        _flag_name(str(token)) in {"-fit", "--fit"} for token in (extra_args or ())
                    ):
                        return fa_cmd
                    # An inherited LLAMA_ARG_FIT=off is the user's too, and only manual mode
                    # scrubs it: the flip would override it on the CLI, and the reserve
                    # _user_fit_disabled already took prices this respawn where it lands.
                    if _user_fit_disabled(extra_args, env = env):
                        return fa_cmd
                    if _placement_is_fitter_proof(fa_cmd, gpu_layers = _manual_gpu_layers, env = env):
                        return fa_cmd
                    index = fa_cmd.index("--fit")
                    if index + 1 >= len(fa_cmd) or str(fa_cmd[index + 1]).strip().lower() == "on":
                        return fa_cmd
                    flipped = list(fa_cmd)
                    flipped[index + 1] = "on"
                    logger.info(
                        "Re-enabling the fitter for the --flash-attn off retry: the padded "
                        "V cache the retry runs with is bigger than the placement the fit "
                        "priced with flash attention on."
                    )
                    return flipped

                def _drop_fit_load_mode_for_no_flash(fa_cmd: list) -> list:
                    """The fit's ``--load-mode none`` off a --flash-attn off respawn.

                    The FA-off rewrite changes the very footprint the fit priced. On an
                    MLA model it takes the K cache down to f16 with V, because llama.cpp
                    rejects a split K/V there before it rejects a quantized V without
                    flash attention (llama-context.cpp checks type_k == type_v first),
                    and the MLA branch of the KV estimator prices that latent cache at
                    the K width alone. The non-flash attention path also materialises a
                    full [n_kv, n_ubatch, n_head] F32 KQ tensor (llama-graph.cpp
                    build_attn_mha) that the flash-attn-shaped compute-buffer estimate
                    does not model. Under an active fitter the child answers a bigger
                    requirement by offloading fewer layers and spilling more weights to
                    host RAM, where "--load-mode none" makes them anonymous rather than
                    file-backed: an OOM kill where the mapping would only have paged.

                    Only Unsloth's own tokens; a user's --load-mode is theirs. The
                    record is NOT cleared, for the same reason the --fit on retry keeps
                    it: the CPU fallback below still respawns from an argv that carries
                    these tokens and has to be able to name them. Stripping an
                    already-stripped argv is a no-op.
                    """
                    if not self._fit_load_mode_flags:
                        return fa_cmd
                    stripped = _without_subsequence(fa_cmd, self._fit_load_mode_flags)
                    if stripped != fa_cmd:
                        # The record follows the argv: without "none" the child is back
                        # on llama.cpp's auto, which maps, so a record still saying
                        # "reserving" spends a full reload on the next "Don't reserve
                        # system RAM". From the argv, not _mem_managed + _mem_extras:
                        # this descends from _last_spawn_cmd, which carries any page-lock
                        # the --fit on retry appended, and rebuilding would drop it.
                        self._record_memory_state(stripped, env)
                        logger.info(
                            "Load mode: dropping the fit's --load-mode none for the "
                            "--flash-attn off retry; the no-flash launch has a "
                            "footprint the fit did not price."
                        )
                    return stripped

                # Memoed only when the load ends terminally, so a recovering fallback is not blocked.
                _sched_abort_seen = False

                def _spawn_and_wait(run_cmd, *, label = ""):
                    """Start llama-server with run_cmd and wait for health.

                    Up to four launches: the first, one ROCm env correction, one
                    retry without the MoE expert cache, and one --fit recovery.
                    Separate flags, so a library mix does not spend the fit slot
                    and a VRAM crash after the correction can still fit-retry.
                    """
                    # _mem_host_resident too: the --fit on retry re-arms the
                    # page-lock and writes it back, which without this makes the
                    # read below an UnboundLocalError instead.
                    nonlocal _last_spawn_cmd, _mem_host_resident, _did_rocm_retry
                    nonlocal _sched_abort_seen
                    # One revocation point for the tensor-spill plan instead of a
                    # strip per retry site. `label` is empty ONLY on the first
                    # spawn, so a retry added later is covered automatically. The
                    # plan's arithmetic assumed this launch's devices, cache dtype
                    # and projector placement, and each retry changes one of them.
                    if label:
                        run_cmd = self._drop_tensor_spill(run_cmd, label.lstrip("-") or "retry")
                    # The cache only rides the first pinned launch: every retry re-places.
                    if label or not _argv_loads_pinned(run_cmd, env):
                        run_cmd = self._drop_moe_cache(
                            run_cmd, f"{label.lstrip('-') or 'unpinned'} launch"
                        )
                    _fit_retry_allowed = self._fit_off_retry_eligible(run_cmd, use_fit)
                    _did_fit_retry = False
                    _did_moe_cache_retry = False
                    for _spawn_attempt in (0, 1, 2, 3):
                        # Defensive kill: drop an orphan Popen a concurrent load may
                        # have stored before we overwrite the reference (#5161).
                        # Also reaps the crashed first attempt on the retry pass.
                        self._kill_process()
                        self._child_gpu_physical_ids = _child_gpu_physical_ids

                        self._stdout_lines = []
                        # Tee llama-server output to a dedicated log file so a
                        # post-mortem has the full trail even when the parent only
                        # kept the last 50 lines. Path is under the studio home.
                        # ``label`` (MTP fallback) and the attempt index (--fit
                        # off retry) keep a respawn within the same epoch second
                        # from truncating the crash log a retry warning just
                        # pointed the user at.
                        self._llama_log_fh = None
                        try:
                            log_dir = _swa_cache_path().parent / "logs" / "llama-server"
                            log_dir.mkdir(parents = True, exist_ok = True)
                            self._llama_log_path = log_dir / (
                                f"llama-{int(time.time())}{label}-port-{self._port}"
                                f"-try{_spawn_attempt}.log"
                            )
                            self._llama_log_fh = open(
                                self._llama_log_path,
                                "w",
                                encoding = "utf-8",
                                buffering = 1,
                            )
                            # After the open, so the cap counts this file and never deletes
                            # it. On the retry path the earlier attempts' logs are what a
                            # reader wants, and keep-newest holds on to them.
                            prune_log_dir(log_dir, "llama-*.log", protect = self._llama_log_path)
                            logger.info(f"llama-server stdout/stderr -> {self._llama_log_path}")
                        except (OSError, UnicodeDecodeError) as e:
                            # Best-effort; never block the load on logging.
                            logger.warning(f"Could not open llama-server log file: {e}")
                            self._llama_log_path = None
                        _last_spawn_cmd = list(run_cmd)
                        # Read off the argv actually spawned rather than the intent, so
                        # every emitter is covered by construction.
                        self._idle_slot_clearing_active = _idle_slot_clearing_active(
                            run_cmd,
                            supports_cache_ram = bool(server_caps.get("supports_cache_ram")),
                        )
                        # Check with publication under one lock: a spawn either
                        # publishes first and the sweep kills it, or sees the flag
                        # and never starts. Across Popen only, never the wait.
                        with self._spawn_lock:
                            if self._spawn_is_stale():
                                logger.info("app is shutting down; not starting llama-server")
                                self._close_attempt_log()
                                self._health_wait_cancelled = True
                                return False
                            _spawned = _spawn_on_lifetime_thread(
                                lambda: subprocess.Popen(
                                    run_cmd,
                                    stdout = subprocess.PIPE,
                                    stderr = subprocess.STDOUT,
                                    text = True,
                                    encoding = "utf-8",
                                    errors = "replace",
                                    env = env,
                                    cwd = _spawn_cwd,
                                    **_windows_hidden_subprocess_kwargs(),
                                    **_child_popen_kwargs(),
                                )
                            )
                            self._process = _spawned
                            _invalidate_gpu_memory("llama-server spawned")
                            # Inside the lock: written after a sweep reaped the child,
                            # _pid_start_identity yields no start time, and the bare pid
                            # left behind is one a later launch kills blind.
                            self._note_server_pid(_spawned.pid)
                        # mark_process_shutting_down does not take _spawn_lock, so the
                        # check above is not atomic against it for a helper-owned
                        # backend. Without this recheck a child spawned in that gap sits
                        # outside the completed sweep for the whole health wait.
                        if self._spawn_is_stale():
                            logger.info(
                                "shutdown began during the spawn; killing the new llama-server"
                            )
                            self._kill_process()
                            self._close_attempt_log()
                            self._health_wait_cancelled = True
                            return False
                        # is_active covers it from here, so drop the pre-spawn flag.
                        # Re-armed below if this attempt does not come up: the retry rungs
                        # respawn from the SAME captured settings.
                        self._memory_pending_launch = None

                        # Background thread to drain stdout (prevents pipe deadlock)
                        self._stdout_thread = threading.Thread(
                            target = self._drain_stdout, daemon = True, name = "llama-stdout"
                        )
                        self._stdout_thread.start()
                        if self._wait_for_health(
                            timeout = self._HEALTH_STALL_TIMEOUT_S, cancelled = _load_cancelled
                        ):
                            if _did_rocm_retry:
                                # Proved on this host: every later child skips the
                                # prepend instead of crashing into it first.
                                self._remember_bundle_only_rocm(binary)
                            return True
                        if getattr(self, "_health_wait_cancelled", False):
                            return False
                        # The window is open again: the rungs below respawn from the SAME
                        # captured settings while a crashed child leaves is_active false.
                        # Here rather than per rung, so a later rung cannot forget it.
                        self._memory_pending_launch = _mem_settings
                        # Read once, like the wait itself: a cleared reference is a
                        # teardown, not a startup crash, and re-reading the
                        # attribute per term would race the shutdown thread again.
                        _crashed_proc = self._process
                        _startup_crashed = (
                            _crashed_proc is not None
                            and _crashed_proc.poll() is not None
                            and _crashed_proc.returncode != 0
                        )
                        # A split-axis abort (#6415) is fit-independent: skip the
                        # --fit off retry and let the caller latch it. So is a
                        # pre-b9455 build refusing a quantized KV cache in tensor mode:
                        # a capability the binary lacks, where re-spawning to let it
                        # offload spends a second full model load on a theory unrelated
                        # to the failure. The caller latches both, so both skip alike.
                        _startup_output = "\n".join(self._stdout_lines[-50:])
                        # Wider tail: the GGML_ASSERT line scrolls past the [New LWP] dump.
                        if _startup_crashed and self._is_sched_reserve_abort(
                            "\n".join(self._stdout_lines[-200:])
                        ):
                            _sched_abort_seen = True
                        # Scan the full buffer: gdb output can bury the assert.
                        _tensor_capability_crash = self._is_tensor_split_assert(
                            "\n".join(self._stdout_lines)
                        ) or self._is_tensor_quant_kv_unsupported(_startup_output)
                        _hip_rocr_mismatch = self._is_bundled_hip_rocr_mismatch(_startup_output)
                        # No fit retry reaches it, and the rung below needs this
                        # launch's argv. Whole buffer: it arrives with a backtrace.
                        _capability_crash = _tensor_capability_crash or self._is_kv_unified_refused(
                            "\n".join(self._stdout_lines)
                        )
                        _capability_crash = _capability_crash or (
                            self._cuda_kernel_image_error("\n".join(self._stdout_lines)) is not None
                        )
                        if (
                            not _did_rocm_retry
                            and _startup_crashed
                            and not _tensor_capability_crash
                            and _hip_rocr_mismatch
                        ):
                            # The prepend is what a shell launch does not do, which
                            # is why the same binary runs from a terminal. Drop
                            # those dirs only: the rest of env holds the pooling
                            # and memory scrubs.
                            _bundle_only = self._llama_server_env_for_binary(
                                binary, use_system_rocm = False
                            )
                            _retry_ld = _bundle_only.get("LD_LIBRARY_PATH", "")
                            # "" on a host that never had the var: nothing to undo.
                            if _retry_ld and _retry_ld != env.get("LD_LIBRARY_PATH"):
                                logger.warning(
                                    "llama-server crashed during startup (exit code %s) "
                                    "because a bundled ROCm library could not resolve a "
                                    "symbol against the system ROCm; retrying once "
                                    "with the bundled runtime only. Crash log: %s",
                                    _crashed_proc.returncode,
                                    self._llama_log_path,
                                )
                                env["LD_LIBRARY_PATH"] = _retry_ld
                                _did_rocm_retry = True
                                continue
                        if (
                            not _did_moe_cache_retry
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            # OOM or a cache error with the cache on: same placement without it.
                            _uncached = self._drop_moe_cache(run_cmd)
                            _crash_text = "\n".join(self._stdout_lines[-200:])
                            if _uncached != run_cmd and (
                                _MOE_CACHE_ERROR_RE.search(_crash_text)
                                or self._is_gpu_memory_start_failure(_crash_text)
                            ):
                                logger.warning(
                                    "llama-server crashed during startup (exit code %s) "
                                    "with --moe-cache-mib auto; retrying once without "
                                    "the MoE expert cache. Crash log: %s",
                                    _crashed_proc.returncode,
                                    self._llama_log_path,
                                )
                                run_cmd = _uncached
                                _did_moe_cache_retry = True
                                continue
                        if (
                            not _did_fit_retry
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            # A spill-planned launch that crashed on startup. The
                            # two recoveries below cannot reach it: the spill arm
                            # leaves fully_gpu_offloaded False and runs under
                            # use_fit=True with an explicit --fit, so
                            # _fit_off_retry_eligible is False on both counts and
                            # the load would skip to the terminal/CPU fallbacks
                            # even though the --fit on placement it replaced would
                            # have worked. A crash means the prediction was wrong,
                            # so hand placement back to llama.cpp. Self-gating:
                            # _drop_tensor_spill returns run_cmd unchanged when no
                            # plan is present.
                            _reverted = self._drop_tensor_spill(run_cmd, "startup failure")
                            if _reverted != run_cmd:
                                logger.warning(
                                    "llama-server crashed during startup (exit code %s) "
                                    "with a tensor-spill plan; the plan was optimistic, "
                                    "retrying once with --fit on so llama.cpp can place "
                                    "the model itself. Crash log: %s",
                                    _crashed_proc.returncode,
                                    self._llama_log_path,
                                )
                                run_cmd = _reverted
                                self._record_memory_state(run_cmd, env)
                                _did_fit_retry = True
                                self._vram_fit_context_length = None
                                continue
                        if (
                            not _did_fit_retry
                            and fully_gpu_offloaded
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            # We forced --fit off because Unsloth's (conservative) VRAM
                            # math placed the model fully on GPU. A startup crash here
                            # means that estimate was optimistic, so fall back to --fit
                            # on and let llama.cpp offload rather than fail the load.
                            # DSpark takes this path too: fitting only skips its
                            # unmeasurable sidecar reserve, the drafter still loads.
                            logger.warning(
                                "llama-server crashed during startup (exit code %s) "
                                "with forced --fit off; the fit estimate was optimistic, "
                                "retrying once with --fit on so it can offload. "
                                "Crash log: %s",
                                _crashed_proc.returncode,
                                self._llama_log_path,
                            )
                            # Flip Unsloth's own --fit off (added first, before any
                            # user extra args) to on; a user's later --fit still wins
                            # by last-arg. Defensive: if absent, the default is already
                            # --fit on, so leave it.
                            _run = list(run_cmd)
                            if "--fit" in _run:
                                _run[_run.index("--fit") + 1] = "on"
                            # Same reasoning as the page-lock below, one step
                            # earlier: the fit picked "none" because it had proved
                            # the load fits, and that proof is what just failed, so
                            # give the retry llama.cpp's auto back. Only Unsloth's
                            # own tokens; a user's --load-mode is untouched.
                            if self._fit_load_mode_flags:
                                _run = _without_subsequence(_run, self._fit_load_mode_flags)
                                # The record is NOT cleared: it describes `cmd`, which
                                # this retry never touched and the arch-crash and CPU
                                # fallbacks respawn from, so clearing here would leave
                                # them carrying a --load-mode none they can no longer
                                # name. Stripping an already-stripped argv is a no-op.
                                logger.info(
                                    "Load mode: dropping the fit's --load-mode none "
                                    "for the --fit on retry; the fit it was derived "
                                    "from did not hold."
                                )
                            # And the managed pair: the offload it was chosen for did
                            # not hold.
                            _run = self._drop_managed_dio(
                                _run,
                                "the --fit on retry gives up the confirmed full offload",
                                clear_record = False,
                            )
                            # The full-offload prediction that suppressed the
                            # page-lock was the very thing that just proved
                            # wrong, so llama.cpp may now leave weights in host
                            # RAM. Re-arm residency for the retry (last-wins, so
                            # appending is enough) and re-record the state.
                            if _mem_should_mlock and not _mem_host_resident:
                                _run.extend(
                                    ["--load-mode", "mmap+mlock"]
                                    if server_caps.get("supports_load_mode")
                                    else ["--mlock"]
                                )
                                _mem_host_resident = True
                                self._memory_mlock_applicable = True
                                self._memory_policy_active = True
                                logger.info(
                                    "Model Memory: re-applying the page-lock for "
                                    "the --fit on retry; the full-offload "
                                    "prediction did not hold."
                                )
                            run_cmd = _run
                            self._record_memory_state(_run, env)
                            _did_fit_retry = True
                            # The fit that vouched for the ceiling is what just failed.
                            self._vram_fit_context_length = None
                            continue
                        if (
                            not _did_fit_retry
                            and _fit_retry_allowed
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            logger.warning(
                                "llama-server crashed during startup (exit code %s) "
                                "with the default memory-fit step enabled; Unsloth "
                                "already verified the model fits, retrying once "
                                "with --fit off. Crash log: %s",
                                _crashed_proc.returncode,
                                self._llama_log_path,
                            )
                            run_cmd = [*run_cmd, "--fit", "off"]
                            # The mirror of the --fit on retry above. Turning the
                            # fitter off leaves -ngl at its default, which llama.cpp
                            # resolves to every layer, so the fitted attempt's
                            # host-resident verdict can stop holding. Drop a
                            # page-lock the retry no longer needs rather than
                            # reserving a full host copy of a fully offloaded model.
                            if _mem_host_resident and not self._weights_in_host_memory(
                                fully_gpu_offloaded = True,
                                gpu_memory_mode = gpu_memory_mode,
                                gpu_layers = gpu_layers,
                                extra_args = _mem_extra_args,
                                gpu_indices = gpu_indices,
                                is_vulkan_backend = is_vulkan_backend,
                                binary = binary,
                                env = _mem_env,
                            ):
                                _mem_host_resident = False
                                # So a later save is not compared against a lock this
                                # launch dropped.
                                self._memory_mlock_applicable = False
                                if _mem_managed:
                                    run_cmd = _without_subsequence(run_cmd, _mem_managed)
                                    # The managed flag was the policy's only mark on
                                    # this child unless it also scrubbed or stripped,
                                    # and a child equal to an unmanaged one must not
                                    # be torn down when the toggles go off.
                                    self._memory_policy_active = _mem_policy_touched_extras
                                    logger.info(
                                        "Model Memory: dropping the page-lock for "
                                        "the --fit off retry; it offloads every layer."
                                    )

                                # The other direction, which the page-lock arm cannot
                                # reach: nothing was emitted BECAUSE the fitted attempt
                                # read host-resident, the verdict this retry overturns.
                                # Appended, so a hand-typed flag still wins last. The
                                # fitter is OFF here, so -ngl is every layer.
                                _retry_dio, _retry_applicable, _retry_active = _dio_decision_for(
                                    gpu_indices, fully_offloaded = True, child_env = env
                                )
                                self._memory_dio_applicable = _retry_applicable
                                if _retry_dio and not self._memory_dio_flags:
                                    run_cmd = [*run_cmd, *_retry_dio]
                                    self._memory_dio_flags = list(_retry_dio)
                                    self._memory_policy_active = (
                                        _retry_active or self._memory_policy_active
                                    )
                                    logger.info(
                                        "Model Memory: applying %s for the --fit off "
                                        "retry; it offloads every layer.",
                                        " ".join(_retry_dio),
                                    )
                                self._record_memory_state(run_cmd, env)
                            _did_fit_retry = True
                            continue
                        return False

                _launched_mmproj_has_audio = self._mmproj_has_audio

                def _raise_terminal_load_failure(detail: str) -> NoReturn:
                    if _sched_abort_seen and not _load_cancelled():
                        LlamaCppBackend._record_sched_reserve_abort(binary, _abort_memo_model)
                    if intent.cpu_fallback:
                        self._cleanup_failed_cpu_fallback()
                    # No child will carry this budget; left set, the route reads it
                    # ahead of is_active and asks for a reload with nothing loaded.
                    self._vram_fraction_pending = None
                    raise RuntimeError(detail)

                def _cleanup_cancelled_load(message: str) -> None:
                    logger.info(message)
                    self._cleanup_failed_cpu_fallback()
                    self._unload_audio_codec()
                    self._healthy = False
                    self._vram_fraction_pending = None

                def _finish_cancelled_health_wait(message: str) -> bool:
                    if not getattr(self, "_health_wait_cancelled", False):
                        return False
                    _cleanup_cancelled_load(message)
                    return True

                def _try_auto_vulkan_cpu_fallback(
                    failed_cmd: list[str],
                    failed_rc: Optional[int],
                    *,
                    terminal: bool = True,
                ) -> bool:
                    """Replay an eligible signal crash without devices.

                    ``terminal`` off reports a failed CPU spawn as "not recovered"
                    instead of raising, for a caller that still holds another argv
                    worth replaying.
                    """
                    nonlocal intent, gpu_memory_mode, gpu_layers, n_cpu_moe
                    nonlocal _replay_pageable_override
                    nonlocal tensor_parallel, tensor_split, gpu_indices, use_fit, _spawn_cwd
                    # GGML_ASSERT is a signal on POSIX and a CRT abort (exit 3) on
                    # MSVC, so Windows needs both. Cancel-checked like every retry
                    # here: an /unload racing the crash must not leave a server up.
                    _crashed = self._is_signal_crash(failed_rc) or (
                        sys.platform == "win32" and self._is_abort_exit(failed_rc)
                    )
                    if not _crashed or not _cpu_fallback_eligible or _load_cancelled():
                        return False
                    fallback_has_mmproj = self._launch_has_mmproj(failed_cmd, env)
                    prepared = self._prepare_cpu_fallback_launch(
                        binary,
                        failed_cmd,
                        env,
                        server_caps,
                        drop_full_offload_threads = _drop_full_offload_threads,
                        # The same host pool the preflight and the reprice below read,
                        # so the rewrite and the advisory answer one question once.
                        avail_mib = _preflight_avail_mib,
                    )
                    if prepared is None:
                        return False
                    if _load_cancelled():
                        # Staging copies a whole runtime, so an /unload can land after
                        # the gate above with no runtime yet for unload_model() to
                        # remove; take it back here.
                        _cleanup_cancelled_load("Load cancelled while staging the CPU fallback")
                        return False
                    replay, _spawn_cwd, _cpu_pageable_note = prepared
                    # The rewrite mutates the SHARED env in place, so it happens once
                    # even when several argvs are replayed. A non-terminal failure
                    # below returns without the note, and the next replay then finds an
                    # already-pageable env and produces none of its own -- so whichever
                    # attempt finally comes up would report ordinary paging and never
                    # say the user's load mode had been overridden. Carried instead.
                    if _cpu_pageable_note:
                        _replay_pageable_override = _cpu_pageable_note
                    _cpu_pageable_note = _cpu_pageable_note or _replay_pageable_override

                    logger.warning(
                        "The auto-selected Vulkan backend hard-crashed during "
                        "startup; retrying once with llama.cpp devices disabled."
                    )
                    if not _spawn_and_wait(replay, label = "-cpu"):
                        if _finish_cancelled_health_wait(
                            "Load cancelled during the staged CPU replay health wait"
                        ):
                            return False
                        if not terminal:
                            # This argv's drafter may be why it cannot start at all,
                            # which the GPU crash says nothing about. Reap the child
                            # and keep the staged runtime for the caller's next argv.
                            self._kill_process()
                            return False
                        _proc_snap1 = self._process  # snapshot: re-reading races the teardown
                        cpu_rc = _proc_snap1.poll() if _proc_snap1 is not None else None
                        detail = self._classify_llama_start_failure(
                            "\n".join(self._stdout_lines[-50:]),
                            gguf_path,
                            self._model_identifier,
                            cpu_rc,
                            binary,
                            self._llama_log_path,
                            (self._api_key,),
                            self._extra_args,
                        )
                        if _sched_abort_seen:
                            LlamaCppBackend._record_sched_reserve_abort(binary, _abort_memo_model)
                        self._cleanup_failed_cpu_fallback()
                        self._vram_fraction_pending = None
                        raise RuntimeError(detail)

                    # The replay dropped the fit's --load-mode none, so a record taken
                    # off the launch argv describes a reservation this child does not
                    # make, and a later "Don't reserve system RAM" would demand a
                    # pointless reload. From _last_spawn_cmd, the argv that really
                    # started, so this cannot undo a record a --fit retry wrote.
                    #
                    # Or the pageable override rewrote a mode this replay no longer runs
                    # with: the record has to describe the argv that started, whichever
                    # of the two changed it, or the reload comparator judges this child
                    # against a mode it does not have.
                    #
                    # Unconditional now: the replay also withdraws the managed DirectIO, a
                    # third way to go stale, and all of them land in _last_spawn_cmd.
                    self._record_memory_state(_last_spawn_cmd, env)
                    intent = self._apply_cpu_fallback_state(
                        intent,
                        is_vision = fallback_has_mmproj,
                        mmproj_has_audio = (
                            _launched_mmproj_has_audio if fallback_has_mmproj else False
                        ),
                        disable_vision = disable_vision,
                        # The same expression the load's own assignment uses, since the
                        # replay carries the same switch and the same dropped projector.
                        vision_disabled_by_user = bool(
                            is_vision and disable_vision and _dv_dropped_image_projector
                        ),
                    )
                    gpu_memory_mode = intent.gpu_memory_mode
                    gpu_layers = intent.gpu_layers
                    n_cpu_moe = intent.n_cpu_moe
                    tensor_parallel = intent.tensor_parallel
                    tensor_split = intent.tensor_split
                    gpu_indices = None
                    use_fit = False
                    # Everything above records that this session runs on no GPU; the
                    # advisory has to say so too. Here and not at the decision above,
                    # because only a child that came up is evidence of what is running
                    # and a failed replay can still fall through to another argv.
                    self._reprice_after_cpu_only_fallback(
                        host_msg = _host_ram_msg,
                        cpu_cmd = _last_spawn_cmd,
                        env = env,
                        avail_mib = _preflight_avail_mib,
                        pageable_note = _cpu_pageable_note,
                    )
                    # This return skips the commit block below, so write the drafter
                    # records here too: left describing the PREVIOUS load, the comparator
                    # judges this child against another model's drafter and a carried-over
                    # suppressed path stands the refetch down for a load that dropped
                    # nothing. Inert while only virtualised Metal could suppress, where
                    # auto-Vulkan fallback cannot happen; an unloadable sidecar can be
                    # suppressed anywhere.
                    self._mtp_draft_path = launch_mtp_draft_path
                    self._mtp_draft_suppressed_path = _suppressed_draft_path
                    self._mtp_draft_suppressed_reason = _suppressed_draft_reason
                    logger.warning(
                        "llama-server loaded successfully on CPU after the "
                        "auto-selected Vulkan backend crashed. GPU acceleration "
                        "is disabled for this model session."
                    )
                    return True

                # Store the resolved on-disk path, not the caller's kwarg: in
                # HF mode gguf_path is None and ``model_path`` is what
                # llama-server mmap's, which downstream consumers need. Must be
                # set BEFORE the spawn: load_progress() reads _gguf_path for
                # the mmap progress total while the health wait runs.
                self._gguf_path = model_path
                self._gguf_load_identity = self._gguf_load_source_identity(model_path, mmproj_path)
                self._hf_repo = hf_repo
                self._mtp_draft_path = launch_mtp_draft_path
                self._mtp_draft_suppressed_path = _suppressed_draft_path
                self._mtp_draft_suppressed_reason = _suppressed_draft_reason
                # For local GGUF files, extract variant from filename if absent
                if hf_variant:
                    self._hf_variant = hf_variant
                elif gguf_path:
                    try:
                        from utils.models.model_config import _extract_quant_label
                        self._hf_variant = _extract_quant_label(gguf_path)
                    except Exception:
                        self._hf_variant = None
                else:
                    self._hf_variant = None
                self._is_vision = effective_is_vision
                self._disable_vision = bool(disable_vision)
                # is_vision, NOT effective_is_vision: the latter means "a projector is
                # attached", which this load deliberately made false, so it would report
                # False for the one case this field exists to describe. The intent flag
                # is the model's own capability.
                #
                # Three cases must NOT blame the switch, because turning it back on
                # would not restore images: a text-only GGUF with a stale toggle, an
                # audio-only projector (kept on purpose, no image encoder), and an
                # advanced argument that drops the projector (nothing reads the file, so
                # the capability default stays True). All three fall through to the
                # generic "cannot accept images".
                #
                # So this is one signal recorded where the decision was made rather than
                # re-derived from state that cannot tell the cases apart: True only where
                # a usable image projector was resolved and the switch dropped it. None
                # of the three reaches that assignment.
                self._vision_disabled_by_user = bool(
                    is_vision and disable_vision and _dv_dropped_image_projector
                )
                self._model_identifier = model_identifier

                # Store the effective (possibly capped) context separately; do
                # NOT overwrite _context_length (the native length for display).
                self._effective_context_length = (
                    effective_ctx if effective_ctx > 0 else self._context_length
                )
                self._max_context_length = (
                    max_available_ctx if max_available_ctx > 0 else self._effective_context_length
                )
                # The fit priced Unsloth's own placement and the base weights alone: a
                # pass-through placement override, adapter or inherited projector is not.
                if _vram_fit_ctx is not None:
                    _child_extras = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args
                    )
                    _child_env = dict(os.environ)
                    if gpu_memory_mode == "manual":
                        self._clear_manual_placement_env(_child_env)
                    if _gpu_ids_own_device_flags:
                        self._clear_device_placement_env(_child_env)
                    if (
                        _pass_through_overrides_placement(
                            _child_extras,
                            _child_env,
                            len(gpu_indices) if gpu_indices is not None else len(gpus),
                            is_vulkan_backend = is_vulkan_backend,
                            gpu_indices = gpu_indices,
                        )
                        or _sidecar_adapter_bytes(extra_args) != 0
                        # A drafter the planner could only cover with a flat cushion.
                        or _flat_mtp_engages
                        or _child_env.get("LLAMA_ARG_MMPROJ")
                        or _child_env.get("LLAMA_ARG_MMPROJ_URL")
                        or (
                            not launch_mmproj_path
                            and extra_args_mmproj_auto(_child_extras, _child_env)
                        )
                        # A hand-set fitter or split ratio re-places what the fit priced.
                        or _extra_args_set_any_flag(_child_extras, _VRAM_FIT_VOIDING_FLAGS)
                        or any(_child_env.get(name) for name in _VRAM_FIT_VOIDING_ENV_VARS)
                        # Any surviving --device: the cap sweep may have pooled cards it leaves out.
                        or _extra_args_main_device(_child_extras) is not None
                        # And any companion (projector, drafter) pinned to a card of its own.
                        or any(
                            _extra_args_device(_child_extras, flags) is not None
                            for flags in _COMPANION_DEVICE_FLAG_GROUPS
                        )
                        or _child_env.get("LLAMA_ARG_DEVICE")
                        # --no-kv-offload keeps the whole cache in host RAM at any context.
                        or not _kv_offload_from_args(_child_extras, _child_env)
                    ):
                        _vram_fit_ctx = None
                self._vram_fit_context_length = _vram_fit_ctx

                # Past the host-RAM and offload advisories, so it can say the ceiling is
                # the request without costing the user a memory warning it does not
                # replace: appended when one of those spoke, recorded when none did.
                if _unmeasured_ctx_notice:
                    if self._last_load_warning:
                        self._amend_load_warning(" " + _unmeasured_ctx_notice)
                    else:
                        self._record_load_warning(_unmeasured_ctx_notice)

                if _cuda_ctx_notice and use_fit:  # slot reduction may have re-pinned it
                    if self._last_load_warning:
                        self._amend_load_warning(" " + _cuda_ctx_notice)
                    else:
                        self._record_load_warning(_cuda_ctx_notice)

                # LoadRequest carries this flag, so a stale rollback, an API caller or
                # a swapped-out runtime can ask for a replay that never happened. Hold
                # it to the crash path's bar (managed auto Vulkan, no explicit
                # placement) and otherwise run the request as sent.
                if intent.cpu_fallback:
                    if self._auto_vulkan_cpu_fallback_eligible(
                        binary, intent, extra_args, env, allow_manual_cpu = True
                    ):
                        prepared = self._prepare_cpu_fallback_launch(
                            binary,
                            cmd,
                            env,
                            server_caps,
                            drop_full_offload_threads = _drop_full_offload_threads,
                            avail_mib = _preflight_avail_mib,
                        )
                        if prepared is None:
                            if _load_cancelled():
                                _cleanup_cancelled_load(
                                    "Load cancelled while reconstructing the CPU fallback"
                                )
                                return False
                            _raise_terminal_load_failure(
                                "The prior Vulkan CPU fallback could not be reconstructed; run "
                                "`unsloth studio update` to repair the managed llama.cpp runtime."
                            )
                        cmd, _spawn_cwd, _replay_pageable_note = prepared
                        # Same reason as the crash path, recomputed from the stripped
                        # `cmd` before anything spawns. The arch-crash rung re-derives it
                        # from `cmd` too, so the _mem_policy_for_cmd snapshot it restores
                        # cannot put the stale pair back. The pageable override rewrites
                        # the same field, as does the managed DirectIO withdrawal, so the
                        # record follows the argv about to spawn. The snapshot with it:
                        # this replay REPLACED `cmd`.
                        self._record_memory_state(cmd, env)
                        _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                        # The preflight above priced and, where needed, rewrote the argv
                        # this replay was built FROM; an override settled here is news it
                        # could not have carried. Appended to whatever notice is recorded,
                        # the same way the APU preflight's override reaches the message,
                        # and a no-op on a load whose warning the opt-out silenced, which
                        # keeps the log line as its only record.
                        self._amend_load_warning(_replay_pageable_note)
                        # This replay is CPU-only by construction (--device none), so
                        # the advisory has to describe the whole model in host RAM. The
                        # amend above can only add to a notice that already exists, and
                        # on a restored fallback there often is none: an empty Vulkan
                        # probe leaves _child_has_no_gpu False, so the preflight
                        # abstained and memory_warning was null for a session paging
                        # everything from disk. Deferred rather than priced here,
                        # because only a child that came up is evidence of what runs.
                        _restored_cpu_reprice = _replay_pageable_note or True
                        # Same normalization the crash path applies, so a client that
                        # sent only the flag cannot leave a CPU-only server reporting
                        # an Auto/GPU placement (which then fails holds_no_vram).
                        intent = self._apply_cpu_fallback_state(
                            intent,
                            is_vision = self._is_vision,
                            mmproj_has_audio = self._mmproj_has_audio,
                            disable_vision = disable_vision,
                            vision_disabled_by_user = bool(
                                is_vision and disable_vision and _dv_dropped_image_projector
                            ),
                        )
                        gpu_memory_mode = intent.gpu_memory_mode
                        gpu_layers = intent.gpu_layers
                        n_cpu_moe = intent.n_cpu_moe
                        tensor_parallel = intent.tensor_parallel
                        tensor_split = intent.tensor_split
                        gpu_indices = None
                        use_fit = False
                    else:
                        # Phase 1 kept recovery state for a replay this request does
                        # not qualify for. Drop it, or this load reports the old crash
                        # and _preserve_cpu_fallback_intent rewrites later reloads.
                        intent = replace(intent, cpu_fallback = False)
                        self._cpu_fallback_reason = None
                        self._cleanup_cpu_fallback_runtime()
                # The placement above is now fixed for this child, but _process
                # is not set until Popen. Without this, a save landing in that
                # window sees no active backend and reports no reload while the
                # child is already committed. Popen clears it, and so does every
                # exit below, so a failed spawn cannot leave it stuck on.
                self._memory_pending_launch = _mem_settings
                self._vram_fraction_pending = _budget_priced_placement()
                healthy = False
                try:
                    healthy = _spawn_and_wait(cmd)
                    if not healthy:
                        # Every recovery below relaunches a plan the fit never priced.
                        self._vram_fit_context_length = None
                finally:
                    # Only once a child is up, where is_active takes over. An
                    # unconditional clear undid the re-arm and left the outer rungs (arch
                    # gate, KV-unified, flash-attention) respawning with nothing marking
                    # the window. `_serial_load_scope` releases it either way.
                    if healthy:
                        self._memory_pending_launch = None
                if not healthy and _finish_cancelled_health_wait(
                    "Load cancelled during the llama-server health wait"
                ):
                    return False
                # #6415 split-mode tensor warmup abort. Latch it on THIS first spawn:
                # the flash-attn-off retry below can't run tensor (needs flash_attn),
                # so its output drops the marker and recording later would miss it,
                # looping every load. Record and raise to the route's layer fallback,
                # skipping the futile flash-attn/MTP retries.
                if not healthy and self._tensor_parallel and not _load_cancelled():
                    _ts_out = "\n".join(self._stdout_lines[-50:])
                    _proc_snap2 = self._process  # snapshot: re-reading races the teardown
                    _ts_rc = _proc_snap2.poll() if _proc_snap2 is not None else None
                    if self._should_record_tensor_split_abort(
                        _ts_rc, "\n".join(self._stdout_lines)
                    ):
                        # A drafter crash must not disable tensor split for drafterless loads.
                        if not _extra_args_mtp_draft_path(
                            _last_spawn_cmd, env = _child_spec_env(extra_args)
                        ):
                            LlamaCppBackend._record_tensor_split_abort(
                                binary, model_identifier, _planned_cache_pair
                            )
                        self._kill_process()
                        raise RuntimeError(
                            "llama-server aborted on --split-mode tensor "
                            "(split-axis geometry); retrying with layer split."
                        )
                    # Pre-b9455 llama.cpp refusing a quantized KV cache in tensor mode.
                    # Latched on this first spawn for the same reason as the abort
                    # above: the retries below cannot fix a capability the binary
                    # lacks, and only the first spawn carries the marker.
                    if self._is_tensor_quant_kv_unsupported(_ts_out):
                        # Binary-wide: one detection spares every later model and
                        # every other quantized type the same doomed startup.
                        LlamaCppBackend._record_tensor_quant_kv_unsupported(binary)
                        self._kill_process()
                        raise RuntimeError(
                            "This llama.cpp build does not support a quantized KV "
                            "cache with --split-mode tensor (added in b9455, "
                            "ggml-org/llama.cpp#23792). Retrying with layer split; "
                            "update llama.cpp to use tensor parallelism with a "
                            "quantized cache."
                        )
                # "device kernel image is invalid": the auto-pinned device's arch has
                # no kernels in this binary (#7624: an iGPU whose shared-RAM "free
                # memory" outranked the dGPU). The proactive mapped_targets gate cannot
                # see custom-linked builds, so catch the crash and retry once on a
                # narrowed set. Auto selection only; an explicit pick keeps its error.
                if (
                    not healthy
                    and not _load_cancelled()
                    and not gpu_ids
                    and not is_vulkan_backend
                    and gpu_indices
                    and self._kernel_image_invalid("\n".join(self._stdout_lines[-80:]))
                ):
                    _crashed = sorted(set(gpu_indices))
                    # _detected_gpus, not gpus: the post-picker enumeration this launch
                    # planned against, and unlike `gpus` it survives the manual mode that
                    # empties the pool to bypass the planner.
                    _remaining = self._arch_crash_retry_gpu_ids(
                        gpu_indices, [i for i, _free in _detected_gpus]
                    )
                    if _remaining:
                        # APU presence controls the RAM guard; relative pool sizes
                        # determine whether managed allocations help.
                        _retry_wants_unified = self._amd_apu_wants_unified_memory(_remaining)
                        _retry_unified_helps = self._unified_memory_for_launch(
                            _remaining, _unified_need_now(), opted_in = _unified_opt_in
                        )
                        # Everything recorded so far priced the CRASHED selection, and
                        # that placement is gone: the canonical #7624 shape pins the APU
                        # whose shared-pool "free memory" outranked the dGPU, warns that
                        # system RAM cannot hold the weights, then crashes and respawns
                        # on the discrete sibling that holds them in VRAM. First notice
                        # wins, so without this the served child reports a shortfall on
                        # a device it never touches. The two checks below price exactly
                        # the set the respawn reaches, and both are wider than their
                        # upstream twins (the same weights against a NARROWED pool), so
                        # a shortfall that still stands is re-recorded here.
                        #
                        # Both are also kept in a variable: the pageable-load override
                        # at the END of this block, after every rewrite `cmd` still has
                        # coming, reads them as its shortfall condition and amends
                        # whichever one first-notice-wins recorded.
                        self._begin_load_warnings()
                        # ...and the re-price snapshots go with the scope they describe:
                        # the preflight's pair priced the crashed placement, so leaving
                        # them standing would let the text-only fallback re-price the
                        # notice this block is about to record against the wrong pool.
                        _apu_ram_msg = None
                        _host_ram_msg = None
                        _apu_avail_mib = None
                        # The APU RAM preflight upstream ran against the CRASHED
                        # selection. The mirror shape (crash on the dGPU, retry on the
                        # unified-memory sibling) makes the respawn the first load into
                        # system RAM, so price it here too. Advisory: the respawn goes
                        # ahead either way.
                        _retry_apu_msg = None
                        if model_size is not None and _retry_wants_unified:
                            _apu_avail_mib = self._available_system_memory_mib()
                            _apu_resident_model_size = max(
                                1,
                                model_size
                                - self._lazy_read_host_bytes(
                                    model_path,
                                    caps = server_caps,
                                    extra_args = cmd,
                                    env = env,
                                    auto_resolves_off = self._lazy_auto_resolves_off(
                                        gpu_indices = _remaining,
                                        detected_gpus = _detected_gpus,
                                        shared_gpu_ids = _shared_gpu_ids,
                                        is_vulkan_backend = is_vulkan_backend,
                                        probe_integrated_cuda = False,
                                    ),
                                ),
                            )
                            _retry_apu_msg = self._apu_ram_shortfall_message(
                                _apu_resident_model_size + _mmproj_pinned_bytes,
                                _apu_avail_mib,
                            )
                            # Same contract as the preflight above: the opt-out takes
                            # the message and nothing else. _retry_apu_msg itself stays
                            # set, so _retry_is_oversized below still sees the
                            # shortfall and the pageable rewrite still happens.
                            if not self._host_offload_warning_opted_out():
                                _apu_ram_msg = _retry_apu_msg
                                self._record_load_warning(_retry_apu_msg)
                        # host guard credited the whole pool; the respawn reaches only _remaining
                        _retry_rows = [row for row in _detected_gpus if row[0] in set(_remaining)]
                        _retry_host_msg = self._launch_host_shortfall_message(cmd, _retry_rows, env)
                        _host_ram_msg = _retry_host_msg
                        self._record_load_warning(_retry_host_msg)
                        logger.warning(
                            f"llama-server crashed with a HIP kernel-image error on "
                            f"GPU(s) {_crashed} -- the llama.cpp build has no kernels "
                            f"for that arch. Retrying on GPU(s) {_remaining}."
                        )
                        self._kill_process()
                        gpu_indices = _remaining
                        _unified_gpu_indices = _remaining
                        # The tuning was decided against the devices that just crashed,
                        # and this respawn reuses `cmd`, so a discrete-to-APU retry would
                        # carry --cache-ram 0 onto a shared pool and the reverse would
                        # reach a discrete card without it. Re-decide against the retry.
                        if sys.platform == "win32" and full_offload_tuning_active:
                            _retry_shared = _cache_target_unknown or (
                                self._offload_target_shares_system_memory(
                                    is_vulkan_backend = is_vulkan_backend,
                                    shared_gpu_ids = _shared_gpu_ids,
                                    detected_gpus = _retry_rows or _detected_gpus,
                                    gpu_indices = _remaining,
                                )
                            )
                            # Off first, or _retry_cache_tuning_flags reads the cap as a
                            # count the user typed and declines to zero it.
                            if _auto_ckpt_emitted:
                                cmd = self._without_flag_pairs(cmd, _auto_ckpt_emitted)
                                _auto_ckpt_emitted = []
                            if _retry_shared and _cache_flags_emitted:
                                # Exactly the tokens this policy appended, each a flag
                                # with its value, so the strip cannot eat a user extra.
                                cmd = self._without_flag_pairs(cmd, _cache_flags_emitted)
                                _cache_flags_emitted = []
                                logger.info(
                                    "Retry lands on a GPU that shares system memory: dropped "
                                    "--cache-ram 0 and --ctx-checkpoints 0 so the prompt cache "
                                    "survives."
                                )
                            elif not _retry_shared and not _cache_flags_emitted:
                                _retry_flags = self._retry_cache_tuning_flags(
                                    cmd,
                                    cache_ram = cache_ram,
                                    ctx_checkpoints = _ctx_checkpoints_owned,
                                    server_caps = server_caps,
                                )
                                if _retry_flags:
                                    cmd.extend(_retry_flags)
                                    _cache_flags_emitted = list(_retry_flags)
                                    logger.info(
                                        "Retry lands on a GPU with its own memory: applied the "
                                        "Windows full-offload cache tuning."
                                    )
                            _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(cmd)
                        # GGML_CUDA_ENABLE_UNIFIED_MEMORY was decided for the CRASHED
                        # set. The canonical #7624 shape crashes on the APU and retries
                        # on the dGPU, where it is harmful, so withdraw it, but only
                        # the value this launch set. Both directions: a markerless
                        # mixed host can as easily crash on the dGPU and land on the
                        # APU, which needs it.
                        if _unified_env_applied and not _retry_unified_helps:
                            env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
                            _unified_env_applied = False
                            logger.info(
                                "Arch-crash retry targets GPU(s) the unified-memory "
                                "swap cannot help; dropped "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY for the respawn."
                            )
                        elif (
                            _retry_unified_helps
                            and not _unified_opt_out  # the opt-out outlives the respawn
                            and "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
                        ):
                            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
                            _unified_env_applied = True
                            logger.info(
                                "Arch-crash retry targets a unified-memory APU the "
                                "weights outgrow; set "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 for the respawn."
                            )
                        _retry_mask = list(_remaining)
                        if _companion_fit_mask is not None:
                            # tokens are positions in the crashed launch's mask
                            if _companion_added_device is not None:
                                for _k in range(len(cmd) - 1, 0, -1):
                                    if cmd[_k - 1 : _k + 1] == _companion_added_device:
                                        del cmd[_k - 1 : _k + 1]
                                        break
                            _had_main_device = _extra_args_main_device(cmd) is not None
                            _retry_mask, _refit = _widen_pin_ids_for_companion_devices(
                                cmd,
                                list(_remaining),
                                list(_companion_fit_mask),
                                allowed_ids = gpu_ids or None,
                            )
                            _companion_added_device = (
                                list(cmd[-2:])
                                if not _had_main_device and _extra_args_main_device(cmd) is not None
                                else None
                            )
                            _companion_fit_mask = list(_retry_mask)
                            if _refit:
                                logger.info(
                                    "Refitted companion device flags to the retry's mask: %s",
                                    _refit,
                                )
                        self._emit_child_gpu_visibility(
                            env, ",".join(str(i) for i in _retry_mask), prefer_rocr = True
                        )
                        _child_gpu_physical_ids = tuple(int(i) for i in _retry_mask)
                        # Same for the inherited env twins of the flags dropped
                        # below: the new mask re-indexes the survivors under them.
                        self._clear_split_placement_env(env)
                        # The mask re-indexes the survivors from 0, so a --tensor-split
                        # sized for the crashed set weights the wrong cards (see
                        # _without_tensor_split). Drop it and let llama.cpp split by
                        # free VRAM over the narrowed set.
                        _arch_retry_cmd = self._without_tensor_split(cmd)
                        if _arch_retry_cmd is not None:
                            logger.info(
                                "Dropped --tensor-split for the arch-crash retry: "
                                "its shares are positional over the visible "
                                "devices and the narrowed set re-indexes them."
                            )
                            cmd = _arch_retry_cmd
                            # No ratio record, unlike the proactive gate: _tensor_split
                            # is non-None only in manual mode, which leaves gpu_indices
                            # None, and this arm needs it truthy.
                            self._tensor_split = None
                            # The auto twin, which this arm CAN reach: an auto
                            # tensor-parallel fallback emits its ratio with
                            # gpu_indices pinned. Left set it feeds the
                            # `tensor_split` property a ratio the respawned child
                            # does not have.
                            self._auto_tensor_split_emitted = None
                        # A tensor split needs two devices, so narrowing to one makes
                        # --split-mode tensor a no-op still REPORTED as active
                        # (tensor_parallel drives the UI and the MTP watchdog). Strip it.
                        if len(_remaining) < 2 and self._tensor_parallel:
                            _no_tensor_mode = self._without_flags(cmd, ("--split-mode", "-sm"))
                            if _no_tensor_mode is not None:
                                cmd = _no_tensor_mode
                            self._arch_gate_dropped_tensor_parallel = True
                            self._tensor_parallel = False
                            logger.info(
                                "Arch-crash retry leaves one GPU; dropped tensor "
                                "parallelism, which needs at least two."
                            )
                        # The fit chose "none" because the WHOLE footprint fit the
                        # devices this launch PINNED, and the retry runs on a set that
                        # proof never covered. The two guards above are weights-only
                        # floors and cannot re-establish it, so KV and scratch can still
                        # push the load past RAM while mmap stays off. Hand the respawn
                        # llama.cpp's auto back, as the --fit on and CPU fallbacks do.
                        # Only Unsloth's own tokens; a user's --load-mode is theirs.
                        _fit_mode_left_cmd = bool(self._fit_load_mode_flags)
                        if self._fit_load_mode_flags:
                            cmd = _without_subsequence(cmd, self._fit_load_mode_flags)
                            self._fit_load_mode_flags = []
                            logger.info(
                                "Load mode: dropping the fit's --load-mode none for "
                                "the arch-crash retry; it runs on a different device "
                                "set than the fit was proved against."
                            )
                        # This respawn starts from `cmd`, but the crashed launch may
                        # have taken _spawn_and_wait's --fit retry, which appends a
                        # page-lock to its OWN argv and records it. `cmd` never carried
                        # that lock, so leaving the record alone lets
                        # memory_state_satisfies_settings judge the launch satisfied and
                        # dedupe away the reload that would apply the user's page-lock.
                        # Restore what `cmd` means (a no-op when no retry fired).
                        (
                            _mem_host_resident,
                            self._memory_state,
                            self._memory_direct_io,
                            self._memory_dio_applicable,
                            self._memory_dio_flags,
                            self._memory_policy_active,
                            self._memory_mlock_applicable,
                        ) = _mem_policy_for_cmd
                        # AFTER the restore, which would otherwise put back the
                        # applicability the strip cleared. Re-asked on the narrowed set in
                        # BOTH directions: narrowing onto the surviving discrete card can
                        # GAIN an offload the original set never had, so guarding on an
                        # existing pair left the successful retry on mmap.
                        _arch_dio, _arch_applicable, _arch_active = _dio_decision_for(
                            _remaining, fully_offloaded = fully_gpu_offloaded, child_env = env
                        )
                        self._memory_dio_applicable = _arch_applicable
                        _dio_left_cmd = False
                        if self._memory_dio_flags and not _arch_dio:
                            _dio_left_cmd = True
                            cmd = self._drop_managed_dio(
                                cmd,
                                "the arch-crash retry's remaining devices no longer "
                                "confirm a full offload",
                            )
                            self._memory_dio_applicable = _arch_applicable
                        elif _arch_dio and not self._memory_dio_flags:
                            _dio_left_cmd = True
                            cmd = [*cmd, *_arch_dio]
                            self._memory_dio_flags = list(_arch_dio)
                            self._memory_policy_active = _arch_active or self._memory_policy_active
                            logger.info(
                                "Model Memory: applying %s; the arch-crash retry's "
                                "remaining GPU(s) %s confirm a full offload.",
                                " ".join(_arch_dio),
                                _remaining,
                            )
                        # ...except the load-mode pair just removed: the snapshot was
                        # taken while `cmd` still carried it, so restoring it would
                        # record a reservation this respawn no longer makes. From the
                        # stripped argv, NOT rebuilt from _mem_managed + _mem_extras: by
                        # here `cmd` may be a fallback argv those parts no longer add up
                        # to, dropping whatever memory-relevant flags they added.
                        if _fit_mode_left_cmd or _dio_left_cmd:
                            self._record_memory_state(cmd, env)
                        # Residency is a property of the DEVICES, which the retry just
                        # changed: crash on the discrete card, land on the
                        # unified-memory APU, and the weights are host-backed after
                        # all, so the lock the first launch skipped is the one the user
                        # asked for. Left alone the respawn runs unlocked AND records
                        # _memory_mlock_applicable False, reading the missing lock as
                        # deliberate and deduping away the reload that would apply it.
                        # Through the same helpers, so the two cannot drift.
                        _retry_host_resident = self._weights_in_host_memory(
                            fully_gpu_offloaded = fully_gpu_offloaded,
                            gpu_memory_mode = gpu_memory_mode,
                            gpu_layers = gpu_layers,
                            extra_args = _mem_extra_args,
                            gpu_indices = _remaining,
                            is_vulkan_backend = is_vulkan_backend,
                            binary = binary,
                            env = env,
                            # Widened like the other two sites: _mem_should_mlock is always
                            # False under no-reserve, so alone it asked with the probe off,
                            # and an unprobed device answers the conservative True.
                            probe_vulkan = _mem_should_mlock or _mem_probe_for_dio,
                            fit_active = fit_is_effectively_on([*cmd, *(_mem_extra_args or [])], env),
                        )
                        # Lock-ADDING direction only: `cmd` carries a policy-emitted
                        # lock only when the first launch found the weights
                        # host-resident, so this is the one direction that adds without
                        # taking anything back off argv. Removing flags would mean
                        # scanning for --mlock/--no-mmap, VALUELESS in llama.cpp's
                        # parser, so the scan would eat the user extra after them. The
                        # reverse direction leaves the crashed launch's lock in place,
                        # which stays truthful: the child does hold it.
                        if _retry_host_resident and not _mem_host_resident:
                            _retry_managed, _ = apply_model_memory_policy(
                                extra_args,
                                supports_load_mode = bool(server_caps.get("supports_load_mode")),
                                # Not the requested load mode: the first launch
                                # emitted it and this call only ADDS, so a second
                                # copy would land after the extras and mark
                                # _memory_policy_active for a launch it never touched.
                                weights_in_host_memory = _retry_host_resident,
                                # The launch's own pair: a save landing before this retry
                                # would otherwise pick flags this launch never saw.
                                settings = _mem_settings,
                            )
                            if _retry_managed:
                                # After the user extras, so llama.cpp's last-wins parse
                                # gives it the final say over any hand-written memory
                                # flag. Nothing is stripped, so kept extras survive.
                                cmd.extend(_retry_managed)
                            _mem_host_resident = _retry_host_resident
                            self._memory_mlock_applicable = _retry_host_resident
                            # Stays active if it already was: the flag also
                            # covers extras this policy touched and env it
                            # scrubbed, neither of which the respawn undoes.
                            self._memory_policy_active = (
                                bool(_retry_managed) or self._memory_policy_active
                            )
                            # From `cmd`: it may carry the managed pair, which the parts no
                            # longer add up to. Rebuilding from them recorded a mapped load
                            # for a streaming child, and the comparator then looped.
                            self._record_memory_state(cmd, env)
                            logger.info(
                                "Arch-crash retry changed where the weights live; "
                                "recomputed Model Memory (%s).",
                                " ".join(_retry_managed) if _retry_managed else "no page-lock",
                            )

                        # Last, because everything above still rewrites `cmd`: the
                        # tensor-split drop, the split-mode drop, the fit's load-mode
                        # strip and the re-armed page-lock. The narrowed pool is where a
                        # load that FIT its first placement can meet its first shortfall,
                        # and nothing upstream remapped it, so an unmapped respawn would
                        # allocate the whole model in host RAM and be OOM-killed. Same
                        # condition, same helper, before the respawn rather than after.
                        def _retry_is_oversized() -> bool:
                            if _retry_apu_msg or _retry_host_msg:
                                return True
                            # The opt-out silences the message, never the rewrite.
                            if not self._host_offload_warning_opted_out():
                                return False
                            return bool(
                                self._launch_host_shortfall_message(
                                    cmd, _retry_rows, env, honor_opt_out = False
                                )
                            )

                        cmd, _retry_pageable_note = self._page_an_oversized_unmapped_load(
                            cmd, env, _retry_is_oversized
                        )
                        if _retry_pageable_note:
                            # Onto whichever notice first-notice-wins kept above; a
                            # suppressed one leaves the log line as the only record,
                            # which is the intended contract for the opt-out.
                            self._amend_load_warning(_retry_pageable_note)
                            # From the argv, like the fit-strip above: `cmd` is what
                            # the respawn runs, and the record has to match it.
                            self._record_memory_state(cmd, env)
                        # And the carve-out advice with them. _begin_load_warnings()
                        # dropped the one priced for the crashed placement, but the
                        # respawn can land on a unified-memory APU whose allocation the
                        # same weights outgrow. Priced against `cmd` and _remaining, so
                        # the spill reported is this placement's.
                        try:
                            _retry_carveout_need = _unified_need_now(argv = cmd)
                        except Exception:
                            _retry_carveout_need = None
                        self._record_carveout_advice(
                            _remaining,
                            _retry_carveout_need,
                            is_vulkan_backend = is_vulkan_backend,
                            shared_gpu_ids = _shared_gpu_ids,
                            detected_gpus = _retry_rows or _detected_gpus,
                            target_unknown = _cache_target_unknown,
                        )
                        healthy = _spawn_and_wait(cmd, label = "-archfallback")

                # #12842: after the #7624 respawn every rung below keeps the same kernels.
                if not healthy and not _load_cancelled():
                    # Whole buffer: a Linux abort appends a debugger backtrace.
                    _cuda_image_out = "\n".join(self._stdout_lines)
                    if self._cuda_kernel_image_error(_cuda_image_out) is not None:
                        _proc_snap_ki = self._process  # snapshot: re-reading races the teardown
                        _ki_rc = _proc_snap_ki.poll() if _proc_snap_ki is not None else None
                        self._kill_process()
                        _raise_terminal_load_failure(
                            self._classify_llama_start_failure(
                                _cuda_image_out,
                                gguf_path,
                                self._model_identifier,
                                _ki_rc,
                                binary,
                                self._llama_log_path,
                                (self._api_key,),
                                self._extra_args,
                            )
                        )

                # Studio adds --kv-unified itself above one slot, so nothing the user
                # changes reaches it: retry at one slot, context intact. It aborts, so
                # this MUST stay ahead of the flash-attn rung, which takes any signal
                # crash; on the message, not the exit, so Windows lands here too.
                if not healthy and not _load_cancelled():
                    _kvu_cmd = (
                        self._with_single_sequence(_last_spawn_cmd)
                        if self._is_kv_unified_refused("\n".join(self._stdout_lines))
                        else None
                    )
                    if _kvu_cmd is not None:
                        logger.warning(
                            "llama-server refused a unified KV cache with more than "
                            "one sequence; retrying with one slot and --kv-unified "
                            "dropped. Concurrent requests will queue."
                        )
                        self._kill_process()
                        if self._drop_env_single_sequence(env):
                            logger.info(
                                "Dropped inherited LLAMA_ARG_KV_UNIFIED / "
                                "LLAMA_ARG_N_PARALLEL for the single-sequence retry."
                            )
                        cmd = _kvu_cmd
                        # Read by admission control; left as-is Studio over-admits.
                        n_parallel = 1  # allow-slot-clamp: llama-server refused more
                        kv_cache_unified = False
                        # The cap in argv was sized for slots this retry no longer runs;
                        # same re-decide as the arch-crash respawn.
                        if _auto_ckpt_emitted:
                            cmd = self._without_flag_pairs(cmd, _auto_ckpt_emitted)
                            _auto_ckpt_emitted = []
                        _auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()
                        _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(cmd)
                        healthy = _spawn_and_wait(cmd, label = "-single-seq")

                # Flash-attention kernels hard-crash at startup on some ROCm/GPU
                # builds (frequently inside the vision tower). Disabling FA keeps
                # both vision and MTP, so retry that way before dropping either.
                # Only on a hard fault with FA on; a cancel/unload stops respawn.
                if not healthy and not _load_cancelled():
                    _proc_snap3 = self._process  # snapshot: re-reading races the teardown
                    _fa_rc = _proc_snap3.poll() if _proc_snap3 is not None else None
                    _fa_cmd = (
                        self._with_flash_attn_off(
                            _last_spawn_cmd,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        )
                        if self._is_signal_crash(_fa_rc)
                        else None
                    )
                    if _fa_cmd is not None:
                        logger.warning(
                            "llama-server hard-crashed at startup (exit %s) with "
                            "flash attention on; retrying once with --flash-attn "
                            "off (keeps vision and MTP).",
                            _fa_rc,
                        )
                        self._kill_process()
                        # The argv rewrite can't reach an env-only quantized V
                        # cache; drop it so the FA-off child doesn't abort on it.
                        if self._drop_env_quantized_v_cache(
                            env,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        ):
                            logger.info(
                                "Dropped inherited quantized V-cache env for the "
                                "--flash-attn off retry (requires flash attention)."
                            )
                        # Same reach problem for the flag itself: where the rewrite
                        # could only drop a bare --flash-attn, the env var alone
                        # still enables it.
                        if self._drop_env_flash_attn(env):
                            logger.info(
                                "Dropped inherited LLAMA_ARG_FLASH_ATTN for the "
                                "--flash-attn off retry."
                            )
                        _fa_cmd = _drop_fit_load_mode_for_no_flash(_fa_cmd)
                        _fa_cmd = _enable_managed_fit_for_no_flash(_fa_cmd)
                        _flash_attn_known_off = True
                        cmd = _fa_cmd
                        healthy = _spawn_and_wait(_fa_cmd, label = "-noflash")

                # MTP from Unsloth's spec flags or the user's (extra_args
                # --spec-type / LLAMA_ARG_SPEC_TYPE). The env survives to the child only
                # when the extras own --spec-type (the launch scrubs it otherwise), so
                # anything else would judge a server it is not starting.
                _launch_spec_env: Mapping[str, str] = _child_spec_env(extra_args)
                _spec_requested_mtp = any(
                    "mtp" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_mtp(extra_args, env = _launch_spec_env)
                _spec_requested_dspark = any(
                    "draft-dspark" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_dspark(extra_args, env = _launch_spec_env)
                _spec_requested_dflash = any(
                    "draft-dflash" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_dflash(extra_args, env = _launch_spec_env)
                # A paired EAGLE-3 drafter (Studio export) can abort the server the same way.
                _spec_requested_eagle3 = any("draft-eagle3" in str(t).lower() for t in spec_flags)
                # Any MTP launch, not only tensor: a respawn would replay a drafter that
                # aborts every decode (#13253). Cleared if the MTP-drop fallback wins.
                _mtp_active_for_launched_server = bool(_spec_requested_mtp)
                # MTP can pass /health then crash the flash-attn kernel on the
                # first decode under tensor; probe one generation so the fallback
                # catches that too. Tensor-only, so ordinary MTP stays probe-free.
                if (
                    healthy
                    and self._tensor_parallel
                    and _spec_requested_mtp
                    and not _load_cancelled()
                    and not self._probe_mtp_decode()
                ):
                    # A first-decode hard fault is usually the FA kernel: retry
                    # FA-off (keeps MTP) before dropping speculative decoding below.
                    _proc_snap4 = self._process  # snapshot: re-reading races the teardown
                    _probe_rc = _proc_snap4.poll() if _proc_snap4 is not None else None
                    _fa_cmd = (
                        self._with_flash_attn_off(
                            _last_spawn_cmd,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        )
                        if self._is_signal_crash(_probe_rc)
                        else None
                    )
                    healthy = False
                    if _fa_cmd is not None:
                        logger.warning(
                            "MTP first-decode hard-crashed (exit %s) with flash "
                            "attention on; retrying with --flash-attn off.",
                            _probe_rc,
                        )
                        self._kill_process()
                        # The argv rewrite can't reach an env-only quantized V
                        # cache; drop it so the FA-off child doesn't abort on it.
                        if self._drop_env_quantized_v_cache(
                            env,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        ):
                            logger.info(
                                "Dropped inherited quantized V-cache env for the "
                                "--flash-attn off retry (requires flash attention)."
                            )
                        # Same reach problem for the flag itself: where the rewrite
                        # could only drop a bare --flash-attn, the env var alone
                        # still enables it.
                        if self._drop_env_flash_attn(env):
                            logger.info(
                                "Dropped inherited LLAMA_ARG_FLASH_ATTN for the "
                                "--flash-attn off retry."
                            )
                        _fa_cmd = _drop_fit_load_mode_for_no_flash(_fa_cmd)
                        _fa_cmd = _enable_managed_fit_for_no_flash(_fa_cmd)
                        _flash_attn_known_off = True
                        cmd = _fa_cmd
                        healthy = (
                            _spawn_and_wait(_fa_cmd, label = "-noflash-mtp")
                            and self._probe_mtp_decode()
                        )
                    if not healthy:
                        logger.warning(
                            "MTP speculative decoding crashed on the first decode "
                            "under tensor parallelism; retrying without it."
                        )
                # Any MTP request can abort the server: a separate drafter
                # (Gemma) on a binary that predates its arch, or an embedded
                # head (Qwen) the binary cannot build. Retry once with the
                # spec slice replaced by --spec-default so the main model still
                # loads. Gate on the spec block (not the drafter path, which
                # off/ngram local loads also carry) and keep
                # _requested_spec_mode so a duplicate /load doesn't thrash. The
                # cancel check stops an /unload-killed attempt respawning. A
                # decode-probe failure above also routes here.
                # The retry replaces _last_spawn_cmd with a drafterless argv, but a GPU
                # init crash is not the drafter's fault: keep the requested launch so
                # the CPU replay below can still run it.
                _spec_cpu_replay_cmd: Optional[List[str]] = None
                if (
                    not healthy
                    and (
                        _spec_requested_mtp
                        or _spec_requested_dspark
                        or _spec_requested_dflash
                        or _spec_requested_eagle3
                    )
                    and not _load_cancelled()
                ):
                    _spec_cpu_replay_cmd = list(_last_spawn_cmd)
                    # Blame the binary only when the output shows MTP itself
                    # failing (unknown arch / draft or context build); an
                    # unrelated crash (e.g. OOM) gets a neutral message.
                    _lo = "\n".join(self._stdout_lines).lower()
                    # Only an unknown architecture proves the prebuilt predates
                    # this MTP model (an update fixes it). The memory/context
                    # build failures are generic (VRAM / ctx pressure), where an
                    # update may not help, so classify those as runtime_error.
                    _arch_unsupported = "unknown model architecture" in _lo
                    if (
                        _arch_unsupported
                        or "failed to measure draft model memory" in _lo
                        or "failed to measure mtp context memory" in _lo
                        or "failed to create mtp context" in _lo
                        or "failed to create llama_context" in _lo
                    ):
                        _retry_reason = (
                            "the prebuilt may predate it; retrying without "
                            "speculative decoding -- run `unsloth studio "
                            "update` for this drafter"
                        )
                        self._spec_fallback_reason = (
                            "binary_outdated" if _arch_unsupported else "runtime_error"
                        )
                    else:
                        _retry_reason = (
                            "retrying without speculative decoding in case the drafter is the cause"
                        )
                        self._spec_fallback_reason = "runtime_error"
                    _drafter = (
                        Path(launch_mtp_draft_path).name
                        if launch_mtp_draft_path
                        else "embedded head"
                    )
                    logger.warning(
                        "llama-server failed to start with speculative drafter (%s); %s.",
                        _drafter,
                        _retry_reason,
                    )
                    self._kill_process()
                    # Re-locate rather than trust _spec_start: the arch gate may have
                    # narrowed cmd since, which slides the block left and would
                    # otherwise slice --api-key and friends out of the tail.
                    _spec_at = _subsequence_index(cmd, spec_flags, _spec_start)
                    _fb_tail = cmd[_spec_at + len(spec_flags) :]
                    _fb_stripped_extras: Optional[List[str]] = None
                    # Appending --spec-default cannot clear MTP the extras or the env
                    # carry: types accumulate, so it would retry the mode that just
                    # failed and lose a main model that loads fine without it. The
                    # tail loses its spec group and the env goes with it, the same way
                    # the crash replay does it.
                    if (
                        _extra_args_requests_mtp(extra_args, env = _launch_spec_env)
                        or _extra_args_requests_dspark(extra_args, env = _launch_spec_env)
                        or _extra_args_requests_dflash(extra_args, env = _launch_spec_env)
                    ):
                        _fb_tail = strip_shadowing_flags(
                            _fb_tail,
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                        for _fb_spec_var in _SPEC_ENV_VARS:
                            env.pop(_fb_spec_var, None)
                        _fb_stripped_extras = strip_shadowing_flags(
                            extra_args or [],
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                    fallback_cmd = cmd[:_spec_at] + ["--spec-default"] + _fb_tail
                    # Restore the cut slots; if they fail (unified KV refused), retry as cut.
                    _fb_cut = (
                        (
                            fallback_cmd,
                            n_parallel,
                            effective_ctx,
                            (self._effective_context_length, self._max_context_length),
                            _auto_ctx_checkpoints,
                            _auto_ckpt_emitted,
                        )
                        if _mtp_slots_asked is not None
                        else None
                    )
                    if _fb_cut is not None:
                        n_parallel = _mtp_slots_asked
                        fallback_cmd = _with_parallel_slots(
                            fallback_cmd,
                            n_parallel,
                            n_batch = n_batch,
                            supports_kv_unified = bool(server_caps.get("supports_kv_unified")),
                        )
                        # The cut may have bought context; the restored count was priced at this.
                        if _mtp_slots_target_ctx and "-c" in fallback_cmd:
                            effective_ctx = _mtp_slots_target_ctx
                            fallback_cmd[fallback_cmd.index("-c") + 1] = str(effective_ctx)
                            # The ceiling was priced at the cut count; this is the one proven here.
                            self._effective_context_length = self._max_context_length = (
                                effective_ctx
                            )
                        # The checkpoint cap is per slot; same re-decide as the single-seq rung.
                        if _auto_ckpt_emitted:
                            fallback_cmd = self._without_flag_pairs(
                                fallback_cmd, _auto_ckpt_emitted
                            )
                        _auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()
                        _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(fallback_cmd)
                        kv_cache_unified = _kv_unified_from_args(fallback_cmd)
                        # The plan other loads place against was priced at the cut count.
                        _fb_cut_plan = dict(self._pending_plan_mib)
                        if self._pending_plan_mib and self._can_estimate_kv():
                            kv_cache_bytes = self._estimate_kv_cache_bytes(
                                effective_ctx,
                                cache_type_kv,
                                n_parallel = n_parallel,
                                swa_full = swa_full,
                                kv_unified = bool(kv_cache_unified),
                                n_ubatch = _ubatch_for_slots(n_parallel),
                                flash_attn = planned_flash_attn,
                            )
                            self._pending_plan_mib = _gpu_plan_mib(
                                gguf_size + mmproj_size + kv_cache_bytes,
                                list(self._pending_plan_mib),
                                gpus,
                            )
                    _unified_withdraw_if_unneeded(
                        fallback_cmd,
                        mtp_engages = False,
                        why = "The retry without speculative decoding",
                    )
                    healthy = _spawn_and_wait(fallback_cmd, label = "-retry")
                    if not healthy and _fb_cut is not None and not _load_cancelled():
                        logger.warning(
                            "The retry without speculative decoding failed at %d slots; "
                            "retrying at the %d the drafter launch ran.",
                            n_parallel,
                            _fb_cut[1],
                        )
                        self._kill_process()
                        (
                            fallback_cmd,
                            _fb_slots,
                            effective_ctx,
                            (self._effective_context_length, self._max_context_length),
                            _auto_ctx_checkpoints,
                            _auto_ckpt_emitted,
                        ) = _fb_cut
                        self._pending_plan_mib = _fb_cut_plan
                        n_parallel = (
                            _fb_slots  # allow-slot-clamp: the restored count failed to start
                        )
                        kv_cache_unified = _kv_unified_from_args(fallback_cmd)
                        healthy = _spawn_and_wait(fallback_cmd, label = "-retry-cut")
                    if healthy:
                        self._speculative_type = "default"
                        _mtp_active_for_launched_server = False
                        # The record below stores extra_args as what LAUNCHED, but this
                        # server launched without the spec group. Leaving the MTP list
                        # there would have the next inheriting Apply pick it back up and
                        # repeat the crash. Keep it as the requested state only, the way
                        # the runtime-crash replay does.
                        if _fb_stripped_extras is not None:
                            _pv_suppressed_spec_extra_args = (
                                _pv_suppressed_spec_extra_args
                                if _pv_suppressed_spec_extra_args is not None
                                else list(extra_args or [])
                            )
                            extra_args = _fb_stripped_extras

                # Keep a multimodal model multimodal when only its GPU projector
                # placement fails. llama.cpp owns mmproj offload separately from
                # --gpu-layers, so retry it on CPU before removing --mmproj.
                if not healthy:
                    out = "\n".join(self._stdout_lines[-50:])
                    # Read the crash code before _kill_process() clears _process.
                    _proc_snap5 = self._process  # snapshot: re-reading races the teardown
                    _crash_rc = _proc_snap5.poll() if _proc_snap5 is not None else None
                    self._kill_process()
                    # Only when the WAIT itself was cancelled. A cancel that lands later,
                    # while a crashed launch is staging its CPU fallback, must still run
                    # that recovery, so _load_cancelled() alone is the wrong test here.
                    if _finish_cancelled_health_wait(
                        "Load cancelled during the llama-server health wait"
                    ):
                        return False
                    # The #6415 split-axis abort is latched earlier (first spawn).
                    # Skip if a cancel/unload is pending (mirrors the MTP guard).
                    _projector_msg = self._is_projector_incompatibility(out)
                    _projector_memory = self._is_gpu_memory_start_failure(out)
                    _signal_mmproj_guess = self._is_signal_crash(
                        _crash_rc
                    ) and not self._output_has_nonprojector_diagnostic(out)
                    if (
                        launched_with_mmproj
                        and not _load_cancelled()
                        and (_projector_msg or _projector_memory or _signal_mmproj_guess)
                    ):
                        _vision_gpu_cmd = list(_last_spawn_cmd)
                        _cpu_projector_cmd = None
                        if not _projector_msg and _paravirtual_mmproj_pinnable(server_caps):
                            _cpu_projector_cmd = self._with_mmproj_offload_disabled(
                                _vision_gpu_cmd, env
                            )

                        if _cpu_projector_cmd is not None:
                            # The fit priced the projector as GPU-eligible, and this
                            # retry moves those bytes into host RAM, so neither the
                            # VRAM-only proof (which never asked what RAM holds) nor the
                            # VRAM-plus-RAM one (answered with the projector on the card)
                            # covers the launch this respawns. Only Unsloth's own tokens;
                            # a user's --load-mode is theirs. The record is NOT cleared:
                            # the fallbacks below respawn from an argv that carries these
                            # tokens and has to be able to name them.
                            if self._fit_load_mode_flags:
                                _stripped_cpu_projector_cmd = _without_subsequence(
                                    _cpu_projector_cmd, self._fit_load_mode_flags
                                )
                                if _stripped_cpu_projector_cmd != _cpu_projector_cmd:
                                    # Same as the no-flash rung: the record follows the
                                    # argv, or a stale "reserving" spends a full reload
                                    # the running server already satisfies.
                                    self._record_memory_state(_stripped_cpu_projector_cmd, env)
                                    logger.info(
                                        "Load mode: dropping the fit's --load-mode none "
                                        "for the CPU-projector retry; it moves the "
                                        "projector into host RAM the fit credited to VRAM."
                                    )
                                _cpu_projector_cmd = _stripped_cpu_projector_cmd
                            # The managed pair stays: this retry moves the PROJECTOR, which
                            # clip.cpp loads through its own ifstream, and the main model is
                            # still fully offloaded. The block above is the fit's VRAM
                            # accounting, which the move really does void.
                            logger.warning(
                                "llama-server failed while loading this model's GPU "
                                "vision projector (--mmproj); retrying with the "
                                "projector on CPU to preserve image input."
                            )
                            cmd = _cpu_projector_cmd
                            _unified_withdraw_if_unneeded(
                                cmd, why = "The retry with the projector on CPU"
                            )
                            healthy = _spawn_and_wait(cmd, label = "-mmproj-cpu")
                            if healthy:
                                self._mmproj_fallback_reason = "cpu_offload"
                                logger.warning(
                                    "Vision projector loaded on CPU after GPU startup "
                                    "failed; image input remains available for this session."
                                )
                            else:
                                _cpu_projector_out = "\n".join(self._stdout_lines[-50:])
                                # Snapshot: re-reading races the teardown.
                                _proc_snap6 = self._process
                                _cpu_projector_rc = (
                                    _proc_snap6.poll() if _proc_snap6 is not None else None
                                )
                                self._kill_process()
                                if _finish_cancelled_health_wait(
                                    "Load cancelled during the CPU-projector health wait"
                                ):
                                    return False
                                if _load_cancelled():
                                    _cleanup_cancelled_load(
                                        "Load cancelled after the CPU-projector health wait"
                                    )
                                    return False
                                if self._is_projector_incompatibility(_cpu_projector_out):
                                    _projector_msg = True
                                elif self._is_gpu_memory_start_failure(_cpu_projector_out):
                                    # The projector was already off the GPU, so removing it
                                    # cannot repair this allocation failure; keep the real error.
                                    _raise_terminal_load_failure(
                                        self._classify_llama_start_failure(
                                            _cpu_projector_out,
                                            gguf_path,
                                            self._model_identifier,
                                            _cpu_projector_rc,
                                            binary,
                                            self._llama_log_path,
                                            (self._api_key,),
                                            self._extra_args,
                                        )
                                    )
                        elif _projector_memory and _paravirtual_mmproj_pinnable(server_caps):
                            # An env/argv pin already put mmproj on CPU. Text-only
                            # cannot free additional GPU memory, so surface the OOM.
                            _raise_terminal_load_failure(
                                self._classify_llama_start_failure(
                                    out,
                                    gguf_path,
                                    self._model_identifier,
                                    _crash_rc,
                                    binary,
                                    self._llama_log_path,
                                    (self._api_key,),
                                    self._extra_args,
                                )
                            )

                        if not healthy:
                            if _projector_msg:
                                logger.warning(
                                    "llama-server could not load this model's vision "
                                    "projector (--mmproj). The installed llama.cpp build is "
                                    "likely too old for it. Loading text-only for this "
                                    "session; run 'unsloth studio update' to enable vision."
                                )
                                self._mmproj_fallback_reason = "projector_incompatible"
                            else:
                                logger.warning(
                                    "llama-server could not start with this model's vision "
                                    "projector (--mmproj), including the CPU-projector "
                                    "recovery when available. Retrying text-only for this "
                                    "session; check memory, GPU/driver logs, or update Unsloth."
                                )
                                self._mmproj_fallback_reason = "projector_startup_failure"
                            cmd = self._strip_mmproj_args(_vision_gpu_cmd)
                            _unified_withdraw_if_unneeded(cmd, why = "The text-only retry")
                            # This retry bypasses _spawn_and_wait, so refresh the
                            # launched-argv snapshot itself -- the zero-offload
                            # classification below must not see the stripped --mmproj.
                            _last_spawn_cmd = list(cmd)
                            self._is_vision = False
                            self._mmproj_has_audio = False
                            if not self._start_llama_process(
                                cmd,
                                env,
                                child_gpu_physical_ids = _child_gpu_physical_ids,
                            ):
                                # Shutdown refused the retry; self._process still names
                                # the old child the teardown is clearing.
                                _cleanup_cancelled_load("App shut down during the text-only retry")
                                return False
                            if self._wait_for_health(
                                timeout = self._HEALTH_STALL_TIMEOUT_S, cancelled = _load_cancelled
                            ):
                                healthy = True
                                # The child that serves this session never read the
                                # CPU-pinned projector, so the host-memory advisory the
                                # APU preflight charged it against is now describing a
                                # load nothing is running. Priced again here, and not at
                                # the strip above, because a failed text-only retry can
                                # still replay the ORIGINAL vision argv on CPU further
                                # down -- projector and all -- and that child does load
                                # the bytes. Only a healthy text-only server is evidence.
                                self._reprice_after_dropping_pinned_projector(
                                    apu_msg = _apu_ram_msg,
                                    host_msg = _host_ram_msg,
                                    model_size = _apu_resident_model_size,
                                    pinned_bytes = _mmproj_pinned_bytes,
                                    avail_mib = _apu_avail_mib,
                                    part = _apu_ram_part,
                                )
                            else:
                                # Read the exit code before _kill_process() clears it, so
                                # an OS-killed text-only retry still gets the OOM message.
                                # Snapshot: re-reading races the teardown.
                                _retry_proc = self._process
                                _retry_rc = _retry_proc.poll() if _retry_proc is not None else None
                                # This retry bypasses _spawn_and_wait's abort detection.
                                if _retry_rc not in (None, 0) and self._is_sched_reserve_abort(
                                    "\n".join(self._stdout_lines[-200:])
                                ):
                                    _sched_abort_seen = True
                                self._kill_process()
                                if _finish_cancelled_health_wait(
                                    "Load cancelled during the text-only retry health wait"
                                ):
                                    return False
                                # A text-only signal crash is independent evidence of a GPU
                                # startup fault. Keep a confirmed bad projector out of the
                                # replay; a guessed one still gets a CPU try with vision.
                                if self._is_signal_crash(_retry_rc):
                                    _cpu_replay_cmd = (
                                        _last_spawn_cmd if _projector_msg else _vision_gpu_cmd
                                    )
                                    if _try_auto_vulkan_cpu_fallback(
                                        _cpu_replay_cmd,
                                        _retry_rc,
                                    ):
                                        healthy = True
                                        # A whole-runtime CPU replay with the original
                                        # vision argv supersedes the text-only diagnosis.
                                        if not _projector_msg:
                                            self._mmproj_fallback_reason = None
                                    else:
                                        if _finish_cancelled_health_wait(
                                            "Load cancelled during the CPU replay health wait"
                                        ):
                                            return False
                                        _raise_terminal_load_failure(
                                            self._gpu_init_crash_message(binary)
                                        )
                                if not healthy:
                                    _retry_detail = self._classify_llama_start_failure(
                                        "\n".join(self._stdout_lines[-50:]),
                                        gguf_path,
                                        self._model_identifier,
                                        _retry_rc,
                                        binary,
                                        self._llama_log_path,
                                        (self._api_key,),
                                        self._extra_args,
                                    )
                                    if _finish_cancelled_health_wait(
                                        "Load cancelled during the text-only recovery"
                                    ):
                                        return False
                                    _raise_terminal_load_failure(
                                        self._mmproj_retry_failure_message(
                                            projector_confirmed = _projector_msg,
                                            detail = _retry_detail,
                                        )
                                    )
                    else:
                        # Try the drafter launch first, non-terminally: a build that
                        # can neither pin one to CPU nor start with it still recovers
                        # on the drafterless argv. Snapshot that argv, since a spawned
                        # replay rebinds _last_spawn_cmd to its own command line.
                        _drafterless_cpu_replay_cmd = list(_last_spawn_cmd)
                        _replayed = _spec_cpu_replay_cmd is not None and (
                            _try_auto_vulkan_cpu_fallback(
                                _spec_cpu_replay_cmd, _crash_rc, terminal = False
                            )
                        )
                        if _replayed:
                            # The drafter came back with the replay, so the
                            # speculative-disable diagnosis no longer holds.
                            self._spec_fallback_reason = None
                        elif _try_auto_vulkan_cpu_fallback(_drafterless_cpu_replay_cmd, _crash_rc):
                            _replayed = True
                        if _replayed:
                            healthy = True
                        else:
                            if _finish_cancelled_health_wait(
                                "Load cancelled during the CPU replay health wait"
                            ):
                                return False
                            _raise_terminal_load_failure(
                                self._classify_llama_start_failure(
                                    out,
                                    gguf_path,
                                    self._model_identifier,
                                    _crash_rc,
                                    binary,
                                    self._llama_log_path,
                                    (self._api_key,),
                                    self._extra_args,
                                )
                            )

                if _restored_cpu_reprice is not None:
                    # The child that answered is the CPU-only replay, so price the
                    # advisory against the whole model in system RAM, against the pool
                    # the preflight read rather than a live one taken with the weights
                    # already resident.
                    self._reprice_after_cpu_only_fallback(
                        host_msg = _host_ram_msg,
                        cpu_cmd = _last_spawn_cmd or cmd,
                        env = env,
                        avail_mib = _preflight_avail_mib,
                        pageable_note = (
                            _restored_cpu_reprice
                            if isinstance(_restored_cpu_reprice, str)
                            else None
                        ),
                    )
                if not self._publish_healthy():
                    # Teardown began between the 200 and this commit; publishing
                    # would report a model that is gone.
                    _cleanup_cancelled_load("App shut down as the load was completing")
                    return False
                self._commit_effective_parallel_slots(n_parallel)
                self._swa_full = swa_full
                self._kv_cache_unified = kv_cache_unified
                # Re-derived from the slot count that LAUNCHED, not the one the sizing
                # pass saw. The drafter drop and the non-MTP retry both hand slots back
                # after the batch flag is emitted, and this is recorded next to
                # _commit_effective_parallel_slots: pairing those slots with a micro-batch
                # derived at the clamped count under-states the cache the slot save
                # re-estimates from it.
                _launched_ubatch = _ubatch_for_slots(n_parallel)
                self._n_ubatch = max(
                    0,
                    int(self._DEFAULT_N_UBATCH if _launched_ubatch is None else _launched_ubatch),
                )
                # What the child is RUNNING: the same architecture rule the plan above used.
                self._flash_attn_enabled = _flash_attn_enabled_from_args(
                    _last_spawn_cmd, default = not _flash_attn_known_off, env = env
                ) and not _architecture_forces_flash_attn_off(self._architecture)
                self._effective_cache_types = _effective_main_cache_types(
                    _last_spawn_cmd,
                    env,
                )
                # The pair the REQUEST resolved to, kept beside the pair that launched.
                # A launch-time rewrite makes them differ (no --flash-attn resets a
                # quantized V cache to f16, as does the flash-attn crash recovery), and
                # the matcher wants the request: comparing the running pair against an
                # unrewritten request rejects every identical repeat load.
                self._requested_cache_types = _planned_cache_pair
                self._kv_cache_context_total = effective_ctx if effective_ctx > 0 else None

                # Server is up: adopt the real per-request context it allocated
                # -- the length --fit chose, or a --parallel slot split -- so the
                # reported context_length matches reality. (Querying /props
                # before the spawn above always failed; the seeded value was the
                # requested/native length.)
                # Clear first, or a swap into a model without video inherits
                # the previous server's answer.
                self._has_video_input = False
                # The explicit override, not the field fallback: only a flag after
                # Studio's own -c can allocate past the fit, so it is the ceiling.
                self._reconcile_effective_ctx_with_server(ctx_override or 0)
                if self._kv_cache_context_total is not None:
                    self._n_ubatch = min(
                        self._n_ubatch,
                        self._kv_cache_context_total,
                    )

                # Commit caller intent only after _healthy=True so a failed start
                # can't poison the next inheritance check. None keeps prior, []
                # clears, list sets. Source records hf_variant for the route's
                # same_source check.
                if extra_args is not None:
                    # Persist the same device-stripped extras the command used when gpu_ids
                    # owns placement, so a later inheriting reload (after gpu_ids clears)
                    # can't resurrect the dropped --device (#7188).
                    self._extra_args = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else list(extra_args)
                    )
                    # Device-stripped the same way, so both comparator sides share a rule.
                    # _extra_args_as_requested first: it predates the Metal
                    # zero-context strip, whereas the drafter-drop snapshot is taken
                    # after it and would put that strip back into the comparator. The
                    # two never disagree otherwise, since the snapshot is only set
                    # alongside a non-None _extra_args_as_requested and the spec strip
                    # it guards runs later still.
                    _pv_requested = (
                        _extra_args_as_requested
                        if _extra_args_as_requested is not None
                        else (
                            _pv_suppressed_spec_extra_args
                            if _pv_suppressed_spec_extra_args is not None
                            else extra_args
                        )
                    )
                    self._requested_extra_args = (
                        self._strip_device_extra_args(_pv_requested)
                        if _gpu_ids_own_device_flags
                        else list(_pv_requested)
                    )
                    self._extra_args_source = (model_identifier, hf_variant)
                # A re-fit replay records what launched, which is what the client replays next.
                self._requested_n_ctx = int(effective_ctx if _replayed_ctx_refit else n_ctx)
                # Local n_parallel may have been reduced above; the snapshot has the ask.
                self._requested_n_parallel = max(1, int(intent.n_parallel))
                self._slots_cut_for_drafter = (
                    _mtp_slots_asked is not None and n_parallel < _mtp_slots_asked
                )
                # Commit with the rest of the known-good state: only a launch that got
                # this far replaced the runtime, so a rejected preflight leaves the old
                # marker intact, and the diffusion runner returns well before it without
                # consuming, or paying for, any llama-server capability.
                self._capability_probe_inconclusive = _launch_probe_inconclusive
                # The budget this child was sized against, committed with the rest of
                # the known-good state so a duplicate-load fast path or failed launch
                # leaves the previous marker intact and the route keeps asking for the
                # reload. None when placement never used it (_budget_priced_placement).
                self._vram_fraction_launched = _budget_priced_placement()
                self._vram_fraction_pending = None
                self._requested_n_batch = intent.n_batch
                self._requested_n_ubatch = intent.n_ubatch
                self._requested_load_mode = intent.load_mode
                self._requested_spec_draft_cache_type = intent.spec_draft_cache_type
                self._requested_ctx_checkpoints = intent.ctx_checkpoints
                self._requested_cache_ram = intent.cache_ram
                # Commit the known-good snapshot + whether MTP is live, then
                # watch this load for a mid-generation crash. The verified-cache hint is
                # dropped: a respawn replays this snapshot arbitrarily later, so recovery
                # re-resolves the file instead of trusting a path this request verified.
                self._last_load_intent = replace(intent, verified_gguf = None)
                self._mtp_runtime_fallback_active = _mtp_active_for_launched_server
                self._start_mtp_crash_watchdog()

                # Catch silent CPU fallback when GPU was intended (#5106). Manual
                # offload (no picker) leaves gpu_indices None and use_fit False, so
                # include its GPU-layer intent; use the preserved probe since
                # auto-layers/manual empty `gpus`. A deliberate zero-offload load
                # classifies by its launched argv instead: CPU-only by construction, it
                # must read False (not None) or training needlessly unloads a server
                # holding no VRAM. An arch-gated launch is CPU-only for a stronger
                # reason still: the env block masked every device away ("-1").
                _deliberate_cpu_only = (
                    gpu_memory_mode == "manual" and gpu_layers == 0
                ) or _arch_gate_forced_cpu
                # Published on EVERY load, not only when true: load_model phase 1 only
                # kills the old process, so a stale value would answer holds_no_vram
                # for the child that replaced it. A host that gains coverage (a
                # llama.cpp update) would then report a VRAM-holding server as holding
                # none, and the arbiter leave it unclaimed.
                self._arch_gate_forced_cpu = bool(_arch_gate_forced_cpu)
                if _arch_gate_forced_cpu:
                    # False, never None: _classify_gpu_offload answers None here (the
                    # gated probe left _detected_gpus empty) and training spares only an
                    # exactly-False server, so None unloads one that frees no VRAM.
                    self._gpu_offload_active = False
                elif _deliberate_cpu_only:
                    self._gpu_offload_active = self._zero_offload_gpu_flag(
                        _last_spawn_cmd, _detected_gpus, env
                    )
                else:
                    self._gpu_offload_active = self._classify_gpu_offload(
                        gpu_indices is not None or use_fit or gpu_memory_mode == "manual",
                        _detected_gpus,
                    )
                # CPU-only hosts log 0/N too, so a zero needs an expected GPU; a positive count stands.
                _offload_counts = (
                    None
                    if _arch_gate_forced_cpu or _deliberate_cpu_only
                    else parse_gpu_offload_counts(self._stdout_lines)
                )
                if (
                    _offload_counts is not None
                    and _offload_counts[0] <= 0
                    and not (
                        (_detected_gpus or _metal_capable_host())
                        and (gpu_indices is not None or use_fit or gpu_memory_mode == "manual")
                    )
                ):
                    _offload_counts = None
                self._gpu_offload_layers = _offload_counts
                self._gpu_backend_unavailable = bool(_detected_gpus or _metal_capable_host()) and (
                    llama_saw_gpu_device(self._stdout_lines) is False
                )
                # Not _GPU_OFFLOAD_OVERRIDE_FLAGS: --fit on is exactly the split to report.
                self._offload_overridden = (
                    _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
                    or _device_selection_is_cpu(
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args,
                        env,
                    )
                    or _env_fixes_gpu_layers(env)
                )
                if (
                    self._gpu_offload_layers is not None
                    and 0 < self._gpu_offload_layers[0] < self._gpu_offload_layers[1]
                ):
                    logger.warning(
                        "llama-server offloaded %d of %d layers to GPU; the rest run on "
                        "CPU, which is much slower for a dense model.",
                        self._gpu_offload_layers[0],
                        self._gpu_offload_layers[1],
                    )
                if self._gpu_offload_active is False and not _deliberate_cpu_only:
                    logger.warning(
                        "llama-server appears to have loaded the model entirely "
                        "on CPU even though Unsloth detected at least one GPU. "
                        "This usually means the prebuilt binary's GPU backend "
                        "failed to load -- on Windows, cudart64_X.dll / "
                        "cublas64_X.dll could not be resolved. Reinstall the "
                        "Unsloth llama.cpp prebuilt or install a matching CUDA "
                        "toolkit (issue unslothai/unsloth#5106).",
                    )

                logger.info(
                    f"llama-server ready on port {self._port} for model '{model_identifier}'"
                )
                # Poll llama-server /metrics -> vLLM-style engine_stats logs
                # (only when the binary exposes /metrics).
                if server_caps.get("supports_metrics"):
                    try:
                        from core.inference.llama_stats import maybe_start_stats_logger
                        if self._stats_logger is not None:
                            self._stats_logger.stop()
                        self._stats_logger = maybe_start_stats_logger(
                            self.base_url, logger, headers = self._auth_headers
                        )
                    except Exception as e:
                        logger.debug(f"engine-stats logger not started: {e}")
                else:
                    self._stats_logger = None

            # Probe outside _lock (interruptible by /unload); init inside.
            self._is_audio = False
            self._audio_type = None
            self._audio_probed = False
            self._has_audio_input = False
            try:
                detected = self._detect_audio_type_strict()
                self._audio_probed = True
            except Exception as exc:
                logger.debug("Audio probe failed: %s", exc)
                detected = None
            applied_audio = (
                self._apply_detected_audio(detected)
                if intent.audio_codec_path is None
                else self._apply_detected_audio(detected, intent.audio_codec_path)
            )
            if not applied_audio:
                return False

            if not self._healthy:
                return False
            if _load_cancelled():
                # Cancelled during the post-health setup or the audio probe. Publishing
                # now leaves the child resident while the cancel route sees is_loaded.
                _cleanup_cancelled_load(
                    "Load cancelled after llama-server became healthy; unwinding"
                )
                return False
            # Snapshot the files the server actually loaded. If a GGUF shard or a
            # LoRA/control-vector sidecar is swapped on disk afterwards while the
            # old weights stay mapped, save_slots_for_resume() compares against
            # this and refuses to persist KV that a reload could misapply.
            if self._slot_save_dir:
                self._slot_loaded_identity = (
                    self._gguf_file_identity(self._gguf_path),
                    self._slot_launch_fingerprint(),
                )
            return True

    def _build_speculative_flags(
        self,
        *,
        speculative_type: Optional[str],
        spec_draft_n_max: Optional[int],
        extra_args: Optional[List[str]],
        model_identifier: str,
        model_path: Optional[str],
        gpus: bool,
        binary: Optional[str],
        mtp_draft_path: Optional[str] = None,
        dspark_draft_path: Optional[str] = None,
        dspark_fit_sized: bool = True,
        dflash_draft_path: Optional[str] = None,
        dflash_fit_sized: bool = True,
        drafter_no_vram: bool = False,
        embedded_mtp_partial_offload: bool = False,
        draft_device: Optional[str] = None,
        mtp_drafter_unloadable: bool = False,
        eagle3_draft_path: Optional[str] = None,
        mtp_trained: Optional[bool] = None,
    ) -> List[str]:
        """Return the llama-server flag list for the requested spec mode.

        Side effects: sets ``self._speculative_type`` (resolved internal
        emit), ``self._requested_spec_mode`` (canonical UI mode for the
        status round-trip), and ``self._spec_draft_n_max`` (user override
        only; None when the platform default applies).

        Speculative decoding (n-gram self-speculation, zero VRAM):
        ngram-mod uses a ~16 MB shared hash pool, constant memory /
        complexity, variable draft lengths. Helps most when the model
        repeats existing text (code refactor, summarisation, reasoning);
        for low-repetition chat, overhead is ~5 ms.

        Benchmarks from upstream llama.cpp speculative-decoding PRs:
          Scenario                        | Without | With    | Speedup
          gpt-oss-120b code refactor      | 181 t/s | 446 t/s | 2.5x
          Qwen3-235B offloaded            |  12 t/s |  21 t/s | 1.8x
          gpt-oss-120b repeat (92% accept)| 181 t/s | 814 t/s | 4.5x
        Refs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
              https://github.com/ggml-org/llama.cpp/pull/19164
              https://github.com/ggml-org/llama.cpp/pull/18471
              MTP guide: unsloth.ai/docs/models/qwen3.6#mtp-guide

        Sub-3B dense MTP regresses vs spec-off when the head is baked into the
        main GGUF (Qwen): the draft head's per-token cost exceeds the
        acceptance savings at this scale. Q4_K_XL clean bench (each prompt once
        after an unrelated warmup) on B200 + x86 CPU:
          0.8B GPU: draft-mtp n=2 = 0.58x vs OFF; ngram-only = 1.10x
          2B   GPU: draft-mtp n=2 = 0.82x vs OFF; OFF or ngram = 1.00x
          0.8B CPU: chained n=2   = 0.86x vs OFF; ngram-only = 1.19x
          2B   CPU: chained n=2   = 0.83x vs OFF; ngram-only = 1.01x
          4B+ GPU/CPU: spec on is a net win (1.08x-1.46x).
        A separate drafter (Gemma's root mtp-*.gguf) is a different, cheaper
        mechanism that wins even below 3B, so it is exempt from the sub-3B drop
        (``mtp_draft_path`` set -> not too small). B200 Q4_K_XL bench, draft-mtp
        n=2 vs OFF: gemma-4-E2B (2B) = 1.21x, accept ~0.65 (vs ngram = 1.00x);
        gemma-4-E4B (4B) and 12B engage as usual.
        Auto falls back to ngram-mod (zero-VRAM, near-zero idle cost on
        diverse content) for an embedded sub-3B head; forced MTP on a model
        with no head/drafter defaults back (mtp -> spec-default, mtp+ngram ->
        ngram-mod) since llama-server aborts otherwise; a drafter the binary
        cannot build (older prebuilt, or a CUDA kernel limit) aborts the spawn
        and the load retries once without speculative decoding.
        """
        flags: List[str] = []
        # Reset; emit branches re-set on the resolved emission.
        self._spec_draft_n_max = None
        self._speculative_type = None
        self._spec_fallback_reason = None
        # A drafter trained with the fine-tune and exported beside this GGUF (drafter.json).
        trained_steps = None
        if eagle3_draft_path is None or mtp_trained is None:
            sidecar_eagle3, sidecar_trained, trained_steps = _trained_drafter_sidecar(model_path)
            eagle3_draft_path = eagle3_draft_path or sidecar_eagle3
            mtp_trained = sidecar_trained if mtp_trained is None else mtp_trained

        # Canonical UI-facing requested mode (legacy values mapped via
        # _canonicalize_spec_mode).
        canonical_mode = _canonicalize_spec_mode(speculative_type)
        user_owns_spec_type = _extra_args_set_spec_type(extra_args)
        if (
            not user_owns_spec_type
            and canonical_mode in ("auto", "mtp", "mtp+ngram")
            and self._nextn_predict_layers
            and mtp_draft_path
        ):
            # Backstop for callers that bypass load_model's discovery normalization.
            mtp_draft_path = None
        # MTP signals: head baked into the main GGUF (Qwen, via metadata or
        # name), or a separate drafter resolved from the repo (Gemma).
        is_mtp_model = (
            bool(self._nextn_predict_layers)
            or _is_mtp_model_name(model_identifier, model_path)
            or bool(mtp_draft_path)
            # A dropped sidecar was this quant's ONLY MTP signal: the motivating
            # RVN-Q6_K.gguf has no nextn_predict_layers and no -mtp in its name, so
            # clearing mtp_draft_path read as a plain model, skipping every arm below and
            # leaving the drop unexplained. Kept as a signal, it reaches the fallback.
            or mtp_drafter_unloadable
        )
        _mtp_size_b = _extract_model_size_b(model_identifier)
        # The sub-3B regression is an embedded-head cost; a separate drafter
        # (Gemma) is a cheap standalone model that wins below 3B, so exempt it.
        # A head the user trained with this fine-tune is what they asked to draft with, so the
        # size floor (a stock-head cost measurement) and the MLA default-off below do not apply.
        _mtp_too_small = (
            _mtp_size_b is not None
            and _mtp_size_b < _MTP_MIN_SIZE_B
            and not bool(mtp_draft_path)
            and not mtp_trained
        )
        # Drafterless Gemma (name-only MTP, no embedded head), or any quant whose only
        # drafter was dropped as unopenable: emitting MTP would abort llama-server, so
        # every mode below falls back. Without the second case a dropped sidecar on any
        # non-Gemma repo disables MTP in silence.
        _mtp_drafter_missing = (
            not mtp_draft_path
            and not self._nextn_predict_layers
            and (_is_gemma_mtp_name(model_identifier, model_path) or mtp_drafter_unloadable)
        )
        # Embedded MTP head on an MLA model (GLM-5.2/DeepSeek/Kimi, detected by
        # kv_lora_rank): llama.cpp's MLA/DSA MTP path is ~2x slower than no spec,
        # so Auto drops it (override via the Settings dropdown / forced mtp, or
        # UNSLOTH_MLA_MTP_ENABLED=1). Separate drafters (Gemma, mtp_draft_path) and
        # non-MLA embedded heads (Qwen, no kv_lora_rank) are unaffected.
        _auto_mla_embedded_mtp = (
            bool(self._nextn_predict_layers)
            and self._kv_lora_rank is not None
            and not bool(mtp_draft_path)
            and not _mla_mtp_auto_enabled()
            and not _arch_has_fast_mla_mtp(getattr(self, "_architecture", None))
            and not mtp_trained
        )

        if user_owns_spec_type:
            # User --spec-type wins outright; suppress auto-emit to avoid a
            # duplicate spec block.
            self._requested_spec_mode = None
            return flags

        effective_mode = canonical_mode or "auto"
        self._requested_spec_mode = effective_mode
        if effective_mode == "off":
            return flags
        if effective_mode == "ngram-simple":
            flags.extend(["--spec-type", "ngram-simple"])
            self._speculative_type = "ngram-simple"
            return flags
        caps = self.probe_server_capabilities(binary)

        # Which drafter every branch below is about. Set once here, before any
        # fallback can erase the evidence: the UI names the recovery from this,
        # and under Auto neither the requested mode nor the resolved
        # _speculative_type ("default" after a fallback) still carries the kind.
        _auto_mode = (canonical_mode or "auto") == "auto"
        if canonical_mode == "eagle3" or (
            _auto_mode and eagle3_draft_path and caps.get("supports_eagle3")
        ):
            self._spec_drafter_kind = "eagle3"
            if not caps.get("supports_eagle3"):
                logger.warning(
                    "EAGLE-3 requested but llama-server lacks --spec-type draft-eagle3; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not eagle3_draft_path:
                logger.warning(
                    "EAGLE-3 requested but no trained eagle3-*.gguf drafter sits next to this "
                    "model; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            draft_n_max = (
                int(spec_draft_n_max) if spec_draft_n_max is not None else (trained_steps or 3)
            )
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = draft_n_max
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    eagle3_draft_path,
                    "--spec-type",
                    "draft-eagle3",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            if draft_device:
                flags.extend(["--spec-draft-device", draft_device])
            self._speculative_type = "draft-eagle3"
            logger.info("Spec decoding: draft-eagle3 using %s", eagle3_draft_path)
            return flags
        if canonical_mode == "dspark" or (
            _auto_mode and dspark_draft_path and caps.get("supports_dspark")
        ):
            self._spec_drafter_kind = "dspark"
        elif canonical_mode == "dflash" or (
            _auto_mode and dflash_draft_path and caps.get("supports_dflash")
        ):
            self._spec_drafter_kind = "dflash"
        else:
            self._spec_drafter_kind = "mtp"

        def _resolved_draft_n_max() -> int:
            # User override wins; else platform default (the B200 / x86
            # clean-sweep sweet spot from PR #5582 is n=2 GPU, n=3 CPU;
            # past 3 regresses on essay-style low-acceptance prompts).
            if spec_draft_n_max is not None:
                n = int(spec_draft_n_max)
                self._spec_draft_n_max = n
                return n
            if mtp_trained and trained_steps and self._spec_drafter_kind == "mtp":
                return trained_steps  # the depth the exported head was trained for
            return 2 if gpus else 3

        def _emit_dspark() -> None:
            """Append --model-draft <sidecar> --spec-type draft-dspark + n-max.

            Callers have already established the sidecar and the capability; Auto
            reaches this too, since DSpark is the default when both hold.
            """
            if not dspark_fit_sized:
                # Not a blocker: the sidecar ships no token_embd/output and borrows
                # the target's, so llama.cpp's fit step cannot build a standalone
                # draft context to measure it (llama-context.cpp:153-160) and skips
                # the reserve (server-context.cpp:1190-1193). The load still works;
                # only the ~11 GB is missing from the fit budget.
                logger.info(
                    "DSpark under --fit on: llama.cpp cannot size the sidecar during "
                    "fitting, so its ~%.1f GB is not reserved. Pin GPU Layers in "
                    "Manual mode if the load runs out of VRAM.",
                    (self._get_gguf_size_bytes(dspark_draft_path) or 0) / 1e9,
                )
            draft_n_max = int(spec_draft_n_max) if spec_draft_n_max is not None else 3
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = draft_n_max
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    dspark_draft_path,
                    "--spec-type",
                    "draft-dspark",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            self._speculative_type = "draft-dspark"
            logger.info("Spec decoding: draft-dspark using %s", dspark_draft_path)

        if effective_mode == "dspark":
            # Capability first: the fetch is gated on the same answer, so a missing
            # sidecar here is usually the binary's fault. Blaming the file instead
            # would tell the user to place one, and reload on every Apply.
            if not caps.get("supports_dspark"):
                logger.warning(
                    "DSpark requested but llama-server lacks --spec-type "
                    "draft-dspark; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not dspark_draft_path:
                logger.warning(
                    "DSpark requested but no matching dspark-*.gguf sidecar was found; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            _emit_dspark()
            return flags

        def _emit_dflash() -> None:
            """Append --model-draft <sidecar> --spec-type draft-dflash + n-max.

            Callers have already established the sidecar and the capability. Auto
            reaches this too, since DFlash is the default whenever the model
            ships one: unlike the ~11 GB DSpark sidecar it is ~1.5 GiB and comes
            from the model's own GGUF repo.
            """
            if not dflash_fit_sized:
                # Same shape as DSpark: the sidecar ships no token_embd/output and
                # borrows the target's, so llama.cpp cannot build a standalone
                # draft context to measure it (llama-context.cpp:153-160) and
                # skips the reserve (server-context.cpp:1190-1193). The load still
                # works; only the ~1.5 GB is missing from the fit budget, which is
                # an order of magnitude less than the DSpark case.
                logger.info(
                    "DFlash under --fit on: llama.cpp cannot size the sidecar during "
                    "fitting, so its ~%.1f GB is not reserved. Pin GPU Layers in "
                    "Manual mode if the load runs out of VRAM.",
                    (self._get_gguf_size_bytes(dflash_draft_path) or 0) / 1e9,
                )
            draft_n_max = _resolved_draft_n_max()
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    dflash_draft_path,
                    "--spec-type",
                    "draft-dflash",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            if draft_device:
                # A separate drafter, like the Gemma MTP one, so it takes the same
                # device pin (_emit_mtp does this); a baked-in head has nowhere to
                # put it and DSpark's sidecar predates the flag being threaded here.
                flags.extend(["--spec-draft-device", draft_device])
            self._speculative_type = "draft-dflash"
            logger.info("Spec decoding: draft-dflash using %s", dflash_draft_path)

        if effective_mode == "dflash":
            # Capability first, for the same reason as DSpark: the fetch is gated
            # on the same answer, so a missing sidecar here is usually the
            # binary's fault, and blaming the file would tell the user to place
            # one and reload on every Apply.
            if not caps.get("supports_dflash"):
                logger.warning(
                    "DFlash requested but llama-server lacks --spec-type "
                    "draft-dflash; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not dflash_draft_path:
                logger.warning(
                    "DFlash requested but no matching dflash-*.gguf sidecar was found; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            _emit_dflash()
            return flags

        def _emit_mtp(*, chain_ngram: bool) -> bool:
            """Append --spec-type mtp[/draft-mtp][,ngram-mod] + n-max."""
            mtp_token = caps.get("mtp_token") if caps else None
            if not mtp_token:
                inconclusive = bool(caps.get("mtp_probe_inconclusive")) if caps else True
                if inconclusive:
                    logger.info(
                        "Requested MTP speculative decoding but llama-server MTP "
                        "capability probe was inconclusive; loading without "
                        "speculative decoding."
                    )
                else:
                    logger.warning(
                        "Requested MTP speculative decoding but "
                        "llama-server lacks --spec-type mtp/draft-mtp; "
                        "run `unsloth studio update`. Loading without "
                        "speculative decoding."
                    )
                # Override an inherited LLAMA_ARG_SPEC_TYPE=draft-mtp (CLI wins
                # over env) so the child matches the binary-capability gate and
                # the no-MTP budget, like the sibling no-head/non-MTP fallbacks.
                flags.append("--spec-default")
                self._speculative_type = "default"
                if inconclusive:
                    self._spec_fallback_reason = None
                else:
                    self._spec_fallback_reason = "binary_no_mtp"
                return False
            draft_n_max = _resolved_draft_n_max()
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            # Separate-file drafter (Gemma): point llama-server at it. Baked-in
            # heads (Qwen) pass no path -- llama-server reads them from the
            # main GGUF.
            if mtp_draft_path:
                flags.extend(["--model-draft", mtp_draft_path])
                if draft_device:
                    flags.extend(["--spec-draft-device", draft_device])
                logger.info(f"Using separate MTP drafter: {mtp_draft_path}")
            spec_value = mtp_token
            ngram_knobs: list[str] = []
            if chain_ngram:
                ngram_knobs = _build_ngram_mod_flags(caps, chain_with_mtp = True)
                if ngram_knobs:
                    spec_value = f"ngram-mod,{mtp_token}"
                else:
                    logger.warning(
                        "llama-server lacks ngram-mod tuning "
                        "flags; loading MTP only (no ngram chain)"
                    )
            flags.extend(["--spec-type", spec_value, n_max_flag, str(draft_n_max)])
            flags.extend(ngram_knobs)
            self._speculative_type = "draft-mtp"
            chain_label = "chained ngram-mod" if chain_ngram else "MTP-only"
            logger.info(f"Spec decoding: {mtp_token} ({chain_label})")
            return True

        def _emit_ngram_mod() -> bool:
            """Append --spec-type ngram-mod + flag-set knobs."""
            if not caps.get("supports_ngram_mod"):
                logger.warning(
                    "Requested ngram-mod speculative decoding but llama-server "
                    "does not advertise ngram-mod support; run `unsloth studio "
                    "update`. Loading without speculative decoding."
                )
                # The stand-down the warning tells the user to fix, recorded so the
                # update actually takes: an unrecorded one leaves the picker adopting
                # this process forever. "binary_outdated" rather than "binary_no_mtp"
                # because the retry rule for that one asks _SPEC_KIND_CAPABILITY about
                # _spec_drafter_kind, which ngram-mod never sets, having no drafter.
                self._spec_fallback_reason = "binary_outdated"
                return False
            ngram_knobs = _build_ngram_mod_flags(caps)
            flags.extend(["--spec-type", "ngram-mod"])
            if not ngram_knobs:
                logger.warning(
                    "llama-server lacks ngram-mod tuning "
                    "flags; loading without --spec-ngram-mod-* knobs"
                )
            flags.extend(ngram_knobs)
            self._speculative_type = "ngram-mod"
            logger.info("Spec decoding: ngram-mod")
            return True

        def _fallback_drafter_not_found() -> None:
            """Drafterless Gemma: use ngram-mod (or spec-default) and record why."""
            if mtp_drafter_unloadable:
                logger.warning(
                    "Model %s had an MTP drafter that llama-server cannot open as "
                    "--model-draft; falling back without it.",
                    model_identifier,
                )
            else:
                logger.warning(
                    "Model %s is MTP-capable but no drafter or head was found; "
                    "falling back. Check network or run `unsloth studio update`.",
                    model_identifier,
                )
            if caps.get("supports_ngram_mod"):
                _emit_ngram_mod()
            else:
                flags.append("--spec-default")
                self._speculative_type = "default"
            # Its own reason: the drafter_not_found copy tells a local load to place a file
            # already on disk, and offers a refetch this load deliberately stands down.
            self._spec_fallback_reason = (
                "drafter_unloadable" if mtp_drafter_unloadable else "drafter_not_found"
            )

        if effective_mode == "ngram":
            _emit_ngram_mod()
            return flags
        if effective_mode in ("mtp", "mtp+ngram"):
            chain_ngram = effective_mode == "mtp+ngram"
            if not is_mtp_model:
                if chain_ngram:
                    logger.warning(
                        "MTP+Ngram requested but this GGUF has no MTP head or "
                        "drafter; loading ngram-mod only."
                    )
                    _emit_ngram_mod()
                else:
                    logger.warning(
                        "MTP requested but this GGUF has no MTP head or drafter; "
                        "loading without speculative decoding."
                    )
                    flags.append("--spec-default")
                    self._speculative_type = "default"
                return flags
            if _mtp_drafter_missing:
                _fallback_drafter_not_found()
                return flags
            if _mtp_too_small:
                logger.warning(
                    f"Forcing {effective_mode.upper()} on a {_mtp_size_b:.1f}B model; "
                    "the bench shows MTP regresses below 3B. "
                    "Engaging anyway (user override)."
                )
            _emit_mtp(chain_ngram = chain_ngram)
            return flags

        # Embedded Hybrid Mamba MTP shares the target model's device placement.
        # Under an automatic fit, its rollback states push more target layers to CPU;
        # at a fixed partial layer count, the draft and rollback work still regresses
        # decode. Auto therefore preserves ordinary llama-server performance by
        # disabling speculation. Forced MTP remains available as the explicit override.
        # Only where MTP is the thing being stood down. On a build whose --spec-type
        # has no MTP spelling the emit path below already loads without speculation
        # and names the real cause (binary_no_mtp, with the update affordance), so
        # claiming a placement policy would send the user to force a mode this build
        # cannot run, and "none" is not a value every such build's enum carries.
        if embedded_mtp_partial_offload and caps.get("mtp_token"):
            logger.info(
                "Auto: embedded MTP is disabled because the Hybrid Mamba target "
                "requires partial CPU offload; ordinary decoding is faster at this "
                "placement. Choose MTP explicitly to force it."
            )
            flags.extend(["--spec-type", "none"])
            self._speculative_type = "none"
            self._spec_fallback_reason = "mtp_partial_offload"
            # Record the depth this decision was made at (user override only, as
            # everywhere else). The rollback copies priced into the fit are
            # base_recurrent * spec_draft_n_max, so a changed depth can move the
            # placement and must reload -- and without a recorded value the
            # comparison in _runtime_matches_intent would see 0 forever and reload
            # on every Apply instead of once.
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = int(spec_draft_n_max)
            return flags

        # effective_mode == "auto": the promotion path. No vision gate on any drafter
        # kind. MTP's mmproj compatibility is llama.cpp #22673; DFlash is a different
        # --spec-type and #22105 says nothing, so it was measured: Muse-Glimmer-30B
        # UD-Q4_K_XL + mmproj-kquant + dflash-kquant, b10342, B200, n_max=2, greedy,
        # ~545 image tokens gave 92.1 -> 114.2 tok/s at 0.646 acceptance with output
        # byte-identical to the drafter-free run.
        # Which drafter Auto would have emitted had VRAM allowed. An MLA model with no
        # sidecar is already dropped by policy below, and reporting the VRAM reason
        # would invite forcing MTP at a smaller context: slower than the ngram-mod it
        # gets. Let it fall through to the MLA branch, which gives the real reason.
        _no_vram_drops_a_real_drafter = (
            bool(dspark_draft_path and caps.get("supports_dspark"))
            or bool(dflash_draft_path and caps.get("supports_dflash"))
            or not _auto_mla_embedded_mtp
        )
        if drafter_no_vram and _no_vram_drops_a_real_drafter:
            # The fit found room for the target but not for the drafter's reserve,
            # and reserved nothing for it, so emitting one now would OOM the load.
            # Same downgrade as the MLA branch below: ngram-mod costs no VRAM, and
            # spec-off when the build lacks it. The drafter paths stay recorded, so
            # the UI still names the kind and a repeat Apply still dedupes.
            self._spec_fallback_reason = "drafter_no_vram"
            if caps.get("supports_ngram_mod"):
                logger.info(
                    "Auto: the drafter does not fit in VRAM alongside the model at this "
                    "context, so it is dropped and ngram-mod (zero-VRAM) takes its "
                    "place. Choose the drafter in the Speculative Decoding dropdown to "
                    "force it at a smaller context."
                )
                _emit_ngram_mod()
            else:
                # spec-off: --spec-type ngram-mod is not a value this build's enum
                # carries, and llama-server aborts on one it cannot parse rather than
                # ignoring it. Mirrors the MLA and sub-3B branches below.
                logger.info(
                    "Auto: the drafter does not fit in VRAM alongside the model at this "
                    "context, so speculative decoding is disabled (this llama-server "
                    "does not advertise ngram-mod). Choose the drafter in the "
                    "Speculative Decoding dropdown to force it at a smaller context."
                )
        elif dspark_draft_path and caps.get("supports_dspark"):
            # DSpark first: load_model only hands a sidecar down once it has one
            # this binary can launch, and it beats every other Auto outcome for
            # this architecture (1.84x on 4x B200, 1.91x on one). Without it these
            # models fall through to --spec-default, i.e. no drafter at all.
            _emit_dspark()
        elif dflash_draft_path and caps.get("supports_dflash"):
            # DFlash next, on the same reasoning as DSpark and with the same
            # capability gate: load_model only hands a sidecar down once it has
            # one this binary can launch. Unlike DSpark this needs no opt-in: the
            # published sidecar is ~1.5 GiB and already sits in the model's own
            # GGUF repo, so Auto pays roughly what the MTP drafter costs. DSpark
            # keeps first refusal so a repo shipping both is unchanged.
            _emit_dflash()
        elif _auto_mla_embedded_mtp:
            # MLA embedded-MTP (GLM-5.2 et al.): the MTP path regresses vs spec-off
            # on llama.cpp today, so Auto drops it and falls back to ngram-mod (or
            # spec-off if unsupported), mirroring the sub-3B branch. Forced mtp /
            # mtp+ngram (handled above) still engage; UNSLOTH_MLA_MTP_ENABLED=1
            # re-enables this promotion once upstream optimizes the path.
            self._spec_fallback_reason = "mla_mtp_disabled"
            if caps.get("supports_ngram_mod"):
                logger.info(
                    "Auto: MLA embedded-MTP model detected; llama.cpp's MLA/DSA "
                    "MTP path is slower than no speculation, so using ngram-mod "
                    "instead. Override via the Unsloth Speculative Decoding "
                    "dropdown or UNSLOTH_MLA_MTP_ENABLED=1."
                )
                _emit_ngram_mod()
            else:
                logger.info(
                    "Auto: MLA embedded-MTP model detected; disabling speculative "
                    "decoding (this llama-server does not advertise ngram-mod). "
                    "Override via the dropdown or UNSLOTH_MLA_MTP_ENABLED=1."
                )
                # spec-off: emit nothing, mirroring the sub-3B no-ngram path.
        elif is_mtp_model and not _mtp_too_small:
            if _mtp_drafter_missing:
                # Name-only MTP, drafter did not resolve (download failed/absent).
                _fallback_drafter_not_found()
            else:
                # GPU: MTP-only. CPU/Mac: chain ngram-mod + MTP.
                _emit_mtp(chain_ngram = not gpus)
        elif is_mtp_model and _mtp_too_small:
            # Sub-3B fallback: drop the MTP draft head, keep ngram-mod when
            # the binary supports it.
            if _mtp_drafter_missing:
                _fallback_drafter_not_found()
            elif caps.get("supports_ngram_mod"):
                logger.info(
                    f"MTP GGUF detected but model size {_mtp_size_b:.1f}B "
                    "is below the 3B speedup threshold; using ngram-mod "
                    "only (zero-VRAM, no draft head). Override via "
                    "--spec-type or the Unsloth Speculative Decoding "
                    "dropdown."
                )
                _emit_ngram_mod()
            else:
                logger.info(
                    f"MTP GGUF detected but model size {_mtp_size_b:.1f}B "
                    "is below the 3B speedup threshold and the bundled "
                    "llama-server does not advertise ngram-mod; "
                    "auto-disabling speculative decoding."
                )
        else:
            # Non-MTP model: let llama-server choose its default strategy.
            flags.append("--spec-default")
            self._speculative_type = "default"
        return flags

    def adopt_load_intent_if_matched(self, intent: GgufLoadIntent) -> bool:
        """Match live state and adopt the caller's compatible GPU placement."""
        if not self.matches_load_source(intent):
            return False
        # A custom-path save leaves the model intent unchanged, but the
        # resident process still belongs to the previously selected runtime.
        if self._binary_changed_since_launch():
            logger.info("llama-server selection changed since launch; forcing a reload")
            return False
        intent = self._preserve_cpu_fallback_intent(intent, source_matches = True)
        # The stored state is what LAUNCHED, and on a virtualised Metal device that is
        # the CPU-pinned rewrite, not the request. Apply the same rewrite here
        # (idempotent, so load_model's own call is a no-op) or a client that keeps
        # sending Auto mismatches on every call.
        if _metal_device_is_paravirtual():
            (
                _pv_mode,
                _pv_layers,
                _pv_tp,
                _pv_split,
                _pv_cpu_moe,
                _pv_extras,
            ) = paravirtual_normalized_request(
                gpu_memory_mode = intent.gpu_memory_mode,
                gpu_layers = intent.gpu_layers,
                tensor_parallel = intent.tensor_parallel,
                tensor_split = list(intent.tensor_split) if intent.tensor_split else None,
                n_cpu_moe = intent.n_cpu_moe,
                extra_args = list(intent.extra_args) if intent.extra_args is not None else None,
                log_dropped = False,
            )
            intent = replace(
                intent,
                gpu_memory_mode = _pv_mode,
                gpu_layers = _pv_layers,
                tensor_parallel = _pv_tp,
                tensor_split = tuple(_pv_split) if _pv_split else None,
                n_cpu_moe = _pv_cpu_moe,
                extra_args = tuple(_pv_extras) if _pv_extras is not None else None,
            )
        effective_extra_args = (
            list(intent.extra_args) if intent.extra_args is not None else self._extra_args
        )
        candidate_extra_args = list(effective_extra_args) if effective_extra_args else []
        if intent.extra_args is not None and self._gpu_ids_own_placement(intent.gpu_ids):
            candidate_extra_args = self._strip_device_extra_args(candidate_extra_args)
        # A Model Memory toggle changes only the launch flags, so the intent is
        # unchanged and this would otherwise report already-loaded and leave the
        # running process on the old placement.
        if not memory_state_satisfies_settings(
            self._memory_state,
            self._memory_policy_active,
            self._memory_mlock_applicable,
            self._memory_direct_io,
            self._memory_dio_applicable,
            bool(self._memory_dio_flags),
        ):
            logger.info("Model Memory policy changed since launch; forcing a reload")
            return False
        # Same shape as the Model Memory check above: the budget is server-wide, so
        # a save leaves the intent identical and this would report already-loaded with
        # the child still on the old fraction. None means placement never used it
        # (manual, or no discrete GPU), where a reload would change nothing.
        _launched_frac = self._vram_fraction_launched
        # The pending marker, not a fresh read: a load in flight captured its budget
        # under the lock, and re-reading could decide against a different number than
        # the plan that follows. The route's fast path finds it unset and resolves
        # live as before. Reading the marker rather than taking a parameter keeps the
        # signature every caller and test double already has.
        _pending_frac = self._vram_fraction_pending
        _active_frac = _pending_frac if _pending_frac is not None else _active_vram_fraction()
        if _launched_frac is not None and float(_launched_frac) != _active_frac:
            logger.info("VRAM budget changed since launch; forcing a reload")
            return False
        if self._hand_added_projector_since_launch(candidate_extra_args):
            logger.info("A projector was added beside the cached weight; forcing a reload")
            return False
        if not self._runtime_matches_intent(intent, candidate_extra_args):
            return False
        self._record_matching_gpu_request(
            list(intent.gpu_ids) if intent.gpu_ids is not None else None
        )
        return True

    def _hand_added_projector_since_launch(self, extra_args: list) -> bool:
        """A text-only -hf launch whose repo publishes no projector, with one now
        dropped beside the cached weight (#9286). Without this the next Apply dedupes
        onto the projector-less server until Studio restarts."""
        last = getattr(self, "_last_load_intent", None)
        if (
            self._is_vision
            or self._disable_vision
            or not self._hf_repo
            or not self._gguf_path
            or last is None
            or last.is_vision
            or extra_args_disable_mmproj(extra_args)
        ):
            return False
        try:
            from utils.models.model_config import _hf_cached_local_mmproj
            return _hf_cached_local_mmproj(self._gguf_path) is not None
        except Exception as exc:
            logger.debug("Hand-added mmproj probe failed: %s", exc)
            return False

    def matches_load_source(self, intent: GgufLoadIntent) -> bool:
        """Whether the resident model has the intent's identity and weights."""
        if not self.is_loaded or (self._model_identifier or "").lower() != (
            intent.model_identifier.lower()
        ):
            return False
        custom_mmproj = _extra_args_device(intent.extra_args, {"--mmproj", "-mm"})
        source_path = intent.gguf_path
        if source_path is None and custom_mmproj is not None:
            if (self._hf_variant or "").lower() != (intent.hf_variant or "").lower():
                return False
            source_path = self._gguf_path
        if source_path is not None and self._gguf_path:
            resident_identity = getattr(self, "_gguf_load_identity", None)
            if resident_identity is not None:
                candidate_identity = LlamaCppBackend._gguf_load_source_identity(
                    source_path,
                    custom_mmproj or intent.mmproj_path,
                )
                return candidate_identity == resident_identity
            try:
                return Path(self._gguf_path).resolve() == Path(source_path).resolve()
            except OSError:
                return False
        return (self._hf_variant or "").lower() == (intent.hf_variant or "").lower()

    def components_match_intent(self, intent: GgufLoadIntent) -> bool:
        """``_runtime_matches_intent`` minus capacity and placement: same weights and components."""
        if intent.force_reload or not self.matches_load_source(intent):
            return False
        requested = self.requested_extra_args
        # Inherited extras would carry the resident's adapters and drafters unnamed by the caller.
        if intent.extra_args_inherited and requested:
            return False
        extras = requested if intent.extra_args_inherited else intent.extra_args
        if tuple(extras or ()) != tuple(requested or ()):
            return False
        if (self._chat_template_override or None) != (intent.chat_template_override or None):
            return False
        if not self._is_diffusion and (
            bool(self._disable_vision) != bool(intent.disable_vision)
            or self._requested_reasoning_budget
            != resolve_reasoning_budget(extras, intent.reasoning_budget)
            or self._requested_reasoning_budget_message
            != resolve_reasoning_budget_message(extras, intent.reasoning_budget_message)
        ):
            return False
        if _extra_args_set_spec_type(extras):
            return self._requested_spec_mode is None
        return (_canonicalize_spec_mode(intent.speculative_type) or "auto") == (
            self._requested_spec_mode or "auto"
        )

    def _classify_gpu_offload(
        self, expected_gpu: bool, detected_gpus: list[tuple[int, int]]
    ) -> Optional[bool]:
        """True if the model landed on a GPU, False if only CPU buffers landed
        despite GPU intent, None when there's no signal. Delegates to the shared
        classifier so it tracks current llama.cpp logs (offloaded-layer counts /
        device_info), not just the older "model buffer size" lines."""
        if not detected_gpus or not expected_gpu:
            return None
        return classify_gpu_offload_lines(self._stdout_lines)

    @staticmethod
    def _cmd_has_gpu_companion(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """True when the argv/env carries a GPU companion: any --mmproj form, or
        a drafter (Unsloth's --model-draft, the extras aliases, or the
        LLAMA_ARG_SPEC_DRAFT_* env) -- these offload to the GPU regardless of
        the main ``--gpu-layers``. A drafter explicitly forced to CPU
        (--spec-draft-ngl 0 / --spec-draft-device cpu) doesn't count."""
        _args = [str(a) for a in cmd]
        if any(a.startswith("--mmproj") or _flag_name(a) == "-mm" for a in _args):
            # --no-mmproj-offload clears mmproj_use_gpu and clip.cpp gates the whole
            # GPU backend on it, so the projector holds no VRAM. Last flag wins.
            _off = [a for a in _args if a in ("--mmproj-offload", "--no-mmproj-offload")]
            if not _off or _off[-1] != "--no-mmproj-offload":
                return True
        if _extra_args_mtp_draft_path(cmd, env) is None:
            return False
        return not _extra_args_draft_offloaded_to_cpu(cmd, env)

    @staticmethod
    def _zero_offload_keeps_gpu_visible(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """Whether a zero-layer launch still has a reason to use visible GPUs.

        Keep this shared by child masking and post-launch residency bookkeeping:
        a device pin, surviving tensor mode, mmproj, or GPU drafter prevents the
        launch from being a confirmed zero-VRAM server.
        """
        return (
            LlamaCppBackend._cmd_has_gpu_device_pin(cmd, env)
            or _effective_tensor_parallel(cmd, False, env)
            or LlamaCppBackend._cmd_has_gpu_companion(cmd, env)
        )

    @staticmethod
    def _cmd_has_gpu_device_pin(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """True when the effective main or draft ``--device`` pin names a GPU."""
        main_flags = {"--device", "-dev"}
        draft_flags = {"--spec-draft-device", "-devd", "--device-draft"}
        last_main: Optional[str] = None
        last_draft: Optional[str] = None
        args = [str(arg) for arg in cmd]
        for index, raw in enumerate(args):
            flag = _flag_name(raw)
            _, equals, inline = raw.partition("=")
            if flag not in main_flags and flag not in draft_flags:
                continue
            value = inline if equals else (args[index + 1] if index + 1 < len(args) else "")
            if flag in main_flags:
                last_main = value
            else:
                last_draft = value
        if last_main is None:
            last_main = (env or {}).get("LLAMA_ARG_DEVICE")

        def _names_gpu(value: Optional[str]) -> bool:
            if value is None:
                return False
            devices = [item.strip().lower() for item in value.split(",") if item.strip()]
            return not devices or any(item not in ("cpu", "none") for item in devices)

        return _names_gpu(last_main) or _names_gpu(last_draft)

    @staticmethod
    def _zero_offload_gpu_flag(
        spawn_cmd: list,
        detected_gpus: list,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional[bool]:
        """GPU-residency flag for a deliberate manual zero-offload load. The
        main model is CPU-only by construction, but device pins, tensor mode,
        mmproj, and GPU drafters can still make the server hold VRAM. The counted
        offload classifier cannot see those allocations. This uses the same
        predicate as the launch-time zero-VRAM mask; None means no GPU signal."""
        if LlamaCppBackend._is_vulkan_backend():
            # Unlike -ngl 0, the recovery's last-wins --device none removes Vulkan.
            main_device = (_extra_args_main_device(spawn_cmd) or "").strip().lower()
            if main_device in ("cpu", "none"):
                return LlamaCppBackend._zero_offload_keeps_gpu_visible(spawn_cmd, env)
            if detected_gpus:
                return True
        if not detected_gpus:
            return None
        return LlamaCppBackend._zero_offload_keeps_gpu_visible(spawn_cmd, env)
