# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local diffusion (text-to-image) backend.

A torch-only singleton that loads one of three kinds (see ``resolve_model_kind``): a single-file
GGUF transformer dequantised on-device via ``GGUFQuantizationConfig``, a single-file safetensors
transformer (e.g. fp8), or a full diffusers pipeline via ``from_pretrained`` (which re-applies an
embedded quant config such as bnb-4bit). The single-file kinds pull the rest of the pipeline (VAE,
text encoders, scheduler) from the matching base repo; the pipeline kind pulls everything from the
repo itself. Pipeline loads are gated to the ``unsloth/*`` org, the official bases, or a local path
for safety; a single ``.safetensors`` file is trusted per file from any repo (see
``diffusion_single_file_trust``).

torch/diffusers are imported lazily so this stays importable in a no-torch runtime. ``begin_load``
runs on a background thread; poll ``load_progress`` for the download bar. GPU-handoff policy lives
in the arbiter the routes call, not here.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import functools
import inspect
import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from core.inference.inference_gate import holds_inference, pauses_background_embedding
from core._torchao_stub import (
    alias_torchao_fqn_to_config,
    hide_xformers_built_for_another_torch,
    install_torchao_windows_rocm_stub,
    install_xformers_windows_rocm_stub,
)
from hub.utils.hf_errors import modelscope_missing
from hub.utils.hf_tokens import normalize_token
from loggers import get_logger
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
from utils.account_context import account_thread, current_account_id
from utils.hardware import clear_gpu_cache

from .diffusion_content import (
    assert_local_pick_is_dit,
    content_variant_hint,
    local_pick_file,
    whole_pipeline_gguf_family,
)
from .diffusion_gguf_pipeline import load_whole_pipeline_gguf, whole_pipeline_gguf_resident_mib
from . import hub_reachability
from .diffusion_families import (
    SMALL_DIT_BF16_WHEN_RESIDENT,
    small_dit_rule_excludes,
    named_variant_base,
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    IDEOGRAM4_FAMILY_NAME,
    LUMINA2_FAMILY_NAME,
    DiffusionFamily,
    DiffusionModelReplacedError,  # re-exported: callers import it from either module
    LoadIdentity,
    load_identity,
    local_pipeline_components_are_complete,
    _family_override_resolved,
    assert_flux2_gguf_matches_base,
    assert_flux3_checkpoint_matches_family,
    assert_pipeline_class_available,
    _is_local_path,
    _MIRROR_UPSTREAM,
    canonical_base,
    cache_holds_files,
    comfy_flow_shift_for,
    comfy_shift_node_for,
    default_generation_params,
    generation_params_with_grid,
    detect_family_for_pick,
    excluded_model_reason,
    prefer_ungated_mirror,
    mirror_hosts_artifacts,
    prequant_only_repo_ids,
    prequant_repo_role,
    resolve_base_repo,
    resolve_local_gguf_child,
    supported_family_names,
    family_for_checkpoint,
    transformer_config_overrides_for,
    transformer_variant_differs_from_base,
)
from .diffusion_compat import (
    assert_flux2_pick_compatible,
    assert_flux3_pick_compatible,
    assert_pick_is_not_speech,
    flux2_pick_mismatch,
    flux3_pick_mismatch,
    speech_pick_refusal,
)
from .diffusion_device import (
    DiffusionDeviceTarget,
    apply_diffusion_device_ordinal,
    diffusion_device_scope,
    diffusion_device_target_from_torch_device,
    install_frame_pad_fix,
    install_vae_sync,
    pin_cuda_ordinal,
    placed_cuda_ordinal,
    auto_cuda_ordinal_for_load,
    resolve_diffusion_device_target,
    resolve_selected_cuda_ordinal,
    resolve_selected_cuda_ordinals,
)
from .diffusion_cfg_batch import cfg_batch_status, end_cfg_batch_render, install_cfg_batch
from .diffusion_zimage_cfg_split import (
    install_zimage_cfg_split,
    split_aware_request_extra,
    uninstall_zimage_cfg_split,
    zimage_cfg_split_status,
)
from .diffusion_zimage_cfg_split import render as zimage_cfg_split_render
from .diffusion_prefix_kv import release_prefix_kv, track_prefix_kv
from .diffusion_multigpu import (
    AUTOMATIC_ORDINALS,
    end_cfg_render,
    release_cfg_prefix_kv,
    drop_displaced_cfg_router,
    install_cfg_parallel,
    multigpu_mode,
    invalidate_cfg_replica,
    placement_status as multigpu_placement_status,
    refine_plan_for_multigpu,
    uninstall_cfg_parallel,
    with_load_selection,
)
from .diffusion_conditioning import (
    LocalizedEdit,
    check_conditioned_fields,
    check_output_size,
    conditioning_capabilities,
    EDIT_EXTRA_IMAGE_AREA,
    decode_condition_images,
    edit_takes_extra_images,
    effective_reference_resolution,
    match_source_size,
)
from .diffusion_ideogram4 import ideogram4_repo_is_fp8, load_ideogram4_pipeline
from .diffusion_hidream import (
    HIDREAM_FAMILY_NAME,
    HIDREAM_LLAMA_BF16_BYTES,
    HIDREAM_LLAMA_REPO,
    hidream_te4_kwargs,
)
from .diffusion_krea2 import KREA2_FAMILY_NAME, load_krea2_pipeline
from .diffusion_families import MOSS_IMAGE_FAMILY_NAME
from .diffusion_flux3 import (
    FLUX3_COMPONENTS,
    FLUX3_COMPONENT_FOLDERS,
    FLUX3_INDEX,
    is_flux3_family,
    load_flux3_pipeline,
)
from .diffusion_flux3 import REFERENCE_AREA as FLUX3_REFERENCE_AREA
from .diffusion_small_host import (
    INT8_DENOISER_NAMES,
    decide_small_host,
    engaged_on as small_host_engaged_on,
    host_commit_mib,
    host_ram_mib,
    resolve_snapshot_dir,
    stored_components,
    torch_dtype_map as small_host_torch_dtype_map,
)
from .model_ids import hf_cache_repo_id
from .safetensors_pread import pread_available
from .diffusion_memory import (
    commit_load_budget_fraction,
    release_load_budget_fraction,
    vram_budget_scope,
    carry_load_budget,
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_LOW_VRAM,
    ACTIVATION_TILE,
    DeviceMemory,
    ImageActivationShortfallError,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_STREAMING,
    apply_memory_plan,
    calibrated_image_activation,
    engage_vae_tiling,
    estimate_gguf_resident_mib,
    estimate_image_runtime_mib,
    estimate_safetensors_dense_mib,
    file_size_mib,
    largest_streamable_companion_mib,
    normalize_memory_mode,
    plan_diffusion_memory,
    plan_fits_total_capacity,
    denoisers_pinned_resident,
    install_encode_release,
    plan_keeps_transformer_resident,
    prequant_seed_device,
    raise_on_image_activation_shortfall,
    raise_on_unified_memory_shortfall,
    scoped_allow_oversized_load,
    UNIFIED_OVERSIZE_REFUSAL_CODE,
    UnifiedMemoryShortfallError,
    unified_capacity_memory,
    unified_memory_refusal_code,
    reclaimable_snapshot_device_memory,
    reclaim_host_memory,
    reclaim_offload_host_memory,
    release_pinned_host_memory,
    request_background_encoder_pins,
    finish_encoder_pins_at_load,
    start_background_pins,
    stop_background_pins,
    refine_balanced_plan_for_components,
    refine_memory_plan_for_components,
    refine_plan_from_loaded_weights,
    prefer_whole_module_offload,
    whole_tier_request_costs,
    pin_denoiser_for_call,
    measured_image_runtime_mib,
    WHOLE_TIER_ATTR,
    learned_peak_label,
    CFG_OFF_PART,
    cfg_off_peaks_enabled,
    TIER_MOVED_ATTR,
    tier_peaks_enabled,
    without_cfg_off,
    live_vae_saver_tag,
    planned_vae_saver_tag,
    stream_window_pad_mib,
    with_label_part,
    applied_plan,
    release_resident_groups,
    release_for_decode,
    resident_group_mib,
    hook_resident_denoiser,
    measured_request_extra_mib,
    DECODE_LEND_ATTR,
    decode_lend_from_plan,
    with_workflow,
    settled_snapshot_device_memory,
    snapshot_device_memory,
    _torchao_stream_pinnable,
    pipeline_host_mib,
    torchao_offload_plan,
    torchao_scheme_streams,
    torchao_streaming_plan,
    torchao_survives_plan,
    unified_memory_shortfall_message,
    vae_tile_side,
    vae_can_slice,
    vae_is_sliced,
)
from .diffusion_torchao_patches import install_torchao_int_mm_patch
from . import diffusion_hires, diffusion_upscalers
from .diffusion_hires import HiresPass, with_hires
from .image_orientation import exif_upright
from .mcp_images import flattened_rgb
from .media_decode_phase import decode_phase, denoise_phase
from .diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_EAGER,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    engage_pinned_denoisers,
    auto_dynamic_active,
    _gguf_dequant_compilable,
    compile_eligible,
    compiled_shapes_are_static,
    rocm_probe_load_scoped,
    family_compiles_regionally,
    fp16_compile_explicit_only,
    fp16_unet_offloaded,
    fresh_compile_count,
    int8_gemm_live,
    arm_graphs_after_placement,
    normalize_speed_mode,
    resolve_speed_mode,
    family_render_cudnn_scope,
    restore_backend_flags,
    settle_compile_fallback,
    snapshot_backend_flags,
)
from .diffusion_offload_prefetch import cross_forward_render, cross_forward_step
from .diffusion_vae_fp16 import enable_fp16_vae_decode
from .diffusion_vae_upcast import pipeline_upcasts_keep_storage
from .diffusion_vae_tiling import install as install_wide_vae_tiles
from .diffusion_attention import (
    apply_attention_backend,
    attention_status_reason,
    auto_attention_reason,
    normalize_attention_backend,
    sdpa_math_only,
    sdpa_subquadratic_confirmed,
    attention_may_run_native,
    begin_attention_run,
    report_attention_run,
    select_attention_backend,
    _dit_head_dims,
    _ensure_attention_backend_installed,
)
from . import diffusion_compile_cache as compile_cache
from .diffusion_compile_config import family_filters_reductions
from . import diffusion_cond_cache as cond_cache
from . import diffusion_prompt_cache as prompt_cache
from . import diffusion_embeddings
from . import diffusion_te_release as te_release
from . import diffusion_gguf_compile as gguf_compile
from . import diffusion_bg_compile as bg_compile
from . import diffusion_cuda_graph as cuda_graph
from .diffusion_block_graph import compile_below_hooks_enabled, compile_pipe_below_offload_hooks
from . import diffusion_render_thread as render_thread
from .diffusion_postprocess import install as install_device_postprocess
from .diffusion_batched import (
    chunk_jobs,
    is_oom_error,
    resolve_batch_jobs,
    split_chunk,
    uniform_prompt,
)
from .diffusion_cache import (
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    TC_FBCACHE,
    TC_STATIC,
    apply_step_cache,
    auto_static_skip_plan,
    auto_step_cache_mode,
    auto_step_cache_allowed,
    skip_tier,
    cache_breaks_graph,
    effective_denoise_steps,
    effective_request_strength,
    maybe_toggle_step_cache,
    normalize_transformer_cache,
    resolve_auto_step_cache,
    step_cache_supported,
)
from .diffusion_step_skip import (
    auto_static_settings,
    install_static_step_skip,
    mark_step_end,
    reset_static_step_skip,
    static_skip_is_auto,
    static_skip_stats,
    static_skip_view,
    uninstall_static_step_skip,
)
from .diffusion_nvfp4_protect import protect_generation
from .diffusion_precision import (
    effective_te_quant,
    preflight_te_quant,
    normalize_te_quant,
    quantize_text_encoders,
    resolve_te_quant_request,
    te_quant_needs_resident_weights,
    te_quant_supported,
    te_quant_unsupported_reason,
    torchao_quantize_importable,
)
from .diffusion_te_prequant import supplied_component_pipe_kwargs, te_prequant_pipe_kwargs
from .diffusion_fast_load import _snapshot_dir as _load_snapshot_dir
from .diffusion_fast_load import start_load_prefetch, stop_prefetch, te_precast_components
from .diffusion_hosted_vae import hosted_vae_pipe_kwargs, hosted_vae_plan_files
from .diffusion_fp16_variant import (
    fp16_variant_enabled,
    is_fp16_dtype,
    is_fp16_twin,
    local_fp16_variant,
    plan_fp16_twins,
    twin_of,
)
from .diffusion_flow_shift import (
    SAMPLE_SIGMAS_KEY,
    apply_comfy_flow_shift,
    requested_flow_shift,
    scheduler_with_flow_shift,
    shift_control,
    install_sample_sigmas,
    pipe_sample_sigmas,
    sample_sigmas_for_steps,
)
from .diffusion_mask import (
    ONLY_MASKED_UNSUPPORTED_MSG,
    blur_mask,
    inpaint_class_accepts_padding_mask_crop,
    mask_is_empty,
    overlay_with_mask,
    resolve_padding,
)
from .diffusion_sampler import (
    DEFAULT_SAMPLER,
    apply_checkpoint_prediction_type,
    apply_pipeline_prediction_type,
    build_scheduler,
    sampler_options,
)
from . import diffusion_sd_controls as sd_controls_mod
from .diffusion_single_file_converters import (
    CONVERTERS as _ORIGINAL_LAYOUT_CONVERTERS,
    load_original_layout_transformer,
)
from .diffusion_text_length import (
    IDEOGRAM4_COMFY_GUIDANCE,
    ideogram4_comfy_guidance_schedule,
    ideogram4_comfy_mu_std,
    flux_t5_kwarg,
    true_cfg_needs_empty_negative,
)
from .diffusion_denoiser_prequant import (
    DENOISER_COMPONENT,
    PIPELINE_SEED_DECLINED,
    denoiser_prequant_cached,
    denoiser_prequant_source,
    pipeline_seed_supported,
    prequant_artifact_label,
)
from .diffusion_comfy_block import (
    comfy_block_backend,
    comfy_block_backends,
    comfy_block_runtime_layers,
    comfy_nvfp4_runtime_possible,
)
from .diffusion_comfy_quant import (
    header_has_fp8_weights,
    comfy_fp8_backend,
    comfy_fp8_storage,
    comfy_int8_backend,
    comfy_torchao_quantized,
    fp8_storage_wanted,
    fp8_widen_chosen,
    fp8_stored_layers,
    pipe_fp8_stored_layers,
    load_comfy_quant_transformer,
    refuse_comfy_quant,
)
from .diffusion_single_file_trust import assert_safetensors_file, single_file_load_allowed
from .diffusion_prequant import (
    PrequantIntoError,
    PrequantSource,
    hosted_fast_accum_conflict,
    load_prequantized_transformer,
    local_prequant_path_ready,
    prequant_checkpoint_cached,
    prequant_unreadable_reason,
    resolve_prequant_source,
    scoped_local_files_only,
    usable_prequant_source,
)
from .diffusion_auto_policy import (
    RESOLVED_APPLIED,
    RESOLVED_FELL_BACK,
    RESOLVED_UNSUPPORTED,
    base_repo_bf16_components_gb,
    build_resolved_record,
    estimate_dense_quant,
    format_generation_for_log,
    format_resolved_for_log,
    family_bf16_components_gb,
    precision_fallback_allowed,
    precision_refusal_message,
    companion_bytes_from_declared,
    resident_bytes_from_declared,
    resolve_dense_quant_candidate,
)
from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled
from .diffusion_nvfp4_install import nvfp4_backend_fields as _nvfp4_backend_fields
from .diffusion_transformer_quant import (
    TQ_AUTO,
    auto_bf16_when_resident_reason,
    TQ_NVFP4,
    TQ_INT8,
    DEFAULT_MIN_LINEAR_FEATURES,
    DenoiserView as _DenoiserView,
    dense_quant_blocker,
    dense_quant_supported_kind,
    dense_quant_unsupported_kind_reason,
    dense_transformer_supported,
    dense_transformer_unsupported_reason,
    denoiser_modules,
    explain_unusable_scheme,
    family_denies_scheme,
    mark_source_precision,
    nvfp4_gated_base,
    native_int8_act,
    native_offload_host,
    native_quant_host,
    native_quant_scheme,
    NATIVE_OFFLOAD_SCHEMES,
    NATIVE_QUANT_SCHEMES,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
    stored_denoiser_precision,
    transformer_is_quantised,
)
from .diffusion_native_quant import native_quant_reason
from utils.paths.path_utils import (
    any_not_appledouble_metadata,
    is_appledouble_metadata,
)

logger = get_logger(__name__)

_ZERO_THRESHOLD_NEGATIVE_PROMPT_FAMILIES = frozenset(("z-image", KREA2_FAMILY_NAME))


def _negative_prompt_guidance_threshold(family: DiffusionFamily) -> Optional[float]:
    """The guidance above which Studio passes a negative prompt; None when guidance does not gate it. Studio's rule,
    not the guider's: diffusers also runs CFG below 1, steering toward the negative prompt."""
    if family.cfg_kwarg == "true_cfg_scale":
        return 1.0
    if family.cfg_kwarg == "guidance_scale":
        return 0.0 if family.name in _ZERO_THRESHOLD_NEGATIVE_PROMPT_FAMILIES else 1.0
    return None


def _refuse_weight_change_on_streamed_graph(state: Any) -> None:
    """Recapturing a block-streamed graph after a LoRA weight change fails, and the eager fallback then fails every
    later generate, so such a load keeps its baked weights until a reload."""
    handles = getattr(state, "cuda_graphs", ()) or ()
    if any(getattr(getattr(h, "placement", None), "mode", None) == "group" for h in handles):
        raise ValueError(
            "This load streams the transformer through GPU memory, so its LoRA weights are fixed until the "
            "model is reloaded. Reload the model with the new adapter weights."
        )


def _state_shift_node(state: Any) -> Optional[str]:
    """The ComfyUI shift node (diffusion_flow_shift SHIFT_NODE_*) of the loaded checkpoint, or None."""
    return comfy_shift_node_for(
        state.family,
        getattr(state, "gguf_filename", None),
        getattr(state, "repo_id", None),
        getattr(state, "display_repo_id", None),
        getattr(state, "base_repo", None),
    )


def _guidance_supported(state: Any) -> bool:
    """FLUX.2 klein base and distilled share a family; Flux2KleinPipeline turns guidance off on ``is_distilled``."""
    if not state.family.uses_guidance:
        return False
    config = getattr(getattr(state, "pipe", None), "config", None)
    return not (
        state.family.name == "flux.2-klein" and bool(getattr(config, "is_distilled", False))
    )


def _live_preview_available(state: Any) -> bool:
    from .diffusion_preview import preview_available
    return preview_available(getattr(state.family, "name", None), getattr(state, "device", None))


def _negative_prompt_engaged(family: DiffusionFamily, guidance: float) -> bool:
    if not family.uses_negative_prompt:
        return False
    threshold = _negative_prompt_guidance_threshold(family)
    return threshold is None or float(guidance) > threshold


# Every `import diffusers` below is lazy, so this runs first. On Windows ROCm both reach an absent distributed
# backend: diffusers imports xformers on sight, its quantizers torchao.
install_xformers_windows_rocm_stub()
hide_xformers_built_for_another_torch()
install_torchao_windows_rocm_stub()
alias_torchao_fqn_to_config()
install_torchao_int_mm_patch()


# "gguf" and "single_file" take companions from the base repo; "pipeline" is a full diffusers repo.
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})


def _record_revision(out: Optional[dict[str, str]], repo_id: str, info: Any) -> None:
    """Remember the commit a ``model_info`` answer described, when the Hub reports one."""
    sha = getattr(info, "sha", None)
    if out is not None and isinstance(sha, str) and sha:
        out[repo_id] = sha


def _planned_progress_files(
    repo_id: str,
    gguf_filename: Optional[str],
    *,
    kind: str,
    base: str,
    fetch_base: str,
    base_files: Sequence[str],
    file_sizes: dict[str, dict[str, int]],
    dit_prequant: Optional[tuple[str, str, int]] = None,
    te_hub_files: Sequence[tuple[str, str, int]] = (),
) -> tuple[tuple[str, str, int], ...]:
    """``(repo, filename, size)`` for every file a diffusers load downloads, keyed by the repo the bytes LAND in: the
    selected checkpoint, the base files under the repo they are fetched from, the hosted prequant denoiser and the
    pre-cast encoders. One file named twice (a combined repo, a prequant in the companion repo) is listed once. The
    single source of a load's expected_bytes and of the progress scan."""
    out: dict[tuple[str, str], tuple[str, str, int]] = {}

    def add(repo: Optional[str], name: Optional[str], size: Any) -> None:
        if not repo or not name or _is_local_path(repo):
            return
        out.setdefault((repo.lower(), name), (repo, name, int(size or 0)))

    if kind != "pipeline" and gguf_filename:
        add(repo_id, gguf_filename, (file_sizes.get(repo_id) or {}).get(gguf_filename, 0))
    base_sizes = (
        file_sizes.get(base) or file_sizes.get(repo_id if kind == "pipeline" else base) or {}
    )
    for name in base_files:
        add(fetch_base or base, name, base_sizes.get(name, 0))
    if dit_prequant is not None:
        add(dit_prequant[0], dit_prequant[1], dit_prequant[2])
    for repo, name, size in te_hub_files:
        add(repo, name, size)
    return tuple(out.values())


def _sibling_blob_key(sibling: Any) -> Optional[str]:
    """The name huggingface_hub gives this file's blob (and its ``<key>.incomplete`` while downloading): the LFS
    sha256 for an LFS file, else the git blob id. None when the listing did not say."""
    lfs = getattr(sibling, "lfs", None)
    sha = (
        (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
        if lfs
        else None
    )
    key = sha or getattr(sibling, "blob_id", None)
    return str(key) if key else None


def _planned_blob_keys(
    planned: Sequence[tuple[str, str, int]],
    file_keys: dict[str, dict[str, str]],
    *,
    repo_id: str,
    base: str,
    fetch_base: str,
) -> dict[tuple[str, str], str]:
    """Blob keys for ``planned``, read off the listing that sized each file (the checkpoint under ``repo_id``, the
    base files under ``base`` while they land in ``fetch_base``). A file with no key is left out."""
    out: dict[tuple[str, str], str] = {}
    for repo, name, _size in planned:
        for source in dict.fromkeys((repo, repo_id, base) if repo == fetch_base else (repo,)):
            key = (file_keys.get(source) or {}).get(name)
            if key:
                out[(repo.lower(), name)] = key
                break
    return out


def _listing_extra(listing_repo: str, rfilename: str) -> bool:
    """A file an artifact mirror ships beside the upstream folders that no pipeline load reads (its sample renders);
    the root checkpoints are already excluded by every listing filter's top-level rule."""
    return mirror_hosts_artifacts(listing_repo) and rfilename.split("/", 1)[0] in (
        "samples",
        "assets",
    )


_GB_TO_MIB = (1000.0**3) / (1024.0 * 1024.0)
# Upstream releases Studio keeps a byte mirror of: their size is the family table's, not a fine-tune's.
_MIRRORED_UPSTREAMS: frozenset[str] = frozenset(
    u.strip().lower() for u in _MIRROR_UPSTREAM.values()
)


def _incomplete_release_cache_floor(
    fam: Any,
    repo_id: Optional[str],
    complete: bool,
    model_dense_mib: Optional[int],
    companion_mib: Optional[int],
    text_encoder_mib: Optional[int],
) -> Optional[tuple[int, int, int]]:
    """(model, companions, text encoders) MiB floored at the bf16 table for a mirrored release whose cache is incomplete.

    A downloading load plans before bytes land: a text-encoder-only cache priced FLUX.1-dev at 9.3 GB, then OOM'd on
    a 22 GB card. Fine-tunes keep their own bytes."""
    if complete or not repo_id:
        return None
    release = canonical_base(repo_id).strip()
    key = release.lower()
    if not (
        key == str(getattr(fam, "base_repo", "") or "").strip().lower()
        or key in _MIRRORED_UPSTREAMS
        or base_repo_bf16_components_gb(release) is not None
    ):
        return None
    table = family_bf16_components_gb(fam, release)
    if table is None:
        return None
    return (
        max(model_dense_mib or 0, int(sum(table) * _GB_TO_MIB)),
        max(companion_mib or 0, int(sum(table[1:]) * _GB_TO_MIB)),
        max(text_encoder_mib or 0, int(table[1] * _GB_TO_MIB)),
    )


def hub_cache_dir() -> str:
    """Loaders must pin this live root, since diffusers' import-time cache constant can split one load."""
    from utils.hf_cache_settings import active_hf_hub_cache
    return active_hf_hub_cache()


# Gated repos raise .../<owner>/<name>/resolve/... or .../api/models/<owner>/<name>; without the api branch the id would be api/models.
_HUB_REPO_RE = re.compile(
    r"huggingface\.co/(?:api/(?:models|datasets|spaces)/)?([\w.\-]+/[\w.\-]+)"
)


def _gated_in_chain(exc: BaseException) -> Optional[BaseException]:
    """The GatedRepoError in ``exc``'s cause/context chain, or None. Transformers loads re-raise the
    403 wrapped in an OSError, so the outermost error alone misses the case this exists for.
    Screened by class name across the MRO, so no hub import."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(cls.__name__ == "GatedRepoError" for cls in type(exc).__mro__):
            return exc
        # `raise ... from None` means the raiser already wrote a better message.
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return None


def _hf_token_in_play(hf_token: Optional[str]) -> bool:
    """Whether the failing Hub call carried ANY credential, not just Unsloth's own: with token=None
    huggingface_hub still falls back to HF_TOKEN or the cached CLI login, so keying off the
    request token alone loops an already-authenticated user."""
    if hf_token:
        return True
    try:
        # Under HF_HUB_DISABLE_IMPLICIT_TOKEN get_token() answers while the request goes out anonymous.
        from huggingface_hub.utils import get_token_to_send
        return bool(get_token_to_send(None))
    except Exception:  # noqa: BLE001 -- assume none; at worst the message says "add a token"
        return False


_DYNAMO_PARTIAL_RE = re.compile(
    r"partially initialized module 'torch\._dynamo'|"
    r"module 'torch\._dynamo' has no attribute 'utils'"
)


def dynamo_partial_init_message(exc: BaseException) -> Optional[str]:
    """Partial-init torch._dynamo failure: retrying in-process never recovers, so the remedy is a restart; unrelated errors return None."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if _DYNAMO_PARTIAL_RE.search(str(exc)):
            return (
                "PyTorch's compiler module (torch._dynamo) ended up half-initialised in this "
                "process, so the image model could not finish loading. Restart Unsloth and load "
                "it again; this state does not clear on its own."
            )
        # Stops at `raise ... from None`: the hidden inner error is unrelated; restart advice would misfire.
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return None


def hub_access_message(exc: BaseException, *, had_token: bool) -> Optional[str]:
    """Rewrite a gated-repo failure into the step that actually unblocks the user, else None so an
    unrelated load error keeps its own text. Only the toast is affected; the raw exception,
    request id and resolve URL still reach the log."""
    gated = _gated_in_chain(exc)
    if gated is None:
        return None
    found = _HUB_REPO_RE.search(str(gated))
    # An API URL ends at the repo, so a trailing full stop is stripped (a repo name never ends in a dot).
    repo = found.group(1).rstrip(".") if found else None
    # Any other /api/<endpoint> URL (whoami-v2, ...) would parse as the repo "api/<endpoint>"
    if repo and repo.split("/", 1)[0] == "api":
        repo = None
    where = f"https://huggingface.co/{repo}" if repo else "its Hugging Face page"
    subject = repo or "This model"
    if had_token:
        # A token was sent and still bounced, so the account itself lacks access.
        return f"{subject} is gated and this Hugging Face account is not on its access list. Request access at {where}, then load again."
    return f"{subject} is gated. Request access at {where}, then add a Hugging Face token in Settings and load again."


def resolve_model_kind(gguf_filename: Optional[str], model_kind: Optional[str] = None) -> str:
    """Explicit validated model_kind wins; else a .gguf name gives gguf, another file single_file, no name pipeline. Pure, so all load paths agree."""
    if model_kind:
        kind = model_kind.strip().lower()
        if kind not in _MODEL_KINDS:
            raise ValueError(
                f"Unknown model_kind '{model_kind}'. Expected one of {sorted(_MODEL_KINDS)}."
            )
        return kind
    name = (gguf_filename or "").strip()
    if not name:
        return "pipeline"
    if name.lower().endswith(".gguf"):
        return "gguf"
    return "single_file"


def _active_lora_pairs(pipe: Any) -> list:
    """Zero-weight adapters are dropped; shape is tolerated since this runs inside a finished generation."""
    pairs = []
    for entry in getattr(pipe, "_unsloth_loras", ()) or ():
        try:
            if len(entry) == 3:
                name, _path, weight = entry
            elif len(entry) == 2:
                name, weight = entry
            else:
                continue
            weight = float(weight)
        except Exception:  # noqa: BLE001 - an unrecognised marker records no adapter
            continue
        if weight:
            pairs.append((name, weight))
    return pairs


def _baked_lora_names(pipe: Any) -> list:
    """Names of adapters baked in at load (torchao); scale 0 is not the same pipeline, so the recipe must record them."""
    if not getattr(pipe, "_unsloth_loras_baked", False):
        return []
    names = []
    for entry in getattr(pipe, "_unsloth_loras", ()) or ():
        try:
            name = entry[0]
        except Exception:  # noqa: BLE001 - an unrecognised marker records no adapter
            continue
        if name:
            names.append(str(name))
    return names


# ``(module, adapter, base)`` for every baked LoRA scale held as a device tensor; ``base`` is lora_alpha / r.
_PINNED_LORA_SCALES_ATTR = "_unsloth_pinned_lora_scales"
# ``(module, adapter)`` for the baked LoRA layers OUTSIDE the denoisers (a text encoder carrying the same file's
# adapter): their scales stay floats and are set the way set_adapters would set them.
_PINNED_LORA_SIDE_ATTR = "_unsloth_pinned_lora_side"


def _side_lora_layers(pipe: Any) -> list:
    """``(module, adapter)`` for every PEFT LoRA layer on a pipeline component that is not a denoiser."""
    import torch

    skip = {id(m) for _a, m in denoiser_modules(pipe)}
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        components = dict(vars(pipe))
    out = []
    seen: set[int] = set()
    for value in components.values():
        if not isinstance(value, torch.nn.Module) or id(value) in skip or id(value) in seen:
            continue
        seen.add(id(value))
        for module in value.modules():
            scaling = module.__dict__.get("scaling")
            lora_b = getattr(module, "lora_B", None)
            if not isinstance(scaling, dict) or lora_b is None or not hasattr(module, "set_scale"):
                continue
            # Every scaled adapter, an Embedding's (factors in lora_embedding_A/B, not lora_B) included.
            out.extend((module, adapter) for adapter in scaling)
    return out


def _pin_baked_lora_scales(pipe: Any, target_device: Any) -> int:
    """Hold each baked adapter's PEFT scale as a 0-d tensor on its LoRA weights' device, so a later weight change or
    subset switch is an in-place write instead of a new Python float.

    PEFT keeps ``scaling[adapter]`` as a float and multiplies by it inside the LoRA forward. torch.compile guards on
    that float (``self.scaling['b'] == 1.0``), and a CUDA graph records it as a kernel argument, so every new weight
    vector recompiled and recaptured. The adapters and their shapes never change on a baked load, only the scale,
    so a tensor read at run time serves every selection with the first prompt's graphs. Call before compile and
    graph capture. Skipped (floats kept, the reset path stays) on an offloaded denoiser, where the LoRA weights move
    off the scale's device, and when any LoRA weight is not yet on the target device type. Returns the count."""
    if not getattr(pipe, "_unsloth_loras_baked", False) or getattr(
        pipe, _PINNED_LORA_SCALES_ATTR, None
    ):
        return 0
    if _denoiser_hooked(pipe):
        return 0
    import math

    import torch

    # The type off the string form ("cuda:0" -> "cuda"), as torch.device(...).type gives, without a torch.device.
    want = str(target_device).split(":", 1)[0]
    found = []
    for _attr, root in denoiser_modules(pipe):
        modules = getattr(root, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            scaling = module.__dict__.get("scaling")
            lora_b = getattr(module, "lora_B", None)
            if not isinstance(scaling, dict) or lora_b is None:
                continue
            for adapter, value in scaling.items():
                if isinstance(value, torch.Tensor):
                    continue
                if adapter not in lora_b:
                    # A scale with no lora_B entry (a PEFT Embedding keeps lora_embedding_A/B): pinning the rest
                    # would skip set_adapters and leave this one at its bake-time weight, so keep the float path.
                    return 0
                weight = lora_b[adapter].weight
                if weight.device.type != want:
                    return 0
                r = module.r[adapter]
                alpha = module.lora_alpha[adapter]
                rslora = bool(getattr(module, "use_rslora", {}).get(adapter, False))
                base = alpha / (math.sqrt(r) if rslora else r)
                found.append((module, adapter, base, float(value), weight.device))
    if not found:
        return 0
    pinned = []
    # A normal tensor even when the load runs under inference_mode: an inference tensor refuses the later fill_.
    with torch.inference_mode(False):
        for module, adapter, base, value, device in found:
            module.scaling[adapter] = torch.tensor(value, dtype = torch.float32, device = device)
            pinned.append((module, adapter, base))
    setattr(pipe, _PINNED_LORA_SIDE_ATTR, tuple(_side_lora_layers(pipe)))
    setattr(pipe, _PINNED_LORA_SCALES_ATTR, tuple(pinned))
    return len(pinned)


def _set_pinned_lora_scales(pipe: Any, weights: dict) -> bool:
    """Write ``weights`` (adapter name -> weight, omitted = 0) into the pinned scale tensors in place. False, with the
    pin dropped, when there is none or it no longer holds (something rewrote a scale as a float or moved the
    weights): the caller then takes the set_adapters + graph reset path."""
    pinned = getattr(pipe, _PINNED_LORA_SCALES_ATTR, None)
    if not pinned:
        return False
    import torch

    for module, adapter, _base in pinned:
        value = module.scaling.get(adapter)
        if (
            not isinstance(value, torch.Tensor)
            or value.device != module.lora_B[adapter].weight.device
        ):
            setattr(pipe, _PINNED_LORA_SCALES_ATTR, None)
            return False
    with torch.no_grad():
        for module, adapter, base in pinned:
            module.scaling[adapter].fill_(float(weights.get(adapter, 0.0)) * base)
    # The same adapter on a text encoder (SD / SDXL / SD3 / FLUX files can carry one): set_adapters is skipped, so
    # set its scale here, or prompt encoding keeps the bake-time weight while the marker names the new one.
    for module, adapter in getattr(pipe, _PINNED_LORA_SIDE_ATTR, None) or ():
        module.set_scale(adapter, float(weights.get(adapter, 0.0)))
    return True


def resolve_local_single_file(model_path: str) -> Optional[str]:
    """The sole single-file checkpoint basename in a local ``model_path`` directory that is NOT a
    diffusers pipeline (no ``model_index.json`` or FLUX.3 ``modular_model_index.json``) and holds exactly one ``.safetensors`` file, else
    None.

    The On-Device scanner advertises a bare single-file safetensors directory as a text-to-image
    model, but the local picker starts it as a ``pipeline`` with no filename, so a pipeline load
    400s on the missing ``model_index.json``. The images load route uses this to reinterpret such a
    pick as a ``single_file`` load of the sole checkpoint. A real pipeline dir, or an ambiguous one
    (0 or more than 1 ``.safetensors``), returns None and loads unchanged. A PEFT LoRA adapter
    folder is also skipped. Never raises.
    """
    try:
        root = Path(model_path).expanduser()
        if not root.is_dir() or any(
            (root / index).is_file() for index in ("model_index.json", FLUX3_INDEX)
        ):
            return None
        # A PEFT adapter folder is not a base checkpoint; skip it so validation 400s before eviction
        if (root / "adapter_config.json").is_file():
            return None

        checkpoints = [
            p.name
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".safetensors"
            and p.stem.lower() != "adapter_model"
            and not is_appledouble_metadata(p)
        ]
    except OSError:
        return None
    return checkpoints[0] if len(checkpoints) == 1 else None


def split_local_checkpoint_path(model_path: str) -> Optional[tuple[str, str]]:
    """``(dir, name)`` when ``model_path`` is one local ``.safetensors`` file, else None."""
    try:
        path = Path(model_path).expanduser()
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            return None
        if is_appledouble_metadata(path):
            return None
    except (OSError, ValueError):
        return None
    return str(path.parent), path.name


def decode_b64_image(
    data: str,
    *,
    mode: str = "RGB",
    max_side: int = 4096,
    max_pixels: Optional[int] = None,
) -> Any:
    """Decode a base64 or data: URL image; masks decode as single-channel L, source images as RGB."""
    import base64
    import binascii
    import io

    from PIL import Image

    raw = data.strip()
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    try:
        blob = base64.b64decode(raw, validate = False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    try:
        img = Image.open(io.BytesIO(blob))
        # Reject from the header before img.load() so a huge-dimension file cannot spike memory
        w, h = img.size
        if w > max_side or h > max_side:
            raise ValueError(f"Image is too large ({w}x{h}); maximum is {max_side}px per side.")
        if max_pixels is not None and w * h > max_pixels:
            raise ValueError(
                f"Image is too large ({w}x{h}); maximum is {max_pixels:,} source pixels."
            )
        img.load()
        # Below the size guard because the transpose needs the pixels the guard exists to not read.
        img = exif_upright(img)
    except ValueError:
        raise  # the size guard's own message; don't wrap it as a decode error
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 to the client
        raise ValueError(f"Could not decode image: {exc}") from exc
    return flattened_rgb(img) if mode == "RGB" else img.convert(mode)


def _snap_to_multiple(img: Any, multiple: int = 16) -> Any:
    """Snap both sides to a multiple, since image-conditioned pipelines reject sizes not divisible by 16."""
    from PIL import Image

    w, h = img.size
    nw = max(multiple, int(round(w / multiple)) * multiple)
    nh = max(multiple, int(round(h / multiple)) * multiple)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def _clamp_max_side(img: Any, max_side: int) -> Any:
    """Clamps the longest side to 2048, the same ceiling txt2img enforces, so an oversized upload cannot OOM the transformer or VAE."""
    from PIL import Image

    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def _fit_within(img: Any, max_w: int, max_h: int) -> Any:
    """Fit inside the box without ever enlarging; growing a source is the Upscale workflow's job."""
    from PIL import Image

    w, h = img.size
    bw = max(1, int(max_w))
    bh = max(1, int(max_h))
    if w <= bw and h <= bh:
        return img
    scale = min(bw / float(w), bh / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def _prequant_plan_bytes(te_files: Any) -> int:
    """Return the hosted pre-cast encoder bytes in a download plan."""
    return sum(
        int(size or 0) for entry in (te_files or {}).values() for _name, size in (entry[1] or ())
    )


def _predownload_encoder_bf16_bytes(
    fam: Any,
    te_files: Any,
    *,
    pipeline_declared: bool = True,
) -> int:
    """Return separately hosted dense bf16 encoder bytes absent from the pipeline repo."""
    if (
        getattr(fam, "name", None) == HIDREAM_FAMILY_NAME
        and "text_encoder_4" not in (te_files or {})
        and pipeline_declared
    ):
        return HIDREAM_LLAMA_BF16_BYTES
    return 0


def _image_variant_hint(
    family_name: Optional[str],
    single_file_name: Optional[str],
    repo_id: Optional[str],
    base: Optional[str],
) -> str:
    """Include both repo_id and base: the variant marker usually lives in the base, not the GGUF repo."""
    parts: list[str] = []
    for part in (family_name, single_file_name, repo_id, base):
        text = (part or "").strip()
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


def _is_source_sized(workflow: str, family: Any = None) -> bool:
    """Whether ``workflow`` takes its output size from the source image rather than width/height.
    "edit" is source-sized only on an edit-only family (Kontext, Qwen-Image-Edit), or when
    ``family`` is None; the unified edit (Qwen-Image-2.1) renders at the requested size."""
    if workflow == "edit":
        return family is None or bool(getattr(family, "edit", False))
    return workflow in ("img2img", "inpaint", "upscale")


def _release_prefix_kv(pipe: Any) -> None:
    """Drop the denoiser's prefix KV caches (and a CFG-parallel replica's copies) before VAE decode."""
    try:
        released = release_prefix_kv(getattr(pipe, "transformer", None))
        if released:
            release_cfg_prefix_kv(pipe)
            logger.debug("diffusion.prefix_kv: %d MiB released before decode", released >> 20)
    except Exception as exc:  # noqa: BLE001 - a memory saving only
        logger.debug("diffusion.prefix_kv: release failed: %s", exc)


def _compile_shape_dims(
    workflow: str,
    init_pil: Any,
    width: int,
    height: int,
    family: Any = None,
) -> tuple[int, int]:
    """Actual (w, h) a generation runs at, for compile-cache registration; mirrors generate(), since source-sized workflows use the image size."""
    if init_pil is None or not _is_source_sized(workflow, family):
        return int(width), int(height)
    iw, ih = init_pil.size
    if getattr(family, "layer_count", 0):
        return _layered_canvas(family, (iw, ih))
    return int(iw), int(ih)


def _layered_canvas(family: Any, size: tuple[int, int]) -> tuple[int, int]:
    """The (width, height) QwenImageLayeredPipeline decomposes at: the ``layer_resolution`` square's area at the input's
    aspect ratio, each side rounded to 32 (``calculate_dimensions`` in the pipeline). The input image's own size only
    sets the ratio."""
    import math

    iw, ih = size
    area = float(getattr(family, "layer_resolution", 640)) ** 2
    ratio = float(iw) / float(max(1, ih))
    width = math.sqrt(area * ratio)
    height = width / ratio
    return max(32, int(round(width / 32)) * 32), max(32, int(round(height / 32)) * 32)


# Official non-unsloth bases allowed to load as full (non-GGUF) artifacts: safetensors-only, no remote code, exact
# lowercased match. Derived from the registry rows (base_repo, variant / training bases, FamilyProfile.trusted_repos).
from .diffusion_families import _FAMILIES as _IMAGE_REGISTRY  # noqa: E402
from .family_profile import trusted_repo_ids as _trusted_repo_ids  # noqa: E402

_TRUSTED_NON_GGUF_REPOS = _trusted_repo_ids(_IMAGE_REGISTRY)


def _is_trusted_diffusion_repo(repo_id: str) -> bool:
    """Non-GGUF loads deserialise third-party weights, so allow only unsloth/*, the allowlist, or user-named local paths; OSError means not local."""
    try:
        if Path(repo_id).expanduser().exists():
            return True
    except OSError:
        pass
    rid = repo_id.strip().lower()
    return rid.startswith("unsloth/") or rid in _TRUSTED_NON_GGUF_REPOS


def _assert_local_base_is_pipeline(
    base_repo: str,
    *,
    allow_modular: bool = False,
    modular_only: bool = False,
    excluded_components: Sequence[str] = (),
    config_only_model_components: bool = False,
    fixed_component_folders: Optional[Mapping[str, tuple[str, str, str]]] = None,
) -> None:
    """A companion ``base_repo`` fed to ``from_pretrained(base)`` (or ``config=base``) must be a
    diffusers PIPELINE directory (has ``model_index.json``). ``_is_trusted_diffusion_repo`` accepts
    ANY existing local path, so without this a local base that is not a pipeline dir would pass the
    preflight, let the route evict the resident GPU model, then fail deep in the background load:
    the eviction this validation exists to prevent. A non-existent local base is already rejected by
    the trust check; a bare remote id is left for the loader to resolve. Shared by the image, video,
    and training preflights. Never evicts; raises ValueError on a bad local base.

    ``allow_modular`` accepts ``modular_model_index.json`` as well, for a caller whose loader is
    ``ModularPipeline.from_pretrained``: that IS the valid on-disk layout for a Modular Diffusers
    pipeline (MiniMax-H3 ships no ``model_index.json`` at all). Off by default: a conventional
    ``DiffusionPipeline`` load still needs the conventional index. ``modular_only`` accepts nothing else, for a
    loader that reads only the modular index (FLUX.3).

    ``excluded_components`` are supplied by the caller (the checkpoint's denoiser), not the base;
    ``config_only_model_components`` means the base only supplies component configs."""
    base = (base_repo or "").strip()
    if not base:
        return
    try:
        root = Path(base).expanduser()
        exists = root.exists()
    except OSError:
        return  # invalid path characters -> a remote id, not a local path
    if not exists:
        return
    indexes = [] if modular_only else ["model_index.json"]
    if allow_modular or modular_only:
        indexes.append("modular_model_index.json")
    if not root.is_dir() or not any(
        local_pipeline_components_are_complete(
            root,
            name,
            excluded_components = excluded_components,
            config_only_model_components = config_only_model_components,
            fixed_component_folders = fixed_component_folders,
        )
        for name in indexes
    ):
        raise ValueError(
            f"Local base_repo is not a diffusers pipeline directory "
            f"(no valid {' or '.join(indexes)}): {base}"
        )


def _repo_access_message(repo: str, *, gated: bool) -> str:
    """The repo id AND its licence page: the worker's 401/403 names neither, and the base comes from
    a card tag, so the user never saw which repo it is."""
    url = f"https://huggingface.co/{repo}"
    if gated:
        return (
            f"'{repo}' is gated on Hugging Face and this model cannot be downloaded without it. "
            f"Accept its licence at {url}, then add a Hugging Face token that has access in "
            "Unsloth settings and try again."
        )
    return (
        f"'{repo}' could not be read from Hugging Face (private, renamed or removed) and this "
        f"model cannot be downloaded without it. Check {url}, then add a Hugging Face token that "
        "has access in Unsloth settings and try again."
    )


def _assert_base_repo_accessible(
    base_repo: str,
    hf_token: Optional[str],
    probe_file: str = "model_index.json",
    *,
    local_files_only: bool = False,
) -> Optional[str]:
    """Fails up front on an unreadable gated base; probes gated repos only, other errors fail open; under local_files_only only the cache is read."""
    repo = (base_repo or "").strip()
    # Only a remote 'org/name' can be gated; a local base is already on disk.
    if not repo or repo.count("/") != 1:
        return None
    # build_hf_headers sends "" as a literal "Bearer " that 401s.
    hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
    try:
        if Path(repo).expanduser().exists():
            return None
    # OSError (invalid path chars) means a remote id and RuntimeError (no home dir for expanduser) reaches here; an escape would 500 this probe.
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except Exception:  # noqa: BLE001 - an unexpected hub layout leaves today's behaviour
        return None

    # Set when only the import-time root holds the excusing copy, which the pinned from_pretrained would miss.
    other_root_snapshot: Optional[str] = None

    # hf_hub_download serves a cached gated base on 401; excuses access errors only, never a 404.
    def _already_downloaded() -> bool:
        """True when probe_file is cached under either root; a proxy for the diffusers plan. Never raises."""
        nonlocal other_root_snapshot
        try:
            from huggingface_hub import try_to_load_from_cache
            for root in (hub_cache_dir(), None):
                # Only a str is a cached path; a miss is None and an absent file is a sentinel.
                hit = try_to_load_from_cache(repo, probe_file, cache_dir = root)
                if isinstance(hit, str):
                    # Root None means the import-time root holds it; its parent is the snapshot for top-level probes.
                    if root is None and "/" not in probe_file:
                        other_root_snapshot = str(Path(hit).parent)
                    return True
        except Exception:  # noqa: BLE001 - a cache we cannot read is not an access verdict
            pass
        return False

    if local_files_only:
        # Cache read only, no model_info or HEAD; sets other_root_snapshot, and a miss is not a refusal since the loader raises for missing files.
        _already_downloaded()
        return other_root_snapshot

    def _is_auth_error(exc: Any) -> bool:
        """Catches plain 401/403 HfHubHTTPError from expired tokens or missing permissions; others
        fail open."""
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in (401, 403)

    try:
        gated = getattr(hub_reachability.model_info(HfApi(), repo, token = hf_token), "gated", None)
    except GatedRepoError:  # a gated repo can also withhold its metadata
        if _already_downloaded():
            return other_root_snapshot
        raise ValueError(_repo_access_message(repo, gated = True)) from None
    except RepositoryNotFoundError as exc:
        # Raised for 401 too (hf_raise_for_status folds it in with 404): re-read the status; only 401/403 reach the cache escape, a 404 raises.
        if _is_auth_error(exc) and _already_downloaded():
            return other_root_snapshot
        raise ValueError(
            modelscope_missing(exc) or _repo_access_message(repo, gated = False)
        ) from None
    except HfHubHTTPError as exc:
        if not _is_auth_error(exc):
            return None  # a 5xx or rate limit is not an access verdict
        if _already_downloaded():
            return other_root_snapshot
        raise ValueError(_repo_access_message(repo, gated = False)) from None
    except Exception:  # noqa: BLE001 - offline / transient: the download surfaces any real error
        return None
    # Nothing to carry: model_info answered, so each base file resolves via whichever root holds it.
    if not gated or _already_downloaded():
        return None
    try:
        # Metadata HEAD, not hf_hub_download, which would serve a cached pointer for a stale token.
        get_hf_file_metadata(hf_hub_url(repo, probe_file), token = hf_token)
    except GatedRepoError:
        raise ValueError(_repo_access_message(repo, gated = True)) from None
    except HfHubHTTPError as exc:
        if not _is_auth_error(exc):
            return None
        raise ValueError(_repo_access_message(repo, gated = True)) from None
    except Exception:  # noqa: BLE001 - no manifest / offline / transient is not an access verdict
        return None
    return None


def _te_quant_reason(
    outcome: Any, engaged: Optional[str], *, auto: bool, family: Optional[str]
) -> str:
    """AUTO picks are annotated as unrequested, which signals the encoder can be pinned back to bf16."""
    reason = getattr(outcome, "reason", None) or (
        "dense bf16 text encoder(s) loaded"
        if engaged is None
        else "dense text encoder(s) quantised in place"
    )
    if auto and engaged is not None:
        reason = f"{reason}; selected automatically for {family} (no text_encoder_quant requested)"
    elif auto:
        reason = f"{reason}; the automatic '{family}' encoder scheme did not engage"
    return reason


def _with_supplied_components(load: Callable[..., dict]) -> Callable[..., dict]:
    """Supplied text-encoder / VAE files replace the base repo's for this load, via a context variable the
    assembly, memory plan and status read."""

    @functools.wraps(load)
    def wrapper(self: Any, repo_id: str, **kwargs: Any) -> dict:
        from .diffusion_comfy_components import (
            reset_active_component_overrides,
            set_active_component_overrides,
        )

        text_encoder_files = kwargs.pop("text_encoder_files", None)
        vae_file = kwargs.pop("vae_file", None)
        overrides = kwargs.pop("_component_overrides", None)
        if overrides is None and (text_encoder_files or vae_file):
            fam = self.validate_load_request(
                repo_id,
                gguf_filename = kwargs.get("gguf_filename"),
                family_override = kwargs.get("family_override"),
                model_kind = kwargs.get("model_kind"),
                text_encoder_files = text_encoder_files,
                vae_file = vae_file,
            )
            hf_token = (kwargs.get("hf_token") or "").strip() or None
            base = _resolve_base_repo(
                repo_id, kwargs.get("base_repo"), fam, hf_token, kwargs.get("gguf_filename")
            )
            overrides = self._plan_component_overrides(
                fam,
                repo_id,
                base,
                text_encoder_files,
                vae_file,
                hf_token,
                local_files_only = bool(kwargs.get("local_files_only")),
                base_local_dir = kwargs.get("_base_local_dir"),
                text_encoder_quant = kwargs.get("text_encoder_quant"),
            )
        if overrides is not None and overrides.text_encoder_components:
            kwargs["text_encoder_quant"] = _te_quant_for_supplied_encoders(
                kwargs.get("text_encoder_quant")
            )
        from .diffusion_precision import te_quant_is_auto
        from .diffusion_te_prequant import auto_te_request

        token = set_active_component_overrides(overrides)
        try:
            # An unset / "auto" encoder scheme never fails the load: a hosted file that cannot load falls back to dense.
            with auto_te_request(te_quant_is_auto(kwargs.get("text_encoder_quant"))):
                return load(self, repo_id, **kwargs)
        finally:
            reset_active_component_overrides(token)

    return wrapper


def _generation_chunks(
    family: Any,
    jobs: list,
    batch_size: int,
    *,
    layer_count: Any = None,
) -> list:
    """Per-forward chunks of ``(prompt, seed)`` jobs. Layered: one input per forward. FLUX.3 runs one image per
    modular call, so each image is its own chunk: every one then gets its own encode, denoise and decode phase, and
    the text-encoder release that comes with entering denoise (one chunk of several images would release it once,
    and the next image's encode would bring it back for the rest of the chunk)."""
    if layer_count or is_flux3_family(family):
        return [[job] for job in jobs]
    return chunk_jobs(jobs, batch_size)


def _base_manifest_name(fam: Any) -> str:
    """The pipeline index a base repo carries: FLUX.3 ships only ``modular_model_index.json``."""
    return FLUX3_INDEX if is_flux3_family(fam) else "model_index.json"


def _te_quant_for_supplied_encoders(requested: Optional[str]) -> Optional[str]:
    """Unset / auto never re-casts a supplied encoder (it would fetch the hosted encoder the file replaces)."""
    if requested is None or str(requested).strip().lower() in ("", "auto"):
        return "none"
    return requested


def _hub_file_size(repo_id: str, filename: str, hf_token: Optional[str]) -> int:
    # get_paths_info takes no timeout.
    if hub_reachability.hub_unreachable():
        return 0
    try:
        from huggingface_hub import HfApi
        for info in HfApi().get_paths_info(repo_id, [filename], token = hf_token):
            hub_reachability.note_hub_success()
            return int(getattr(info, "size", 0) or 0)
        hub_reachability.note_hub_success()
    except Exception as exc:  # noqa: BLE001 - a size is a progress-bar hint, never a refusal
        hub_reachability.note_hub_failure(exc)
    return 0


def _active_supplied_text_encoders() -> tuple[str, ...]:
    from .diffusion_comfy_components import active_component_overrides
    overrides = active_component_overrides()
    return tuple(c for c in overrides.paths if c.startswith("text_encoder")) if overrides else ()


def _active_component_summary() -> Optional[dict]:
    from .diffusion_comfy_components import active_component_overrides
    overrides = active_component_overrides()
    return overrides.summary() if overrides is not None and overrides.files else None


@dataclass(frozen = True)
class _LoadState:
    """Everything about the currently-loaded pipeline, swapped as one unit."""

    pipe: Any
    family: Any
    repo_id: str
    base_repo: str
    device: str
    dtype: str
    cpu_offload: bool
    display_repo_id: Optional[str] = None
    # Defaulted so older positional constructions keep working.
    offload_policy: str = OFFLOAD_NONE
    vae_tiling: bool = False
    memory_mode: str = "auto"
    # Resolved load kind ("gguf"|"single_file"|"pipeline"); lets the UI gate GGUF-only controls.
    kind: str = "gguf"
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    # Torch backend flags (TF32 / cudnn.benchmark) captured before the speed layer mutated them.
    backend_flags_before: Optional[dict] = None
    # Text-encoder quant engaged: "fp8" | "nvfp4" | None.
    text_encoder_quant: Optional[str] = None
    # Transformer quant on the dense fast path ("int8"|"fp8"|"nvfp4"|"mxfp8"), or None when GGUF loaded.
    transformer_quant: Optional[str] = None
    # Attention backend via the diffusers dispatcher, or None for default SDPA.
    attention_backend: Optional[str] = None
    # Caller original attention request, so deferred engagement re-runs the same selection.
    attention_request: Optional[str] = None
    # Step cache engaged ("fbcache" | "static") or None; only fbcache breaks the graph.
    transformer_cache: Optional[str] = None
    # Auto only: generate() toggles it across FBCACHE_MIN_STEPS; explicit never toggles
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Shared eager patches installed for this load; uninstalled on unload.
    eager_patched: bool = False
    # Deferred speed auto: the load stays eager, generate() engages `default` at the 3rd generation
    speed_deferred: bool = False
    # That deferred engagement compiled the denoiser: LoRAs are off until a reload, which defers again.
    speed_auto_compiled: bool = False
    # Auto static skip plan for that deferred `default` tier, installed when it engages.
    deferred_static_plan: Optional[dict] = None
    # Successful generations on this load; drives the deferred engagement above.
    generation_count: int = 0
    # Pre-warmed torch.compile cache context when a compiled tier ran, else None.
    compile_cache_ctx: Any = None
    bg_compile: Any = None
    # GraphedForward handles installed on the denoiser modules.
    cuda_graphs: tuple = ()
    # Token kept so LoRA adapters selected at generate time can be fetched.
    hf_token: Optional[str] = None
    # Per-control provenance {control: {value, source, reason}}, for status badges.
    resolved: Optional[dict] = None
    # The single-file checkpoint basename this load committed (None for a pipeline). Part of the build identity.
    gguf_filename: Optional[str] = None
    component_files: Optional[dict] = None
    # The torch ordinal this pipeline's weights were placed on, or None for an automatic pick. Committed WITH the
    # pipeline, so a load in flight never moves the resident model's card.
    gpu_ordinal: Optional[int] = None
    # Actual card incl. automatic picks; only for re-pinning a pooled generate worker.
    placed_ordinal: Optional[int] = None
    # Stored so generate()'s re-check uses the same multipliers the load did.
    variant_hint: str = ""
    # Promoted tiers leave no room for a later ControlNet.
    calibrated_placement: bool = False
    # Unified memory only: frees the text encoders during each denoise and reloads them before they run again.
    te_releaser: Any = None


@dataclass
class _LoadingState:
    """An in-flight background load, polled for download progress."""

    repo_id: str
    base_repo: str
    expected_bytes: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None
    # Where the companion BYTES land: ``base_repo``, or its mirror when one was swapped in. Never surfaced
    # (``base_repo`` stays the id status() reports), but the cache scan and the delete guard must look here.
    fetch_repo: Optional[str] = None
    # Third repos this load downloads from; the cache scan and the delete guard both need them.
    asset_repos: tuple[str, ...] = ()
    # (repo, filename, declared size, repo bytes when claimed) for one named file; the scan counts only the bytes this load adds.
    asset_files: tuple[tuple[str, str, int, int], ...] = ()
    account_id: Optional[str] = None
    # The single file this load reads from ``repo_id``, so progress through an artifact mirror counts that one
    # checkpoint rather than every sibling quant cached beside it.
    gguf_filename: Optional[str] = None
    # ``(repo, filename, size)`` for every file this load downloads, from ``_planned_progress_files``: the ONE answer
    # to "which files of which repos does this load count", which expected_bytes is the sum of and load_progress
    # scans. Empty when the size estimate failed; load_progress then keeps its repo-level scan.
    planned_files: tuple[tuple[str, str, int], ...] = ()
    # ``{incomplete blob path: size}`` in the planned repos when this load claimed them: a cancelled sibling download
    # leaves ``*.incomplete`` blobs behind, and only growth past this snapshot is this load's own bytes.
    incomplete_baseline: dict[str, int] = field(default_factory = dict)
    # ``{(repo_lower, filename): blob key}`` for the planned files whose listing named one. huggingface_hub writes a
    # download to ``blobs/<key>.incomplete``, so an in-flight blob with a planned key IS that file, resumed bytes
    # included, and every other ``.incomplete`` (a cancelled sibling's) is ignored outright.
    planned_blob_keys: dict[tuple[str, str], str] = field(default_factory = dict)
    # (expected_bytes, downloaded) once every expected byte was on disk.
    finalized_scan: Optional[tuple[int, int]] = None


def _account_owned_load(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        request = object()
        inherited = kwargs.get("_load_token")
        while True:
            if inherited is None:
                # Wait the in-flight ejects out and take the epoch they leave; a worker inherits one.
                self._wait_for_pending_unloads()
            with self._load_cancel_lock:
                token = self._load_token if inherited is None else inherited
                # Read under the lock that registers, or an eject in the gap admits this load into teardown.
                if inherited is None and self._unload_waiters:
                    continue
                self._raise_if_load_cancelled(token)
                self._load_accounts[request] = (token, current_account_id())
                break
        try:
            return method(self, *args, **{**kwargs, "_load_token": token})
        finally:
            with self._load_cancel_lock:
                self._load_accounts.pop(request, None)

    return wrapped


def _release_render_on_unload(method):
    """Free VRAM when an unload lands mid-render: the cancelled render's traceback pins ``pipe`` past
    the teardown's cache clear, so drop its frames and clear again once the last reference is gone."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        token, teardowns = self._load_token, self._teardown_epoch
        try:
            return method(self, *args, **kwargs)
        except BaseException as exc:
            _clear_exception_frames(exc)
            raise
        finally:
            # Epoch too: a replacing load bumps the token before this render starts, then tears down.
            if self._load_token != token or self._teardown_epoch != teardowns:
                try:
                    clear_gpu_cache()
                except Exception as exc:
                    logger.debug("diffusion.generate: cache release after unload failed: %s", exc)

    return wrapped


@dataclass
class _GenState:
    """An in-flight generation, updated per denoising step for the progress bar."""

    total_steps: int
    step: int = 0
    # Set when the first step finishes, so the slower warmup step does not skew the ETA rate.
    first_step_at: float = 0.0
    # Computed once per step (in the callback) so it's stable between polls.
    eta_seconds: Optional[float] = None
    # encode -> denoise (pipe() entered its loop) -> decode (pipe() entered its decoder).
    phase: str = "encode"
    preview: Optional[str] = None
    preview_seq: int = 0


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    """Seconds remaining, from the average step time measured after the first step. None until at
    least one step has elapsed since the first."""
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


def _resolve_diffusion_compute_dtype(fam: Optional[DiffusionFamily], dtype: Any) -> Any:
    """Promote float16 -> float32 for fp16-incompatible families without a usable fp16 guard; else unchanged."""
    from .diffusion_fp16_guard import fp16_promotes_to_fp32

    if fam is None or not getattr(fam, "fp16_incompatible", False):
        return dtype
    import torch

    if dtype != torch.float16 or not fp16_promotes_to_fp32(fam):
        return dtype
    return torch.float32


def _float_load_itemsize(dtype: Any) -> Optional[int]:
    try:
        import torch
        if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
            return int(dtype.itemsize)
    except Exception:  # noqa: BLE001 - no torch: keep on-disk sizing
        pass
    return None


def _class_pins_fp32(class_name: str, load_dtype: Any) -> tuple[str, ...]:
    """Diffusers pins at both halves; Transformers pins ``_keep_in_fp32_modules`` only at fp16."""
    for lib in ("diffusers", "transformers"):
        module = sys.modules.get(lib)
        try:
            cls = getattr(module, class_name, None) if module is not None else None
        except Exception:  # noqa: BLE001 - an unimportable class tells us nothing
            cls = None
        if cls is None:
            continue
        pinned = list(getattr(cls, "_keep_in_fp32_modules_strict", None) or ())
        if lib == "diffusers" or str(load_dtype) == "torch.float16":
            pinned += list(getattr(cls, "_keep_in_fp32_modules", None) or ())
        # Diffusers tests each pin against single name segments, so a dotted pin never matches there.
        return tuple(str(name) for name in pinned if lib != "diffusers" or "." not in str(name))
    return ()


def _component_pins_fp32(folder: Path, load_dtype: Any) -> tuple[str, ...]:
    try:
        config = json.loads((folder / "config.json").read_text(encoding = "utf-8"))
        names = [config.get("_class_name"), *(config.get("architectures") or ())]
    except Exception:  # noqa: BLE001 - no readable config: nothing is pinned by name
        return ()
    return tuple(
        pin
        for name in names
        if isinstance(name, str) and name
        for pin in _class_pins_fp32(name, load_dtype)
    )


def _install_gguf_prefix_strip(transformer_cls: Any, logger: Any) -> None:
    """Strips the model.diffusion_model. prefix sd.cpp GGUFs add, since diffusers handles it inconsistently; idempotent and best-effort."""
    try:
        from diffusers.loaders import single_file_model as sfm

        entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get(
            getattr(transformer_cls, "__name__", str(transformer_cls))
        )
        if not isinstance(entry, dict):
            return
        original = entry.get("checkpoint_mapping_fn")
        if not callable(original) or getattr(original, "_unsloth_prefix_strip", False):
            return
        prefix = "model.diffusion_model."

        def _stripped_mapping_fn(checkpoint = None, **kwargs):
            checkpoint = {
                (key[len(prefix) :] if key.startswith(prefix) else key): value
                for key, value in (checkpoint or {}).items()
            }
            return original(checkpoint = checkpoint, **kwargs)

        _stripped_mapping_fn._unsloth_prefix_strip = True
        entry["checkpoint_mapping_fn"] = _stripped_mapping_fn
    except Exception as exc:  # noqa: BLE001 - loader-compat shim only, never fail the load
        logger.warning("diffusion.gguf: prefix-strip shim not installed: %s", exc)


def _qwen_image_21_checkpoint_to_diffusers(checkpoint = None, **kwargs):
    """Splits fused img_mlp.gate_up into gate then proj (swapped order is off by up to 1.08); dequantises 1-D norms, which break if left packed."""
    prefix = "model.diffusion_model."
    fused = "img_mlp.gate_up.weight"
    converted = {}
    for key, value in (checkpoint or {}).items():
        if key.startswith(prefix):
            key = key[len(prefix) :]
        quant_shape = getattr(value, "quant_shape", None)
        if quant_shape is not None and len(quant_shape) == 1:
            from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
            value = dequantize_gguf_tensor(value)
        if key.endswith("." + fused):
            rows = int(value.shape[0])
            if rows % 2:
                raise ValueError(
                    f"{key}: a fused gate_up with an odd row count ({rows}) cannot be split into "
                    "gate_layer and proj; this is not a Qwen-Image-2.1 checkpoint"
                )
            stem = key[: -len("gate_up.weight")]
            converted[stem + "gate_layer.weight"] = value[: rows // 2]
            converted[stem + "proj.weight"] = value[rows // 2 :]
            continue
        converted[key] = value
    return converted


# Without an entry from_single_file refuses, and GPU hosts route every GGUF to diffusers; native sd.cpp has no Windows GPU build.
_UNREGISTERED_SINGLE_FILE_CLASSES: dict = {
    "QwenImage21Transformer2DModel": _qwen_image_21_checkpoint_to_diffusers,
    **_ORIGINAL_LAYOUT_CONVERTERS,
}


def _register_unregistered_single_file_classes(logger: Any = None) -> tuple:
    """Adds missing classes only, never overwriting upstream entries; skips absent ones, since diffusers getattr-resolves every entry on each load."""
    added: list = []
    try:
        import diffusers
        from diffusers.loaders import single_file_model as sfm
    except Exception:  # noqa: BLE001 - no diffusers, nothing to register into
        return ()
    for name, mapping_fn in _UNREGISTERED_SINGLE_FILE_CLASSES.items():
        if name in sfm.SINGLE_FILE_LOADABLE_CLASSES:
            continue
        try:
            if getattr(diffusers, name, None) is None:
                continue
        except Exception:  # noqa: BLE001 - a class whose module fails to import is not loadable
            continue
        sfm.SINGLE_FILE_LOADABLE_CLASSES[name] = {
            "checkpoint_mapping_fn": mapping_fn,
            "default_subfolder": "transformer",
        }
        added.append(name)
    if added and logger is not None:
        logger.info("diffusion.single_file: registered %s", ", ".join(added))
    return tuple(added)


def _restore_gguf_trimmed_dims(model: Any, state_dict: Any) -> Any:
    """Re-adds the leading size-1 dims GGUF drops, and only those; a wrong tensor still raises."""
    expected = model.state_dict()
    for name, want in expected.items():
        have = state_dict.get(name)
        if have is None or tuple(have.shape) == tuple(want.shape):
            continue
        want_shape, have_shape = tuple(want.shape), tuple(have.shape)
        if have.numel() != want.numel() or len(want_shape) <= len(have_shape):
            continue
        pad = len(want_shape) - len(have_shape)
        if want_shape[:pad] != (1,) * pad or want_shape[pad:] != have_shape:
            continue
        state_dict[name] = have.reshape(want_shape)
    return state_dict


def _model_index_sample_sigmas(base: str, base_local_dir: Optional[str]) -> Any:
    """``sample_sigmas`` from the model_index.json the pipeline was just built from, never from the network."""
    from .diffusion_comfy_components import read_model_index

    try:
        index = read_model_index(
            base,
            base_local_dir = base_local_dir,
            local_files_only = True,
            cache_dir = hub_cache_dir(),
        )
    except Exception:  # noqa: BLE001 - an uncached or unreadable index carries no grid
        return None
    return index.get(SAMPLE_SIGMAS_KEY) if isinstance(index, dict) else None


def _generation_defaults_for(
    repo_id: Optional[str],
    gguf_filename: Optional[str],
    base_repo: Optional[str],
    grid: Optional[tuple[float, ...]] = None,
    family: Any = None,
) -> Optional[dict[str, float]]:
    """The (steps, guidance) the OpenAI route renders this load with, header variant before base (FLUX.1's base is
    schnell), so the Images form can match it."""
    try:
        # FLUX.3 first: the variant check has matched the checkpoint to the family, while the names cannot tell the
        # variants apart (BFL's ComfyUI base files are flux3_image_klein_*.safetensors, no "base", and a base file can
        # sit on the distilled repo's companions).
        identifiers = (
            gguf_filename,
            content_variant_hint(repo_id, gguf_filename),
            repo_id,
            base_repo,
        )
        if is_flux3_family(family):
            identifiers = (family.name,) + identifiers
        steps, guidance = generation_params_with_grid(grid, *identifiers)
    except Exception:  # noqa: BLE001 - status must not fail on a recipe lookup
        return None
    return {"steps": int(steps), "guidance": float(guidance)}


def _refuse_lora_without_loader(fam: Any, loras: Any) -> None:
    """A family whose pipeline has no LoRA loader refuses an active selection before anything loads."""
    from core.inference import diffusion_lora

    name = getattr(fam, "name", None)
    if _has_active_lora(loras) and not diffusion_lora.diffusers_lora_family_supported(name):
        raise ValueError(
            diffusion_lora.diffusers_lora_family_message(getattr(fam, "label", None) or name)
        )


_WHOLE_PIPELINE_GGUF_LORA_MSG = (
    "LoRA is not available on a whole-pipeline GGUF: its UNet runs as the stored GGUF weights, which cannot carry "
    "adapters. Load the .safetensors checkpoint to use a LoRA, or clear the LoRA selection."
)


def _whole_pipeline_gguf_pick(repo_id: Optional[str], gguf_filename: Optional[str]) -> bool:
    """True when the pick is an on-disk GGUF carrying a whole single-file pipeline (UNet + text encoders + VAE)."""
    return whole_pipeline_gguf_family(local_pick_file(repo_id, gguf_filename)) is not None


def _dequantize_gguf_outside_linears(
    model: Any,
    dtype: Any,
    logger: Any = None,
) -> int:
    """Dequantise GGUF-packed params outside GGUFLinear (only those dequantise per forward; numpy has no bf16, so BF16
    stays raw bytes too). Qwen-Image-Layered's BF16 addition_t_embedding otherwise fails the first step with 3072 vs
    6144. A few KB, kept dense at ``dtype``. Returns the count."""
    try:
        import torch
        from diffusers.quantizers.gguf.utils import (
            GGUFLinear,
            GGUFParameter,
            dequantize_gguf_tensor,
        )
    except Exception:  # noqa: BLE001 - no GGUF support, nothing packed to fix
        return 0
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return 0
    fixed: list = []
    for module_name, module in named_modules():
        if isinstance(module, GGUFLinear):
            continue
        for name, param in list(module.named_parameters(recurse = False)):
            if not isinstance(param, GGUFParameter):
                continue
            dense = dequantize_gguf_tensor(param).to(dtype)
            setattr(
                module,
                name,
                torch.nn.Parameter(dense.as_subclass(torch.Tensor), requires_grad = False),
            )
            fixed.append(f"{module_name}.{name}" if module_name else name)
    if fixed and logger is not None:
        logger.info(
            "diffusion.gguf: dequantised %d non-linear parameter(s): %s",
            len(fixed),
            ", ".join(fixed[:8]),
        )
    return len(fixed)


def _install_gguf_dim_restore(logger: Any) -> None:
    """Restores trimmed GGUF dims; patch both names, since single_file_model holds its own module-level import of the function. Idempotent."""
    try:
        from diffusers.loaders import single_file_model as sfm
        from diffusers.models import model_loading_utils as mlu

        original = mlu.load_model_dict_into_meta
        if getattr(original, "_unsloth_dim_restore", False):
            return

        def _restoring_loader(model, state_dict, *args: Any, **kwargs: Any):
            try:
                state_dict = _restore_gguf_trimmed_dims(model, state_dict)
            except Exception:  # noqa: BLE001 - never turn a load failure into a different one
                pass
            return original(model, state_dict, *args, **kwargs)

        _restoring_loader._unsloth_dim_restore = True
        mlu.load_model_dict_into_meta = _restoring_loader
        if getattr(sfm, "load_model_dict_into_meta", None) is not None:
            sfm.load_model_dict_into_meta = _restoring_loader
    except Exception as exc:  # noqa: BLE001 - loader-compat shim only, never fail the load
        logger.warning("diffusion.gguf: dim-restore shim not installed: %s", exc)


def _family_model_class(diffusers: Any, name: str) -> Any:
    """``getattr(diffusers, name)``, or Studio's in-tree class of that name (MOSS-Image), imported on first use."""
    if name in ("MOSSImagePipeline", "MOSSImageTransformer", "MOSSImageScheduler"):
        from .diffusion_moss_image import in_tree_class
        return in_tree_class(name)
    return getattr(diffusers, name)


@functools.lru_cache(maxsize = None)
def _no_recast_pipeline_class(pipe_cls: Any) -> Any:
    """Drops the dtype half of .to(): from_pipe shares resident modules, so a cast rewrites the loaded model; cached, one subclass per pipe_cls."""

    def to(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        args = tuple(arg for arg in args if not isinstance(arg, torch.dtype))
        kwargs.pop("dtype", None)
        if not args and not kwargs:
            return self
        return super(no_recast, self).to(*args, **kwargs)

    no_recast = type(pipe_cls.__name__, (pipe_cls,), {"to": to})
    return no_recast


_NO_WEIGHT = object()


def _torchao_offload_decline(what: str, plan: Any) -> str:
    """Why a torchao build was declined under ``plan``, quoting the floor that keeps it resident."""
    estimates = getattr(plan, "estimates", None) or {}
    floor = estimates.get("resident_transformer_floor_mib")
    streamed = f", {floor} MiB even with the text encoders streamed" if floor is not None else ""
    return (
        f"{what} '{plan.offload_policy}' offload here "
        f"({estimates.get('resident_required_mib')} MiB required vs a "
        f"{estimates.get('safe_device_budget_mib')} MiB budget{streamed}), "
        "and torchao tensors cannot be offloaded"
    )


def _has_active_lora(loras: Any) -> bool:
    """Reads .weight first, as unpacking a LoraSpec yields field pairs and a 0 weight would read active."""
    for entry in loras or ():
        weight = getattr(entry, "weight", _NO_WEIGHT)
        if weight is _NO_WEIGHT:
            try:
                _lid, weight = entry
            except (TypeError, ValueError):  # an unrecognised shape is not a disabled adapter
                return True
        try:
            if float(weight) != 0:
                return True
        except (TypeError, ValueError):  # likewise: an unreadable weight must not skip the bake
            return True
    return False


def _planned_quant_scheme(
    fam: Optional[DiffusionFamily],
    target: Any,
    requested: Optional[str],
    *,
    base_repo: Optional[str],
    prequant_path: Optional[str],
) -> Optional[str]:
    """The scheme the load will resolve, asked with the same base and hosted-checkpoint probe as the load."""
    return select_transformer_quant_scheme(
        target,
        requested,
        family = getattr(fam, "name", None),
        base_repo = base_repo,
        has_prequant = _prequant_probe(fam, base_repo = base_repo, prequant_path = prequant_path),
    )


def _is_local_cache_entry(source: Any) -> bool:
    """A source resolved from Studio's local pre-quant cache, not a hosted repo or an operator's file."""
    if getattr(source, "kind", None) != "path":
        return False
    try:
        from .diffusion_prequant_cache import path_in_cache
        return bool(path_in_cache(getattr(source, "location", None)))
    except Exception:  # noqa: BLE001
        return False


def _prequant_probe(
    fam: Optional[DiffusionFamily], *, base_repo: Optional[str], prequant_path: Optional[str]
) -> Callable[[str], bool]:
    return lambda candidate: (
        fam is not None
        and usable_prequant_source(fam, candidate, path_override = prequant_path, base_repo = base_repo)
        is not None
    )


def _uncached_prequant_repo(
    fam: Optional[DiffusionFamily],
    target: Any,
    requested: Optional[str],
    *,
    base_repo: Optional[str],
    prequant_path: Optional[str],
    online: Optional[bool] = None,
) -> Optional[str]:
    """The hosted prequant repo an auto quant would download, or None; load and download plan must agree. Never raises."""
    try:
        scheme = _planned_quant_scheme(
            fam, target, requested, base_repo = base_repo, prequant_path = prequant_path
        )
        if scheme is None:
            return None
        source = usable_prequant_source(
            fam, scheme, path_override = prequant_path, base_repo = base_repo
        )
        if source is None or source.kind != "repo":
            return None
        if prequant_checkpoint_cached(source, cache_dir = hub_cache_dir(), online = online):
            return None
        return source.location
    except Exception:  # noqa: BLE001 - a probe that cannot answer keeps the prequant shortcut
        return None


def _dense_transformer_cached(
    base_repo: Optional[str],
    *,
    companion_files: Optional[Sequence[str]] = None,
    transformer_files: Optional[Sequence[str]] = None,
) -> bool:
    """True only if EVERY transformer shard is cached in the repo the prefetch fetches (mirror judged on the full list); no listing means False. Never raises."""
    base = (base_repo or "").strip()
    if not base or not transformer_files:
        return False
    try:
        fetch_repo = prefer_ungated_mirror(
            base, files = [*(companion_files or ()), *transformer_files]
        )
        return cache_holds_files(fetch_repo, transformer_files)
    except Exception:  # noqa: BLE001 -- a cache we cannot read is not a verdict
        return False


def _local_base_transformer_present(base_repo: Optional[str]) -> bool:
    """A local base has no Hub listing, so an empty file list must not read as a missing transformer."""
    base = (base_repo or "").strip()
    if not base:
        return False
    try:
        transformer = Path(base).expanduser() / "transformer"
        return transformer.is_dir() and any_not_appledouble_metadata(
            transformer.glob("*.safetensors")
        )
    except OSError:  # an id with invalid path characters is simply not a directory
        return False


_UNSHARDED_PIPELINE_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)\.(?:bin|safetensors)$"
)


def _transformer_folder_complete(folder: Path) -> bool:
    """Complete checkpoint for diffusers' default load: safetensors index first, never a ``.bin`` index."""
    if not folder.is_dir():
        return False
    names = {f.name for f in folder.iterdir() if f.is_file()}
    indexes = [
        n
        for n in names
        if n.endswith(".safetensors.index.json") and _DEFAULT_PIPELINE_WEIGHT_INDEX_RE.match(n)
    ]
    if indexes:
        for index in indexes:
            with open(folder / index, encoding = "utf-8") as fh:
                shards = set((json.load(fh).get("weight_map") or {}).values())
            if shards and shards <= names:
                return True
        return False
    return any(_UNSHARDED_PIPELINE_WEIGHT_RE.match(n) for n in names)


def _dense_candidate_is_prequant(
    fam: Optional[DiffusionFamily],
    target: Any,
    requested: Optional[str],
    *,
    base_repo: Optional[str],
    prequant_path: Optional[str],
) -> bool:
    """A None candidate may be a download-sized disk gate, so re-ask the prequant resolver; never raises, and unanswerable reads as dense."""
    try:
        candidate = resolve_dense_quant_candidate(
            fam = fam,
            target = target,
            requested = requested,
            base_repo = base_repo,
            prequant_path = prequant_path,
            force_dense = False,
            logger = None,
        )
        if candidate is not None:
            return bool(candidate.prequant)
        scheme = _planned_quant_scheme(
            fam, target, requested, base_repo = base_repo, prequant_path = prequant_path
        )
        if scheme is None:
            return False
        return (
            usable_prequant_source(fam, scheme, path_override = prequant_path, base_repo = base_repo)
            is not None
        )
    except Exception:  # noqa: BLE001 - a probe that cannot answer keeps the decline
        return False


# Unset outside a load, which then plans on the max tier's (larger) activations.
_PLANNED_SPEED_MODE: contextvars.ContextVar[Any] = contextvars.ContextVar("_PLANNED_SPEED_MODE")
_SPEED_UNKNOWN = object()


def _plans_at_requested_speed(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _PLANNED_SPEED_MODE.set(kwargs.get("speed_mode"))
        try:
            return fn(*args, **kwargs)
        finally:
            _PLANNED_SPEED_MODE.reset(token)

    return wrapper


def _vae_on_other_card(pipe: Any, guard_target: Any) -> Any:
    """The CUDA device of ``pipe.vae`` when a multi-GPU split put it on another card than ``guard_target``, else None."""
    from .diffusion_multigpu import _module_device, _split_home, is_split

    if not is_split(pipe) and _split_home(pipe) is None:
        return None
    vae = getattr(pipe, "vae", None)
    try:
        device = _module_device(vae) if vae is not None else None
    except Exception:  # noqa: BLE001 - an unreadable VAE keeps the primary's verdict
        return None
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    if device.index in (None, placed_cuda_ordinal(guard_target)):
        return None
    return device


def _guard_vae_card(
    pipe: Any, guard_target: Any, guard_kwargs: dict, restore_vae: Any, logger: Any
) -> tuple[Any, bool]:
    """Re-run the generate-time activation guard on the VAE's card when a multi-GPU split put it on another card than
    the denoiser: the decode allocates there, and the primary's free memory says nothing about it. Engages VAE tiling
    when only a tiled decode fits that card (unless already tiled) and raises the guard's refusal when nothing fits.
    Returns ``(restore_vae, tiled)`` for a tiling it engaged, else ``(None, False)``. Conservative: the whole request's
    estimate is checked against the VAE's card."""
    device = _vae_on_other_card(pipe, guard_target)
    if device is None:
        return None, False
    with diffusion_device_scope(device.index):
        vae_kwargs = {
            **guard_kwargs,
            "device_memory": reclaimable_snapshot_device_memory(guard_target),
        }
    verdict = raise_on_image_activation_shortfall(**vae_kwargs)
    if verdict.action != ACTIVATION_TILE or restore_vae is not None:
        return None, False
    restore, tiled = engage_vae_tiling(pipe, logger = logger)
    try:
        if not tiled and not verdict.overridden:
            raise_on_image_activation_shortfall(**{**vae_kwargs, "vae_tile_side": None})
        elif (
            int(vae_kwargs.get("batch_size") or 1) > 1
            and vae_kwargs.get("vae_sliced")
            and not verdict.overridden
            and not vae_is_sliced(getattr(pipe, "vae", None))
        ):
            raise_on_image_activation_shortfall(**{**vae_kwargs, "vae_sliced": False})
    except ValueError:
        # The refusal: the caller never receives the handle, so undo the tiling here.
        if restore is not None:
            restore()
        raise
    except Exception as exc:  # noqa: BLE001 - fail OPEN on a broken probe and keep the tiling, as the caller does
        (logger or get_logger(__name__)).warning(
            "diffusion.generate: VAE card headroom re-check skipped (%s)", exc
        )
    except BaseException:
        if restore is not None:
            restore()
        raise
    return restore, tiled


def _guard_layer_cards(pipe: Any, guard_target: Any, guard_kwargs: dict) -> int:
    """The generate-time check on every other card a layer split runs blocks on. Such a card holds only blocks: no
    text encode, no VAE decode, and one block's activations at a time, so the whole-request estimate the primary is
    checked against does not apply. It needs what the plan reserved there at the default resolution, scaled by how
    much this request's activations grow past that default. ``pipe`` is the LOADED pipeline (a from_pipe workflow
    pipeline carries no split record). Raises the guard's refusal (a ValueError); returns the number of cards
    checked. Fails open on anything unknown."""
    from .diffusion_layer_split_plan import applied_layer_split, secondary_block_cards
    from .diffusion_memory import _device_user_budget_reserve_mib, _oversized_generate_override

    split = applied_layer_split(pipe)
    if split is None or guard_kwargs.get("allow_oversized") or _oversized_generate_override():
        return 0
    try:
        sized = dict(
            batch_size = guard_kwargs.get("batch_size") or 1, family = guard_kwargs.get("family")
        )
        needed = estimate_image_runtime_mib(
            width = guard_kwargs.get("width"),
            height = guard_kwargs.get("height"),
            condition_pixels = int(guard_kwargs.get("condition_pixels") or 0),
            **sized,
        )
        planned = estimate_image_runtime_mib(width = None, height = None, **{**sized, "batch_size": 1})
        scale = max(1.0, float(needed) / float(planned)) if planned else 1.0
    except Exception:  # noqa: BLE001 - a broken estimate never blocks a generation
        return 0
    checked = 0
    for ordinal in secondary_block_cards(pipe):
        working, reserve = split.working_mib(ordinal)
        if working + reserve <= 0:
            continue
        with diffusion_device_scope(int(ordinal)):
            card = reclaimable_snapshot_device_memory(guard_target)
        free = card.free_mib
        if free is None:
            continue
        free = int(free) - _device_user_budget_reserve_mib(card, card.total_mib or free)
        checked += 1
        need = int(working * scale) + reserve
        if need > int(free):
            raise ImageActivationShortfallError(
                f"Generating at {guard_kwargs.get('width')}x{guard_kwargs.get('height')} needs about "
                f"{need / 1024:.2f} GB of working memory on cuda:{int(ordinal)}, which runs part of the model split "
                f"across GPUs, but only about {max(0, int(free)) / 1024:.2f} GB is free there. Generate at a smaller "
                "resolution, or free GPU memory on that GPU. To try anyway, turn on 'Allow oversized generations' "
                "under Advanced on the Images page."
            )
    return checked


def _layer_split_expected(
    pipe: Any,
    plan: Any,
    refine: Optional[Callable[..., Any]] = None,
) -> bool:
    """Whether placement will cut the denoiser across cards: the multi-GPU refinement asked early, with the sizes
    loaded so far. The speed layer runs before placement, and on a split every block is resident on some card, so it
    is set up for that (regional compile with fullgraph, int8 kernels, graphs) and not for the one-card offload plan
    the split replaces. A split declined later lands its one-card fallback, the same path a transient OOM re-plan
    takes. ``refine``: the backend's own refinement (video declines some families)."""
    if plan is None or getattr(plan, "offload_policy", OFFLOAD_NONE) == OFFLOAD_NONE:
        return False
    try:
        preview = (refine or refine_plan_for_multigpu)(pipe, plan, None)
    except Exception:  # noqa: BLE001 - the one-card plan decides
        return False
    return getattr(preview, "layer_split", None) is not None


def _companion_split_expected(
    pipe: Any,
    plan: Any,
    refine: Optional[Callable[..., Any]] = None,
) -> bool:
    """Whether placement will keep the denoiser resident by moving other components to a second card (the multi-GPU
    component split), previewed like :func:`_layer_split_expected`. The one-card plan offloads there, so only this
    preview tells the quant compile rule that the denoiser ends up resident."""
    if plan is None or getattr(plan, "offload_policy", OFFLOAD_NONE) == OFFLOAD_NONE:
        return False
    try:
        preview = (refine or refine_plan_for_multigpu)(pipe, plan, None)
    except Exception:  # noqa: BLE001 - the one-card plan decides
        return False
    return (
        getattr(preview, "secondary_ordinal", None) is not None
        and getattr(preview, "offload_policy", None) == OFFLOAD_NONE
    )


def _native_quant_needs_compile(target: Any, native_scheme: Optional[str], resident: bool) -> bool:
    """Whether a torchao-free quantised DiT must be compiled like a torchao one: the NVIDIA int8 scheme (W8A8
    ``torch._int_mm``, chosen for an offload plan) once the DiT stays resident, on one card or split across cards,
    where eager runs it ~3.5x slower (LTX-2 split: 0.90-0.97 vs 0.26 s/step). ROCm and Apple keep their own native
    path and its compile policy; a streamed DiT keeps the offload path's."""
    return (
        native_scheme is not None
        and bool(resident)
        and native_offload_host(target)
        and not native_quant_host(target)
    )


def _restore_step_cache_after_layer_split(
    pipe: Any, views: Any, **cache_kwargs: Any
) -> Optional[str]:
    """The split takes the load-time First-Block-Cache off before it moves the blocks; once they sit on their cards the
    cache goes back on (``diffusion_cache`` copies its residuals between the head and tail cards). Returns the mode
    that engaged on ``pipe`` (None: it declined, the denoiser runs uncached)."""
    engaged = None
    for view in views:
        mode = apply_step_cache(view, mode = TC_FBCACHE, **cache_kwargs)
        if view is pipe:
            engaged = mode
    return engaged


def _step_cache_dropped_by_layer_split(pipe: Any) -> bool:
    """Whether the layer split placement took the load-time First-Block-Cache off ``pipe.transformer``
    (``diffusion_layer_split._drop_step_cache``): the marker is gone, so the load's cache locals are stale."""
    transformer = getattr(pipe, "transformer", None)
    return transformer is not None and not getattr(transformer, "_unsloth_step_cache", None)


def _drop_measured_reserve_for_layer_split(pipe: Any, plan: Any) -> bool:
    """A layer split replaced the one-card plan whose partial residency the measured reserve priced. Left in place, a
    conditioned request past it would give the split denoiser on-demand offload hooks onto the primary
    (``hook_resident_denoiser``), pulling every block off its card. True when dropped."""
    if getattr(plan, "layer_split", None) is None or not getattr(
        pipe, "_unsloth_measured_reserve", None
    ):
        return False
    try:
        pipe._unsloth_measured_reserve = None
        setattr(
            pipe, DECODE_LEND_ATTR, None
        )  # the lent encoder residency was priced on the same one-card plan
    except Exception:  # noqa: BLE001
        return False
    return True


RELEASE_BEFORE_TILING_ENV = "UNSLOTH_DIFFUSION_RELEASE_BEFORE_TILING"
RELEASE_BEFORE_TILING_LAYER = "release_before_tiling"
RELEASE_BEFORE_TILING_STATS = "_unsloth_release_before_tiling_stats"


def _release_before_tiling_off() -> bool:
    return os.environ.get(RELEASE_BEFORE_TILING_ENV, "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )


def _untiled_decode_shortfall_mib(vae: Any, z: Any) -> Optional[int]:
    """MiB short for decoding ``z`` untiled, by ``decode_tile_budget``'s rule on the whole frame. None off CUDA,
    on unknown figures, or when ``decode_platform_cap`` is under the frame."""
    from .diffusion_vae_tiling import (
        FREE_FRACTION,
        _free_mib,
        _fused,
        _geometry,
        decode_mib_per_latent,
        decode_platform_cap,
    )
    try:
        cap = decode_platform_cap(vae, z)
        if cap is not None and cap < int(z.shape[-2]) * int(z.shape[-1]):
            return None
        free = _free_mib(vae, z)
        if free is None:
            return None
        mib, scale = free
        frames = int(z.shape[2]) if z.dim() == 5 else 1
        batch = 1 if bool(getattr(vae, "use_slicing", False)) else int(z.shape[0])
        per_latent = decode_mib_per_latent(_geometry(vae)[0], _fused(vae), type(vae).__name__)
        need = batch * frames * int(z.shape[-2]) * int(z.shape[-1]) * per_latent * scale
        return int(-(-(need / FREE_FRACTION - mib) // 1))
    except Exception:  # noqa: BLE001 - unknown figures: decode as planned (tiled)
        return None


def _untiled_encode_shortfall_mib(vae: Any, x: Any) -> Optional[int]:
    """MiB short for encoding ``x`` untiled, the decode's rule (free less the VRAM Budget reserve); None off CUDA
    or on unknown figures."""
    from .diffusion_memory import cuda_reusable_cache_bytes, user_budget_reserve_bytes
    from .diffusion_vae_tiling import FREE_FRACTION, _fused, _geometry, encode_mib_per_latent

    try:
        import torch

        if getattr(x, "device", None) is None or x.device.type != "cuda":
            return None
        ratio = int(_geometry(vae)[0])
        per_latent = encode_mib_per_latent(ratio, _fused(vae), type(vae).__name__)
        if per_latent is None or ratio <= 0:
            return None
        free, total = torch.cuda.mem_get_info(x.device)
        free += cuda_reusable_cache_bytes(x.device)
        free -= user_budget_reserve_bytes(x.device, total)
        elem = next(vae.encoder.parameters()).element_size()
        # enable_fp16_vae_decode's fp32_encode upcasts the fp16 encoder and input for the encode itself
        if getattr(vae, "_unsloth_fp16_decode", False):
            elem = max(elem, 4)
        scale = elem / 2
        frames = int(x.shape[2]) if x.dim() == 5 else 1
        batch = 1 if bool(getattr(vae, "use_slicing", False)) else int(x.shape[0])
        latents = -(-int(x.shape[-2]) // ratio) * -(-int(x.shape[-1]) // ratio)
        need = batch * frames * latents * per_latent * scale
        return int(-(-(need / FREE_FRACTION - free / 2**20) // 1))
    except Exception:  # noqa: BLE001 - unknown figures: encode tiled
        return None


def _release_for_untiled(
    state_pipe: Any, shortfall: Callable[[], Optional[int]], device: Any, logger: Any, reason: str
) -> tuple[bool, Optional[Callable[[], None]]]:
    """(untiled?, restore). Streams the shortfall plus margin, then the rest once; nothing when even every resident
    group could not cover it (a freed MiB frees at most a MiB). Still short: pin back and tile."""
    short = shortfall()
    if short is None:
        return False, None
    if short <= 0:
        return True, None
    resident = resident_group_mib(state_pipe)
    if short > resident:
        return False, None
    restore: Optional[Callable[[], None]] = None
    try:
        for need in (min(resident, short + short // 4 + 256), None):
            if need is None:
                need = resident_group_mib(state_pipe)
                if need <= 0:
                    break
            step = release_resident_groups(state_pipe, need, logger, reason = reason, device = device)
            if step is None:
                break
            restore = _chain_restores(step, restore)
            again = shortfall()
            if again is not None and again <= 0:
                return True, restore
            if need >= resident:
                break
    except Exception as exc:  # noqa: BLE001 - a broken release keeps the tiled pass
        if logger is not None:
            logger.warning("diffusion.memory: streaming before %s failed (%s)", reason, exc)
    except BaseException:
        # a cancel between a release and its handover: pin back before propagating
        if restore is not None:
            restore()
        raise
    if restore is not None:
        restore()
    return False, None


def _release_for_untiled_decode(
    state_pipe: Any, vae: Any, z: Any, device: Any, logger: Any
) -> tuple[bool, Optional[Callable[[], None]]]:
    return _release_for_untiled(
        state_pipe,
        lambda: _untiled_decode_shortfall_mib(vae, z),
        device,
        logger,
        "an untiled VAE decode",
    )


def _release_for_untiled_encode(
    state_pipe: Any, vae: Any, x: Any, device: Any, logger: Any
) -> tuple[bool, Optional[Callable[[], None]]]:
    """Untiled only on a positive fit check: on WDDM an overrun spills to host memory instead of raising."""
    return _release_for_untiled(
        state_pipe,
        lambda: _untiled_encode_shortfall_mib(vae, x),
        device,
        logger,
        "an untiled VAE encode",
    )


def _defer_tiling_to_decode(
    pipe: Any,
    state_pipe: Any,
    verdict: Any,
    guard_target: Any,
    guard_kwargs: dict,
    restore_vae: Any,
    logger: Any,
) -> Any:
    """Tile verdict with resident offload groups: keep them through the denoise (as ComfyUI does) and decide at each
    VAE decode / encode, on its own input, whether streaming lets it run untiled (tiling changes pixels, streaming
    only costs time); groups pin back right after. An untiled encode OOM retries tiled. Wrappers sit on the public
    ``decode`` / ``encode`` slots, outside compiled regions. Returns the VAE restore extended to drop them."""
    vae = getattr(pipe, "vae", None)
    if (
        vae is None
        or restore_vae is None
        or verdict is None
        or verdict.action != ACTIVATION_TILE
        or verdict.overridden
        or _release_before_tiling_off()
        or not bool(getattr(vae, "use_tiling", False))
        or not callable(getattr(vae, "decode", None))
    ):
        return restore_vae
    resident = resident_group_mib(state_pipe)
    if resident <= 0:
        return restore_vae
    # No request-wide "hopeless" gate: the verdict prices the denoise, gone by decode time.
    try:
        if _vae_on_other_card(pipe, guard_target) is not None:
            return restore_vae
    except Exception:  # noqa: BLE001 - unreadable split: the primary's card decodes
        pass
    from . import decode_layers
    from .diffusion_vae_oom import _compiling, clear_partial_decode, drop_tracebacks, is_oom

    device = getattr(guard_target, "torch_device", None)
    stats = {"decodes": 0, "untiled": 0, "tiled": 0, "release_s": 0.0, "restore_s": 0.0,
             "released_mib": 0, "encodes": 0, "encode_untiled": 0, "encode_tiled": 0,
             "encode_released_mib": 0, "encode_oom": 0}  # fmt: skip
    vae.__dict__[RELEASE_BEFORE_TILING_STATS] = stats

    def make_decode(inner: Callable, layer: Any) -> Callable:
        depth = [0]

        def decode(*args: Any, **kwargs: Any) -> Any:
            if depth[0] or _compiling():
                return inner(*args, **kwargs)
            depth[0] += 1
            restore = None
            untiled = False
            try:
                z = args[0] if args else kwargs.get("z")
                t0 = time.perf_counter()
                before = resident_group_mib(state_pipe)
                untiled, restore = _release_for_untiled_decode(state_pipe, vae, z, device, logger)
                stats["decodes"] += 1
                stats["release_s"] += time.perf_counter() - t0
                stats["untiled" if untiled else "tiled"] += 1
                if restore is not None:
                    stats["released_mib"] += max(0, before - resident_group_mib(state_pipe))
                vae.use_tiling = not untiled
                return inner(*args, **kwargs)
            finally:
                vae.use_tiling = False
                if restore is not None:
                    t1 = time.perf_counter()
                    try:
                        restore()
                    except Exception as exc:  # noqa: BLE001 - the groups keep streaming
                        if logger is not None:
                            logger.warning(
                                "diffusion.memory: re-residency after the decode failed: %s", exc
                            )
                    stats["restore_s"] += time.perf_counter() - t1
                depth[0] -= 1

        return decode

    def make_encode(inner: Callable, layer: Any) -> Callable:
        depth = [0]

        def encode(*args: Any, **kwargs: Any) -> Any:
            if depth[0] or _compiling() or bool(getattr(vae, "use_tiling", False)):
                return inner(*args, **kwargs)
            depth[0] += 1
            restore = None
            try:
                x = args[0] if args else kwargs.get("x")
                before = resident_group_mib(state_pipe)
                untiled, restore = _release_for_untiled_encode(state_pipe, vae, x, device, logger)
                stats["encodes"] += 1
                stats["encode_untiled" if untiled else "encode_tiled"] += 1
                if restore is not None:
                    stats["encode_released_mib"] += max(0, before - resident_group_mib(state_pipe))
                vae.use_tiling = not untiled
                if not untiled:
                    return inner(*args, **kwargs)  # oversized or unknown: tiled, as main
                try:
                    return inner(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - the backstop behind a fit check that was wrong
                    if not is_oom(exc):
                        raise
                    drop_tracebacks(exc)
                stats["encode_oom"] += 1
                clear_partial_decode(vae, x)
                vae.use_tiling = True
                return inner(*args, **kwargs)
            finally:
                vae.use_tiling = False
                if restore is not None:
                    try:
                        restore()
                    except Exception as exc:  # noqa: BLE001 - the groups keep streaming
                        if logger is not None:
                            logger.warning(
                                "diffusion.memory: re-residency after the encode failed: %s", exc
                            )
                depth[0] -= 1

        return encode

    layers = [decode_layers.wrap(vae, "decode", RELEASE_BEFORE_TILING_LAYER, make_decode)]
    if callable(getattr(vae, "encode", None)):
        layers.append(decode_layers.wrap(vae, "encode", RELEASE_BEFORE_TILING_LAYER, make_encode))
    vae.use_tiling = False
    if logger is not None:
        logger.info(
            "diffusion.memory: VAE tiling waits for the decode; resident groups stay resident for the denoise"
        )

    def restore_all() -> None:
        for layer in reversed(layers):
            decode_layers.unwrap(layer)
        vae.use_tiling = True  # what the request's tiling left, for its own undo
        restore_vae()

    return restore_all


def _chain_restores(
    first: Optional[Callable[[], None]], then: Optional[Callable[[], None]]
) -> Optional[Callable[[], None]]:
    """One restore running ``first`` (the later release) and then ``then``; None when both are None."""
    if first is None:
        return then
    if then is None:
        return first

    def restore() -> None:
        try:
            first()
        finally:
            then()

    return restore


def _tile_for_verdict(
    pipe: Any,
    verdict: Any,
    guard_target: Any,
    guard_kwargs: dict,
    guard_batch: int,
    restore_vae: Any,
    logger: Any,
) -> Any:
    """Engage VAE tiling when the guard says only a tiled decode fits; returns the VAE restore handle. A multi-GPU split
    that decodes on another card is decided by that card's verdict alone: the primary never runs the decode, so its
    tile verdict must not switch the secondary's VAE to the (not bit-identical) tiled path."""
    vae_restore, vae_card_tiled = _guard_vae_card(
        pipe, guard_target, guard_kwargs, restore_vae, logger
    )
    if vae_card_tiled:
        return vae_restore
    if verdict.action != ACTIVATION_TILE or _vae_on_other_card(pipe, guard_target):
        return restore_vae
    restore_vae, vae_tiled = engage_vae_tiling(pipe, logger = logger)
    try:
        if not vae_tiled and not verdict.overridden:
            # Passed only because it would be tiled; tiling failed, so refuse on the untiled figure.
            raise_on_image_activation_shortfall(**{**guard_kwargs, "vae_tile_side": None})
        elif (
            guard_batch > 1
            and guard_kwargs["vae_sliced"]
            and not verdict.overridden
            and not vae_is_sliced(getattr(pipe, "vae", None))
        ):
            raise_on_image_activation_shortfall(**{**guard_kwargs, "vae_sliced": False})
    except ValueError:
        # the caller never receives the handle on a refusal: undo the tiling here
        if restore_vae is not None:
            restore_vae()
        raise
    except Exception as exc:  # noqa: BLE001 - fail OPEN on a broken probe and keep the tiling, as generate() does
        (logger or get_logger(__name__)).warning(
            "diffusion.generate: activation headroom re-check skipped (%s)", exc
        )
    except BaseException:
        if restore_vae is not None:
            restore_vae()
        raise
    return restore_vae


def plan_is_calibrated(plan: Any) -> bool:
    """Whether ``plan`` was fitted to the measured activation row rather than the flat estimate: the one-card plan, or
    a multi-GPU split priced with ``calibrated_split_headroom_mib``. The generate guard then charges conditioning
    (ControlNet, references) on top and drops the flat default-resolution exemption."""
    estimates = getattr(plan, "estimates", None) or {}
    if "calibrated_headroom_mib" in estimates:
        return True
    split = (
        getattr(plan, "layer_split", None) is not None
        or getattr(plan, "secondary_ordinal", None) is not None
    )
    return split and "calibrated_split_headroom_mib" in estimates


def _controlnet_streams(state: Any) -> bool:
    """Whether a ControlNet added at generate time streams from host memory instead of loading whole beside the base
    model: under any offload policy, a calibrated placement (budgets only the base model), and a multi-GPU split
    (reports "none", but its fit priced only the base pipeline, which one card would have offloaded)."""
    from .diffusion_multigpu import is_split
    return (
        getattr(state, "offload_policy", OFFLOAD_NONE) != OFFLOAD_NONE
        or bool(getattr(state, "calibrated_placement", False))
        or is_split(getattr(state, "pipe", None))
    )


def _measured_target(target: Any) -> bool:
    """Whether ``target`` is where the activation tables were measured: NVIDIA CUDA with a sub-quadratic SDPA kernel seen
    to run. Math-only attention, ROCm and an unanswered probe are not."""
    if getattr(target, "backend", None) != "cuda" or getattr(target, "vendor", None) != "nvidia":
        return False
    try:
        return bool(sdpa_subquadratic_confirmed(target))
    except Exception:  # noqa: BLE001 - unprobed attention keeps the flat estimate
        return False


def _calibrated_activation(
    fam: Any,
    target: Any,
    *,
    unpriced_references_ok: bool = False,
) -> Any:
    """Measured activations apply only where measured: NVIDIA with a sub-quadratic attention kernel.
    ``unpriced_references_ok``: skip the reference rule; the caller sizes references via measured_request_extra_mib."""
    if getattr(target, "backend", None) != "cuda" or getattr(target, "vendor", None) != "nvidia":
        return None
    # Measured without input images: references the guard cannot size (no reference_resolutions) would overflow.
    # FLUX.3 can: every reference is resized to 1 MP and the request guard charges those pixels on top.
    if (
        not unpriced_references_ok
        and getattr(fam, "reference", False)
        and not tuple(getattr(fam, "reference_resolutions", ()) or ())
        and not is_flux3_family(fam)
    ):
        return None
    # Measured at 16-bit only; fp32-promoted families (Z-Image pre-Ampere) do not qualify.
    compute = _resolve_diffusion_compute_dtype(fam, getattr(target, "dtype", None))
    if (_float_load_itemsize(compute) or 2) > 2:
        return None
    requested = _PLANNED_SPEED_MODE.get(_SPEED_UNKNOWN)
    try:
        max_speed = requested is _SPEED_UNKNOWN or normalize_speed_mode(requested) == SPEED_MAX
    except ValueError:
        max_speed = True
    activation = calibrated_image_activation(getattr(fam, "name", None), max_speed = max_speed)
    if activation is None:
        return None
    try:
        return activation if sdpa_subquadratic_confirmed(target) else None
    except Exception:  # noqa: BLE001 - unprobed attention keeps the flat estimate
        return None


def _quadratic_attention(
    target: Any,
    engaged_backend: Optional[str] = None,
    pipe: Any = None,
) -> bool:
    """Whether attention can only run on SDPA math (quadratic memory); False when unknown.
    An engaged non-native backend is a fused kernel, so the SDPA probe only speaks for native, unless that backend can
    hand calls to native SDPA (ROCm flash: masked calls, lengths it does not serve): then budget as native."""
    if engaged_backend is not None and str(engaged_backend) != "native":
        try:
            if pipe is None or not attention_may_run_native(pipe):
                return False
        except Exception:  # noqa: BLE001
            return False
    try:
        if not sdpa_math_only(target, _dit_head_dims(pipe)):
            return False
    except Exception:  # noqa: BLE001 - a broken probe must never block a generation
        return False
    try:
        from .diffusion_qwenimage21_math import bounded_math_attention
        return not bounded_math_attention(pipe)
    except Exception:  # noqa: BLE001 - known math stays quadratic unless every processor is verified
        return True


def _attention_scaling_known(target: Any, engaged_backend: Optional[str] = None) -> bool:
    """Whether ``_quadratic_attention``'s answer is evidence: a non-native backend engaged, or the SDPA probe saw math
    alone or a sub-quadratic kernel run. An unanswered probe (or no device target) is unknown."""
    if engaged_backend is not None and str(engaged_backend) != "native":
        return True
    if target is None:
        return False
    try:
        return bool(sdpa_math_only(target) or sdpa_subquadratic_confirmed(target))
    except Exception:  # noqa: BLE001
        return False


def _start_peak_probe(state: Any) -> Any:
    """Arm the learned-peak measurement for one render (CUDA loads that carry a peak identity), else None."""
    pipe = getattr(state, "pipe", None)
    _follow_live_graph_mode(state)
    identity = getattr(pipe, "_unsloth_peak_identity", None)
    if identity is None:
        return None
    try:
        from .diffusion_measured_peaks import PeakProbe, enabled

        index = _peak_device_index(state)
        graphs = cuda_graph.live_graph_count(index)
        generation = cuda_graph.capture_generation(index)
        accounted = getattr(pipe, "_unsloth_peak_graphs", None)
        if accounted is None or graphs < accounted:
            # the load's own graphs (every render starts above them), or fewer after a reset freed some
            first = accounted is None
            pipe._unsloth_peak_graphs = accounted = graphs
            if first or graphs == 0:
                # nothing alive on the card: no unrecorded capture can remain. A drop to a non-zero count may be
                # another pipeline's reset, so a capture this one made unrecorded keeps the generation dirty
                pipe._unsloth_peak_graph_gen = generation
        if getattr(pipe, "_unsloth_peak_graph_gen", None) is None:
            pipe._unsloth_peak_graph_gen = generation
        if graphs > accounted or generation > pipe._unsloth_peak_graph_gen:
            # an unrecorded render (cache-served, refused, failed) left graphs allocated, or replaced one in a full
            # cache (same count): a later peak above that start omits memory a fresh process allocates on its first
            # render
            return None
        if not _peak_identity_matches(state, identity):
            # this render runs a configuration the key does not name (a speed tier or step cache that toggled, a VAE
            # saver the guard engaged for it, another attention backend): its peak belongs to no key
            return None
        if not enabled():
            return None
        window_pad = stream_window_pad_mib(pipe)
        if window_pad is None:
            return (
                None  # the streamed baseline is unknown: the peak cannot be put on the common scale
            )
        probe = PeakProbe(getattr(state, "device", None))
        if probe.device is None:
            return None
        probe.window_pad = window_pad
        probe.failed_captures = cuda_graph.failed_capture_count()
        probe.conditioning_hits = _conditioning_cache_hits(pipe)
        return probe
    except Exception:  # noqa: BLE001 - measurement only
        return None


def _graphs_settled_eager(state: Any) -> bool:
    """No CUDA graph can still record over the denoiser placement (graphs key on weight addresses)."""
    pipe = getattr(state, "pipe", None)
    handles = tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ()) or tuple(
        getattr(state, "cuda_graphs", ()) or ()
    )
    try:
        if cuda_graph.live_graph_count(_peak_device_index(state)):
            return False
    except Exception:  # noqa: BLE001 - unreadable: something may be recorded
        return False
    for handle in handles:
        if getattr(handle, "poisoned", False):
            continue
        declined = getattr(handle, "_placement_declined", None)
        if (
            callable(declined)
            and declined()
            and getattr(handle, "fallback_handle", None) is None
            and getattr(handle, "fallback", None) is None
        ):
            continue
        return False
    return True


def _learned_peak_keys(
    state: Any,
    label: str,
    speed: Optional[str],
    *,
    eager_stand_in: bool = True,
) -> list:
    """Learned-peak keys, most specific first. ``eager_stand_in=False``: live key only (a dropped graph can peak above
    the capturing render)."""
    pipe = getattr(state, "pipe", None)
    cfg_off = CFG_OFF_PART in str(label).split("|")
    keys = [(label, speed)]
    if not eager_stand_in:
        return keys if speed is not None else []
    # a deferred compile tier records nothing: its eager peak (not exceeded when compiled) stands in
    keys.append((label, SPEED_OFF))
    loaded = getattr(pipe, "_unsloth_peak_identity", None)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        loaded_label = str(loaded[1])
        if cfg_off:
            loaded_label = f"{without_cfg_off(loaded_label)}|{CFG_OFF_PART}"
        keys += [(loaded_label, tier) for tier in (loaded[2], SPEED_OFF) if tier is not None]
    tier_info = getattr(pipe, WHOLE_TIER_ATTR, None)
    eager_label = tier_info.get("eager_peak_label") if isinstance(tier_info, dict) else None
    if eager_label:
        if cfg_off:
            eager_label = f"{eager_label}|{CFG_OFF_PART}"
        keys.append((eager_label, SPEED_OFF))
    return list(dict.fromkeys(keys))


def _learned_peak_for_keys(
    family: Any,
    keys: Sequence[tuple],
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int,
    workflow: Optional[str] = None,
    phase: Optional[str] = None,
) -> Optional[int]:
    for key_label, tier in keys:
        found = measured_image_runtime_mib(
            family,
            tier,
            width = width,
            height = height,
            batch_size = batch_size,
            quant = with_workflow(key_label, workflow),
            static_seed = False,
            phase = phase,
        )
        if found is not None:
            return found
    return None


def _follow_live_graph_mode(state: Any) -> None:
    """Keep a deferred profile's identity on the CUDA graph mode actually running (armed, then possibly dropped)."""
    pipe = getattr(state, "pipe", None)
    identity = getattr(pipe, "_unsloth_peak_identity", None)
    if not getattr(pipe, TIER_MOVED_ATTR, False) or not isinstance(identity, tuple):
        return
    if len(identity) != 3 or "|graphs:" not in str(identity[1]):
        return
    try:
        live = with_label_part(str(identity[1]), "graphs", _tier_graph_mode(state))
        if live != identity[1]:
            pipe._unsloth_peak_identity = (identity[0], live, identity[2])
    except Exception:  # noqa: BLE001 - recording is optional
        pass


def _tier_request_peak(
    state: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int,
    cfg: Optional[bool],
    workflow: Optional[str] = None,
) -> Optional[int]:
    """Learned peak of a deferred profile's engaged tier; None until that tier has its own measurement."""
    pipe = getattr(state, "pipe", None)
    if not getattr(pipe, TIER_MOVED_ATTR, False):
        return None
    try:
        family, label, speed = _live_peak_identity(state)
    except Exception:  # noqa: BLE001
        return None
    if speed is None:
        return None
    label = without_cfg_off(str(label))
    if (
        cfg is False
        and getattr(pipe, "_unsloth_peak_expects_cfg", False)
        and cfg_off_peaks_enabled()
    ):
        label = f"{label}|{CFG_OFF_PART}"
    return _learned_peak_for_keys(
        family,
        _learned_peak_keys(state, label, speed, eager_stand_in = False),
        width = width,
        height = height,
        batch_size = batch_size,
        workflow = workflow,
    )


def _hold_captured_release(state: Any, extra_mib: int) -> int:
    """At least the release a live CUDA graph was captured over: graphs key on weight addresses, so re-pinning
    groups would re-capture."""
    pipe = getattr(state, "pipe", None)
    held = getattr(pipe, "_unsloth_captured_release_mib", None)
    if held is None:
        return extra_mib
    try:
        if not cuda_graph.live_graph_count(_peak_device_index(state)):
            pipe._unsloth_captured_release_mib = None
            return extra_mib
    except Exception:  # noqa: BLE001 - unknown: keep the placement the graph was captured over
        pass
    return max(int(extra_mib), int(held))


def _note_captured_release(state: Any, extra_mib: int, generation_before: Optional[int]) -> None:
    pipe = getattr(state, "pipe", None)
    if generation_before is None or not getattr(pipe, TIER_MOVED_ATTR, False):
        return
    try:
        if cuda_graph.capture_generation(_peak_device_index(state)) > generation_before:
            pipe._unsloth_captured_release_mib = int(max(0, extra_mib))
    except Exception:  # noqa: BLE001
        pass


def _whole_tier_for_request(
    state: Any,
    *,
    steps: int,
    width: Optional[int],
    height: Optional[int],
    batch_size: int,
    cfg: Optional[bool],
    condition_pixels: int,
    workflow: Optional[str] = None,
    logger: Any = None,
) -> Optional[tuple[float, float, int]]:
    """``(whole s, streamed s, activation MiB)`` if this request should run the denoiser whole, else None.
    Blocks compile below their offload hooks, so compiled code never sees the pin; a pending background compile or a
    CUDA graph that could still record keeps the streamed plan. Without a learned pre-decode peak the call streams."""
    pipe = getattr(state, "pipe", None)
    if getattr(state, "offload_policy", None) != OFFLOAD_GROUP or getattr(
        state, "transformer_cache", None
    ):
        return None
    if int(condition_pixels or 0) > 0:
        # no learned peak covers condition images
        return None
    try:
        family, label, speed = _live_peak_identity(state)
    except Exception:  # noqa: BLE001
        return None
    tier_info = getattr(pipe, WHOLE_TIER_ATTR, None)
    if isinstance(tier_info, dict) and speed == SPEED_OFF:
        # remembered: a deferred profile later rewrites the load's key, and post-capture renders record under none
        tier_info.setdefault("eager_peak_label", without_cfg_off(str(label)))
    costs = whole_tier_request_costs(
        pipe,
        steps = steps,
        width = width,
        height = height,
        batch_size = batch_size,
        cfg = cfg,
        condition_pixels = condition_pixels,
    )
    if costs is None or costs[0] >= costs[1]:
        return None
    bg = getattr(state, "bg_compile", None)
    if bg is not None and bg.pending():
        return None
    if not _graphs_settled_eager(state):
        return None
    if speed is None:
        return None
    activation = _learned_peak_for_keys(
        family,
        _learned_peak_keys(state, label, speed),
        width = width,
        height = height,
        batch_size = batch_size,
        workflow = workflow,
        phase = "pre_decode",
    )
    if activation is None:
        if logger is not None:
            logger.info(
                "diffusion.memory: a %d-step render prices the denoiser whole (%.2f s vs %.2f s streamed), but no "
                "pre-decode peak is learned for its size yet: it streams",
                int(steps),
                costs[0],
                costs[1],
            )
        return None
    return costs[0], costs[1], int(activation)


def _peak_device_index(state: Any) -> Optional[int]:
    """CUDA ordinal the pipeline renders on (``state.device`` may be un-indexed: the thread's current card)."""
    try:
        import torch
        dev = torch.device(getattr(state, "device", None) or "cuda")
        return dev.index if dev.index is not None else torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        return None


def _live_peak_identity(state: Any) -> tuple:
    """The load's learned-peak identity (the canonical configuration captured at load) with every part that can change
    after the load read from the live state: speed tier, step cache, VAE savers, attention backend. A render records
    only when this equals the identity."""
    pipe = getattr(state, "pipe", None)
    family, label, _ = tuple(getattr(pipe, "_unsloth_peak_identity", ("", "", None)))[:3]
    label = with_label_part(
        str(label), "cache", str(getattr(state, "transformer_cache", None) or "off")
    )
    label = with_label_part(label, "vae", live_vae_saver_tag(pipe))
    label = with_label_part(label, "attn", getattr(state, "attention_backend", None) or "native")
    if "|graphs:" in label and getattr(pipe, TIER_MOVED_ATTR, False):
        label = with_label_part(label, "graphs", _tier_graph_mode(state))
    elif "|graphs:" in label:
        label = with_label_part(label, "graphs", _live_graph_mode(state))
    speed = getattr(state, "speed_mode", None)
    if "compile_fallback_eager" in tuple(getattr(state, "speed_optims", ()) or ()):
        speed = None  # the compiled tier fell back to eager: an execution path no key names
    try:
        from .diffusion_bg_compile import eager_forced
        if eager_forced():
            speed = None  # a background compile is still pending: this render runs eager under a compiled tier
    except Exception:  # noqa: BLE001
        speed = None
    # Unknown guidance, or CFG on a family that never runs it, records nothing.
    family_obj = getattr(state, "family", None)
    expects_cfg = bool(getattr(family_obj, "uses_negative_prompt", False))
    cfg = getattr(pipe, "_unsloth_peak_cfg", expects_cfg)
    if cfg is None:
        speed = None
    elif bool(cfg) != expects_cfg:
        if expects_cfg and cfg_off_peaks_enabled():
            label = f"{label}|{CFG_OFF_PART}"
        else:
            speed = None
    if getattr(pipe, "_unsloth_loras", None):
        speed = None  # LoRA adapters add rank-dependent activations no key names
    return (family, label, speed)


def _peak_identity_matches(state: Any, identity: Any) -> bool:
    """The render runs the configuration the load's identity names (``cfg:off`` aside)."""
    family, label, speed = _live_peak_identity(state)
    return (family, without_cfg_off(label), speed) == tuple(identity)


def _live_graph_mode(state: Any) -> str:
    """``step`` / ``blocks`` (``_engaged_graph_mode``) while the load's CUDA graphs are armed and not dropped
    (``live_status``), else ``off``."""
    try:
        _, optims = cuda_graph.live_status(
            getattr(state, "resolved", None),
            getattr(state, "speed_optims", ()),
            getattr(state, "cuda_graphs", ()),
        )
        if "cuda_graph" not in optims:
            return "off"
        return _engaged_graph_mode(getattr(state, "pipe", None), {"cuda_graph": True})
    except Exception:  # noqa: BLE001
        return "unknown"


TIER_FALLBACK_GRAPH_MODE = "step_blocks"


def _tier_graph_mode(state: Any) -> str:
    """``_live_graph_mode``, but the armed mode before the first capture."""
    try:
        if "cuda_graph" not in tuple(getattr(state, "speed_optims", ()) or ()):
            return "off"
        pipe = getattr(state, "pipe", None)
        handles = tuple(getattr(state, "cuda_graphs", ()) or ()) or tuple(
            getattr(pipe, "_unsloth_cuda_graphs", ()) or ()
        )
        armed = _engaged_graph_mode(pipe, {"cuda_graph": True})
        if not handles:
            return "off"
        if any(getattr(h, "fallback_handle", None) not in (None, h) for h in handles):
            # per-block fallback (GraphedForward._engage_fallback) is invisible to stats(): its own peak mode
            return TIER_FALLBACK_GRAPH_MODE
        totals = cuda_graph.stats(handles)
        if totals.get("graphs"):
            return armed
        if totals.get("captures") or totals.get("poisoned") or _graphs_settled_eager(state):
            return "off"
        return armed
    except Exception:  # noqa: BLE001
        return "unknown"


def _peak_te_label(te_quant: Any, outcome: Any) -> str:
    """The text encoders' precision as the learned-peak key names it. A partial cast (some encoders quantised, others
    left dense) is no build the key can name: ``unknown``, so it neither reads nor records a learned peak."""
    if getattr(outcome, "partial", False):
        return "unknown"
    return str(te_quant or "dense")


def _engaged_graph_mode(pipe: Any, applied: Any) -> str:
    """The graph implementation placement armed: whole-step and per-block graphs hold different statics, slot rings
    and pools, so their peaks never share a key."""
    if not (applied or {}).get("cuda_graph"):
        return "off"
    return "blocks" if getattr(pipe, "_unsloth_cuda_graph_mode", None) == "blocks" else "step"


def _local_weights_digest(path: Any) -> Optional[str]:
    """Size and mtime of every weight file under a local pipeline folder (``checkpoint_fingerprint`` reads only its
    manifests); None for a repo id or an unreadable folder."""
    try:
        import hashlib

        root = Path(str(path)).expanduser()
        if not root.is_dir():
            return None
        digest = hashlib.sha256()
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in (
                ".safetensors",
                ".bin",
                ".gguf",
                ".pt",
                ".pth",
                ".ckpt",
            ):
                st = f.stat()
                digest.update(f"{f.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}\0".encode())
        return digest.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _peak_model_identity(
    repo_id: Any,
    gguf_filename: Optional[str],
    kind: Optional[str],
    base: Any,
    fetch_base: Optional[str] = None,
) -> str:
    """The exact build for the learned-peak key: the denoiser source and its weights' revision / fingerprint, plus the
    companion base and its revision for a GGUF / single-file load. ``unknown`` (no learned peak used or recorded)
    when a revision cannot be read. ``fetch_base``: the repo the load actually read ``base`` from."""
    from .diffusion_component_sizes import checkpoint_fingerprint
    from .diffusion_families import mirror_repo

    def _fingerprint(
        repo: Any,
        filename: Optional[str] = None,
        fetched: Optional[str] = None,
    ) -> tuple[Any, Any]:
        if fetched and str(fetched) != str(repo):
            return fetched, checkpoint_fingerprint(fetched, filename)
        revision = checkpoint_fingerprint(repo, filename)
        if revision or filename or fetched:
            return repo, revision
        # No load decision: a fresh install may hold only the unsloth/ mirror.
        mirror = mirror_repo(str(repo)) if repo else None
        if mirror and mirror != str(repo):
            mirrored = checkpoint_fingerprint(mirror)
            if mirrored:
                return mirror, mirrored
        return repo, revision

    pipeline = (kind or "pipeline") == "pipeline"
    repo_id, revision = _fingerprint(
        repo_id, gguf_filename, fetch_base if pipeline and str(base) == str(repo_id) else None
    )
    source = f"{repo_id}/{gguf_filename}" if gguf_filename else str(repo_id)
    if not revision:
        return "unknown"
    if str(revision).startswith("local:"):
        weights = _local_weights_digest(repo_id)
        if not weights:
            return "unknown"
        revision = f"{revision}:{weights}"
    identity = f"{source}@{revision}"
    if not pipeline and base and str(base) != str(repo_id):
        base, base_revision = _fingerprint(base, None, fetch_base)
        if not base_revision:
            return "unknown"
        if str(base_revision).startswith("local:"):
            weights = _local_weights_digest(base)
            if not weights:
                return "unknown"
            base_revision = f"{base_revision}:{weights}"
        identity += f"+{base}@{base_revision}"
    return identity


def _conditioning_cache_hits(pipe: Any) -> int:
    """Encodes served from the in-memory prompt cache or the persistent conditioning cache so far: a render with a
    hit skipped the text-encoder phase, so its peak covers only part of what a new prompt needs."""
    hits = 0
    try:
        cache = prompt_cache.cache_for(pipe)
        if cache is not None:
            hits += int(cache.stats.get("hits", 0))
        stats = getattr(pipe, "_unsloth_cond_cache_stats", None)
        if isinstance(stats, dict):
            hits += int(stats.get("hits", 0))
    except Exception:  # noqa: BLE001 - measurement only
        pass
    return hits


def _record_peak_probe(
    state: Any,
    probe: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int,
    target: Any = None,
    workflow: Optional[str] = None,
) -> None:
    """Persist a completed render's peak above its starting allocation under the load's peak identity, keyed for
    ``workflow`` (``with_workflow``: text-to-image keeps the load's key), with its pre-decode / decode phases."""
    if probe is None or not width or not height:
        return
    identity = getattr(getattr(state, "pipe", None), "_unsloth_peak_identity", None)
    try:
        from .diffusion_measured_peaks import (
            PHASE_DECODE,
            PHASE_DECODE_UNPADDED,
            PHASE_PRE_DECODE,
            PHASE_UNPADDED,
            device_name,
            record,
        )

        peak = probe.finish()
        if peak is None or not isinstance(identity, tuple) or len(identity) != 3:
            return
        if getattr(probe, "decode_released", False):
            # the decode streamed lent groups out, so its peak is net of their bytes: a sample would under-size the
            # bucket (the placement that lent them already held a sample covering this decode)
            return
        if not _peak_identity_matches(state, identity):
            # the configuration moved during the render (a compile fallback, a toggled cache): no key names it
            return
        if cuda_graph.failed_capture_count() != getattr(probe, "failed_captures", 0):
            # a capture failed (an absorbed OOM): the peak is the failed attempt's, not the render's need
            return
        if _conditioning_cache_hits(state.pipe) != getattr(probe, "conditioning_hits", 0):
            # a cached prompt skipped the text encoder: the peak misses that phase, and an unseeded family places
            # its next load from this value alone
            return
        engaged = getattr(state, "attention_backend", None)
        if not _attention_scaling_known(target, engaged):
            # the SDPA probe could not tell math from a fused kernel: a quadratic peak stored as linear would
            # under-size every larger request
            return
        family, quant, speed_mode = _live_peak_identity(
            state
        )  # the key this render ran (``cfg:off`` included)
        pipe = state.pipe
        index = getattr(probe.device, "index", None)
        graphs = cuda_graph.live_graph_count(index)
        generation = cuda_graph.capture_generation(index)
        pad = int(getattr(probe, "window_pad", 0) or 0)
        peak += pad
        split = getattr(probe, "phase_peaks", None) or {}
        phases = {}
        if PHASE_PRE_DECODE in split and PHASE_DECODE in split:
            pre, dec = int(split[PHASE_PRE_DECODE]), int(split[PHASE_DECODE])
            phases = {
                PHASE_PRE_DECODE: pre + pad,
                PHASE_DECODE: dec + pad,
                PHASE_DECODE_UNPADDED: dec,
                PHASE_UNPADDED: max(pre, dec),
            }
        quant = with_workflow(quant, _peak_workflow(getattr(state, "family", None), workflow))
        held: list = []
        raised = record(
            family = family,
            quant = quant,
            speed_mode = speed_mode,
            width = width,
            height = height,
            batch_size = batch_size,
            peak_mib = peak,
            device_name = device_name(probe.device),
            quadratic = _quadratic_attention(target, engaged, state.pipe),
            held = held,
            phases = phases,
        )
        if held:
            # the store holds this sample, so it covers the graphs the render recorded; unpersisted, they stay dirty
            pipe._unsloth_peak_graphs = graphs
            pipe._unsloth_peak_graph_gen = generation
        if raised:
            logger.info(
                "diffusion.memory: learned activation peak %d MiB recorded for %s / %s / %s at %dx%d x%d",
                peak,
                family,
                quant,
                speed_mode,
                int(width),
                int(height),
                int(batch_size),
            )
    except Exception as exc:  # noqa: BLE001 - recording must never fail a finished render
        logger.debug("diffusion.memory: learned activation peak not recorded (%s)", exc)


def _activation_guard_batch(chunks: Sequence[Sequence[Any]]) -> int:
    """One image, as OOM backoff covers bigger batches; Windows WDDM never raises, so budget the largest."""
    if sys.platform != "win32":
        return 1
    return max((len(chunk) for chunk in chunks), default = 1)


def _inplace_torchao_placement(
    plan: Any,
    scheme: Optional[str],
    estimate: Any,
    largest_companion_mib: Callable[[], Optional[int]],
) -> tuple[Any, Optional[str]]:
    """(placement, None) for the in-place torchao quant, or (``plan``, why not). Model offload checks LOADED encoders."""
    if plan_keeps_transformer_resident(plan):
        return plan, None
    if plan.offload_policy == OFFLOAD_MODEL:
        estimates = getattr(plan, "estimates", None) or {}
        budget = int(estimates.get("safe_device_budget_mib") or 0)
        overhead = int(estimates.get("runtime_headroom_mib") or 0) + int(
            estimates.get("base_overhead_mib") or 0
        )
        if (
            estimate is not None
            and estimate.steady_transformer_mib + overhead <= budget
            and (largest_companion_mib() or 0) <= budget
        ):
            return plan, None
        if torchao_scheme_streams(scheme) and _torchao_stream_pinnable(plan):
            return torchao_streaming_plan(plan), None
        return plan, (
            "whole-module offload onloads each component whole, and the quantised transformer or a text "
            "encoder is not known to fit the device budget"
        )
    placed = torchao_offload_plan(plan, scheme)
    if placed is not None:
        return placed, None
    return plan, (
        f"'{plan.offload_policy}' offload streams the transformer through hooks torchao weights do not survive "
        f"({scheme} on this torchao). Pin low_vram or a resident memory mode to combine the two"
    )


def _denoiser_hooked(pipe: Any) -> bool:
    """Whether an offload hook moves a denoiser of ``pipe``; a CUDA graph over it would replay stale pointers."""
    from .diffusion_autoopt.targets import hooked_denoisers

    for module in hooked_denoisers(pipe):
        if getattr(module, "_hf_hook", None) is not None:
            return True
        registry = getattr(module, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None) or {}
        if any("offload" in str(key) for key in hooks):
            return True
    return False


def _torchao_render_needs_no_grad(state: Any) -> bool:
    """inference_mode rejects moving torchao weights between devices, so offloaded quant renders use no_grad.
    Stored fp8 codes too: moved under inference_mode they become unversionable, re-copied to host every offload."""
    if getattr(state, "offload_policy", OFFLOAD_NONE) == OFFLOAD_NONE:
        return False
    return bool(
        getattr(state, "transformer_quant", None)
        or getattr(state, "text_encoder_quant", None)
        or pipe_fp8_stored_layers(getattr(state, "pipe", None))
    )


def _memory_request_forces_offload(memory_mode: Optional[str], cpu_offload: bool) -> bool:
    """Balanced and low_vram force offload; cpu_offload only if no mode is set; fast/auto are measured."""
    mode = normalize_memory_mode(memory_mode)
    if mode in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM):
        return True
    return mode is None and bool(cpu_offload)


def _pipeline_quant_uncompilable_reason(
    target: Any,
    fam: Any,
    speed_mode: Optional[str],
    *,
    model_kind: str,
    probe: bool = False,
) -> Optional[str]:
    """Why a PIPELINE load must keep its dense weights rather than quantise them, or None.

    An uncompiled torchao transformer is ~30x slower than the bf16 it replaced, so converting
    without a compile is a pessimisation. Both callers ask here: the route preflight, so the refusal
    lands before the arbiter evicts anything, and the loader, so an automatic request declines to
    bf16. GGUF is out of scope, substituting dense base weights and falling back to the packed file.

    speed=off is absent on purpose: an engaged quant upgrades it to `default`, and an AUTO request
    under it is rewritten to off long before either caller.

    ``probe``: only the loader, which owns the evicted card, runs the allocating ROCm compile probe."""
    if model_kind != "pipeline":
        return None
    # A string compare, not resolve_speed_mode: this runs on the route, where an unvalidated value must not raise.
    if str(speed_mode or "").strip().lower() == SPEED_EAGER:
        return (
            "Speed is set to 'eager', and a quantised transformer that is not compiled runs far "
            "slower than the bf16 weights it replaces. Pick a compiling speed mode to combine the two"
        )
    if not compile_eligible(target, is_gguf = False, family = fam, probe = probe):
        return (
            "this process cannot run a torch.compile (no Triton, TORCHDYNAMO_DISABLE, or a "
            "family/device that does not compile), and a quantised transformer that is not "
            "compiled runs far slower than the bf16 weights it replaces"
        )
    return None


def _compile_cache_wanted(
    pipe: Any, target: Any, fam: Any, *, is_gguf: bool, speed_mode: str, offload_active: bool
) -> bool:
    """``compile_eligible`` for the compile-cache gate; on a dense ROCm load the allocating probe runs only once the
    plan finds something in the model to compile."""
    if is_gguf or getattr(target, "backend", None) != "rocm":
        return compile_eligible(target, is_gguf = is_gguf, family = fam)
    if not compile_eligible(target, is_gguf = False, family = fam, probe = False):
        return False
    try:
        from .diffusion_autoopt import plan as plan_mod
        step = plan_mod.build_plan(
            pipe,
            target,
            family = fam,
            backend = "image",
            speed_mode = speed_mode,
            offload_active = offload_active,
        ).step("compiled")
    except Exception:  # noqa: BLE001 - no plan: the probing predicate decides as before
        return compile_eligible(target, is_gguf = False, family = fam)
    return bool(step is not None and step.attempt)


def _plan_proves_resident(plan: Any) -> bool:
    """Resident AND fits; the planner also stays resident when it cannot read the card."""
    estimates = getattr(plan, "estimates", None) or {}
    # An explicit fast proves its fit against the budget it was placed with.
    budget = estimates.get("resident_budget_mib", estimates.get("safe_device_budget_mib"))
    required = estimates.get("resident_required_mib")
    # Unified memory plans 'none' whatever the size (offload frees nothing there), so compare the two as well.
    return (
        getattr(plan, "offload_policy", None) == OFFLOAD_NONE
        and budget is not None
        and required is not None
        and int(required) <= int(budget)
    )


def _auto_keeps_bf16_reason(
    fam: Any,
    kind: Optional[str] = "pipeline",
    base: Optional[str] = None,
) -> Optional[str]:
    """Why AUTO keeps bf16 when it fits (measured rule: pipeline loads only), or None; both deciders read this.
    ``base`` is the resolved base repo: the small-DiT rule skips a family's large variants (FLUX.2 klein 9B)."""
    if not family_compiles_regionally(fam):
        return (
            f"'{getattr(fam, 'name', None)}' cannot be regionally compiled (its transformer declares no "
            "repeated blocks), and an uncompiled quantised transformer runs far slower than the weights "
            "it replaces; they fit on this GPU as they are"
        )
    if kind != "pipeline":
        return None
    measured = auto_bf16_when_resident_reason(getattr(fam, "name", None))
    if measured is None:
        return None
    if measured == SMALL_DIT_BF16_WHEN_RESIDENT and small_dit_rule_excludes(base):
        return None
    return (
        f"'{getattr(fam, 'name', None)}': {measured}; the bf16 weights fit on this GPU as they are"
    )


def _auto_quant_eager_reason(
    fam: Any,
    plan: Any,
    prequant_path: Optional[str] = None,
    kind: Optional[str] = "pipeline",
    base: Optional[str] = None,
) -> Optional[str]:
    """Why AUTO keeps bf16 for this load, or None; an operator's own checkpoint wins."""
    if not _plan_proves_resident(plan):
        return None
    reason = _auto_keeps_bf16_reason(fam, kind, base)
    if reason is None:
        return None
    if prequant_path and local_prequant_path_ready(prequant_path):
        return None
    return reason


def _uninstall_fused_dit_patches() -> None:
    """Restore the process-global fused DiT patches so the next load honours its own kill switches. The group is the
    registry's (diffusion_autoopt.patches): every spec registered under it is restored, no hand-kept list."""
    try:
        from .diffusion_autoopt.patches import GROUP_FUSED_DIT, uninstall_group
        uninstall_group(GROUP_FUSED_DIT)
    except Exception:  # noqa: BLE001 - teardown is best effort
        pass


def _uninstall_eager_patches() -> None:
    """Restore the shared compile-safe and per-arch fusions (the registry's eager group)."""
    try:
        from .diffusion_autoopt.patches import GROUP_EAGER, uninstall_group
        uninstall_group(GROUP_EAGER)
    except Exception:  # noqa: BLE001 - teardown is best effort
        pass


def _clear_exception_frames(exc: BaseException) -> None:
    """Release failed-call locals while preserving traceback locations."""
    errors, seen = [exc], set()
    while errors:
        error = errors.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        errors.extend(cause for cause in (error.__cause__, error.__context__) if cause is not None)


def _dense_fast_path_reason(
    fam: Any,
    scheme: Optional[str],
    base: Optional[str],
    kind: str,
    path_override: Optional[str],
    loras: Any = None,
    failure: Optional[str] = None,
) -> str:
    """Names an unreadable hosted prequant only when one was tried; only the load knows why it failed."""
    note = (
        prequant_unreadable_reason(fam, scheme, base_repo = base)
        if kind == "pipeline" and not path_override and not _has_active_lora(loras)
        else None
    )
    if note:
        return f"engaged on the dense fast path; {note}, so the dense bf16 transformer was quantized instead"
    if failure:
        return (
            f"engaged on the dense fast path; the pre-quantized {scheme} checkpoint did not load "
            f"({failure}), so the dense bf16 transformer was quantized instead"
        )
    return "engaged on the dense fast path"


def _progress_latch_enabled() -> bool:
    """UNSLOTH_DIFFUSION_PROGRESS_LATCH=0 rescans the cache on every load-progress poll, as before."""
    return os.environ.get("UNSLOTH_DIFFUSION_PROGRESS_LATCH", "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _bg_compile_module(
    pipe: Any,
    *,
    speed_optims: Any,
    speed_mode: Any,
    transformer_quant: Any,
    gguf_transformer: bool,
    offload_policy: Any,
    transformer_cache: Any,
    cache_auto: bool,
    target: Any,
) -> Any:
    """The denoiser whose compile may run in the background (``diffusion_bg_compile.select_module``), or None."""
    return bg_compile.select_module(
        pipe,
        speed_optims = speed_optims,
        default_tier = speed_mode == SPEED_DEFAULT,
        quantized = transformer_quant is not None,
        gguf = bool(gguf_transformer),
        step_cache = bool(transformer_cache) or bool(cache_auto),
        device = getattr(target, "device", None),
        backend = getattr(target, "backend", None),
        denoiser_hooked = offload_policy != OFFLOAD_NONE and _denoiser_hooked(pipe),
    )


def _prequant_bakes_auto_accumulate(source: Any) -> bool:
    """Hosted repos and local cache entries bake the auto fp8 accumulate (forced-mode builds are never cached)."""
    kind = getattr(source, "kind", None)
    if kind == "repo":
        return True
    if kind != "path":
        return False
    try:
        from .diffusion_prequant_cache import path_in_cache
        return path_in_cache(getattr(source, "location", None) or "")
    except Exception:  # noqa: BLE001
        return False


def _single_file_fp8_upcast(single_file_path: Any, fam: Any) -> bool:
    """Whether a single-file denoiser stores fp8 weights a widened load upcasts (~2x resident). Read from the header
    (the basename only when it is unreadable); SDXL single-file pipelines are already bf16."""
    if not single_file_path or getattr(fam, "single_file_is_pipeline", False):
        return False
    stored = header_has_fp8_weights(str(single_file_path))
    return "fp8" in Path(single_file_path).name.lower() if stored is None else stored


def _comfy_keep_fp32_pricing(fam: Any, target: Any) -> dict:
    """``comfy_resident_mib`` kwargs for the DiT's ``_keep_in_fp32_modules``: never stored, fp32 on an fp16 load."""
    import torch

    from .diffusion_comfy_quant import keep_fp32_modules_for

    keep = keep_fp32_modules_for(fam)
    if not keep:
        return {}
    fp16 = getattr(target, "dtype", None) is torch.float16
    return {"keep_fp32": keep, "keep_fp32_bytes": 4 if fp16 else None}


def _image_cast_candidates(
    fam: Any,
    source: Any,
    *,
    dtype: Any,
    base: Optional[str],
    skip: tuple = (),
) -> list:
    """Generic diffusers assembly only (per-component assemblers are not wired). Never raises."""
    try:
        from .diffusion_cast_cache import candidates, image_enabled

        if not image_enabled() or not source:
            return []
        if is_flux3_family(fam) or getattr(fam, "name", None) in (
            MOSS_IMAGE_FAMILY_NAME,
            KREA2_FAMILY_NAME,
            IDEOGRAM4_FAMILY_NAME,
        ):
            return []
        return candidates(
            source,
            torch_dtype = dtype,
            cache_dir = None if os.path.isdir(str(source)) else hub_cache_dir(),
            skip = skip,
            base_repo = base,
            logger = logger,
        )
    except Exception as exc:  # noqa: BLE001 - a cache problem is a miss
        logger.warning("diffusion.cast_cache: skipped (%s)", exc)
        return []


def _image_cast_cached(cands: list, variant: Optional[str]) -> list:
    if not cands or variant is not None:
        return []
    from .diffusion_cast_cache import cached_components
    return cached_components(cands)


def _converted_shards_pread(dtype: Any, small_host: Any) -> Any:
    """Windows: transformers components converted to ``dtype`` read through ``pread`` (no copy-on-write commit for the
    stored bytes, #14513); components the small-host route loads at their stored dtype stay memory-mapped."""
    from .safetensors_pread import pread_safetensors

    keep = tuple(small_host.storage_dtypes) if small_host is not None and small_host.engaged else ()
    return pread_safetensors(convert_to = dtype, keep = keep)


class DiffusionBackend:
    """Holds at most one loaded diffusers pipeline. All mutations are serialised."""

    def __init__(self) -> None:
        # _lock serialises pipeline mutations; the status/progress readers stay lock-free.
        self._lock = threading.Lock()
        # Guards load metadata without waiting on construction. Never acquire pipeline locks while holding this lock.
        self._load_cancel_lock = threading.Lock()
        self._unload_waiters = 0
        # _generate_lock serialises generations and is the ONLY lock the denoise holds.
        self._generate_lock = threading.Lock()
        self._state: Optional[_LoadState] = None
        self._loading: Optional[_LoadingState] = None
        # Bumped on begin_load/unload so a superseded worker neither commits nor stamps progress
        self._load_token = 0
        # Includes preflight and direct loads that have no background marker.
        self._load_accounts: dict[object, tuple[int, str]] = {}
        # Replaced per load so cancelled workers stay cancelled.
        self._cancel_event = threading.Event()
        # Keep Stop responsive while a replacement holds _lock.
        self._generation_cancel_lock = threading.Lock()
        # Cancel Event of the in-flight generation; per-generation so a cancel can't be lost or leak
        self._active_generate_cancel: Optional[threading.Event] = None
        # Bound with the event under the same lock, so a cancel cannot land on the wrong generation.
        self._active_generate_account: Optional[str] = None
        # Queued requests; cancel_generate() decides which Stop may signal.
        self._queued_generate_cancels: set[threading.Event] = set()
        self._generation_owns_slot = False

        self._transition_owns_slot = False
        # Teardowns waiting to free this pipeline; a count supports concurrent reservations.
        self._teardown_waiters = 0
        # Monotonic count of reserved teardowns; lets a render detect one ran under it.
        self._teardown_epoch = 0
        # Set when no teardown is reserved; an Event keeps waiting independent of _lock.
        self._teardown_drained = threading.Event()
        self._teardown_drained.set()
        # Repos a cancelled load is still reading, per epoch; see draining_repo_ids().
        self._draining_repos: dict[int, set[str]] = {}
        # Separate event: _teardown_drained stays SET until construction releases _lock, so waiting on it spun.
        self._unload_fence_clear = threading.Event()
        self._unload_fence_clear.set()
        # Written by the callback, read lock-free by generate_progress().
        self._gen: Optional[_GenState] = None
        # img2img/inpaint pipes built via from_pipe (shared modules, no extra VRAM); cleared on unload
        self._aux_pipes: dict[str, Any] = {}
        # Loaded ControlNets and their from_pipe pipelines, reusing resident modules
        self._cn_models: dict[str, Any] = {}
        self._cn_pipes: dict[tuple[str, str], Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    def _pick_device_and_dtype(self, ordinal: Optional[int] = None) -> tuple[str, Any]:
        """(device, dtype) for the current host. Thin wrapper over the device policy module, kept as
        a method so tests can still monkeypatch it."""
        if ordinal is None:
            target = resolve_diffusion_device_target()
            return target.device, target.dtype
        target = resolve_diffusion_device_target(ordinal = ordinal)
        # The INDEXED string, so _resolve_device_target can rebuild a selection an override would erase.
        return target.torch_device, target.dtype

    def _stop_load_prefetch(self) -> None:
        """End a load's page-cache prefetch (diffusion_fast_load.start_load_prefetch), if one is running."""
        stop_prefetch(getattr(self, "_load_prefetch", None))
        self._load_prefetch = None

    def _raise_if_load_cancelled(self, token: int) -> None:
        # Token only: unload() bumps _load_token first; a bare _unload_waiters check would 409 model switches, which must wait instead.
        if token != self._load_token:
            raise RuntimeError("Diffusion load was cancelled.")

    def _wait_for_pending_unloads(self, timeout: float = 900.0) -> None:
        """Fresh loads (no epoch of their own) queue behind pending ejects, including failed ones; holds no lock, so no deadlock."""
        deadline = time.monotonic() + timeout
        while True:
            with self._load_cancel_lock:
                if not self._unload_waiters:
                    return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for a diffusion unload to finish.")
            # A sleep with an early wake, not the condition itself: the counter above decides.
            self._unload_fence_clear.wait(timeout = 0.1)

    def _reserve_teardown_locked(self) -> None:
        """Fence queued generations off the pipeline this teardown is about to free. Call only while
        holding ``_lock``, so the count and the gate move together."""
        self._teardown_waiters += 1
        self._teardown_epoch += 1
        self._teardown_drained.clear()

    def _release_teardown_locked(self) -> None:
        """Release one teardown reservation and wake generations when the last one leaves. Call only
        while holding ``_lock``. A count is necessary because an unload and a superseding load
        can both be queued behind the active generation."""
        assert self._teardown_waiters > 0, "teardown reservation released without an owner"
        self._teardown_waiters -= 1
        if self._teardown_waiters == 0:
            self._teardown_drained.set()

    @contextmanager
    def _model_transition_slot(self):
        """Sets the owner flag under the cancel lock before the teardown fence drops, so a queued cancel sees the reservation or owner."""
        self._generate_lock.acquire()
        with self._generation_cancel_lock:
            assert not self._transition_owns_slot, "two model transitions own one slot"
            self._transition_owns_slot = True
        try:
            yield
        finally:
            with self._generation_cancel_lock:
                self._transition_owns_slot = False
                self._generate_lock.release()

    @contextmanager
    def _generation_slot(self, cancel: threading.Event):
        """Yields to teardown fences after taking the lock (not FIFO); the zero-fence check and cancel registration must share one _lock section."""
        admitted = False
        with self._generation_cancel_lock:
            self._queued_generate_cancels.add(cancel)
        try:
            while True:
                while True:
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    if self._generate_lock.acquire(timeout = 0.1):
                        break
                with self._lock:
                    if not self._teardown_waiters:
                        with self._generation_cancel_lock:
                            cancelled = cancel.is_set()
                            if not cancelled and not self._unload_waiters:
                                self._queued_generate_cancels.discard(cancel)
                                self._active_generate_cancel = cancel
                                self._active_generate_account = current_account_id()
                                self._generation_owns_slot = True
                                admitted = True
                    else:
                        cancelled = cancel.is_set()
                if admitted:
                    break
                self._generate_lock.release()
                if cancelled:
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                while True:
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    # Not _teardown_drained: it is still SET before the teardown is reserved, so waiting on it spins.
                    with self._load_cancel_lock:
                        fenced_by_unload = bool(self._unload_waiters)
                    gate = self._unload_fence_clear if fenced_by_unload else self._teardown_drained
                    if gate.wait(timeout = 0.1):
                        break
            from hub.services.models.account_access import media_generation_slot
            try:
                with media_generation_slot("diffusion"):
                    yield
            finally:
                with self._generation_cancel_lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None
                    self._generation_owns_slot = False
                    self._generate_lock.release()
        finally:
            if not admitted:
                with self._generation_cancel_lock:
                    self._queued_generate_cancels.discard(cancel)

    # Offload policies fixed by the request; `fast`/`auto` are measurements and cannot be judged network-free.
    def assert_precision_available(
        self,
        fam: Optional[DiffusionFamily],
        *,
        model_kind: str,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        memory_mode: Optional[str] = None,
        cpu_offload: bool = False,
        speed_mode: Optional[str] = None,
        gpu_ordinal: Optional[int] = None,
        repo_id: Optional[str] = None,
        base_repo: Optional[str] = None,
        local_files_only: bool = False,
        text_encoder_files: Optional[Sequence[str]] = None,
        gguf_filename: Optional[str] = None,
    ) -> None:
        """Explicit precision only: RuntimeError (409) for host-level impossibilities; auto is never refused; the route calls this before taking the GPU."""
        from .diffusion_nvfp4_flag import refuse_disabled_nvfp4

        refuse_disabled_nvfp4(
            transformer_quant = transformer_quant, text_encoder_quant = text_encoder_quant
        )
        if precision_fallback_allowed():
            return
        pinned = normalize_transformer_quant(transformer_quant)
        te_mode = normalize_te_quant(text_encoder_quant)
        if pinned is None and te_mode is None:
            return
        # One probe for both checks, asked of the card THIS load will use: on a mixed box the default device can be a
        # different generation, which would refuse a scheme the selected card supports, or pass one it does not and
        # fail only after eviction. Scoped, because the route calls this on a pooled thread and the probes below use
        # argument-less CUDA calls.
        if model_kind != "pipeline" and not (base_repo or "").strip() and fam is not None:
            # The variant base the loader derives from the pick's name (a FLUX.2 Klein 9B checkpoint), network-free:
            # judged on the family default, the int8 gate would check another variant's encoder file.
            base_repo = named_variant_base(fam, repo_id, gguf_filename) or base_repo
        with diffusion_device_scope(gpu_ordinal):
            target = self._target_for_ordinal(fam, gpu_ordinal)
            return self._assert_precision_for_target(
                fam,
                target,
                model_kind = model_kind,
                pinned = pinned,
                te_mode = te_mode,
                memory_mode = memory_mode,
                cpu_offload = cpu_offload,
                speed_mode = speed_mode,
                base_repo = repo_id if model_kind == "pipeline" else base_repo,
                local_files_only = local_files_only,
                supplied_text_encoders = self._planned_supplied_text_encoders(
                    fam,
                    repo_id,
                    repo_id if model_kind == "pipeline" else base_repo,
                    text_encoder_files,
                    te_mode,
                ),
            )

    def _planned_supplied_text_encoders(
        self,
        fam: Optional[DiffusionFamily],
        repo_id: Optional[str],
        base_repo: Optional[str],
        text_encoder_files: Optional[Sequence[str]],
        te_mode: Optional[str],
    ) -> tuple[str, ...]:
        """The encoder slots the supplied files fill, for the preflight: it runs before the load installs the
        component overrides. Network-free (local files, a cached model index); () when that cannot tell, and the
        load's own check then refuses a mixed build."""
        if not text_encoder_files or fam is None or te_mode != "int8":
            return ()
        try:
            from .diffusion_comfy_components import parse_component_file

            refs = [
                parse_component_file(
                    spec, model_path = repo_id, trusted_repo = _is_trusted_diffusion_repo
                )
                for spec in text_encoder_files
            ]
            if any(not ref.local_path for ref in refs):
                return ()
            overrides = self._plan_component_overrides(
                fam,
                repo_id or "",
                base_repo or getattr(fam, "base_repo", None) or repo_id or "",
                list(text_encoder_files),
                None,
                None,
                local_files_only = True,
                download = False,
            )
            return tuple(overrides.text_encoder_components) if overrides is not None else ()
        except Exception:  # noqa: BLE001 -- unplannable here: the load refuses it
            return ()

    def _assert_precision_for_target(
        self,
        fam: Optional[DiffusionFamily],
        target: DiffusionDeviceTarget,
        *,
        model_kind: str,
        pinned: Optional[str],
        te_mode: Optional[str],
        memory_mode: Optional[str],
        cpu_offload: bool,
        speed_mode: Optional[str] = None,
        base_repo: Optional[str] = None,
        local_files_only: bool = False,
        supplied_text_encoders: tuple[str, ...] = (),
    ) -> None:
        """The body of ``assert_precision_available``, run with the selected card current."""
        if pinned is not None and pinned != TQ_AUTO:
            reason = None
            family_name = getattr(fam, "name", None)
            if (
                not base_repo
                and model_kind != "pipeline"
                and pinned == TQ_NVFP4
                and family_denies_scheme(family_name, pinned)
            ):
                # GGUF pick without a base: judge network-free on a gate-lifted base; the loader refuses any other.
                base_repo = nvfp4_gated_base(family_name)
            if not dense_quant_supported_kind(model_kind):
                reason = dense_quant_unsupported_kind_reason(model_kind)
            elif getattr(fam, "denoiser_attr", "transformer") != "transformer":
                # Refuse from the family table, before eviction and download.
                reason = (
                    f"'{getattr(fam, 'name', None)}' denoises with a "
                    f"{getattr(fam, 'denoiser_attr', 'unet')}, not a transformer, and the dense torchao schemes "
                    "do not cover it"
                )
            elif (
                model_kind == "pipeline"
                and pinned in NATIVE_QUANT_SCHEMES
                and native_quant_host(target)
            ):
                if native_quant_scheme(target, pinned, family = getattr(fam, "name", None)) is None:
                    reason = explain_unusable_scheme(getattr(fam, "name", None), pinned)
            elif (
                model_kind == "pipeline"
                and pinned in NATIVE_OFFLOAD_SCHEMES
                and _memory_request_forces_offload(memory_mode, cpu_offload)
                and native_offload_host(target)
            ):
                if (
                    native_quant_scheme(
                        target, pinned, family = getattr(fam, "name", None), offload = True
                    )
                    is None
                ):
                    reason = explain_unusable_scheme(getattr(fam, "name", None), pinned)
            elif _memory_request_forces_offload(memory_mode, cpu_offload) and (
                model_kind != "pipeline"
                or normalize_memory_mode(memory_mode) == MEMORY_MODE_BALANCED
            ):
                # Request-named offload policies are known now; refusing later would land after the eviction.
                requested_memory = normalize_memory_mode(memory_mode) or "cpu_offload"
                reason = (
                    f"'{requested_memory}' memory places the transformer under CPU offload, and a "
                    "GGUF pick only swaps in the quantised dense transformer when it fits resident"
                    if model_kind != "pipeline"
                    else "'balanced' memory streams the transformer through group-offload hooks "
                    "torchao weights do not survive; use low_vram to combine the two"
                )
            elif not dense_transformer_supported(target):
                reason = dense_transformer_unsupported_reason(target)
            elif (
                uncompilable := _pipeline_quant_uncompilable_reason(
                    target, fam, speed_mode, model_kind = model_kind
                )
            ) is not None:
                # Deterministic from the request: else the loader refuses only after the eviction and download.
                reason = uncompilable
            elif (
                select_transformer_quant_scheme(
                    target,
                    pinned,
                    family = getattr(fam, "name", None),
                    base_repo = base_repo,
                    # Runs before eviction, so an OOM smoke probe only means the GPU is still full, not unusable.
                    unproven_ok = True,
                )
                is None
            ):
                # Explicit schemes are never swapped, so None means this GPU or deny list rules it out; pass the same base so a lifted gate is not blamed.
                reason = explain_unusable_scheme(getattr(fam, "name", None), pinned, base_repo)
            if reason is not None:
                raise RuntimeError(
                    precision_refusal_message(
                        "transformer_quant",
                        pinned,
                        reason,
                        off_label = "Off to run the checkpoint as-is",
                    )
                )
        # Check the mode the loader will attempt, not the raw request (explicit int8 without keep-bf16 becomes layerwise fp8), else valid loads are refused.
        te_effective = effective_te_quant(te_mode, getattr(fam, "name", None))
        if te_effective is not None and normalize_te_quant(te_mode) == "int8":
            # int8 depends on whether the hosted int8 ConvRot file will really load for this base and device.
            te_effective = preflight_te_quant(
                te_mode,
                fam,
                base_repo,
                target,
                supplied = supplied_text_encoders or _active_supplied_text_encoders(),
                local_files_only = local_files_only,
            )
        # preflight_te_quant judges the hosted int8 file itself (None when it will load, on Apple Silicon too), so a
        # file that is not held under a cache-only load stays refused.
        te_reason = None
        if te_quant_needs_resident_weights(te_effective) and not torchao_quantize_importable():
            # The casters import torchao only after download, so check the install here for a clean 409.
            te_reason = (
                "torchao is not importable on this install, and these encoder modes are torchao "
                "quantisations"
            )
        elif te_effective is not None and not te_quant_supported(target, te_effective):
            te_reason = te_quant_unsupported_reason(te_effective)
            if normalize_te_quant(te_mode) == "int8":
                from .diffusion_te_prequant import explicit_int8_unavailable_reason
                missing = explicit_int8_unavailable_reason(
                    fam,
                    base_repo,
                    target,
                    supplied = supplied_text_encoders or _active_supplied_text_encoders(),
                    local_files_only = local_files_only,
                )
                if missing:
                    te_reason = f"{missing}, and the fp8 fallback cannot run here ({te_reason})"
        elif te_quant_needs_resident_weights(te_effective) and _memory_request_forces_offload(
            memory_mode, cpu_offload
        ):
            # Offload hooks move modules with Module.to(), which torchao tensors do not survive; refuse before eviction. Layerwise fp8 is unaffected.
            requested_memory = normalize_memory_mode(memory_mode) or "cpu_offload"
            te_reason = (
                f"'{requested_memory}' memory places the text encoder under CPU offload, and "
                "torchao quantised tensors cannot be moved by the offload hooks"
            )
        if te_reason is not None:
            raise RuntimeError(
                precision_refusal_message(
                    "text_encoder_quant",
                    te_mode,
                    te_reason,
                    off_label = "leave it unset to keep the dense bf16 encoder",
                    auto_available = False,
                )
            )

    def _resolve_device_target(
        self,
        fam: Optional[DiffusionFamily],
        *,
        ordinal: Optional[int] = None,
    ) -> DiffusionDeviceTarget:
        """Promotes float16 to float32 for fp16-incompatible families, rebuilding the target so its flags match; runs after _pick_device_and_dtype."""
        device, dtype = (
            self._pick_device_and_dtype()
            if ordinal is None
            else self._pick_device_and_dtype(ordinal)
        )
        effective = _resolve_diffusion_compute_dtype(fam, dtype)
        if effective is not dtype:
            logger.warning(
                "diffusion.dtype_promoted: family=%s float16 -> float32 (fp16-incompatible)",
                getattr(fam, "name", None),
            )
        # Builds only; pinning is the caller's call (pooled threads must not leak a set_device).
        return diffusion_device_target_from_torch_device(device, effective)

    def _fp16_load_target(self, fam: Optional[DiffusionFamily], ordinal: Optional[int]) -> bool:
        """True when this pick loads at fp16 (a pre-Ampere card), so the plan may stage ``.fp16`` twins. Never raises."""
        if not fp16_variant_enabled():
            return False
        try:
            return is_fp16_dtype(getattr(self._target_for_ordinal(fam, ordinal), "dtype", None))
        except Exception:  # noqa: BLE001 - no target: the defaults always load
            return False

    def _target_for_ordinal(
        self, fam: Optional[DiffusionFamily], ordinal: Optional[int]
    ) -> DiffusionDeviceTarget:
        """``_resolve_device_target`` with the ordinal only when there is one, so a monkeypatched
        seam taking just ``fam`` keeps working on the automatic path."""
        if ordinal is None:
            return self._resolve_device_target(fam)
        return self._resolve_device_target(fam, ordinal = ordinal)

    def _state_device_target(self, state: _LoadState) -> DiffusionDeviceTarget:
        """Pins the calling thread to the resident GPU of the model; every worker must use this, since state.device is un-indexed."""
        target = self._target_for_ordinal(state.family, state.gpu_ordinal)
        apply_diffusion_device_ordinal(target)
        # And put an AUTOMATIC load back on its own card. generate() runs on a pooled asyncio.to_thread worker, so a
        # previous pinned model leaves that thread set to its GPU for good; with no ordinal to pin, the bare "cuda"
        # Generators below would then target that card while these weights sit on the default one. Skipped when the
        # pin above already set that card.
        if state.gpu_ordinal is None or state.placed_ordinal != state.gpu_ordinal:
            pin_cuda_ordinal(state.placed_ordinal)
        return target

    def _resolve_gguf_path(
        self,
        repo_id: str,
        gguf_filename: str,
        hf_token: Optional[str],
        local_files_only: bool = False,
    ) -> str:
        """Under local_files_only, a Hub re-resolve could start a multi-GB pull that unload cannot
        preempt."""
        local_root = Path(repo_id).expanduser()
        if local_root.exists():
            return str(resolve_local_gguf_child(local_root, gguf_filename))
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        # Pin the LIVE cache root, else a mid-session cache change re-pulls the GGUF inside the load lock.
        cache_dir = hub_cache_dir()
        offline_ok = local_files_only or hub_reachability.hub_unreachable()
        cached_here = False
        # Best-effort: an unreadable cache raises here, and letting that escape would abort a load that only had to
        # download. Any failure falls through to the pinned download below.
        try:
            cached_here = isinstance(
                try_to_load_from_cache(repo_id, gguf_filename, cache_dir = cache_dir), str
            )
            if not cached_here:
                # Same other-root reuse as the pre-quant checkpoint, reached THROUGH that root rather than returned
                # raw, so the ref still resolves and a republished GGUF is picked up; the blob is reused, and offline
                # the cached pointer comes back anyway.
                elsewhere = try_to_load_from_cache(repo_id, gguf_filename, cache_dir = None)
                if isinstance(elsewhere, str) and Path(elsewhere).is_file():
                    try:
                        return hf_hub_download(
                            repo_id,
                            gguf_filename,
                            token = hf_token,
                            cache_dir = None,
                            local_files_only = offline_ok,
                        )
                    except Exception:  # noqa: BLE001 - revalidation is a bonus, never a new failure
                        return elsewhere
        except Exception:  # noqa: BLE001 - an unreadable cache is not a verdict, just download
            pass
        return hf_hub_download(
            repo_id,
            gguf_filename,
            token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only or (cached_here and offline_ok),
        )

    def _dense_quant_prefetch_needed(
        self,
        fam: DiffusionFamily,
        kwargs: dict,
        *,
        companion_files: Optional[Sequence[str]] = None,
        transformer_files: Optional[Sequence[str]] = None,
    ) -> bool:
        """True when dense transformer shards would download under the load lock, so prefetch them; keep in step with the load_pipeline gates."""
        raw = kwargs.get("transformer_quant")
        auto = raw is None or str(raw).strip().lower() in ("", "auto")
        if auto:
            mode = TQ_AUTO
        else:
            mode = normalize_transformer_quant(raw)
        if mode is None:
            return False
        # An explicit Speed="off" load stays GGUF-as-is (dense path never runs); don't widen the prefetch.
        speed = kwargs.get("speed_mode")
        if speed is not None and str(speed).strip().lower() == SPEED_OFF:
            return False
        if transformer_variant_differs_from_base(
            fam,
            kwargs.get("base_repo"),
            kwargs.get("gguf_filename"),
            content_variant_hint(kwargs.get("repo_id"), kwargs.get("gguf_filename")),
            kwargs.get("repo_id"),
            kwargs.get("display_repo_id"),
        ):
            return False
        try:
            # A definite-offload policy skips the dense build, so widening would waste a multi-GB pull.
            mm = normalize_memory_mode(kwargs.get("memory_mode"))
            if mm in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM):
                return False
            if mm is None and kwargs.get("cpu_offload"):
                return False
            # Scoped to the target card: the selectors below read the current device.
            with diffusion_device_scope(kwargs.get("gpu_ordinal")):
                return self._dense_quant_prefetch_decision(
                    fam,
                    kwargs,
                    auto = auto,
                    mode = mode,
                    companion_files = companion_files,
                    transformer_files = transformer_files,
                )
        except Exception:  # noqa: BLE001 -- an unanswerable probe keeps the GGUF
            return False

    def _dense_quant_prefetch_decision(
        self,
        fam: DiffusionFamily,
        kwargs: dict,
        *,
        auto: bool,
        mode: str,
        companion_files: Optional[Sequence[str]] = None,
        transformer_files: Optional[Sequence[str]] = None,
    ) -> bool:
        """``_dense_quant_prefetch_needed``'s body, run with the selected card current."""
        target = self._target_for_ordinal(fam, kwargs.get("gpu_ordinal"))
        # Auto pick with an uncached hosted prequant keeps the GGUF, so skip base transformer/ shards; cached shards and explicit transformer_quant still stage.
        if (
            auto
            and not _has_active_lora(kwargs.get("loras"))
            and _uncached_prequant_repo(
                fam,
                target,
                mode,
                base_repo = kwargs.get("base_repo"),
                prequant_path = kwargs.get("transformer_prequant_path"),
            )
            is not None
        ):
            return False
        if (
            auto
            and not _has_active_lora(kwargs.get("loras"))
            and not _dense_transformer_cached(
                kwargs.get("base_repo"),
                companion_files = companion_files,
                transformer_files = transformer_files,
            )
        ):
            return False
        # Only widen when the loader would take the dense path; same candidate load_pipeline re-plans against.
        candidate = resolve_dense_quant_candidate(
            fam = fam,
            target = target,
            requested = mode,
            base_repo = kwargs.get("base_repo"),
            prequant_path = kwargs.get("transformer_prequant_path"),
            # An all-zero list bakes nothing; sizing it dense stages shards the load never reads.
            force_dense = _has_active_lora(kwargs.get("loras")),
            logger = None,
        )
        if candidate is None:
            return False
        if candidate.prequant:
            scheme = getattr(candidate, "scheme", None)
            if self._hosted_prequant_reachable(fam, scheme, kwargs):
                return False
            kwargs["_prequant_unreachable"] = tuple(
                dict.fromkeys((*kwargs.get("_prequant_unreachable", ()), scheme))
            )
        # Mirror plan_fits_total_capacity, else load_pipeline declines the dense path anyway.
        from .diffusion_memory import (
            snapshot_device_memory,
            total_capacity_budget_mib,
        )

        budget = total_capacity_budget_mib(snapshot_device_memory(target))
        steady = getattr(candidate, "steady_total_mib", None)
        if budget is not None and steady is not None and int(steady) > budget:
            return False
        return True

    def _hosted_prequant_reachable(self, fam: Any, scheme: Optional[str], kwargs: dict) -> bool:
        """Family table alone calls private or gated prequant repos available; a Hub refusal makes it unavailable unless cached or local."""
        try:
            source = usable_prequant_source(
                fam,
                scheme,
                path_override = kwargs.get("transformer_prequant_path"),
                base_repo = kwargs.get("base_repo"),
                local_files_only = bool(kwargs.get("local_files_only")),
                hf_token = kwargs.get("hf_token"),
                cache_dir = hub_cache_dir(),
            )
            if source is None:
                # With the load's token the loader would refuse it, so stage dense; only if a source resolves at all.
                return (
                    resolve_prequant_source(
                        fam,
                        scheme,
                        path_override = kwargs.get("transformer_prequant_path"),
                        base_repo = kwargs.get("base_repo"),
                    )
                    is None
                )
            if getattr(source, "kind", None) != "repo":
                return True
            if prequant_checkpoint_cached(
                source,
                cache_dir = hub_cache_dir(),
                online = False if kwargs.get("local_files_only") else None,
            ):
                return True
            if kwargs.get("local_files_only"):
                return False
            return (
                self._prequant_source_hub_entry(source, kwargs.get("hf_token"), scheme = scheme)
                is not None
            )
        except Exception:  # noqa: BLE001 -- a transient probe failure keeps the candidate's verdict
            return True

    @staticmethod
    def _auto_prequant_retry_scheme(
        target: Any,
        fam: Any,
        requested: Optional[str],
        chosen: Optional[str],
        *,
        base_repo: Optional[str],
        path_override: Optional[str],
        loras: Any,
    ) -> Optional[str]:
        """A lower auto rung than ``chosen`` that HAS a usable prequant, or None. Only when ``auto``
        was requested: an explicit scheme is honored or refused, never swapped. A baked LoRA
        takes the dense path regardless, so it gets no retry."""
        if normalize_transformer_quant(requested) != TQ_AUTO or _has_active_lora(loras):
            return None
        try:
            from .diffusion_transformer_quant import auto_scheme_candidates
            candidates = auto_scheme_candidates(
                target,
                getattr(fam, "name", None),
                base_repo = base_repo,
                has_prequant = lambda candidate: usable_prequant_source(
                    fam, candidate, path_override = path_override, base_repo = base_repo
                )
                is not None,
            )
        except Exception:  # noqa: BLE001 -- no candidates is just "no retry"
            return None
        seen_chosen = False
        for candidate in candidates:
            if candidate == chosen:
                seen_chosen = True
                continue
            # Strictly BELOW the winner: a higher rung was already rejected by the ladder.
            if not seen_chosen:
                continue
            source = usable_prequant_source(
                fam, candidate, path_override = path_override, base_repo = base_repo
            )
            if source is None or _is_local_cache_entry(source):
                # Lower rungs never come from the local cache: one earlier explicit load must not change auto's pick.
                continue
            # Retries must be cached too: _uncached_prequant_repo only guards the winner.
            if source.kind == "repo" and not prequant_checkpoint_cached(
                source, cache_dir = hub_cache_dir()
            ):
                continue
            return candidate
        return None

    def _restore_skipped_transformer_shards(
        self,
        repo_id: str,
        base: str,
        skipped: Sequence[str],
        hf_token: Optional[str],
        *,
        fetch_base: Optional[str] = None,
        base_local_dir: Optional[str] = None,
        local_files_only: bool = False,
        manifest: str = "model_index.json",
    ) -> Optional[str]:
        """Stage the shards a seeded plan left out of the pull; returns the snapshot dir to assemble
        from, or None for the hub id. ``from_pretrained`` treats a local dir as terminal, so a
        failed seed has nothing. ``manifest`` is the base's pipeline index (FLUX.3's is the modular one), fetched
        with the shards so the top-up resolves the same snapshot."""
        files = [name for name in (skipped or ()) if name]
        if not files:
            return base_local_dir
        logger.warning(
            "diffusion.denoiser_prequant: restoring the %d released denoiser shard(s) the "
            "pre-download skipped",
            len(files),
        )
        restored = self._prefetch_files(
            repo_id,
            None,
            base,
            [*files, manifest],
            hf_token,
            fetch_base = fetch_base,
            local_files_only = local_files_only,
        )
        # Accept only a snapshot both staging and this top-up resolved; split roots fall back to the hub id.
        if restored is None or base_local_dir is None or Path(restored) != Path(base_local_dir):
            return None
        return base_local_dir

    def _fetch_denoiser_prequant(
        self,
        entry: Optional[tuple[str, str, int]],
        hf_token: Optional[str],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Stage the seeded denoiser, best-effort; done here so an unload can preempt the fetch."""
        if entry is None:
            return
        repo, filename, _size = entry
        try:
            from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
            hf_hub_download_with_xet_fallback(
                repo,
                filename,
                hf_token,
                cancel_event = cancel_event if cancel_event is not None else self._cancel_event,
                reuse_other_cache_root = True,
            )
        except Exception as exc:  # noqa: BLE001 -- the loader re-resolves and falls back to dense
            if isinstance(exc, RuntimeError) and str(exc) == "Cancelled":
                raise
            logger.warning(
                "diffusion.denoiser_prequant_fetch_failed: %s/%s: %s", repo, filename, exc
            )

    def _prefetch_files(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        base_files: list[str],
        hf_token: Optional[str],
        cancel_event: Optional[threading.Event] = None,
        fetch_base: Optional[str] = None,
        local_files_only: bool = False,
    ) -> Optional[str]:
        """Downloads without the lock, honouring the cancel_event of this load; local_files_only must make every resolution a cache lookup."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # Only bytes move; caller keeps the upstream id. Reuse fetch_base when set, so every later fetch agrees with what was staged.
        base = fetch_base or prefer_ungated_mirror(base, hf_token, files = base_files)
        # Callers without a per-load event (tests, direct use) fall back to the current one.
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        # Resolve via the import-time root too, else a cache-folder change turns a cached gated base into
        # a Hub token error.
        if gguf_filename and not Path(repo_id).expanduser().exists():
            hf_hub_download_with_xet_fallback(
                repo_id,
                gguf_filename,
                hf_token,
                cancel_event = cancel,
                reuse_other_cache_root = True,
                local_files_only = local_files_only,
                gguf_header_delta = True,
            )
        # Base repo (VAE / text-encoder / scheduler); list comes from the estimate.
        snapshot_root: Optional[str] = None
        # Files may resolve through different roots; only return a snapshot that holds all of them.
        roots: set[str] = set()
        for rfilename in base_files:
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            local = hf_hub_download_with_xet_fallback(
                base,
                rfilename,
                hf_token,
                cancel_event = cancel,
                reuse_other_cache_root = True,
                local_files_only = local_files_only,
            )
            try:
                root = str(Path(local).parents[len(Path(rfilename).parts) - 1])
            except (IndexError, ValueError, OSError):
                root = ""
            roots.add(root)
            # FLUX.3's manifest is the modular one; either marks the snapshot the companions were staged into.
            if rfilename in ("model_index.json", FLUX3_INDEX):
                snapshot_root = root or None
        # Return a snapshot only if all files share it; else the hub id resolves each file via its root.
        if snapshot_root is not None and roots != {snapshot_root}:
            return None
        return snapshot_root

    @staticmethod
    def _validate_component_files(
        fam: DiffusionFamily,
        kind: str,
        repo_id: str,
        text_encoder_files: Optional[Sequence[str]],
        vae_file: Optional[str],
    ) -> None:
        """Header-only, so a bad pick is a 400 before eviction. Single-file / GGUF denoisers only."""
        from .diffusion_comfy_components import validate_component_specs

        if (
            kind not in ("gguf", "single_file")
            or fam.single_file_is_pipeline
            or fam.pipeline_only
            or getattr(fam, "native_transformer_files_only", False)
        ):
            raise ValueError(
                "text_encoder_file / vae_file load beside a single-file or GGUF transformer; a full pipeline "
                "(or a whole-pipeline single file) already carries its own text encoders and VAE."
            )
        if is_flux3_family(fam):
            # The supplied-component loader resolves classes from a model_index.json, which FLUX.3 does not ship, and
            # its encoder is a Qwen3-VL with its processor in the same folder: refuse up front rather than load the
            # official encoder and VAE in place of the files the user picked.
            raise ValueError(
                f"separate text encoder / VAE files are not supported for '{fam.name}' yet; omit "
                "text_encoder_files and vae_file (the base repo's encoder and VAE are used)."
            )
        if fam.name == MOSS_IMAGE_FAMILY_NAME and (vae_file or text_encoder_files):
            raise ValueError(
                f"separate text encoder / VAE files are not supported for '{fam.name}' yet; omit "
                "text_encoder_files and vae_file (the base repo's encoder and VAE are used)."
            )
        if vae_file and fam.name == KREA2_FAMILY_NAME:
            raise ValueError(
                f"a separate VAE file is not supported for '{fam.name}' yet; omit vae_file."
            )
        validate_component_specs(
            list(text_encoder_files or ()),
            vae_file,
            model_path = repo_id,
            trusted_repo = _is_trusted_diffusion_repo,
        )

    def _plan_component_overrides(
        self,
        fam: DiffusionFamily,
        repo_id: str,
        base: str,
        text_encoder_files: Optional[Sequence[str]],
        vae_file: Optional[str],
        hf_token: Optional[str],
        *,
        local_files_only: bool = False,
        base_local_dir: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        download: bool = True,
        text_encoder_quant: Optional[str] = None,
    ) -> Any:
        if not text_encoder_files and not vae_file:
            return None
        from .diffusion_comfy_components import plan_component_overrides, read_model_index

        index = read_model_index(
            prefer_ungated_mirror(base, hf_token),
            base_local_dir = base_local_dir or (base if _is_local_path(base) else None),
            hf_token = hf_token,
            local_files_only = local_files_only,
            cache_dir = hub_cache_dir(),
        )

        def _resolve_hub(ref: Any) -> str:
            from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
            return hf_hub_download_with_xet_fallback(
                ref.repo_id,
                ref.filename,
                hf_token,
                cancel_event = cancel_event if cancel_event is not None else self._cancel_event,
                reuse_other_cache_root = True,
                local_files_only = local_files_only,
            )

        overrides = plan_component_overrides(
            text_encoder_files = list(text_encoder_files or ()),
            vae_file = vae_file,
            model_path = repo_id,
            model_index = index,
            family = fam.name,
            trusted_repo = _is_trusted_diffusion_repo,
            hf_token = hf_token,
            resolve_hub = _resolve_hub if download else None,
        )
        if download and overrides is not None:
            # The load path: fit every file to its real class before anything resident is unloaded.
            from .diffusion_comfy_components import verify_override_fit
            verify_override_fit(
                overrides,
                base = base_local_dir or prefer_ungated_mirror(base, hf_token),
                hf_token = hf_token,
                local_files_only = local_files_only,
                family = fam.name,
                cache_dir = hub_cache_dir(),
                text_encoder_quant = text_encoder_quant,
            )
        return overrides

    def _comfy_text_encoders(
        self,
        fam: Optional[DiffusionFamily],
        kind: str,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        hf_token: Optional[str],
        *,
        local_files_only: bool = False,
        text_encoder_quant: Optional[str] = None,
    ) -> list[str]:
        """Text-encoder files from the ComfyUI layout beside a local DiT, used when the request names none, so a
        ComfyUI user's ``text_encoders/`` copy replaces the base repo's multi-GB download. Empty whenever any part
        cannot be settled: the load then takes the base repo's encoder, as before."""
        if (
            fam is None
            or kind not in ("gguf", "single_file")
            or not gguf_filename
            or not _is_local_path(repo_id)
            or getattr(fam, "single_file_is_pipeline", False)
            or getattr(fam, "pipeline_only", False)
            or getattr(fam, "native_transformer_files_only", False)
            or os.environ.get("UNSLOTH_DIFFUSION_COMFY_TEXT_ENCODERS", "").strip().lower()
            in ("0", "false", "off", "no")
        ):
            return []
        started = time.monotonic()
        try:
            from hub.utils.comfy_models import comfy_text_encoder_files

            dit = resolve_local_gguf_child(Path(repo_id).expanduser(), gguf_filename)
            try:
                from hub.storage.scan_folders import list_scan_folders
                scan_folders = [row["path"] for row in list_scan_folders()]
            except Exception:  # noqa: BLE001 - the layout above the file still answers
                scan_folders = []
            candidates = [str(p) for p in comfy_text_encoder_files(dit, scan_folders)]
            if candidates:
                from hub.services.models import account_access
                if account_access.managed_account():
                    candidates = [
                        p for p in candidates if account_access.model_visible(os.path.realpath(p))
                    ]
            if not candidates:
                return []
            from .diffusion_comfy_components import (
                pick_comfy_text_encoders,
                read_model_index,
                verify_override_fit,
            )

            mirror = prefer_ungated_mirror(base, hf_token)
            index = read_model_index(
                mirror,
                base_local_dir = base if _is_local_path(base) else None,
                hf_token = hf_token,
                local_files_only = local_files_only,
                cache_dir = hub_cache_dir(),
            )
            planned_quant, auto = resolve_te_quant_request(
                text_encoder_quant, getattr(fam, "te_quant_auto", None)
            )
            auto_scheme = planned_quant if auto else None
            if auto_scheme is not None:
                from .diffusion_te_prequant import (
                    te_prequant_sources_confirmed,
                    te_prequant_sources_for_base,
                )
                hosted = te_prequant_sources_for_base(
                    fam,
                    base,
                    te_quant_mode = planned_quant,
                    target = self._target_for_ordinal(fam, None),
                )
                if not hosted or not te_prequant_sources_confirmed(
                    hosted, local_files_only = local_files_only
                ):
                    # Without its hosted file the default loads dense, and the ComfyUI copy is that encoder.
                    auto_scheme = None

            def _fits(overrides: Any) -> None:
                verify_override_fit(
                    overrides,
                    base = base if _is_local_path(base) else mirror,
                    hf_token = hf_token,
                    local_files_only = local_files_only,
                    family = fam.name,
                    cache_dir = hub_cache_dir(),
                    text_encoder_quant = text_encoder_quant,
                )

            picked = pick_comfy_text_encoders(
                candidates,
                model_index = index,
                family = fam.name,
                fits = _fits,
                auto_scheme = auto_scheme,
            )
        except Exception as exc:  # noqa: BLE001 - the base repo's encoder is always the fallback
            logger.info(
                "diffusion.comfy_text_encoders: none taken from the ComfyUI folder (%s)",
                type(exc).__name__,
            )
            return []
        if picked:
            logger.info(
                "diffusion.comfy_text_encoders: %s from the ComfyUI folder beside %s (%.1fs)",
                ", ".join(os.path.basename(p) for p in picked),
                gguf_filename,
                time.monotonic() - started,
            )
        return picked

    def validate_load_request(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        base_repo: Optional[str] = None,
        text_encoder_files: Optional[Sequence[str]] = None,
        vae_file: Optional[str] = None,
    ) -> DiffusionFamily:
        """Network-free and before the GPU handoff, so an unloadable pick fails before anything is
        evicted."""
        kind = resolve_model_kind(gguf_filename, model_kind)
        assert_local_pick_is_dit(repo_id, gguf_filename, "image")
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            # An excluded model gets its stated reason, not the unknown-family message that invites a doomed retry
            excluded = excluded_model_reason(repo_id)
            if excluded:
                raise ValueError(f"'{repo_id}' cannot be loaded: {excluded}")
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name. (Video models and image models "
                f"whose diffusers transformer has no single-file loader are not supported.)"
            )
        # Prequant repos have no model_index.json (a pipeline pick staged GBs, then 404d); a same-named local dir is a real pipeline.
        if (
            kind == "pipeline"
            and repo_id.strip().lower() in prequant_only_repo_ids()
            and not _is_local_path(repo_id)
        ):
            bases, schemes, te_schemes = prequant_repo_role(fam, repo_id)
            if not bases and mirror_hosts_artifacts(repo_id):
                # An artifact mirror: its folders are its base's pipeline, so name that base, not the family default.
                bases = (canonical_base(repo_id),)
            shown = " or ".join(bases) or fam.base_repo
            knobs = []
            if schemes:
                knobs.append(f"the transformer precision to {' or '.join(schemes)}")
            if te_schemes:
                knobs.append(f"the text encoder precision to {' or '.join(te_schemes)}")
            raise ValueError(
                f"'{repo_id}' hosts pre-quantised checkpoints for {shown}; it is not a pipeline to pick "
                f"directly. Pick {shown} and set {' and '.join(knobs) or 'a precision'} instead."
            )
        # Too-old diffusers matters only for a diffusers build (not native sd.cpp); imported here since the router imports this module's siblings.
        from .diffusion_engine_router import family_buildable_here

        # MOSS-Image GGUFs are sd.cpp only: refuse before eviction and before the diffusers class check.
        if getattr(fam, "native_transformer_files_only", False) and (
            kind == "single_file"
            or (
                kind == "gguf"
                and not family_buildable_here(fam, model_kind = kind, gguf_filename = gguf_filename)
            )
        ):
            raise ValueError(
                f"'{fam.name}' runs a {'GGUF' if kind == 'gguf' else 'single-file'} transformer only on the native "
                f"stable-diffusion.cpp engine, from a build that supports it (upstream master-975 or newer); the "
                f"diffusers engine loads the full pipeline repo. Select the pipeline repo, or a GGUF with the native "
                f"engine available."
            )
        if not family_buildable_here(fam, model_kind = kind, gguf_filename = gguf_filename):
            assert_pipeline_class_available(fam.pipeline_class, fam.name)
        # Whole-pipeline families take a GGUF only when it is whole too (stable-diffusion.cpp ``convert``).
        if (
            kind == "gguf"
            and fam.single_file_is_pipeline
            and not _whole_pipeline_gguf_pick(repo_id, gguf_filename)
        ):
            raise ValueError(
                f"'{fam.name}' checkpoints are whole-pipeline single files, so a GGUF loads only when it "
                f"also carries the text encoders and VAE under the checkpoint's own tensor names (a current "
                f"stable-diffusion.cpp convert of the checkpoint, on disk). Load the .safetensors checkpoint, or "
                f"such a GGUF, instead."
            )
        # A multi-denoiser family (Ideogram 4) has no transformer-only path; reject before eviction
        if kind in ("gguf", "single_file") and fam.pipeline_only:
            raise ValueError(
                f"'{fam.name}' loads only as a full diffusers pipeline (it assembles "
                f"multiple transformers), not from a single-file or GGUF checkpoint; "
                f"select the pipeline repo."
            )
        # Non-GGUF loads deserialise weights: repo-gated, except a lone .safetensors file (diffusion_single_file_trust).
        if kind != "gguf" and not single_file_load_allowed(
            _is_trusted_diffusion_repo(repo_id), kind, gguf_filename
        ):
            raise ValueError(
                f"Non-GGUF diffusion loads from '{repo_id}' are limited to a single .safetensors "
                f"checkpoint; full pipelines and other weight formats load only from unsloth/* "
                f"repos, the official base repos, or a local path. Pass a gguf_filename to load "
                f"a GGUF instead."
            )
        # The companion base repo also loads via from_pretrained, so it must clear the same trust bar
        if base_repo and base_repo.strip() and not _is_trusted_diffusion_repo(base_repo):
            raise ValueError(
                f"base_repo is restricted to unsloth/* repos (or a local path); got '{base_repo}'."
            )
        # A local base_repo loads as a full pipeline; reject a non-pipeline one before eviction
        whole_file = kind in ("single_file", "gguf") and fam.single_file_is_pipeline
        excluded = (
            (fam.denoiser_attr,) if kind in ("gguf", "single_file") and not whole_file else ()
        )
        _assert_local_base_is_pipeline(
            base_repo,
            # FLUX.3 companions come from a Modular Diffusers base, and its loader reads only modular_model_index.json.
            modular_only = is_flux3_family(fam),
            excluded_components = excluded,
            config_only_model_components = whole_file,
            # The FLUX.3 assembler reads each component from its own folder here, never from a spec's source repo.
            fixed_component_folders = FLUX3_COMPONENT_FOLDERS if is_flux3_family(fam) else None,
        )
        local_root = Path(repo_id).expanduser()
        # Path-shaped: "."/".." prefix, a backslash (never in "org/name"), or an absolute path.
        path_shaped = (
            repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or local_root.is_absolute()
        )
        if kind in ("gguf", "single_file"):
            if not gguf_filename:
                raise ValueError(f"a single-file checkpoint name is required for a '{kind}' load.")
            # Fail a kind/extension mismatch before the handoff: gguf needs .gguf, single_file must not.
            is_gguf_name = gguf_filename.lower().endswith(".gguf")
            if kind == "gguf" and not is_gguf_name:
                raise ValueError("a 'gguf' load requires a .gguf checkpoint name.")
            if kind == "single_file" and is_gguf_name:
                raise ValueError("a .gguf checkpoint needs model_kind 'gguf', not 'single_file'.")
            # A single-file load must name a real .safetensors, else it evicts chat then fails in background
            if kind == "single_file" and not gguf_filename.lower().endswith(".safetensors"):
                raise ValueError(
                    f"'{gguf_filename}' is not a loadable single-file checkpoint "
                    f"(expected a .safetensors name; use a .gguf name for a GGUF load)."
                )
            if local_root.exists():
                resolve_local_gguf_child(local_root, gguf_filename)
            elif path_shaped:
                raise FileNotFoundError(f"Local model path does not exist: {repo_id}")
        else:  # pipeline
            if gguf_filename:
                raise ValueError(
                    "a 'pipeline' load takes a full diffusers repo, not a single-file name."
                )
            if local_root.exists():
                # FLUX.3 is a Modular Diffusers layout: its manifest is modular_model_index.json.
                index_name = FLUX3_INDEX if is_flux3_family(fam) else "model_index.json"
                if not local_pipeline_components_are_complete(
                    local_root,
                    index_name,
                    fixed_component_folders = (
                        FLUX3_COMPONENT_FOLDERS if is_flux3_family(fam) else None
                    ),
                ):
                    raise FileNotFoundError(
                        f"Local pipeline directory has no complete valid {index_name}: {repo_id}"
                    )
            elif path_shaped:
                raise FileNotFoundError(f"Local model path does not exist: {repo_id}")
            elif repo_id.upper().endswith("-GGUF"):
                # A remote "*-GGUF" id is not a pipeline; reject here instead of evicting chat then failing
                raise ValueError(
                    f"'{repo_id}' is a single-file GGUF repo; load it with model_kind 'gguf' "
                    f"and a .gguf filename, not as a full pipeline."
                )
        if text_encoder_files or vae_file:
            self._validate_component_files(fam, kind, repo_id, text_encoder_files, vae_file)
        return fam

    def preflight_base_access(
        self,
        repo_id: str,
        fam: Optional[DiffusionFamily],
        *,
        gguf_filename: Optional[str] = None,
        model_kind: Optional[str] = None,
        base_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        allow_network: bool = True,
    ) -> None:
        """Base-access and FLUX.2 refusals, run before eviction; fails open when offline or transient."""
        kind = resolve_model_kind(gguf_filename, model_kind)
        if kind == "pipeline":
            base = repo_id  # the full pipeline IS the repo
        else:
            base = _resolve_base_repo(repo_id, base_repo, fam, hf_token, gguf_filename)
        # Probe the repo the load will FETCH from: refusing the upstream id would reject the gated picks the ungated
        # mirror rescues, and the swap is pure, so it decides the same here as on the load thread. Only the raise
        # matters; _run_load recomputes the excused snapshot.
        _assert_base_repo_accessible(
            prefer_ungated_mirror(base, hf_token), hf_token, _base_manifest_name(fam)
        )
        # Same reasoning for the FLUX.2 size pairing, and this is the only place it can be caught before a teardown:
        # the loader's own guard opens the downloaded checkpoint, so it fires after ~19 GB of base shards AND after
        # the resident pipeline was freed. Metadata only, and fails open on anything it cannot read. The UPSTREAM
        # base, not the mirror: the size tables key on vendor ids.
        assert_flux2_pick_compatible(fam, repo_id, gguf_filename, base, hf_token)
        # FLUX.3's variant check (distilled vs base, BFL quantized packages) off the header, for the same reason.
        assert_flux3_pick_compatible(fam, repo_id, gguf_filename, hf_token)
        # No media backend decodes a speech GGUF, and detect_family_for_pick answers from the folder name, so a csm
        # file beside a denoiser reaches this loader as one of its own. Cache-only for a load nobody asked for: this
        # probe would otherwise spend a revision HEAD, or a range request, on the one path that promised to stay off
        # the Hub.
        assert_pick_is_not_speech(repo_id, gguf_filename, hf_token, allow_network)

    @_account_owned_load
    @pauses_background_embedding
    @vram_budget_scope("image", load = True)
    def begin_load(
        self,
        repo_id: str,
        *,
        display_repo_id: Optional[str] = None,
        local_files_only: bool = False,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
        gpu_ids: Optional[list[int]] = None,
        # The ordinal the ROUTE already ranked, so the preflight and the load agree on one card.
        gpu_ordinal: Optional[int] = None,
        text_encoder_files: Optional[Sequence[str]] = None,
        vae_file: Optional[str] = None,
        allow_oversized_load: bool = False,
        _load_token: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        with self._load_cancel_lock:
            entry_token = _load_token
            self._raise_if_load_cancelled(entry_token)
        # The probes this load adds read the Hub directly; they keep a managed account's False (anonymous) sentinel.
        hub_token = normalize_token(hf_token)
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        # Resolved ONCE, here, and carried to the worker: outside it so a bad pick is the route's 400 rather than a
        # load that dies mid-download, and only once so free VRAM cannot re-rank the choice after the weights land.
        # Gated on the resolved backend, since XPU / MPS / CPU ignore physical ids and would otherwise 400 a selection
        # the contract says to drop. Re-ranked only when the caller did not already do it: free VRAM moves between the
        # route's preflight and here, so resolving twice can approve a scheme against one card and place the weights
        # on another.
        auto_ordinal = gpu_ordinal is None and not gpu_ids
        if gpu_ordinal is None and gpu_ids:
            gpu_ordinal = (
                resolve_selected_cuda_ordinal(gpu_ids)
                if resolve_diffusion_device_target().device == "cuda"
                else None
            )
        gpu_ordinals = (
            resolve_selected_cuda_ordinals(gpu_ids)
            if gpu_ids
            and len(set(gpu_ids)) > 1
            and multigpu_mode() != "off"
            and resolve_diffusion_device_target().device == "cuda"
            else AUTOMATIC_ORDINALS
            if not gpu_ids
            else []
        )
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            family_override = family_override,
            model_kind = model_kind,
            text_encoder_files = text_encoder_files,
            vae_file = vae_file,
        )
        if (
            _has_active_lora(loras)
            and getattr(fam, "single_file_is_pipeline", False)
            and resolve_model_kind(gguf_filename, model_kind) == "gguf"
        ):
            raise ValueError(_WHOLE_PIPELINE_GGUF_LORA_MSG)
        _refuse_lora_without_loader(fam, loras)
        if auto_ordinal:
            # Automatic, after validation: one GPU, Apple, CPU and ROCm skip every probe here.
            gpu_ordinal = auto_cuda_ordinal_for_load()
        # Refuse an EXPLICIT precision this host can never honor BEFORE the load starts, so the route answers 409 with
        # the reason instead of evicting the resident model, downloading several GB and only then failing. The
        # declines that need the real footprint can only be found mid-load and surface through load-progress.
        self.assert_precision_available(
            fam,
            model_kind = resolve_model_kind(gguf_filename, model_kind),
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
            memory_mode = memory_mode,
            cpu_offload = cpu_offload,
            # Uncompiled torchao loses to bf16, so 'eager' stays dense.
            speed_mode = speed_mode,
            gpu_ordinal = gpu_ordinal,
            repo_id = repo_id,
            base_repo = base_repo,
            local_files_only = local_files_only,
            text_encoder_files = text_encoder_files,
            gguf_filename = gguf_filename,
        )

        with self._lock, self._load_cancel_lock:
            self._raise_if_load_cancelled(entry_token)
            # Allow starting over a previously-failed load, but not over a live one.
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            self._load_token += 1
            token = self._load_token
            # New event per load, never clear(): unload() sets the event the running worker holds, so clearing would un-cancel it. The token is the commit guard.
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            # Seed with the family fallback; the worker resolves the real base and updates this.
            self._loading = _LoadingState(
                repo_id = repo_id,
                base_repo = fam.base_repo,
                account_id = current_account_id(),
                gguf_filename = gguf_filename,
            )

        account_thread(
            target = carry_load_budget(self._run_load),
            kwargs = dict(
                repo_id = repo_id,
                display_repo_id = display_repo_id,
                local_files_only = local_files_only,
                gguf_filename = gguf_filename,
                base_repo = base_repo,
                family_override = family_override,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                text_encoder_quant = text_encoder_quant,
                transformer_quant = transformer_quant,
                transformer_quant_fast_accum = transformer_quant_fast_accum,
                transformer_prequant_path = transformer_prequant_path,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                model_kind = model_kind,
                loras = loras,
                gpu_ordinal = gpu_ordinal,
                text_encoder_files = list(text_encoder_files or ()) or None,
                vae_file = vae_file,
                allow_oversized_load = bool(allow_oversized_load),
                _load_token = token,
                _cancel_event = cancel_event,
                _gpu_ordinals = gpu_ordinals,
                _hub_token = hub_token,
            ),
            daemon = True,
        ).start()
        return self.status()

    @scoped_allow_oversized_load
    # The multi-GPU selection (begin_load's ``_gpu_ordinals``) is visible to this load's planner and placement only.
    @with_load_selection
    @scoped_local_files_only
    @vram_budget_scope("image", load = True)
    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        # This load's own event: a later load replaces self._cancel_event rather than clearing it.
        cancel_event = kwargs.pop("_cancel_event", None) or self._cancel_event
        # API-initiated loads must download nothing; read local_files_only, do not pop it, since load_pipeline reads it too.
        local_files_only = bool(kwargs.get("local_files_only"))
        # Before any download: unsloth_zoo enters the dynamo cycle from the inductor side, racing the warm.
        try:
            from utils.torch_warmup import gate_torch_stack_import
            gate_torch_stack_import("image load", logger)
        except Exception as exc:  # noqa: BLE001 - a safety net, never a new failure
            logger.debug("dynamo import gate skipped: %r", exc)
        try:
            # Resolve the base repo and estimate sizes here (both network) so begin_load returns instantly.
            fam = detect_family_for_pick(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            kind = resolve_model_kind(kwargs.get("gguf_filename"), kwargs.get("model_kind"))
            if kind == "pipeline":
                # The full pipeline IS the repo, so the base repo is the repo itself.
                base = kwargs["repo_id"]
            else:
                base = _resolve_base_repo(
                    kwargs["repo_id"],
                    kwargs.get("base_repo"),
                    fam,
                    kwargs.get("hf_token"),
                    kwargs.get("gguf_filename"),
                )
            kwargs["base_repo"] = base
            discovered = False
            if not kwargs.get("text_encoder_files") and not kwargs.get("vae_file"):
                kwargs["text_encoder_files"] = (
                    self._comfy_text_encoders(
                        fam,
                        kind,
                        kwargs["repo_id"],
                        kwargs.get("gguf_filename"),
                        base,
                        kwargs.get("hf_token"),
                        local_files_only = local_files_only,
                        text_encoder_quant = kwargs.get("text_encoder_quant"),
                    )
                    or None
                )
                discovered = bool(kwargs["text_encoder_files"])
            # Claimed before a byte moves, so the cache-delete guard sees a Hub component repo.
            from .diffusion_comfy_components import hub_repo_ids

            component_repos = hub_repo_ids(
                kwargs.get("text_encoder_files") or (),
                kwargs.get("vae_file"),
                model_path = kwargs["repo_id"],
            )
            if component_repos:
                with self._load_cancel_lock:
                    if self._load_token == token and self._loading is not None:
                        self._loading.asset_repos = tuple(
                            dict.fromkeys(self._loading.asset_repos + component_repos)
                        )
            # Assigned before the plan so replaced components' shards are never staged.
            component_overrides = self._plan_component_overrides(
                fam,
                kwargs["repo_id"],
                base,
                kwargs.get("text_encoder_files"),
                kwargs.get("vae_file"),
                kwargs.get("hf_token"),
                local_files_only = local_files_only,
                cancel_event = cancel_event,
                text_encoder_quant = kwargs.get("text_encoder_quant"),
            )
            if discovered and component_overrides is not None:
                component_overrides.discovered = True
                component_overrides.te_quant_request = kwargs.get("text_encoder_quant")
            kwargs["_component_overrides"] = component_overrides
            replaced = tuple(component_overrides.components) if component_overrides else ()
            if component_overrides is not None and component_overrides.text_encoder_components:
                kwargs["text_encoder_quant"] = _te_quant_for_supplied_encoders(
                    kwargs.get("text_encoder_quant")
                )
            # Skip the dense shards the pre-cast encoder replaces, using the same resolver as the injection; kwargs keep the raw request.
            te_quant_planned, _ = resolve_te_quant_request(
                kwargs.get("text_encoder_quant"), getattr(fam, "te_quant_auto", None)
            )
            te_prequant_files = {
                c: v
                for c, v in self._te_prequant_plan_files(
                    fam,
                    te_quant_planned,
                    kwargs.get("hf_token"),
                    kwargs.get("gpu_ordinal"),
                    local_files_only = local_files_only,
                    base_repo = base,
                ).items()
                if c not in replaced
            }
            vae_hosted_files = (
                {}
                if "vae" in replaced
                else self._hosted_vae_plan_files(
                    fam,
                    kind,
                    base,
                    kwargs.get("gpu_ordinal"),
                    kwargs.get("hf_token"),
                    local_files_only,
                )
            )
            kwargs["_hosted_vae_planned"] = bool(vae_hosted_files)
            pipeline_planned = self._pipeline_planned_denoiser_scheme(
                fam,
                base = base,
                kind = kind,
                transformer_quant = kwargs.get("transformer_quant"),
                speed_mode = kwargs.get("speed_mode"),
                memory_mode = kwargs.get("memory_mode"),
                cpu_offload = bool(kwargs.get("cpu_offload")),
                loras = kwargs.get("loras"),
                transformer_prequant_path = kwargs.get("transformer_prequant_path"),
                gpu_ordinal = kwargs.get("gpu_ordinal"),
                repo_id = kwargs["repo_id"],
                fast_accum = kwargs.get("transformer_quant_fast_accum"),
                text_encoder_quant = te_quant_planned if te_prequant_files else None,
                local_files_only = local_files_only,
            )
            if local_files_only and pipeline_planned not in (None, PIPELINE_SEED_DECLINED):
                # Offline: without cached released shards, suppressing the seed would leave nothing to assemble.
                if not denoiser_prequant_cached(
                    fam,
                    pipeline_planned,
                    base_repo = base,
                    path_override = kwargs.get("transformer_prequant_path"),
                    cache_dir = hub_cache_dir(),
                ):
                    pipeline_planned = None
            dit_prequant = (
                None
                if local_files_only
                else self._dit_prequant_plan_source(
                    fam,
                    kind,
                    kwargs.get("hf_token"),
                    {**kwargs, "_pipeline_prequant_planned": pipeline_planned},
                )
            )
            skip_transformer_weights = kind == "pipeline" and dit_prequant is not None
            resident_sizes: list[tuple[str, int]] = []
            skipped_transformer_files: list[str] = []
            fetch_repos: dict[str, str] = {}
            planned_sizes: dict[str, dict[str, int]] = {}
            planned_keys: dict[str, dict[str, str]] = {}
            expected, base_files = self._estimate_download_bytes(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                kwargs.get("hf_token"),
                kind = kind,
                pipeline_components = _explicit_pipeline_components(fam),
                single_file_is_pipeline = bool(fam and fam.single_file_is_pipeline),
                companion_components = FLUX3_COMPONENTS if is_flux3_family(fam) else None,
                base_manifest = _base_manifest_name(fam),
                include_transformer = (
                    (
                        lambda companions, transformer_files: self._dense_quant_prefetch_needed(
                            fam,
                            kwargs,
                            companion_files = companions,
                            transformer_files = transformer_files,
                        )
                    )
                    if kind == "gguf"
                    else False
                ),
                skip_te_components = tuple(te_prequant_files) + replaced + tuple(vae_hosted_files),
                skip_transformer_weights = skip_transformer_weights,
                skipped_files_out = skipped_transformer_files,
                local_files_only = local_files_only,
                fp16_twins = kind == "pipeline"
                and self._fp16_load_target(fam, kwargs.get("gpu_ordinal")),
                resident_file_sizes_out = resident_sizes,
                fetch_repos_out = fetch_repos,
                file_sizes_out = planned_sizes,
                file_keys_out = planned_keys,
            )
            # A local operator checkpoint has no Hub entry, so the seed costs nothing and must be kept.
            local_seed = (
                bool(kwargs.get("transformer_prequant_path"))
                and pipeline_planned not in (None, PIPELINE_SEED_DECLINED)
                and denoiser_prequant_cached(
                    fam,
                    pipeline_planned,
                    base_repo = base,
                    path_override = kwargs.get("transformer_prequant_path"),
                    cache_dir = hub_cache_dir(),
                )
            )
            # Pinned into the load, decline included: the pull was scoped on this answer.
            kwargs["_pipeline_prequant_planned"] = (
                PIPELINE_SEED_DECLINED
                if pipeline_planned == PIPELINE_SEED_DECLINED
                else (
                    pipeline_planned
                    if skip_transformer_weights or local_files_only or local_seed
                    else None
                )
            )
            kwargs["_pipeline_prequant_skipped"] = tuple(skipped_transformer_files)
            kwargs["_te_prequant_resolved"] = bool(te_prequant_files)
            if dit_prequant is not None:
                expected += int(dit_prequant[2])
            # load_pipeline downloads the pre-cast encoder. A mirrored file is read in place, never cached.
            from .diffusion_te_prequant import te_prequant_unmirrored

            te_hub_files = [
                (repo, name, int(size or 0))
                for repo, files in (*te_prequant_files.values(), *vae_hosted_files.values())
                for name, size in te_prequant_unmirrored(repo, files)
            ]
            expected += sum(size for _repo, _name, size in te_hub_files)
            # Dense fallback may only use shards this prefetch staged; a local base counts as staged
            # on disk.
            kwargs["_transformer_prefetched"] = any(
                f.startswith("transformer/") for f in base_files
            ) or _local_base_transformer_present(base)
            # One mirror decision per load, so staging and assembly use the same repo.
            fetch_base = fetch_repos.get(base) or prefer_ungated_mirror(
                base, kwargs.get("hf_token"), files = base_files
            )
            kwargs["_fetch_base"] = fetch_base
            # Catch a gated base before the prefetch moves any bytes; probes fetch_base.
            base_snapshot = _assert_base_repo_accessible(
                fetch_base,
                kwargs.get("hf_token"),
                _base_manifest_name(fam),
                local_files_only = local_files_only,
            )
            # Also here, since a direct begin_load skips the plan and route preflights; keep it before _prefetch_files and the unload.
            assert_flux2_pick_compatible(
                fam, kwargs["repo_id"], kwargs.get("gguf_filename"), base, kwargs.get("hf_token")
            )
            assert_flux3_pick_compatible(
                fam,
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                kwargs.get("hf_token"),
                allow_network = not local_files_only,
            )
            # Same verdict here, so a direct begin_load is covered too.
            assert_pick_is_not_speech(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                kwargs.get("hf_token"),
                allow_network = not local_files_only,
            )
            shortfall = self.declared_footprint_shortfall(
                fam,
                kwargs["repo_id"],
                base,
                kind = kind,
                declared_files = resident_sizes,
                prequant_bytes = _prequant_plan_bytes(te_prequant_files)
                + _prequant_plan_bytes(vae_hosted_files)
                + (int(dit_prequant[2]) if skip_transformer_weights else 0),
                extra_bf16_bytes = _predownload_encoder_bf16_bytes(
                    fam, te_prequant_files, pipeline_declared = bool(resident_sizes)
                ),
                gguf_filename = kwargs.get("gguf_filename"),
                gpu_ordinal = kwargs.get("gpu_ordinal"),
                memory_mode = kwargs.get("memory_mode"),
                cpu_offload = bool(kwargs.get("cpu_offload")),
            )
            if shortfall is not None:
                logger.error(
                    "diffusion.memory: refusing oversized unified-memory load before download: %s",
                    shortfall,
                )
                raise UnifiedMemoryShortfallError(shortfall)
            # Outside the lock: a disk scan, and nothing of ours has written to the artifact repo yet.
            asset_baseline = self._cache_bytes(dit_prequant[0]) if skip_transformer_weights else 0
            planned_files = _planned_progress_files(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                kind = kind,
                base = base,
                fetch_base = fetch_base,
                base_files = base_files,
                file_sizes = planned_sizes,
                dit_prequant = dit_prequant,
                te_hub_files = te_hub_files,
            )
            with self._load_cancel_lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.fetch_repo = fetch_base
                    # One list for both numbers, so they cannot disagree. Used only when it accounts for the whole
                    # estimate: a total that came back without per-file sizes keeps the repo-level scan.
                    planned_total = sum(size for _r, _n, size in planned_files)
                    if planned_total and planned_total == expected:
                        self._loading.planned_files = planned_files
                        self._loading.planned_blob_keys = _planned_blob_keys(
                            planned_files,
                            planned_keys,
                            repo_id = kwargs["repo_id"],
                            base = base,
                            fetch_base = fetch_base,
                        )
                        self._loading.incomplete_baseline = self._incomplete_blob_sizes(
                            {repo for repo, _n, _s in planned_files}
                        )
                    self._loading.expected_bytes = expected
                    if skip_transformer_weights:
                        # Claimed before a byte moves. File entry first: load_progress reads lock-free.
                        self._loading.asset_files += (
                            (
                                dit_prequant[0],
                                dit_prequant[1],
                                int(dit_prequant[2]),
                                asset_baseline,
                            ),
                        )
                        self._loading.asset_repos = tuple(
                            dict.fromkeys(self._loading.asset_repos + (dit_prequant[0],))
                        )
            if skip_transformer_weights:
                self._fetch_denoiser_prequant(
                    dit_prequant,
                    kwargs.get("hf_token"),
                    cancel_event = cancel_event,
                )
            if te_hub_files:
                # Baselined AFTER the denoiser fetch: one repo can hold both, and its bytes are not the encoder's.
                te_baselines = {
                    repo: self._cache_bytes(repo) for repo in {r for r, _n, _s in te_hub_files}
                }
                with self._load_cancel_lock:
                    if self._load_token == token and self._loading is not None:
                        self._loading.asset_files += tuple(
                            (repo, name, size, te_baselines[repo])
                            for repo, name, size in te_hub_files
                        )
                        self._loading.asset_repos = tuple(
                            dict.fromkeys(
                                self._loading.asset_repos
                                + tuple(repo for repo, _n, _s in te_hub_files)
                            )
                        )
            # Download outside the lock so unload/an eviction can preempt the pull. The carried snapshot is the
            # fallback, never the override: it fires only when the estimate came back empty, since the metadata that
            # fills it is the same call whose failure earned the escape. Without it the load 401s with every byte
            # already on disk.
            # Where diffusers' own "Couldn't connect" fallback lands, minus its untimed revision / model_info calls.
            forced_offline = False
            if not local_files_only and _pick_cached_while_hub_unreachable(
                fam, kind, kwargs["repo_id"], kwargs.get("gguf_filename"), base, fetch_base
            ):
                logger.info(
                    "diffusion.hub_unreachable: %s is cached and the Hub did not connect moments ago; loading "
                    "from the cache",
                    kwargs["repo_id"],
                )
                forced_offline = True

            def _prefetch_and_load(offline: bool) -> None:
                kwargs["local_files_only"] = offline
                kwargs["_base_local_dir"] = (
                    self._prefetch_files(
                        kwargs["repo_id"],
                        kwargs.get("gguf_filename"),
                        base,
                        base_files,
                        kwargs.get("hf_token"),
                        cancel_event = cancel_event,
                        fetch_base = fetch_base,
                        local_files_only = offline,
                    )
                    or base_snapshot
                )
                self.load_pipeline(**kwargs)

            try:
                _prefetch_and_load(local_files_only or forced_offline)
            except OSError as exc:
                # Only the manifest is checked up front: an interrupted download still needs the online load.
                if (
                    not forced_offline
                    or self._load_token != token
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    raise
                logger.info(
                    "diffusion.hub_unreachable: %s is not fully cached (%s); retrying online",
                    kwargs["repo_id"],
                    exc,
                )
                _prefetch_and_load(False)
            with self._load_cancel_lock:
                # Only clear the marker if this load is still current (a superseder has its own token).
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the client via load_progress
            # A cancelled/superseded load raised below; don't log/stamp it onto the current load.
            if self._load_token != token:
                return
            # exc_info: clients only get str(exc), so the traceback is only available here.
            logger.error("diffusion.load_failed: %s", exc, exc_info = True)
            if self._state is not None:
                from .gpu_arbiter import DIFFUSION, restore_media_need, restore_owner_account
                from hub.services.models.account_access import restore_resident_metadata

                restore_owner_account(DIFFUSION)
                restore_media_need(DIFFUSION)
                restore_resident_metadata(DIFFUSION)
            try:
                clear_gpu_cache()
            except Exception:  # noqa: BLE001
                pass
            # Surfaced verbatim: rewrite gated 403s and redact native paths. Guarded so _loading.error is always set.
            from utils.native_path_leases import redact_native_paths

            try:
                text = (
                    modelscope_missing(exc)
                    or hub_access_message(exc, had_token = _hf_token_in_play(kwargs.get("hf_token")))
                    or dynamo_partial_init_message(exc)
                    or str(exc)
                )
            except Exception:  # noqa: BLE001
                text = str(exc)
            with self._load_cancel_lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(text)
                    self._loading.error_code = unified_memory_refusal_code(exc)
        finally:
            # Nothing else knows when the prefetch returned, so this is the drain's only owner.
            with self._load_cancel_lock:
                self._draining_repos.pop(token, None)

    def load_progress(self) -> dict[str, Any]:
        """Phase + downloaded/total bytes for the in-flight load (cache-scan based)."""
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error, error_code = loading.error_code)
        if loading is None:
            return _progress("ready" if self._state is not None else None)
        finalized = loading.finalized_scan
        if (
            finalized is not None
            and finalized[0] == loading.expected_bytes
            and _progress_latch_enabled()
        ):
            # Nothing left to download: another tree walk only holds the GIL the load thread needs.
            return _progress("finalizing", finalized[1], finalized[0], 1.0)

        if loading.planned_files and sum(size for _r, _n, size in loading.planned_files) > 0:
            downloaded = DiffusionBackend._planned_progress_bytes(
                loading.planned_files, loading.incomplete_baseline, loading.planned_blob_keys
            )
            return DiffusionBackend._progress_from(loading, downloaded)
        # Sum checkpoint + companion cache, scanning the repo the bytes LAND in (the mirror when one was swapped in),
        # else a mirrored companion download reads as zero and the bar sits still.
        companion = loading.fetch_repo or loading.base_repo

        def _companion_progress_bytes(repo_id: str) -> int:
            # An artifact mirror (``mirror_hosts_artifacts``) also holds root GGUF / prequant checkpoints the plan
            # never counted, so only its diffusers part is scanned, plus the in-flight blobs (this load's own
            # pipeline files). Every other repo is scanned exactly as before.
            if mirror_hosts_artifacts(repo_id):
                return self._cache_bytes(repo_id, pipeline_only = True, in_flight = True)
            return self._cache_bytes(repo_id)

        if loading.base_repo and loading.base_repo == loading.repo_id:
            # Full pipeline: base IS the repo, so count it once. Summing would add the upstream's stale partial blobs,
            # the very thing that selects the mirror, to the mirror's live bytes, pushing the bar to 100% / finalizing
            # mid-download.
            downloaded = _companion_progress_bytes(companion or loading.repo_id)
        elif loading.gguf_filename and mirror_hosts_artifacts(loading.repo_id):
            # A GGUF picked straight from an artifact mirror: the repo is also the companion, and also holds sibling
            # quants and sample renders the plan never counted. The selected file, the pipeline folders and the
            # in-flight blobs are what this load downloads.
            downloaded = self._cache_file_bytes(
                loading.repo_id, loading.gguf_filename
            ) + _companion_progress_bytes(loading.repo_id)
            if companion and companion.lower() != loading.repo_id.lower():
                downloaded += _companion_progress_bytes(companion)
        else:
            downloaded = self._cache_bytes(loading.repo_id)
            if companion and companion != loading.repo_id:
                downloaded += _companion_progress_bytes(companion)
        # Repos read before files (_run_load stores files first), so every repo seen has its file entry.
        asset_repos = loading.asset_repos
        asset_files = loading.asset_files
        scoped = {entry[0] for entry in asset_files}
        for asset in asset_repos:
            if asset and asset not in (loading.repo_id, companion) and asset not in scoped:
                downloaded += self._cache_bytes(asset)
        for repo, filename, size, baseline in asset_files:
            if not repo or repo in (loading.repo_id, companion):
                continue
            # Capped at the one file; a sibling scheme sits in the baseline.
            downloaded += min(
                size,
                self._cache_file_bytes(repo, filename) + max(0, self._cache_bytes(repo) - baseline),
            )
        return DiffusionBackend._progress_from(loading, downloaded)

    @staticmethod
    def _progress_from(loading: "_LoadingState", downloaded: int) -> dict[str, Any]:
        expected = loading.expected_bytes
        # Downloads done, still finalizing. The cache scan can exceed the estimate, so clamp to 100%.
        if expected > 0 and downloaded >= expected * 0.999:
            loading.finalized_scan = (expected, min(downloaded, expected))
            return _progress("finalizing", min(downloaded, expected), expected, 1.0)
        if expected <= 0:
            # No size estimate: report the phase with no byte claim, since `downloaded` scans what is PRESENT.
            return _progress("downloading")
        return _progress("downloading", downloaded, expected, min(downloaded / expected, 1.0))

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repos an in-flight load downloads, incl. a swapped-in mirror and seeded denoiser (the delete guard needs them); uses only _load_cancel_lock."""
        with self._load_cancel_lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            ids = (
                loading.repo_id,
                loading.base_repo,
                loading.fetch_repo,
                *loading.asset_repos,
            )
            return tuple(dict.fromkeys(r for r in ids if r))

    def draining_repo_ids(self) -> tuple[str, ...]:
        """Repos a cancelled load still reads, for the delete-cached guard only; not in loading_repo_ids, which the arbiter reads as ownership."""
        with self._load_cancel_lock:
            return tuple(dict.fromkeys(r for repos in self._draining_repos.values() for r in repos))

    def _hosted_vae_plan_files(
        self,
        fam: Any,
        kind: Optional[str],
        base: Optional[str],
        gpu_ordinal: Optional[int],
        hf_token: Optional[str],
        local_files_only: bool,
    ) -> dict[str, tuple[str, list[tuple[str, int]]]]:
        """``diffusion_hosted_vae.hosted_vae_plan_files`` at this pick's load dtype; {} for a family without one."""
        if not getattr(fam, "hosted_bf16_vae", None):
            return {}
        try:
            dtype = getattr(self._target_for_ordinal(fam, gpu_ordinal), "dtype", None)
        except Exception:  # noqa: BLE001 -- no target: keep the dense VAE
            return {}
        return hosted_vae_plan_files(
            fam,
            base,
            kind = kind,
            dtype = dtype,
            hf_token = hf_token,
            local_files_only = local_files_only,
            logger = logger,
        )

    @staticmethod
    def _te_prequant_plan_files(
        fam: Any,
        text_encoder_quant: Optional[str],
        hf_token: Optional[str],
        gpu_ordinal: Optional[int] = None,
        local_files_only: bool = False,
        base_repo: Optional[str] = None,
    ) -> dict[str, tuple[str, list[tuple[str, int]]]]:
        """Pre-cast text encoders from a hosted checkpoint; empty when unresolvable or local_files_only (no model_info lookups), so dense shards stay."""
        if local_files_only:
            return {}
        try:
            from .diffusion_te_prequant import TE_PREQUANT_SCHEMES

            if normalize_te_quant(text_encoder_quant) not in TE_PREQUANT_SCHEMES:
                return {}
            from huggingface_hub import HfApi

            from .diffusion_te_prequant import (
                TE_PREQUANT_COMPONENTS,
                te_prequant_hub_files,
                te_prequant_sources_for_base,
            )

            source_kwargs = {
                "te_quant_mode": text_encoder_quant,
                # The selected card, for the same reason the dense-quant probe uses it.
                "target": (
                    resolve_diffusion_device_target()
                    if gpu_ordinal is None
                    else resolve_diffusion_device_target(ordinal = gpu_ordinal)
                ),
            }
            standalone_component_bases = None
            if getattr(fam, "name", None) == HIDREAM_FAMILY_NAME:
                source_kwargs["components"] = (*TE_PREQUANT_COMPONENTS, "text_encoder_4")
                standalone_component_bases = {"text_encoder_4": HIDREAM_LLAMA_REPO}
            sources = te_prequant_sources_for_base(
                fam,
                base_repo or str(getattr(fam, "base_repo", "") or ""),
                standalone_component_bases = standalone_component_bases,
                **source_kwargs,
            )
            if not sources:
                return {}
            files = te_prequant_hub_files(sources, HfApi(token = hf_token or None), logger)
            return {c: (sources[c].location, f) for c, f in files.items()}
        except Exception as exc:  # noqa: BLE001 -- an unresolvable pre-cast keeps the dense encoder
            logger.warning("diffusion.te_prequant_plan_failed: %s", exc)
            return {}

    @_plans_at_requested_speed
    def _pipeline_planned_denoiser_scheme(
        self,
        fam: Any,
        *,
        base: Optional[str],
        kind: str,
        transformer_quant: Optional[str],
        speed_mode: Optional[str],
        memory_mode: Optional[str] = None,
        cpu_offload: bool = False,
        loras: Any = None,
        transformer_prequant_path: Optional[str] = None,
        gpu_ordinal: Optional[int] = None,
        repo_id: Optional[str] = None,
        fetch_base: Optional[str] = None,
        fast_accum: Optional[bool] = None,
        text_encoder_quant: Optional[str] = None,
        local_files_only: bool = False,
    ) -> Optional[str]:
        """The scheme an official ``kind == "pipeline"`` pick seeds its denoiser from, settled BEFORE
        anything is downloaded; None keeps the released bf16 weights and ``PIPELINE_SEED_DECLINED``
        marks an artifact-sized plan that would still offload. Never raises."""
        try:
            if kind != "pipeline" or fam is None:
                return None
            if not pipeline_seed_supported(fam):
                return None
            raw = transformer_quant
            auto = raw is None or str(raw).strip().lower() in ("", "auto")
            # Speed=off rewrites AUTO to off (no seed); an explicit scheme upgrades speed to default, so seed it to keep the load off the released bf16 shards.
            if auto and speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF:
                return None
            mode = TQ_AUTO if auto else normalize_transformer_quant(raw)
            if mode is None:
                return None
            if _has_active_lora(loras):
                return None
            # 'balanced' group-offload hooks refuse torchao weights. low_vram / cpu_offload (whole-module) seed on the
            # host when the weights survive, only the winning rung (a lower one would change the runtime's scheme),
            # and never a pinned scheme the loader serves torchao-free (native int8).
            if normalize_memory_mode(memory_mode) == MEMORY_MODE_BALANCED:
                return None
            forced_offload = _memory_request_forces_offload(memory_mode, cpu_offload)
            # SCOPED, not pinned: the pooled asyncio.to_thread thread must not be handed back set to this card.
            with diffusion_device_scope(gpu_ordinal):
                target = self._target_for_ordinal(fam, gpu_ordinal)
                if not dense_transformer_supported(target):
                    return None
                if (
                    _pipeline_quant_uncompilable_reason(target, fam, speed_mode, model_kind = kind)
                    is not None
                ):
                    return None
                if auto and _auto_keeps_bf16_reason(fam, base = base or repo_id) is not None:
                    bf16_memory = snapshot_device_memory(target)
                    bf16_plan = self._bf16_table_plan(
                        target,
                        fam,
                        base or repo_id or "",
                        memory_mode,
                        cpu_offload,
                        kind = kind,
                        repo_id = repo_id,
                        fetch_base = fetch_base,
                        text_encoder_quant = None if local_files_only else text_encoder_quant,
                        device_memory_override = replace(bf16_memory, free_mib = bf16_memory.total_mib),
                    )
                    # Offline without the released shards, bf16 cannot assemble: a cached seed is the only way.
                    if (
                        bf16_plan is not None
                        and _auto_quant_eager_reason(
                            fam, bf16_plan, transformer_prequant_path, base = base or repo_id
                        )
                        is not None
                        and not (
                            local_files_only
                            and not any(
                                self._released_transformer_cached(candidate)
                                for candidate in dict.fromkeys(
                                    (
                                        fetch_base,
                                        base,
                                        repo_id,
                                        prefer_ungated_mirror(base or repo_id or ""),
                                    )
                                )
                                if candidate
                            )
                        )
                    ):
                        return None
                scheme = _planned_quant_scheme(
                    fam,
                    target,
                    mode,
                    base_repo = base,
                    prequant_path = transformer_prequant_path,
                )
                if scheme is None or scheme == TQ_AUTO:
                    return None
                # Under auto, a rung whose plan still offloads yields to the next resident rung; an explicit scheme is never swapped, only honored or refused.
                rungs: list[str] = [scheme]
                if auto and not forced_offload:
                    try:
                        from .diffusion_transformer_quant import auto_scheme_candidates
                        below = list(
                            auto_scheme_candidates(
                                target,
                                getattr(fam, "name", None),
                                base_repo = base,
                                has_prequant = _prequant_probe(
                                    fam,
                                    base_repo = base,
                                    prequant_path = transformer_prequant_path,
                                ),
                            )
                        )
                        if scheme in below:
                            rungs.extend(below[below.index(scheme) + 1 :])
                    except Exception:  # noqa: BLE001 -- no lower rungs is just "no retry"
                        pass
                declined = False
                # An offloaded quantised rung beats bf16 shards; a later resident rung beats both.
                offloaded_rung: Optional[str] = None
                memory = snapshot_device_memory(target)
                # Like the VRAM override below: the outgoing pipeline's host weights are gone before this one pins.
                reclaimable_host_mib = pipeline_host_mib(
                    getattr(getattr(self, "_state", None), "pipe", None)
                )
                for rung in rungs:
                    if (
                        forced_offload
                        and not auto
                        and native_quant_scheme(
                            target, rung, family = getattr(fam, "name", None), offload = True
                        )
                        is not None
                    ):
                        continue
                    source = denoiser_prequant_source(
                        fam,
                        rung,
                        base_repo = base,
                        path_override = transformer_prequant_path,
                        local_files_only = local_files_only,
                        cache_dir = hub_cache_dir(),
                    )
                    if source is None:
                        continue
                    if rung != scheme and _is_local_cache_entry(source):
                        # Cache seeds only auto's pick: one explicit int8 load must not turn later auto loads into int8.
                        continue
                    # The loader's checkpoint validation refuses a FORCED fp8 accumulate the artifact does not bake,
                    # but only after the plan dropped the released shards. Hosted artifacts and Studio's own local
                    # pre-quant cache entries both bake the auto value; an operator's own file is left to it.
                    if _prequant_bakes_auto_accumulate(source) and hosted_fast_accum_conflict(
                        rung, fast_accum
                    ):
                        continue
                    candidate = resolve_dense_quant_candidate(
                        fam = fam,
                        target = target,
                        requested = rung,
                        base_repo = base,
                        prequant_path = transformer_prequant_path,
                        force_dense = False,
                        logger = None,
                    )
                    if candidate is None or not candidate.prequant:
                        continue
                    planned = self._plan_memory(
                        target,
                        None,
                        base or repo_id or "",
                        fam,
                        memory_mode,
                        cpu_offload,
                        kind = kind,
                        repo_id = repo_id,
                        fetch_base = fetch_base,
                        transformer_resident_override_mib = candidate.steady_transformer_mib,
                        **self._candidate_companion_overrides(
                            candidate, fam, base or repo_id or "", target, text_encoder_quant
                        ),
                        device_memory_override = replace(memory, free_mib = memory.total_mib),
                    )
                    if not torchao_survives_plan(
                        planned, rung, reclaimable_host_mib = reclaimable_host_mib
                    ):
                        logger.info(
                            "diffusion.denoiser_prequant: an artifact-sized plan for %s places the "
                            "denoiser with '%s' offload, which its torchao weights do not survive "
                            "here, so the released shards are kept",
                            rung,
                            planned.offload_policy,
                        )
                        declined = True
                        continue
                    if not plan_keeps_transformer_resident(planned):
                        offloaded_rung = offloaded_rung or rung
                        continue
                    return rung
                if offloaded_rung is not None:
                    return offloaded_rung
                return PIPELINE_SEED_DECLINED if declined else None
        except Exception as exc:  # noqa: BLE001 -- an unanswerable probe keeps the released shards
            logger.warning("diffusion.denoiser_prequant_plan_failed: %s", exc)
            return None

    def _dit_prequant_plan_source(
        self,
        fam: Any,
        kind: str,
        hf_token: Optional[str],
        kwargs: dict[str, Any],
        failures_out: Optional[list] = None,
    ) -> Optional[tuple[str, str, int]]:
        """Hosted prequant denoiser replacing the dense shards; without it the plan undercounts and the file is pulled inline. Mirrors _load_dense_quant_pipeline gates."""
        if fam is None:
            return None
        planned = kwargs.get("_pipeline_prequant_planned") if kind == "pipeline" else None
        if kind == "pipeline":
            if not planned or planned == PIPELINE_SEED_DECLINED:
                return None
        elif kind != "gguf":
            return None
        try:
            if planned:
                with diffusion_device_scope(kwargs.get("gpu_ordinal")):
                    source = denoiser_prequant_source(
                        fam,
                        planned,
                        base_repo = kwargs.get("base_repo"),
                        path_override = kwargs.get("transformer_prequant_path"),
                        local_files_only = bool(kwargs.get("local_files_only")),
                        hf_token = hf_token,
                        cache_dir = hub_cache_dir(),
                    )
                return self._prequant_source_hub_entry(source, hf_token, scheme = planned)
            raw = kwargs.get("transformer_quant")
            auto = raw is None or str(raw).strip().lower() in ("", "auto")
            # Speed=off forces an AUTO quant off (no prequant fetched); an explicit quant still takes it.
            speed = kwargs.get("speed_mode")
            if auto and speed is not None and str(speed).strip().lower() == SPEED_OFF:
                return None
            mode = TQ_AUTO if auto else normalize_transformer_quant(raw)
            if mode is None:
                return None
            # A LoRA bake forces the DENSE path, so no prequant is fetched.
            if _has_active_lora(kwargs.get("loras")):
                return None
            # balanced/low_vram offload by mode, so no replan can enable the fast path; keep the GGUF.
            mm = normalize_memory_mode(kwargs.get("memory_mode"))
            if mm in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM):
                return None
            if mm is None and kwargs.get("cpu_offload"):
                return None
            # The card this load will land on, and SCOPED for the same reason as the dense-quant probe
            with diffusion_device_scope(kwargs.get("gpu_ordinal")):
                target = self._target_for_ordinal(fam, kwargs.get("gpu_ordinal"))
                # Auto declines an uncached hosted checkpoint, so only a cached one or an explicit request counts.
                if (
                    auto
                    and _uncached_prequant_repo(
                        fam,
                        target,
                        mode,
                        base_repo = kwargs.get("base_repo"),
                        prequant_path = kwargs.get("transformer_prequant_path"),
                    )
                    is not None
                ):
                    return None
                scheme = _planned_quant_scheme(
                    fam,
                    target,
                    mode,
                    base_repo = kwargs.get("base_repo"),
                    prequant_path = kwargs.get("transformer_prequant_path"),
                )
                if scheme is None:
                    return None
                source = usable_prequant_source(
                    fam,
                    scheme,
                    path_override = kwargs.get("transformer_prequant_path"),
                    base_repo = kwargs.get("base_repo"),
                    local_files_only = bool(kwargs.get("local_files_only")),
                    hf_token = hf_token,
                    cache_dir = hub_cache_dir(),
                )

                if source is None and auto:
                    retry = self._auto_prequant_retry_scheme(
                        target,
                        fam,
                        mode,
                        scheme,
                        base_repo = kwargs.get("base_repo"),
                        path_override = kwargs.get("transformer_prequant_path"),
                        loras = kwargs.get("loras"),
                    )
                    if retry is not None:
                        source = usable_prequant_source(
                            fam,
                            retry,
                            path_override = kwargs.get("transformer_prequant_path"),
                            base_repo = kwargs.get("base_repo"),
                            local_files_only = bool(kwargs.get("local_files_only")),
                            hf_token = hf_token,
                            cache_dir = hub_cache_dir(),
                        )
                        # The retry replaces the pick, so the readability question below is about the rung actually planned.
                        scheme = retry
                return self._prequant_source_hub_entry(
                    source, hf_token, failures_out, scheme = scheme
                )
        except Exception as exc:  # noqa: BLE001 -- an unsizable prequant must not fail the plan
            logger.warning("diffusion.dit_prequant_plan_failed: %s", exc)
            # Best-effort for the UI only: dense shards are already excluded, so a swallowed lookup must mark the plan partial or a download-only caller reports success.
            if failures_out is not None:
                failures_out.append(exc)
            return None

    @staticmethod
    def _prequant_source_hub_entry(
        source: Any,
        hf_token: Optional[str],
        failures_out: Optional[list] = None,
        scheme: Optional[str] = None,
    ) -> Optional[tuple[str, str, int]]:
        """Must be readable by this install, not merely present; unreadable names are skipped for a later candidate, and a full miss keeps the dense shards."""
        if source is None or getattr(source, "kind", None) != "repo":
            return None
        from .diffusion_nvfp4_flag import nvfp4_blocked, nvfp4_repo_blocked

        if nvfp4_blocked(scheme) or nvfp4_repo_blocked(source.location):
            return None
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            # Decides what loads: always asked, only bounded.
            info = hub_reachability.model_info(
                HfApi(token = hf_token or None),
                source.location,
                files_metadata = True,
                skip_when_unreachable = False,
            )
        except RepositoryNotFoundError as exc:
            # 401/403/404 on the repo = no hosted checkpoint, not a partial listing (GatedRepoError subclasses this).
            logger.info(
                "diffusion.prequant_repo_unavailable: %s (%s)", source.location, type(exc).__name__
            )
            return None
        sizes = {s.rfilename: int(getattr(s, "size", 0) or 0) for s in (info.siblings or [])}
        # All candidates in loader order (safetensors, then pickle spellings).
        from .diffusion_prequant import (
            candidate_filenames_of,
            prefer_cached_pickle_twins,
            restricted_prequant_load_supported,
        )

        # The resolver's order, so a cached .pt is priced and staged instead of its uncached twin.
        ordered = prefer_cached_pickle_twins(
            source.location,
            candidate_filenames_of(source),
            readable = lambda n: restricted_prequant_load_supported(scheme, n),
        )
        for name in ordered:
            if name and name in sizes and restricted_prequant_load_supported(scheme, name):
                return (source.location, name, int(sizes[name]))
        # Holds neither name: dense shards are already excluded, so the plan is partial.
        if failures_out is not None:
            failures_out.append(
                RuntimeError(
                    f"prequant artifact missing from {source.location}: "
                    f"tried {list(candidate_filenames_of(source))}"
                )
            )
        return None

    def _nvfp4_checkpoint_will_load(
        self,
        fam: Any,
        base: Optional[str],
        path_override: Optional[str],
        hf_token: Optional[str],
        *,
        local_files_only: bool = False,
        loras: Any = None,
    ) -> bool:
        """Whether an NVFP4 load opens a pre-quantised checkpoint (the only FlashInfer path); unanswerable -> False."""
        if not nvfp4_diffusion_enabled():
            return False
        try:
            if _has_active_lora(loras):
                return False
            source = usable_prequant_source(
                fam,
                TQ_NVFP4,
                path_override = path_override,
                base_repo = base,
                local_files_only = local_files_only,
                hf_token = hf_token,
                cache_dir = hub_cache_dir(),
            )
            if source is None:
                return False
            if getattr(source, "kind", None) != "repo":
                return True
            if prequant_checkpoint_cached(
                source, cache_dir = hub_cache_dir(), online = False if local_files_only else None
            ):
                return True
            if local_files_only:
                return False
            return self._prequant_source_hub_entry(source, hf_token, scheme = TQ_NVFP4) is not None
        except Exception as exc:  # noqa: BLE001 -- a refused or unreachable listing is no checkpoint
            logger.info(
                "diffusion.nvfp4_install: no reachable NVFP4 checkpoint for %s, so the load runs on "
                "torchao and FlashInfer is not installed (%s)",
                base,
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _estimate_download_bytes(
        repo_id: str,
        gguf_filename: Optional[str],
        base_repo: str,
        hf_token: Optional[str],
        *,
        kind: str = "gguf",
        pipeline_components: Optional[Sequence[str]] = None,
        single_file_is_pipeline: bool = False,
        include_transformer: "bool | Callable[[Sequence[str], Sequence[str]], bool]" = False,
        companion_components: Optional[Sequence[str]] = None,
        base_manifest: str = "model_index.json",
        sizes_out: Optional[dict[str, int]] = None,
        file_sizes_out: Optional[dict[str, dict[str, int]]] = None,
        resident_file_sizes_out: Optional[list[tuple[str, int]]] = None,
        revisions_out: Optional[dict[str, str]] = None,
        fetch_repos_out: Optional[dict[str, str]] = None,
        file_keys_out: Optional[dict[str, dict[str, str]]] = None,
        skip_te_components: tuple[str, ...] = (),
        skip_transformer_weights: bool = False,
        skipped_files_out: Optional[list[str]] = None,
        failures_out: Optional[list] = None,
        local_files_only: bool = False,
        fp16_twins: bool = False,
    ) -> tuple[int, list[str]]:
        """Total download size for the progress bar, plus the base-repo files to fetch (the prefetch
        reuses this list, so the base is listed only once).

        ``sizes_out`` takes per-repo byte totals, so the download plan can size one job per repo off
        this same pair of Hub lookups. ``resident_file_sizes_out`` takes only the selected pipeline
        components and default weight variant, the set ``from_pretrained`` materialises in memory.
        ``revisions_out`` records the commit each lookup described, so a cache probe can ask about
        the SAME revision. ``fetch_repos_out`` records the repo that supplied pipeline metadata, so
        staging reads that pinned file set rather than making a second mirror choice.

        For a ``pipeline`` load the whole repo IS the pipeline, so transformer/ is INCLUDED, unlike
        the GGUF / single-file paths. For a ``single_file_is_pipeline`` family (SDXL) the base repo
        supplies only config/tokenizer and its weight files are skipped.

        ``include_transformer`` may be a CALLABLE ``(companions, transformer_files) -> bool``, called
        once with this repo's listing split either side of ``transformer/``: the widening decision
        turns on what those sets say about the cache and the fetch repo, and that listing lives here.

        ``skip_te_components`` names the text encoders this pick loads PRE-CAST, so their dense
        shards are not counted or fetched (tens of wasted GB nothing ever opens). The rest of the
        component folder is kept: the pre-cast loader meta-inits from the base repo's config.
        ``skip_transformer_weights`` does the same for a seeded pick's DENOISER, keeping
        ``transformer/config.json`` for the same reason.

        ``local_files_only`` returns the metadata-unavailable answer ``(0, [])`` instead of asking:
        the bytes belong to a download that is not permitted, and the empty list stages nothing.

        ``fp16_twins`` (the load casts to fp16): a ``pipeline`` whose every listed component ships a single-file
        ``.fp16`` twin stages the twins instead of the defaults (``diffusion_fp16_variant``), unless the defaults are
        already cached and the twins are not. The load then reads them with ``variant="fp16"``.
        """
        if local_files_only:
            return 0, []
        from huggingface_hub import HfApi

        # Metadata comes from the repo the bytes will: the base's unsloth mirror unless the upstream already holds
        # them (prefer_ungated_mirror), so a fresh install asks the vendor nothing, and a cached upstream keeps the
        # listing and the commit pin its cache probe needs.
        from .diffusion_te_prequant import is_prequant_covered_weight

        api = HfApi()
        total = 0
        base_files: list[str] = []

        def _dense_te_shard(rfilename: str) -> bool:
            return bool(skip_te_components) and is_prequant_covered_weight(
                rfilename, skip_te_components
            )

        def _seeded_denoiser_shard(rfilename: str) -> bool:
            return skip_transformer_weights and is_prequant_covered_weight(
                rfilename, (DENOISER_COMPONENT,)
            )

        try:
            if kind == "pipeline":

                def _pipeline_listing(metadata_repo: str) -> tuple[Any, Any, list[Any]]:
                    info = hub_reachability.model_info(
                        api, metadata_repo, files_metadata = True, token = hf_token
                    )
                    components = _pipeline_components_from_index(
                        metadata_repo,
                        info,
                        hf_token,
                        explicit_components = pipeline_components,
                        failures_out = failures_out,
                    )
                    candidates = [
                        s
                        for s in info.siblings
                        if _pipeline_file_downloaded(s.rfilename)
                        and not _dense_te_shard(s.rfilename)
                        and (components is None or _pipeline_selected_file(s.rfilename, components))
                    ]
                    # Prefer safetensors over sibling .bin weights.
                    st_dirs = {
                        s.rfilename.rsplit("/", 1)[0]
                        for s in candidates
                        if s.rfilename.endswith(".safetensors")
                    }
                    picked = [
                        s
                        for s in candidates
                        if not (
                            s.rfilename.endswith(".bin")
                            and s.rfilename.rsplit("/", 1)[0] in st_dirs
                        )
                    ]
                    # Split off AFTER the variant preference, else a .bin twin fetches weights nothing opens.
                    seeded = [s for s in picked if _seeded_denoiser_shard(s.rfilename)]
                    if seeded:
                        picked = [s for s in picked if not _seeded_denoiser_shard(s.rfilename)]
                    if skipped_files_out is not None:
                        skipped_files_out[:] = [s.rfilename for s in seeded]
                    if fp16_twins and not seeded and components is not None:
                        picked = _swap_in_fp16_twins(metadata_repo, info, picked, hf_token)
                    return info, components, picked

                # The family's own index: FLUX.3 ships only modular_model_index.json, so probing
                # model_index.json would never find a cached upstream and restage the mirror.
                metadata_repo = prefer_ungated_mirror(repo_id, hf_token, files = (base_manifest,))
                info, components, picked = _pipeline_listing(metadata_repo)
                picked = [s for s in picked if not _listing_extra(metadata_repo, s.rfilename)]
                base_files = [s.rfilename for s in picked]
                fetch_repo = prefer_ungated_mirror(repo_id, hf_token, files = base_files)
                if fetch_repo.lower() != metadata_repo.lower():
                    metadata_repo = fetch_repo
                    info, components, picked = _pipeline_listing(metadata_repo)
                    picked = [s for s in picked if not _listing_extra(metadata_repo, s.rfilename)]
                    base_files = [s.rfilename for s in picked]
                total = sum(s.size or 0 for s in picked)
                if fetch_repos_out is not None:
                    fetch_repos_out[repo_id] = metadata_repo
                if sizes_out is not None:
                    sizes_out[repo_id] = total
                if file_sizes_out is not None:
                    file_sizes_out[repo_id] = {s.rfilename: int(s.size or 0) for s in picked}
                if file_keys_out is not None:
                    file_keys_out[repo_id] = {
                        s.rfilename: k for s in picked if (k := _sibling_blob_key(s))
                    }
                if resident_file_sizes_out is not None and components is not None:
                    resident_file_sizes_out.extend(
                        (s.rfilename, int(s.size or 0))
                        for s in picked
                        if _pipeline_resident_weight(s.rfilename)
                    )
                _record_revision(revisions_out, metadata_repo, info)
                return total, base_files
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = hub_reachability.model_info(
                    api, repo_id, files_metadata = True, token = hf_token
                )
                gguf_bytes = sum(s.size or 0 for s in info.siblings if s.rfilename == gguf_filename)
                total += gguf_bytes
                if sizes_out is not None:
                    sizes_out[repo_id] = gguf_bytes
                if file_sizes_out is not None:
                    file_sizes_out[repo_id] = {
                        s.rfilename: int(s.size or 0)
                        for s in info.siblings
                        if s.rfilename == gguf_filename
                    }
                if file_keys_out is not None:
                    file_keys_out[repo_id] = {
                        s.rfilename: k
                        for s in info.siblings
                        if s.rfilename == gguf_filename and (k := _sibling_blob_key(s))
                    }
                _record_revision(revisions_out, repo_id, info)
            # A whole-pipeline single file (SDXL, or its GGUF) needs only the base's config/tokenizer, not its weights.
            if kind in ("single_file", "gguf") and single_file_is_pipeline:
                base_filter = _base_config_file_downloaded
            else:

                def base_filter(rfilename: str) -> bool:
                    # ``companion_components``: an assembler that opens only these folders (FLUX.3), so the rest of
                    # the base (BFL's variants/ packages) is never staged; transformer/ still follows the quant path.
                    if (
                        companion_components is not None
                        and "/" in rfilename
                        and rfilename.split("/", 1)[0] not in companion_components
                    ):
                        return False
                    return _base_file_downloaded(
                        rfilename, include_transformer = include_transformer, manifest = base_manifest
                    )

            def _base_listing(listing_repo: str) -> list[Any]:
                info = hub_reachability.model_info(
                    api, listing_repo, files_metadata = True, token = hf_token
                )
                # Under the repo it describes: a mirror's commit says nothing about a cached upstream snapshot.
                _record_revision(revisions_out, listing_repo, info)
                return [s for s in info.siblings if not _listing_extra(listing_repo, s.rfilename)]

            listing_repo = prefer_ungated_mirror(base_repo, hf_token)
            base_siblings = _base_listing(listing_repo)
            # Same two-step choice as the pipeline listing above: "any weight cached" picks the listing repo, the exact
            # companion set then picks the repo they come from, and a different answer re-lists from that repo. The
            # plan still takes its own fetch decision from the final file list. FLUX.3's mirror is its FP8 repo, whose
            # companion folders are BFL's byte for byte and whose extra root files base_filter never stages, so its
            # listing stays on the repo the prefetch pulls: an anonymous load never asks the gated vendor repo anything.
            if companion_components is None:
                wanted = [
                    s.rfilename
                    for s in base_siblings
                    if _base_file_downloaded(
                        s.rfilename, include_transformer = False, manifest = base_manifest
                    )
                ]
                fetch_repo = prefer_ungated_mirror(base_repo, hf_token, files = wanted)
                if fetch_repo.lower() != listing_repo.lower():
                    listing_repo = fetch_repo
                    base_siblings = _base_listing(listing_repo)
            if callable(include_transformer):
                kept = [s.rfilename for s in base_siblings if not _dense_te_shard(s.rfilename)]
                companions = tuple(
                    f
                    for f in kept
                    if _base_file_downloaded(f, include_transformer = False, manifest = base_manifest)
                    and (
                        companion_components is None
                        or "/" not in f
                        or f.split("/", 1)[0] in companion_components
                    )
                )
                skip = set(companions)
                transformer_files = tuple(
                    f
                    for f in kept
                    if f not in skip
                    and _base_file_downloaded(f, include_transformer = True, manifest = base_manifest)
                    # A scoped assembler's dense loader opens only transformer/, not BFL's variants/ packages.
                    and (companion_components is None or f.split("/", 1)[0] == "transformer")
                )
                # Rebinding before base_filter is ever CALLED, so the loop below reads the answer.
                include_transformer = bool(include_transformer(companions, transformer_files))
            # The mirror choice above saw only the companions. Once the verdict may have widened the set to the dense
            # transformer shards, decide again on the FINAL list: a gated upstream holding the companions but not the
            # shards must not stay the fetch repo, or the shards 401 for an anonymous user. Recorded for the plan and
            # the load, which stage and read from this one answer.
            final_files = [
                s.rfilename
                for s in base_siblings
                if base_filter(s.rfilename) and not _dense_te_shard(s.rfilename)
            ]
            final_repo = prefer_ungated_mirror(base_repo, hf_token, files = final_files)
            if final_repo.lower() != listing_repo.lower():
                listing_repo = final_repo
                base_siblings = _base_listing(listing_repo)
            if fetch_repos_out is not None and not _is_local_path(base_repo):
                fetch_repos_out.setdefault(base_repo, listing_repo)
            # Do not stage a combined repo's checkpoint again as a companion.
            counted = {gguf_filename} if base_repo == repo_id and gguf_filename else set()
            base_bytes = 0
            base_sizes: dict[str, int] = {}
            for s in base_siblings:
                if not base_filter(s.rfilename) or _dense_te_shard(s.rfilename):
                    continue
                if s.rfilename in counted:
                    continue
                if file_keys_out is not None and (k := _sibling_blob_key(s)):
                    file_keys_out.setdefault(base_repo, {})[s.rfilename] = k
                base_files.append(s.rfilename)
                base_bytes += s.size or 0
                base_sizes[s.rfilename] = int(s.size or 0)
            total += base_bytes
            # Accumulate, never assign: on a combined repo both branches key the same repo id, and assignment drops the checkpoint's size and file map.
            if sizes_out is not None:
                sizes_out[base_repo] = sizes_out.get(base_repo, 0) + base_bytes
            if file_sizes_out is not None:
                file_sizes_out.setdefault(base_repo, {}).update(base_sizes)
        except Exception as exc:  # noqa: BLE001 - estimate is best-effort
            logger.warning("diffusion.size_estimate_failed: %s", exc)
            # Lets no-download callers tell a partial file list from a complete one.
            if failures_out is not None:
                failures_out.append(exc)
        return total, base_files

    @vram_budget_scope("image", load = True)
    def download_plan(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        hf_token: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        allow_device_probe: bool = True,
        memory_verdict: bool = True,
        discover_text_encoders: bool = True,
        **load_kwargs: Any,
    ) -> dict[str, Any]:
        """The repos + exact files this pick needs, so the Hub download manager can fetch them with the
        same file scope the loader would.

        A plain snapshot_download would also pull what the loader deliberately skips (the packaged
        root single, transformer/ shards, fp16 twins), tens of GB per FLUX repo. Resolves
        family/kind/base exactly as ``_run_load`` does. Local paths are already on disk and yield no
        entries.

        ``text_encoder_quant`` is read for the same reason as the DiT quant: an fp8 request loads a
        hosted PRE-CAST encoder, so the base repo's dense encoder shards must not be staged and the
        pre-cast checkpoint must be.

        ``allow_device_probe=False`` skips device-dependent precision planning and the memory
        verdict while training owns the GPU. It is not free: the pre-cast encoder and the DiT
        prequant need a target to resolve, so clearing it also changes WHICH FILES the plan counts.
        ``memory_verdict=False`` suppresses only the oversized-unified-memory refusal.
        ``discover_text_encoders=False`` keeps the base repo's encoder for an engine that never opens a single-file one.
        """
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        kind = resolve_model_kind(gguf_filename, model_kind)
        if kind == "pipeline":
            base = repo_id  # the full pipeline IS the repo
        elif fam is None and not (base_repo or "").strip():
            # Unrecognised GGUF: no companions to resolve, so an empty plan; raising would 500 the picker route, which asks for a plan on every hub pick.
            return {"entries": [], "total_bytes": 0, "required_bytes": 0, "checkpoint_bytes": 0}
        else:
            base = _resolve_base_repo(repo_id, base_repo, fam, hf_token, gguf_filename)
        # Reported, not raised. The images page falls back to /images/load on ANY plan failure, so a 400 here would
        # start the very download this is meant to prevent; carried in the envelope instead, the picker can refuse at
        # SELECTION time. Metadata only, and None whenever nothing is known to be wrong. The speech verdict belongs
        # here too, not only on the load preflight: the Images page stages and downloads before it calls load, so a
        # later refusal arrives after the bytes.
        incompatible = (
            flux2_pick_mismatch(fam, repo_id, gguf_filename, base, hf_token)
            # The FLUX.3 variant / BFL-package verdict, off the header or the manifest, for the same reason.
            or flux3_pick_mismatch(fam, repo_id, gguf_filename, hf_token)
            or speech_pick_refusal(repo_id, gguf_filename, hf_token)
        )
        component_overrides = None
        if (
            discover_text_encoders
            and not load_kwargs.get("text_encoder_files")
            and not load_kwargs.get("vae_file")
        ):
            # The same pick the load makes, so the plan stages no encoder the load will not open.
            load_kwargs["text_encoder_files"] = (
                self._comfy_text_encoders(
                    fam,
                    kind,
                    repo_id,
                    gguf_filename,
                    base,
                    hf_token,
                    text_encoder_quant = text_encoder_quant,
                )
                or None
            )
        if (
            load_kwargs.get("text_encoder_files") or load_kwargs.get("vae_file")
        ) and fam is not None:
            self._validate_component_files(
                fam,
                kind,
                repo_id,
                load_kwargs.get("text_encoder_files"),
                load_kwargs.get("vae_file"),
            )
            component_overrides = self._plan_component_overrides(
                fam,
                repo_id,
                base,
                load_kwargs.get("text_encoder_files"),
                load_kwargs.get("vae_file"),
                hf_token,
                download = False,
            )
        replaced = tuple(component_overrides.components) if component_overrides else ()
        if component_overrides is not None and component_overrides.text_encoder_components:
            text_encoder_quant = _te_quant_for_supplied_encoders(text_encoder_quant)
        # Only a checkpoint that really resolves on the Hub earns the right to drop dense shards
        # Resolved through the same tri-state the loader uses, so the size the picker SHOWS is the size the load will
        # actually move on a family whose unset request takes a hosted encoder.
        te_quant_planned, _ = resolve_te_quant_request(
            text_encoder_quant, getattr(fam, "te_quant_auto", None)
        )
        te_files = (
            {
                c: v
                for c, v in self._te_prequant_plan_files(
                    fam,
                    te_quant_planned,
                    hf_token,
                    load_kwargs.get("gpu_ordinal"),
                    base_repo = base,
                ).items()
                if c not in replaced
            }
            if allow_device_probe
            else {}
        )
        vae_hosted_files = (
            self._hosted_vae_plan_files(
                fam, kind, base, load_kwargs.get("gpu_ordinal"), hf_token, False
            )
            if allow_device_probe and "vae" not in replaced
            else {}
        )
        # Settled before the listing: only a checkpoint that resolves on the Hub may drop the released shards.
        pipeline_planned = (
            self._pipeline_planned_denoiser_scheme(
                fam,
                base = base,
                kind = kind,
                transformer_quant = load_kwargs.get("transformer_quant"),
                speed_mode = load_kwargs.get("speed_mode"),
                memory_mode = load_kwargs.get("memory_mode"),
                cpu_offload = bool(load_kwargs.get("cpu_offload")),
                loras = load_kwargs.get("loras"),
                transformer_prequant_path = load_kwargs.get("transformer_prequant_path"),
                gpu_ordinal = load_kwargs.get("gpu_ordinal"),
                repo_id = repo_id,
                fast_accum = load_kwargs.get("transformer_quant_fast_accum"),
                text_encoder_quant = te_quant_planned if te_files else None,
            )
            if allow_device_probe
            else None
        )
        # Declared before the prequant lookup so a repo missing both artifacts marks the plan partial; the dense shards are already dropped.
        plan_failures: list = []
        dit_prequant = (
            self._dit_prequant_plan_source(
                fam,
                kind,
                hf_token,
                {
                    **load_kwargs,
                    "base_repo": base,
                    "_pipeline_prequant_planned": pipeline_planned,
                },
                plan_failures,
            )
            if allow_device_probe
            else None
        )
        skip_transformer_weights = kind == "pipeline" and dit_prequant is not None
        sizes: dict[str, int] = {}
        file_sizes: dict[str, dict[str, int]] = {}
        resident_file_sizes: list[tuple[str, int]] = []
        revisions: dict[str, str] = {}
        fetch_repos: dict[str, str] = {}
        required_total, base_files = self._estimate_download_bytes(
            repo_id,
            gguf_filename,
            base,
            hf_token,
            kind = kind,
            pipeline_components = _explicit_pipeline_components(fam),
            single_file_is_pipeline = bool(fam and fam.single_file_is_pipeline),
            companion_components = FLUX3_COMPONENTS if is_flux3_family(fam) else None,
            base_manifest = _base_manifest_name(fam),
            # The RESOLVED base, as the load passes it: a variant base picks its own pre-quant repo. Deferred exactly
            # as _run_load defers it. Called eagerly this runs before the base listing exists, so the cache gate sees
            # no transformer shards, always declines, and the plan scopes narrower than the load: the registry then
            # reports scope_file_mismatch and the plan cannot adopt the load's own in-flight job.
            include_transformer = (
                (
                    lambda companions, transformer_files: self._dense_quant_prefetch_needed(
                        fam,
                        {
                            **load_kwargs,
                            "repo_id": repo_id,
                            "gguf_filename": gguf_filename,
                            "base_repo": base,
                            "hf_token": hf_token,
                        },
                        companion_files = companions,
                        transformer_files = transformer_files,
                    )
                )
                if kind == "gguf" and allow_device_probe
                else False
            ),
            sizes_out = sizes,
            file_sizes_out = file_sizes,
            resident_file_sizes_out = resident_file_sizes,
            revisions_out = revisions,
            fetch_repos_out = fetch_repos,
            skip_te_components = tuple(te_files) + replaced + tuple(vae_hosted_files),
            skip_transformer_weights = skip_transformer_weights,
            failures_out = plan_failures,
            fp16_twins = kind == "pipeline"
            and allow_device_probe
            and self._fp16_load_target(fam, load_kwargs.get("gpu_ordinal")),
        )
        # Probe the staged id: a gated base answers model_info anonymously, then 401s mid-download.
        fetch_base = fetch_repos.get(base) or prefer_ungated_mirror(
            base, hf_token, files = base_files
        )
        _assert_base_repo_accessible(fetch_base, hf_token, _base_manifest_name(fam))
        entries: list[dict[str, Any]] = []
        checkpoint_bytes = 0
        if gguf_filename:
            checkpoint_bytes = int(
                file_sizes.get(repo_id, {}).get(gguf_filename, sizes.get(repo_id, 0))
            )
        te_prequant_bytes = _prequant_plan_bytes(te_files)
        required_total += te_prequant_bytes + _prequant_plan_bytes(vae_hosted_files)
        te_prequant_bytes += _prequant_plan_bytes(vae_hosted_files)
        # The prequant replacing the dropped transformer/ shards is footprint nothing else reports.
        dit_prequant_bytes = int(dit_prequant[2]) if dit_prequant is not None else 0
        required_total += dit_prequant_bytes

        scoped_files: dict[str, list[str]] = {}
        scoped_gguf: dict[str, Optional[str]] = {}
        missing_checkpoints: set[str] = set()

        def add_missing_entry(
            repo: str,
            files: list[str],
            declared_sizes: dict[str, int],
            *,
            gguf: Optional[str] = None,
            checkpoint: bool = False,
        ) -> None:
            """Adds only files neither cache root resolves, probed at the plan's revision; one entry per repo, else two jobs fight over progress and cancel."""

            scope = scoped_files.setdefault(repo, [])
            scope.extend(name for name in files if name not in scope)
            scoped_gguf[repo] = scoped_gguf.get(repo) or gguf
            revision = revisions.get(repo)
            live = hub_cache_dir()

            # Adoption needs the same complete file set; the fast path drops a repo only if one root serves it.
            if self._files_already_cached(repo, files, revision, declared_sizes).issuperset(files):
                for entry in entries:
                    if entry["repo_id"] == repo:
                        entry["files"].extend(name for name in scope if name not in entry["files"])
                        entry["gguf_filename"] = scoped_gguf[repo]
                        break
                return

            def _where(name: str) -> Optional[str]:
                size = declared_sizes.get(name)
                if self._hub_file_is_cached(repo, name, revision, size, roots = (live,)):
                    return "live"
                if not self._hub_file_is_cached(repo, name, revision, size, roots = (None,)):
                    return None
                # reuse_other_cache_root only switches when the live lookup finds nothing, so a stale live copy wins.
                if self._hub_file_is_cached(repo, name, revision, None, roots = (live,)):
                    return None
                return "other"

            where = {name: _where(name) for name in files}
            # Files straddling two roots cannot be one snapshot, so stage the old-root subset into the live root; a file missing everywhere counts as live.
            split = {w or "live" for w in where.values()} == {"live", "other"}
            missing = [n for n, w in where.items() if w is None or (split and w != "live")]
            if not missing:
                return

            # Staged but not counted: the worker links unchanged files from an older snapshot (hub.utils.snapshot_reuse).
            reusable = self._reusable_from_older_snapshot(
                repo,
                [n for n in missing if where.get(n) is None],
                revision,
                declared_sizes,
                hf_token,
            )
            if checkpoint:
                missing_checkpoints.add(repo)
            for entry in entries:
                if entry["repo_id"] != repo:
                    continue
                newly_missing = [n for n in missing if n not in entry["files"]]
                added = [n for n in scope if n not in entry["files"]]
                entry["files"].extend(added)
                entry["bytes"] += int(
                    sum(declared_sizes.get(n, 0) for n in newly_missing if n not in reusable)
                )
                entry["gguf_filename"] = scoped_gguf[repo]
                entry["checkpoint"] = repo in missing_checkpoints
                return
            entries.append(
                {
                    "repo_id": repo,
                    # Stable list so a second pick adopts the same job; cached names are no-op downloads.
                    "files": list(scope),
                    "bytes": int(
                        sum(declared_sizes.get(name, 0) for name in missing if name not in reusable)
                    ),
                    "gguf_filename": scoped_gguf[repo],
                    "checkpoint": repo in missing_checkpoints,
                }
            )

        from .diffusion_te_prequant import te_prequant_unmirrored

        for repo, files in (*te_files.values(), *vae_hosted_files.values()):
            # A mirrored encoder still drops the dense shards above but is never staged: the Hub may not hold it.
            files = te_prequant_unmirrored(repo, files)
            if not files:
                continue
            add_missing_entry(
                repo,
                [name for name, _size in files],
                {name: int(size) for name, size in files},
            )
        if gguf_filename and not Path(repo_id).expanduser().exists():
            add_missing_entry(
                repo_id,
                [gguf_filename],
                file_sizes.get(repo_id, {gguf_filename: int(sizes.get(repo_id, 0))}),
                gguf = gguf_filename,
                checkpoint = True,
            )
        if base_files and not Path(base).expanduser().exists():
            # Staging must name the MIRROR: a gated upstream 401s anonymous users before the swap; status, saved configs and LoRA tags keep the vendor id.
            if fetch_base != base and fetch_base not in revisions:
                mirror_revision = self._current_sha(fetch_base, hf_token)
                if mirror_revision:
                    revisions[fetch_base] = mirror_revision
            add_missing_entry(
                fetch_base,
                base_files,
                file_sizes.get(base, {}),
                # Flag explicitly: the mirror swap above renamed this entry, so repo id comparison fails.
                checkpoint = kind == "pipeline",
            )
        if dit_prequant is not None:
            # Staged, not just counted: leaving it out means a multi-GB inline pull under the load lock.
            prequant_repo, prequant_file, prequant_size = dit_prequant
            add_missing_entry(prequant_repo, [prequant_file], {prequant_file: prequant_size})
        for ref in component_overrides.files.values() if component_overrides else ():
            if not ref.is_hub:
                continue
            ref_size = _hub_file_size(ref.repo_id, ref.filename, hf_token)
            required_total += ref_size
            add_missing_entry(ref.repo_id, [ref.filename], {ref.filename: ref_size})
        incompatible_code: Optional[str] = None
        if incompatible is None and allow_device_probe and memory_verdict:
            incompatible = self.declared_footprint_shortfall(
                fam,
                repo_id,
                base,
                kind = kind,
                declared_files = resident_file_sizes,
                # The artifact carries the whole denoiser share, already stored at the precision it loads at.
                prequant_bytes = te_prequant_bytes + dit_prequant_bytes
                if skip_transformer_weights
                else te_prequant_bytes,
                extra_bf16_bytes = _predownload_encoder_bf16_bytes(
                    fam, te_files, pipeline_declared = bool(resident_file_sizes)
                ),
                gguf_filename = gguf_filename,
                gpu_ordinal = load_kwargs.get("gpu_ordinal"),
                memory_mode = load_kwargs.get("memory_mode"),
                cpu_offload = bool(load_kwargs.get("cpu_offload")),
            )
            if incompatible is not None:
                incompatible_code = UNIFIED_OVERSIZE_REFUSAL_CODE
        return {
            "entries": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "required_bytes": int(required_total),
            "checkpoint_bytes": checkpoint_bytes,
            "incompatible_reason": incompatible,
            "incompatible_code": incompatible_code,
            # Companion discovery failed, so the file list above is partial. The UI stages what it can and the loader
            # pulls the rest inline; a caller that must not download at all has to treat this plan as unverified
            # rather than as nothing missing.
            "plan_failed": bool(plan_failures),
        }

    @staticmethod
    def _files_already_cached(
        repo_id: str,
        files: list[str],
        revision: Optional[str] = None,
        declared_sizes: Optional[dict[str, int]] = None,
    ) -> set[str]:
        """All files only if one cache root serves the whole pinned snapshot, not a union; no revision is no verdict, and never trust a stale main ref."""
        if not revision or not files:
            return set()
        try:
            from huggingface_hub import try_to_load_from_cache
            roots = (hub_cache_dir(), None)
        except Exception:  # noqa: BLE001 -- unreadable cache means stage, never fail the plan
            return set()
        wanted = set(files)

        def _hits(root: Optional[str]) -> set[str]:
            found: set[str] = set()
            for name in wanted:
                try:
                    value = try_to_load_from_cache(repo_id, name, cache_dir = root, revision = revision)
                except Exception:  # noqa: BLE001 -- keep checking the other files/root
                    continue
                if not isinstance(value, str):
                    continue
                path = Path(value)
                if not path.is_file():
                    continue
                expected = (declared_sizes or {}).get(name)
                if expected and expected > 0:
                    try:
                        if path.stat().st_size != expected:
                            continue
                    except OSError:
                        continue
                found.add(name)
            return found

        live = _hits(roots[0])
        if live == wanted:
            return wanted
        if live:
            return set()
        return wanted if _hits(roots[1]) == wanted else set()

    @staticmethod
    def _reusable_from_older_snapshot(
        repo_id: str,
        names: list[str],
        revision: Optional[str],
        declared_sizes: dict[str, int],
        hf_token: Optional[str],
    ) -> set[str]:
        """Missing files an older snapshot holds with the same content; never hashes on the request path."""
        if not names:
            return set()
        try:
            from hub.utils.snapshot_reuse import (
                cached_ref_commit,
                hub_remote_digests,
                reusable_paths,
            )

            # Unpinned entries were probed against refs/main; the worker fetches the Hub head.
            digest_revision = None if revision else "main"
            revision = revision or cached_ref_commit("model", repo_id, hub_cache_dir())
            if not revision:
                return set()
            return reusable_paths(
                "model",
                repo_id,
                revision,
                {name: int(declared_sizes.get(name) or 0) for name in names},
                hub_cache = hub_cache_dir(),
                # False is the managed-account anonymous sentinel; None would send the installation token.
                remote_digests = hub_remote_digests(
                    "model", repo_id, hf_token if hf_token is False else (hf_token or None)
                ),
                digest_revision = digest_revision,
            )
        except Exception:  # noqa: BLE001 -- counting the bytes is the conservative answer
            return set()

    @staticmethod
    def _current_sha(repo_id: str, hf_token: Optional[str]) -> Optional[str]:
        """The current Hub commit for a mirror repo, or None when metadata is unavailable."""
        try:
            from huggingface_hub import HfApi
            return (
                getattr(hub_reachability.model_info(HfApi(), repo_id, token = hf_token), "sha", None)
                or None
            )
        except Exception:  # noqa: BLE001 -- no SHA means conservative staging
            return None

    @staticmethod
    def _hub_file_is_loadable(
        repo_id: str,
        filename: str,
        revision: Optional[str] = None,
        expected_size: Optional[int] = None,
    ) -> bool:
        """A stale live copy can shadow a good one in the other root, so validity is checked, not
        presence."""
        live = hub_cache_dir()
        if DiffusionBackend._hub_file_is_cached(
            repo_id, filename, revision, expected_size, roots = (live,)
        ):
            return True
        if not DiffusionBackend._hub_file_is_cached(
            repo_id, filename, revision, expected_size, roots = (None,)
        ):
            return False
        return not DiffusionBackend._hub_file_is_cached(
            repo_id, filename, revision, None, roots = (live,)
        )

    @staticmethod
    def _hub_file_is_cached(
        repo_id: str,
        filename: str,
        revision: Optional[str] = None,
        expected_size: Optional[int] = None,
        roots: Optional[tuple[Optional[str], ...]] = None,
    ) -> bool:
        """Network-free; a pinned miss proves nothing; a hit must match the declared size, be a real file, and be the path the unpinned main ref resolves to."""
        try:
            from huggingface_hub import try_to_load_from_cache

            search = (hub_cache_dir(), None) if roots is None else roots

            def hit(root: Optional[str], rev: Optional[str]) -> Optional[str]:
                value = try_to_load_from_cache(repo_id, filename, cache_dir = root, revision = rev)
                return value if isinstance(value, str) and Path(value).is_file() else None

            def sound(hit: str) -> bool:
                """A hit must also match the declared size, since a truncated file can sit at the
                right path."""
                if not expected_size or expected_size <= 0:
                    return True  # nothing declared to check against; trust the ref
                try:
                    return Path(hit).stat().st_size == expected_size
                except OSError:
                    return False

            unpinned = [(root, value) for root in search if (value := hit(root, None))]
            if revision is None:
                return any(sound(value) for _root, value in unpinned)

            # Branch on presence, not soundness: a damaged snapshot must not bless an older same-size one.
            pinned = [(root, value) for root in search if (value := hit(root, revision))]
            if not pinned:
                return any(sound(value) for _root, value in unpinned)

            def snapshot_path(value: str) -> Path:
                # Do not resolve symlinks: blobs are shared; the snapshots/<sha> path encodes the choice.
                return Path(value).absolute()

            return any(
                pinned_root == main_root
                and sound(pinned_value)
                and sound(main_value)
                and snapshot_path(pinned_value) == snapshot_path(main_value)
                for pinned_root, pinned_value in pinned
                for main_root, main_value in unpinned
            )
        except Exception:  # noqa: BLE001 -- an unreadable cache is a miss, never a plan failure
            pass
        return False

    @staticmethod
    def _hub_cache_repo_dir(repo_id: str) -> Path:
        """Reads the live cache root: the import-time constant goes stale when the cache folder changes."""
        return Path(hub_cache_dir()) / f"models--{repo_id.replace('/', '--')}"

    @staticmethod
    def _hub_cache_repo_dirs(repo_id: str) -> list[Path]:
        """Every cache root repo_id can resolve through, live root first; the constant is read per call."""
        # Read the live root once: load_progress polls from another thread.
        live_root = hub_cache_dir()
        folder = f"models--{repo_id.replace('/', '--')}"
        live = Path(live_root) / folder
        try:
            from huggingface_hub import constants
            other = str(constants.HF_HUB_CACHE or "").strip()
        except Exception:  # noqa: BLE001 - no hub package: the live root still stands
            return [live]
        if not other:
            return [live]
        # Windows roots can differ by case alone.
        if os.path.normcase(os.path.normpath(other)) == os.path.normcase(
            os.path.normpath(live_root)
        ):
            return [live]
        return [live, Path(other) / folder]

    @staticmethod
    def _pipeline_source_complete(
        local_repo: Optional[Path], staged_repo: Optional[Path], cache_repo: Optional[str]
    ) -> bool:
        """True if a local, staged or live cached snapshot holds every manifest component."""
        trees = [p for p in (local_repo, staged_repo) if p is not None]
        if cache_repo:
            trees += [
                DiffusionBackend._live_snapshot_dir(d)
                for d in DiffusionBackend._hub_cache_repo_dirs(cache_repo)
            ]
        for tree in trees:
            if tree is None:
                continue
            try:
                if tree.is_dir() and local_pipeline_components_are_complete(
                    tree, "model_index.json"
                ):
                    return True
            except OSError:
                continue
        return False

    @staticmethod
    def _live_snapshot_dir(repo_dir: Path) -> Optional[Path]:
        """The one snapshot a load reads from this root, via refs/main; None means read the whole root."""
        try:
            rev = (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip()
        except (OSError, ValueError):
            return None
        # A ref file holds a bare commit hash; a separator would join out of the cache.
        if not rev or rev in (".", "..") or "/" in rev or "\\" in rev:
            return None
        snapshot = repo_dir / "snapshots" / rev
        return snapshot if snapshot.is_dir() else None

    @staticmethod
    def _cache_file_bytes(repo_id: str, filename: str) -> int:
        """Largest copy across roots, not the sum: the same file in two roots is one logical file."""
        best = 0
        for repo_dir in DiffusionBackend._hub_cache_repo_dirs(repo_id):
            snapshot = DiffusionBackend._live_snapshot_dir(repo_dir)
            if snapshot is None:
                continue
            try:
                best = max(best, (snapshot / filename).stat().st_size)
            except OSError:
                continue  # not in this root / broken symlink / unreadable
        return best

    @staticmethod
    def _incomplete_blob_sizes(repos) -> dict[str, int]:
        """``{path: size}`` of every ``blobs/*.incomplete`` of ``repos`` across cache roots. Never raises."""
        out: dict[str, int] = {}
        for repo in repos:
            for repo_dir in DiffusionBackend._hub_cache_repo_dirs(repo):
                try:
                    for entry in (repo_dir / "blobs").iterdir():
                        if entry.name.endswith(".incomplete"):
                            out[str(entry)] = entry.stat().st_size
                except OSError:
                    continue
        return out

    @staticmethod
    def _planned_progress_bytes(
        planned: Sequence[tuple[str, str, int]],
        baseline: Optional[dict[str, int]] = None,
        blob_keys: Optional[dict[tuple[str, str], str]] = None,
    ) -> int:
        """Bytes on disk of exactly the files ``_planned_progress_files`` chose. Each finished file counts up to its
        planned size (its largest copy across cache roots); a repo's ``*.incomplete`` blobs are the file it is
        writing right now, capped so the repo never counts more than its planned files. Sibling quants, sample
        renders, root checkpoints and other schemes already in the same repo are simply not in the list."""
        by_repo: dict[str, list[tuple[str, int]]] = {}
        for repo, name, size in planned:
            by_repo.setdefault(repo, []).append((name, int(size or 0)))
        total = 0
        for repo, files in by_repo.items():
            planned_total = sum(size for _n, size in files)
            done_by_file = {
                name: (min(size, DiffusionBackend._cache_file_bytes(repo, name)) if size else 0)
                for name, size in files
            }
            # A planned file's in-flight blob is ``<its blob key>.incomplete``: counted in full (resumed bytes are
            # this file's), up to what the file still lacks. Any other ``.incomplete`` belongs to some other download
            # and is ignored, unless a planned file's key is unknown: only then does growth past the load's baseline
            # stand in for it.
            remaining_by_key: dict[str, int] = {}
            unkeyed = False
            for name, size in files:
                key = (blob_keys or {}).get((repo.lower(), name))
                if key:
                    remaining_by_key[key] = remaining_by_key.get(key, 0) + max(
                        0, size - done_by_file[name]
                    )
                elif size:
                    unkeyed = True
            in_flight = 0
            for path, size in DiffusionBackend._incomplete_blob_sizes((repo,)).items():
                stem = Path(path).name[: -len(".incomplete")]
                if stem in remaining_by_key:
                    in_flight += min(size, remaining_by_key[stem])
                elif unkeyed:
                    in_flight += max(0, size - (baseline or {}).get(path, 0))
            total += min(planned_total, sum(done_by_file.values()) + in_flight)
        return total

    @staticmethod
    def _cache_bytes(
        repo_id: str,
        *,
        pipeline_only: bool = False,
        in_flight: bool = False,
    ) -> int:
        """Bytes of ``repo_id`` on disk across every cache root, for progress and the pipeline plan.

        Scoped per root to the revision that root serves (``refs/main``), because ``blobs/`` is
        append-only: a republished repo keeps the superseded revision's blobs forever under
        different etags, so summing the whole dir counts a stale full copy on top of the live
        partial one and ``load_progress`` reports finalizing through the rest of a multi-GB pull.
        The in-flight ``blobs/*.incomplete`` still counts: those are this download's own bytes.

        ``pipeline_only`` counts just the diffusers part of the snapshot (``model_index.json`` and the component
        folders), for a repo that also carries root GGUF / prequant checkpoints the pipeline never reads. Blobs
        that cannot be attributed to a path are left out there, unless ``in_flight`` asks for the
        ``*.incomplete`` ones: a load's progress counts what it is downloading right now.

        Keyed by blob filename, not summed per root: a blob is named after the file's etag, so a
        copy present in both roots counts once. Scanning only the live root reports 0 for a load a
        moved cache serves entirely off disk.
        """
        sizes: dict[str, int] = {}

        def _add(key: str, path: Path) -> None:
            try:
                size = path.stat().st_size
            except OSError:
                return  # broken symlink / unreadable
            sizes[key] = max(sizes.get(key, 0), size)

        for repo_dir in DiffusionBackend._hub_cache_repo_dirs(repo_id):
            snapshot = DiffusionBackend._live_snapshot_dir(repo_dir)
            try:
                blobs = list((repo_dir / "blobs").iterdir())
            except OSError:
                blobs = []  # repo not in this root yet / unreadable / no-symlink cache mode
            if snapshot is None:
                if not pipeline_only:
                    for entry in blobs:
                        _add(entry.name, entry)
                continue
            if not pipeline_only or in_flight:
                for entry in blobs:
                    if entry.name.endswith(".incomplete"):
                        _add(entry.name, entry)
            try:
                for f in snapshot.rglob("*"):
                    if not f.is_file():
                        continue
                    if pipeline_only:
                        rel = f.relative_to(snapshot).as_posix()
                        if "/" not in rel and rel != "model_index.json":
                            continue
                        # The same predicate the plan's listing drops these with, so the two cannot drift.
                        if _listing_extra(repo_id, rel):
                            continue
                    # Dedupe on the path INSIDE the snapshot, so one logical file counts once however many roots hold
                    # it. The etag would not: each root serves its own revision, so a republished file summed both
                    # copies and could report finalizing at half the real progress. Works for a no-symlink cache too,
                    # which stores the file in the snapshot rather than linking to blobs/.
                    _add(f"path:{f.relative_to(snapshot).as_posix()}", f)
            except OSError:
                pass  # the download is writing this tree under the walk; report what we counted
        return sum(sizes.values())

    @staticmethod
    def _local_dir_weight_sizes(
        path: Path,
        *,
        exclude_transformer: bool,
        load_dtype: Any = None,
    ) -> dict[str, int]:
        """Per-file weight sizes under a local diffusers dir, priced at load_dtype (default variant, safetensors over .bin, cast floats) for dedupe by path."""
        load_itemsize = _float_load_itemsize(load_dtype)
        sizes: dict[str, int] = {}
        for f in path.rglob("*"):
            if f.suffix.lower() not in (".safetensors", ".bin", ".pt", ".ckpt"):
                continue
            try:
                rel = f.relative_to(path)
            except ValueError:
                continue
            if exclude_transformer and rel.parts and rel.parts[0] == "transformer":
                continue
            # from_pretrained reads <component>/<file> only: a deeper tree (FLUX.3's variants/fp8r, variants/nvfp4
            # quantised packages) is never loaded and must not be priced.
            if load_itemsize is not None and (
                len(rel.parts) != 2
                or not (
                    _pipeline_default_variant_file(rel.as_posix()) or is_fp16_twin(rel.as_posix())
                )
            ):
                continue
            try:
                sizes[rel.as_posix()] = f.stat().st_size
            except OSError:
                continue
        if load_itemsize is None:
            return sizes
        # A twin beside a default would price the same tensors twice.
        default_dirs = {
            rel.rsplit("/", 1)[0]
            for rel in sizes
            if not is_fp16_twin(rel) and _pipeline_resident_weight(rel)
        }
        for rel in [r for r in sizes if is_fp16_twin(r) and r.rsplit("/", 1)[0] in default_dirs]:
            del sizes[rel]
        # A .bin only loses to a safetensors checkpoint the loader can select: unsharded, or shards with their index.
        st_dirs = {
            rel.rsplit("/", 1)[0]
            for rel in sizes
            if _SELECTABLE_SAFETENSORS_RE.match(rel.rsplit("/", 1)[-1])
        } | {
            index.parent.relative_to(path).as_posix()
            for index in path.glob("*/*.safetensors.index.json")
            if _DEFAULT_PIPELINE_WEIGHT_INDEX_RE.match(index.name)
        }
        for rel in list(sizes):
            if rel.endswith(".bin") and rel.rsplit("/", 1)[0] in st_dirs:
                del sizes[rel]
            elif rel.endswith((".safetensors", ".bin")):
                pinned = (
                    _component_pins_fp32((path / rel).parent, load_dtype)
                    if load_itemsize < 4
                    else ()
                )
                reader = (
                    DiffusionBackend._safetensors_cast_bytes
                    if rel.endswith(".safetensors")
                    else DiffusionBackend._bin_cast_bytes
                )
                cast = reader(path / rel, load_itemsize, pinned)
                if cast is not None:
                    sizes[rel] = cast
        return sizes

    @staticmethod
    def _safetensors_cast_bytes(
        path: Path,
        itemsize: int,
        pinned: tuple[str, ...] = (),
    ) -> Optional[int]:
        """A half load narrows only a file stored wider throughout (a mixed file pins its fp32 norms)."""
        from .diffusion_component_sizes import read_safetensors_header
        try:
            header = read_safetensors_header(path, max_bytes = 100_000_000)
            if header is None:
                return None
            tensors = []
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                dtype = str(meta.get("dtype", ""))
                if dtype not in _SAFETENSORS_FLOAT_WIDTH and dtype not in ("I64", "I32", "BOOL"):
                    return None
                begin, end = meta["data_offsets"]
                numel = 1
                for dim in meta.get("shape", []):
                    numel *= int(dim)
                tensors.append(
                    (name, _SAFETENSORS_FLOAT_WIDTH.get(dtype), numel, int(end) - int(begin))
                )
            return DiffusionBackend._cast_tensor_bytes(tensors, itemsize, pinned)
        except Exception:  # noqa: BLE001 - corrupt/crafted header keeps the on-disk size
            return None

    @staticmethod
    def _bin_cast_bytes(
        path: Path,
        itemsize: int,
        pinned: tuple[str, ...] = (),
    ) -> Optional[int]:
        try:
            import torch

            state = torch.load(path, map_location = "cpu", mmap = True, weights_only = True)
            if not isinstance(state, dict):
                return None
            tensors = []
            seen: set = set()
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    return None
                # Tied weights are saved once and load as one tensor: count an exact alias once.
                alias = (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                )
                if alias in seen:
                    continue
                seen.add(alias)
                width = int(tensor.element_size()) if tensor.is_floating_point() else None
                numel = int(tensor.numel())
                tensors.append((str(name), width, numel, numel * int(tensor.element_size())))
            del state
            return DiffusionBackend._cast_tensor_bytes(tensors, itemsize, pinned)
        except Exception:  # noqa: BLE001 - unreadable pickle keeps the on-disk size
            return None

    @staticmethod
    def _cast_tensor_bytes(
        tensors: list[tuple[str, Optional[int], int, int]],
        itemsize: int,
        pinned: tuple[str, ...] = (),
    ) -> Optional[int]:
        widths = [width for _, width, _, _ in tensors if width is not None]
        if not widths:
            return None
        narrows = min(widths) > itemsize
        # The loaders' own match: a pin names whole dot-separated segments, dotted pins included.
        pin_match = (
            re.compile("|".join(rf"(^|\.){re.escape(pin)}($|\.)" for pin in pinned)).search
            if pinned
            else None
        )
        total = 0
        for name, width, numel, stored in tensors:
            if width is None:
                total += stored
            elif itemsize >= 4 or (pin_match is not None and pin_match(name)):
                total += numel * 4
            elif narrows:
                total += numel * itemsize
            else:
                total += stored
        return total

    @staticmethod
    def _local_dir_weight_bytes(
        path: Path,
        *,
        exclude_transformer: bool,
        load_dtype: Any = None,
    ) -> int:
        """Only size signal for a local dir's auto memory plan, else a large model folds to zero; exclude_transformer drops transformer/ for GGUF/single-file only."""
        return sum(
            DiffusionBackend._local_dir_weight_sizes(
                path, exclude_transformer = exclude_transformer, load_dtype = load_dtype
            ).values()
        )

    @staticmethod
    def _union_over_cached_revs(
        base: str,
        fn: Callable[[Path], dict[str, int]],
        staged_dir: Optional[str] = None,
    ) -> int:
        """Unions per-file sizes over all cached revisions; the roots hold disjoint parts, so max
        under-counts."""
        local = Path(base).expanduser()
        if local.is_dir():
            return sum(fn(local).values())
        candidates: list[Path] = [Path(staged_dir).expanduser()] if staged_dir else []
        for repo_dir in DiffusionBackend._hub_cache_repo_dirs(base):
            # One revision per root, else repacked shards of an old revision double-count; a root naming none falls back to all, since over-counting beats reading zero.
            live_rev = DiffusionBackend._live_snapshot_dir(repo_dir)
            if live_rev is not None:
                candidates.append(live_rev)
                continue
            try:
                candidates.extend(rev for rev in (repo_dir / "snapshots").iterdir() if rev.is_dir())
            except OSError:
                continue  # a root with no copy of this repo contributes nothing
        merged: dict[str, int] = {}
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            for rel, count in fn(candidate).items():
                merged[rel] = max(merged.get(rel, 0), count)
        return sum(merged.values())

    @staticmethod
    def _supplied_component_mib(
        base: str, staged_dir: Optional[str], load_dtype: Any
    ) -> Optional[tuple[int, int, int, int]]:
        """``(scanned MiB, scanned TE MiB, supplied MiB, supplied TE MiB)``; scanned = what the base-repo walk
        counted for the replaced components."""
        from .diffusion_comfy_components import active_component_overrides, scanned_component_bytes

        overrides = active_component_overrides()
        if overrides is None or not overrides.paths:
            return None
        replaced = set(overrides.paths)
        replaced_te = {c for c in replaced if c.startswith("text_encoder")}

        def _scan(components: set) -> int:
            return DiffusionBackend._union_over_cached_revs(
                base,
                lambda d: scanned_component_bytes(
                    DiffusionBackend._local_dir_weight_sizes(
                        d, exclude_transformer = True, load_dtype = load_dtype
                    ),
                    components,
                ),
                staged_dir,
            )

        mib = 1024 * 1024
        resident = overrides.resident_mib(_float_load_itemsize(load_dtype) or 2)
        return (
            _scan(replaced) // mib,
            (_scan(replaced_te) // mib) if replaced_te else 0,
            int(sum(resident.values())),
            int(sum(v for c, v in resident.items() if c in replaced_te)),
        )

    @staticmethod
    def _companion_cache_bytes(
        base: str,
        staged_dir: Optional[str] = None,
        load_dtype: Any = None,
    ) -> int:
        """Companion size excluding transformer/, which the GGUF supplies; walks snapshots, not blobs/."""
        return DiffusionBackend._union_over_cached_revs(
            base,
            lambda d: DiffusionBackend._local_dir_weight_sizes(
                d, exclude_transformer = True, load_dtype = load_dtype
            ),
            staged_dir,
        )

    @staticmethod
    def _local_dir_text_encoder_sizes(path: Path, load_dtype: Any = None) -> dict[str, int]:
        """Text-encoder weight files, taken from the same walk so they stay a subset of the
        companion total."""
        return {
            rel: size
            for rel, size in DiffusionBackend._local_dir_weight_sizes(
                path, exclude_transformer = True, load_dtype = load_dtype
            ).items()
            # Prefix, not equality: families ship text_encoder, text_encoder_2, text_encoder_3.
            if rel.split("/", 1)[0].startswith("text_encoder")
        }

    @staticmethod
    def _text_encoder_cache_bytes(
        base: str,
        staged_dir: Optional[str] = None,
        load_dtype: Any = None,
    ) -> int:
        """Text-encoder share of the companion total, unioned over cache roots on the same relative
        paths."""
        return DiffusionBackend._union_over_cached_revs(
            base,
            lambda d: DiffusionBackend._local_dir_text_encoder_sizes(d, load_dtype),
            staged_dir,
        )

    @staticmethod
    def _precast_text_encoder_mib(
        fam: Any,
        base: str,
        target: Any,
        text_encoder_quant: Optional[str],
        staged_dir: Optional[str] = None,
        skip_components: tuple[str, ...] = (),
    ) -> Optional[tuple[int, tuple[str, ...], bool]]:
        """``(MiB, components, exact)`` for the hosted pre-cast text encoder(s) this pick loads, or None.

        Uncached checkpoints are priced from the family table (``exact=False``). Never raises."""
        try:
            from .diffusion_te_prequant import (
                TE_PREQUANT_BUDGET_SCALE,
                TE_PREQUANT_COMPONENTS,
                int8_convrot_te_runs_on,
                te_candidate_filenames,
                te_candidate_is_readable,
                te_int8_convrot_budget_scale,
                te_prequant_sources_for_base,
            )

            extra: dict[str, Any] = {}
            if getattr(fam, "name", None) == HIDREAM_FAMILY_NAME:
                # TE4 comes from its own repo (hidream_te4_kwargs), as in the download planner.
                extra["components"] = (*TE_PREQUANT_COMPONENTS, "text_encoder_4")
                extra["standalone_component_bases"] = {"text_encoder_4": HIDREAM_LLAMA_REPO}
            sources = te_prequant_sources_for_base(
                fam, base, te_quant_mode = text_encoder_quant, target = target, **extra
            )
            sources = {c: v for c, v in (sources or {}).items() if c not in skip_components}
            if not sources:
                return None
            from .diffusion_precision import te_quant_picked_automatically
            from .diffusion_te_prequant import te_prequant_sources_confirmed

            if te_quant_picked_automatically(
                text_encoder_quant
            ) and not te_prequant_sources_confirmed(sources):
                # An automatic pick without a confirmed file loads the dense encoder: no pre-cast saving to price.
                return None
            total = 0
            uncached = 0
            table_priced = 0
            for component, source in sources.items():
                size = 0
                kind = getattr(source, "kind", None)
                if kind == "path":
                    try:
                        size = Path(str(source.location)).expanduser().stat().st_size
                    except OSError:
                        size = 0
                elif kind == "repo":
                    from .diffusion_prequant import (
                        first_cached_as_resolved,
                        prefer_cached_pickle_twins,
                    )

                    repo = str(source.location)
                    sized: dict[str, int] = {}

                    def _size_of(name: str) -> int:
                        def _sizes(d: Path) -> dict[str, int]:
                            f = d / name
                            if f.is_file():
                                try:
                                    return {"precast": f.stat().st_size}
                                except OSError:
                                    return {}
                            return {}

                        if name not in sized:
                            sized[name] = DiffusionBackend._union_over_cached_revs(repo, _sizes)
                        return sized[name]

                    # The file the load opens: a cached older encoder does not make an uncached one ahead free.
                    names = [
                        n for n in te_candidate_filenames(source) if te_candidate_is_readable(n)
                    ]
                    hit = first_cached_as_resolved(
                        repo,
                        prefer_cached_pickle_twins(
                            repo,
                            names,
                            readable = te_candidate_is_readable,
                            cache_dir = hub_cache_dir(),
                            log = False,
                        ),
                        is_cached = lambda n: _size_of(n) > 0,
                        cache_dir = hub_cache_dir(),
                    )
                    size = _size_of(hit) if hit else 0
                if size > 0:
                    total += int(size)
                    continue
                uncached += 1
                if component == "text_encoder_4":
                    # Outside the table's encoder term: priced from its own dense size.
                    total += int(HIDREAM_LLAMA_BF16_BYTES * TE_PREQUANT_BUDGET_SCALE)
                else:
                    table_priced += 1
            if table_priced:
                table = family_bf16_components_gb(fam, base)
                if table is None:
                    return None
                # The table's encoder term holds HiDream's TE4 too, which is priced on its own above (cached or from
                # its dense size): out of both the share and the hosted bytes.
                te4_in_table = "text_encoder_4" in sources
                share = table_priced / max(1, len(sources) - (1 if te4_in_table else 0))
                # MPS and an automatic int8 pick only take the int8 ConvRot file, smaller than the fp8 one.
                from .diffusion_te_prequant import auto_int8_takes_file_only

                scale = (
                    te_int8_convrot_budget_scale(fam)
                    if int8_convrot_te_runs_on(target)
                    or auto_int8_takes_file_only(text_encoder_quant)
                    else TE_PREQUANT_BUDGET_SCALE
                )
                from .diffusion_te_prequant import _TE_DENSE_SIBLING_GB

                # Only the hosted encoders: a sibling no file replaces (CLIP-L, ByT5) is priced by its own dense
                # shards, not as part of the pre-cast file.
                hosted_gb = max(
                    0.0,
                    float(table[1])
                    - _TE_DENSE_SIBLING_GB.get(str(getattr(fam, "name", "") or ""), 0.0)
                    - (HIDREAM_LLAMA_BF16_BYTES / 1000.0**3 if te4_in_table else 0.0),
                )
                total += int(hosted_gb * (1000.0**3) * scale * share)
            if total <= 0:
                return None
            return max(1, total // (1024 * 1024)), tuple(sources), uncached == 0
        except Exception:  # noqa: BLE001 -- sizing aid only; the scanned terms stand
            return None

    @staticmethod
    def _safetensors_param_count(path: Path) -> int:
        """Total tensor elements in a safetensors file, read from its JSON header without touching
        the tensor data. 0 on any read/parse failure."""
        from .diffusion_component_sizes import read_safetensors_header
        try:
            header = read_safetensors_header(path)
            if header is None:
                return 0
            total = 0
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                numel = 1
                for dim in meta.get("shape", []):
                    numel *= dim
                total += numel
            return total
        except Exception:  # noqa: BLE001 - corrupt/crafted shard degrades to 0, never crashes the load
            return 0

    @staticmethod
    def _dense_transformer_resident_bytes(base: str, staged_dir: Optional[str] = None) -> int:
        """bf16 resident size (num_params times 2), not on-disk bytes (2x for F32); 0 when no shards, which skips the fit check and can OOM the dense build."""

        def _params(d: Path) -> dict[str, int]:
            tdir = d / "transformer"
            if not tdir.is_dir():
                return {}
            return {
                f"transformer/{s.name}": DiffusionBackend._safetensors_param_count(s)
                for s in tdir.glob("*.safetensors")
            }

        return DiffusionBackend._union_over_cached_revs(base, _params, staged_dir) * 2

    @_invalidates_gpu_memory("diffusion load")
    @scoped_allow_oversized_load
    @rocm_probe_load_scoped
    @_account_owned_load
    @_plans_at_requested_speed
    @scoped_local_files_only
    @_with_supplied_components
    @vram_budget_scope("image", load = True)
    def load_pipeline(
        self,
        repo_id: str,
        *,
        display_repo_id: Optional[str] = None,
        local_files_only: bool = False,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
        gpu_ordinal: Optional[int] = None,
        # Read by ``scoped_allow_oversized_load``.
        allow_oversized_load: bool = False,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
        # Only when True may the dense-quant fallback materialise transformer/ shards; default for direct calls.
        _transformer_prefetched: bool = True,
        # Repo the background prefetch staged companions from; direct calls re-derive it below.
        _fetch_base: Optional[str] = None,
        # The scheme the plan settled, or PIPELINE_SEED_DECLINED; None for a direct call, which the pull never scoped.
        _pipeline_prequant_planned: Optional[str] = None,
        _pipeline_prequant_skipped: tuple[str, ...] = (),
        _prequant_unreachable: tuple[str, ...] = (),
        _te_prequant_resolved: bool = True,
        _hosted_vae_planned: bool = False,
        # begin_load's ``normalize_token`` of the request token: False stays the anonymous sentinel.
        _hub_token: Any = None,
    ) -> dict[str, Any]:
        with self._load_cancel_lock:
            if _load_token is None:
                _load_token = self._load_token
            self._raise_if_load_cancelled(_load_token)
        hub_token = _hub_token if _hub_token is not None else normalize_token(hf_token)
        # A blank token must degrade to anonymous, not be passed as a credential. Normalize once.
        hf_token = hf_token.strip() if isinstance(hf_token, str) else hf_token
        hf_token = hf_token or None

        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            family_override = family_override,
            model_kind = model_kind,
        )
        _refuse_lora_without_loader(fam, loras)
        kind = resolve_model_kind(gguf_filename, model_kind)
        # Validate every mode string that can raise BEFORE this load evicts the previous pipeline.
        normalize_transformer_quant(transformer_quant)
        normalize_speed_mode(speed_mode)
        normalize_attention_backend(attention_backend)
        normalize_transformer_cache(transformer_cache)
        # Text-encoder tri-state resolved once so every consumer agrees; must run before the load evicts the previous pipeline, since it validates the raw request.
        text_encoder_quant_requested = text_encoder_quant
        from .diffusion_comfy_components import active_component_overrides

        discovered_components = active_component_overrides()
        if discovered_components is not None and discovered_components.discovered:
            text_encoder_quant_requested = discovered_components.te_quant_request
        text_encoder_quant, text_encoder_quant_auto = resolve_te_quant_request(
            text_encoder_quant, getattr(fam, "te_quant_auto", None)
        )
        if text_encoder_quant_auto:
            logger.info(
                "diffusion.text_encoder_quant: '%s' selected automatically for %s (unset request)",
                text_encoder_quant,
                fam.name,
            )
        # A full pipeline is its own base; single-file kinds resolve the companion base repo.
        base = (
            repo_id
            if kind == "pipeline"
            else _resolve_base_repo(repo_id, base_repo, fam, hf_token, gguf_filename)
        )
        # ``base`` is the upstream id for reports; ``fetch_base`` is the only id handed to downloads.
        fetch_base = _fetch_base or prefer_ungated_mirror(base, hf_token)
        target = self._target_for_ordinal(fam, gpu_ordinal)
        # Dedicated worker thread: the pin is permanent, so placement, offload and state.device share a card.
        apply_diffusion_device_ordinal(target)
        device, dtype = target.device, target.dtype

        # Must run before the first import of diffusers (pulls in torch._dynamo); later consumers would see a module poisoned by a concurrent import race.
        try:
            from utils.torch_warmup import close_dynamo_import_window
            close_dynamo_import_window(logger)
        except Exception as exc:  # noqa: BLE001 - optimisation only
            logger.debug("dynamo pre-import skipped: %r", exc)

        import diffusers

        # diffusers hard-codes _tqdm_active = True and ignores env vars; silence it so it cannot corrupt logs.
        from loggers.config import quiet_third_party_progress_bars

        quiet_third_party_progress_bars()

        # Pre-install the optional attention kernel before the load locks: the pip install can block unload for 600s.
        try:
            preinstall_backend = select_attention_backend(
                target, attention_backend, speed_active = True
            )
            if preinstall_backend is not None:
                _ensure_attention_backend_installed(preinstall_backend, logger)
        except Exception:  # noqa: BLE001 - the locked path re-resolves and validates
            pass
        # Install FlashInfer only when a pre-quantised checkpoint (incl. a ComfyUI nvfp4 file) will load: on-the-fly is torchao.
        if (
            kind == "single_file"
            and nvfp4_diffusion_enabled()
            and comfy_nvfp4_runtime_possible(getattr(fam, "name", None))
            and not _has_active_lora(loras)
            and self._comfy_single_file_holds_nvfp4(repo_id, gguf_filename)
        ) or (
            dense_quant_supported_kind(kind)
            and nvfp4_diffusion_enabled()
            and TQ_NVFP4
            in (
                normalize_transformer_quant(transformer_quant),
                _pipeline_prequant_planned,
            )
            # A declined seed loads the released denoiser (torchao or no quant), so no checkpoint for FlashInfer.
            and _pipeline_prequant_planned != PIPELINE_SEED_DECLINED
            and (
                _pipeline_prequant_planned == TQ_NVFP4
                or self._nvfp4_checkpoint_will_load(
                    fam,
                    base,
                    transformer_prequant_path,
                    hf_token,
                    local_files_only = local_files_only,
                    loras = loras,
                )
            )
            # The locked re-plan (after teardown) may drop the capacity-settled seed; ask it now with teardown memory credited.
            and not (
                kind == "pipeline"
                and _pipeline_prequant_planned == TQ_NVFP4
                and not self._seed_plan_stays_resident(
                    TQ_NVFP4,
                    target,
                    base,
                    fam,
                    memory_mode,
                    cpu_offload,
                    repo_id = repo_id,
                    base_local_dir = _base_local_dir,
                    fetch_base = fetch_base,
                    text_encoder_quant = text_encoder_quant,
                )
            )
        ):
            from .diffusion_nvfp4_install import ensure_flashinfer_for_nvfp4
            _nvfp4_install_outcome = ensure_flashinfer_for_nvfp4(
                device, logger = logger, local_files_only = local_files_only
            )
        else:
            _nvfp4_install_outcome = None

        with self._lock:
            self._raise_if_load_cancelled(_load_token)
            with self._generation_cancel_lock:
                if self._active_generate_cancel is not None:
                    self._active_generate_cancel.set()
            self._reserve_teardown_locked()
        with self._model_transition_slot():
            # Per load: a previous load's pre-quant failure must never reach this one's status.
            self._prequant_fallback_note = None
            with self._lock:
                try:
                    self._raise_if_load_cancelled(_load_token)

                    self._unload_locked()
                finally:
                    self._release_teardown_locked()
                # Bind even when the gate skipped, so no stale install reason survives.
                from .diffusion_nvfp4_install import record_install_reason

                record_install_reason(self, *(_nvfp4_install_outcome or (True, None)), device)
                # Drop the previous load's NVFP4 cache tag; an NVFP4 load sets it again.
                try:
                    from .diffusion_nvfp4_act_quant import reset_compile_cache_tag
                    reset_compile_cache_tag()
                except Exception as exc:  # noqa: BLE001 - keying only
                    logger.debug("nvfp4 cache tag reset skipped: %r", exc)

                # Single-file kinds resolve a checkpoint path; the pipeline kind has none.
                single_file_path = (
                    self._resolve_gguf_path(
                        repo_id, gguf_filename, hf_token, local_files_only = local_files_only
                    )
                    if kind in ("gguf", "single_file")
                    else None
                )
                # A renamed or hand-picked FLUX.2 GGUF can still land on a different-size base, and no name-based rule
                # catches that. Say so here, naming the file and the repo, rather than letting the GGUF quantizer
                # raise a bare shape mismatch.
                if kind == "single_file" and not _is_trusted_diffusion_repo(repo_id):
                    # Admitted per file: prove it is a safetensors container before any probe or loader opens it.
                    assert_safetensors_file(single_file_path)
                assert_flux2_gguf_matches_base(fam, base, single_file_path)
                assert_flux3_checkpoint_matches_family(fam, single_file_path)
                # A ComfyUI-quantized file loads through its own path below; a format it cannot run is refused here,
                # from the header, before planning or reading a weight, rather than loaded with its scales dropped.
                comfy_scan = refuse_comfy_quant(single_file_path) if kind == "single_file" else None
                # torchao weights from a ComfyUI file: compile like Studio's own quantized transformer
                comfy_compile = False
                transformer_cls = _family_model_class(diffusers, fam.transformer_class)
                pipeline_cls = _family_model_class(diffusers, fam.pipeline_class)

                # Decide placement up front (weights still on CPU). Budgets the GGUF; dense is preflighted below.
                plan = self._plan_memory(
                    target,
                    single_file_path,
                    base,
                    fam,
                    memory_mode,
                    cpu_offload,
                    kind = kind,
                    lora = _has_active_lora(loras),
                    repo_id = repo_id,
                    # The base may resolve from the OTHER cache root, which live-root scans read as zero companions.
                    base_local_dir = _base_local_dir,
                    fetch_base = fetch_base,
                    text_encoder_quant = text_encoder_quant,
                )
                pipeline_seed_scheme: Optional[str] = None
                bf16_pipeline_plan = plan
                if kind == "pipeline" and _pipeline_prequant_planned not in (
                    None,
                    PIPELINE_SEED_DECLINED,
                ):
                    pipeline_seed_scheme = _pipeline_prequant_planned
                    seeded_plan = self._seeded_pipeline_plan(
                        pipeline_seed_scheme,
                        target,
                        base,
                        fam,
                        memory_mode,
                        cpu_offload,
                        repo_id = repo_id,
                        base_local_dir = _base_local_dir,
                        fetch_base = fetch_base,
                        text_encoder_quant = text_encoder_quant,
                    )
                    seeded_placement = (
                        None
                        if seeded_plan is None
                        else torchao_offload_plan(seeded_plan, pipeline_seed_scheme)
                    )
                    if seeded_placement is None:
                        # Live free memory can undercut the CAPACITY the plan settled this against.
                        logger.info(
                            "diffusion.denoiser_prequant: an artifact-sized plan for %s places the "
                            "denoiser where its torchao weights do not survive on this card, so the "
                            "released denoiser is loaded instead",
                            pipeline_seed_scheme,
                        )
                        pipeline_seed_scheme = None
                    else:
                        plan = seeded_placement
                # Unified memory: none is a placement, not a fit; refuse a shortfall after eviction, before weights load, judged on the resident table total, not shard bytes.
                raise_on_unified_memory_shortfall(
                    self._resident_sized_plan(
                        plan, fam, base, target, kind, text_encoder_quant = text_encoder_quant
                    ),
                    family = getattr(fam, "name", None),
                    logger = logger,
                )

                # Raw caller request, saved before the tri-state rewrite so status still shows a declined fp8 ask.
                transformer_quant_requested = transformer_quant
                # Dtype tri-state: unset/"auto" -> hardware ladder; "none"/"off" -> GGUF as-is; explicit pins.
                transformer_quant_auto = transformer_quant is None or str(
                    transformer_quant
                ).strip().lower() in ("", "auto")
                if transformer_quant_auto:
                    # An explicit Speed="off" stays GGUF-as-is: auto-quant would break the bit-exact request.
                    speed_off = (
                        speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
                    )
                    transformer_quant = "off" if speed_off else TQ_AUTO
                    if kind == "gguf" and fam.single_file_is_pipeline:
                        # A whole-pipeline GGUF (SDXL) has no dense twin in the base repo: its UNet runs as stored.
                        if _has_active_lora(loras):
                            raise ValueError(_WHOLE_PIPELINE_GGUF_LORA_MSG)
                        transformer_quant = "off"
                # The one case that must fail closed: a named scheme (not auto, not off). Normalized here so "FP8" and
                # "fp8" refuse identically; a bogus value already raised above.
                transformer_quant_pinned = (
                    None
                    if transformer_quant_auto
                    else normalize_transformer_quant(transformer_quant_requested)
                )

                pipe = transformer = None
                pipe_kwargs: dict[str, Any] = {}
                transformer_quant_engaged = None
                transformer_quant_artifact: Optional[str] = None
                quant_plan = None
                # Why the dense quant did not engage, in caller terms; shown in `resolved` and the refusal message.
                transformer_quant_decline: Optional[str] = None
                small_dit_kept_bf16 = False
                transformer_quant_decline_status = RESOLVED_FELL_BACK
                # A GGUF is excluded while it bakes LoRAs: only the dense build can carry them.
                if (
                    transformer_quant == TQ_AUTO
                    and pipeline_seed_scheme is None
                    and dense_quant_supported_kind(kind)
                    and dense_transformer_supported(target)
                    and not (kind == "gguf" and _has_active_lora(loras))
                    and _auto_keeps_bf16_reason(fam, kind, base) is not None
                    and (
                        eager_reason := _auto_quant_eager_reason(
                            fam,
                            eager_plan := self._bf16_resident_plan(
                                plan,
                                target,
                                fam,
                                base,
                                memory_mode,
                                cpu_offload,
                                kind = kind,
                                repo_id = repo_id,
                                fetch_base = fetch_base,
                                base_local_dir = _base_local_dir,
                                text_encoder_quant = text_encoder_quant
                                if _te_prequant_resolved and not local_files_only
                                else None,
                            ),
                            transformer_prequant_path,
                            kind,
                            base,
                        )
                    )
                    is not None
                ):
                    logger.info(
                        "diffusion.transformer_quant: auto keeps %s (%s)", kind, eager_reason
                    )
                    transformer_quant = "off"
                    transformer_quant_decline = eager_reason
                    # Place the released weights by the plan that proved they fit, not by fp32 shard bytes.
                    plan = eager_plan
                    small_dit_kept_bf16 = (
                        auto_bf16_when_resident_reason(getattr(fam, "name", None))
                        == SMALL_DIT_BF16_WHEN_RESIDENT
                    )
                native_scheme = (
                    native_quant_scheme(
                        target, transformer_quant_pinned, family = getattr(fam, "name", None)
                    )
                    if kind == "pipeline"
                    else None
                )
                native_offload_scheme = (
                    native_quant_scheme(
                        target,
                        transformer_quant_pinned,
                        family = getattr(fam, "name", None),
                        offload = True,
                    )
                    if kind == "pipeline" and native_scheme is None
                    else None
                )
                native_offload_forced = native_offload_scheme is not None and (
                    _memory_request_forces_offload(memory_mode, cpu_offload)
                )
                # The only build the native cache can serve back (MPS / ROCm set native_scheme up front).
                native_cache_ok = (
                    native_offload_forced
                    and native_scheme is None
                    and getattr(target, "device", None) == "cuda"
                )
                if transformer_quant_pinned is not None and not dense_quant_supported_kind(kind):
                    transformer_quant_decline = dense_quant_unsupported_kind_reason(kind)
                    transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                elif (
                    transformer_quant_pinned is not None
                    and native_scheme is None
                    and not dense_transformer_supported(target)
                ):
                    # The helper knows truer reasons on ROCm and the Windows torchao stub.
                    transformer_quant_decline = dense_transformer_unsupported_reason(target)
                    transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                elif (
                    transformer_quant_pinned is not None
                    and native_scheme is None
                    and not native_offload_forced
                    and _planned_quant_scheme(
                        fam,
                        target,
                        transformer_quant_pinned,
                        base_repo = base,
                        prequant_path = transformer_prequant_path,
                    )
                    is None
                ):
                    transformer_quant_decline = explain_unusable_scheme(
                        getattr(fam, "name", None),
                        transformer_quant_pinned,
                        base_repo = base,
                        prequant_missing = normalize_transformer_quant(transformer_quant_pinned)
                        == TQ_NVFP4
                        and usable_prequant_source(
                            fam,
                            TQ_NVFP4,
                            path_override = transformer_prequant_path,
                            base_repo = base,
                            local_files_only = local_files_only,
                            hf_token = hf_token,
                            cache_dir = hub_cache_dir(),
                        )
                        is None,
                    )
                    transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                # GGUF size can mis-budget the fast path: preflight footprint; dense_declined set on auto decline.
                dense_declined = False
                # No dense fallback for a prequant-sized plan, so a failed prequant load raises; or if transformer/ was not prefetched, which would pull it inline.
                dense_fallback_allowed = bool(_transformer_prefetched)
                # 2509 / original Edit GGUFs resolve to the 2511 base, whose transformer/ is another model.
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and transformer_variant_differs_from_base(
                        fam,
                        base,
                        gguf_filename,
                        content_variant_hint(repo_id, gguf_filename),
                        repo_id,
                        display_repo_id,
                    )
                ):
                    dense_declined = True
                    if transformer_quant_decline is None:
                        transformer_quant_decline = (
                            f"{base} has a different transformer than this GGUF, so only the GGUF "
                            "itself can run it"
                        )
                    # Baking adapters needs the dense build; staying on auto lets the load fail loudly below.
                    if transformer_quant_pinned is None and not _has_active_lora(loras):
                        transformer_quant = "off"
                    else:
                        transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                # Set when an offloading GGUF pick loads the pre-quantised checkpoint instead (diffusion_gguf_route).
                gguf_offload_placement: Optional[Any] = None
                gguf_offload_scheme: Optional[str] = None
                gguf_offload_swap = False
                # Auto on GGUF: a hosted pre-quant is free only if cached, else a second multi-GB denoiser the GGUF never uses; LoRA bakes and explicit schemes are exempt.
                baking_loras = _has_active_lora(loras)
                # Only while a scheme is enabled: Speed=off GGUF is deliberate, not a decline.
                if (
                    kind == "gguf"
                    and transformer_quant_auto
                    and not baking_loras
                    and normalize_transformer_quant(transformer_quant) is not None
                    # Request-named offload skips the dense build: plan omits transformer/ by policy, not missing cache.
                    and not _memory_request_forces_offload(memory_mode, cpu_offload)
                ):
                    uncached_prequant = _uncached_prequant_repo(
                        fam,
                        target,
                        transformer_quant,
                        base_repo = base,
                        prequant_path = transformer_prequant_path,
                        online = False if local_files_only else None,
                    )
                    if uncached_prequant is not None:
                        logger.info(
                            "diffusion.transformer_quant_declined: pre-quant checkpoint in %s is "
                            "not cached (an auto quant never downloads a second transformer for a "
                            "GGUF pick); loading the GGUF",
                            uncached_prequant,
                        )
                        dense_declined = True
                        transformer_quant_decline = (
                            f"the hosted pre-quant checkpoint in {uncached_prequant} is not "
                            "cached, and an auto quant never downloads a second transformer for "
                            "a GGUF pick"
                        )
                    # Dense path would pull un-prefetched transformer/ shards under the load lock, so decline; not for a cached prequant, which stages none by design.
                    elif (
                        not _transformer_prefetched
                        and dense_transformer_supported(target)
                        and _planned_quant_scheme(
                            fam,
                            target,
                            transformer_quant,
                            base_repo = base,
                            prequant_path = transformer_prequant_path,
                        )
                        is not None
                        and not _dense_candidate_is_prequant(
                            fam,
                            target,
                            transformer_quant,
                            base_repo = base,
                            prequant_path = transformer_prequant_path,
                        )
                        and self._auto_prequant_retry_scheme(
                            target,
                            fam,
                            transformer_quant,
                            _planned_quant_scheme(
                                fam,
                                target,
                                transformer_quant,
                                base_repo = base,
                                prequant_path = transformer_prequant_path,
                            ),
                            base_repo = base,
                            path_override = transformer_prequant_path,
                            loras = loras,
                        )
                        is None
                    ):
                        logger.info(
                            "diffusion.transformer_quant_declined: %s transformer/ shards are not "
                            "staged (an auto quant never downloads a second transformer for a "
                            "GGUF pick); loading the GGUF",
                            base,
                        )
                        dense_declined = True
                        transformer_quant_decline = (
                            f"{base} transformer/ shards are not staged, and an auto quant never "
                            "downloads a second transformer for a GGUF pick"
                        )
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                    and not dense_declined
                ):
                    if plan.offload_policy != OFFLOAD_NONE:
                        candidate = resolve_dense_quant_candidate(
                            fam = fam,
                            target = target,
                            requested = transformer_quant,
                            base_repo = base,
                            prequant_path = transformer_prequant_path,
                            force_dense = _has_active_lora(loras),
                            logger = logger,
                        )
                        if (
                            candidate is not None
                            and candidate.prequant
                            and getattr(candidate, "scheme", None) in _prequant_unreachable
                        ):
                            candidate = resolve_dense_quant_candidate(
                                fam = fam,
                                target = target,
                                requested = transformer_quant,
                                base_repo = base,
                                prequant_path = transformer_prequant_path,
                                force_dense = True,
                                logger = logger,
                            )

                        # Defined outside the check: the retry rebinds candidate and calls this, so nesting raises UnboundLocalError under the load lock.
                        def _replan_candidate():
                            return self._plan_memory(
                                target,
                                single_file_path,
                                base,
                                fam,
                                memory_mode,
                                cpu_offload,
                                kind = kind,
                                repo_id = repo_id,
                                fetch_base = fetch_base,
                                transformer_resident_override_mib = (
                                    candidate.transient_transformer_mib
                                ),
                                # Companions skip prefetched shards (no double count); encoder share keeps streamed tiers priceable.
                                **self._candidate_companion_overrides(
                                    candidate, fam, base, target, text_encoder_quant
                                ),
                                text_encoder_quant = text_encoder_quant,
                            )

                        if candidate is None:
                            # No basis to re-plan, so the GGUF plan's offload stands.
                            transformer_quant_decline = (
                                f"the memory plan picked '{plan.offload_policy}' offload and there "
                                "is no quantised candidate to re-plan against on this host (see "
                                "the server log)"
                            )
                        if candidate is not None:
                            replanned = _replan_candidate()
                            if (
                                not plan_keeps_transformer_resident(replanned)
                                # Explicit balanced/low_vram picks offload by mode, so a fresh snapshot cannot change it.
                                and normalize_memory_mode(memory_mode)
                                not in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM)
                                and plan_fits_total_capacity(replanned)
                            ):
                                # Fits total capacity but not live free memory: re-snapshot and replan once.
                                replanned = _replan_candidate()
                            if not plan_keeps_transformer_resident(replanned):
                                logger.info(
                                    "diffusion.transformer_quant_declined: required=%s MiB "
                                    "budget=%s MiB free=%s MiB policy=%s (%s)",
                                    replanned.estimates.get("resident_required_mib"),
                                    replanned.estimates.get("safe_device_budget_mib"),
                                    getattr(replanned.device_memory, "free_mib", None),
                                    replanned.offload_policy,
                                    "; ".join(replanned.reasons),
                                )
                                transformer_quant_decline = _torchao_offload_decline(
                                    "the quantised build still needs", replanned
                                )
                                # Held until the resident retry below had its turn.
                                gguf_offload_placement, gguf_offload_reason = (
                                    self._gguf_offload_prequant_placement(
                                        replanned,
                                        candidate,
                                        fam = fam,
                                        target = target,
                                        auto = transformer_quant_auto,
                                        gguf_filename = gguf_filename,
                                        base = base,
                                        prequant_path = transformer_prequant_path,
                                    )
                                )
                                # The retry below may rebind ``candidate``; the placement is this one's.
                                gguf_offload_scheme = getattr(candidate, "scheme", None)
                                if gguf_offload_reason is not None:
                                    transformer_quant_decline = gguf_offload_reason
                            if plan_keeps_transformer_resident(replanned):
                                quant_plan = replanned
                                # The GGUF plan declined resident; a prequant-sized replan says nothing about the dense build.
                                if candidate.prequant:
                                    dense_fallback_allowed = False
                        if quant_plan is None:
                            # Auto winner has no resident option: try a lower rung, else the fast path is skipped.
                            retry = self._auto_prequant_retry_scheme(
                                target,
                                fam,
                                transformer_quant,
                                _planned_quant_scheme(
                                    fam,
                                    target,
                                    transformer_quant,
                                    base_repo = base,
                                    prequant_path = transformer_prequant_path,
                                ),
                                base_repo = base,
                                path_override = transformer_prequant_path,
                                loras = loras,
                            )
                            retry_candidate = (
                                resolve_dense_quant_candidate(
                                    fam = fam,
                                    target = target,
                                    requested = retry,
                                    base_repo = base,
                                    prequant_path = transformer_prequant_path,
                                    force_dense = _has_active_lora(loras),
                                    logger = logger,
                                )
                                if retry is not None
                                else None
                            )
                            if retry_candidate is not None:
                                candidate = retry_candidate
                                retry_plan = _replan_candidate()
                                if plan_keeps_transformer_resident(retry_plan):
                                    logger.info(
                                        "diffusion.transformer_quant: auto's pick does not fit "
                                        "resident; retrying at %s, whose checkpoint is cached",
                                        retry,
                                    )
                                    transformer_quant = retry
                                    quant_plan = retry_plan
                                    if candidate.prequant:
                                        dense_fallback_allowed = False
                        if quant_plan is None and gguf_offload_placement is not None:
                            logger.info(
                                "diffusion.transformer_quant: the GGUF pick offloads ('%s'); loading the "
                                "pre-quantised %s checkpoint under the same placement instead",
                                gguf_offload_placement.offload_policy,
                                gguf_offload_scheme,
                            )
                            quant_plan = gguf_offload_placement
                            gguf_offload_swap = True
                            transformer_quant_decline = None
                            # A dense bf16 fallback would land the whole denoiser on a card it does not fit.
                            dense_fallback_allowed = False
                    else:
                        # Dense bf16 build: re-check the fit before eviction rather than OOM after; skipped for a prequant.
                        scheme = _planned_quant_scheme(
                            fam,
                            target,
                            transformer_quant,  # normalized above
                            base_repo = base,
                            prequant_path = transformer_prequant_path,
                        )
                        # usable_prequant_source (not resolve_): a missing or non-allowlisted local path must not skip the dense-fit re-check.
                        prequant = (
                            # A LoRA bake skips the prequant shortcut, so gate the fast path as if no prequant existed
                            None
                            # Nor one this user cannot fetch: the loader would build dense, so the check must too
                            if _has_active_lora(loras) or scheme in _prequant_unreachable
                            else usable_prequant_source(
                                fam,
                                scheme,
                                path_override = transformer_prequant_path,
                                base_repo = base,
                                local_files_only = local_files_only,
                                hf_token = hf_token,
                                cache_dir = hub_cache_dir(),
                            )
                            if scheme is not None
                            else None
                        )
                        # Read where the prefetch wrote: a base on the import-time root sizes 0 and skips this check.
                        dense_mib = int(
                            self._dense_transformer_resident_bytes(fetch_base, _base_local_dir)
                            // (1024 * 1024)
                        )
                        dense_possible = True
                        # Why dense bf16 is out, in caller terms; the load's decline reason only if no auto rung fits.
                        dense_decline_reason: Optional[str] = None
                        if dense_mib > 0:
                            dense_plan = self._plan_memory(
                                target,
                                single_file_path,
                                base,
                                fam,
                                memory_mode,
                                cpu_offload,
                                kind = kind,
                                repo_id = repo_id,
                                fetch_base = fetch_base,
                                transformer_resident_override_mib = dense_mib,
                                # No companion_override, so this reads cache: point it at the snapshot the load reads, not live root.
                                base_local_dir = _base_local_dir,
                                text_encoder_quant = text_encoder_quant,
                            )
                            # Unified memory: the planner says none for any size, so size the dense build explicitly and decline to the packed GGUF rather than raise.
                            unified_shortfall = unified_memory_shortfall_message(
                                dense_plan, family = getattr(fam, "name", None)
                            )
                            if unified_shortfall is not None:
                                dense_possible = False
                                dense_fallback_allowed = False
                                dense_decline_reason = (
                                    "the dense bf16 transformer this quant is built from does not "
                                    f"fit this device's unified memory ({dense_mib} MiB)"
                                )
                            elif dense_plan.offload_policy != OFFLOAD_NONE:
                                dense_possible = False
                                dense_fallback_allowed = False
                                dense_decline_reason = (
                                    "the dense bf16 transformer this quant is built from does not "
                                    f"fit resident ({dense_mib} MiB; the plan picked "
                                    f"'{dense_plan.offload_policy}' offload)"
                                )
                        elif not _transformer_prefetched:
                            # 0 with nothing staged means the capacity gate declined transformer/, so dense cannot run.
                            dense_possible = False
                            dense_decline_reason = (
                                "the base transformer this quant is built from was not staged (the "
                                "prefetch's capacity gate declined its shards), so the dense bf16 "
                                "build cannot run on this host"
                            )
                        # Unified memory: a hosted prequant can outweigh the pool, so size the artifact and decline to the GGUF when it does not fit.
                        if (
                            prequant is not None
                            and getattr(plan.device_memory, "memory_kind", None) == "unified_memory"
                        ):
                            prequant_candidate = resolve_dense_quant_candidate(
                                fam = fam,
                                target = target,
                                requested = transformer_quant,
                                base_repo = base,
                                prequant_path = transformer_prequant_path,
                                force_dense = _has_active_lora(loras),
                                logger = logger,
                            )
                            if (
                                prequant_candidate is not None
                                and unified_memory_shortfall_message(
                                    self._plan_memory(
                                        target,
                                        single_file_path,
                                        base,
                                        fam,
                                        memory_mode,
                                        cpu_offload,
                                        kind = kind,
                                        repo_id = repo_id,
                                        fetch_base = fetch_base,
                                        transformer_resident_override_mib = (
                                            prequant_candidate.transient_transformer_mib
                                        ),
                                        # Price the pre-cast encoder share, as the load-level plan does.
                                        companion_override_mib = (
                                            self._precast_scaled_companions_mib(
                                                prequant_candidate,
                                                fam,
                                                base,
                                                target,
                                                text_encoder_quant,
                                            )
                                        ),
                                    ),
                                    family = getattr(fam, "name", None),
                                )
                                is not None
                            ):
                                dense_declined = True
                                dense_possible = False
                                dense_fallback_allowed = False
                                transformer_quant_decline = (
                                    "the pre-quantised transformer does not fit this device's "
                                    "unified memory, where offloading it would free nothing, so "
                                    "the packed GGUF was loaded instead"
                                )
                                logger.info(
                                    "diffusion.transformer_quant_declined: the pre-quantised "
                                    "transformer (%s MiB) does not fit unified memory; "
                                    "loading the GGUF",
                                    prequant_candidate.transient_transformer_mib,
                                )
                        # The INT8 artifact outweighs the GGUF, so plan it on its own; a plan that streams the transformer cannot carry torchao.
                        if (
                            prequant is not None
                            and not dense_declined
                            and getattr(plan.device_memory, "memory_kind", None) != "unified_memory"
                        ):
                            sized_candidate = resolve_dense_quant_candidate(
                                fam = fam,
                                target = target,
                                requested = transformer_quant,
                                base_repo = base,
                                prequant_path = transformer_prequant_path,
                                force_dense = False,
                                logger = logger,
                            )
                            if sized_candidate is not None and sized_candidate.prequant:
                                sized_plan = self._plan_memory(
                                    target,
                                    single_file_path,
                                    base,
                                    fam,
                                    memory_mode,
                                    cpu_offload,
                                    kind = kind,
                                    repo_id = repo_id,
                                    fetch_base = fetch_base,
                                    transformer_resident_override_mib = (
                                        sized_candidate.transient_transformer_mib
                                    ),
                                    **self._candidate_companion_overrides(
                                        sized_candidate, fam, base, target, text_encoder_quant
                                    ),
                                    text_encoder_quant = text_encoder_quant,
                                )
                                sized_placement, sized_reason = (
                                    (None, None)
                                    if plan_keeps_transformer_resident(sized_plan)
                                    else self._gguf_offload_prequant_placement(
                                        sized_plan,
                                        sized_candidate,
                                        fam = fam,
                                        target = target,
                                        auto = transformer_quant_auto,
                                        gguf_filename = gguf_filename,
                                        base = base,
                                        prequant_path = transformer_prequant_path,
                                    )
                                )
                                if plan_keeps_transformer_resident(sized_plan):
                                    quant_plan = sized_plan
                                    if sized_plan.offload_policy != OFFLOAD_NONE:
                                        dense_fallback_allowed = False
                                elif sized_placement is not None:
                                    logger.info(
                                        "diffusion.transformer_quant: the pre-quantised %s checkpoint "
                                        "offloads ('%s'); loading it under that placement instead of the GGUF",
                                        sized_candidate.scheme,
                                        sized_placement.offload_policy,
                                    )
                                    quant_plan = sized_placement
                                    gguf_offload_swap = True
                                    dense_fallback_allowed = False
                                else:
                                    dense_declined = True
                                    dense_fallback_allowed = False
                                    transformer_quant_decline = (
                                        sized_reason
                                        or _torchao_offload_decline(
                                            "the pre-quantised transformer needs", sized_plan
                                        )
                                    )
                                    logger.info(
                                        "diffusion.transformer_quant_declined: %s",
                                        transformer_quant_decline,
                                    )
                        # No prequant for the winner: auto may try a lower rung with a hosted checkpoint; an explicit scheme is never swapped.
                        if not dense_possible and prequant is None:
                            retry = self._auto_prequant_retry_scheme(
                                target,
                                fam,
                                transformer_quant,
                                scheme,
                                base_repo = base,
                                path_override = transformer_prequant_path,
                                loras = loras,
                            )
                            # An int8 checkpoint can outweigh the Q4 GGUF, so size the retried rung and replan.
                            retry_candidate = (
                                resolve_dense_quant_candidate(
                                    fam = fam,
                                    target = target,
                                    requested = retry,
                                    base_repo = base,
                                    prequant_path = transformer_prequant_path,
                                    force_dense = _has_active_lora(loras),
                                    logger = logger,
                                )
                                if retry is not None
                                else None
                            )
                            retry_plan = (
                                self._plan_memory(
                                    target,
                                    single_file_path,
                                    base,
                                    fam,
                                    memory_mode,
                                    cpu_offload,
                                    kind = kind,
                                    repo_id = repo_id,
                                    fetch_base = fetch_base,
                                    transformer_resident_override_mib = (
                                        retry_candidate.transient_transformer_mib
                                    ),
                                    # Same pre-cast encoder pricing as the fit check above.
                                    **self._candidate_companion_overrides(
                                        retry_candidate,
                                        fam,
                                        base,
                                        target,
                                        text_encoder_quant,
                                    ),
                                )
                                if retry_candidate is not None
                                else None
                            )
                            if (
                                retry_plan is not None
                                and plan_keeps_transformer_resident(retry_plan)
                                # Same unified caveat: 'none' is not a fit here, so size the retried rung explicitly before pinning.
                                and unified_memory_shortfall_message(
                                    retry_plan, family = getattr(fam, "name", None)
                                )
                                is None
                            ):
                                logger.info(
                                    "diffusion.transformer_quant: %s has no prequant and cannot "
                                    "build dense; retrying auto at %s, whose checkpoint is cached "
                                    "and plans resident",
                                    scheme,
                                    retry,
                                )
                                transformer_quant = retry
                                quant_plan = retry_plan
                                dense_fallback_allowed = False
                            else:
                                # Reason surfaces as the 409 detail or the Auto: OFF badge, so name which half failed.
                                dense_declined = True
                                transformer_quant_decline = (
                                    f"{dense_decline_reason}, and no hosted pre-quantised "
                                    "checkpoint is available to build it from"
                                    if dense_decline_reason
                                    else "no hosted pre-quantised checkpoint is available and the "
                                    "dense bf16 transformer cannot be built on this host"
                                )
                backend_flags_before = None
                eager_patched = False
                load_audit = None
                compile_ctx = None
                state_committed = False
                load_bg_compile = None
                state = None
                pending_cache_write = None
                pending_cast_write = None
                native_streamed = False
                try:
                    self._raise_if_load_cancelled(_load_token)
                    if (
                        kind == "gguf"
                        and normalize_transformer_quant(transformer_quant) is not None
                        and dense_transformer_supported(target)
                        and not dense_declined
                        and (plan.offload_policy == OFFLOAD_NONE or quant_plan is not None)
                    ):
                        try:
                            pipe, transformer_quant_engaged = self._load_dense_quant_pipeline(
                                transformer_cls,
                                pipeline_cls,
                                base,
                                device,
                                dtype,
                                hf_token,
                                target,
                                transformer_quant,
                                transformer_quant_fast_accum,
                                fam = fam,
                                base_local_dir = _base_local_dir,
                                prequant_path = transformer_prequant_path,
                                allow_dense_fallback = dense_fallback_allowed,
                                lora_specs = loras,
                                text_encoder_quant = text_encoder_quant,
                                fetch_base = fetch_base,
                                local_files_only = local_files_only,
                                _load_token = _load_token,
                                seed_plan = quant_plan if quant_plan is not None else plan,
                            )
                            transformer_quant_artifact = self._gguf_route_artifact(
                                pipe,
                                fam,
                                transformer_quant_engaged,
                                base = base,
                                prequant_path = transformer_prequant_path,
                            )
                        except Exception as exc:  # noqa: BLE001 - fall back to the GGUF build
                            self._raise_if_load_cancelled(_load_token)
                            logger.warning(
                                "diffusion.transformer_quant_fallback: %s (loading GGUF)", exc
                            )
                            pipe = None
                            transformer_quant_engaged = None
                            # Reaches the client (tooltip/refusal), so format before cleanup and redact.
                            from utils.native_path_leases import redact_native_paths

                            transformer_quant_decline = redact_native_paths(
                                f"the quantised transformer build failed ({exc})"
                            )
                            _clear_exception_frames(exc)
                            # The NVFP4 caches hold VIEWS of the failed transformer, which clear_gpu_cache() cannot free.
                            try:
                                from .diffusion_nvfp4_linear import reset_nvfp4_state
                                reset_nvfp4_state()
                            except Exception:  # noqa: BLE001 - cleanup is best effort
                                pass
                            try:
                                clear_gpu_cache()
                            except Exception:  # noqa: BLE001
                                pass
                    if transformer_quant_engaged is not None and quant_plan is not None:
                        # The engaged dense build uses the re-planned placement; the GGUF-size plan stays for fallback.
                        plan = quant_plan

                    if (
                        pipe is None
                        and kind == "gguf"
                        and normalize_transformer_quant(transformer_quant) is not None
                        and _has_active_lora(loras)
                    ):
                        # Baked adapters cannot ride the GGUF fallback; fail loudly.
                        raise RuntimeError(
                            "The requested LoRA adapters could not be applied: baking adapters "
                            "requires the quantized (int8/fp8) transformer build, which was "
                            "declined or failed on this device (see the server log), and the "
                            "GGUF fallback cannot carry them. Retry without transformer_quant "
                            "adapters, free VRAM, or pick a smaller model."
                        )

                    # GGUF reaches its verdict here; pipeline quantisation is attempted after assembly.
                    def _refuse_pinned_precision() -> None:
                        raise RuntimeError(
                            precision_refusal_message(
                                "transformer_quant",
                                transformer_quant_pinned,
                                transformer_quant_decline
                                or "the quantised transformer build did not engage on this host",
                                off_label = "Off to run the checkpoint as-is",
                            )
                        )

                    if (
                        transformer_quant_pinned is not None
                        and not precision_fallback_allowed()
                        # Refuse known failures before downloading a pipeline.
                        and (
                            transformer_quant_decline is not None
                            or (pipe is None and kind == "gguf")
                        )
                    ):
                        _refuse_pinned_precision()

                    if pipe is None:
                        if kind == "pipeline":
                            if fam.name == IDEOGRAM4_FAMILY_NAME:
                                # ideogram ships the same transformers-5.x Qwen stack as krea; assemble per-component too,
                                # from the mirror for the same reason. A hosted pre-cast fp8 encoder replaces the dense
                                # vendor-fp8 one (whose shards the download plan then leaves out), as on krea.
                                ideogram_te = te_prequant_pipe_kwargs(
                                    fam,
                                    fetch_base,
                                    te_quant_mode = text_encoder_quant,
                                    target = target,
                                    dtype = dtype,
                                    hf_token = hf_token,
                                    logger = logger,
                                    local_files_only = local_files_only,
                                    # assembly reads fetch_base (never the staged dir), like krea
                                    dense_source = fetch_base if local_files_only else None,
                                ).get("text_encoder")
                                self._raise_if_load_cancelled(_load_token)
                                pipe = load_ideogram4_pipeline(
                                    fetch_base,
                                    dtype,
                                    hf_token = hf_token,
                                    check_cancelled = lambda: self._raise_if_load_cancelled(
                                        _load_token
                                    ),
                                    text_encoder = ideogram_te,
                                )
                            else:
                                pipe_kwargs = {
                                    "local_files_only": local_files_only,
                                    "torch_dtype": dtype,
                                    "cache_dir": hub_cache_dir(),
                                }
                                if hf_token:
                                    pipe_kwargs["token"] = hf_token
                                # fp16 load with a twin for every component reads the twins (bit-identical after cast).
                                _te_replaced = te_precast_components(
                                    fam, fetch_base, text_encoder_quant, target
                                )
                                _variant_snapshot = _load_snapshot_dir(
                                    fetch_base, _base_local_dir, hub_cache_dir()
                                )
                                _fp16_variant_eligible = not (
                                    pipeline_seed_scheme is not None
                                    or _pipeline_prequant_skipped
                                    or is_flux3_family(fam)
                                    or fam.name in (MOSS_IMAGE_FAMILY_NAME, KREA2_FAMILY_NAME)
                                )
                                _fp16_variant = (
                                    local_fp16_variant(
                                        _variant_snapshot,
                                        load_dtype = dtype,
                                        supplied = _te_replaced,
                                    )
                                    if _fp16_variant_eligible
                                    else None
                                )
                                _cast_cands = _image_cast_candidates(
                                    fam,
                                    _base_local_dir or fetch_base,
                                    dtype = dtype,
                                    base = base,
                                    skip = tuple(_te_replaced or ()),
                                )
                                # Stopped once the pipeline is built.
                                self._stop_load_prefetch()
                                self._load_prefetch = start_load_prefetch(
                                    fam,
                                    _base_local_dir or fetch_base,
                                    prequant_scheme = pipeline_seed_scheme,
                                    prequant_path_override = transformer_prequant_path,
                                    prequant_base_repo = base,
                                    text_encoders_replaced = tuple(_te_replaced or ())
                                    + tuple(_image_cast_cached(_cast_cands, _fp16_variant)),
                                    cache_dir = hub_cache_dir(),
                                    logger = logger,
                                    variant = _fp16_variant,
                                )
                                if fam.name == HIDREAM_FAMILY_NAME:
                                    # The repo names a text_encoder_4 it does not ship; supply it from the open mirror.
                                    pipe_kwargs.update(
                                        hidream_te4_kwargs(
                                            dtype,
                                            hf_token,
                                            fam = fam,
                                            te_quant_mode = text_encoder_quant,
                                            target = target,
                                            local_files_only = local_files_only,
                                        )
                                    )
                                    self._raise_if_load_cancelled(_load_token)
                                # A hosted pre-cast fp8 text encoder skips the dense TE download; the cast re-applies idempotently.
                                pipe_kwargs.update(
                                    te_prequant_pipe_kwargs(
                                        fam,
                                        fetch_base,
                                        te_quant_mode = text_encoder_quant,
                                        target = target,
                                        dtype = dtype,
                                        hf_token = hf_token,
                                        logger = logger,
                                        local_files_only = local_files_only,
                                        # krea assembles from fetch_base below, never the staged dir
                                        dense_source = (
                                            None
                                            if fam.name == KREA2_FAMILY_NAME
                                            else _base_local_dir
                                        )
                                        or (fetch_base if local_files_only else None),
                                        # the config from the dir assembly reads, the id for checkpoint compatibility
                                        config_source = (
                                            None
                                            if fam.name == KREA2_FAMILY_NAME
                                            else _base_local_dir
                                        ),
                                    )
                                )
                                # A miss reads vae/, fetched here when the plan dropped it.
                                if "vae" not in pipe_kwargs:
                                    pipe_kwargs.update(
                                        hosted_vae_pipe_kwargs(
                                            fam,
                                            fetch_base,
                                            kind = kind,
                                            dtype = dtype,
                                            hf_token = hf_token,
                                            local_files_only = local_files_only,
                                            config_source = _base_local_dir,
                                            cache_dir = hub_cache_dir(),
                                            dense_staged = not _hosted_vae_planned,
                                            logger = logger,
                                        )
                                    )
                                if pipeline_seed_scheme is not None:
                                    from .diffusion_denoiser_prequant import (
                                        denoiser_prequant_pipe_kwargs,
                                    )

                                    seeded = denoiser_prequant_pipe_kwargs(
                                        fam,
                                        fetch_base,
                                        scheme = pipeline_seed_scheme,
                                        dtype = dtype,
                                        device = device,
                                        # offloading plans seed it on the host: a GPU seed OOMed 8 GB cards
                                        placement_device = prequant_seed_device(
                                            plan, device, pipeline_seed_scheme
                                        ),
                                        hf_token = hf_token,
                                        target = target,
                                        path_override = transformer_prequant_path,
                                        fast_accum = transformer_quant_fast_accum,
                                        local_files_only = local_files_only,
                                        cache_dir = hub_cache_dir(),
                                        logger = logger,
                                    )
                                    pipe_kwargs.update(seeded)
                                    if seeded:
                                        transformer_quant_engaged = pipeline_seed_scheme
                                        transformer_quant_artifact = prequant_artifact_label(
                                            denoiser_prequant_source(
                                                fam,
                                                pipeline_seed_scheme,
                                                base_repo = base,
                                                path_override = transformer_prequant_path,
                                                local_files_only = local_files_only,
                                                hf_token = hf_token,
                                                cache_dir = hub_cache_dir(),
                                            ),
                                            next(iter(seeded.values()), None),
                                        )
                                    else:
                                        # The plan was priced on the seed landing, so re-plan at bf16 without it.
                                        from utils.native_path_leases import redact_native_paths

                                        from .diffusion_prequant import last_prequant_failure

                                        failure = last_prequant_failure()
                                        self._prequant_fallback_note = (
                                            redact_native_paths(failure)
                                            if failure
                                            else "the checkpoint was unavailable"
                                        )
                                        logger.warning(
                                            "diffusion.denoiser_prequant: no pre-quantized denoiser was "
                                            "seeded (%s); re-planning memory at the released bf16 size",
                                            self._prequant_fallback_note,
                                        )
                                        pipeline_seed_scheme = None
                                        plan = bf16_pipeline_plan
                                        raise_on_unified_memory_shortfall(
                                            self._resident_sized_plan(
                                                plan,
                                                fam,
                                                base,
                                                target,
                                                kind,
                                                text_encoder_quant = text_encoder_quant,
                                            ),
                                            family = getattr(fam, "name", None),
                                            logger = logger,
                                        )
                                if (
                                    pipeline_seed_scheme is None
                                    and native_cache_ok
                                    and not _has_active_lora(loras)
                                    and not _pipeline_prequant_skipped
                                    and "transformer" not in pipe_kwargs
                                ):
                                    native_cached = self._native_quant_from_cache(
                                        fam,
                                        transformer_cls,
                                        native_offload_scheme,
                                        target,
                                        base = base,
                                        source_base = _base_local_dir or fetch_base,
                                        config_kwargs = {
                                            "local_files_only": local_files_only,
                                            "cache_dir": hub_cache_dir(),
                                            **({"token": hf_token} if hf_token else {}),
                                        },
                                        cancel_check = lambda: self._raise_if_load_cancelled(
                                            _load_token
                                        ),
                                    )
                                    if native_cached is None:
                                        # Miss: build per tensor so the dense module is never resident beside the codes.
                                        native_cached = self._native_quant_build_streamed(
                                            fam,
                                            transformer_cls,
                                            native_offload_scheme,
                                            target,
                                            dtype,
                                            source_base = _base_local_dir or fetch_base,
                                            config_kwargs = {
                                                "local_files_only": local_files_only,
                                                "cache_dir": hub_cache_dir(),
                                                **({"token": hf_token} if hf_token else {}),
                                            },
                                            cancel_check = lambda: self._raise_if_load_cancelled(
                                                _load_token
                                            ),
                                        )
                                        native_streamed = native_cached is not None
                                    if native_cached is not None:
                                        pipe_kwargs["transformer"] = native_cached
                                        native_scheme = native_offload_scheme
                                        transformer_quant_engaged = native_offload_scheme
                                if pipeline_seed_scheme is None and _pipeline_prequant_skipped:
                                    # from_pretrained cannot re-fetch these from a local snapshot dir, so top them up here.
                                    _base_local_dir = self._restore_skipped_transformer_shards(
                                        repo_id,
                                        base,
                                        _pipeline_prequant_skipped,
                                        hf_token,
                                        fetch_base = fetch_base,
                                        base_local_dir = _base_local_dir,
                                        local_files_only = local_files_only,
                                        manifest = _base_manifest_name(fam),
                                    )
                                    # Plan saw companions only; re-plan once transformer/ is cached, else 'none' keeps bf16 resident.
                                    plan = self._plan_memory(
                                        target,
                                        single_file_path,
                                        base,
                                        fam,
                                        memory_mode,
                                        cpu_offload,
                                        kind = kind,
                                        lora = _has_active_lora(loras),
                                        repo_id = repo_id,
                                        base_local_dir = _base_local_dir,
                                        fetch_base = fetch_base,
                                        text_encoder_quant = text_encoder_quant,
                                    )
                                    bf16_pipeline_plan = plan
                                self._raise_if_load_cancelled(_load_token)
                                small_host = None
                                if is_flux3_family(fam):
                                    # FLUX.3 ships modular blocks only: assemble per component behind the classic
                                    # face (diffusion_flux3), from the staged dir when there is one.
                                    try:
                                        pipe = load_flux3_pipeline(
                                            _base_local_dir or fetch_base,
                                            dtype,
                                            hf_token = hf_token,
                                            check_cancelled = lambda: self._raise_if_load_cancelled(
                                                _load_token
                                            ),
                                            local_files_only = local_files_only,
                                            text_encoder = pipe_kwargs.get("text_encoder"),
                                            transformer = pipe_kwargs.get("transformer"),
                                            family_name = fam.name,
                                        )
                                    finally:
                                        self._stop_load_prefetch()
                                elif fam.name == MOSS_IMAGE_FAMILY_NAME:
                                    from .diffusion_moss_image import load_moss_image_pipeline
                                    try:
                                        pipe = load_moss_image_pipeline(
                                            _base_local_dir or fetch_base,
                                            dtype,
                                            hf_token = hf_token,
                                            check_cancelled = lambda: self._raise_if_load_cancelled(
                                                _load_token
                                            ),
                                            local_files_only = local_files_only,
                                            text_encoder = pipe_kwargs.get("text_encoder"),
                                            transformer = pipe_kwargs.get("transformer"),
                                        )
                                    finally:
                                        self._stop_load_prefetch()
                                elif fam.name == KREA2_FAMILY_NAME:
                                    # krea's transformers-5.x configs break the 4.x line: assemble per component from
                                    # pipe_kwargs (pre-cast TE, seeded denoiser or None = dense shards).
                                    try:
                                        pipe = load_krea2_pipeline(
                                            fetch_base,
                                            dtype,
                                            hf_token = hf_token,
                                            check_cancelled = lambda: self._raise_if_load_cancelled(
                                                _load_token
                                            ),
                                            local_files_only = local_files_only,
                                            text_encoder = pipe_kwargs.get("text_encoder"),
                                            transformer = pipe_kwargs.get("transformer"),
                                        )
                                    finally:
                                        self._stop_load_prefetch()
                                else:
                                    # bf16 -> fp16 conversion materialises in host RAM (35 GB for FLUX.1)
                                    small_host = self._small_host_decision(
                                        _base_local_dir or fetch_base,
                                        pipe_kwargs,
                                        target,
                                        dtype,
                                        lora_active = _has_active_lora(loras),
                                    )
                                    if small_host is not None and small_host.engaged:
                                        pipe_kwargs["torch_dtype"] = small_host_torch_dtype_map(
                                            small_host, dtype
                                        )
                                    if _fp16_variant_eligible:
                                        # Final kwargs: a dtype map or a supplied component (a default-only cached
                                        # file it replaces blocked the first check) changes what twins must cover.
                                        _fp16_variant = local_fp16_variant(
                                            _variant_snapshot,
                                            load_dtype = dtype,
                                            torch_dtype = pipe_kwargs.get("torch_dtype"),
                                            supplied = tuple(pipe_kwargs),
                                        )
                                    if _fp16_variant is not None:
                                        pipe_kwargs["variant"] = _fp16_variant
                                        logger.info(
                                            "diffusion.fp16_variant: reading the .%s weight files (%s load)",
                                            _fp16_variant,
                                            str(dtype).replace("torch.", ""),
                                        )
                                    _cast_missed = []
                                    if (
                                        _cast_cands
                                        and _fp16_variant is None
                                        and not (small_host is not None and small_host.engaged)
                                    ):
                                        from .diffusion_cast_cache import inject as _cast_inject

                                        # Twins (fp16 variant) and the small-host dtype map read other bytes.
                                        _cast_missed = _cast_inject(
                                            pipe_kwargs,
                                            _cast_cands,
                                            logger = logger,
                                            cancel_check = lambda: self._raise_if_load_cancelled(
                                                _load_token
                                            ),
                                            label = "diffusion",
                                        )
                                    # The prefetched snapshot dir keeps from_pretrained off the hub (24 GB per FLUX.1
                                    # otherwise)
                                    try:
                                        with _converted_shards_pread(dtype, small_host):
                                            pipe = pipeline_cls.from_pretrained(
                                                _base_local_dir or fetch_base, **pipe_kwargs
                                            )
                                    finally:
                                        self._stop_load_prefetch()
                                    if _cast_missed:
                                        from .diffusion_cast_cache import defer_writes

                                        # Names only: the write reads the live tensors once the load is published.
                                        pending_cast_write = defer_writes(
                                            pipe, _cast_missed, logger = logger
                                        )
                                    if fam.single_file_is_pipeline:
                                        # A v-pred SDXL repo can ship an epsilon scheduler config beside the root
                                        # single file that declares v_pred / ztsnr (NoobAI-XL V-Pred 1.0).
                                        # A staged snapshot omits that root file, so its header is read off the Hub.
                                        apply_pipeline_prediction_type(
                                            pipe,
                                            _base_local_dir or fetch_base,
                                            logger,
                                            repo_id = fetch_base,
                                            # Not hf_token: that collapsed False (anonymous) to None (ambient).
                                            hf_token = hub_token,
                                            allow_network = not local_files_only,
                                        )
                                if small_host is not None and small_host.engaged:
                                    plan = self._apply_small_host_route(
                                        pipe,
                                        plan,
                                        small_host,
                                        target,
                                        dtype,
                                        fam,
                                        logger,
                                        _load_token,
                                    )
                        elif kind in ("single_file", "gguf") and fam.single_file_is_pipeline:
                            # A single-file SDXL-style checkpoint is the WHOLE pipeline: load it through the pipeline
                            # class with ``config`` on the base repo. Its GGUF (validated whole) loads the same way.
                            sf_pipe_kwargs: dict[str, Any] = {
                                "local_files_only": local_files_only,
                                "torch_dtype": dtype,
                                # ``config`` is a repo fetch ahead of the mirrored load, so a gated id would 401 here first.
                                "config": fetch_base,
                                "cache_dir": hub_cache_dir(),
                            }
                            if hf_token:
                                sf_pipe_kwargs["token"] = hf_token
                            if comfy_scan is not None:
                                raise ValueError(
                                    "A ComfyUI-quantized checkpoint holds only a denoiser; this family's single "
                                    "file is a whole pipeline, so it cannot be loaded here."
                                )
                            if kind == "gguf":
                                pipe = load_whole_pipeline_gguf(
                                    pipeline_cls,
                                    getattr(diffusers, fam.transformer_class),
                                    single_file_path,
                                    sf_pipe_kwargs,
                                    dtype = dtype,
                                    denoiser_attr = fam.denoiser_attr,
                                    logger = logger,
                                )
                                _dequantize_gguf_outside_linears(
                                    getattr(pipe, fam.denoiser_attr), dtype, logger
                                )
                            else:
                                pipe = pipeline_cls.from_single_file(
                                    single_file_path, **sf_pipe_kwargs
                                )
                                # The base repo's scheduler is epsilon; a v-pred checkpoint says so only in its header.
                                apply_checkpoint_prediction_type(pipe, single_file_path, logger)
                        else:
                            # Transformer-only single file; VAE/text-encoder/scheduler come from the base repo.
                            sf_kwargs: dict[str, Any] = {
                                "torch_dtype": dtype,
                                # FLUX.3 reads the config from the staged companions, as its assembly below does: the
                                # validated base holds transformer/config.json, and fetch_base would re-resolve it.
                                "config": (_base_local_dir or fetch_base)
                                if is_flux3_family(fam)
                                else fetch_base,
                                "subfolder": "transformer",
                                "token": hf_token,
                                "cache_dir": hub_cache_dir(),
                                # config is a REPO ID, and diffusers forwards this into the load_config() that resolves
                                # it, so without the flag this branch reaches the Hub on a load nobody asked for. The
                                # pipeline assembly below was already guarded; this call was not.
                                "local_files_only": local_files_only
                                or _cached_while_hub_unreachable(
                                    fetch_base, "transformer/config.json"
                                ),
                                # config is the family base's (2511); 2509 / original Edit lack its zero_cond_t
                                **transformer_config_overrides_for(
                                    fam, gguf_filename, repo_id, display_repo_id, base
                                ),
                            }
                            # Before the prefix shim, which cannot wrap an unregistered class.
                            _register_unregistered_single_file_classes(logger)
                            if kind == "gguf":
                                # Dequantise the GGUF transformer on-device at the compute dtype.
                                sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                                    compute_dtype = dtype
                                )
                                # sd.cpp GGUFs prefix tensors with model.diffusion_model.; the converters reject it.
                                _install_gguf_prefix_strip(transformer_cls, logger)
                                _install_gguf_dim_restore(logger)
                            if comfy_scan is not None:
                                # int8 / fp8 codes go to Studio's runtime unchanged where it runs; the rest dequantize.
                                _comfy_offload = self._comfy_offload_for(plan, single_file_path)
                                transformer = load_comfy_quant_transformer(
                                    transformer_cls,
                                    single_file_path,
                                    comfy_scan,
                                    sf_kwargs,
                                    int8_backend = comfy_int8_backend(
                                        target,
                                        fam.name,
                                        base,
                                        offload = _comfy_offload,
                                    ),
                                    fp8_backend = comfy_fp8_backend(
                                        target,
                                        fam.name,
                                        base,
                                        offload = _comfy_offload,
                                    ),
                                    **comfy_block_backends(
                                        comfy_scan,
                                        target,
                                        fam.name,
                                        dtype = dtype,
                                        logger = logger,
                                        lora = _has_active_lora(loras),
                                    ),
                                    family = fam.name,
                                    target = target,
                                    fast_accum = transformer_quant_fast_accum,
                                    logger = logger,
                                    fp8_storage = self._fp8_storage_for(
                                        single_file_path, target, loras
                                    ),
                                    fp8_widen_chosen = self._fp8_widen_for(
                                        single_file_path, target, loras
                                    ),
                                )
                                comfy_compile = comfy_torchao_quantized(transformer)
                            else:
                                if kind != "gguf" and (
                                    not hasattr(transformer_cls, "from_single_file")
                                    # diffusers 0.41's from_single_file rounds Krea 2's fp32 norms to bf16.
                                    or getattr(transformer_cls, "__name__", "")
                                    == "Krea2Transformer2DModel"
                                ):
                                    transformer = load_original_layout_transformer(
                                        transformer_cls,
                                        single_file_path,
                                        sf_kwargs,
                                        logger,
                                        fp8_storage = self._fp8_storage_for(
                                            single_file_path, target, loras
                                        ),
                                        fp8_widen_chosen = self._fp8_widen_for(
                                            single_file_path, target, loras
                                        ),
                                    )
                                else:
                                    # A safetensors single-file (fp8) carries its own dtype: no GGUF dequant config.
                                    transformer = transformer_cls.from_single_file(
                                        single_file_path, **sf_kwargs
                                    )
                                if kind == "gguf":
                                    _dequantize_gguf_outside_linears(transformer, dtype, logger)
                            self._raise_if_load_cancelled(_load_token)

                            if is_flux3_family(fam):
                                # The single file / GGUF is the denoiser; the modular companions come from the base.
                                # From the staged dir when the prefetch resolved one, as every other assembly does: a plan
                                # that found the companions under another cache root must not re-resolve them here.
                                pipe = load_flux3_pipeline(
                                    _base_local_dir or fetch_base,
                                    dtype,
                                    hf_token = hf_token,
                                    check_cancelled = lambda: self._raise_if_load_cancelled(
                                        _load_token
                                    ),
                                    transformer = transformer,
                                    local_files_only = local_files_only,
                                    text_encoder = te_prequant_pipe_kwargs(
                                        fam,
                                        fetch_base,
                                        te_quant_mode = text_encoder_quant,
                                        target = target,
                                        dtype = dtype,
                                        hf_token = hf_token,
                                        logger = logger,
                                        local_files_only = local_files_only,
                                        dense_source = _base_local_dir
                                        or (fetch_base if local_files_only else None),
                                        config_source = _base_local_dir,
                                    ).get("text_encoder"),
                                    family_name = fam.name,
                                )
                            elif fam.name == KREA2_FAMILY_NAME:
                                pipe = load_krea2_pipeline(
                                    fetch_base,
                                    dtype,
                                    hf_token = hf_token,
                                    check_cancelled = lambda: self._raise_if_load_cancelled(
                                        _load_token
                                    ),
                                    transformer = transformer,
                                    # Same as full-pipeline: single file supplies only the denoiser; other GB parts must not be fetched.
                                    local_files_only = local_files_only,
                                    text_encoder = te_prequant_pipe_kwargs(
                                        fam,
                                        fetch_base,
                                        te_quant_mode = text_encoder_quant,
                                        target = target,
                                        dtype = dtype,
                                        hf_token = hf_token,
                                        logger = logger,
                                        local_files_only = local_files_only,
                                        dense_source = fetch_base if local_files_only else None,
                                    ).get("text_encoder"),
                                )
                            else:
                                pipe_kwargs = {
                                    "local_files_only": local_files_only,
                                    "torch_dtype": dtype,
                                    "transformer": transformer,
                                    "cache_dir": hub_cache_dir(),
                                }
                                if hf_token:
                                    pipe_kwargs["token"] = hf_token
                                if fam.name == "hunyuanimage-2.1" and getattr(
                                    getattr(transformer, "config", None), "guidance_embeds", False
                                ):
                                    # Distilled file on the base repo: the base's CFG guiders would double every step.
                                    pipe_kwargs["guider"] = None
                                    pipe_kwargs["ocr_guider"] = None
                                if fam.name == HIDREAM_FAMILY_NAME:
                                    pipe_kwargs.update(
                                        hidream_te4_kwargs(
                                            dtype,
                                            hf_token,
                                            fam = fam,
                                            te_quant_mode = text_encoder_quant,
                                            target = target,
                                            local_files_only = local_files_only,
                                        )
                                    )
                                    self._raise_if_load_cancelled(_load_token)
                                # Same pre-cast TE as above: the GGUF supplies the transformer, so the TE is the big download.
                                pipe_kwargs.update(
                                    te_prequant_pipe_kwargs(
                                        fam,
                                        fetch_base,
                                        te_quant_mode = text_encoder_quant,
                                        target = target,
                                        dtype = dtype,
                                        hf_token = hf_token,
                                        logger = logger,
                                        local_files_only = local_files_only,
                                        dense_source = _base_local_dir
                                        or (fetch_base if local_files_only else None),
                                        config_source = _base_local_dir,
                                    )
                                )
                                self._raise_if_load_cancelled(_load_token)
                                # the companions (15.8 GB bf16 Qwen2.5-VL at fp32) fill the host here
                                small_host = self._small_host_decision(
                                    _base_local_dir or fetch_base,
                                    pipe_kwargs,
                                    target,
                                    dtype,
                                    lora_active = _has_active_lora(loras),
                                )
                                if small_host is not None and small_host.engaged:
                                    pipe_kwargs["torch_dtype"] = small_host_torch_dtype_map(
                                        small_host, dtype
                                    )
                                with _converted_shards_pread(dtype, small_host):
                                    pipe = pipeline_cls.from_pretrained(
                                        _base_local_dir or fetch_base, **pipe_kwargs
                                    )
                                if small_host is not None and small_host.engaged:
                                    plan = self._apply_small_host_route(
                                        pipe,
                                        plan,
                                        small_host,
                                        target,
                                        dtype,
                                        fam,
                                        logger,
                                        _load_token,
                                    )

                    if native_streamed and pipe is not None:
                        pending_cache_write = self._save_pipeline_quant_to_cache(
                            pipe,
                            fam,
                            denoiser_modules(pipe),
                            loras = loras,
                            base = base,
                            source_base = _base_local_dir or fetch_base,
                            cancel_check = lambda: self._raise_if_load_cancelled(_load_token),
                            defer = True,
                            native_cache = native_cache_ok,
                        )
                        if not hasattr(pending_cache_write, "start"):
                            pending_cache_write = None
                        self._raise_if_load_cancelled(_load_token)
                    # Quantise dense bf16 pipeline denoisers in place. The blocker excludes UNet and pre-quantised
                    # pipelines; offloaded plans stay dense unless whole-module.
                    if (
                        pipe is not None
                        and kind == "pipeline"
                        and transformer_quant_engaged is None
                        and normalize_transformer_quant(transformer_quant) is not None
                        and (dense_transformer_supported(target) or native_scheme is not None)
                    ):
                        # The same helper the route preflight asked; probed only for an engaged torchao request
                        # (a native scheme never reads it).
                        pipeline_quant_uncompilable = (
                            None
                            if native_scheme is not None
                            else _pipeline_quant_uncompilable_reason(
                                target, fam, speed_mode, model_kind = kind, probe = True
                            )
                        )
                        # from_pretrained widens a raw fp8/int8 checkpoint to bf16, erasing the one thing the blocker
                        # below reads, so recover it from the shard header (Ideogram's loader stamps its own).
                        # `fetch_base` too: a LOCAL diffusers directory stages nothing, so `_base_local_dir` is None
                        # exactly where a hand-converted fp8 checkpoint is likeliest. Non-directories are ignored.
                        source_precision = stored_denoiser_precision(_base_local_dir or fetch_base)
                        if source_precision is not None:
                            for _attr, denoiser in denoiser_modules(pipe):
                                mark_source_precision(denoiser, source_precision)
                        pipeline_quant_blocker = (
                            None
                            if native_scheme is not None or native_offload_scheme is not None
                            else pipeline_quant_uncompilable
                        ) or dense_quant_blocker(pipe)
                        if pipeline_quant_blocker is not None:
                            logger.info(
                                "diffusion.transformer_quant: skipped (%s)", pipeline_quant_blocker
                            )
                            transformer_quant_decline = pipeline_quant_blocker
                            transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                        else:
                            # Re-plan against the quantised steady size. The build peak remains bf16.
                            bf16_plan = plan
                            if plan.offload_policy != OFFLOAD_NONE:
                                preview_scheme = native_scheme or select_transformer_quant_scheme(
                                    target,
                                    transformer_quant,
                                    family = getattr(fam, "name", None),
                                    base_repo = base,
                                )
                                # This in-memory rewrite needs no cache-space or hosted-checkpoint checks.
                                estimate = (
                                    estimate_dense_quant(fam, preview_scheme, base_repo = base)
                                    if preview_scheme is not None
                                    else None
                                )
                                if estimate is not None:
                                    replanned = self._plan_memory(
                                        target,
                                        single_file_path,
                                        base,
                                        fam,
                                        memory_mode,
                                        cpu_offload,
                                        kind = kind,
                                        repo_id = repo_id,
                                        fetch_base = fetch_base,
                                        base_local_dir = _base_local_dir,
                                        transformer_resident_override_mib = (
                                            estimate.steady_transformer_mib
                                        ),
                                        **self._candidate_companion_overrides(
                                            estimate, fam, base, target, text_encoder_quant
                                        ),
                                        text_encoder_quant = text_encoder_quant,
                                    )
                                    # Native schemes (incl. explicit int8 offloaded) keep the resident-only rule.
                                    placed = (
                                        (
                                            replanned
                                            if plan_keeps_transformer_resident(replanned)
                                            else None
                                        )
                                        if native_scheme is not None
                                        or native_offload_scheme is not None
                                        else torchao_offload_plan(replanned, preview_scheme)
                                    )
                                    if placed is not None:
                                        replanned = placed
                                        logger.info(
                                            "diffusion.transformer_quant: %s fits the '%s' placement "
                                            "(%d MiB steady, denoiser streamed=%s, encoders streamed=%s); "
                                            "replacing the bf16 plan's '%s' offload",
                                            preview_scheme,
                                            replanned.offload_policy,
                                            estimate.steady_transformer_mib,
                                            not plan_keeps_transformer_resident(replanned),
                                            bool(getattr(replanned, "stream_text_encoders", False)),
                                            plan.offload_policy,
                                        )
                                        plan = replanned
                            keeps_resident = plan_keeps_transformer_resident(plan)
                            if (
                                not keeps_resident
                                and native_scheme is None
                                and native_offload_scheme is not None
                            ):
                                native_scheme = native_offload_scheme
                                logger.info(
                                    "diffusion.transformer_quant: %s runs torchao-free under the plan's '%s' "
                                    "offload (torchao tensors reject the hooks' Module.to())",
                                    native_scheme,
                                    plan.offload_policy,
                                )
                            inplace_decline = None
                            if not keeps_resident and native_scheme is None:
                                plan, inplace_decline = _inplace_torchao_placement(
                                    plan,
                                    preview_scheme,
                                    estimate,
                                    lambda: largest_streamable_companion_mib(pipe),
                                )
                            if inplace_decline is not None:
                                transformer_quant_decline = inplace_decline
                                logger.info(
                                    "diffusion.transformer_quant: skipped (%s)",
                                    transformer_quant_decline,
                                )
                            elif native_scheme is None and pipeline_quant_uncompilable is not None:
                                # Resident plan: torchao still needs the compile check skipped above.
                                logger.info(
                                    "diffusion.transformer_quant: skipped (%s)",
                                    pipeline_quant_uncompilable,
                                )
                                transformer_quant_decline = pipeline_quant_uncompilable
                                transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                            else:
                                if _has_active_lora(loras):
                                    # PEFT must wrap dense Linears before torchao converts their base layers.
                                    baked = self._resolve_lora_set(
                                        [(i, w) for (i, w) in loras if w != 0],
                                        family = getattr(fam, "name", None),
                                        hf_token = hf_token,
                                    )
                                    for name, path, _weight in baked:
                                        pipe.load_lora_weights(path, adapter_name = name)
                                    pipe.set_adapters(
                                        [n for (n, _p, _w) in baked],
                                        adapter_weights = [w for (_n, _p, w) in baked],
                                    )
                                    pipe._unsloth_loras = baked
                                    pipe._unsloth_loras_baked = True
                                    logger.info(
                                        "diffusion.lora_bake: %d adapter(s) attached before the pipeline "
                                        "quantize",
                                        len(baked),
                                    )
                                # Convert every denoiser so multi-branch pipelines use one precision.
                                denoisers = denoiser_modules(pipe)
                                native_kwargs = (
                                    {
                                        "offload": plan.offload_policy != OFFLOAD_NONE,
                                        "act_int8": native_scheme == TQ_INT8
                                        and native_int8_act(target),
                                    }
                                    if native_scheme is not None
                                    else {}
                                )
                                engaged: list[str] = []
                                cached_scheme = (
                                    self._pipeline_quant_from_cache(
                                        pipe,
                                        fam,
                                        denoisers,
                                        target = target,
                                        transformer_quant = transformer_quant,
                                        fast_accum = transformer_quant_fast_accum,
                                        loras = loras,
                                        base = base,
                                        fetch_base = fetch_base,
                                        source_base = _base_local_dir or fetch_base,
                                        hf_token = hf_token,
                                    )
                                    if native_scheme is None
                                    else None
                                )
                                if cached_scheme is not None:
                                    engaged.append(cached_scheme)
                                for attr, _module in denoisers if cached_scheme is None else ():
                                    scheme = quantize_transformer(
                                        pipe
                                        if attr == "transformer"
                                        else _DenoiserView(pipe, attr),
                                        target,
                                        mode = transformer_quant,
                                        family = getattr(fam, "name", None),
                                        base_repo = base,
                                        fast_accum = transformer_quant_fast_accum,
                                        logger = logger,
                                        **native_kwargs,
                                    )
                                    if scheme is None:
                                        break
                                    engaged.append(scheme)
                                if (
                                    engaged
                                    and len(engaged) == len(denoisers)
                                    and cached_scheme is not None
                                ):
                                    transformer_quant_engaged = cached_scheme
                                elif engaged and len(engaged) == len(denoisers):
                                    transformer_quant_engaged = engaged[0]
                                    pending_cache_write = self._save_pipeline_quant_to_cache(
                                        pipe,
                                        fam,
                                        denoisers,
                                        loras = loras,
                                        base = base,
                                        source_base = _base_local_dir or fetch_base,
                                        cancel_check = lambda: self._raise_if_load_cancelled(
                                            _load_token
                                        ),
                                        defer = True,
                                        native_cache = native_cache_ok,
                                    )
                                    if not hasattr(pending_cache_write, "start"):
                                        pending_cache_write = None
                                    self._raise_if_load_cancelled(_load_token)
                                else:
                                    # A clean decline can remain bf16; a partial in-place conversion is unusable.
                                    dirty = [
                                        attr
                                        for attr, module in denoisers
                                        if transformer_is_quantised(module)
                                    ]
                                    if dirty:
                                        # Rebound, not deleted: the enclosing finally reads `pipe`.
                                        pipe = None
                                        try:
                                            clear_gpu_cache()
                                        except Exception:  # noqa: BLE001
                                            pass
                                        raise RuntimeError(
                                            f"transformer_quant='{transformer_quant}' converted "
                                            f"{', '.join(dirty)} and then failed, leaving the "
                                            "transformer neither dense nor usable. Reload with "
                                            "Precision set to Off to run the checkpoint as-is."
                                        )
                                    transformer_quant_decline = (
                                        f"'{transformer_quant}' did not engage on family "
                                        f"'{getattr(fam, 'name', None)}' with this GPU (the scheme is "
                                        "unsupported here, or the family's measured deny list rules it "
                                        "out); see the server log"
                                    )
                                    transformer_quant_decline_status = RESOLVED_UNSUPPORTED
                            if plan is not bf16_plan and transformer_quant_engaged is None:
                                # The quant-sized placement was only ever valid for the quantised build.
                                plan = bf16_plan
                        # Pinned schemes fail closed, including blocker declines that skipped conversion.
                        if (
                            transformer_quant_engaged is None
                            and transformer_quant_pinned is not None
                            and not precision_fallback_allowed()
                        ):
                            # Rebound, not deleted: the enclosing finally reads `pipe`.
                            pipe = None
                            try:
                                clear_gpu_cache()
                            except Exception:  # noqa: BLE001 -- the refusal matters more than the sweep
                                pass
                            _refuse_pinned_precision()

                    self._raise_if_load_cancelled(_load_token)
                    # Lazy import (the optimizer's submodules import torch lazily) keeps diffusion.py torch-free to
                    # import.
                    from . import diffusion_autoopt as autoopt

                    load_audit = autoopt.new_audit(backend = "image", family = fam)

                    # GGUF defaults to `default` (below quant noise floor); dense stays bit-identical `off`.
                    effective_speed = resolve_speed_mode(speed_mode, is_gguf = kind == "gguf")
                    split_expected = _layer_split_expected(pipe, plan)
                    # A torchao (or torchao-free resident) int8 DiT must be compiled: eager is ~30x slower, losing to GGUF.
                    quant_forces_compile = (
                        transformer_quant_engaged is not None or comfy_compile
                    ) and (
                        native_scheme is None
                        or _native_quant_needs_compile(
                            target,
                            native_scheme,
                            split_expected
                            or plan_keeps_transformer_resident(plan)
                            or _companion_split_expected(pipe, plan),
                        )
                    )
                    if quant_forces_compile and effective_speed == SPEED_OFF:
                        logger.info(
                            "diffusion.transformer_quant: forcing speed_mode=default "
                            "(quantized transformer must be compiled; eager is ~30x slower)"
                        )
                        effective_speed = SPEED_DEFAULT
                    if (
                        small_dit_kept_bf16
                        and speed_mode is None
                        and effective_speed == SPEED_OFF
                        and not comfy_compile
                        and compile_eligible(target, is_gguf = False, family = fam, probe = False)
                        and not fp16_compile_explicit_only(target)
                    ):
                        logger.info(
                            "diffusion.speed: speed_mode=default from render 1 (a small denoiser kept bf16 in "
                            "place of the INT8 default, which compiled from render 1 too)"
                        )
                        effective_speed = SPEED_DEFAULT
                    # Deferred speed auto for dense: stay eager and engage `default` on the 3rd image, where compile
                    # amortises. Only when speed was unset.
                    speed_deferred = (
                        speed_mode is None
                        and effective_speed == SPEED_OFF
                        and transformer_quant_engaged is None
                        and not comfy_compile
                        # Non-probing: _engage_deferred_profile runs the ROCm probe if the 3rd image arrives.
                        and compile_eligible(target, is_gguf = False, family = fam, probe = False)
                        and not fp16_compile_explicit_only(target)
                    )
                    # Speed optims run before placement: snapshot the global backend flags for unload restore.
                    backend_flags_before = snapshot_backend_flags()
                    # Pick the attention kernel BEFORE compile: auto upgrades to cuDNN fused attention on NVIDIA (~1.18x)
                    attention_engaged = apply_attention_backend(
                        pipe,
                        select_attention_backend(
                            target,
                            attention_backend,
                            speed_active = effective_speed != SPEED_OFF,
                            family = fam,
                            speed_unset = speed_mode is None,
                        ),
                        logger = logger,
                        target = target,
                    )
                    self._raise_if_load_cancelled(_load_token)
                    # Step cache drops compile fullgraph; off/fbcache pin it, auto only on max by FBCACHE_MIN_STEPS.
                    cache_request = normalize_transformer_cache(transformer_cache)
                    cache_auto = transformer_cache is None or cache_request == TC_AUTO
                    cache_auto_live = cache_auto and auto_step_cache_allowed(effective_speed)
                    cache_quant_active = transformer_quant_engaged is not None or bool(
                        gguf_filename
                    )
                    default_steps: Optional[int] = None
                    static_plan: Optional[dict] = None
                    if cache_auto:
                        default_steps, _ = default_generation_params(
                            # FLUX.3's family names its variant; its file names may not (see _generation_defaults_for).
                            *((fam.name,) if is_flux3_family(fam) else ()),
                            gguf_filename,
                            content_variant_hint(repo_id, gguf_filename),
                            repo_id,
                            base,
                            fam.name,
                        )
                        static_plan = auto_static_skip_plan(
                            (repo_id, base), skip_tier(speed_mode, effective_speed), default_steps
                        )
                        cache_request = resolve_auto_step_cache(
                            effective_speed, default_steps, static_plan = static_plan
                        )
                    cache_engaged = None
                    if cache_request == TC_STATIC:
                        cache_engaged = (
                            install_static_step_skip(
                                pipe,
                                settings = auto_static_settings(static_plan, logger = logger),
                                logger = logger,
                            )
                            if static_plan
                            else install_static_step_skip(pipe, logger = logger)
                        )
                        if cache_engaged is None and static_plan:
                            # Declined (second denoiser, no forward): auto falls back to what it picked before.
                            static_plan = None
                            cache_request = resolve_auto_step_cache(effective_speed, default_steps)
                    if cache_request != TC_STATIC:
                        cache_engaged = apply_step_cache(
                            pipe,
                            mode = cache_request,
                            threshold = transformer_cache_threshold,
                            # GGUF transformers are quantized too, so the cache needs the higher threshold.
                            quant_active = cache_quant_active,
                            # Prefix-KV families cache only when asked: the default threshold is too lossy at 40 steps.
                            length_changes_ok = not cache_auto,
                            logger = logger,
                        )
                    # Speed unset on a dense load: `default` engages on the 3rd image, and the skip with it.
                    deferred_static_plan = (
                        auto_static_skip_plan((repo_id, base), SPEED_DEFAULT, default_steps)
                        if cache_auto and speed_deferred and cache_engaged is None
                        else None
                    )
                    cache_graph_break = cache_breaks_graph(cache_engaged)
                    self._raise_if_load_cancelled(_load_token)
                    # Arm only where FBCache can engage: a live toggle drops fullgraph and retries every generation.
                    cache_may_toggle = (
                        cache_auto_live
                        and cache_engaged != TC_STATIC
                        and (
                            cache_engaged is not None
                            or (cache_request is None and step_cache_supported(pipe, logger = logger))
                        )
                    )
                    if cache_auto:
                        if cache_engaged == TC_STATIC:
                            cache_reason = (
                                f"auto: static step skip (every {static_plan['every']}) for this model "
                                f"at {static_plan['min_steps']}+ steps; set "
                                "UNSLOTH_DIFFUSION_AUTO_STEP_SKIP=0 to turn it off"
                            )
                        elif deferred_static_plan:
                            cache_reason = (
                                f"auto: static step skip (every {deferred_static_plan['every']}) engages "
                                "with the compiled profile on the 3rd image; set "
                                "UNSLOTH_DIFFUSION_AUTO_STEP_SKIP=0 to turn it off"
                            )
                        elif not cache_auto_live:
                            # Only name the max tier where it would help: SDXL / LTX-2 never cache on any tier.
                            cache_reason = (
                                "auto: step caching engages on the max speed tier only"
                                if step_cache_supported(pipe, logger = logger)
                                else "auto: model does not support step caching"
                            )
                        elif cache_engaged:
                            cache_reason = (
                                f"auto: {default_steps}-step default schedule reaches "
                                f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                            )
                        elif cache_request is not None or not cache_may_toggle:
                            cache_reason = "auto: model does not support step caching"
                        else:
                            cache_reason = (
                                f"auto: {default_steps}-step default schedule is below "
                                f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                            )
                    elif cache_request is not None and not cache_engaged:
                        cache_reason = "requested, but this model does not support step caching; running uncached"
                    else:
                        cache_reason = "requested"
                    # The dense fast path sets gguf_filename, but its transformer is dense.
                    gguf_transformer = kind == "gguf" and transformer_quant_engaged is None

                    # Registered patches (diffusion_autoopt.patches), in their original order: the shared and per-arch
                    # compile-safe fusions on a speed tier (uninstalled on `off`, tracked by eager_patched), then the
                    # per-pipe installs that gate themselves (Q2.1 fast step, FLUX.2 fused RoPE, ROCm FLUX.1 / FLUX.2
                    # one-kernel RoPE + AdaLN).
                    autoopt.install_patches(
                        autoopt.STAGE_LOAD_EAGER,
                        pipe = pipe,
                        target = target,
                        family = fam,
                        dtype = dtype,
                        device = device,
                        speed_mode = effective_speed,
                        backend = "image",
                        audit = load_audit,
                        logger = logger,
                    )
                    eager_patched = effective_speed != SPEED_OFF
                    autoopt.install_patches(
                        autoopt.STAGE_LOAD,
                        pipe = pipe,
                        target = target,
                        family = fam,
                        dtype = dtype,
                        device = device,
                        speed_mode = effective_speed,
                        backend = "image",
                        audit = load_audit,
                        logger = logger,
                    )
                    # fp16-only cards: a guarded family stays float16; patch its overflow sites.
                    from .diffusion_fp16_guard import family_fp16_guard, install_fp16_guard

                    install_fp16_guard(
                        getattr(pipe, "transformer", None),
                        family_fp16_guard(fam),
                        getattr(target, "dtype", None),
                        logger = logger,
                    )
                    # After quantization, so int8 weight-only layers keep their storage.
                    from .diffusion_mps_fp16 import install_mps_fp16

                    install_mps_fp16(getattr(pipe, "transformer", None), target, logger = logger)

                    self._raise_if_load_cancelled(_load_token)
                    speed_offload = plan.offload_policy != OFFLOAD_NONE and not split_expected
                    # Pre-warmed torch.compile cache: a per-fingerprint inductor dir plus a bundle loaded before the
                    # first compiled forward pays the 25-58s compile once.
                    if (
                        effective_speed in (SPEED_DEFAULT, SPEED_MAX)
                        # ROCm GGUF default compiles only the dequant: gate on that predicate, never the regional probe.
                        # On max the dequant is the fallback when the regional compile is refused, so it keeps the cache.
                        and (
                            _gguf_dequant_compilable(
                                True,
                                getattr(target, "device", None) == "cuda",
                                bool(getattr(fam, "supports_torch_compile", True)),
                                target,
                            )
                            if gguf_transformer
                            and effective_speed == SPEED_DEFAULT
                            and getattr(target, "backend", None) == "rocm"
                            else _compile_cache_wanted(
                                pipe,
                                target,
                                fam,
                                is_gguf = gguf_transformer,
                                speed_mode = effective_speed,
                                offload_active = speed_offload,
                            )
                            or (
                                gguf_transformer
                                and getattr(target, "backend", None) == "rocm"
                                and _gguf_dequant_compilable(
                                    True,
                                    getattr(target, "device", None) == "cuda",
                                    bool(getattr(fam, "supports_torch_compile", True)),
                                    target,
                                )
                            )
                        )
                        and not fp16_unet_offloaded(target, pipe, offload_active = speed_offload)
                    ):
                        compile_ctx = compile_cache.begin(
                            family = fam.name,
                            transformer = getattr(pipe, "transformer", None)
                            or getattr(pipe, "unet", None),
                            dtype = getattr(target, "dtype", None),
                            # A GGUF transformer compiles a different graph than a dense load, so it keys its own bundles.
                            quant = "gguf" if gguf_transformer else transformer_quant_engaged,
                            attention_backend = attention_engaged,
                            # One resolver (diffusion_autoopt.plan) for the kwargs the compile will use: fullgraph
                            # mirrors apply_speed_optims (a step cache or planned offload graph-breaks), dynamic is
                            # the value it resolves, and an offloaded denoiser compiling below its hooks keys apart.
                            compile_kwargs = autoopt.compile_cache_kwargs(
                                pipe,
                                speed_mode = effective_speed,
                                fullgraph = not cache_graph_break
                                and not cache_may_toggle
                                and not speed_offload,
                                offload_below_hooks = speed_offload and compile_below_hooks_enabled(),
                            ),
                            logger = logger,
                            reduction_filter = family_filters_reductions(fam),
                        )

                    self._raise_if_load_cancelled(_load_token)
                    # A shipped grid was tuned on the shipped scheduler: no ComfyUI shift.
                    raw_grid = _model_index_sample_sigmas(fetch_base, _base_local_dir)
                    if install_sample_sigmas(pipe, raw_grid, logger) is None:
                        # Before from_pipe copies the scheduler.
                        apply_comfy_flow_shift(
                            pipe,
                            comfy_flow_shift_for(
                                fam,
                                gguf_filename,
                                content_variant_hint(repo_id, gguf_filename),
                                repo_id,
                                display_repo_id,
                                base,
                            ),
                            logger,
                        )
                    # Before the speed optims, so the fused batched tile decode does not replace it.
                    try:
                        install_wide_vae_tiles(getattr(pipe, "vae", None), logger)
                    except Exception as exc:  # noqa: BLE001 - keep the stock tiled decode
                        logger.warning("diffusion.vae_tiling: not installed: %s", exc)
                    install_frame_pad_fix(pipe, target, logger = logger)
                    install_vae_sync(pipe, target, logger = logger)
                    # Before the speed optims so their decode compile lands inside the non-finite check; `off` keeps fp32.
                    vae_fp16 = str(
                        speed_mode or ""
                    ).strip().lower() != SPEED_OFF and enable_fp16_vae_decode(
                        pipe, target, logger = logger
                    )
                    # The optimizer's speed layer (diffusion_autoopt): plan, family hooks, the speed primitive below,
                    # the per-step audit.
                    speed_applied = autoopt.run_speed_layer(
                        pipe,
                        target,
                        family = fam,
                        backend = "image",
                        speed_mode = effective_speed,
                        is_gguf = gguf_transformer,
                        offload_active = speed_offload,
                        audit = load_audit,
                        logger = logger,
                        apply = lambda family, log: apply_speed_optims(
                            pipe,
                            target,
                            is_gguf = gguf_transformer,
                            family = family,
                            speed_mode = effective_speed,
                            cache_active = cache_graph_break or cache_may_toggle,
                            cache_engaged = cache_graph_break,
                            offload_active = speed_offload,
                            denoiser_offloaded = not split_expected
                            and not plan_keeps_transformer_resident(plan),
                            stream_int8_gemm = True,
                            logger = log,
                        ),
                    )
                    if vae_fp16:
                        speed_applied["vae_fp16_decode"] = True
                    # After the speed layer, so it wraps the fused / single-frame _decode the layer installed.
                    try:
                        from .diffusion_vae_oom import install as install_vae_oom_retry
                        install_vae_oom_retry(getattr(pipe, "vae", None), logger, pipe = pipe)
                    except Exception as exc:  # noqa: BLE001 - a decode OOM then fails as before
                        logger.warning("diffusion.vae_oom: not installed: %s", exc)
                    self._raise_if_load_cancelled(_load_token)
                    if (
                        transformer_quant_engaged is not None
                        and native_scheme is None
                        and not speed_applied.get("compiled")
                    ):
                        # Compile did not engage: quantized transformer runs eager, far slower than the GGUF it replaced.
                        logger.warning(
                            "diffusion.transformer_quant: %s engaged but the transformer is NOT "
                            "compiled; eager torchao quant is ~30x slower than GGUF here",
                            transformer_quant_engaged,
                        )
                    # Before the cast and placement, so the unused head is never cast or moved.
                    from .diffusion_text_encoder_trim import trim_text_encoder

                    te_trim = trim_text_encoder(
                        getattr(pipe, "text_encoder", None), family = fam.name
                    )
                    if te_trim.get("lm_head") == "dropped" and te_trim.get("params"):
                        logger.info(
                            "diffusion.text_encoder: dropped unused lm_head (%.2fM params); hidden states unchanged",
                            te_trim["params"] / 1e6,
                        )
                    from .diffusion_te_truncate import truncate_text_encoders

                    truncate_text_encoders(pipe, logger = logger)
                    # Quantise the dense companion text encoder(s) before placement so offload moves the smaller
                    # weights.
                    te_outcome = quantize_text_encoders(
                        pipe,
                        target,
                        mode = text_encoder_quant,
                        family = fam.name,
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        logger = logger,
                    )
                    self._raise_if_load_cancelled(_load_token)
                    te_quant = te_outcome.mode
                    # An explicit encoder mode that engaged nothing or partially must fail like the transformer.
                    if (
                        (te_quant is None or te_outcome.partial)
                        and not text_encoder_quant_auto
                        and normalize_te_quant(text_encoder_quant_requested) is not None
                        and not precision_fallback_allowed()
                    ):
                        raise RuntimeError(
                            precision_refusal_message(
                                "text_encoder_quant",
                                normalize_te_quant(text_encoder_quant_requested) or "",
                                te_outcome.reason or "no text encoder could be cast",
                                off_label = "leave it unset to keep the dense bf16 encoder",
                                auto_available = False,
                            )
                        )

                    # Whole-module offload needs a full component; refine from loaded weights so big TEs leaf-stream.
                    refined_plan = refine_memory_plan_for_components(
                        pipe, refine_balanced_plan_for_components(pipe, plan)
                    )
                    if refined_plan.offload_policy != plan.offload_policy:
                        logger.info(
                            "diffusion.memory: refined policy %s -> %s (%s)",
                            plan.offload_policy,
                            refined_plan.offload_policy,
                            "; ".join(refined_plan.reasons),
                        )
                    plan = refined_plan
                    peak_model = _peak_model_identity(
                        repo_id, gguf_filename, kind, base, fetch_base
                    )
                    if getattr(pipe, "_unsloth_loras", None):
                        peak_model = (
                            "unknown"  # adapters baked in at load: a build the key cannot name
                        )
                    try:
                        # the phase-table seed holds only where it was measured; unpriced references size flat
                        pipe._unsloth_phase_seeds = (
                            _calibrated_activation(fam, target, unpriced_references_ok = True)
                            is not None
                        )
                        # the dense default / max tier seeds hold on the measured target only
                        pipe._unsloth_dense_tier_seeds = _measured_target(target)
                        # a CFG-capable family may place from a peak recorded without CFG (``cfg:off``)
                        pipe._unsloth_peak_expects_cfg = bool(
                            getattr(fam, "uses_negative_prompt", False)
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    peak_vae = planned_vae_saver_tag(pipe, plan)
                    peak_te = _peak_te_label(te_quant, te_outcome)
                    # an auto step cache toggles per request: only uncached renders (the full transformer) record
                    peak_cache = "off" if cache_may_toggle else str(cache_engaged or "off")
                    # graph statics and pools sit above a graph-free peak; a graph armed after an offloaded placement
                    # counts as on
                    peak_graphs = (
                        "on"
                        if speed_applied.get("cuda_graph")
                        or getattr(pipe, "_unsloth_cuda_graph_after_placement", False)
                        else "off"
                    )
                    try:
                        # the learned-peak key: recorded at every completed render, read by later loads
                        pipe._unsloth_peak_identity = (
                            fam.name,
                            learned_peak_label(
                                pipe,
                                attention_engaged or "native",
                                peak_model,
                                peak_vae,
                                peak_te,
                                peak_cache,
                                peak_graphs,
                            ),
                            effective_speed,
                        )
                    except Exception:  # noqa: BLE001 - recording is optional
                        pass
                    # not under the legacy cpu_offload flag alone: that request asked for offload
                    if not cpu_offload or normalize_memory_mode(memory_mode) is not None:
                        plan = refine_plan_from_loaded_weights(
                            pipe,
                            plan,
                            family = fam.name,
                            speed_mode = effective_speed,
                            logger = logger,
                            attention = attention_engaged or "native",
                            model = peak_model,
                            vae = peak_vae,
                            te = peak_te,
                            cache = peak_cache,
                            graphs = peak_graphs,
                        )
                        try:
                            plan_steps, _ = default_generation_params(
                                *((fam.name,) if is_flux3_family(fam) else ()),
                                gguf_filename,
                                content_variant_hint(repo_id, gguf_filename),
                                repo_id,
                                base,
                                fam.name,
                            )
                        except Exception:  # noqa: BLE001 - unpriced: the unconditional switch
                            plan_steps = None
                        plan = prefer_whole_module_offload(
                            pipe, plan, logger = logger, steps = plan_steps
                        )
                        headroom = plan.estimates.get("measured_runtime_headroom_mib")
                        if headroom and (
                            plan.resident_transformer_mib or plan.resident_text_encoder_mib
                        ):
                            pipe._unsloth_measured_reserve = (
                                int(headroom),
                                fam.name,
                                effective_speed,
                                plan.estimates.get("measured_dense_transformer_mib"),
                                plan.estimates.get("measured_compute_bytes"),
                                plan.estimates.get("measured_learned_peak_mib"),
                                plan.estimates.get("measured_learned_peak_pixels"),
                                plan.estimates.get("measured_learned_peak_quadratic"),
                                plan.estimates.get("measured_static_seed", True),
                                plan.estimates.get("measured_learned_cfg_off", False),
                                plan.estimates.get("measured_dense_bf16", False),
                                plan.estimates.get("measured_small_host_int8", False),
                            )
                            setattr(pipe, DECODE_LEND_ATTR, decode_lend_from_plan(plan))

                    from .diffusion_qwenimage21_vision import configure_vision_attention

                    configure_vision_attention(pipe, family = fam.name, target = target, logger = logger)

                    # After the TE quant, so the cache key reflects the encoders that run.
                    cond_cache.install(
                        pipe,
                        family = fam.name,
                        repo_id = repo_id,
                        base_repo = base,
                        dtype = dtype,
                        te_quant = te_quant,
                        logger = logger,
                    )
                    # Outermost, so a repeat skips both the disk cache and the encoders.
                    prompt_cache.install(
                        pipe,
                        identity = {
                            "family": fam.name,
                            "repo": str(repo_id),
                            "base": str(base or repo_id),
                            "dtype": str(dtype),
                            "te_quant": str(te_quant),
                        },
                        logger = logger,
                    )

                    # Apply planned placement; apply_memory_plan returns what actually engaged, so status stays honest.
                    self._raise_if_load_cancelled(_load_token)
                    # a refinement may have re-tiered the plan; routed encoders need group offload
                    plan = self._small_host_plan(pipe, plan, fam, logger)
                    plan = refine_plan_for_multigpu(pipe, plan, logger)
                    # streamed text encoders pin after the load returns, before or during the first encode
                    request_background_encoder_pins(pipe)
                    effective_policy, effective_tiling = apply_memory_plan(
                        pipe,
                        plan,
                        device = device,
                        # Indexed: CPU-offload APIs default to cuda:0 without an index.
                        placement_device = target.torch_device,
                        logger = logger,
                    )
                    start_background_pins(pipe)
                    # a transient OOM during setup may have re-planned to lower residency
                    plan = applied_plan(pipe, plan)
                    _drop_measured_reserve_for_layer_split(pipe, plan)
                    identity = getattr(pipe, "_unsloth_peak_identity", None)
                    if isinstance(identity, tuple) and len(identity) == 3:
                        # a saver the plan asked for may not have engaged: key the renders by what did
                        live_vae = with_label_part(identity[1], "vae", live_vae_saver_tag(pipe))
                        if live_vae != identity[1]:
                            pipe._unsloth_peak_identity = (identity[0], live_vae, identity[2])
                            # placed from the planned saver's peak: requests size from the flat estimate instead
                            pipe._unsloth_peak_static = True
                    if cache_engaged == TC_FBCACHE and _step_cache_dropped_by_layer_split(pipe):
                        cache_engaged = _restore_step_cache_after_layer_split(
                            pipe,
                            (pipe,),
                            threshold = transformer_cache_threshold,
                            quant_active = cache_quant_active,
                            length_changes_ok = not cache_auto,
                            logger = logger,
                        )
                        if cache_engaged is None and getattr(plan, "layer_split", None) is not None:
                            cache_may_toggle = False
                            cache_reason = (
                                "off: the step cache declined the denoiser split across cards"
                            )
                    # streams the whole-resident denoiser back to the flat room while the encoders run
                    install_encode_release(pipe, plan, logger)
                    # Placed, not yet compiled or captured: a later LoRA weight / subset change then writes in place.
                    pinned_scales = _pin_baked_lora_scales(pipe, target.device)
                    if pinned_scales:
                        logger.info(
                            "diffusion.lora_bake: %d adapter scale(s) held as device tensors (weight changes "
                            "keep the compiled graphs)",
                            pinned_scales,
                        )
                    # the speed layer saw only the plan; placement may have pinned every denoiser group since
                    if (
                        denoisers_pinned_resident(pipe)
                        or getattr(plan, "layer_split", None) is not None
                    ):
                        engage_pinned_denoisers(pipe, speed_applied, logger)
                    # Whole-forward graph holds only while no hook moves the denoiser; else, or if unsafe, per-block.
                    if not speed_deferred:
                        if _denoiser_hooked(pipe):
                            compile_pipe_below_offload_hooks(pipe, logger)
                        # the whole-step graph, with the offload copies recorded in it, where placement allows one
                        arm_graphs_after_placement(pipe, speed_applied, logger)
                        cuda_graph.arm_block_graphs(
                            pipe,
                            speed_applied,
                            target = target,
                            family = fam,
                            hooked = _denoiser_hooked(pipe),
                            pinned = denoisers_pinned_resident(pipe),
                            cache_engaged = bool(cache_graph_break),
                            speed_mode = effective_speed,
                            logger = logger,
                        )
                    identity = getattr(pipe, "_unsloth_peak_identity", None)
                    if (
                        isinstance(identity, tuple)
                        and len(identity) == 3
                        and "graphs:on" in str(identity[1]).split("|")
                    ):
                        # placed from the larger of both graph modes' peaks: renders record under the one that armed
                        pipe._unsloth_peak_identity = (
                            identity[0],
                            with_label_part(
                                identity[1], "graphs", _engaged_graph_mode(pipe, speed_applied)
                            ),
                            identity[2],
                        )

                    # Per-control provenance; cpu_offload=False is the unset default, so only True is explicit.
                    resolved = build_resolved_record(
                        {
                            "family_override": _family_override_resolved(family_override, fam),
                            "speed_mode": (
                                speed_mode,
                                "deferred" if speed_deferred else effective_speed,
                                "quantized transformer requires compile"
                                if quant_forces_compile
                                and normalize_speed_mode(speed_mode) in (None, SPEED_OFF)
                                else "auto: exact eager for the first two images; "
                                "the compile profile engages on the 3rd"
                                if speed_deferred
                                else "per-kind default"
                                if speed_mode is None
                                else "requested",
                            ),
                            "transformer_quant": (
                                # Raw request so 'Auto: OFF' and a declined 'FP8' differ; captured before auto retries pin a rung.
                                transformer_quant_requested,
                                transformer_quant_engaged or "off",
                                # A recorded decline wins: it names WHY, where the generic strings below only say the
                                # GGUF ran
                                (
                                    transformer_quant_decline
                                    or (
                                        "not engaged (GGUF transformer loaded)"
                                        if kind == "gguf"
                                        else "dense transformer kept unquantized"
                                    )
                                )
                                if transformer_quant_engaged is None
                                else native_quant_reason(
                                    getattr(pipe, "transformer", None), transformer_quant_engaged
                                )
                                if native_scheme is not None
                                else f"seeded from the hosted checkpoint "
                                f"{transformer_quant_artifact.split(':', 1)[1]}"
                                if transformer_quant_artifact is not None
                                else "re-planned resident for the quantised artifact"
                                if quant_plan is not None
                                else _dense_fast_path_reason(
                                    fam,
                                    transformer_quant_engaged,
                                    base,
                                    kind,
                                    transformer_prequant_path,
                                    loras,
                                    failure = getattr(self, "_prequant_fallback_note", None),
                                ),
                                # Honored when the quant engaged, and when the ask was 'off' (a no-quant request GGUF satisfies).
                                RESOLVED_APPLIED
                                if transformer_quant_engaged is not None
                                or transformer_quant_pinned is None
                                else transformer_quant_decline_status,
                            ),
                            "text_encoder_quant": (
                                # Raw request, as for transformer: the badge must not show a scheme the caller never typed.
                                text_encoder_quant_requested,
                                te_quant or "off",
                                _te_quant_reason(
                                    te_outcome,
                                    te_quant,
                                    auto = text_encoder_quant_auto,
                                    family = fam.name,
                                ),
                                te_outcome.status,
                            ),
                            "attention_backend": (
                                attention_backend,
                                attention_engaged or "native",
                                attention_status_reason(attention_backend, attention_engaged),
                            ),
                            "memory_mode": (
                                memory_mode,
                                effective_policy,
                                "everything fits on the GPU, no offload needed"
                                if effective_policy == OFFLOAD_NONE
                                else "a loaded component exceeds the device budget; "
                                "streaming transformer blocks and text-encoder layers"
                                if effective_policy == OFFLOAD_STREAMING
                                else "planned from measured free VRAM vs estimated footprint",
                            ),
                            "transformer_cache": (
                                None if cache_auto else transformer_cache,
                                cache_engaged or "off",
                                cache_reason,
                            ),
                            "cuda_graph": (
                                None,
                                "on" if speed_applied.get("cuda_graph") else "off",
                                cuda_graph.status_reason(
                                    pipe, bool(speed_applied.get("cuda_graph"))
                                ),
                            ),
                            "cpu_offload": (
                                True if cpu_offload else None,
                                effective_policy != OFFLOAD_NONE,
                                "legacy flag" if cpu_offload else "from the memory plan",
                            ),
                        }
                    )
                    if transformer_quant_artifact is not None:
                        # Beside ``source``, never in it: the frontend branches on "auto"/"explicit".
                        resolved["transformer_quant"]["artifact"] = transformer_quant_artifact
                    # The optimizer's per-step audit (engaged / declined + reason / failed + error), beside the
                    # speed_mode entry.
                    autoopt.record_load_steps(
                        load_audit,
                        attention = attention_engaged,
                        step_cache = cache_engaged,
                        step_cache_reason = cache_reason,
                        compile_cache = compile_ctx is not None,
                        speed_deferred = speed_deferred,
                    )
                    autoopt.finalize(
                        load_audit,
                        tuple(k for k, v in speed_applied.items() if v),
                        resolved,
                    )
                    if gguf_offload_swap and transformer_quant_engaged is not None:
                        from .diffusion_gguf_route import gguf_offload_swap_reason
                        resolved["transformer_quant"]["reason"] = gguf_offload_swap_reason(
                            transformer_quant_engaged,
                            transformer_quant_artifact,
                            plan,
                            gguf_filename,
                        )
                        resolved["transformer_quant"]["replaced"] = f"gguf:{gguf_filename}"

                    # Opt-in (UNSLOTH_DIFFUSION_BG_COMPILE=1): off by default so the same seed twice repeats.
                    bg_module = (
                        None
                        if not bg_compile.load_time_enabled()
                        else _bg_compile_module(
                            pipe,
                            speed_optims = tuple(k for k, v in speed_applied.items() if v),
                            speed_mode = effective_speed,
                            transformer_quant = transformer_quant_engaged,
                            gguf_transformer = gguf_transformer,
                            offload_policy = effective_policy,
                            transformer_cache = cache_engaged,
                            cache_auto = cache_may_toggle,
                            target = target,
                        )
                    )
                    load_bg_compile = (
                        bg_compile.arm(
                            bg_module,
                            logger = logger,
                            seed_dir = getattr(compile_ctx, "dir", None),
                        )
                        if bg_module is not None
                        else None
                    )
                    # After every forward wrapper of the speed / graph layers, so the replica sees the final module.
                    install_cfg_parallel(pipe, offload_policy = effective_policy, logger = logger)
                    install_cfg_batch(
                        pipe,
                        offload_policy = effective_policy,
                        speed_mode = speed_mode,
                        transformer_quant = transformer_quant_engaged,
                        logger = logger,
                    )
                    # Z-Image guided steps at the guidance-off batch: turning CFG on compiles and records nothing new.
                    # compiled_only: an eager denoiser keeps the batched guided call (no graph to protect).
                    install_zimage_cfg_split(
                        pipe, offload_policy = effective_policy, logger = logger, compiled_only = True
                    )
                    te_releaser = te_release.maybe_install(pipe, plan, logger = logger)
                    state = _LoadState(
                        pipe = pipe,
                        family = family_for_checkpoint(
                            fam, gguf_filename, repo_id, display_repo_id, base
                        ),
                        repo_id = repo_id,
                        display_repo_id = display_repo_id,
                        base_repo = base,
                        device = device,
                        gpu_ordinal = target.ordinal,
                        placed_ordinal = placed_cuda_ordinal(target),
                        dtype = str(dtype).replace("torch.", ""),
                        kind = kind,
                        cpu_offload = effective_policy != OFFLOAD_NONE,
                        offload_policy = effective_policy,
                        calibrated_placement = plan_is_calibrated(plan),
                        vae_tiling = effective_tiling,
                        memory_mode = plan.requested_mode,
                        speed_mode = effective_speed,
                        speed_optims = tuple(k for k, v in speed_applied.items() if v),
                        cuda_graphs = tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ()),
                        backend_flags_before = backend_flags_before,
                        text_encoder_quant = te_quant,
                        transformer_quant = transformer_quant_engaged,
                        attention_backend = attention_engaged,
                        attention_request = attention_backend,
                        transformer_cache = cache_engaged,
                        cache_auto = cache_may_toggle,
                        cache_quant_active = cache_quant_active,
                        cache_threshold = transformer_cache_threshold,
                        eager_patched = eager_patched,
                        speed_deferred = speed_deferred,
                        deferred_static_plan = deferred_static_plan,
                        compile_cache_ctx = compile_ctx,
                        bg_compile = load_bg_compile,
                        hf_token = hf_token,
                        resolved = resolved,
                        gguf_filename = gguf_filename,
                        component_files = _active_component_summary(),
                        variant_hint = _image_variant_hint(
                            fam.name,
                            Path(single_file_path).name if single_file_path else None,
                            repo_id,
                            base,
                        ),
                        te_releaser = te_releaser,
                    )
                    # Serialize publication with cancellation.
                    with self._load_cancel_lock:
                        self._raise_if_load_cancelled(_load_token)
                        self._state = state
                        state_committed = True
                        commit_load_budget_fraction("image")
                    try:
                        from . import diffusion_vae_prebuild
                        diffusion_vae_prebuild.maybe_kick(pipe, compile_ctx, logger)
                    except Exception:  # noqa: BLE001 - a prebuild, never a failed load
                        pass
                except BaseException as exc:
                    _clear_exception_frames(exc)
                    raise
                finally:
                    self._stop_load_prefetch()
                    if pending_cache_write is not None:
                        try:
                            if state_committed:
                                pending_cache_write.start()
                            else:
                                pending_cache_write.discard()
                        except Exception:  # noqa: BLE001 - a cache write never fails a load
                            pass
                        pending_cache_write = None
                    if pending_cast_write is not None:
                        try:
                            if state_committed:
                                pending_cast_write.start()
                            else:
                                pending_cast_write.discard()
                        except Exception:  # noqa: BLE001 - a cache write never fails a load
                            pass
                        pending_cast_write = None
                    # Pre-commit failure: roll back the process-wide mutations (symmetric with _unload_locked).
                    if not state_committed:
                        # Never published, so _unload_locked cannot reach them: the pinners would keep the encoders.
                        stop_background_pins(pipe)
                        # First: its gate sits in front of the CUDA-graph layer uninstalled by identity below.
                        if load_bg_compile is not None:
                            load_bg_compile.close()
                            load_bg_compile = None
                        restore_backend_flags(backend_flags_before)
                        compile_cache.restore(compile_ctx, logger = logger)
                        gguf_compile.uninstall_all()  # idempotent
                        cuda_graph.uninstall_all(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())
                        uninstall_static_step_skip(pipe)
                        # After the graphs: the NVFP4 caches hold views that would pin the aborted transformer.
                        try:
                            from .diffusion_nvfp4_linear import reset_nvfp4_state
                            reset_nvfp4_state()
                        except Exception:  # noqa: BLE001 - teardown is best effort
                            pass
                        _uninstall_fused_dit_patches()
                        if eager_patched:
                            _uninstall_eager_patches()
                            _uninstall_fused_dit_patches()
                        state = pipe = transformer = None
                        pipe_kwargs.clear()
                        clear_gpu_cache()

        logger.info(
            # Estimates too: a verdict alone can't be checked from logs; a None term explains a skipped tier.
            "diffusion.loaded: repo=%s base=%s device=%s offload=%s tiling=%s resolved=[%s] reasons=%s "
            "estimates=%s",
            repo_id,
            base,
            device,
            effective_policy,
            effective_tiling,
            format_resolved_for_log(state.resolved),
            "; ".join(plan.reasons),
            plan.estimates,
        )
        finish_encoder_pins_at_load(getattr(state, "pipe", None), logger)
        return self.status()

    @staticmethod
    def _gguf_route_artifact(
        pipe: Any,
        fam: Any,
        scheme: Optional[str],
        *,
        base: Optional[str],
        prequant_path: Optional[str],
    ) -> Optional[str]:
        """``prequant:<repo>/<file>`` when the build loaded a pre-quantized checkpoint, else None."""
        try:
            transformer = getattr(pipe, "transformer", None)
            if scheme is None or getattr(transformer, "_unsloth_prequant_path", None) is None:
                return None
            from .diffusion_denoiser_prequant import prequant_artifact_label

            return prequant_artifact_label(
                resolve_prequant_source(fam, scheme, path_override = prequant_path, base_repo = base),
                transformer,
            )
        except Exception:  # noqa: BLE001 - status only
            return None

    @staticmethod
    def _comfy_single_file_holds_nvfp4(repo_id: Optional[str], filename: Optional[str]) -> bool:
        """Whether an already-on-disk single file holds ComfyUI nvfp4 layers; never downloads, never raises."""
        try:
            if not repo_id or not filename:
                return False
            local_root = Path(str(repo_id)).expanduser()
            if local_root.exists():
                path = str(resolve_local_gguf_child(local_root, filename))
            else:
                from huggingface_hub import try_to_load_from_cache
                path = try_to_load_from_cache(repo_id, filename, cache_dir = hub_cache_dir())
                if not isinstance(path, str):
                    # the resolver falls back to Hugging Face's default cache too
                    path = try_to_load_from_cache(repo_id, filename, cache_dir = None)
            if not isinstance(path, str):
                return False
            from .diffusion_comfy_quant import scan_comfy_quant

            scan = scan_comfy_quant(path)
            return bool(scan is not None and not scan.problems and scan.counts().get("nvfp4"))
        except Exception:  # noqa: BLE001 - an install hint only: the load decides on its own
            return False

    def _comfy_offload_for(self, plan: Any, single_file_path: Any) -> bool:
        """Probe the ComfyUI backends as offloaded: the plan offloads, or it was priced with the offloaded backends."""
        priced = getattr(self, "_comfy_priced_offload", None) or {}
        return not plan_keeps_transformer_resident(plan) or bool(priced.get(str(single_file_path)))

    def _fp8_widen_for(self, single_file_path: Any, target: Any, loras: Any) -> bool:
        """The planner's ``fp8_widen_chosen`` pick for this file, re-checked against device / LoRA / env: only then
        does the host RAM refusal reserve what that rule keeps free."""
        picks = getattr(self, "_fp8_widen_picks", None) or {}
        return bool(picks.get(str(single_file_path))) and comfy_fp8_storage(
            target, lora = _has_active_lora(loras)
        )

    def _fp8_storage_for(self, single_file_path: Any, target: Any, loras: Any) -> bool:
        """The planner's ``fp8_storage_wanted`` pick for this file, re-checked against device / LoRA / env."""
        picks = getattr(self, "_fp8_storage_picks", None) or {}
        return bool(picks.get(str(single_file_path))) and comfy_fp8_storage(
            target, lora = _has_active_lora(loras)
        )

    @staticmethod
    def _comfy_single_file_resident_mib(
        single_file_path: Optional[str],
        fam: Any,
        target: Any,
        base: Optional[str],
        *,
        lora: bool = False,
        fp8_storage: bool = False,
        offload: bool = False,
    ) -> Optional[int]:
        """``comfy_resident_mib`` for a ComfyUI-quantized single file, else None. ``offload`` prices the backends an
        offloaded load keeps (the loader probes them the same way)."""
        try:
            from .diffusion_comfy_quant import comfy_resident_mib, scan_comfy_quant
            from .diffusion_transformer_quant import (
                DEFAULT_MIN_LINEAR_FEATURES,
                TQ_FP8,
                TQ_MXFP8,
                TQ_NVFP4,
                divisible_for_scheme,
            )

            scan = scan_comfy_quant(single_file_path)
            if scan is None or scan.problems:
                return None
            name = getattr(fam, "name", None)
            return comfy_resident_mib(
                single_file_path,
                scan,
                keep_int8 = comfy_int8_backend(target, name, base, offload = offload) is not None,
                keep_fp8 = comfy_fp8_backend(target, name, base, offload = offload) is not None,
                fp8_stored = fp8_storage and comfy_fp8_storage(target, lora = lora),
                **_comfy_keep_fp32_pricing(fam, target),
                # the loader's runtime filter: layers it skips are priced dequantized
                min_features = DEFAULT_MIN_LINEAR_FEATURES,
                fp8_divisible = divisible_for_scheme(TQ_FP8),
                block_divisible = {
                    "nvfp4": divisible_for_scheme(TQ_NVFP4),
                    "mxfp8": divisible_for_scheme(TQ_MXFP8),
                },
                **{
                    # a LoRA dequantizes the block layers (comfy_block_backends)
                    f"keep_{fmt}": backend is not None and not lora
                    for fmt, backend in (
                        ("nvfp4", comfy_block_backend("nvfp4", target, name)[0]),
                        (
                            "mxfp8",
                            comfy_block_backend(
                                "mxfp8", target, name, dtype = getattr(target, "dtype", None)
                            )[0],
                        ),
                    )
                },
            )
        except Exception:  # noqa: BLE001 - a planning aid: the file-size estimate stands
            return None

    @staticmethod
    def _gguf_offload_prequant_placement(
        replanned: Any,
        candidate: Any,
        *,
        fam: Any,
        target: Any,
        auto: bool,
        gguf_filename: Optional[str],
        base: Optional[str],
        prequant_path: Optional[str],
    ) -> tuple[Optional[Any], Optional[str]]:
        """``diffusion_gguf_route.gguf_offload_prequant_placement`` with the cache state resolved."""
        from .diffusion_gguf_route import gguf_offload_prequant_placement

        scheme = getattr(candidate, "scheme", None)
        if candidate is None or not scheme:
            return None, None
        try:
            prequant_cached = (
                _uncached_prequant_repo(
                    fam, target, scheme, base_repo = base, prequant_path = prequant_path
                )
                is None
            )
        except Exception:  # noqa: BLE001 - unknown cache state: treat as uncached
            prequant_cached = False
        return gguf_offload_prequant_placement(
            replanned,
            candidate,
            scheme = scheme,
            auto = auto,
            gguf_filename = gguf_filename,
            prequant_cached = prequant_cached,
            torchao_offload_plan = torchao_offload_plan,
        )

    def _load_dense_quant_pipeline(
        self,
        transformer_cls: Any,
        pipeline_cls: Any,
        base: str,
        device: str,
        dtype: Any,
        hf_token: Optional[str],
        target: DiffusionDeviceTarget,
        mode: Optional[str],
        fast_accum: Optional[bool] = None,
        *,
        fam: Optional[DiffusionFamily] = None,
        prequant_path: Optional[str] = None,
        base_local_dir: Optional[str] = None,
        allow_dense_fallback: bool = True,
        lora_specs: Optional[list[tuple[str, float]]] = None,
        text_encoder_quant: Optional[str] = None,
        fetch_base: Optional[str] = None,
        local_files_only: bool = False,
        _load_token: Optional[int] = None,
        seed_plan: Any = None,
    ) -> tuple[Any, str]:
        """Prefers a prequant checkpoint, else quantises the dense base before compile; a baked LoRA forces the dense path; raises so load_pipeline falls back to GGUF."""
        if _load_token is None:
            _load_token = self._load_token

        def check_cancelled() -> None:
            self._raise_if_load_cancelled(_load_token)

        check_cancelled()
        self._prequant_fallback_note = None
        fetch_base = fetch_base or prefer_ungated_mirror(base, hf_token)
        # 1. Pre-quantized checkpoint, when one is configured for the resolved scheme.
        scheme = _planned_quant_scheme(
            fam, target, mode, base_repo = base, prequant_path = prequant_path
        )
        if scheme is None:
            # Bail before the multi-GB dense download; unsupported schemes (fp8 on Ampere) fail after eviction.
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        if fam is not None and not _has_active_lora(lora_specs):
            # LoRA bake needs dense weights (adapters attach before quantize_); skip prequant for active LoRAs.
            source = resolve_prequant_source(
                fam, scheme, path_override = prequant_path, base_repo = base
            )
            check_cancelled()
            if source is not None:
                # A GPU seed under offload OOMed 8 GB Qwen-Image-2.1 in the streaming hooks.
                seed_device = (
                    prequant_seed_device(seed_plan, device, scheme)
                    if seed_plan is not None
                    else device
                )
                transformer = load_prequantized_transformer(
                    transformer_cls,
                    fetch_base,
                    source,
                    device = device,
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = scheme,
                    # Reject a checkpoint with a different Linear filter so prequant matches runtime-quant.
                    min_features = DEFAULT_MIN_LINEAR_FEATURES,
                    # Unrequested loads may not fetch this multi-GB checkpoint; a miss falls to the refused dense path.
                    local_files_only = local_files_only,
                    # Enforced only when the caller forces fp8 fast-accum; a checkpoint baked otherwise falls to dense.
                    fast_accum = fast_accum,
                    # Reuse the root _uncached_prequant_repo cleared this load against, not the import-time constant.
                    cache_dir = hub_cache_dir(),
                    logger = logger,
                    placement_device = None if seed_device == device else seed_device,
                    family = fam.name,
                )
                check_cancelled()
                if transformer is None:
                    # Never a silent swap: the status names why the checkpoint was not used.
                    from utils.native_path_leases import redact_native_paths

                    from .diffusion_prequant import last_prequant_failure

                    failure = last_prequant_failure()
                    self._prequant_fallback_note = (
                        redact_native_paths(failure)
                        if failure
                        else "the checkpoint was unavailable"
                    )
                    logger.warning(
                        "diffusion.prequant: %s checkpoint from %s not used (%s); quantizing the dense "
                        "transformer instead",
                        scheme,
                        getattr(source, "location", "?"),
                        self._prequant_fallback_note,
                    )
                if transformer is not None:
                    from .diffusion_prequant_cache import note_hit, path_in_cache

                    if getattr(source, "kind", None) == "path" and path_in_cache(
                        getattr(source, "location", "")
                    ):
                        note_hit(source.location, logger)
                    if scheme == TQ_NVFP4:
                        from .diffusion_nvfp4_linear import nvfp4_prewarm
                        nvfp4_prewarm(transformer, (1,), logger = logger)
                    pipe = self._assemble_pipe(
                        pipeline_cls,
                        base,
                        transformer,
                        dtype,
                        hf_token,
                        seed_device,
                        base_local_dir,
                        fam = fam,
                        te_quant_mode = text_encoder_quant,
                        target = target,
                        fetch_base = fetch_base,
                        local_files_only = local_files_only,
                        check_cancelled = check_cancelled,
                    )
                    check_cancelled()
                    return pipe, scheme

        # 2. Fallback: materialise the dense bf16 transformer and quantise it on-device.
        if not allow_dense_fallback:
            # Plan budgeted only the prequant build; the dense bf16 transformer would exceed it after eviction.
            note = getattr(self, "_prequant_fallback_note", None)
            raise RuntimeError(
                "prequant checkpoint unavailable"
                + (f" ({note})" if note else "")
                + " and the dense transformer does not fit resident"
            )
        # Hub id, not base_local_dir: diffusers treats a local dir as terminal and would not refetch shards.
        check_cancelled()
        transformer = transformer_cls.from_pretrained(
            fetch_base,
            subfolder = "transformer",
            torch_dtype = dtype,
            token = hf_token,
            cache_dir = hub_cache_dir(),
            # Dense bf16 transformer is the largest fetch on this path, so API-initiated loads are refused here.
            local_files_only = local_files_only,
        )
        check_cancelled()
        pipe = self._assemble_pipe(
            pipeline_cls,
            base,
            transformer,
            dtype,
            hf_token,
            device,
            base_local_dir,
            fam = fam,
            te_quant_mode = text_encoder_quant,
            target = target,
            fetch_base = fetch_base,
            local_files_only = local_files_only,
            check_cancelled = check_cancelled,
        )
        check_cancelled()
        if _has_active_lora(lora_specs):
            # Bake LoRA BEFORE quantize_: post-quant torchao dispatch TypeErrors on peft wrappers.
            baked = self._resolve_lora_set(
                [(i, w) for (i, w) in lora_specs if w != 0],
                family = getattr(fam, "name", None),
                hf_token = hf_token,
            )
            for name, path, _weight in baked:
                check_cancelled()
                pipe.load_lora_weights(path, adapter_name = name)
            check_cancelled()
            pipe.set_adapters(
                [n for (n, _p, _w) in baked],
                adapter_weights = [w for (_n, _p, w) in baked],
            )
            pipe._unsloth_loras = baked
            pipe._unsloth_loras_baked = True
            logger.info(
                "diffusion.lora_bake: %d adapter(s) attached before %s quantize",
                len(baked),
                scheme,
            )
        check_cancelled()
        scheme = quantize_transformer(
            pipe,
            target,
            mode = mode,
            family = getattr(fam, "name", None),
            # The upstream id, not ``fetch_base``: a policy is a claim about one set of weights.
            base_repo = base,
            fast_accum = fast_accum,
            logger = logger,
        )
        check_cancelled()
        if scheme is None:
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        if fam is not None and not _has_active_lora(lora_specs):
            # Best-effort: cache this build so the next load of the same weights skips the dense transformer.
            from .diffusion_prequant_cache import save_runtime_quant
            save_runtime_quant(
                getattr(pipe, "transformer", None),
                fam,
                base_repo = base,
                source_base = base_local_dir or fetch_base,
                logger = logger,
                cancel_check = check_cancelled,
            )
            check_cancelled()
        return pipe, scheme

    @staticmethod
    def _pipeline_quant_from_cache(
        pipe: Any,
        fam: Any,
        denoisers: Any,
        *,
        target: Any,
        transformer_quant: Optional[str],
        fast_accum: Optional[bool],
        loras: Any,
        base: Optional[str],
        fetch_base: Optional[str],
        source_base: Optional[str],
        hf_token: Optional[str],
    ) -> Optional[str]:
        """Hosted checkpoints are not read: the published FLUX.2-klein-4B INT8 renders different pixels."""
        try:
            names = [attr for attr, _module in (denoisers or ())]
            transformer = getattr(pipe, "transformer", None)
            if (
                fam is None
                or transformer is None
                or names != ["transformer"]
                or _has_active_lora(loras)
            ):
                return None
            family = getattr(fam, "name", None)
            if native_quant_scheme(target, transformer_quant, family = family) is not None:
                return None
            scheme = select_transformer_quant_scheme(
                target, transformer_quant, family = family, base_repo = base
            )
            from .diffusion_prequant_cache import CACHEABLE_SCHEMES, lookup, note_hit

            if scheme not in CACHEABLE_SCHEMES or transformer_is_quantised(transformer):
                return None
            cached = lookup(fam, scheme, base_repo = base, source_base = source_base, logger = logger)
            if not cached:
                return None
            first = next(iter(transformer.parameters()), None)
            here = str(first.device) if first is not None else "cpu"
            t0 = time.perf_counter()
            loaded = load_prequantized_transformer(
                type(transformer),
                fetch_base or base,
                PrequantSource(kind = "path", location = cached, filename = None),
                device = str(getattr(target, "device", None) or here),
                dtype = getattr(first, "dtype", None),
                hf_token = hf_token,
                scheme = scheme,
                min_features = DEFAULT_MIN_LINEAR_FEATURES,
                fast_accum = fast_accum,
                cache_dir = hub_cache_dir(),
                local_files_only = True,
                logger = logger,
                placement_device = here,
                family = family,
                into = transformer,
            )
            if loaded is None:
                if transformer_is_quantised(transformer):
                    raise PrequantIntoError("the cached build was partly applied and then failed")
                return None
            note_hit(cached, logger)
            logger.info(
                "diffusion.transformer_quant: %s read from the local prequant cache in %.1f s (no re-quantize)",
                scheme,
                time.perf_counter() - t0,
            )
            return scheme
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, PrequantIntoError):
                raise RuntimeError(
                    f"transformer_quant='{transformer_quant}' could not finish loading its cached build ({exc}); "
                    "reload to quantize it fresh"
                ) from exc
            logger.warning("diffusion.prequant_cache: read skipped (%s)", exc)
            return None

    @staticmethod
    def _native_quant_from_cache(
        fam: Any,
        transformer_cls: Any,
        scheme: Optional[str],
        target: Any,
        *,
        base: Optional[str],
        source_base: Optional[str],
        config_kwargs: Optional[dict] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Any:
        """The forced-offload native ``transformer`` from the local prequant cache, or None. CUDA only. Never raises."""
        try:
            if fam is None or scheme is None or getattr(target, "device", None) != "cuda":
                return None
            from .diffusion_native_cache import native_spec, read_for_load

            spec = native_spec(
                scheme,
                getattr(fam, "name", None),
                act_int8 = scheme == TQ_INT8 and native_int8_act(target),
            )
            module = read_for_load(
                fam,
                spec,
                transformer_cls,
                base_repo = base,
                source_base = source_base,
                config_kwargs = config_kwargs,
                logger = logger,
                cancel_check = cancel_check,
            )
            if module is not None:
                logger.info(
                    "diffusion.transformer_quant: native %s read from the local prequant cache "
                    "(the dense transformer is not loaded, no re-quantize)",
                    scheme,
                )
            return module
        except Exception as exc:  # noqa: BLE001 - a cache problem is a miss
            logger.warning("diffusion.prequant_cache: native read skipped (%s)", exc)
            return None

    @staticmethod
    def _native_quant_build_streamed(
        fam: Any,
        transformer_cls: Any,
        scheme: Optional[str],
        target: Any,
        dtype: Any,
        *,
        source_base: Optional[str],
        config_kwargs: Optional[dict] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Any:
        """Forced-offload native ``transformer`` built per tensor, or None for the dense load. Never raises."""
        try:
            if (
                fam is None
                or scheme is None
                or not source_base
                or getattr(target, "device", None) != "cuda"
            ):
                return None
            from .diffusion_native_cache import build_from_shards, native_spec

            family = getattr(fam, "name", None)
            spec = native_spec(
                scheme, family, act_int8 = scheme == TQ_INT8 and native_int8_act(target)
            )
            return build_from_shards(
                transformer_cls,
                str(source_base),
                spec,
                family = family,
                dtype = dtype,
                config_kwargs = config_kwargs,
                logger = logger,
                cancel_check = cancel_check,
            )
        except Exception as exc:  # noqa: BLE001 - the dense load still runs
            logger.warning("diffusion.transformer_quant: per-tensor native build skipped (%s)", exc)
            return None

    @staticmethod
    def _save_pipeline_quant_to_cache(
        pipe: Any,
        fam: Any,
        denoisers: Any,
        *,
        loras: Any,
        base: Optional[str],
        source_base: Optional[str],
        cancel_check: Optional[Callable[[], None]] = None,
        defer: bool = False,
        native_cache: bool = False,
    ) -> Any:
        """Cache a load-time quantized ``transformer`` (single denoiser only, the cache keys that component; no LoRA).
        ``defer`` may return a ``PendingWrite`` for the caller to start once the load is published. A native build is
        written only when ``native_cache`` says the load could read it back. Never raises."""
        try:
            names = [attr for attr, _module in (denoisers or ())]
            if fam is None or names != ["transformer"] or _has_active_lora(loras):
                return None
            transformer = getattr(pipe, "transformer", None)
            if (getattr(transformer, "_unsloth_runtime_quant_info", None) or {}).get("native"):
                if not native_cache:
                    return None
                from .diffusion_native_cache import save_deferred

                if not defer:
                    from .diffusion_native_cache import build_spec, save
                    spec = build_spec(transformer, getattr(fam, "name", None))
                    return (
                        None
                        if spec is None
                        else save(
                            transformer,
                            fam,
                            spec,
                            base_repo = base,
                            source_base = source_base,
                            logger = logger,
                            cancel_check = cancel_check,
                        )
                    )
                return save_deferred(
                    transformer,
                    fam,
                    base_repo = base,
                    source_base = source_base,
                    logger = logger,
                    cancel_check = cancel_check,
                )
            from .diffusion_prequant_cache import save_runtime_quant

            return save_runtime_quant(
                getattr(pipe, "transformer", None),
                fam,
                base_repo = base,
                source_base = source_base,
                logger = logger,
                cancel_check = cancel_check,
                defer = defer,
            )
        except Exception as exc:  # noqa: BLE001 - a cache write never fails a load
            logger.warning("diffusion.prequant_cache: skipped (%s)", exc)
            return None

    @staticmethod
    def _assemble_pipe(
        pipeline_cls: Any,
        base: str,
        transformer: Any,
        dtype: Any,
        hf_token: Optional[str],
        device: str,
        base_local_dir: Optional[str] = None,
        fam: Optional[DiffusionFamily] = None,
        te_quant_mode: Optional[str] = None,
        target: Any = None,
        fetch_base: Optional[str] = None,
        local_files_only: bool = False,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> Any:
        """Fetches via fetch_base, since a gated upstream id would 401 when nothing was staged locally."""
        check_cancelled = check_cancelled or (lambda: None)
        check_cancelled()
        base = fetch_base or prefer_ungated_mirror(base, hf_token)
        if is_flux3_family(fam):
            # FLUX.3 ships modular blocks only, so from_pretrained has no model_index.json to read.
            flux3_te = None
            if target is not None:
                flux3_te = te_prequant_pipe_kwargs(
                    fam,
                    base,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    dtype = dtype,
                    hf_token = hf_token,
                    logger = logger,
                    local_files_only = local_files_only,
                    dense_source = base_local_dir or (base if local_files_only else None),
                    config_source = base_local_dir,
                ).get("text_encoder")
            check_cancelled()
            pipe = load_flux3_pipeline(
                base_local_dir or base,
                dtype,
                hf_token = hf_token,
                transformer = transformer,
                text_encoder = flux3_te,
                check_cancelled = check_cancelled,
                local_files_only = local_files_only,
                family_name = getattr(fam, "name", None),
            )
            check_cancelled()
            pipe.to(device)
            return pipe
        if getattr(fam, "name", None) == MOSS_IMAGE_FAMILY_NAME:
            from .diffusion_moss_image import load_moss_image_pipeline

            moss_te = None
            if target is not None:
                moss_te = te_prequant_pipe_kwargs(
                    fam,
                    base,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    dtype = dtype,
                    hf_token = hf_token,
                    logger = logger,
                    local_files_only = local_files_only,
                    dense_source = base_local_dir or (base if local_files_only else None),
                    config_source = base_local_dir,
                ).get("text_encoder")
            check_cancelled()
            pipe = load_moss_image_pipeline(
                base_local_dir or base,
                dtype,
                hf_token = hf_token,
                transformer = transformer,
                text_encoder = moss_te,
                check_cancelled = check_cancelled,
                local_files_only = local_files_only,
            )
            check_cancelled()
            pipe.to(device)
            return pipe
        if getattr(fam, "name", None) == KREA2_FAMILY_NAME:
            # krea ships transformers-5.x configs and no top-level tokenizer files, so assemble per-component.
            krea_te = None
            if target is not None:
                krea_te = te_prequant_pipe_kwargs(
                    fam,
                    base,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    dtype = dtype,
                    hf_token = hf_token,
                    logger = logger,
                    local_files_only = local_files_only,
                    dense_source = base_local_dir or (base if local_files_only else None),
                    config_source = base_local_dir,
                ).get("text_encoder")
            check_cancelled()
            pipe = load_krea2_pipeline(
                base_local_dir or base,
                dtype,
                hf_token = hf_token,
                transformer = transformer,
                text_encoder = krea_te,
                check_cancelled = check_cancelled,
                # ``base_local_dir`` is None if nothing was staged, leaving a repo id; pipe_kwargs guards it too.
                local_files_only = local_files_only,
            )
            check_cancelled()
            pipe.to(device)
            return pipe
        pipe_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "torch_dtype": dtype,
            "transformer": transformer,
            "cache_dir": hub_cache_dir(),
        }
        if hf_token:
            pipe_kwargs["token"] = hf_token
        if getattr(fam, "name", None) == HIDREAM_FAMILY_NAME:
            pipe_kwargs.update(
                hidream_te4_kwargs(
                    dtype,
                    hf_token,
                    fam = fam,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    local_files_only = local_files_only,
                )
            )
        if target is not None:
            check_cancelled()
            pipe_kwargs.update(
                te_prequant_pipe_kwargs(
                    fam,
                    base,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    dtype = dtype,
                    hf_token = hf_token,
                    logger = logger,
                    local_files_only = local_files_only,
                    dense_source = base_local_dir or (base if local_files_only else None),
                    config_source = base_local_dir,
                )
            )
        check_cancelled()
        pipe_kwargs.update(
            supplied_component_pipe_kwargs(
                base_local_dir or base,
                dtype = dtype,
                hf_token = hf_token,
                local_files_only = local_files_only,
                family = getattr(fam, "name", None),
            )
        )
        with _converted_shards_pread(dtype, None):
            pipe = pipeline_cls.from_pretrained(base_local_dir or base, **pipe_kwargs)
        check_cancelled()
        pipe.to(device)
        return pipe

    def _bf16_table_plan(
        self,
        target: DiffusionDeviceTarget,
        fam: DiffusionFamily,
        base: str,
        memory_mode: Optional[str],
        cpu_offload: bool,
        *,
        kind: str,
        repo_id: Optional[str],
        fetch_base: Optional[str] = None,
        base_local_dir: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        device_memory_override: Optional[DeviceMemory] = None,
    ) -> Any:
        """Plan priced from the family's bf16-resident table, or None for an unknown family."""
        table = family_bf16_components_gb(fam, base)
        if table is None:
            return None
        text_encoder_gb = table[1]
        try:
            from .diffusion_te_prequant import (
                TE_PREQUANT_BUDGET_SCALE,
                te_prequant_budget_scale,
                te_prequant_sources_for_base,
            )

            te_scale = te_prequant_budget_scale(
                fam, te_quant_mode = text_encoder_quant, target = target, base = base
            )
            text_encoder_gb = table[1] * te_scale
            # HiDream's Llama text_encoder_4 comes pre-cast from its own repo, outside the components the scale probes.
            if (
                te_scale == 1.0
                and getattr(fam, "name", None) == HIDREAM_FAMILY_NAME
                and te_prequant_sources_for_base(
                    fam,
                    HIDREAM_LLAMA_REPO,
                    te_quant_mode = text_encoder_quant,
                    target = target,
                    components = ("text_encoder_4",),
                    standalone_component_bases = {"text_encoder_4": HIDREAM_LLAMA_REPO},
                ).get("text_encoder_4")
                is not None
            ):
                llama_gb = min(HIDREAM_LLAMA_BF16_BYTES / 1000.0**3, table[1])
                text_encoder_gb = table[1] - llama_gb * (1.0 - TE_PREQUANT_BUDGET_SCALE)
        except Exception:  # noqa: BLE001 -- sizing aid only; budget the dense encoder
            text_encoder_gb = table[1]
        mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
        text_encoder_mib = int(text_encoder_gb * mib_per_gb)
        return self._plan_memory(
            target,
            None,
            base,
            fam,
            memory_mode,
            cpu_offload,
            kind = kind,
            repo_id = repo_id,
            fetch_base = fetch_base,
            base_local_dir = base_local_dir,
            transformer_resident_override_mib = int(table[0] * mib_per_gb),
            companion_override_mib = text_encoder_mib + int(table[2] * mib_per_gb),
            text_encoder_override_mib = text_encoder_mib,
            device_memory_override = device_memory_override,
        )

    def _bf16_resident_plan(
        self,
        plan: Any,
        target: DiffusionDeviceTarget,
        fam: DiffusionFamily,
        base: str,
        memory_mode: Optional[str],
        cpu_offload: bool,
        *,
        kind: str,
        repo_id: Optional[str],
        fetch_base: Optional[str] = None,
        base_local_dir: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
    ) -> Any:
        """``plan`` re-priced when the table lowers it (cached shards wider than bf16).

        HiDream always: no cached plan counts its separately loaded text_encoder_4."""
        if kind != "pipeline":
            return plan
        if (
            getattr(fam, "name", None) != HIDREAM_FAMILY_NAME
            and self._resident_sized_plan(
                plan, fam, base, target, kind, text_encoder_quant = text_encoder_quant
            )
            is plan
        ):
            return plan
        try:
            repriced = self._bf16_table_plan(
                target,
                fam,
                base,
                memory_mode,
                cpu_offload,
                kind = kind,
                repo_id = repo_id,
                fetch_base = fetch_base,
                base_local_dir = base_local_dir,
                text_encoder_quant = text_encoder_quant,
                device_memory_override = getattr(plan, "device_memory", None),
            )
        except Exception:  # noqa: BLE001 -- sizing aid only; decide on the plan as built
            return plan
        return plan if repriced is None else repriced

    @staticmethod
    def _released_transformer_cached(base: Optional[str]) -> bool:
        """Complete ``transformer/`` in the snapshot the loader reads; roots in ``_assert_base_repo_accessible`` order."""
        if not base:
            return False
        try:
            local = Path(base).expanduser()
            if local.is_dir():
                return _transformer_folder_complete(local / "transformer")
            for repo_dir in DiffusionBackend._hub_cache_repo_dirs(base):
                snapshot = DiffusionBackend._live_snapshot_dir(repo_dir)
                if snapshot is not None and (snapshot / "model_index.json").is_file():
                    return _transformer_folder_complete(snapshot / "transformer")
            return False
        except Exception:  # noqa: BLE001 -- unreadable cache: treat the shards as missing
            return False

    @staticmethod
    def _candidate_companion_overrides(
        candidate: Any,
        fam: DiffusionFamily,
        base: str,
        target: Any,
        text_encoder_quant: Optional[str],
    ) -> dict[str, Optional[int]]:
        """``_plan_memory`` overrides for a dense-quant candidate, pricing the encoder the load actually opens."""
        companions = DiffusionBackend._precast_scaled_companions_mib(
            candidate, fam, base, target, text_encoder_quant
        )
        encoders = getattr(candidate, "text_encoders_mib", None)
        if encoders and companions is not None:
            raw = getattr(candidate, "companions_mib", None)
            if raw is not None:
                # companions - VAE is the priced encoder share; the VAE term is unscaled in both.
                encoders = max(0, int(encoders) - (int(raw) - int(companions)))
        return {
            "companion_override_mib": companions,
            "text_encoder_override_mib": encoders,
        }

    @staticmethod
    def _precast_scaled_companions_mib(
        candidate: Any,
        fam: DiffusionFamily,
        base: str,
        target: Any,
        text_encoder_quant: Optional[str],
    ) -> Optional[int]:
        """Prices the text encoder at pre-cast size when one is injected, so the fit check matches
        the load."""
        companions = getattr(candidate, "companions_mib", None)
        if companions is None:
            return None
        try:
            from .diffusion_te_prequant import te_prequant_budget_scale

            scale = te_prequant_budget_scale(
                fam,
                te_quant_mode = text_encoder_quant,
                target = target,
                base = base,
            )
            encoders = int(getattr(candidate, "text_encoders_mib", 0) or 0)
            if scale == 1.0 or encoders <= 0:
                return int(companions)
            return int(companions) - encoders + int(encoders * scale)
        except Exception:  # noqa: BLE001 -- sizing aid only; the dense total still refuses safely
            return int(companions)

    def _seeded_pipeline_plan(
        self,
        scheme: str,
        target: Any,
        base: str,
        fam: Any,
        memory_mode: Optional[str],
        cpu_offload: bool,
        *,
        repo_id: str,
        base_local_dir: Optional[str],
        fetch_base: Optional[str],
        device_memory_override: Optional[DeviceMemory] = None,
        text_encoder_quant: Optional[str] = None,
    ):
        """Full-pipeline plan priced on the pre-quantised ``scheme`` seed, or None when the family table cannot size it."""
        seed_estimate = estimate_dense_quant(fam, scheme, base_repo = base, prequant_available = True)
        if seed_estimate is None:
            return None
        return self._plan_memory(
            target,
            None,
            base,
            fam,
            memory_mode,
            cpu_offload,
            kind = "pipeline",
            repo_id = repo_id,
            base_local_dir = base_local_dir,
            fetch_base = fetch_base,
            transformer_resident_override_mib = seed_estimate.steady_transformer_mib,
            **self._candidate_companion_overrides(
                seed_estimate, fam, base, target, text_encoder_quant
            ),
            device_memory_override = device_memory_override,
            text_encoder_quant = text_encoder_quant,
        )

    def _seed_plan_stays_resident(
        self, scheme: str, target: Any, *args: Any, **kwargs: Any
    ) -> bool:
        """Whether the post-teardown seeded plan keeps ``scheme``, asked before teardown with this process's allocation
        credited as free (errs toward keeping the seed). Unanswerable keeps it."""
        try:
            memory = snapshot_device_memory(target)
            if getattr(target, "device", None) == "cuda" and memory.free_mib is not None:
                import torch

                free = int(memory.free_mib) + int(torch.cuda.memory_reserved()) // (1024 * 1024)
                if memory.total_mib is not None:
                    free = min(free, int(memory.total_mib))
                memory = DeviceMemory(
                    memory.backend, memory.device, memory.memory_kind, free, memory.total_mib
                )
            plan = self._seeded_pipeline_plan(
                scheme, target, *args, device_memory_override = memory, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - a gate on an optional install must not fail the load
            logger.debug("diffusion.nvfp4_install: seed plan preview skipped: %r", exc)
            return True
        return plan is not None and plan.offload_policy == OFFLOAD_NONE

    def _resident_sized_plan(
        self,
        plan: Any,
        fam: DiffusionFamily,
        base: str,
        target: DiffusionDeviceTarget,
        kind: str,
        text_encoder_quant: Optional[str] = None,
    ) -> Any:
        """Family bf16-resident weight total for the memory refusal; only ever lowers the estimate, and is skipped for GGUF/single-file, non-bf16 and local dirs."""
        try:
            # Companion bytes are double-counted for SDXL-style single files; fine as a hint, not as
            # a refusal.
            if kind in ("single_file", "gguf") and getattr(fam, "single_file_is_pipeline", False):
                companion = plan.estimates.get("companion_dense_mib")
                current = plan.estimates.get("model_dense_mib")
                if companion and current is not None and int(companion) < int(current):
                    return replace(
                        plan,
                        estimates = {
                            **plan.estimates,
                            "model_dense_mib": int(current) - int(companion),
                        },
                    )
                return plan
            if kind != "pipeline":
                return plan
            table_base = base
            if _is_local_path(base):
                from utils.hf_cache_settings import known_hf_hub_caches

                table_base = hf_cache_repo_id(base)
                snapshot = Path(base).expanduser().resolve()
                if table_base is None or not any(
                    snapshot.is_relative_to(Path(root).expanduser().resolve())
                    for root in known_hf_hub_caches()
                ):
                    return plan
            import torch

            if getattr(target, "dtype", None) not in (torch.bfloat16, torch.float16):
                return plan
            # Recognised bases only: a renamed 9B derivative would otherwise read the 4B family entry.
            canonical = canonical_base(table_base)
            if (
                base_repo_bf16_components_gb(table_base) is None
                and canonical.lower() != str(getattr(fam, "base_repo", "") or "").lower()
            ):
                return plan
            table = family_bf16_components_gb(fam, table_base)
            if table is None:
                return plan
            # Dense encoder term over-refuses: a hosted fp8 pre-cast encoder is ~0.65x, so budget that instead.
            from .diffusion_te_prequant import te_prequant_budget_scale

            te_scale = te_prequant_budget_scale(
                fam,
                te_quant_mode = text_encoder_quant,
                target = target,
                base = table_base,
            )
            transformer_gb, text_encoders_gb, vae_gb = table
            resident_gb = transformer_gb + text_encoders_gb * te_scale + vae_gb
            current = plan.estimates.get("model_dense_mib")
            table_mib = int(resident_gb * (1000.0**3) / (1024.0 * 1024.0))
            if current is None or table_mib >= int(current):
                return plan
            return replace(
                plan,
                estimates = {**plan.estimates, "model_dense_mib": table_mib},
            )
        except Exception:  # noqa: BLE001 - sizing aid only; refuse on the plan as built
            return plan

    def declared_footprint_shortfall(
        self,
        fam: Optional[DiffusionFamily],
        repo_id: str,
        base: str,
        *,
        kind: str,
        declared_files: Any,
        prequant_bytes: int = 0,
        extra_bf16_bytes: int = 0,
        gguf_filename: Optional[str] = None,
        gpu_ordinal: Optional[int] = None,
        memory_mode: Optional[str] = None,
        cpu_offload: bool = False,
    ) -> Optional[str]:
        """Pre-download unified-memory refusal for pipeline picks only, from Hub sizes, against total capacity so a resident model is ignored."""
        if fam is None or kind != "pipeline":
            return None
        declared_files = list(declared_files or ())
        if (
            max(0, int(prequant_bytes or 0)) <= 0
            and max(0, int(extra_bf16_bytes or 0)) <= 0
            and not any(int(size or 0) > 0 for _name, size in declared_files)
        ):
            return None
        try:
            target = self._target_for_ordinal(fam, gpu_ordinal)
            with diffusion_device_scope(target.ordinal if target.is_cuda_torch_device else None):
                device_memory = snapshot_device_memory(target)
            if device_memory.memory_kind != "unified_memory" or device_memory.total_mib is None:
                return None
            dtype_scale = 2.0 if "float32" in str(getattr(target, "dtype", "")).lower() else 1.0
            resident_bytes = resident_bytes_from_declared(
                base,
                declared_files,
                prequant_bytes = prequant_bytes,
                extra_bf16_bytes = extra_bf16_bytes,
                dtype_scale = dtype_scale,
            )
            if not resident_bytes:
                return None
            capacity = unified_capacity_memory(device_memory)
            resident_mib = int(resident_bytes) // (1024 * 1024)
            # Plain pipelines only: pre-cast or separately hosted parts are not split by file here.
            companion_bytes = (
                companion_bytes_from_declared(base, declared_files, dtype_scale = dtype_scale)
                if not prequant_bytes and not extra_bf16_bytes
                else None
            )
            variant_hint = _image_variant_hint(fam.name, gguf_filename, repo_id, base)
            plan = plan_diffusion_memory(
                target = target,
                device_memory = capacity,
                model_dense_mib = resident_mib,
                companion_dense_mib = (
                    None if companion_bytes is None else int(companion_bytes) // (1024 * 1024)
                ),
                runtime_headroom_mib = estimate_image_runtime_mib(
                    width = None, height = None, family = variant_hint
                ),
                requested_mode = memory_mode,
                explicit_offload = bool(cpu_offload),
            )
            # The estimate uses capacity, not current free memory.
            plan = replace(plan, device_memory = replace(capacity, free_mib = None))
        except Exception:  # noqa: BLE001 -- failed sizing must not refuse a load
            return None
        return unified_memory_shortfall_message(plan, family = fam.name)

    def _small_host_decision(
        self, source: Any, pipe_kwargs: dict, target: Any, dtype: Any, *, lora_active: bool
    ) -> Any:
        """Small-host decision for a pipeline load; raises its refusal, None without a local snapshot."""
        snapshot = resolve_snapshot_dir(source, hub_cache_dir())
        if snapshot is None:
            return None
        comps = {
            name: comp
            for name, comp in stored_components(snapshot).items()
            # pre-built components (hosted fp8 encoder, prequant denoiser) are not loaded here
            if name not in pipe_kwargs
        }
        total, available = host_ram_mib()
        decision = decide_small_host(
            comps,
            dtype,
            device = str(getattr(target, "backend", None) or getattr(target, "device", "")),
            host_total_mib = total,
            host_available_mib = available,
            lora_active = lora_active,
            host_commit_mib = host_commit_mib(),
            pread_libraries = ("transformers",) if pread_available() else (),
        )
        if decision.engaged or decision.refuse:
            logger.info(
                "diffusion.small_host: engaged=%s (%s); stored-dtype components=%s",
                decision.engaged,
                decision.reason,
                decision.storage_dtypes,
            )
        elif decision.available_mib is not None:
            logger.info("diffusion.small_host: engaged=False (%s)", decision.reason)
        if decision.refuse:
            raise RuntimeError(decision.refuse)
        return decision

    def _apply_small_host_route(
        self,
        pipe: Any,
        plan: Any,
        decision: Any,
        target: Any,
        dtype: Any,
        fam: Any,
        logger: Any,
        load_token: Any = None,
    ) -> Any:
        """Convert the components loaded at their stored dtype and place the plan to keep them memory-mapped."""
        import torch

        from .diffusion_small_host import (
            cast_resident_,
            int8_act_device_ok,
            int8_act_family,
            mark,
            prepare_streamed_encoder_,
            quantize_int8_weight_,
        )

        device = target.torch_device
        resident = plan.offload_policy == OFFLOAD_NONE
        info: dict[str, Any] = {"reason": decision.reason, "components": {}}
        for name in decision.storage_dtypes:
            module = getattr(pipe, name, None)
            if not isinstance(module, torch.nn.Module):
                continue
            if resident:
                cast_resident_(module, device, dtype)
                info["components"][name] = "cast on device"
            elif name.startswith("text_encoder"):
                mib = prepare_streamed_encoder_(module, dtype)
                info["components"][name] = f"memory-mapped, layerwise cast ({mib} MiB)"
            elif name not in INT8_DENOISER_NAMES:
                cast_resident_(module, "cpu", dtype)
                info["components"][name] = "converted"
            else:
                # pageable: pinning rounds blocks to powers of two (11.3 GB of FLUX.1 int8 held 18 GB pinned)
                act_int8 = int8_act_family(fam) and int8_act_device_ok(device)
                stats = quantize_int8_weight_(
                    module,
                    compute_dtype = dtype,
                    work_device = device,
                    keep_device = "cpu",
                    act_int8 = act_int8,
                )
                info["components"][name] = (
                    f"int8 weights ({stats['int8_bytes'] >> 20} MiB, {stats['linears']} linears"
                    + (", int8 activations" if act_int8 else "")
                    + ")"
                )
            if load_token is not None:
                self._raise_if_load_cancelled(load_token)
        mark(pipe, info)
        logger.info("diffusion.small_host: %s", info["components"])
        return self._small_host_plan(pipe, plan, fam, logger)

    def _small_host_plan(self, pipe: Any, plan: Any, fam: Any, logger: Any) -> Any:
        """Group offload, routed encoders streamed; the denoiser keeps what fits after the runtime reserve resident."""
        info = small_host_engaged_on(pipe)
        if not info or plan.offload_policy == OFFLOAD_NONE:
            return plan
        if "small_host_loaded_transformer_mib" in plan.estimates and plan.stream_text_encoders:
            return plan  # already placed by the route
        from .diffusion_memory import (
            DEFAULT_BASE_OVERHEAD_MIB,
            OFFLOAD_GROUP,
            _loaded_component_mib,
        )

        sizes = _loaded_component_mib(pipe) or {}
        dit = sum(m for m, r in sizes.values() if r == "dit")
        other = sum(m for m, r in sizes.values() if r == "other")
        est = dict(plan.estimates)
        budget = est.get("safe_device_budget_mib")
        headroom = int(est.get("runtime_headroom_mib") or 0)
        overhead = int(est.get("base_overhead_mib") or DEFAULT_BASE_OVERHEAD_MIB)
        room = None if budget is None else int(budget) - headroom - overhead - other
        stream_dit = room is None or dit <= 0 or room < dit
        resident_dit = None if not stream_dit or room is None or room <= 0 else int(room)
        new = replace(
            plan,
            offload_policy = OFFLOAD_GROUP,
            stream_text_encoders = True,
            stream_transformer = stream_dit,
            resident_transformer_mib = resident_dit
            if plan.resident_transformer_mib is None
            else plan.resident_transformer_mib,
            estimates = {**est, "small_host_loaded_transformer_mib": dit},
            reasons = plan.reasons
            + (
                f"small-host route: encoders stream from memory-mapped storage; transformer {dit} MiB "
                + (
                    "resident"
                    if not stream_dit
                    else f"streams with {resident_dit or 0} MiB resident"
                ),
            ),
        )
        if (
            new.offload_policy != plan.offload_policy
            or new.stream_transformer != plan.stream_transformer
            or new.resident_transformer_mib != plan.resident_transformer_mib
        ):
            logger.info("diffusion.small_host: placement %s", new.reasons[-1])
        return new

    def _plan_memory(
        self,
        target: DiffusionDeviceTarget,
        single_file_path: Optional[str],
        base: str,
        fam: DiffusionFamily,
        memory_mode: Optional[str],
        cpu_offload: bool,
        *,
        kind: str = "gguf",
        repo_id: Optional[str] = None,
        transformer_resident_override_mib: Optional[int] = None,
        companion_override_mib: Optional[int] = None,
        text_encoder_override_mib: Optional[int] = None,
        base_local_dir: Optional[str] = None,
        fetch_base: Optional[str] = None,
        device_memory_override: Optional[DeviceMemory] = None,
        text_encoder_quant: Optional[str] = None,
        lora: bool = False,
    ):
        """Resident size per kind; cache scans must read fetch_base or companions fold to zero; base_local_dir adds to the scan, never replaces it."""
        single_file_fp8_stored_mib: Optional[int] = None
        single_file_resident_at: Any = (
            None  # (fp8_storage, offload) -> MiB, set for a single-file DiT
        )
        single_file_stored_at: Any = None
        device_memory = (
            device_memory_override
            if device_memory_override is not None
            else settled_snapshot_device_memory(target)
        )
        load_dtype = getattr(target, "dtype", None)
        if _float_load_itemsize(load_dtype) is None:
            load_dtype = None
        companions_from_cache = False
        supplied_te: tuple[str, ...] = ()
        if kind == "pipeline" and transformer_resident_override_mib is not None:
            # Dense-quant re-plan: skip the cache scan, which would price the bf16 transformer this replaces.
            companion_mib = companion_override_mib
            text_encoder_mib = text_encoder_override_mib
            model_dense_mib = transformer_resident_override_mib + (companion_mib or 0)
        elif kind == "pipeline":
            # Repo cached bytes give the resident estimate; a LOCAL path is not cached, so sum on-disk weights.
            local_repo = Path(repo_id).expanduser() if repo_id else None
            staged_repo = Path(base_local_dir).expanduser() if base_local_dir else None
            # Bytes sit under the STAGED repo, so scan the mirror, else an empty upstream cache reads as zero.
            cache_repo = fetch_base or repo_id
            # Largest source wins: a staged snapshot in the OTHER root, or a split prefetch, reads as unknown.
            cached = max(
                self._local_dir_weight_bytes(local_repo, exclude_transformer = False)
                if local_repo is not None and local_repo.is_dir()
                else 0,
                self._local_dir_weight_bytes(staged_repo, exclude_transformer = False)
                if staged_repo is not None and staged_repo.is_dir()
                else 0,
                self._cache_bytes(cache_repo, pipeline_only = mirror_hosts_artifacts(cache_repo))
                if cache_repo
                else 0,
            )
            if load_dtype is not None:
                # Cached bytes are what the repo STORES, not what the load holds at this dtype.
                loaded = max(
                    self._local_dir_weight_bytes(
                        local_repo, exclude_transformer = False, load_dtype = load_dtype
                    )
                    if local_repo is not None and local_repo.is_dir()
                    else 0,
                    self._local_dir_weight_bytes(
                        staged_repo, exclude_transformer = False, load_dtype = load_dtype
                    )
                    if staged_repo is not None and staged_repo.is_dir()
                    else 0,
                    self._union_over_cached_revs(
                        cache_repo,
                        lambda d: self._local_dir_weight_sizes(
                            d, exclude_transformer = False, load_dtype = load_dtype
                        ),
                    )
                    if cache_repo
                    else 0,
                )
                if loaded:
                    cached = loaded
            cached_mib = int(cached // (1024 * 1024)) if cached else None
            model_dense_mib = estimate_safetensors_dense_mib(cached_mib)
            # Repos storing narrower weights (ideogram-4 fp8) undershoot bf16 ~2x; use the size table.
            is_narrow_base = (
                bool(repo_id) and canonical_base(repo_id).lower() == fam.base_repo.strip().lower()
            )
            if (
                not is_narrow_base
                and fam.name == IDEOGRAM4_FAMILY_NAME
                and local_repo is not None
                and local_repo.is_dir()
            ):
                # Local fp8 mirror never matches base_repo by name; detect fp8 from shard headers (nf4 stays packed).
                is_narrow_base = ideogram4_repo_is_fp8(repo_id)
            table = None
            if is_narrow_base:
                table = family_bf16_components_gb(fam, fam.base_repo)
                if table is not None:
                    # Reserve bf16 size from this network-free constant, else model_dense_mib None reads resident: OOM.
                    table_mib = int(sum(table) * (1000.0**3) / (1024.0 * 1024.0))
                    model_dense_mib = (
                        table_mib if model_dense_mib is None else max(model_dense_mib, table_mib)
                    )
            # Companions sit in subfolders: split them like the GGUF branch, or group offload tiers fail on None.
            companion = self._companion_cache_bytes(fetch_base or base, base_local_dir, load_dtype)
            companion_mib = int(companion // (1024 * 1024)) if companion else None
            text_encoder = self._text_encoder_cache_bytes(
                fetch_base or base, base_local_dir, load_dtype
            )
            text_encoder_mib = int(text_encoder // (1024 * 1024)) if text_encoder else None
            companions_from_cache = True
            if is_narrow_base and table is not None:
                # Companions upcast with the raised bf16 model_dense_mib, so take their share from the same table.
                companion_mib = int(sum(table[1:]) * (1000.0**3) / (1024.0 * 1024.0))
                text_encoder_mib = int(table[1] * (1000.0**3) / (1024.0 * 1024.0))
                # Already the dense bf16 encoders, an upper bound on any pre-cast one: swapping would add it on top.
                companions_from_cache = False
            elif table is None:
                floored = _incomplete_release_cache_floor(
                    fam,
                    repo_id,
                    self._pipeline_source_complete(local_repo, staged_repo, cache_repo),
                    model_dense_mib,
                    companion_mib,
                    text_encoder_mib,
                )
                if floored is not None:
                    model_dense_mib, companion_mib, text_encoder_mib = floored
                    companions_from_cache = False
            if companion_mib is not None and model_dense_mib is not None:
                # The two terms merge different cache roots, so companions are clamped to the model size.
                companion_mib = min(companion_mib, model_dense_mib)
                if text_encoder_mib is not None:
                    text_encoder_mib = min(text_encoder_mib, companion_mib)
        else:
            if transformer_resident_override_mib is not None:
                # Planning the dense-quant candidate: the auto-policy estimate replaces the file-size derivation.
                transformer_resident = transformer_resident_override_mib
            elif kind == "single_file":

                def _single_file_resident(
                    fp8_storage: bool, offload: bool = False
                ) -> Optional[int]:
                    fp8_upcast = _single_file_fp8_upcast(single_file_path, fam)
                    if fp8_upcast and fp8_storage and fam.name == KREA2_FAMILY_NAME:
                        # storage keeps only OCP fp8 codes; FNUZ ones still widen
                        fp8_upcast = bool(
                            header_has_fp8_weights(str(single_file_path), fnuz_only = True)
                        )
                    mib = estimate_safetensors_dense_mib(
                        file_size_mib(single_file_path), fp8_upcast = fp8_upcast
                    )
                    if not getattr(fam, "single_file_is_pipeline", False):
                        # Priced from the header: layers a runtime keeps at stored size, dequantized ones at 2x.
                        _comfy_mib = self._comfy_single_file_resident_mib(
                            single_file_path,
                            fam,
                            target,
                            base,
                            lora = lora,
                            fp8_storage = fp8_storage,
                            offload = offload,
                        )
                        if _comfy_mib is not None:
                            mib = _comfy_mib
                    return mib

                def _stored_mib(widened: Optional[int], offload: bool = False) -> Optional[int]:
                    if not (comfy_fp8_storage(target, lora = lora) and single_file_path):
                        return None
                    stored = _single_file_resident(True, offload)
                    if stored is not None and widened is not None and stored < widened:
                        return stored
                    return None

                transformer_resident = _single_file_resident(False)
                single_file_fp8_stored_mib = _stored_mib(transformer_resident)
                single_file_resident_at = _single_file_resident
                single_file_stored_at = _stored_mib
            elif getattr(fam, "single_file_is_pipeline", False):
                # Only the denoiser's linears stay packed; the text encoders, VAE and the rest load dense.
                transformer_resident = whole_pipeline_gguf_resident_mib(
                    single_file_path, dense_bytes = _float_load_itemsize(load_dtype) or 2
                ) or estimate_gguf_resident_mib(file_size_mib(single_file_path))
            else:
                transformer_resident = estimate_gguf_resident_mib(file_size_mib(single_file_path))
            # Companions (VAE, text encoders) load near on-disk size: sum the base cache, or LOCAL weights.
            if companion_override_mib is not None:
                # Dense re-plan: prefetched transformer/ shards share the cache; the estimate avoids double count.
                companion_mib = companion_override_mib
                # Text-encoder share of that override, from the same table so the terms agree; None without a split.
                text_encoder_mib = text_encoder_override_mib
            else:
                # Scan the staged repo and its snapshot; a base in the import-time root is invisible to a live scan.
                companion = self._companion_cache_bytes(
                    fetch_base or base, base_local_dir, load_dtype
                )
                companion_mib = int(companion // (1024 * 1024)) if companion else None
                # Text-encoder share from the same walk as companions, so the subtraction is exact; 0 = no split.
                text_encoder = self._text_encoder_cache_bytes(
                    fetch_base or base, base_local_dir, load_dtype
                )
                text_encoder_mib = int(text_encoder // (1024 * 1024)) if text_encoder else None
                companions_from_cache = True
                # Price supplied files instead of the replaced components a cached base repo still holds.
                supplied = self._supplied_component_mib(
                    fetch_base or base, base_local_dir, load_dtype
                )
                if supplied is not None:
                    scanned_mib, scanned_te_mib, supplied_mib, supplied_te_mib = supplied
                    companion_mib = max(0, int(companion_mib or 0) - scanned_mib) + supplied_mib
                    text_encoder_mib = (
                        max(0, int(text_encoder_mib or 0) - scanned_te_mib) + supplied_te_mib
                    ) or None
                    supplied_te = _active_supplied_text_encoders()
            model_dense_mib = None
            if transformer_resident is not None:
                model_dense_mib = transformer_resident + (companion_mib or 0)
        if companions_from_cache and text_encoder_quant is not None:
            # Supplied encoders are priced above; only the slots they leave to a hosted pre-cast count here.
            precast = self._precast_text_encoder_mib(
                fam,
                base,
                target,
                text_encoder_quant,
                base_local_dir,
                **({"skip_components": supplied_te} if supplied_te else {}),
            )
            if precast:
                precast_mib, precast_components, _exact = precast
                covered = frozenset(precast_components)
                scanned_te = int(
                    self._union_over_cached_revs(
                        fetch_base or base,
                        lambda d: {
                            rel: size
                            for rel, size in self._local_dir_text_encoder_sizes(
                                d, load_dtype
                            ).items()
                            if rel.split("/", 1)[0] in covered
                        },
                        base_local_dir,
                    )
                ) // (1024 * 1024)
                from .diffusion_te_prequant import int8_convrot_te_runs_on

                # Never below scanned dense shards: a bad checkpoint falls back to opening them. Not on MPS, where
                # that failure raises instead (_te_prequant_pipe_kwargs, #11792). An automatic pick whose file is not
                # confirmed never gets here: _precast_text_encoder_mib prices nothing for it, so the dense terms stand.
                if not int8_convrot_te_runs_on(target):
                    precast_mib = max(int(precast_mib), scanned_te)
                text_encoder_mib = max(0, int(text_encoder_mib or 0) - scanned_te) + int(
                    precast_mib
                )
                companion_mib = max(0, int(companion_mib or 0) - scanned_te) + int(precast_mib)
                if model_dense_mib is not None:
                    model_dense_mib = max(0, int(model_dense_mib) - scanned_te) + int(precast_mib)
        # Variant hint lets estimate_image_runtime_mib see distilled markers (they need ~15% less headroom).
        variant_hint = _image_variant_hint(
            fam.name,
            Path(single_file_path).name if single_file_path else None,
            repo_id,
            base,
        )
        # Budgets the 1024x1024 default; generate() re-checks the real resolution.
        runtime_headroom = estimate_image_runtime_mib(width = None, height = None, family = variant_hint)

        def _plan(model_mib: Optional[int]) -> Any:
            return plan_diffusion_memory(
                target = target,
                device_memory = device_memory,
                model_dense_mib = model_mib,
                companion_dense_mib = companion_mib,
                text_encoder_dense_mib = text_encoder_mib,
                runtime_headroom_mib = runtime_headroom,
                requested_mode = memory_mode,
                explicit_offload = cpu_offload,
                calibrated_activation = _calibrated_activation(fam, target),
            )

        plan = _plan(model_dense_mib)
        if kind == "single_file" and single_file_path:
            self.__dict__.setdefault("_comfy_priced_offload", {})[str(single_file_path)] = False
        if (
            single_file_resident_at is not None
            and model_dense_mib is not None
            and transformer_resident is not None
            and not plan_keeps_transformer_resident(plan)
        ):
            # Offloaded: the loader probes the backends with offload=True (torchao fp8 does not survive the hooks), so
            # price what that load builds, widened and fp8-stored, before choosing storage.
            offloaded = single_file_resident_at(False, True)
            if offloaded is not None and offloaded != transformer_resident:
                # Smaller (the int8 / fp8 twin only offload runs): the load keeps those backends even if the new plan
                # fits resident, else it dequantizes past the size it was planned at.
                self.__dict__.setdefault("_comfy_priced_offload", {})[str(single_file_path)] = int(
                    offloaded
                ) < int(transformer_resident)
                model_dense_mib = model_dense_mib - int(transformer_resident) + int(offloaded)
                transformer_resident = offloaded
                single_file_fp8_stored_mib = single_file_stored_at(offloaded, True)
                plan = _plan(model_dense_mib)
        if single_file_fp8_stored_mib is not None and model_dense_mib is not None:
            # Re-plan at the fp8-stored size: a widened load that had to offload may now stay resident.
            from .diffusion_memory import _available_system_memory_mib

            fp8_rule = dict(
                lora = lora,
                offload = not plan_keeps_transformer_resident(plan),
                widened_mib = transformer_resident,
                host_available_mib = _available_system_memory_mib(),
            )
            store = fp8_storage_wanted(target, **fp8_rule)
            picks = self.__dict__.setdefault("_fp8_storage_picks", {})
            picks[str(single_file_path)] = store
            self.__dict__.setdefault("_fp8_widen_picks", {})[str(single_file_path)] = (
                not store and fp8_widen_chosen(target, **fp8_rule)
            )
            if store:
                plan = _plan(
                    model_dense_mib - int(transformer_resident) + single_file_fp8_stored_mib
                )
        elif kind == "single_file" and single_file_path:
            self.__dict__.setdefault("_fp8_storage_picks", {})[str(single_file_path)] = False
            self.__dict__.setdefault("_fp8_widen_picks", {})[str(single_file_path)] = False
        return plan

    @staticmethod
    def _from_pipe_no_recast(pipe: Any, pipe_cls: Any, **extra: Any) -> Any:
        """from_pipe recasts reused modules to float32, and a mid-way raise leaves earlier ones
        already cast."""
        return _no_recast_pipeline_class(pipe_cls).from_pipe(pipe, **extra)

    def _workflow_pipe(self, state: _LoadState, class_name: Optional[str], workflow: str) -> Any:
        """Built with from_pipe over the resident modules: no extra VRAM or reload; ValueError if
        unsupported."""
        if not class_name:
            raise ValueError(
                f"{workflow} is not supported for the '{state.family.name}' model family."
            )
        cached = self._aux_pipes.get(class_name)
        if cached is not None:
            return cached
        import diffusers

        pipe = self._from_pipe_no_recast(state.pipe, getattr(diffusers, class_name))
        prompt_cache.install(
            pipe,
            identity = {
                "family": state.family.name,
                "repo": str(state.repo_id),
                "workflow": class_name,
            },
            lora_owner = state.pipe,
            logger = logger,
        )
        # from_pipe runs without _lock: only publish if this load is still current.
        with self._lock:
            if self._state is state:
                self._aux_pipes[class_name] = pipe
        return pipe

    def _controlnet_pipe(self, state: _LoadState, resolved_cn: Any, cancel: threading.Event) -> Any:
        """Built from the resident pipe without recasting; ValueError if the family lacks ControlNet
        classes."""
        fam = state.family
        pipe_cls_name = getattr(fam, "controlnet_pipeline_class", None)
        model_cls_name = getattr(fam, "controlnet_model_class", None)
        if not pipe_cls_name or not model_cls_name:
            raise ValueError(f"ControlNet is not supported for the '{fam.name}' model family.")
        import diffusers

        cn_model = self._cn_models.get(resolved_cn.id)
        if cn_model is None:
            if cancel.is_set():
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            # resolve_controlnet skips the trust gate, so from_pretrained could run a malicious pickle; run the malware preflight (fails open); force safetensors if remote.
            remote_cn = not getattr(resolved_cn, "is_local", False)
            if remote_cn:
                from utils.security import evaluate_file_security
                _cn_fs = evaluate_file_security(resolved_cn.path, hf_token = state.hf_token or None)
                if _cn_fs.blocked:
                    raise ValueError(_cn_fs.reason)
            # Keep at most one ControlNet resident, else swapping ControlNets accumulates until OOM.
            if self._cn_models or self._cn_pipes:
                self._cn_models.clear()
                self._cn_pipes.clear()
                clear_gpu_cache()
            import torch

            # state.dtype is the display string ("bfloat16"), so pass the real dtype and avoid a float32 load.
            cn_dtype = getattr(torch, str(state.dtype).replace("torch.", ""), None)
            # Force safetensors for untrusted remote repos; if the Hub scan failed open, pickles still load.
            cn_from_pretrained_kwargs: dict[str, Any] = {"cache_dir": hub_cache_dir()}
            if remote_cn:
                cn_from_pretrained_kwargs["use_safetensors"] = True
            cn_model = getattr(diffusers, model_cls_name).from_pretrained(
                resolved_cn.path,
                torch_dtype = cn_dtype,
                token = state.hf_token or None,  # blank -> anonymous
                **cn_from_pretrained_kwargs,
            )
            if cancel.is_set():
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            # Follows the base offload policy; calibrated tiers budget only the base, so ControlNet must stream.
            calibrated = bool(getattr(state, "calibrated_placement", False))
            from .diffusion_multigpu import is_split

            # A split's resident fit also priced the base pipeline alone (one card would have offloaded it).
            if _controlnet_streams(state) and _offload_controlnet_module(
                cn_model, state.device, logger
            ):
                pass
            elif calibrated or is_split(state.pipe):
                del cn_model
                clear_gpu_cache()
                raise ValueError(
                    "This ControlNet could not be streamed from system memory, and loading it fully onto "
                    "the GPU beside the model could run out of memory. Reload the model with the balanced "
                    "memory mode to use ControlNet, or generate without it."
                )
            else:
                cn_model = cn_model.to(state.device)
            if cancel.is_set():
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            self._cn_models[resolved_cn.id] = cn_model
        key = (pipe_cls_name, resolved_cn.id)
        pipe = self._cn_pipes.get(key)
        if pipe is None:
            pipe = self._from_pipe_no_recast(
                state.pipe, getattr(diffusers, pipe_cls_name), controlnet = cn_model
            )
            # from_pipe copies components, not the base pipe's wrapped encode_prompt.
            prompt_cache.install(
                pipe,
                identity = {
                    "family": fam.name,
                    "repo": str(getattr(state, "repo_id", "")),
                    "workflow": pipe_cls_name,
                },
                lora_owner = state.pipe,
                logger = logger,
            )
            with self._lock:
                # Same race as the model cache: an unload may have cleared _cn_pipes while from_pipe ran.
                if cancel.is_set() or self._state is not state:
                    del pipe
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                self._cn_pipes[key] = pipe
        return pipe

    @staticmethod
    def _align_vae_dtype(pipe: Any, denoiser_attr: str = "transformer") -> None:
        """A prior txt2img decode can leave the VAE upcast to fp32; re-align it to the denoiser's dtype."""
        denoiser = getattr(pipe, denoiser_attr, None)
        vae = getattr(pipe, "vae", None)
        if denoiser is None or vae is None:
            return
        try:
            # Read dtype from parameters: a compiled nn.Module may hide .dtype; take the first floating one.
            target_dtype = next(
                (p.dtype for p in denoiser.parameters() if p.dtype.is_floating_point),
                None,
            )
            if target_dtype is None:
                return
            if next(vae.parameters()).dtype != target_dtype:
                vae.to(dtype = target_dtype)
        except (StopIteration, AttributeError, RuntimeError, TypeError):
            pass

    @staticmethod
    def _make_vae_encode_dtype_safe(pipe: Any) -> None:
        """Casts encode inputs to the VAE's dtype; img2img uses the text encoder's dtype, which can
        mismatch."""
        vae = getattr(pipe, "vae", None)
        if vae is None or getattr(vae, "_unsloth_dtype_safe_encode", False):
            return
        try:
            import torch

            original = vae.encode

            @functools.wraps(original)
            def _encode(x: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    if isinstance(x, torch.Tensor):
                        target = next(
                            (p.dtype for p in vae.parameters() if p.dtype.is_floating_point),
                            None,
                        )
                        if target is not None and x.dtype != target:
                            x = x.to(dtype = target)
                except (AttributeError, RuntimeError, StopIteration, TypeError):
                    pass
                return original(x, *args, **kwargs)

            vae.encode = _encode
            vae._unsloth_dtype_safe_encode = True
        except (AttributeError, RuntimeError, TypeError):
            pass

    @staticmethod
    def _resolve_lora_set(
        specs: list[tuple[str, float]],
        *,
        family: Optional[str],
        hf_token: Optional[str],
        cancel: Optional[threading.Event] = None,
    ) -> tuple[tuple[str, str, float], ...]:
        """Resolve (id, weight) specs to a ``(name, path, weight)`` tuple set for diffusers. Shared
        by the generation-time apply path and the quant load-time bake so both produce IDENTICAL
        tuples for the same request (the no-op / weight-only comparisons depend on it)."""
        from core.inference import diffusion_lora

        resolved = diffusion_lora.resolve_specs(
            specs,
            family = family,
            hf_token = hf_token,
            cancel_event = cancel,
        )
        # diffusers load_lora_weights takes safetensors only; reject a .gguf adapter as a clean 400.
        bad = [r.id for r in resolved if r.fmt != "safetensors"]
        if bad:
            raise ValueError(
                "GGUF LoRA adapters are not supported on the diffusers engine "
                f"({', '.join(bad)}); use a .safetensors adapter, or the native engine."
            )
        # Unique adapter names (diffusers requires distinct names; sanitized stems can collide).
        uniq: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for r in resolved:
            name = r.alias
            n = 1
            while name in seen:
                n += 1
                name = f"{r.alias}_{n}"
            seen.add(name)
            uniq.append((name, r.path, r.weight))
        return tuple(uniq)

    def _apply_loras(
        self, state: Any, loras: Optional[list[tuple[str, float]]], cancel: threading.Event
    ) -> None:
        """Unfused: fusing breaks quantised transformers; baked adapters need a reload to add or remove."""
        from core.inference import diffusion_lora

        pipe = state.pipe
        current = getattr(pipe, "_unsloth_loras", ())
        specs = [(i, w) for (i, w) in (loras or []) if w != 0]

        if specs:
            _refuse_lora_without_loader(getattr(state, "family", None), specs)
        quant_baked = bool(getattr(pipe, "_unsloth_loras_baked", False))
        quant = (state.transformer_quant or "").lower()
        if quant in ("int8", "fp8", "nvfp4", "mxfp8"):
            self._adjust_baked_loras(state, pipe, specs, current, quant_baked, cancel)
            return

        if not specs:
            if current:
                try:
                    pipe.unload_lora_weights()
                except Exception:  # noqa: BLE001 -- best-effort clear
                    pass
                pipe._unsloth_loras = ()
                cuda_graph.reset_all(getattr(state, "cuda_graphs", ()))
                invalidate_cfg_replica(pipe, "the LoRA adapters changed", logger)
            return

        if _small_host_int8(pipe):
            raise ValueError(
                "LoRA is not available on this load: system RAM was too small for the dense model, so Unsloth "
                "stored the transformer as int8 weights (small-host route), which cannot carry adapters. Load the "
                "model with the LoRA selected (Unsloth then keeps the dense transformer), or use a host with more RAM."
            )
        if comfy_block_runtime_layers(getattr(pipe, "transformer", None)):
            raise ValueError(
                "LoRA is not available on this load: the ComfyUI file's nvfp4 / mxfp8 layers run on their own "
                "quantized Linears, which cannot carry adapters. Load the file with the LoRA selected (Unsloth then "
                "dequantizes those layers), or set UNSLOTH_DIFFUSION_COMFY_NVFP4=0 / UNSLOTH_DIFFUSION_COMFY_MXFP8=0."
            )
        if fp8_stored_layers(getattr(pipe, "transformer", None)):
            raise ValueError(
                "LoRA is not available on this load: the file's fp8 layers are stored as fp8 and widened per layer "
                "(this GPU has no fp8 GEMM), which cannot carry adapters. Load the file with the LoRA selected "
                "(Unsloth then widens those layers at load), or set UNSLOTH_DIFFUSION_FP8_STORAGE=0."
            )
        if not diffusion_lora.supports_lora(
            engine = "diffusers",
            family = getattr(state.family, "name", None),
            model_kind = state.kind,
            transformer_quant = state.transformer_quant,
            compiled = "compiled" in (getattr(state, "speed_optims", ()) or ()),
        ):
            # Name only routes the user can actually reach: a GPU host never selects the native engine on its own
            raise ValueError(
                "LoRA is not supported for this model/quantisation on the diffusers engine "
                "(GGUF-via-diffusers, or a torch.compile'd Speed=default/max load). Reload with "
                "transformer_quant int8 or fp8, which rebuilds the GGUF into a LoRA-capable dense "
                "transformer, or use a bf16 / bnb-4bit load at Speed=off/eager. To keep the GGUF "
                "weights themselves, reload it on the native engine (Advanced > Engine > Native, "
                "or engine=sd_cpp on the load), which a GPU host does not select on its own."
            )

        desired = self._resolve_lora_set(
            specs,
            family = getattr(state.family, "name", None),
            hf_token = state.hf_token,
            cancel = cancel,
        )
        uniq = list(desired)
        if desired == current:
            return
        try:
            if current:
                pipe.unload_lora_weights()
            for name, path, _weight in uniq:
                pipe.load_lora_weights(path, adapter_name = name)
            pipe.set_adapters(
                [name for name, _p, _w in uniq], adapter_weights = [w for _n, _p, w in uniq]
            )
        except Exception as exc:  # noqa: BLE001 -- surface as a clean 400
            try:
                pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
            pipe._unsloth_loras = ()
            # Reset graphs too, else a base-model render replays the removed adapters.
            cuda_graph.reset_all(getattr(state, "cuda_graphs", ()))
            invalidate_cfg_replica(pipe, "the LoRA adapters changed", logger)
            raise ValueError(f"Failed to apply LoRA: {exc}") from exc
        pipe._unsloth_loras = desired
        # load_lora_weights can re-materialise parameters the graphs baked pointers to.
        cuda_graph.reset_all(getattr(state, "cuda_graphs", ()))
        invalidate_cfg_replica(pipe, "the LoRA adapters changed", logger)

    def _adjust_baked_loras(
        self,
        state: Any,
        pipe: Any,
        specs: list[tuple[str, float]],
        current: tuple,
        quant_baked: bool,
        cancel: threading.Event,
    ) -> None:
        """Generation-time LoRA handling for a torchao-quantized pipe. The adapters (if any) were
        baked at load time, before quantize_ + compile, so the module topology is immutable here.
        Allowed without a reload: weight tweaks on the baked set and disabling any of it (scale 0
        reproduces the quantized base exactly; set_adapters is value-level, so torch.compile guards
        absorb it). Anything that would change topology raises a clean 400 telling the client to
        reload with the new selection."""
        if not quant_baked:
            if not specs:
                return  # no adapters baked, none requested
            raise ValueError(
                "This quantized (int8/fp8) load was built without LoRA adapters. Reload the "
                "model with the adapter selection to bake it into the quantized transformer."
            )
        if not specs:
            # Disable every baked adapter: scale 0 reproduces the quantized base exactly.
            names = [n for (n, _p, _w) in current]
            if any(w != 0 for (_n, _p, w) in current):
                _refuse_weight_change_on_streamed_graph(state)
                disabled = tuple((n, p, 0.0) for (n, p, _w) in current)
                # Pinned scales are tensors the compiled code and the graphs read at run time: no recompile, no recapture.
                if _set_pinned_lora_scales(pipe, {}):
                    pipe._unsloth_loras = disabled
                    return
                pipe.set_adapters(names, adapter_weights = [0.0] * len(names))
                pipe._unsloth_loras = disabled
                # Adapter scales are Python values read inside the LoRA forward, so a graph bakes them in.
                cuda_graph.reset_all(getattr(state, "cuda_graphs", ()))
                invalidate_cfg_replica(pipe, "the LoRA adapters changed", logger)
            return
        desired = self._resolve_lora_set(
            specs,
            family = getattr(state.family, "name", None),
            hf_token = state.hf_token,
            cancel = cancel,
        )
        # Requests drop zero-weight rows, so ``desired`` is the active subset of the bake. Match by file: names
        # are deduplicated per resolved set, so a subset can rename an adapter. Omitted adapters run at 0.
        wanted: dict[str, list[float]] = {}
        for _n, path, weight in desired:
            wanted.setdefault(path, []).append(weight)
        updated = tuple(
            (name, path, wanted[path].pop(0) if wanted.get(path) else 0.0)
            for (name, path, _w) in current
        )
        if any(wanted.values()):
            raise ValueError(
                "The LoRA selection changed, but a quantized (int8/fp8) transformer bakes its "
                "adapters at load time. Reload the model with the new adapter selection."
            )
        if updated == current:
            return
        _refuse_weight_change_on_streamed_graph(state)
        if _set_pinned_lora_scales(pipe, {n: w for (n, _p, w) in updated}):
            pipe._unsloth_loras = updated
            return
        pipe.set_adapters(
            [n for (n, _p, _w) in updated],
            adapter_weights = [w for (_n, _p, w) in updated],
        )
        pipe._unsloth_loras = updated
        cuda_graph.reset_all(getattr(state, "cuda_graphs", ()))
        invalidate_cfg_replica(pipe, "the LoRA adapters changed", logger)

    @staticmethod
    def _reset_step_cache(pipe: Any) -> None:
        """Clears FBCache residuals a failed call left on the transformer via _reset_stateful_cache; reset_stateful_hooks is a silent no-op, best-effort."""
        transformer = getattr(pipe, "transformer", None)
        reset = getattr(transformer, "_reset_stateful_cache", None) or getattr(
            transformer, "reset_stateful_hooks", None
        )
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 - reset is best-effort, never fail a generation
                pass

    def _engage_deferred_speed(self, state: _LoadState) -> None:
        """The deferred `default` profile plus its auto static skip; a failed profile takes the skip back off."""
        object.__setattr__(state, "speed_deferred", False)
        plan = state.deferred_static_plan
        object.__setattr__(state, "deferred_static_plan", None)
        tc = (state.resolved or {}).get("transformer_cache")
        tc_before = dict(tc) if isinstance(tc, dict) else None
        installed = False
        # Before compile, as at load: the skip is an outer forward on the transformer.
        if plan and state.transformer_cache is None:
            installed = (
                install_static_step_skip(
                    state.pipe, settings = auto_static_settings(plan, logger = logger), logger = logger
                )
                == TC_STATIC
            )
            if installed:
                object.__setattr__(state, "transformer_cache", TC_STATIC)
                if isinstance(tc, dict):
                    tc["value"] = TC_STATIC
                    tc["reason"] = (
                        f"auto: static step skip (every {plan['every']}) for this model at "
                        f"{plan['min_steps']}+ steps, engaged with the compiled profile; set "
                        "UNSLOTH_DIFFUSION_AUTO_STEP_SKIP=0 to turn it off"
                    )
        try:
            self._engage_deferred_profile(state)
        except BaseException:
            invalidate_cfg_replica(state.pipe, "the deferred speed profile failed part way", logger)
            if installed:
                uninstall_static_step_skip(state.pipe)
                object.__setattr__(state, "transformer_cache", None)
                if isinstance(tc, dict) and tc_before is not None:
                    tc.clear()
                    tc.update(tc_before)
            raise
        # the profile patched / compiled the source denoiser: a replica built during the eager images is stale
        invalidate_cfg_replica(
            state.pipe, "the deferred speed profile changed the denoiser", logger
        )
        drop_displaced_cfg_router(
            state.pipe, "the deferred speed profile replaced the denoiser's forward", logger
        )

    # Outside load_pipeline (3rd generation): its own probe scope, so one OOM is not re-probed by plan and apply.
    @rocm_probe_load_scoped
    def _engage_deferred_profile(self, state: _LoadState) -> None:
        """Runs the deferred speed profile at the 3rd generation under _generate_lock; the flag clears first so it never retries."""
        object.__setattr__(state, "speed_deferred", False)
        from . import diffusion_autoopt as autoopt

        target = self._state_device_target(state)
        audit = autoopt.new_audit(backend = "image", family = state.family)
        autoopt.install_patches(
            autoopt.STAGE_LOAD_EAGER,
            pipe = state.pipe,
            target = target,
            family = state.family,
            speed_mode = SPEED_DEFAULT,
            backend = "image",
            audit = audit,
            logger = logger,
        )
        object.__setattr__(state, "eager_patched", True)
        # Re-run the load-time selection with the caller's ORIGINAL request: auto still upgrades to cuDNN here.
        attention_engaged = apply_attention_backend(
            state.pipe,
            select_attention_backend(
                target, state.attention_request, speed_active = True, family = state.family
            ),
            logger = logger,
            target = target,
        )
        object.__setattr__(state, "attention_backend", attention_engaged)
        identity = getattr(state.pipe, "_unsloth_peak_identity", None)
        if isinstance(identity, tuple) and len(identity) == 3:
            # later renders record under the backend that now runs
            try:
                # the profile engages the default tier too (state.speed_mode below)
                state.pipe._unsloth_peak_identity = (
                    identity[0],
                    with_label_part(identity[1], "attn", attention_engaged or "native"),
                    SPEED_DEFAULT,
                )
            except Exception:  # noqa: BLE001 - recording is optional
                pass
            # reserve was placed for the load's tier: flat estimate until this tier records its own peak
            if tier_peaks_enabled():
                setattr(state.pipe, TIER_MOVED_ATTR, True)
            else:
                state.pipe._unsloth_peak_static = True
        # Keep the badge in step with the state it describes: the top-level field moved, so leaving `resolved` on the
        # load-time value would make the panel report an attention backend that is no longer the one running.
        if isinstance(state.resolved, dict) and "attention_backend" in state.resolved:
            entry = dict(state.resolved["attention_backend"])
            entry["value"] = attention_engaged or "native"
            object.__setattr__(state, "resolved", {**state.resolved, "attention_backend": entry})
        gguf_transformer = state.kind == "gguf" and state.transformer_quant is None
        if _compile_cache_wanted(
            state.pipe,
            target,
            state.family,
            is_gguf = gguf_transformer,
            speed_mode = SPEED_DEFAULT,
            offload_active = state.offload_policy != OFFLOAD_NONE,
        ):
            compile_ctx = compile_cache.begin(
                family = state.family.name,
                # U-Net families (SDXL) carry the denoiser as pipe.unet.
                transformer = getattr(state.pipe, "transformer", None)
                or getattr(state.pipe, "unet", None),
                dtype = getattr(target, "dtype", None),
                # Same GGUF-vs-dense graph distinction as the load-time begin().
                quant = "gguf" if gguf_transformer else state.transformer_quant,
                attention_backend = attention_engaged,
                # Mirrors the load-time fullgraph decision: a step cache or an offload graph-breaks.
                compile_kwargs = autoopt.compile_cache_kwargs(
                    state.pipe,
                    speed_mode = SPEED_DEFAULT,
                    deferred = True,
                    fullgraph = not cache_breaks_graph(state.transformer_cache)
                    and not state.cache_auto
                    and state.offload_policy == OFFLOAD_NONE,
                    offload_below_hooks = state.offload_policy != OFFLOAD_NONE
                    and compile_below_hooks_enabled(),
                ),
                logger = logger,
                reduction_filter = family_filters_reductions(state.family),
            )
            object.__setattr__(state, "compile_cache_ctx", compile_ctx)
        speed_applied = autoopt.run_speed_layer(
            state.pipe,
            target,
            family = state.family,
            backend = "image",
            speed_mode = SPEED_DEFAULT,
            is_gguf = gguf_transformer,
            offload_active = state.offload_policy != OFFLOAD_NONE,
            audit = audit,
            logger = logger,
            apply = lambda family, log: apply_speed_optims(
                state.pipe,
                target,
                is_gguf = gguf_transformer,
                family = family,
                speed_mode = SPEED_DEFAULT,
                cache_active = cache_breaks_graph(state.transformer_cache) or state.cache_auto,
                cache_engaged = cache_breaks_graph(state.transformer_cache),
                offload_active = state.offload_policy != OFFLOAD_NONE,
                denoiser_offloaded = state.offload_policy != OFFLOAD_NONE
                and _denoiser_hooked(state.pipe),
                stream_int8_gemm = True,
                logger = log,
            ),
        )
        if denoisers_pinned_resident(state.pipe):
            engage_pinned_denoisers(state.pipe, speed_applied, logger)
        if _denoiser_hooked(state.pipe):
            compile_pipe_below_offload_hooks(state.pipe, logger)
        arm_graphs_after_placement(state.pipe, speed_applied, logger)
        # the speed layer just rewrapped vae.decode (a compiled decode guard): the OOM retry's scope goes back on top
        try:
            from .diffusion_vae_oom import ensure_outermost
            ensure_outermost(getattr(state.pipe, "vae", None), logger)
        except Exception as exc:  # noqa: BLE001 - a decode OOM then fails as before
            logger.warning(
                "diffusion.vae_oom: scope not restored after the deferred profile: %s", exc
            )
        cuda_graph.arm_block_graphs(
            state.pipe,
            speed_applied,
            target = target,
            family = state.family,
            hooked = _denoiser_hooked(state.pipe),
            pinned = denoisers_pinned_resident(state.pipe),
            cache_engaged = cache_breaks_graph(state.transformer_cache),
            speed_mode = SPEED_DEFAULT,
            logger = logger,
        )
        if getattr(getattr(state.pipe, "vae", None), "_unsloth_fp16_decode", False):
            speed_applied["vae_fp16_decode"] = True
        object.__setattr__(state, "speed_mode", SPEED_DEFAULT)
        object.__setattr__(state, "speed_optims", tuple(k for k, v in speed_applied.items() if v))
        object.__setattr__(state, "speed_auto_compiled", bool(speed_applied.get("compiled")))
        object.__setattr__(
            state, "cuda_graphs", tuple(getattr(state.pipe, "_unsloth_cuda_graphs", ()) or ())
        )
        _follow_live_graph_mode(state)
        entry = (state.resolved or {}).get("speed_mode")
        if isinstance(entry, dict):
            entry["value"] = SPEED_DEFAULT
            entry["reason"] = (
                "auto: compiled on the 3rd image this session "
                "(repeated use amortises the one-time compile)"
            )
        autoopt.record_load_steps(
            audit,
            attention = attention_engaged,
            step_cache = state.transformer_cache,
            step_cache_reason = ((state.resolved or {}).get("transformer_cache") or {}).get("reason"),
            compile_cache = state.compile_cache_ctx is not None,
        )
        autoopt.finalize(audit, state.speed_optims, state.resolved)
        att = (state.resolved or {}).get("attention_backend")
        if isinstance(att, dict) and att.get("source") == "auto":
            att["value"] = attention_engaged or "native"
            att["reason"] = auto_attention_reason(attention_engaged)
        # Re-derive cuda_graph as the load does, else the badge keeps 'off' after a profile arms graphs.
        graph = (state.resolved or {}).get("cuda_graph")
        if isinstance(graph, dict):
            graph["value"] = "on" if speed_applied.get("cuda_graph") else "off"
            graph["reason"] = cuda_graph.status_reason(
                state.pipe, bool(speed_applied.get("cuda_graph"))
            )
        bg_module = _bg_compile_module(
            state.pipe,
            speed_optims = state.speed_optims,
            speed_mode = SPEED_DEFAULT,
            transformer_quant = state.transformer_quant,
            gguf_transformer = gguf_transformer,
            offload_policy = state.offload_policy,
            transformer_cache = state.transformer_cache,
            cache_auto = state.cache_auto,
            target = target,
        )
        if bg_module is not None:
            object.__setattr__(
                state,
                "bg_compile",
                bg_compile.arm(
                    bg_module,
                    logger = logger,
                    seed_dir = getattr(state.compile_cache_ctx, "dir", None),
                ),
            )
        logger.info(
            "diffusion.speed: deferred profile engaged on generation 3 (optims=%s, attention=%s%s)",
            ",".join(state.speed_optims) or "none",
            attention_engaged or "native",
            "; compiling in the background, this render runs eager"
            if state.bg_compile is not None
            else "",
        )

    @_release_render_on_unload
    @holds_inference
    @with_hires
    @vram_budget_scope("image")
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        # None on both (unified edit only) matches Image 1's aspect ratio.
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        # Fallbacks; the route always sends the per-model values the UI seeds.
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        # Prompts (txt2img only) or seeds: one image each; own Generator keeps same-seed repeats identical.
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        # init alone = img2img, init + mask = inpaint; strength 0 keeps source, 1 redraws.
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        # Upscale (hires fix): factor > 1 with an init image enlarges then re-denoises at low strength.
        upscale: Optional[float] = None,
        # Reference / unified edit: additional images after init_image, in order. The family bounds the total.
        reference_images: Optional[list[str]] = None,
        # Explicit "edit" / "reference"; None keeps the workflow the other arguments imply.
        workflow: Optional[str] = None,
        reference_resolution: Optional[int] = None,
        localized_edit: Optional[LocalizedEdit] = None,
        loras: Optional[list[tuple[str, float]]] = None,
        # ControlNet (id, control_image_b64, control_type, strength, guidance_start, guidance_end). None = off.
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
        # load_identity() of the caller's status() read; refuse rather than run a different load (#9448)
        expected_load: Optional[LoadIdentity] = None,
        allow_oversized: bool = False,
        # None = on unless UNSLOTH_DIFFUSION_PREVIEW=0.
        live_preview: Optional[bool] = None,
        sampler: Optional[str] = None,
        schedule_type: Optional[str] = None,
        sd_controls: Optional["sd_controls_mod.SdControls"] = None,
        shift: Optional[float] = None,
        base_shift: Optional[float] = None,
        max_shift: Optional[float] = None,
        # Inpaint only: None / 0 / False keep the plain inpaint call.
        mask_blur: Optional[int] = None,
        inpaint_only_masked: Optional[bool] = None,
        inpaint_padding: Optional[int] = None,
        # Upscale workflow: a model upscaler id (diffusion_upscalers) in place of the Lanczos resize. None = Lanczos.
        upscale_model: Optional[str] = None,
        _hires_pass: Optional[HiresPass] = None,
    ) -> dict[str, Any]:
        import torch
        from contextlib import nullcontext

        from PIL import Image

        # Per-generation cancel Event that unload()/a superseding load set (under _lock) to abort just this denoise.
        cancel = _hires_pass.cancel if _hires_pass is not None else threading.Event()
        from core.inference.generation_ticket import bind_generation_cancel

        if _hires_pass is None:
            bind_generation_cancel(cancel)
        with nullcontext() if _hires_pass is not None else self._generation_slot(cancel):
            # Never render beside a background compile; wait outside the state lock so status polls keep answering.
            pending_bg = getattr(self._state, "bg_compile", None)
            if pending_bg is not None and pending_bg.compiling():
                logger.info(
                    "diffusion.bg_compile: render waits for the background compile to finish"
                )
                waited = pending_bg.wait(cancel)
                logger.info(
                    "diffusion.bg_compile: render waited %.1f s for the background compile", waited
                )
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # The slot admits on a zero fence, which a COMMITTED replacement also satisfies (#9448)
                loaded_id = load_identity(state.repo_id, state.base_repo, state.family.name)
                if expected_load is not None and expected_load != loaded_id:
                    raise DiffusionModelReplacedError(expected_load, loaded_id)
                # Publish active step-0 state before slow setup, so a reload's mount probe does not read idle.
                self._gen = _GenState(total_steps = steps)
                begin_attention_run(state)
            # Reset in the finally, so a failed or cancelled generation frees its reused outputs.
            static_skip_pipe = None
            restore_scheduler: Optional[tuple[Any, Any]] = None
            applied_embeddings: Optional[diffusion_embeddings.AppliedEmbeddings] = None
            applied_shift: Optional[float] = None
            restore_vae: Optional[Callable[[], None]] = None
            restore_resident: Optional[Callable[[], None]] = None
            bg = None
            # Kicked from the finally below: an offloaded denoiser's placement (restore_resident) must be back first.
            kick_bg = False
            bg_eager = False
            bg_token = None
            # A Detailer crop inpaint (diffusion_bg_compile.auxiliary_pass): eager, uncounted, nothing recorded.
            aux_pass = bg_compile.auxiliary()
            restore_eager_gate: Optional[Callable[[], None]] = None
            try:
                self._state_device_target(state)
                # Local `state` keeps the pipe alive past unload() nulling _state; resolve (prompt, seed) jobs first.
                jobs, seed = resolve_batch_jobs(
                    prompt = prompt,
                    prompts = prompts,
                    seed = seed,
                    seeds = seeds,
                    batch_size = batch_size,
                    draw_seed = torch.Generator(device = state.device).seed,
                )
                # A batch keeps ONE base seed so it renders variations of a single image.
                variation_seed_list: Optional[list[int]] = None
                if sd_controls is not None and sd_controls.variation_strength:
                    if seeds is None and prompts is None:
                        jobs = [(p, seed) for p, _ in jobs]
                    base_variation = (
                        int(sd_controls.variation_seed)
                        if sd_controls.variation_seed is not None
                        else secrets.randbelow(2**32)
                    )
                    variation_seed_list = sd_controls_mod.variation_seeds(base_variation, len(jobs))

                # Deferred speed auto compiles on the 3rd image, but never with LoRA (compile rejects LoRA).
                lora_requested = _has_active_lora(loras)
                # Stay eager while prior adapters are attached: compiling would bake them in.
                loras_attached = bool(getattr(state.pipe, "_unsloth_loras", ()))
                if (
                    state.speed_deferred
                    and state.generation_count >= 2
                    and not aux_pass
                    and not lora_requested
                    and not loras_attached
                ):
                    try:
                        self._engage_deferred_speed(state)
                    except Exception as exc:  # noqa: BLE001 - speed is best-effort
                        logger.warning(
                            "diffusion.speed: deferred engagement failed, staying eager: %s",
                            exc,
                        )

                # Compile still pending: this generation runs eager and records the denoiser inputs for it.
                bg = state.bg_compile
                bg_eager = bg is not None and bg.pending()
                if bg_eager:
                    bg_token = bg_compile._FORCE_EAGER.set(True)
                if aux_pass:
                    # The force-eager signal never reaches a whole-module compiled U-Net on its own.
                    restore_eager_gate = bg_compile.gate_compiled_eager(
                        cuda_graph._denoiser_modules(state.pipe)
                    )

                # Apply/adjust LoRA before picking the workflow pipe; from_pipe pipes share the transformer.
                self._apply_loras(state, loras, cancel)

                pipe = state.pipe
                fam = state.family
                init_pil = mask_pil = None
                control_pil = None
                cn_scale = cn_gstart = cn_gend = cn_mode = None
                ref_extra: list = []
                requested_workflow = workflow
                if requested_workflow not in (None, "edit", "reference"):
                    raise ValueError(f"Unknown workflow '{requested_workflow}'.")
                if init_image is None:
                    if mask_image is not None:
                        raise ValueError("mask_image requires an input image (init_image).")
                    if upscale is not None and upscale > 1.0:
                        raise ValueError("upscale requires an input image (init_image).")
                    if reference_images:
                        raise ValueError("reference_images require an input image (init_image).")
                    if requested_workflow is not None:
                        raise ValueError(
                            f"The {requested_workflow} workflow requires a source image (init_image)."
                        )
                check_conditioned_fields(
                    requested_workflow,
                    fam,
                    mask_image = mask_image,
                    strength = strength,
                    upscale = upscale,
                    controlnet = controlnet,
                    localized_edit = localized_edit,
                )
                if (
                    reference_images
                    and not getattr(fam, "reference", False)
                    and not edit_takes_extra_images(fam)
                ):
                    raise ValueError(
                        f"Reference images are not supported for the '{fam.name}' model family."
                    )
                conditioned = False
                if getattr(fam, "edit", False):
                    if requested_workflow == "reference":
                        raise ValueError(
                            f"The reference workflow is not supported for the '{fam.name}' "
                            "model family."
                        )
                    if init_image is None:
                        raise ValueError(
                            f"{fam.name} is an image-editing model: provide an input image."
                        )
                    if mask_image is not None:
                        raise ValueError(
                            f"{fam.name} is an image-editing model and does not "
                            "support masks (mask_image)."
                        )
                    workflow = "edit"
                    if any(reference_images or ()):
                        # Multi-image edit (Qwen-Image-Edit-Plus): the source, then each extra image in request order,
                        # bounded and budgeted like the reference workflow's. The output still takes the source's size.
                        images_in = decode_condition_images(fam, init_image, reference_images)
                        init_pil, ref_extra = images_in[0], images_in[1:]
                    else:
                        # RGB for the instruction editors; the layered VAE reads the alpha as a fourth channel.
                        init_pil = decode_b64_image(
                            init_image, mode = getattr(fam, "condition_image_mode", "RGB") or "RGB"
                        )
                elif requested_workflow == "edit":
                    if not (
                        getattr(fam, "unified_edit", False) or getattr(fam, "reference_edit", False)
                    ):
                        raise ValueError(
                            f"Instruction editing is not supported for the '{fam.name}' "
                            "model family."
                        )
                    workflow = "edit"
                    conditioned = True
                elif requested_workflow == "reference":
                    if not getattr(fam, "reference", False):
                        raise ValueError(
                            f"The reference workflow is not supported for the '{fam.name}' "
                            "model family."
                        )
                    workflow = "reference"
                    conditioned = True
                elif _hires_pass is not None and _hires_pass.refine:
                    workflow = "upscale"
                    pipe = self._workflow_pipe(state, fam.img2img_pipeline_class, workflow)
                    init_pil = _hires_pass.source
                    steps = diffusion_hires.refine_inference_steps(
                        steps, float(strength), diffusion_hires.timestep_rule(pipe)
                    )
                elif mask_image is not None and init_image is not None:
                    workflow = "inpaint"
                    pipe = self._workflow_pipe(state, fam.inpaint_pipeline_class, workflow)
                    init_pil = decode_b64_image(init_image, mode = "RGB")
                    mask_pil = decode_b64_image(mask_image, mode = "L")
                elif init_image is not None and upscale is not None and upscale > 1.0:
                    # Upscale (hires fix): enlarge with Lanczos, then re-run img2img at low strength to add detail.
                    workflow = "upscale"
                    pipe = self._workflow_pipe(state, fam.img2img_pipeline_class, workflow)
                    init_pil = decode_b64_image(init_image, mode = "RGB")
                    iw, ih = init_pil.size
                    # Cap factor, then longest side 2048; round to 16 (VAE downsample + patch).
                    factor = max(1.0, min(float(upscale), 4.0))
                    tw_f, th_f = iw * factor, ih * factor
                    max_side = 2048
                    fit = min(1.0, max_side / max(tw_f, th_f))
                    tw = max(16, int(round(tw_f * fit / 16.0)) * 16)
                    th = max(16, int(round(th_f * fit / 16.0)) * 16)
                    # After the cap, the target must still exceed the input (else upscale shrinks it).
                    if max(tw, th) <= max(iw, ih):
                        raise ValueError(
                            f"Upscale would not enlarge this image: its longest side "
                            f"({max(iw, ih)}px) already meets the {max_side}px output limit. "
                            f"Use a smaller source image."
                        )
                    if upscale_model:
                        # ESRGAN-family model, then resized to the target (ComfyUI ImageUpscaleWithModel -> ImageScale).
                        init_pil = diffusion_upscalers.upscale_with_model(
                            init_pil,
                            upscale_model,
                            (tw, th),
                            device = state.device,
                            keep_resident = getattr(state, "offload_policy", None)
                            in (None, OFFLOAD_NONE),
                            cancel = cancel,
                        ).image
                    else:
                        init_pil = init_pil.resize((tw, th), Image.LANCZOS)
                    upscale = round(
                        max(tw / iw, th / ih), 2
                    )  # the factor after both caps, for the log
                    if strength is None:
                        strength = 0.35  # hires-fix default: preserve content, add detail
                elif upscale_model:
                    raise ValueError(
                        "upscale_model applies only to the Upscale workflow (init_image + upscale > 1)."
                    )
                elif getattr(fam, "reference", False) and init_image is not None:
                    workflow = "reference"
                    conditioned = True
                elif init_image is not None:
                    workflow = "img2img"
                    pipe = self._workflow_pipe(state, fam.img2img_pipeline_class, workflow)
                    init_pil = decode_b64_image(init_image, mode = "RGB")
                else:
                    workflow = "txt2img"
                ref_resolution = None
                if conditioned:
                    images_in = decode_condition_images(
                        fam, init_image, reference_images, localized_edit
                    )
                    init_pil, ref_extra = images_in[0], images_in[1:]
                    ref_resolution = effective_reference_resolution(fam, reference_resolution)
                elif reference_resolution is not None:
                    raise ValueError(
                        "reference_resolution applies only to the reference and edit workflows."
                    )
                source_sized = _is_source_sized(workflow, fam)
                if not source_sized:
                    if width is None or height is None:
                        if not conditioned:
                            raise ValueError("width and height are required for this workflow.")
                        width, height = match_source_size(
                            fam, init_pil.size, ref_resolution or 1024
                        )
                    check_output_size(fam, int(width), int(height))
                elif width is None or height is None:
                    width = height = 1024

                # ControlNet (diffusers): txt2img only. Builds the family CN pipeline around resident modules.
                if controlnet is not None:
                    from core.inference import diffusion_controlnet
                    cn_id, cn_image_b64, cn_type, cn_strength, cn_gs, cn_ge = controlnet
                    # strength 0 disables CN: skip the whole path so a no-op never pays the download/VRAM.
                    if cn_strength in (None, 0, 0.0):
                        controlnet = None
                    else:
                        if workflow != "txt2img":
                            raise ValueError(
                                "ControlNet currently combines with plain text-to-image only, not "
                                f"the {workflow} workflow."
                            )
                        if not diffusion_controlnet.supports_controlnet(
                            engine = "diffusers",
                            family = state.family.name,
                            has_controlnet_pipeline = bool(
                                getattr(state.family, "controlnet_pipeline_class", None)
                            ),
                            model_kind = state.kind,
                            transformer_quant = state.transformer_quant,
                        ):
                            raise ValueError(
                                "ControlNet is not supported for this model/quantisation on the "
                                "diffusers engine (needs a bf16 or bnb-4bit load of a family with a "
                                "ControlNet pipeline; not GGUF-via-diffusers or torchao fp8/int8)."
                            )
                        # Decode the control image first so a bad one 400s before any CN download, at the OUTPUT size.
                        src = decode_b64_image(cn_image_b64, mode = "RGB")
                        control_pil = diffusion_controlnet.preprocess_control(src, cn_type).resize(
                            (width, height), Image.LANCZOS
                        )
                        try:
                            resolved_cn = diffusion_controlnet.resolve_controlnet(
                                cn_id, family = state.family.name
                            )
                        except FileNotFoundError as exc:
                            # An unknown CN id -> 400, not 500 (the route maps ValueError).
                            raise ValueError(str(exc)) from exc
                        pipe = self._controlnet_pipe(state, resolved_cn, cancel)
                        workflow = "controlnet"
                        cn_scale, cn_gstart, cn_gend = cn_strength, cn_gs, cn_ge
                        # Flux Union CN selects its head by an integer control_mode; map the type.
                        cn_mode = diffusion_controlnet.union_control_mode(cn_id, cn_type)
                # Prompt lists batch txt2img only: conditioned workflows take one image per call, else a silent broadcast pairs every prompt with it.
                if uniform_prompt(jobs) is None and workflow != "txt2img":
                    raise ValueError(
                        "A prompts list is supported for plain text-to-image only; the "
                        f"{workflow} workflow takes one prompt per call (seed lists still work)."
                    )
                # Snap odd-sized inputs and mask to a multiple of 16 when the OUTPUT size comes from the image.
                if init_pil is not None and source_sized and workflow != "upscale":
                    # Bound uploads to 2048 on the longest side (a phone photo would OOM).
                    if workflow == "img2img":
                        # Bound Transform by the requested size so Resolution caps output; img2img only, since shrinking an inpaint canvas breaks its mask.
                        init_pil = _fit_within(init_pil, min(2048, width), min(2048, height))
                    elif workflow == "inpaint":
                        init_pil = _clamp_max_side(init_pil, 2048)
                    init_pil = _snap_to_multiple(init_pil, 16)
                    if mask_pil is not None and mask_pil.size != init_pil.size:
                        from PIL import Image as _PILImage
                        mask_pil = mask_pil.resize(init_pil.size, _PILImage.NEAREST)
                if init_pil is not None:
                    # Keep the VAE encode dtype consistent with the input image.
                    self._align_vae_dtype(pipe, state.family.denoiser_attr)
                    # Also match the dtype the pipeline itself hands the encoder.
                    self._make_vae_encode_dtype_safe(pipe)

                # getattr like the status poll: a pipeline without a scheduler offers no samplers, so the default
                # request keeps it as is and any other sampler is refused by build_scheduler.
                swapped, resolved_schedule = build_scheduler(
                    getattr(pipe, "scheduler", None), sampler, schedule_type
                )
                if swapped is not None:
                    restore_scheduler = (pipe, pipe.scheduler)
                    pipe.scheduler = swapped
                # Pipelines vary in accepted kwargs, so gate every optional one on the signature.
                call_params = inspect.signature(pipe.__call__).parameters
                sd_controls_mod.validate(sd_controls, fam, pipe, call_params, workflow)
                mask_blur_applied = inpaint_padding_applied = None
                soft_overlay = None
                if workflow == "inpaint" and mask_pil is not None:
                    inpaint_padding_applied = resolve_padding(inpaint_only_masked, inpaint_padding)
                    if (
                        inpaint_padding_applied is not None
                        and "padding_mask_crop" not in call_params
                    ):
                        raise ValueError(ONLY_MASKED_UNSUPPORTED_MSG)
                    if inpaint_padding_applied is not None and mask_is_empty(mask_pil):
                        # diffusers' crop box needs a nonzero pixel; with none this is a plain inpaint.
                        inpaint_padding_applied = None
                    if mask_blur:
                        # After the snap/resize above, so the radius is in source-image pixels.
                        mask_pil = blur_mask(mask_pil, mask_blur)
                        mask_blur_applied = int(mask_blur)
                        # The pipeline binarizes the mask, so the blur only reaches the picture through this
                        # overlay. padding_mask_crop already overlays with the blurred mask.
                        if inpaint_padding_applied is None and init_pil is not None:
                            soft_overlay = (init_pil, mask_pil)

                kwargs: dict[str, Any] = {
                    # prompt / generator are set per chunk below, one Generator per image.
                    "num_inference_steps": steps,
                    # Most pipelines use "guidance_scale"; Qwen-Image uses "true_cfg_scale".
                    state.family.cfg_kwarg: guidance,
                }
                # Explicit: the pinned pipeline never reads the grid, and a newer one overrides num_inference_steps.
                sample_sigmas = pipe_sample_sigmas(state.pipe)
                if sample_sigmas is not None and "sigmas" in call_params:
                    kwargs["sigmas"] = sample_sigmas_for_steps(sample_sigmas, steps)
                if state.family.name == IDEOGRAM4_FAMILY_NAME:
                    # Ideogram 4 takes guidance_scale OR guidance_schedule, never both: at the advertised 48 steps and 7.0 guidance drop the constant, else null the schedule.
                    if steps == 48 and abs(float(guidance) - 7.0) < 1e-6:
                        kwargs.pop(state.family.cfg_kwarg, None)
                    else:
                        # ComfyUI preset for this step count; the 7 -> 3 CFG override only at guidance 7.
                        mu, std = ideogram4_comfy_mu_std(steps)
                        if "mu" in call_params:
                            kwargs["mu"] = mu
                        if "std" in call_params:
                            kwargs["std"] = std
                        if abs(float(guidance) - IDEOGRAM4_COMFY_GUIDANCE) < 1e-6:
                            kwargs.pop(state.family.cfg_kwarg, None)
                            kwargs["guidance_schedule"] = ideogram4_comfy_guidance_schedule(
                                steps, width, height
                            )
                        else:
                            kwargs["guidance_schedule"] = None
                layer_count = int(getattr(fam, "layer_count", 0) or 0)
                if layer_count:
                    if "layers" in call_params:
                        kwargs["layers"] = layer_count
                    if "resolution" in call_params:
                        kwargs["resolution"] = int(getattr(fam, "layer_resolution", 640))
                if state.family.name == LUMINA2_FAMILY_NAME and "cfg_trunc_ratio" in call_params:
                    # Lumina 2 card recipe truncates CFG to the first quarter (cfg_trunc_ratio=0.25); the 1.0 default oversaturates.
                    kwargs["cfg_trunc_ratio"] = 0.25
                if init_pil is not None:
                    # Reference / unified edit pass the whole ordered list; others take the single image.
                    kwargs["image"] = [init_pil, *ref_extra] if ref_extra else init_pil
                    if ref_resolution is not None:
                        if "output_resolution" not in call_params:
                            raise ValueError(
                                "This pipeline does not accept a reference resolution."
                            )
                        kwargs["output_resolution"] = ref_resolution
                    if mask_pil is not None and "mask_image" in call_params:
                        kwargs["mask_image"] = mask_pil
                    if inpaint_padding_applied is not None:
                        kwargs["padding_mask_crop"] = inpaint_padding_applied
                    if strength is not None and "strength" in call_params:
                        kwargs["strength"] = strength
                    if _hires_pass is not None and _hires_pass.latent_from is not None:
                        # Latent upscaler: the SDXL img2img pipe takes 4-channel latents as the image.
                        kwargs["image"] = diffusion_hires.upscale_latents(
                            pipe, _hires_pass.latent_from, init_pil.size
                        )
                # width/height: txt2img uses the slider; image-conditioned pipes must use the INPUT IMAGE's size or
                # the latents mismatch, and many drop them, so pass only when accepted.
                if not source_sized:
                    # These generate at the REQUESTED size (reference/control image resized to match).
                    kwargs["width"] = width
                    kwargs["height"] = height
                elif init_pil is not None:
                    iw, ih = init_pil.size
                    if "width" in call_params:
                        kwargs["width"] = iw
                    if "height" in call_params:
                        kwargs["height"] = ih
                if shift is not None or base_shift is not None or max_shift is not None:
                    shift_w, shift_h = kwargs.get("width"), kwargs.get("height")
                    if (not shift_w or not shift_h) and init_pil is not None:
                        shift_w, shift_h = init_pil.size
                    if pipe_sample_sigmas(state.pipe) is not None:
                        # Tuned on the shipped scheduler, like the load-time ComfyUI shift (which skips it too).
                        raise ValueError(
                            "This checkpoint ships its own sampling schedule, so Shift is not available."
                        )
                    shift_node = _state_shift_node(state)
                    applied_shift = requested_flow_shift(
                        shift_node,
                        shift = shift,
                        base_shift = base_shift,
                        max_shift = max_shift,
                        width = shift_w or width,
                        height = shift_h or height,
                        defaults = shift_control(shift_node, pipe.scheduler),
                    )
                    shifted = scheduler_with_flow_shift(pipe.scheduler, applied_shift)
                    if shifted is not pipe.scheduler:
                        if restore_scheduler is None:
                            restore_scheduler = (pipe, pipe.scheduler)
                        pipe.scheduler = shifted
                if (
                    negative_prompt
                    and _negative_prompt_engaged(state.family, guidance)
                    and "negative_prompt" in call_params
                ):
                    kwargs["negative_prompt"] = negative_prompt
                elif "negative_prompt" in call_params and true_cfg_needs_empty_negative(
                    state.family.cfg_kwarg, guidance
                ):
                    # Qwen-Image true CFG needs a negative present; ComfyUI encodes an empty one.
                    kwargs["negative_prompt"] = ""
                applied_embeddings = diffusion_embeddings.apply(
                    pipe,
                    getattr(state.family, "textual_inversion", None),
                    {
                        "prompt": [p for p, _ in jobs],
                        "negative_prompt": kwargs.get("negative_prompt"),
                    },
                    marker_owners = (state.pipe,),
                    logger = logger,
                )
                if workflow == "controlnet" and control_pil is not None:
                    # CN takes the control map and scale; guidance start/end bound steps; kwargs are signature-gated.
                    if "control_image" in call_params:
                        kwargs["control_image"] = control_pil
                    elif "image" in call_params:  # some CN pipelines name it "image"
                        kwargs["image"] = control_pil
                    if "controlnet_conditioning_scale" in call_params and cn_scale is not None:
                        kwargs["controlnet_conditioning_scale"] = cn_scale
                    if "control_guidance_start" in call_params and cn_gstart is not None:
                        kwargs["control_guidance_start"] = cn_gstart
                    if "control_guidance_end" in call_params and cn_gend is not None:
                        kwargs["control_guidance_end"] = cn_gend
                    # Union CN mode index (Flux); only when accepted and the type maps to a mode.
                    if "control_mode" in call_params and cn_mode is not None:
                        kwargs["control_mode"] = cn_mode

                # Per-forward chunks: the whole job list in ONE forward by default, bounded by an explicit batch_size
                # cap; the OOM backoff below halves a failed chunk. Layered: one input per forward (the pipeline's
                # per-prompt layer stride is only right for one).
                chunks = _generation_chunks(state.family, jobs, batch_size, layer_count = layer_count)

                # EFFECTIVE steps: img2img at strength < 1 denoises a fraction of `steps`.
                strength_applied = effective_request_strength(
                    strength,
                    init_pil is not None,
                    "strength" in call_params,
                    call_params["strength"].default if "strength" in call_params else None,
                )
                denoise_steps = effective_denoise_steps(steps, strength_applied)
                auto_step_cache_on = False
                # (mode,) once the pre-guard disable ran, so the toggle below is not repeated
                pre_guard_toggle = None
                if state.cache_auto:
                    # Judged on the pipeline that RUNS: from_pipe workflows open no cache_context for FBCache.
                    from .diffusion_autoopt.runtime import auto_cache_steps

                    cache_steps, runner_reason = auto_cache_steps(state.pipe, pipe, denoise_steps)
                    auto_step_cache_on = auto_step_cache_mode(cache_steps) is not None
                    if not auto_step_cache_on:
                        pre_guard_toggle = (
                            maybe_toggle_step_cache(
                                state.pipe,
                                steps = cache_steps,
                                quant_active = state.cache_quant_active,
                                threshold = state.cache_threshold,
                                logger = logger,
                            ),
                        )
                guard_width = guard_height = guard_target = None
                guard_release_mib, guard_capture_gen = 0, None
                guard_batch = 1
                guard_condition_pixels = 0

                try:
                    # Forward dims, not sliders: conditioned workflows size from the input image; same derivation as the compile-cache registration below.
                    guard_width, guard_height = _compile_shape_dims(
                        workflow, init_pil, width, height, fam
                    )
                    guard_batch = _activation_guard_batch(chunks)
                    guard_target = self._state_device_target(state)
                    guard_condition_pixels = (
                        int(
                            (1 + len(ref_extra))
                            * ref_resolution
                            * ref_resolution
                            * getattr(fam, "condition_pixel_weight", 1.0)
                        )
                        if ref_resolution is not None
                        # FLUX.3 resizes every condition image to 1 MP itself (diffusion_flux3.resize_reference).
                        else (
                            int(
                                (1 + len(ref_extra))
                                * FLUX3_REFERENCE_AREA
                                * getattr(fam, "condition_pixel_weight", 1.0)
                            )
                            if is_flux3_family(fam) and init_pil is not None
                            # A multi-image edit VAE-encodes each extra image at ~1 MP beside the source.
                            else int(
                                len(ref_extra)
                                * EDIT_EXTRA_IMAGE_AREA
                                * getattr(fam, "condition_pixel_weight", 1.0)
                            )
                        )
                    )
                    if layer_count:
                        # layers + 1 extra canvas-sized frames beside the output-sized one.
                        guard_condition_pixels += (layer_count + 1) * guard_width * guard_height
                    # past what the measured placement reserved: stream resident groups again for this call
                    try:
                        guard_cfg = _negative_prompt_engaged(fam, guidance)
                    except Exception:  # noqa: BLE001 - unknown: sized as a CFG request
                        guard_cfg = None
                    # a reference family's source image joins the denoiser's sequence, and only FLUX.3 and the
                    # ``reference_resolutions`` families price it in guard_condition_pixels
                    guard_unpriced_refs = bool(
                        getattr(fam, "reference", False)
                        and init_pil is not None
                        and ref_resolution is None
                        and not is_flux3_family(fam)
                    )
                    _follow_live_graph_mode(state)
                    guard_cfg, extra_mib = split_aware_request_extra(
                        state.pipe,
                        guard_cfg,
                        workflow,
                        lambda cfg: measured_request_extra_mib(
                            state.pipe,
                            width = guard_width,
                            height = guard_height,
                            batch_size = guard_batch,
                            condition_pixels = guard_condition_pixels,
                            cfg = cfg,
                            unpriced_references = guard_unpriced_refs,
                            workflow = _peak_workflow(fam, workflow),
                            tier_peak_mib = _tier_request_peak(
                                state,
                                width = guard_width,
                                height = guard_height,
                                batch_size = guard_batch,
                                cfg = cfg,
                                workflow = _peak_workflow(fam, workflow),
                            ),
                        ),
                        step_cache = auto_step_cache_on,
                    )
                    if getattr(state.pipe, TIER_MOVED_ATTR, False):
                        extra_mib = _hold_captured_release(state, extra_mib)
                        guard_release_mib = extra_mib
                        guard_capture_gen = cuda_graph.capture_generation(_peak_device_index(state))
                    if extra_mib > 0:
                        releasable_mib = resident_group_mib(state.pipe)
                        if (
                            guard_condition_pixels > 0 or guard_unpriced_refs
                        ) and extra_mib > releasable_mib:
                            # The guard credits a conditioned request nothing the load reserved: hook the resident
                            # transformer so this request can stream part of it.
                            hook_resident_denoiser(state.pipe, guard_target.torch_device, logger)
                        restore_resident = release_resident_groups(state.pipe, extra_mib, logger)
                    guard_kwargs = dict(
                        # Not the settled snapshot: its empty_cache() is wrong per generation (re-mallocs activations).
                        device_memory = reclaimable_snapshot_device_memory(guard_target),
                        width = guard_width,
                        height = guard_height,
                        batch_size = guard_batch,
                        # The hint the load planned with, so the distilled / edit multipliers match.
                        family = state.variant_hint,
                        # Only img2img is bounded by the Resolution control (_fit_within); others need an upload-side fix.
                        source_driven = source_sized and workflow != "img2img",
                        condition_pixels = guard_condition_pixels,
                        vae_tile_side = vae_tile_side(getattr(pipe, "vae", None)),
                        vae_sliced = vae_can_slice(getattr(pipe, "vae", None)),
                        quadratic_attention = _quadratic_attention(
                            guard_target, getattr(state, "attention_backend", None), pipe
                        ),
                        allow_oversized = allow_oversized,
                        calibrated_placement = bool(getattr(state, "calibrated_placement", False)),
                        controlnet = workflow == "controlnet" and control_pil is not None,
                        logger = logger,
                    )
                    verdict = raise_on_image_activation_shortfall(**guard_kwargs)
                    if _hires_pass is not None and _hires_pass.preflight is not None:
                        _hires_pass.preflight(guard_kwargs)
                    _guard_layer_cards(state.pipe, guard_target, guard_kwargs)
                    tiled_before = bool(getattr(getattr(pipe, "vae", None), "use_tiling", False))
                    restore_vae = _tile_for_verdict(
                        pipe, verdict, guard_target, guard_kwargs, guard_batch, restore_vae, logger
                    )
                    # tiling changes pixels, streaming only costs time: defer the choice to the decode
                    if not tiled_before:
                        restore_vae = _defer_tiling_to_decode(
                            pipe,
                            state.pipe,
                            verdict,
                            guard_target,
                            guard_kwargs,
                            restore_vae,
                            logger,
                        )
                except ValueError:
                    raise  # the refusal itself: the route turns this into a 400 with the reason
                except Exception as exc:  # noqa: BLE001 - fail OPEN on a broken probe
                    logger.warning(
                        "diffusion.generate: activation headroom re-check skipped (%s)", exc
                    )

                # The steps the pipe runs: a Hires refine converts its count to a longer schedule it enters partway.
                progress_steps = (
                    int(_hires_pass.stats.get("refine_steps") or steps)
                    if _hires_pass is not None and _hires_pass.refine
                    else steps
                )
                gen = _GenState(total_steps = progress_steps * len(chunks))
                # Steps completed by FINISHED chunks, so the bar spans the whole multi-chunk call (mutable cell for
                # _on_step).
                steps_done = [0]

                static_skip = state.transformer_cache == TC_STATIC
                static_chunks_run = 0

                def _publish_preview(
                    url: str,
                    seq: int,
                    gen = gen,
                ) -> None:
                    gen.preview = url
                    gen.preview_seq = seq

                # Started inside the try below, whose exits finish() it: its worker thread polls until then.
                previewer = None

                def _start_previewer():
                    if "callback_on_step_end" not in call_params:
                        return None
                    from .diffusion_preview import LatentPreviewer

                    try:
                        preview_w, preview_h = _compile_shape_dims(
                            workflow, init_pil, width, height, fam
                        )
                    except Exception:  # noqa: BLE001 - no size, no preview
                        preview_w = preview_h = None
                    return LatentPreviewer.create(
                        family = state.family.name,
                        requested = live_preview,
                        height = preview_h,
                        width = preview_w,
                        device = state.device,
                        publish = _publish_preview,
                        total_steps = progress_steps,
                    )

                # Count GPU-finished steps: under CUDA graphs the host enqueues all steps early.
                from .video import (
                    _BOUNDARY_MARK_ATTEMPTS,
                    _BOUNDARY_MARK_RETRY_SECONDS,
                    _CompletedStepTicker,
                    _completed_step_poller,
                    _hold_off_cuda_graph_capture,
                )

                # Replaced per chunk: a finished chunk's boundary must not end the next one's denoise.
                chunk_ticker = [_CompletedStepTicker(progress_steps)]

                def _report(done_in_chunk: int) -> None:
                    if gen.phase not in ("encode", "denoise"):
                        return
                    done = steps_done[0] + max(0, min(int(done_in_chunk), progress_steps))
                    if done <= gen.step:
                        return
                    gen.phase = "denoise"
                    # Monotonic: a wall-clock adjustment (NTP) mid-denoise would skew the ETA.
                    now = time.monotonic()
                    gen.step = done
                    if gen.first_step_at == 0.0:
                        gen.first_step_at = now
                    gen.eta_seconds = _estimate_eta(
                        gen.total_steps, gen.step, gen.first_step_at, now
                    )

                def _flip_to_decode(gen = gen) -> None:
                    # The chunk's denoise is provably over: complete it even if a skipped event left the count short.
                    _report(progress_steps)
                    gen.phase = "decode"
                    gen.eta_seconds = None

                def _pump() -> None:
                    if gen.phase not in ("encode", "denoise"):
                        return
                    ticker = chunk_ticker[0]
                    if ticker.boundary_marked and ticker.boundary_reached():
                        _flip_to_decode()
                        return
                    _report(ticker.completed())

                def _on_step(pipe, step_index, timestep, callback_kwargs):
                    if static_skip:
                        mark_step_end(state.pipe)
                    if gen.phase == "encode":
                        gen.phase = "denoise"
                    if previewer is not None:
                        previewer.on_step(
                            callback_kwargs.get("latents"), getattr(pipe, "scheduler", None)
                        )
                    # Runs after scheduler.step, so this step's update is already enqueued.
                    with _hold_off_cuda_graph_capture() as clear:
                        if clear:
                            chunk_ticker[0].record(step_index + 1)
                            _report(chunk_ticker[0].completed())
                    if cancel.is_set():
                        pipe._interrupt = True
                    cross_forward_step(pipe, step_index)
                    # Last denoise step issued: the prefix KV cache (Qwen-Image-2.1) is dead weight through decode.
                    num_steps = getattr(pipe, "_num_timesteps", None)
                    if isinstance(num_steps, int) and step_index + 1 >= num_steps:
                        _release_prefix_kv(pipe)
                    return callback_kwargs

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step
                    track_prefix_kv(getattr(pipe, "transformer", None))
                # Hires fix latent upscaler (ComfyUI LatentUpscale): the base pass hands the refine pass its latents,
                # so it skips the VAE decode and the image post-processing that only the discarded pixels needed.
                latent_output = bool(
                    _hires_pass is not None and _hires_pass.latent_output and not _hires_pass.refine
                )
                if latent_output:
                    if "output_type" not in call_params or layer_count or soft_overlay is not None:
                        raise ValueError(
                            "The latent upscaler needs a pipeline that returns latents."
                        )
                    kwargs["output_type"] = "latent"
                kwargs.update(sd_controls_mod.guidance_rescale_kwargs(sd_controls, call_params))
                weighting_used = [False]

                def _render_chunk(
                    chunk_kwargs: dict[str, Any], chunk_variation: Optional[list[int]]
                ):
                    if sd_controls_mod.apply_prompt_controls(
                        pipe, fam, sd_controls, chunk_kwargs, guidance = guidance
                    ):
                        weighting_used[0] = True
                    with (
                        sd_controls_mod.variation_latents(
                            pipe,
                            chunk_variation,
                            sd_controls.variation_strength if sd_controls is not None else None,
                        ),
                        pipeline_upcasts_keep_storage(pipe),
                        # Only with the step callback: it stops the carry before the last step's decode.
                        cross_forward_render(pipe, enabled = "callback_on_step_end" in chunk_kwargs),
                    ):
                        return pipe(**chunk_kwargs).images

                # Re-check an AUTO cache decision against the ACTUAL step count; explicit choices never toggle.
                if state.cache_auto:
                    if pre_guard_toggle is not None:
                        toggled = pre_guard_toggle[0]
                    else:
                        toggled = maybe_toggle_step_cache(
                            state.pipe,
                            steps = cache_steps,
                            quant_active = state.cache_quant_active,
                            threshold = state.cache_threshold,
                            logger = logger,
                        )
                    if toggled != state.transformer_cache:
                        # _LoadState is frozen; this in-place update is deliberate so status reports the pipe-level toggle.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {runner_reason}"
                                if runner_reason
                                else f"auto: {denoise_steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                self._gen = gen
                images: list[Any] = []
                per_image_seeds: list[int] = []
                chunk_shapes: list[int] = []
                graphs_before = fresh_compile_count()
                compile_cache.note_use(state.compile_cache_ctx)
                try:
                    state.pipe._unsloth_peak_cfg = _negative_prompt_engaged(state.family, guidance)
                except Exception:  # noqa: BLE001 - an unknown guidance mode records nothing
                    state.pipe._unsloth_peak_cfg = None
                whole_tier = None
                if _hires_pass is None and not aux_pass and not bg_eager:
                    try:
                        whole_tier = _whole_tier_for_request(
                            state,
                            steps = denoise_steps,
                            width = guard_width,
                            height = guard_height,
                            # the largest forward this request runs, not the activation guard's singleton floor
                            batch_size = max((len(chunk) for chunk in chunks), default = 1),
                            cfg = getattr(state.pipe, "_unsloth_peak_cfg", None),
                            condition_pixels = guard_condition_pixels,
                            workflow = workflow,
                            logger = logger,
                        )
                    except Exception as exc:  # noqa: BLE001 - the load's streamed plan
                        logger.debug(
                            "diffusion.memory: per-request whole denoiser skipped (%s)", exc
                        )
                        whole_tier = None
                whole_release: list = []

                def _release_whole_denoiser() -> None:
                    while whole_release:
                        try:
                            whole_release.pop()()
                        except Exception as release_exc:  # noqa: BLE001 - the groups keep their state
                            logger.warning(
                                "diffusion.memory: streaming the denoiser again failed (%s)",
                                release_exc,
                            )

                # a whole-denoiser call's peak is not the streamed placement's
                peak_probe = None if whole_tier is not None else _start_peak_probe(state)
                decode_restores: list = []
                try:
                    previewer = _start_previewer()
                    pending = list(chunks)
                    while pending:
                        chunk = pending.pop(0)
                        chunk_kwargs = dict(kwargs)
                        shared = uniform_prompt(chunk)
                        generators = [
                            torch.Generator(device = state.device).manual_seed(s) for _, s in chunk
                        ]
                        if len(jobs) == 1:
                            chunk_kwargs["prompt"] = shared
                            chunk_kwargs["generator"] = generators[0]
                            chunk_kwargs["num_images_per_prompt"] = 1
                        elif shared is not None:
                            chunk_kwargs["prompt"] = shared
                            chunk_kwargs["generator"] = generators
                            chunk_kwargs["num_images_per_prompt"] = len(chunk)
                        else:
                            # Negative prompt must be broadcast to match, else the pipeline fails in the concat.
                            chunk_kwargs["prompt"] = [p for p, _ in chunk]
                            chunk_kwargs["generator"] = generators
                            chunk_kwargs["num_images_per_prompt"] = 1
                            if isinstance(chunk_kwargs.get("negative_prompt"), str):
                                chunk_kwargs["negative_prompt"] = [
                                    chunk_kwargs["negative_prompt"]
                                ] * len(chunk)
                        # Per chunk, since a prompts list varies in length.
                        t5_len = flux_t5_kwarg(state.family.name, pipe, call_params, chunk_kwargs)
                        if t5_len is not None:
                            chunk_kwargs["max_sequence_length"] = t5_len
                        # A step cache keys residuals on cond/uncond context, which a graph key cannot see.
                        if state.cuda_graphs:
                            cuda_graph.set_bypass(
                                state.cuda_graphs, cache_breaks_graph(state.transformer_cache)
                            )
                        if static_skip:
                            # Auto skip was measured on txt2img only; explicit "static" applies to every workflow.
                            auto_only_txt2img = workflow != "txt2img" and static_skip_is_auto(
                                state.pipe
                            )
                            reset_static_step_skip(
                                state.pipe,
                                denoise_steps,
                                compute_all = auto_only_txt2img,
                                step_signal = "callback_on_step_end" in chunk_kwargs,
                                keep_stats = static_chunks_run > 0,
                                owner = current_account_id(),
                            )
                            static_chunks_run += 1
                            static_skip_pipe = state.pipe
                        elif state.transformer_cache:
                            # diffusers resets FBCache only after a SUCCESSFUL __call__; a raised call leaves a stale residual.
                            self._reset_step_cache(state.pipe)
                        if applied_embeddings is not None:
                            chunk_kwargs = applied_embeddings.rewrite(chunk_kwargs)
                        # Per pipe, idempotent: a workflow pipe built by from_pipe has its own image processor.
                        install_device_postprocess(pipe, logger)
                        protect_ctx = protect_generation(pipe, denoise_steps, logger = logger)
                        # Per chunk: a later chunk encodes again after an earlier one decoded.
                        gen.phase = "encode"

                        def _enter_denoise_phase(gen = gen) -> None:
                            if gen.phase == "encode":
                                gen.phase = "denoise"
                            # The prompt is encoded: the encoders are idle until the next prompt, so free them.
                            if state.te_releaser is not None:
                                state.te_releaser.release()
                            if whole_tier is not None and not whole_release:
                                # pinned after the encode, released before the decode
                                release = pin_denoiser_for_call(
                                    state.pipe,
                                    getattr(guard_target, "torch_device", None) or state.device,
                                    activation_mib = whole_tier[2],
                                    logger = logger,
                                )
                                if release is not None:
                                    whole_release.append(release)
                                    logger.info(
                                        "diffusion.memory: this %d-step render runs the denoiser whole (priced "
                                        "%.2f s vs %.2f s streamed)",
                                        int(denoise_steps),
                                        whole_tier[0],
                                        whole_tier[1],
                                    )

                        chunk_ticker[0] = _CompletedStepTicker(progress_steps)

                        def _enter_decode_phase() -> None:
                            """The decoder was entered: a HOST position, possibly far ahead of the GPU. Mark the
                            boundary in the stream and let the poller flip once the GPU reaches it; with no event
                            to wait on (CPU, MPS) there is no queue to be ahead of, so flip now."""
                            if peak_probe is not None:
                                peak_probe.mark_decode()
                            _release_whole_denoiser()
                            # encoder residency lent to the decode streams again only if free memory is short now
                            restore = release_for_decode(
                                state.pipe,
                                width = guard_width,
                                height = guard_height,
                                batch_size = len(chunk),
                                device = getattr(guard_target, "torch_device", None),
                                logger = logger,
                            )
                            if restore is not None:
                                decode_restores.append(restore)
                                if peak_probe is not None:
                                    # the decode ran with fewer weights than the probe's baseline: its peak understates
                                    peak_probe.decode_released = True
                            marked = False
                            for _ in range(_BOUNDARY_MARK_ATTEMPTS):
                                with _hold_off_cuda_graph_capture() as clear:
                                    if clear:
                                        marked = chunk_ticker[0].mark_boundary()
                                        break
                                time.sleep(_BOUNDARY_MARK_RETRY_SECONDS)
                            if not marked:
                                _flip_to_decode()

                        try:
                            # torchao aten.to fails torch's aliasing check under inference_mode once offloaded.
                            with (
                                (
                                    torch.no_grad()
                                    if _torchao_render_needs_no_grad(state)
                                    else torch.inference_mode()
                                ),
                                protect_ctx,
                                denoise_phase(pipe, _enter_denoise_phase),
                                zimage_cfg_split_render(pipe, len(chunk)),
                                decode_phase(pipe, _enter_decode_phase),
                                _completed_step_poller(_pump),
                                family_render_cudnn_scope(state.family, state.device),
                            ):
                                chunk_variation = (
                                    variation_seed_list[
                                        len(per_image_seeds) : len(per_image_seeds) + len(chunk)
                                    ]
                                    if variation_seed_list
                                    else None
                                )
                                out = render_thread.run(
                                    "diffusion",
                                    lambda: _render_chunk(chunk_kwargs, chunk_variation),
                                )
                            if gen.phase in ("encode", "denoise"):
                                _flip_to_decode()
                        except Exception as exc:  # noqa: BLE001 - reraised unless a splittable OOM
                            oom = is_oom_error(exc)
                            if oom:
                                # Drop graphs on ANY OOM before the split decision: empty_cache() cannot reclaim a graph pool.
                                cuda_graph.reset_all(state.cuda_graphs)
                            if len(chunk) < 2 or not oom:
                                raise
                            # OOM backoff: halve the failed chunk and retry; per-image seeds keep every retry reproducible.
                            empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
                            if callable(empty_cache):
                                empty_cache()
                            first_half, second_half = split_chunk(chunk)
                            pending[:0] = [first_half, second_half]
                            peak_probe = None  # the failed attempt's peak is the card's limit, not this request's
                            gen.total_steps += steps  # one extra chunk to run
                            logger.warning(
                                "diffusion.generate: batch of %d hit OOM; retrying as %d + %d",
                                len(chunk),
                                len(first_half),
                                len(second_half),
                            )
                            continue
                        finally:
                            _release_whole_denoiser()
                            while decode_restores:
                                try:
                                    decode_restores.pop()()
                                except Exception as restore_exc:  # noqa: BLE001 - the groups keep streaming
                                    logger.warning(
                                        "diffusion.memory: re-pinning groups after the decode failed (%s)",
                                        restore_exc,
                                    )
                            # A guarded block that fell back to eager (compile failed at its first forward) no longer
                            # runs compiled: report it on every exit, cancel and error included, so status, LoRA gating
                            # and the compile-cache shape registry stop treating it as compiled.
                            settle_compile_fallback(state, state.pipe, logger)
                        if cancel.is_set():
                            raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                        if layer_count:
                            # One list of layers per input; each layer is an output image with its input's seed.
                            for (_, s), layers_out in zip(chunk, out):
                                images.extend(layers_out)
                                per_image_seeds.extend(s for _ in layers_out)
                        elif latent_output:
                            # One (1, C, h, w) per image; a copy, so no later forward can reuse its storage.
                            images.extend(out[i : i + 1].clone() for i in range(out.shape[0]))
                            per_image_seeds.extend(s for _, s in chunk)
                        else:
                            if soft_overlay is not None:
                                out = [overlay_with_mask(o, *soft_overlay) for o in out]
                            images.extend(out)
                            per_image_seeds.extend(s for _, s in chunk)
                        chunk_shapes.append(len(chunk))
                        steps_done[0] += progress_steps
                except BaseException:
                    if previewer is not None:
                        previewer.finish()
                    # A cancelled or failed render may already have generalised a reused graph, so the success path never sees the count grow: dirty it now.
                    try:
                        if (
                            auto_dynamic_active(state.pipe)
                            and fresh_compile_count() > graphs_before
                        ):
                            compile_cache.mark_recompiled(state.compile_cache_ctx)
                    except Exception:  # noqa: BLE001 - bookkeeping must not mask the render's own error
                        pass
                    raise
                if previewer is not None:
                    previewer.finish()
                # A denoiser whose text length follows the prompt records no new prompt length after this render.
                cuda_graph.seal_lengths(state.cuda_graphs)
                _record_peak_probe(
                    state,
                    # A pass that skipped its VAE decode misses that phase's peak: it must not size a plain render.
                    None if latent_output else peak_probe,
                    width = guard_width,
                    height = guard_height,
                    batch_size = max(chunk_shapes or [1]),
                    target = guard_target,
                    workflow = workflow,
                )
                _note_captured_release(state, guard_release_mib, guard_capture_gen)
                if static_skip:
                    logger.debug("diffusion.step_skip: %s", static_skip_stats(state.pipe))
                # Keep progress ACTIVE through the post-denoise work: the route persists the image after this returns,
                # so a mount probe reading idle would refresh the gallery too early. Persist the warm compile bundle;
                # a STATIC compile makes new artifacts per (w,h,batch), so register this shape. The write itself is
                # QUEUED, not performed: nothing in this response depends on it (only the NEXT process reads the
                # bundle), so save_async hands it to the shared worker and the user stops waiting on it.
                if aux_pass:
                    pass
                elif bg_eager:
                    # Nothing compiled ran, so nothing to register or save yet.
                    bg.note_eager_generation()
                    if _hires_pass is None or not _hires_pass.defer_bg_kick:
                        kick_bg = True
                else:
                    try:
                        # Register the dims the forward ACTUALLY compiled with, and each distinct chunk size: a static compile makes one artifact per batch size.
                        reg_width, reg_height = _compile_shape_dims(
                            workflow, init_pil, width, height, fam
                        )
                        static_shapes = "compiled" in (
                            state.speed_optims or ()
                        ) and compiled_shapes_are_static(state.pipe, state.speed_mode)
                        for chunk_batch in sorted(set(chunk_shapes)):
                            compile_cache.register_shape(
                                state.compile_cache_ctx,
                                (reg_width, reg_height, int(chunk_batch)),
                                static = static_shapes,
                            )
                        if (
                            auto_dynamic_active(state.pipe)
                            and fresh_compile_count() > graphs_before
                        ):
                            # Dynamic recompiles on new text lengths: persist those graphs too, or every fresh process pays them again.
                            compile_cache.mark_recompiled(state.compile_cache_ctx)
                        compile_cache.save_async(state.compile_cache_ctx, logger = logger)
                    except Exception:  # noqa: BLE001 - cache persistence is best-effort
                        pass
                # The status badge reports the attention routes this run took (report only: the recipe keeps the
                # engaged backend).
                report_attention_run(state)
                # Last word on cancellation, AFTER the post-denoise work: the event stays registered through the
                # compile-cache bookkeeping and the page still shows Stop for as long as progress reads active, so a Stop
                # landing there was answered cancelled = true and then contradicted by the image the route persisted.
                # Check and deregister under the cancellation lock, which cancel_generate takes, so the two cannot
                # interleave. The finally below repeats the clear for every other exit.
                with self._generation_cancel_lock:
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None

                # Components are offloaded and transfer copies dropped; return the pages now.
                reclaim_offload_host_memory(state.offload_policy, logger = logger)
                # Count the finished generation (drives deferred speed); a batch is one generation.
                if (_hires_pass is None or not _hires_pass.refine) and not aux_pass:
                    object.__setattr__(state, "generation_count", state.generation_count + 1)
                result = {
                    "images": list(images),
                    "seed": int(seed),
                    "seeds": [int(s) for s in per_image_seeds],
                    "negative_prompt": kwargs.get("negative_prompt") or None,
                    "repo_id": state.display_repo_id or state.repo_id,
                    # The build this ran on, not just the repo id: a GGUF quant or torchao scheme each change the pixels.
                    "model_kind": state.kind,
                    "gguf_filename": state.gguf_filename,
                    "transformer_quant": state.transformer_quant,
                    # Read off the engaged state, never the request.
                    "text_encoder_quant": state.text_encoder_quant,
                    "memory_mode": state.memory_mode,
                    "offload_policy": state.offload_policy,
                    # Read after the deferred speed / step-cache toggles, so this is what ran.
                    "speed_mode": state.speed_mode,
                    "attention_backend": state.attention_backend,
                    "transformer_cache": state.transformer_cache,
                    "cpu_offload": state.cpu_offload,
                    # Adapters baked at LOAD time: disabling them at generate time is a different build, so record them.
                    "baked_loras": _baked_lora_names(state.pipe),
                    # The adapters ACTUALLY attached, at non-zero weight, for this generation.
                    "active_loras": _active_lora_pairs(state.pipe),
                    # Workflow actually run, so a conditioned image is not replayed as a plain Create recipe.
                    "workflow": workflow,
                    "reference_resolution": ref_resolution,
                    "localized_edit": localized_edit.mode if localized_edit is not None else None,
                    "sampler": sampler
                    if swapped is not None and sampler != DEFAULT_SAMPLER
                    else None,
                    "schedule_type": resolved_schedule,
                    **sd_controls_mod.record_fields(
                        sd_controls,
                        weighting_used = weighting_used[0],
                        variation_seed_list = variation_seed_list,
                    ),
                    # Recorded only when applied, so a plain inpaint recipe stays unchanged.
                    "mask_blur": mask_blur_applied,
                    "inpaint_only_masked": True if inpaint_padding_applied is not None else None,
                    "inpaint_padding": inpaint_padding_applied,
                }
                if upscale_model and workflow == "upscale":
                    result["upscale_model"] = upscale_model
                if applied_shift is not None:
                    result["shift"] = shift
                    result["base_shift"] = base_shift
                    result["max_shift"] = max_shift
                    result["applied_flow_shift"] = applied_shift
                if applied_embeddings is not None:
                    result["embeddings"] = list(applied_embeddings.used)
                    result["embeddings_skipped"] = list(applied_embeddings.skipped)
                logger.info(
                    "diffusion.generated: %s",
                    format_generation_for_log(
                        result,
                        engine = "diffusers",
                        steps = steps,
                        strength = strength_applied,
                        upscale = upscale if workflow == "upscale" else None,
                    ),
                )
                return result
            finally:
                if restore_scheduler is not None:
                    restore_scheduler[0].scheduler = restore_scheduler[1]
                if applied_embeddings is not None:
                    applied_embeddings.restore()
                if bg_token is not None:
                    bg_compile._FORCE_EAGER.reset(bg_token)
                if restore_eager_gate is not None:
                    restore_eager_gate()
                if static_skip_pipe is not None:
                    try:
                        reset_static_step_skip(static_skip_pipe, None)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("diffusion.step_skip: reset failed: %s", exc)
                if restore_vae is not None:
                    restore_vae()
                end_cfg_render(getattr(state, "pipe", None))
                end_cfg_batch_render(getattr(state, "pipe", None))
                if restore_resident is not None:
                    try:
                        restore_resident()
                    except Exception as exc:  # noqa: BLE001 - the groups keep streaming
                        logger.warning("diffusion.memory: re-residency failed: %s", exc)
                if kick_bg and bg is not None:
                    try:
                        bg.kick()
                    except Exception as exc:  # noqa: BLE001 - the next render compiles inline
                        logger.warning("diffusion.bg_compile: kick failed: %s", exc)
                with self._generation_cancel_lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None
                with self._lock:
                    # Sole clear of published progress, on every exit.
                    self._gen = None

    def generate_progress(self) -> dict[str, Any]:
        """Live per-step progress for an in-flight generation (lock-free read)."""
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
                "phase": "denoise",
            }
        return {
            "active": True,
            "step": gen.step,
            "total_steps": gen.total_steps,
            "fraction": gen.step / gen.total_steps,  # step is 1..total, never over 1.0
            "eta_seconds": gen.eta_seconds,
            "phase": gen.phase,
            "preview": gen.preview,
            "preview_seq": gen.preview_seq,
        }

    def cancel_generate(self, expected_account: Optional[str] = None) -> bool:
        """Stops the in-flight generation at the next step callback; returns False when nothing runs; a cancel in VAE decode or encode lands after it."""
        with self._generation_cancel_lock:
            # Stop targets the denoising generation, not a serialized waiter.
            active = self._active_generate_cancel
            if active is not None:
                if expected_account is not None and self._active_generate_account not in (
                    None,
                    expected_account,
                ):
                    return False
                active.set()
                return True
            if self._generation_owns_slot:
                return False
            # An accepted eject raises _unload_waiters long before teardown; without this Stop answered False.
            if (
                not self._teardown_waiters
                and not self._transition_owns_slot
                and not self._unload_waiters
            ):
                return False
            # Recheck live state so timed waiters observe a replacement handoff.
            cancels = set(self._queued_generate_cancels)
            if not cancels:
                return False
            for cancel in cancels:
                cancel.set()
            return True

    @_invalidates_gpu_memory("diffusion unload")
    def unload(
        self,
        *,
        expected_account: Optional[str] = None,
        only_if: Optional[Callable[[], bool]] = None,
    ) -> Optional[dict[str, Any]]:
        # fenced, and the try that owns it, start BEFORE the counter moves: a leaked _unload_waiters
        # would make _wait_for_pending_unloads block every later load for the life of the process,
        # and the flag keeps the finally honest when authorization refuses before the fence is raised.
        fenced = False
        try:
            with self._generation_cancel_lock, self._load_cancel_lock:
                # Rechecked under the load worker's lock, so a load that ended since costs nothing.
                if only_if is not None and not only_if():
                    return None
                if expected_account is not None:
                    from .gpu_arbiter import DIFFUSION, GpuBusyForAnotherAccountError
                    from hub.services.models.account_access import require_resident_control

                    # Authorize before changing either cancellation event.
                    if (
                        self._active_generate_cancel is not None
                        and self._active_generate_account != expected_account
                    ):
                        raise GpuBusyForAnotherAccountError(DIFFUSION, 1)
                    require_resident_control(DIFFUSION, getattr(self._state, "repo_id", None))
                    # A published resident outranks a pending load for eject authorization; with no resident, pending ownership decides.
                    loading = self._loading
                    if self._state is not None:
                        foreign_load = False
                    elif loading is not None and loading.error is None:
                        foreign_load = loading.account_id != expected_account
                    else:
                        foreign_load = any(
                            token == self._load_token and account != expected_account
                            for token, account in self._load_accounts.values()
                        )
                    if foreign_load:
                        raise GpuBusyForAnotherAccountError(DIFFUSION, 1)
                # Fence loads and generations before waiting for construction.
                self._unload_waiters += 1
                self._unload_fence_clear.clear()
                fenced = True
                # The cancelled load's own epoch; _load_token is bumped just below.
                cancelled_token = self._load_token
                loading = self._loading
                if loading is not None and loading.error is None:
                    # _run_load's finally drops this, so it spans the prefetch too, where nothing
                    # is registered in _load_accounts. Asset repos too: the encoder download ignores the cancel event.
                    self._draining_repos.setdefault(cancelled_token, set()).update(
                        r
                        for r in (
                            loading.repo_id,
                            loading.base_repo,
                            loading.fetch_repo,
                            *loading.asset_repos,
                        )
                        if r
                    )
                self._cancel_event.set()
                self._load_token += 1
                self._loading = None
                if self._active_generate_cancel is not None:
                    self._active_generate_cancel.set()
            with self._lock:
                # Queued generations are not cancelled by lifecycle changes, so they wait for post-teardown state.
                self._reserve_teardown_locked()
            # Wait for denoise to exit first: _unload_locked uninstalls process-wide state it uses.
            with self._model_transition_slot():
                with self._lock:
                    try:
                        self._unload_locked()
                    finally:
                        # In a finally: clear_gpu_cache() can raise on a sticky CUDA fault, leaving the fence up.
                        self._release_teardown_locked()
        finally:
            if fenced:
                with self._load_cancel_lock:
                    self._unload_waiters -= 1
                    if not self._unload_waiters:
                        self._unload_fence_clear.set()
        return self.status()

    def _unload_locked(self) -> None:
        release_load_budget_fraction("image")
        # Let a background prequant write finish so the next load never shares host RAM with it.
        prequant_cache = sys.modules.get(f"{__package__}.diffusion_prequant_cache")
        if prequant_cache is not None:
            try:
                prequant_cache.finish_background_writes(logger = logger)
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
        state = self._state
        if state is None:
            return
        # A pinner still copying would hold its host copies past release_pinned_host_memory().
        stop_background_pins(getattr(state, "pipe", None))
        # An in-flight background compile still runs the denoiser the teardown below frees.
        if state.bg_compile is not None:
            try:
                state.bg_compile.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
        # Restore process-wide backend flags so the next `off` load is bit-identical; all calls idempotent.
        restore_backend_flags(state.backend_flags_before)
        prebuild = sys.modules.get(f"{__package__}.diffusion_vae_prebuild")
        if prebuild is not None:
            prebuild.cancel_all()
        compile_cache.restore(state.compile_cache_ctx, logger = logger)
        uninstall_cfg_parallel(state.pipe)
        uninstall_zimage_cfg_split(state.pipe)
        # Before clear_gpu_cache(), or the graph pool stays reserved for the life of the process.
        cuda_graph.uninstall_all(state.cuda_graphs)
        uninstall_static_step_skip(state.pipe)
        prompt_cache.release(state.pipe)
        if state.te_releaser is not None:
            state.te_releaser.close()
        for aux in (*self._aux_pipes.values(), *self._cn_pipes.values()):
            prompt_cache.release(aux)
        gguf_compile.uninstall_all()
        try:
            from .diffusion_nvfp4_linear import reset_nvfp4_state
            reset_nvfp4_state()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass
        _uninstall_fused_dit_patches()
        if state.eager_patched:
            _uninstall_eager_patches()
            # Again: the deferred profile layers the eager patch over the fused AdaLN.
            _uninstall_fused_dit_patches()
        # Deliberately NOT unload_lora_weights(): the whole pipe is dropped below, adapters with it; clear aux pipes so they do not pin freed modules.
        self._aux_pipes.clear()
        self._cn_pipes.clear()
        self._cn_models.clear()
        diffusion_upscalers.release_models()
        self._state = None
        del state
        try:
            clear_gpu_cache()
        finally:
            # finally: a sticky CUDA fault must not keep the pinned chunks locked.
            release_pinned_host_memory()
        # Must follow clear_gpu_cache() (runs gc) so the freed staging buffers can be returned.
        reclaim_host_memory(logger = logger)

    def static_skip_view(self) -> tuple:
        """(owner, ``transformer_cache_stats``) from one read; the owner is never part of status()."""
        state = self._state
        return static_skip_view(state.pipe) if state is not None else (None, None)

    def status(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "loaded": False,
                "repo_id": None,
                "display_repo_id": None,
                "family": None,
                "base_repo": None,
                "device": None,
                "dtype": None,
                "model_kind": None,
                "gguf_variant": None,
                "cpu_offload": False,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "text_encoder_quant": None,
                "transformer_quant": None,
                "transformer_quant_backend": None,
                "transformer_quant_backend_reason": None,
                "attention_backend": None,
                "transformer_cache": None,
                "transformer_cache_stats": None,
                "baked_loras": [],
                "workflows": [],
                "conditioning": None,
                "supports_lora": False,
                "lora_after_reload": False,
                "supports_controlnet": False,
                "resolved": None,
                "generation_defaults": None,
            }
        from core.inference import diffusion_controlnet, diffusion_lora
        from hub.utils.gguf import extract_quant_token

        resolved, speed_optims = cuda_graph.live_status(
            state.resolved, state.speed_optims, getattr(state, "cuda_graphs", ())
        )
        speed_optims = int8_gemm_live(state.pipe, speed_optims)
        lora_gate = dict(
            engine = "diffusers",
            family = state.family.name,
            model_kind = state.kind,
            transformer_quant = state.transformer_quant,
        )
        compiled = "compiled" in (getattr(state, "speed_optims", ()) or ())
        lora_load_ok = (
            not _small_host_int8(state.pipe)
            and not comfy_block_runtime_layers(getattr(state.pipe, "transformer", None))
            and not fp8_stored_layers(getattr(state.pipe, "transformer", None))
        )
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "display_repo_id": state.display_repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": state.dtype,
            "model_kind": state.kind,
            "gguf_filename": state.gguf_filename,
            "component_files": state.component_files,
            "gguf_variant": (
                extract_quant_token(state.gguf_filename)
                if state.kind == "gguf" and state.gguf_filename
                else None
            ),
            "cpu_offload": state.cpu_offload,
            "offload_policy": state.offload_policy,
            "vae_tiling": state.vae_tiling,
            "memory_mode": state.memory_mode,
            "speed_mode": state.speed_mode,
            "speed_optims": speed_optims,
            "bg_compile": state.bg_compile.describe() if state.bg_compile is not None else None,
            "text_encoder_quant": state.text_encoder_quant,
            "transformer_quant": state.transformer_quant,
            **_nvfp4_backend_fields(_transformer_quant_backend(state), owner = self),
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
            "transformer_cache_stats": static_skip_stats(state.pipe),
            "baked_loras": _baked_lora_names(state.pipe),
            "multi_gpu": multigpu_placement_status(state.pipe),
            "graph_engagement": cuda_graph.engagement(
                tuple(getattr(state.pipe, "_unsloth_cuda_graphs", ()) or ())
                or getattr(state, "cuda_graphs", ())
            ),
            "cfg_batch": cfg_batch_status(state.pipe),
            "zimage_cfg_split": zimage_cfg_split_status(state.pipe),
            "resolved": resolved,
            "generation_defaults": _generation_defaults_for(
                state.repo_id,
                state.gguf_filename,
                state.base_repo,
                pipe_sample_sigmas(state.pipe),
                family = state.family,
            ),
            # Workflows the loaded family supports, so the UI can gate its tabs.
            "workflows": _family_workflows(state.family),
            "supports_negative_prompt": state.family.uses_negative_prompt,
            "supports_inpaint_only_masked": inpaint_class_accepts_padding_mask_crop(
                getattr(state.family, "inpaint_pipeline_class", None)
            ),
            "negative_prompt_min_guidance": _negative_prompt_guidance_threshold(state.family),
            "supports_guidance": _guidance_supported(state),
            "supports_live_preview": _live_preview_available(state),
            "prompt_hint": state.family.prompt_hint,
            "samplers": sampler_options(getattr(state.pipe, "scheduler", None)),
            "sd_controls": sd_controls_mod.status_fields(state.family, state.pipe),
            # Hidden for a checkpoint that ships its own sampling grid (the load-time ComfyUI shift skips it too).
            "shift_control": None
            if pipe_sample_sigmas(state.pipe) is not None
            else shift_control(_state_shift_node(state), getattr(state.pipe, "scheduler", None)),
            "textual_inversion": getattr(state.family, "textual_inversion", None),
            **_hires_status(state.family),
            "conditioning": conditioning_capabilities(
                state.family, _family_workflows(state.family)
            ),
            "supports_lora": lora_load_ok
            and diffusion_lora.supports_lora(compiled = compiled, **lora_gate),
            # Only Speed=Auto's 3rd-image compile blocks LoRA: a reload stays eager until then.
            "lora_after_reload": lora_load_ok
            and compiled
            and getattr(state, "speed_auto_compiled", False)
            and diffusion_lora.supports_lora(compiled = False, **lora_gate),
            "small_host": small_host_engaged_on(state.pipe),
            "supports_controlnet": diffusion_controlnet.supports_controlnet(
                engine = "diffusers",
                family = state.family.name,
                has_controlnet_pipeline = bool(
                    getattr(state.family, "controlnet_pipeline_class", None)
                ),
                model_kind = state.kind,
                transformer_quant = state.transformer_quant,
            ),
        }


def _small_host_int8(pipe: Any) -> bool:
    """Whether the small-host route stored a denoiser as int8 weights on this pipe."""
    info = small_host_engaged_on(pipe) or {}
    return any(str(v).startswith("int8") for v in (info.get("components") or {}).values())


def _transformer_quant_backend(state: Any) -> Optional[str]:
    """Which NVFP4 kernel path the loaded denoiser runs, read from the MODULE TREE, or None. Never raises."""
    if getattr(state, "transformer_quant", None) != TQ_NVFP4:
        return None
    try:
        pipe = getattr(state, "pipe", None)
        denoiser_attr = getattr(getattr(state, "family", None), "denoiser_attr", "transformer")
        denoiser = getattr(pipe, denoiser_attr or "transformer", None)
        if denoiser is None:
            return None
        declared = getattr(denoiser, "_unsloth_nvfp4_backend", None)
        if declared:
            return str(declared)
        from .diffusion_nvfp4_linear import is_nvfp4_flashinfer_linear

        for module in denoiser.modules():
            if is_nvfp4_flashinfer_linear(module):
                return "flashinfer"
        return "torchao"
    except Exception:  # noqa: BLE001 -- see the docstring: a poll must not fail on a probe
        return None


def _hires_status(fam: DiffusionFamily) -> dict[str, Any]:
    """Status keys for Hires fix (hidden without an img2img path) and the Upscale workflow's model upscalers."""
    try:
        models = diffusion_upscalers.model_upscalers()
    except Exception as exc:  # noqa: BLE001 - a broken scan hides model upscalers, never the status
        logger.debug("diffusion.upscalers: scan failed: %s", exc)
        models = []
    hires = diffusion_hires.capability(fam, model_rows = models)
    return {
        "hires": hires,
        # Why the control is hidden (no img2img path yet), shown in its place.
        "hires_unavailable_reason": diffusion_hires.unavailable_reason(fam)
        if hires is None
        else None,
        "upscale_models": models if getattr(fam, "img2img_pipeline_class", None) else [],
    }


def _peak_workflow(fam: Any, workflow: Optional[str]) -> Optional[str]:
    """``workflow`` for the learned-peak key, None when it is the family's only workflow (an edit-only family): those
    renders are what its loads place from, so they stay on the load key."""
    try:
        if fam is not None and _family_workflows(fam) == [str(workflow or "").strip().lower()]:
            return None
    except Exception:  # noqa: BLE001
        pass
    return workflow


def _family_workflows(fam: DiffusionFamily) -> list[str]:
    """Instruction-editing families have no txt2img mode, so expose only the edit workflow."""
    if getattr(fam, "edit", False):
        return ["edit"]
    workflows = ["txt2img"]
    # Reference families (FLUX.2) add reference conditioning via their pipeline's image arg.
    if getattr(fam, "reference", False):
        workflows.append("reference")
    # Unified (Qwen-Image-2.1) and reference-edit (FLUX.3) families also edit through that same pipeline.
    if getattr(fam, "unified_edit", False) or getattr(fam, "reference_edit", False):
        workflows.append("edit")
    if getattr(fam, "img2img_pipeline_class", None):
        # Upscale runs on the img2img pipeline, so available exactly when img2img is.
        workflows.append("img2img")
        workflows.append("upscale")
    if getattr(fam, "inpaint_pipeline_class", None):
        workflows.append("inpaint")
        # Outpaint reuses the inpaint pipeline with a padded canvas, so needs one that preserves size.
        if getattr(fam, "inpaint_preserves_size", True):
            workflows.append("outpaint")
    return workflows


def _resolve_base_repo(
    repo_id: str,
    base_repo: Optional[str],
    fam: DiffusionFamily,
    hf_token: Optional[str],
    gguf_filename: Optional[str] = None,
) -> str:
    """Card base_model tags are attacker-controlled: an untrusted tag falls back to the family default, since from_pretrained would unpickle it."""
    base = (base_repo or "").strip()
    if not base:
        tag = _hf_base_model(repo_id, hf_token)
        if tag and _is_trusted_diffusion_repo(tag):
            # Map mirrors back to the vendor id (status base_repo, adapter base_model).
            base = canonical_base(tag)
        elif tag is None:
            # No card tag: only a curated variant the name spells out.
            base = named_variant_base(fam, repo_id, gguf_filename) or ""
    # Returns the UPSTREAM id; the swap happens at the fetch sites only.
    resolved = resolve_base_repo(fam, base)
    _remember_companion_base(repo_id, resolved)
    return resolved


def _remember_companion_base(repo_id: str, base: str) -> None:
    """Records the repo-to-base link where it is decided, since an offline scan cannot rebuild the card tag; best-effort, must never fail a load."""
    try:
        from hub.utils.companion_assets import record_companion_link
        record_companion_link(repo_id, base)
    except Exception as exc:  # noqa: BLE001 -- bookkeeping only
        logger.debug("diffusion.companion_link_record_failed: %s", exc)


def _cached_while_hub_unreachable(repo_id: Any, filename: str) -> bool:
    """Hub just failed to connect AND ``filename`` is cached: ``local_files_only=True`` then returns what the online
    call would after its HEAD timed out. False on any doubt, so a first download still asks."""
    if not hub_reachability.hub_unreachable():
        return False
    try:
        if not isinstance(repo_id, str) or Path(repo_id).expanduser().exists():
            return False
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo_id, filename, cache_dir = hub_cache_dir()), str)
    except Exception:  # noqa: BLE001
        return False


def _pick_cached_while_hub_unreachable(
    fam: Any,
    kind: str,
    repo_id: str,
    gguf_filename: Optional[str],
    base: Optional[str],
    fetch_base: Optional[str],
) -> bool:
    """Base manifest cached (or local) and, for a GGUF / single-file pick, the checkpoint too."""
    if not hub_reachability.hub_unreachable():
        return False
    try:
        manifest = _base_manifest_name(fam)
        if not (
            (bool(base) and _is_local_path(base))
            or any(
                _cached_while_hub_unreachable(repo, manifest)
                for repo in dict.fromkeys(r for r in (fetch_base, base) if r)
            )
        ):
            return False
        if kind != "pipeline" and gguf_filename and not _is_local_path(repo_id):
            return _cached_while_hub_unreachable(repo_id, gguf_filename)
        return True
    except Exception:  # noqa: BLE001 - any doubt keeps the online load
        return False


def _hf_base_model(repo_id: str, hf_token: Optional[str]) -> Optional[str]:
    """The diffusers base repo from a GGUF repo's ``base_model`` tag, or None. Lets one family entry
    cover every variant (Turbo/full, schnell/dev, the 2512 Qwen revision). Skipped for local
    paths; None on any lookup failure."""
    if Path(repo_id).expanduser().exists():
        return None
    try:
        from huggingface_hub import HfApi

        # Picks the base: never skipped, or a recovered network would get the family default.
        meta = (
            hub_reachability.model_info(
                HfApi(), repo_id, token = hf_token, skip_when_unreachable = False
            ).cardData
            or {}
        )
    except Exception:  # noqa: BLE001 - best-effort; fall back to the family default
        return None
    base = meta.get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    return base if isinstance(base, str) and base.strip() else None


def _offload_controlnet_module(cn_model: Any, device: str, logger: Any) -> bool:
    """Stream a ControlNet module through ``device`` via diffusers group offloading. Used when the
    base model was loaded with an offload policy: forcing the ControlNet fully resident with
    ``.to(device)`` would defeat that low-VRAM placement and can OOM. Group offloading is applied
    to this single module, so it is isolated and reversible. Returns True on success; on any
    failure the caller falls back to a resident placement (a calibrated tier refuses instead).

    On CUDA: ``record_stream`` (without it diffusers synchronizes the compute stream on every group offload),
    and the event-fenced ``GroupPrefetcher`` replaces diffusers' host-synced prefetch chain."""
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        from .diffusion_memory import _apply_group_offloading_packed
        from .host_pinning import host_pinning_allowed

        onload = torch.device(device)
        use_stream = onload.type == "cuda" and host_pinning_allowed(onload)
        kwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": use_stream,
        }
        params = inspect.signature(apply_group_offloading).parameters
        prefetch = use_stream and "record_stream" in params and "non_blocking" in params
        if prefetch:
            from .diffusion_memory import (
                _pin_top_level_group,
                _skip_top_level_copy_back,
                install_group_offload_buffer_restore,
                install_group_offload_hooks_eager,
                install_group_prefetch,
            )

            install_group_offload_buffer_restore()
            install_group_offload_hooks_eager()
            kwargs["non_blocking"] = True
            kwargs["record_stream"] = True
        _apply_group_offloading_packed(apply_group_offloading, cn_model, kwargs, logger)
        if prefetch:
            if not _pin_top_level_group(cn_model, logger):
                _skip_top_level_copy_back(cn_model, logger)
            install_group_prefetch(cn_model, onload, logger)
        return True
    except Exception as exc:  # noqa: BLE001 - offload is best-effort; resident is the fallback
        if logger is not None:
            logger.warning("diffusion.controlnet: group offload failed (%s); loading resident", exc)
        return False


def _base_file_downloaded(
    rfilename: str,
    *,
    include_transformer: bool = False,
    manifest: str = "model_index.json",
) -> bool:
    """True for base-repo files ``from_pretrained`` actually fetches. The transformer is supplied by
    the GGUF, and repo docs (``assets/``, the top-level README/PDF/images) are never downloaded:
    counting them would peg the progress estimate above what lands on disk, so the bar would sit
    short of 100% for the whole pipeline-load phase. ``include_transformer`` admits the
    ``transformer/`` shards for loads where the dense transformer-quant path will fetch them
    anyway. ``manifest`` is the base's pipeline index (FLUX.3's is ``modular_model_index.json``)."""
    if rfilename == "transformer/config.json":
        # from_single_file(config=repo, subfolder="transformer") always fetches this config. Video keeps
        # the same exception in video.py's snapshot filter.
        return True
    if rfilename.startswith("transformer/"):
        return include_transformer
    if "/" not in rfilename:  # top-level: only the pipeline manifest is fetched
        return rfilename == manifest
    return not rfilename.startswith("assets/")


# No base weights when a single file is the whole pipeline (SDXL); config=base reads structure only.
_BASE_WEIGHT_EXTS = (
    ".safetensors",
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".gguf",
    ".onnx",
    ".onnx_data",
    ".msgpack",
    ".h5",
    ".pb",
)


def _base_config_file_downloaded(rfilename: str) -> bool:
    """True for base-repo files needed to BUILD a pipeline structure around a whole-pipeline single
    file WITHOUT its weights: config / tokenizer / scheduler JSON, but no weight tensors. Used
    for ``single_file_is_pipeline`` families."""
    if not _base_file_downloaded(rfilename):
        return False
    return not rfilename.lower().endswith(_BASE_WEIGHT_EXTS)


def _pipeline_file_downloaded(rfilename: str) -> bool:
    """True for files a full-pipeline ``from_pretrained`` fetches. Like ``_base_file_downloaded``
    but for the ``pipeline`` kind, where the repo supplies its OWN transformer weights, so the
    ``transformer/`` subfolder is kept. Top-level docs and ``assets/`` are skipped, and so are
    artifacts the torch loader never touches (ONNX / OpenVINO / Flax exports and dtype-variant
    twins), so an official repo that ships many formats does not prefetch tens of GB it will not
    load. A Modular Diffusers repo (FLUX.3) has ``modular_model_index.json`` as its manifest."""
    if "/" not in rfilename:  # top-level: only the pipeline manifest is fetched
        return rfilename in ("model_index.json", FLUX3_INDEX)
    lower = rfilename.lower()
    if lower.startswith(("assets/", "onnx/", "openvino/")):
        return False
    name = lower.rsplit("/", 1)[1]
    if name.startswith(("openvino_", "flax_")):
        return False
    if name.endswith((".onnx", ".onnx_data", ".pb", ".msgpack", ".h5", ".ckpt")):
        return False
    if ".fp16." in name or ".bf16." in name or ".non_ema." in name:
        return False
    return True


_DEFAULT_PIPELINE_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)"
    r"(?:-\d{5}-of-\d{5})?\.(?:bin|safetensors)$"
)
_DEFAULT_PIPELINE_WEIGHT_INDEX_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)\.(?:bin|safetensors)\.index\.json$"
)
_SELECTABLE_SAFETENSORS_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)\.safetensors$"
)
_SAFETENSORS_FLOAT_WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
_PIPELINE_WEIGHT_PREFIX_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)[.-].*"
    r"(?:bin|safetensors)(?:\.index.*\.json)?$"
)


def _explicit_pipeline_components(fam: Any) -> Optional[tuple[str, ...]]:
    """Component folders opened by image pipelines assembled outside generic Diffusers."""
    name = getattr(fam, "name", None)
    if is_flux3_family(fam):
        return FLUX3_COMPONENTS
    if name in (KREA2_FAMILY_NAME, MOSS_IMAGE_FAMILY_NAME):
        return ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
    if name == IDEOGRAM4_FAMILY_NAME:
        return (
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "unconditional_transformer",
            "vae",
        )
    return None


def _pipeline_components_from_index(
    repo_id: str,
    info: Any,
    hf_token: Optional[str],
    *,
    explicit_components: Optional[Sequence[str]] = None,
    failures_out: Optional[list] = None,
) -> Optional[tuple[frozenset[str], frozenset[str]]]:
    """Return selected component folders and exact model-index ignores. The manifest is small but
    authoritative. If it cannot be read, staging retains the prior best-effort listing and
    resident sizing declines to issue a hard refusal."""
    try:
        from huggingface_hub import hf_hub_download

        siblings = {s.rfilename for s in (info.siblings or ())}
        if "model_index.json" not in siblings:
            if explicit_components is not None and FLUX3_INDEX in siblings:
                # A modular repo (FLUX.3) whose assembler names its own folders: nothing else to read.
                return frozenset(str(name) for name in explicit_components), frozenset()
            raise FileNotFoundError(f"{repo_id} has no model_index.json")
        path = hf_hub_download(
            repo_id,
            "model_index.json",
            revision = getattr(info, "sha", None) or None,
            cache_dir = hub_cache_dir(),
            token = hf_token or None,
        )
        payload = json.loads(Path(path).read_text(encoding = "utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{repo_id}/model_index.json is not an object")
        if explicit_components is None:
            from .diffusion_component_sizes import manifest_entry_loaded
            selected = {
                str(name) for name, spec in payload.items() if manifest_entry_loaded(name, spec)
            }
        else:
            selected = {str(name) for name in explicit_components}
        ignored = payload.get("_ignore_files", ())
        if not isinstance(ignored, (list, tuple, set)):
            ignored = ()
        if not selected:
            raise ValueError(f"{repo_id}/model_index.json names no loadable components")
        return frozenset(selected), frozenset(str(name) for name in ignored)
    except Exception as exc:  # noqa: BLE001 -- no manifest means no hard memory verdict
        logger.warning("diffusion.pipeline_manifest_failed: %s", exc)
        if failures_out is not None:
            failures_out.append(exc)
        return None


def _pipeline_default_variant_file(rfilename: str) -> bool:
    """Whether a model-looking filename belongs to Diffusers' default variant."""
    name = rfilename.rsplit("/", 1)[-1].lower()
    if not _PIPELINE_WEIGHT_PREFIX_RE.match(name):
        return True
    return bool(
        _DEFAULT_PIPELINE_WEIGHT_RE.match(name) or _DEFAULT_PIPELINE_WEIGHT_INDEX_RE.match(name)
    )


def _pipeline_selected_file(
    rfilename: str, selection: tuple[frozenset[str], frozenset[str]]
) -> bool:
    """Match the files loaded by the selected components at ``variant=None``."""
    components, ignored = selection
    if rfilename in ignored:
        return False
    if rfilename in ("model_index.json", FLUX3_INDEX):
        return True
    if "/" not in rfilename or rfilename.split("/", 1)[0] not in components:
        return False
    return _pipeline_default_variant_file(rfilename)


def _pipeline_resident_weight(rfilename: str) -> bool:
    """Whether a selected file becomes tensor storage in the loaded pipeline (fp16 twins included)."""
    name = rfilename.rsplit("/", 1)[-1].lower()
    return bool(_DEFAULT_PIPELINE_WEIGHT_RE.match(name)) or is_fp16_twin(name)


def _swap_in_fp16_twins(repo_id: str, info: Any, picked: list[Any], hf_token: Any) -> list[Any]:
    """``picked`` with defaults swapped for ``.fp16`` twins; unchanged when only the defaults are cached."""
    if not fp16_variant_enabled():
        return picked
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        siblings = {s.rfilename: s for s in (info.siblings or ())}
        picked_names = [s.rfilename for s in picked]
        folders = {
            name.split("/", 1)[0]
            for name in picked_names
            if "/" in name and _pipeline_resident_weight(name)
        }
        if not folders:
            return picked
        revision = getattr(info, "sha", None) or None
        index_path = hf_hub_download(
            repo_id,
            "model_index.json",
            revision = revision,
            cache_dir = hub_cache_dir(),
            token = hf_token or None,
        )
        index = json.loads(Path(index_path).read_text(encoding = "utf-8"))
        plan = plan_fp16_twins(siblings, index, components = folders)
        if plan is None or set(plan) != folders:
            return picked
        defaults = {
            n for n in picked_names if n.split("/", 1)[0] in plan and _pipeline_resident_weight(n)
        }
        if len(defaults) != len(plan) or any(
            twin_of(n) != plan[n.split("/", 1)[0]] for n in defaults
        ):
            return picked

        def _cached(name: str) -> bool:
            hit = try_to_load_from_cache(
                repo_id, name, cache_dir = hub_cache_dir(), revision = revision
            )
            return isinstance(hit, str)

        if all(_cached(n) for n in defaults) and not all(_cached(t) for t in plan.values()):
            return picked
        twins = [siblings[t] for t in plan.values()]
        return [s for s in picked if s.rfilename not in defaults] + twins
    except Exception as exc:  # noqa: BLE001 - the defaults always load
        logger.debug("diffusion.fp16_variant: plan kept the default files (%r)", exc)
        return picked


def _progress(
    phase: Optional[str],
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
    fraction: float = 0.0,
    *,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
        "fraction": fraction,
        "error": error,
        "error_code": error_code,
    }


_diffusion_backend: Optional[DiffusionBackend] = None


def get_diffusion_backend() -> DiffusionBackend:
    global _diffusion_backend
    if _diffusion_backend is None:
        _diffusion_backend = DiffusionBackend()
    return _diffusion_backend


def generation_in_flight() -> bool:
    """Read the active-generation marker without constructing or locking the backend."""
    backend = _diffusion_backend
    return backend is not None and backend._gen is not None
