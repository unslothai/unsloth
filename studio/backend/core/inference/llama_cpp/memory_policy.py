# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VRAM budgets, KV cache sizing and memory policy constants."""

import os

from typing import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _flag_name,
    _LAYER_OFFLOAD_FLAGS,
    _MOE_OFFLOAD_FLAGS,
    _SPLIT_MODE_FLAGS,
    _TENSOR_SPLIT_FLAGS,
    parse_cache_override_per_axis,
)

# Below this many B params, draft-mtp regresses vs spec-off (bench in
# _build_speculative_flags); auto mode drops MTP under it.
_MTP_MIN_SIZE_B = 3.0

# Cap total GPU occupancy at this fraction of the card. The fit reserves an
# absolute (1 - frac) * total per GPU when total VRAM is known, else a fraction
# of free (see _fit_context_to_vram), plus a byte-accurate MTP draft reserve.
# 3%: the context-linear compute buffer is now modelled (_compute_buffer_ctx_bytes),
# so this cushion no longer covers it - only fragmentation, the per-device CUDA
# context on a multi-GPU split, and MoE routing, which measure ~2-3% (Qwen3.5-397B on
# 3 GPUs under-predicts by 2.7%). Below 3% one fragmentation spike overflows to CPU.
_CTX_FIT_VRAM_FRACTION = 0.97

# Apple unified memory is shared with the OS, so tighter than VRAM. Matches the
# 0.85 MLX uses in mlx_inference.py (_configure_memory_limits); not kept in sync.
_APPLE_UNIFIED_MEMORY_FRACTION = 0.85

# The 5% of wired headroom held back absorbs error in the footprint estimate.
_APPLE_WIRED_CEILING_FRACTION = 0.95

# _fit_context_to_vram's floor: the shortest context the search will settle for, so a
# value equal to it is that floor rather than a measurement. The Metal branch re-prices
# from _FIT_FLOOR_MIN_CTX (the search's 256 alignment step) before trusting it.
#
# 8192, not llama.cpp's own 4096 (common_params.fit_params_min_ctx). 4096 is too short
# to be useful for chat once tools and a system prompt are in the window, and the fit
# search would settle there rather than spilling weights it could spill instead. The two
# floors are now deliberately different numbers: llama.cpp's still governs ITS fitter,
# and Unsloth does not pass -fitc, so a launch that hands llama.cpp an explicit -c can
# still be reduced below this by the child's own fit step.
#
# Every arm now floors here, Metal included. The Apple arm used to hold a separate
# hardcoded 4096, which made a Mac publish half the context a discrete GPU published
# for the same model. Being BELOW llama.cpp's own floor is what made that safe to
# raise: -c 8192 is a request the child's --fit can still reduce, down to its 4096,
# so the arm hands over a ceiling rather than a commitment. The one arm still on a
# bare 4096 is _plan_tensor_parallel's no-KV fallback, which cannot prove a safe cap
# at all and is a different question from this floor.
_FIT_MIN_CTX = 8192
_FIT_FLOOR_MIN_CTX = 256

# llama.cpp's OWN reduction floor: common_params.fit_params_min_ctx, 4096
# (common/common.h), the smallest context its --fit step will shrink a request to.
# Named because the paragraph above spends the whole justification for _FIT_MIN_CTX on
# it: -c 8192 is safe to hand a Mac only as a ceiling the child's fitter can still come
# down from, and this is how far down. A launch that hands over 8192 with the fitter
# NOT running has no such backstop, so it must hand over this number itself rather than
# the ceiling that assumed one.
_LLAMA_FIT_MIN_CTX = 4096

# _plan_tensor_parallel's no-KV fallback. A GUESS: may shrink Auto, never a request (#9653).
_TP_UNMEASURED_CTX = 4096

# Auto only reaches this path when no discrete-GPU subset can hold the model.
# llama.cpp must offload layers to host RAM either way, so prefer a context that
# remains useful for chat. Separate from _FIT_MIN_CTX: that value is also a search
# and Apple unified-memory safety floor.
_AUTO_OFFLOAD_CTX = 8192

# Keep this at or above the fit search floor. Not a style rule, a correctness one:
# the Auto offload branch re-checks whether a subset holds the model at the reduced
# context, and that re-check can only award residency BELOW the floor, since a subset
# winnable at or above it was already taken by the fit loop that runs first. Moving
# this constant is therefore free of placement changes only while it stays on the dead
# side of that floor. Below it the re-check hands over a device and flips --fit off,
# which is a placement decision rather than a display one.
#
# The floor in question is the ``min_ctx = 4096`` DEFAULT on _fit_context_to_vram and
# _cap_ctx_to_per_device_reserve, not _FIT_MIN_CTX: neither auto call site passes the
# argument, and _FIT_MIN_CTX is only handed in explicitly on the Apple arm. The three
# numbers agree today and nothing here makes them; test_auto_offload_ctx_invariants.py
# pins both defaults to _FIT_MIN_CTX and this constant above it, so the agreement
# cannot lapse silently.

# How far amd-smi's total VRAM may sit from HIP's before the two are reporting
# different memory scopes rather than one pool (an APU carve-out against the GTT
# pool, a partition against the whole card). Same 10% margin
# utils/hardware/hardware.py::_rocm_system_wide_vram_by_index gates its overlay on.
_AMD_SMI_TOTAL_TOLERANCE = 0.10

# Flat MTP reserve, used only when GGUF dims are too sparse for the byte-accurate
# reserve (_estimate_mtp_overhead_bytes). Applied to both the fit budget and pin.
_MTP_VRAM_RESERVE_FRAC = 0.05


# Left on every card even at 100%, matching the --fit-target margin llama.cpp
# keeps for its own fitter.
_VRAM_FLOOR_RESERVE_MIB = 512.0

# llama-server's own --fit-target default ("target margin per device for --fit,
# ... default: 1024"): what the fallback fitter leaves when we pass nothing.
_LLAMA_FIT_TARGET_DEFAULT_MIB = 1024.0

# Windows + discrete NVIDIA: since driver 536.40 WDDM serves VRAM overflow from host RAM
# instead of failing (unslothai/unsloth#11349), so match llama.cpp's own fit margin.
_WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB = _LLAMA_FIT_TARGET_DEFAULT_MIB
_WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION = 0.125


def _vram_reserve_floor_mib(total_mib: float, *, sysmem_fallback: bool = False) -> float:
    """Smallest margin a card keeps: 512 MiB, or the default's own reserve if smaller.

    Capping it at the default keeps the budget monotonic. A flat 512 MiB would
    RAISE the reserve the moment the slider passed the default on any card under
    about 17 GiB, where 3% is already less than that: an 8 GiB card offers 7946 MiB
    at 0.97 but only 7680 at 0.971, so nudging the slider up would hand back less
    context, the opposite of what the control promises. Capped, the reserve never
    rises with the fraction, and at or below the default the percentage term always
    wins, so an unset budget is untouched on every card size.

    ``sysmem_fallback`` raises the floor (flat in ``frac``, so still monotonic) on hosts
    whose driver spills an overshoot into system RAM; it overrides a 100% budget on purpose.
    """
    floor = min(_VRAM_FLOOR_RESERVE_MIB, (1.0 - _CTX_FIT_VRAM_FRACTION) * total_mib)
    if not sysmem_fallback:
        return floor
    return max(
        floor,
        min(
            _WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB,
            _WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION * total_mib,
        ),
    )


def _vram_usable_mib(
    free_mib: float,
    total_mib: float,
    frac: float,
    *,
    pooled: bool = False,
    sysmem_fallback: bool = False,
) -> float:
    """Free MiB one card offers a load at ``frac``, unclamped.

    ``pooled`` marks an already absolute budget summed over cards that have each
    paid their own reserve; charging it again would double count. Everything else
    is one device, including a probe reporting free memory with no total, which
    still has to keep a margin: there the free reading is the only scale available,
    and it agrees with the known-total form at the default.

    ``sysmem_fallback``: see ``LlamaCppBackend._sysmem_fallback_risk()``; ``pooled`` ignores it.
    """
    if total_mib and total_mib > 0:
        reserve = max(
            (1.0 - frac) * total_mib,
            _vram_reserve_floor_mib(total_mib, sysmem_fallback = sysmem_fallback),
        )
        return free_mib - reserve
    if pooled:
        return free_mib * frac
    return min(
        free_mib * frac,
        free_mib - _vram_reserve_floor_mib(free_mib, sysmem_fallback = sysmem_fallback),
    )


def _model_memory_settings_or_none() -> Optional[tuple[bool, bool]]:
    try:
        from utils.model_memory_settings import get_model_memory_settings
        return get_model_memory_settings()
    except Exception:
        return None


def _active_vram_fraction() -> float:
    """The user's VRAM budget, or ``_CTX_FIT_VRAM_FRACTION`` when they set none.

    The settings module clamps to a supported range and swallows a corrupt stored
    value, so this cannot widen the budget past 1.0 or hand back a NaN. Imported
    lazily to keep the settings/storage layer off this module's import path, and
    read per call so a change applies to the next load without a restart.
    """
    try:
        from utils.vram_budget_settings import get_vram_budget_fraction
        return get_vram_budget_fraction()
    except Exception:
        # Settings are a convenience; a load must never fail because of them.
        return _CTX_FIT_VRAM_FRACTION


def _kv_bytes_per_elem(cache_type: Optional[str]) -> float:
    """Bytes per KV-cache element for a llama.cpp cache type (f16 default)."""
    return {
        "f32": 4.0,
        "f16": 2.0,
        "bf16": 2.0,
        "q8_0": 34 / 32,
        "q5_1": 0.75,
        "q5_0": 0.6875,
        "q4_1": 0.625,
        "q4_0": 0.5625,
        "iq4_nl": 0.5625,
    }.get((cache_type or "f16").strip().lower(), 2.0)


# Not a CUDA/HIP FlashAttention KV type (ggml_cuda_fattn_kv_type_supported), so attention runs
# on the CPU. q4_1 / q5_0 / q5_1 did too until ggml-org/llama.cpp#28079 added an f16 fallback.
# Mapped to a same-size type that has the kernel, so the KV budget is unchanged (#10336).
_GPU_UNACCELERATED_CACHE_TYPES = {"iq4_nl": "q4_0"}


def _kv_cache_gpu_fallback_warning(
    cache_type_kv: Optional[str],
    gpu_memory_mode: str,
    gpu_layers: int,
    installed_backends: Callable[[], frozenset[str]],
) -> Optional[str]:
    """Warning for a KV cache type that runs attention on the CPU of a CUDA/HIP launch."""
    if cache_type_kv not in _GPU_UNACCELERATED_CACHE_TYPES:
        return None
    if gpu_memory_mode == "manual" and gpu_layers == 0:
        return None
    if not {"cuda", "hip"} & installed_backends():
        return None
    return (
        f"KV cache type {cache_type_kv} has no CUDA/HIP flash-attention kernel, so attention "
        "would fall back to CPU, causing high CPU load and slow generation. Using "
        f"{_GPU_UNACCELERATED_CACHE_TYPES[cache_type_kv]} instead, the same size on the GPU."
    )


# Upper bound on any current tokenizer, for output rows when a truncated header drops
# the token array. Above Llama 4 / Gemma 3 (256k), the widest shipping.
_ASSUMED_MAX_VOCAB = 262144
# Ceiling on the float32 activations a compute graph keeps per micro-batch token, for a
# header that cannot be read: 4 * n_ff at n_ff = 65536, above Llama 3.1 405B's 53056.
_ASSUMED_MAX_ACTIVATION_WIDTH = 262144


def _pad_kv_cells(cells: int) -> int:
    return ((cells + 255) // 256) * 256


def _kv_cache_cell_layout(n_ctx: int, n_parallel: int, kv_unified: bool) -> tuple[int, int, int]:
    """Return llama.cpp's slot count, stream count, and cells per stream."""
    slots = max(1, n_parallel)
    padded_ctx = _pad_kv_cells(n_ctx)
    streams = 1 if kv_unified else slots
    if padded_ctx <= 0:
        return slots, streams, 0
    cells_per_stream = padded_ctx if kv_unified else _pad_kv_cells(padded_ctx // slots)
    return slots, streams, cells_per_stream


def _env_main_cache_type_for_budget(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Heavier (max bytes/elem) of the set LLAMA_ARG_CACHE_TYPE_K/_V env types
    when it budgets differently from the f16 default, else None. Unsloth emits
    --cache-type only for the param/extras path, so an env type would otherwise
    reach the child unbudgeted: f32 under-reserves the KV, and a quantized type
    skips the dequant compute scratch (on Inkling the dense fallback it forces
    dwarfs the banded rate). The caller adopts the type for the reserve only;
    _cache_type_from_env keeps it out of the emitted flags."""
    e = os.environ if env is None else env
    # An UNSET axis is not absent, it is llama.cpp's f16 default, so it joins the max.
    # Dropping it let a lone CACHE_TYPE_V=q4_0 return q4_0 and price K as q4 while the
    # child ran it at f16, under-reserving the larger tensor when key_length >
    # value_length -- and tensor mode has no --fit valve to absorb that.
    axis_types = [
        (e.get(var) or "").strip().lower() or "f16"
        for var in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V")
    ]
    if all(t == "f16" for t in axis_types):
        return None
    heaviest = max(axis_types, key = _kv_bytes_per_elem)
    if _kv_bytes_per_elem(heaviest) == _kv_bytes_per_elem("f16"):
        return None
    return heaviest


def _extra_args_main_cache_type_for_budget(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Heavier (max bytes/elem) of the explicit --cache-type-k/-v extras, or None.

    Extras are appended last and win per axis, so an asymmetric K=f32,V=f16 must be
    budgeted by its heavier axis. resolve_cache_type_kv returns only the last-wins
    single type, which under-reserves the heavier axis when the lighter one is last."""
    k, v = parse_cache_override_per_axis(extra_args)
    candidates = [c for c in (k, v) if c]
    if not candidates:
        return None
    return max(candidates, key = _kv_bytes_per_elem)


def _effective_main_cache_types(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[str, str]:
    """Effective main K/V cache types after environment and CLI precedence."""
    source_env = os.environ if env is None else env
    env_k = (source_env.get("LLAMA_ARG_CACHE_TYPE_K") or "f16").strip().lower()
    env_v = (source_env.get("LLAMA_ARG_CACHE_TYPE_V") or "f16").strip().lower()
    arg_k, arg_v = parse_cache_override_per_axis(args)
    return (
        (arg_k or env_k).strip().lower(),
        (arg_v or env_v).strip().lower(),
    )


def _planned_main_cache_types(
    cache_type_kv: Optional[str],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """Main K/V types the loader's managed flags and user extras will produce."""
    args = list(extra_args or ())
    emitted_type = _extra_args_main_cache_type_for_budget(args) or cache_type_kv
    if emitted_type:
        args = [
            "--cache-type-k",
            emitted_type,
            "--cache-type-v",
            emitted_type,
            *args,
        ]
    return _effective_main_cache_types(args, env)


# What kv_cache_type_from_str accepts, so the only types Unsloth emits. Module scope
# because the budget must know whether a type will reach the child before the command
# is assembled.
_VALID_CACHE_TYPES = frozenset(
    {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32"}
)


def _planned_scratch_cache_type(
    cache_type_kv: Optional[str],
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """LIGHTEST planned axis: the type that decides the compute-buffer scratch rate.

    The KV-byte reserve takes the HEAVIER axis (max over _planned_main_cache_types),
    which is conservative for storage but hides a paired quantized axis -- an f16
    budget hides a q4_0 K. _compute_buffer_ctx_bytes charges the per-device dequant
    scratch only when its cache type is under f16, so handing it the heavier axis
    silently drops that scratch for an asymmetric pair. Taking the lighter axis here
    makes any quantized axis select the quantized rate, leaving the two terms
    independently conservative (heavier for bytes, lighter for scratch)."""
    return min(_planned_main_cache_types(cache_type_kv, extra_args, env), key = _kv_bytes_per_elem)


def _auto_mode_drops_mtp(
    req_mode: Optional[str],
    size_b: Optional[float],
    *,
    has_separate_drafter: bool = False,
) -> bool:
    """Auto mode drops MTP below _MTP_MIN_SIZE_B for an embedded draft head
    (its per-token cost regresses there); a separate drafter (Gemma) is a tiny
    standalone model that still speeds up below 3B, so it never drops. Forced
    mtp / mtp+ngram engage regardless of size."""
    if has_separate_drafter:
        return False
    return req_mode == "auto" and size_b is not None and size_b < _MTP_MIN_SIZE_B


# MLA archs whose MTP context covers only the NextN block instead of duplicating the
# trunk KV: glm5next holds 4+3 MiB over one layer where its trunk holds 48+36 over
# twelve. That one fact is why Auto keeps MTP (gate below) and why the fit must not
# reserve the copy (_estimate_mtp_overhead_bytes). "glm5-next" = upstream name
# (ggml-org/llama.cpp#27773); builds lacking its NextN graph retry without speculation.
_MLA_MTP_FAST_ARCHS = frozenset({"glm5next", "glm5-next"})
# Smallest context Auto shrinks to so a fast MLA head still launches.
_FAST_MTP_MIN_CTX = 65536
# Max share of the target-only Auto context a drafter may take at one slot.
_MTP_SLOT_CTX_TRIM = 0.10


def _arch_has_fast_mla_mtp(architecture: Optional[str]) -> bool:
    """Whether this MLA architecture's embedded MTP head is worth promoting in Auto."""
    return bool(architecture) and str(architecture).strip().lower() in _MLA_MTP_FAST_ARCHS


def _arch_mtp_skips_target_kv_copy(architecture: Optional[str]) -> bool:
    """Whether this MLA architecture's MTP context skips the duplicated target KV."""
    return _arch_has_fast_mla_mtp(architecture)


def _mla_mtp_auto_enabled() -> bool:
    """Whether Auto may pick embedded MTP for an MLA model (GLM-5.2/DeepSeek/Kimi).

    Off by default: llama.cpp's MLA/DSA MTP path keeps a duplicated full target-KV
    context and recomputes the sparse-attention indexer every draft step, so it runs
    ~2x slower than no speculation (GLM-5.2 bench: 27 vs 45 tok/s, flat across draft
    depth and 96-100% acceptance) -- the opposite of the vLLM/SGLang speedup on the
    same model. Set UNSLOTH_MLA_MTP_ENABLED=1 to let Auto promote MLA MTP again once
    that path is optimized upstream. Forced mtp / mtp+ngram ignore this gate."""
    return os.environ.get("UNSLOTH_MLA_MTP_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _extra_args_set_spec_type(extra_args: Optional[Iterable[str]]) -> bool:
    """User passed --spec-type / --spec-default? llama-server takes one
    --spec-type (comma-separated to chain), so suppress auto-emit."""
    return _extra_args_set_any_flag(extra_args, {"--spec-type", "--spec-default"})


# Layer-offload override detection. Single-sourced from llama_server_args, which
# also strips these (plus the MoE flags) from inherited extras; sharing the layer
# set keeps detection and stripping from drifting.
_GPU_OFFLOAD_OVERRIDE_FLAGS = _LAYER_OFFLOAD_FLAGS
# Extras that keep weights on the CPU regardless of the layer count, so a
# full offload cannot be inferred from -ngl alone. -ncmoe carries a COUNT, and
# zero places nothing, so it is parsed rather than merely spotted; the rest are
# presence-only (-cmoe takes no value, and an -ot pattern is not worth guessing
# at, which matches how the env side reads LLAMA_ARG_OVERRIDE_TENSOR).
_CPU_MOE_COUNT_FLAGS = frozenset({"-ncmoe", "--n-cpu-moe"})
# The dense-model twin: common/arg.cpp turns a positive -ncffn into CPU overrides for the
# first N layers' ffn_up/down/gate, exactly as -ncmoe does for expert weights.
_CPU_FFN_COUNT_FLAGS = frozenset({"-ncffn", "--n-cpu-ffn"})
# User controls over llama.cpp's own fitter. Studio emits neither on Metal, so a launch carrying
# one is placed by a fitter the Metal memory probe did not run.
_FIT_CTX_FLAGS = frozenset({"-fitc", "--fit-ctx"})
_FIT_CONTROL_FLAGS = frozenset({"-fitt", "--fit-target"}) | _FIT_CTX_FLAGS
_FIT_CONTROL_ENV_VARS = ("LLAMA_ARG_FIT_TARGET", "LLAMA_ARG_FIT_CTX")
_CPU_PLACEMENT_PRESENCE_FLAGS = (_MOE_OFFLOAD_FLAGS - _CPU_MOE_COUNT_FLAGS) | frozenset(
    {"-ot", "--override-tensor"}
)
_CPU_PLACEMENT_FLAGS = _MOE_OFFLOAD_FLAGS | frozenset({"-ot", "--override-tensor"})
# Pass-through that re-places what a fit priced: the fitter, split, remote (RPC) devices,
# draft tensors moved onto a card of their own, or an unpriced vocoder GGUF.
_VRAM_FIT_VOIDING_FLAGS = (
    frozenset(
        {
            "-fit",
            "--fit",
            "--rpc",
            "-otd",
            "--override-tensor-draft",
            "--spec-draft-override-tensor",
            "-mv",
            "--model-vocoder",
        }
    )
    | _FIT_CONTROL_FLAGS
    | _TENSOR_SPLIT_FLAGS
    | _SPLIT_MODE_FLAGS
)
_VRAM_FIT_VOIDING_ENV_VARS = (
    "LLAMA_ARG_FIT",
    "LLAMA_ARG_RPC",
    *_FIT_CONTROL_ENV_VARS,
    "LLAMA_ARG_TENSOR_SPLIT",
    "LLAMA_ARG_SPLIT_MODE",
)
_THREAD_OVERRIDE_FLAGS = frozenset({"-t", "--threads"})
_GENERATION_CPU_AFFINITY_FLAGS = frozenset({"-C", "--cpu-mask", "-Cr", "--cpu-range"})
_TENSOR_SPLIT_FLAGS = frozenset({"--tensor-split", "-ts"})


def _extra_args_set_any_flag(extra_args: Optional[Iterable[str]], flags: Collection[str]) -> bool:
    if not extra_args:
        return False
    for raw in extra_args:
        flag = _flag_name(str(raw))
        if flag in flags:
            return True
    return False
