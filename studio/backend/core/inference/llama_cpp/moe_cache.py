# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MoE expert cache and lazy tensor reads."""

import os
import re

from typing import (
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _CACHE_RAM_FLAGS,
    _DEVICE_FLAGS,
    _GPU_LAYER_FLAGS,
    _last_flag_value,
    _MOE_OFFLOAD_FLAGS,
    _SPLIT_MODE_FLAGS,
    resolve_effective_load_state,
)

from .memory_policy import (
    _CPU_FFN_COUNT_FLAGS,
    _TENSOR_SPLIT_FLAGS,
)

# GPU cache of host-resident experts (ggml-org/llama.cpp#29887); "auto" is sized by the
# fork's fitter (unslothai/llama.cpp#251). One run, so a retry can strip it.
_MOE_CACHE_AUTO_TOKENS = ("--moe-cache-mib", "auto")
# --moe-cache-mib auto only beats the --fit split at a middling resident expert share r
# (unknown r keeps the split). Cache vs split decode / prompt t/s, L4 gemma-4-26B-A4B:
# r 0.23 23.5/384 vs 24.5/448, r 0.53 44.9/556 vs 31.2/593, r 0.82 48.5/592 vs 42.0/897.
# RTX PRO 6000 Qwen3.8-Flash-Next: r <= 0.745 decode +24-42%, r 0.846 prompt -6-40%.
_MOE_CACHE_MIN_RESIDENT_SHARE = 0.35
_MOE_CACHE_MAX_RESIDENT_SHARE = 0.75
# Drafted launches may get the cache. L4 gemma-4 decode cache/split at r 0.53/0.82/0.23:
# MTP 1.76x/1.25x/0.96x, ngram-mod 1.19x/0.87x/1.11x (the band covers the losses).
_MOE_CACHE_WITH_DRAFTER = True
# Skip the spill micro-batch raise if it pushes more than this share of resident experts
# off the GPU. L4 gemma-4: r 0.23 lost 14% (decode -3%), r 0.53/0.82 lost 4-6%.
_MOE_SPILL_UB_MAX_RESIDENT_LOSS = 0.10
# "on" = ungated v0.1.906, "off" = v0.1.905 (no cache / no raise), else auto.
_MOE_CACHE_ENV = "UNSLOTH_MOE_CACHE"
_MOE_SPILL_UBATCH_ENV = "UNSLOTH_MOE_SPILL_UBATCH"
_MOE_SWITCH_ON = frozenset({"1", "on", "true", "yes", "force"})
_MOE_SWITCH_OFF = frozenset({"0", "off", "false", "no", "never"})


def _moe_switch(name: str, env: Optional[Mapping[str, str]] = None) -> str:
    """``"on"``, ``"off"`` or ``"auto"`` for one of the MoE override env vars."""
    raw = str((os.environ if env is None else env).get(name) or "").strip().lower()
    if raw in _MOE_SWITCH_ON:
        return "on"
    if raw in _MOE_SWITCH_OFF:
        return "off"
    return "auto"


def _moe_resident_expert_share(
    shortfall_bytes: Optional[float], expert_bytes: Optional[int]
) -> Optional[float]:
    """1 - shortfall / expert bytes (the fitter offloads experts first); None if unknown."""
    if shortfall_bytes is None or not expert_bytes or expert_bytes <= 0:
        return None
    return min(1.0, max(0.0, 1.0 - float(shortfall_bytes) / float(expert_bytes)))


def _moe_cache_share_admits(
    resident_share: Optional[float],
    *,
    drafter: bool,
    switch: str = "auto",
) -> bool:
    """Whether ``--moe-cache-mib auto`` beats the --fit split at this resident share."""
    if switch != "auto":
        return switch == "on"
    if drafter and not _MOE_CACHE_WITH_DRAFTER:
        return False
    return (
        resident_share is not None
        and _MOE_CACHE_MIN_RESIDENT_SHARE <= resident_share <= _MOE_CACHE_MAX_RESIDENT_SHARE
    )


def _moe_spill_raise_keeps_experts(
    share_before: Optional[float],
    share_after: Optional[float],
    *,
    switch: str = "auto",
) -> bool:
    """Keep the spill micro-batch raise? Nothing spilling = no; unpriceable = yes."""
    if switch != "auto":
        return switch == "on"
    if share_before is None:
        return True
    if share_before >= 1.0:
        return False
    if share_after is None:
        return True
    return share_before - share_after <= _MOE_SPILL_UB_MAX_RESIDENT_LOSS * share_before


# User placement: the fitter aborts on most of it, and the cache refuses >1 device (--rpc adds one).
_MOE_CACHE_USER_OWNED_FLAGS = (
    frozenset({"--moe-cache-mib", "-ot", "--override-tensor", "--rpc"})
    | _CPU_FFN_COUNT_FLAGS
    | _DEVICE_FLAGS
    | _GPU_LAYER_FLAGS
    | _MOE_OFFLOAD_FLAGS
    | _SPLIT_MODE_FLAGS
    | _TENSOR_SPLIT_FLAGS
)
_MOE_CACHE_USER_OWNED_ENV = (
    "LLAMA_ARG_MOE_CACHE_MIB",
    "LLAMA_ARG_OVERRIDE_TENSOR",
    "LLAMA_ARG_CPU_MOE",
    "LLAMA_ARG_N_CPU_MOE",
    "LLAMA_ARG_N_CPU_FFN",
    "LLAMA_ARG_DEVICE",
    "LLAMA_ARG_MAIN_GPU",
    "LLAMA_ARG_SPLIT_MODE",
    "LLAMA_ARG_TENSOR_SPLIT",
    "LLAMA_ARG_RPC",
)
# Exact startup errors: healthy launches also log "moe cache auto:" and "MoE cache size =".
_MOE_CACHE_ERROR_RE = re.compile(
    r"MoE cache (?:requires a GPU backend|does not support tensor parallelism"
    r"|requires a MoE model|is too small to hold the experts of one token)"
    r"|failed to (?:create the MoE cache context|allocate the MoE cache buffers)"
    r"|the MoE cache is too small for the experts selected in layer",
    re.IGNORECASE,
)

# TENSOR_READ_LAZY tensors (src/models/gemma4.cpp, qwen4exp.cpp): under --lazy-mode auto one
# over 4 GiB stays mmap'd under every load mode (lazy_read::add), needing no pinned RAM or VRAM.
_LAZY_READ_TENSORS: dict[str, tuple[str, ...]] = {
    "gemma4": ("per_layer_token_embd.weight",),
    "qwen4exp": ("per_layer_token_embd.weight",),
}
_LAZY_READ_AUTO_MIN_BYTES = 4 * 1024**3
# Embedded NextN/MTP heads llama.cpp skips at load unless --spec-type draft-mtp, measured:
# Nemotron 3.5 Super loads 2987 MiB less with MTP off (b11642 + ggml-org#30520).
_NEXTN_SKIPPED_WITHOUT_MTP_ARCHS = frozenset({"nemotron_h_moe"})
# --tensor-read-lazy predates ggml-org/llama.cpp#27969; a build reads only its own spelling.
_LAZY_MODE_SPELLINGS: dict[str, tuple[frozenset[str], str]] = {
    "--lazy-mode": (frozenset({"-lzm", "--lazy-mode"}), "LLAMA_ARG_LAZY_MODE"),
    "--tensor-read-lazy": (frozenset({"--tensor-read-lazy"}), "LLAMA_ARG_TENSOR_READ_LAZY"),
}


# llama-server's --cache-ram default (MiB): the host-RAM prompt cache it grows into.
_LLAMA_CACHE_RAM_DEFAULT_MIB = 8192


def _prompt_cache_host_bytes(
    cache_ram: Optional[int],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    *,
    supports_cache_ram: bool,
) -> Optional[int]:
    """The host RAM llama-server's prompt cache may grow into, or None when it is
    unbounded (``-1``) or unreadable. Env first, then the launch's own value, then a
    last-wins pass-through flag; zero on a build without the cache."""
    if not supports_cache_ram:
        return 0
    source_env = os.environ if env is None else env
    value: object = _LLAMA_CACHE_RAM_DEFAULT_MIB
    if str(source_env.get("LLAMA_ARG_CACHE_RAM") or "").strip():
        value = source_env.get("LLAMA_ARG_CACHE_RAM")
    if cache_ram is not None:
        value = cache_ram
    try:
        override = _last_flag_value(extra_args, _CACHE_RAM_FLAGS)
        if override is not None:
            value = override
        mib = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if mib < 0 else mib * 1024 * 1024


def _argv_loads_pinned(argv: Iterable[str], env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``argv`` (with the env twins) loads the weights into a pinned host
    copy, i.e. ``--load-mode none``: no mmap, no page-lock, no DirectIO."""
    mlock, reserves_ram, direct_io = resolve_effective_load_state(list(argv), env)
    return bool(reserves_ram and not mlock and not direct_io)


def _effective_lazy_mode(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    *,
    flag: str = "--lazy-mode",
) -> str:
    """llama.cpp's lazy mode for this launch: env first, then last-wins argv, both in
    the build's own spelling ``flag``.

    A value llama-server would refuse reads as "off", which discounts nothing."""
    source_env = os.environ if env is None else env
    flags, env_name = _LAZY_MODE_SPELLINGS.get(flag, _LAZY_MODE_SPELLINGS["--lazy-mode"])
    mode = "auto"
    value = str(source_env.get(env_name) or "").strip().lower()
    if value:
        mode = value
    try:
        value = _last_flag_value(extra_args, flags)
    except ValueError:
        return "off"
    if value:
        mode = value.strip().lower()
    return mode if mode in ("auto", "on") else "off"


def _lazy_read_bytes(
    architecture: Optional[str], tensor_bytes: Mapping[str, int], mode: str
) -> int:
    """Bytes of ``tensor_bytes`` llama.cpp reads lazily under ``mode``."""
    if mode not in ("auto", "on"):
        return 0
    names = _LAZY_READ_TENSORS.get(architecture or "", ())
    return sum(
        int(size)
        for name, size in tensor_bytes.items()
        if name in names and (mode == "on" or int(size) > _LAZY_READ_AUTO_MIN_BYTES)
    )
