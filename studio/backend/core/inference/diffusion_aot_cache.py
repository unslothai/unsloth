# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Let AOTAutogradCache store the denoiser graphs that hold an ``sdpa_kernel`` block.

Dynamo bakes ``sdpa_kernel`` into the graph as ``_backend_from_string`` / ``_sdpa_kernel`` calls, which are not on
AOTAutogradCache's allowlist, so every regional block of FLUX.1, Z-Image and Qwen-Image re-ran AOT tracing on each
process start (pytorch/pytorch#194007; 2.6 / 4.7 / 1.8 s of a B200 restart's first render). Their arguments are graph
constants, so the graph keys them apart; the one ambient input, SDPA's backend priority order, goes into the marked
value torch folds into the cache key. UNSLOTH_DIFFUSION_SDPA_AOT_CACHE=0 disables it.
"""

from __future__ import annotations

import os
from typing import Any

from . import diffusion_compile_config as compile_config

_ENV = "UNSLOTH_DIFFUSION_SDPA_AOT_CACHE"
_INDUCTOR_MODULE = "torch._inductor.config"
_KNOB = "unsafe_marked_cacheable_functions"
# The names check_node_safe builds: f"{target.__module__}.{target.__name__}".
SDPA_HELPERS = ("torch.nn.attention._backend_from_string", "torch.nn.attention._sdpa_kernel")


def enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _cache_key_value(torch: Any) -> str:
    try:
        priority = tuple(int(b) for b in torch._C._get_sdp_priority_order())
    except Exception:  # noqa: BLE001 - no priority API on this build: key on the version alone
        priority = None
    return f"{torch.__version__}|sdp_priority={priority}"


def install(logger: Any = None) -> bool:
    """Mark the two SDPA helpers cacheable (idempotent, process-wide). False when disabled or unsupported."""
    if not enabled():
        return False
    try:
        import torch
        import torch.nn.attention as attention

        if not all(
            callable(getattr(attention, name.rsplit(".", 1)[1], None)) for name in SDPA_HELPERS
        ):
            return False
        # Both the recorded knob (re-applied on render threads) and this thread's live value.
        recorded = compile_config.get_knob(_INDUCTOR_MODULE, _KNOB)
        live = getattr(getattr(torch._inductor, "config", None), _KNOB, None)
        if not isinstance(recorded, dict) or not isinstance(live, dict):
            return False
        value = _cache_key_value(torch)
        if all(recorded.get(n) == value and live.get(n) == value for n in SDPA_HELPERS):
            return True
        merged = {**recorded, **live}
        merged.update({name: value for name in SDPA_HELPERS})
        if not compile_config.set_knob(_INDUCTOR_MODULE, _KNOB, merged):
            return False
        if logger is not None:
            logger.info(
                "diffusion.aot_cache: sdpa_kernel graphs are AOTAutograd-cacheable (%s)", value
            )
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.debug("diffusion.aot_cache: not installed (%s: %s)", type(exc).__name__, exc)
        return False


# ------------------------------------------------------------------------------------- libdevice pin (restart hits)
# Under ``emulate_precision_casts`` (which every Studio regional compile sets), torch switches Triton from its bundled
# libdevice to the CUDA toolkit's copy, for eager-matching numerics, but only LAZILY: inside the first Triton kernel
# COMPILE of the process (torch._inductor.runtime.compile_tasks._set_triton_libdevice_path). Since torch records
# ``extern_libs_key`` (the libdevice file hash) on every FX graph cache entry and checks it on lookup, the entry a
# compile wrote carries the toolkit's hash while a fresh process looks it up BEFORE any kernel compile, still on the
# bundled file: a silent ``guard_miss``, so every restart recompiled every graph (and the AOTAutograd entry pointing at
# it missed too), persistent inductor dir and Mega-cache bundle notwithstanding. Measured on Wan2.2-TI2V-5B (torch
# 2.14.1, Triton 3.8, A100 and B200): ``fxgraph_cache_miss`` on every restart, with identical FX / AOT keys.
#
# Pinning the path up front, with torch's own resolver, makes the lookup see what the save saw. It changes nothing a
# compile would not have changed anyway (torch sets the same knob and env var at the first kernel compile and never
# resets them), so the kernels and the output are the ones a cold compile produces.
# UNSLOTH_DIFFUSION_PIN_LIBDEVICE=0 disables it.
_LIBDEVICE_ENV = "UNSLOTH_DIFFUSION_PIN_LIBDEVICE"


def pin_libdevice_enabled() -> bool:
    raw = (os.environ.get(_LIBDEVICE_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def pin_triton_libdevice(logger: Any = None) -> Any:
    """Resolve Triton's libdevice now, exactly as torch would at the first kernel compile. Returns the pinned path, or
    None when disabled or when this torch / Triton build has nothing to pin (no extern_libs cache check, no lazy
    resolver, no CUDA). Never raises. Call only when emulate_precision_casts (or use_pytorch_libdevice) is on: that is
    the condition under which torch makes the switch."""
    if not pin_libdevice_enabled():
        return None
    try:
        import warnings

        import torch

        if getattr(torch.version, "cuda", None) is None:
            return None
        # Probe the capability, not the version: only builds that key cache entries on extern libs can miss on it.
        if not callable(getattr(getattr(torch.utils, "_triton", None), "_extern_libs_key", None)):
            return None
        from torch._inductor.runtime import compile_tasks

        resolve = getattr(compile_tasks, "_set_triton_libdevice_path_impl", None)
        if not callable(resolve):
            return None
        from triton import knobs

        before = knobs.nvidia.libdevice_path
        with warnings.catch_warnings():
            # The CUDA_HOME-missing warning is torch's to give at compile time; the bundled file is then consistent.
            warnings.simplefilter("ignore")
            resolve()
        after = knobs.nvidia.libdevice_path
        if logger is not None and after != before:
            logger.info("diffusion.aot_cache: Triton libdevice pinned before the first compile (%s)", after)
        return after
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.debug("diffusion.aot_cache: libdevice not pinned (%s: %s)", type(exc).__name__, exc)
        return None
