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


# libdevice pin: under emulate_precision_casts torch switches Triton to the CUDA toolkit libdevice only inside the
# first kernel compile, but FX cache entries are keyed on its hash (extern_libs_key) and looked up before that, so
# every restart silently missed (Wan2.2-TI2V-5B, torch 2.14.1). Pinning up front with torch's own resolver sets the
# same knob a compile would. UNSLOTH_DIFFUSION_PIN_LIBDEVICE=0 disables it.
_LIBDEVICE_ENV = "UNSLOTH_DIFFUSION_PIN_LIBDEVICE"


def pin_libdevice_enabled() -> bool:
    raw = (os.environ.get(_LIBDEVICE_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def pin_triton_libdevice(logger: Any = None) -> Any:
    """Resolve libdevice now as torch would at first compile; returns the path or None. Never raises.
    Call after emulate_precision_casts is set: that is when torch 2.13+ switches."""
    if not pin_libdevice_enabled():
        return None
    try:
        import warnings

        import torch

        if getattr(torch.version, "cuda", None) is None:
            return None
        # Probe capability, not version; import it since the submodule is not an attribute until imported.
        try:
            import importlib
            utils_triton = importlib.import_module("torch.utils._triton")
        except Exception:  # noqa: BLE001 - no such module on this build
            return None
        if not callable(getattr(utils_triton, "_extern_libs_key", None)):
            return None
        from torch._inductor.runtime import compile_tasks

        # The GATED resolver, never ``_set_triton_libdevice_path_impl``: on torch 2.12 the impl
        # would pick a libdevice file torch itself never selects.
        resolve = getattr(compile_tasks, "_set_triton_libdevice_path", None)
        if not callable(resolve):
            return None
        from triton import knobs

        before = knobs.nvidia.libdevice_path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolve()
        after = knobs.nvidia.libdevice_path
        if logger is not None and after != before:
            logger.info(
                "diffusion.aot_cache: Triton libdevice pinned before the first compile (%s)", after
            )
        return after
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.debug(
                "diffusion.aot_cache: libdevice not pinned (%s: %s)", type(exc).__name__, exc
            )
        return None
