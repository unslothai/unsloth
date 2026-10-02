# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Let AOTAutogradCache store the denoiser graphs that hold an ``sdpa_kernel`` block.

Dynamo bakes every ``sdpa_kernel`` context into the FX graph as two helper calls, ``_backend_from_string`` and
``_sdpa_kernel`` (``torch/_dynamo/variables/ctx_manager.py``), and neither is on AOTAutogradCache's allowlist, so
every such graph logs "Bypassing autograd cache due to: Unsupported call_function target _backend_from_string" and
re-runs AOT tracing + partitioning on EVERY process start, warm Mega-cache bundle or not (pytorch/pytorch#194007).
diffusers' ``_native_cudnn`` / ``_native_efficient`` / ``_native_flash`` attention backends wrap each SDPA call in
one, so this hits every regional block of FLUX.1, Z-Image and Qwen-Image. Measured on a B200 restart (torch 2.11,
int8 denoisers): AOT dispatch 2.6 / 4.7 / 1.8 s of the first render, of which inductor's own (cached) codegen is only
0.5 / 0.8 / 0.5 s.

Both helpers are cache-safe for the calls dynamo emits here: their arguments are constants in the graph (backend
names, ``set_priority``), and the graph is the cache key, so two graphs that pick different backends key apart.
The one piece of ambient state is SDPA's backend priority order, which ``_sdpa_kernel(..., set_priority=True)``
reads; that order goes into the value of ``unsafe_marked_cacheable_functions``, which torch folds into the cache key,
so a process running a different order never hits these entries. A cache hit returns the artifact the same graph
compiled to before, so outputs are unchanged.

UNSLOTH_DIFFUSION_SDPA_AOT_CACHE=0 disables it (the graphs bypass the cache, as before).
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
        # The recorded knob (what render threads re-apply) and this thread's live value: both must carry the helpers.
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
