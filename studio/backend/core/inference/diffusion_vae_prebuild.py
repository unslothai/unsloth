# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the fused VAE's Triton kernels in a short-lived child process while a first start compiles its denoiser.

The fused VAE passes (``diffusion_vae_fused``) are plain ``@triton.jit`` kernels: the first decode of a process with an
empty Triton cache JIT-compiles every one of them, 2.8-3.4 s of render 1 on an RTX PRO 6000 (first decode + post
3.4-3.9 s on a first start vs 0.5-0.6 s on a restart, FLUX.1 / Z-Image / Qwen-Image). Triton keeps compiled kernels in
``TRITON_CACHE_DIR``, which every later start already reads.

At the end of a load that found no compile bundle (the first start for that model), a spawned child builds the same
VAE class from the same config with no weights, installs the same fused passes and decodes zeros once at a small
latent (every integer the kernels specialise on keeps its divisibility class). It writes the kernels into the shared
Triton cache while the parent spends ~8-12 s compiling its denoiser on render 1, so the parent's first decode loads
them instead of compiling. Its own process (no GIL, no thread shared with the parent's compiles), its own CUDA
context, gone when it finishes; the parent never waits for it, and kernels it has not finished are simply compiled
by the parent as before. Same kernel source, same specialisation, so the same binary either way.

Kill switch: ``UNSLOTH_DIFFUSION_VAE_PREBUILD=0``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

_ENV = "UNSLOTH_DIFFUSION_VAE_PREBUILD"
_FALSE = ("0", "false", "no", "off")
# Latent side the child decodes: small enough to cost well under 1 GiB, a multiple of 16 at every decoder stage.
_LATENT_SIDE = 64
# Free VRAM the parent must still have after its load before a second CUDA context is opened next to it.
_MIN_FREE_BYTES = 6 * 1024**3
_CHILD_TIMEOUT_S = 180.0


def enabled() -> bool:
    return (os.environ.get(_ENV) or "").strip().lower() not in _FALSE


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def plan(pipe: Any) -> Optional[dict]:
    """What the child needs to rebuild ``pipe.vae``'s fused decode, or None when there is nothing to prebuild."""
    import torch

    vae = getattr(pipe, "vae", None)
    if vae is None or not getattr(vae, "_unsloth_vae_fused_installed", 0):
        return None
    config = getattr(vae, "config", None)
    if config is None:
        return None
    try:
        decoder = getattr(vae, "decoder", None) or vae
        weight = next(p for p in decoder.parameters() if p.is_floating_point())
    except StopIteration:
        return None
    if weight.device.type != "cuda":
        return None
    cfg = _jsonable(dict(config))
    z_dim = cfg.get("z_dim")
    if isinstance(z_dim, int):
        shape = [1, z_dim, 1, _LATENT_SIDE, _LATENT_SIDE]
    elif isinstance(cfg.get("latent_channels"), int):
        shape = [1, cfg["latent_channels"], _LATENT_SIDE, _LATENT_SIDE]
    else:
        return None
    return {
        "module": type(vae).__module__,
        "name": type(vae).__name__,
        "config": cfg,
        "dtype": str(weight.dtype).replace("torch.", ""),
        "device": int(
            weight.device.index if weight.device.index is not None else torch.cuda.current_device()
        ),
        "shape": shape,
    }


def maybe_kick(
    pipe: Any,
    compile_ctx: Any,
    logger: Any = None,
) -> bool:
    """Spawn the prebuild child for a first-start load (no compile-bundle hit). Never blocks, never raises."""
    if not enabled():
        return False
    try:
        import torch

        if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            return False
        if compile_ctx is None or getattr(compile_ctx, "hit", False):
            return False  # a restart's Triton cache already holds the kernels
        job = plan(pipe)
        if job is None:
            return False
        free, _total = torch.cuda.mem_get_info(job["device"])
        if free < _MIN_FREE_BYTES:
            return False
        return _spawn(job, logger)
    except Exception as exc:  # noqa: BLE001 - a prebuild, never a failed load
        if logger is not None:
            logger.info("diffusion.vae_prebuild: skipped (%s: %s)", type(exc).__name__, exc)
        return False


def _spawn(job: dict, logger: Any) -> bool:
    import multiprocessing as mp

    from utils.native_path_leases import (
        native_path_secret_removed_for_child_start,
        run_without_native_path_secret,
    )

    ctx = mp.get_context("spawn")
    with native_path_secret_removed_for_child_start():
        proc = ctx.Process(
            target = run_without_native_path_secret,
            args = (__name__, "_child_entry", {}, job),
            daemon = True,
        )
        proc.start()
    try:
        from utils.process_lifetime import adopt_pid
        adopt_pid(proc.pid)
    except Exception:  # noqa: BLE001
        pass

    def reap() -> None:
        proc.join(_CHILD_TIMEOUT_S)
        if proc.is_alive():
            proc.kill()
            proc.join(5.0)
        try:
            from utils.process_lifetime import forget_pid
            if not proc.is_alive():
                forget_pid(proc.pid)
        except Exception:  # noqa: BLE001
            pass
        if logger is not None:
            logger.info("diffusion.vae_prebuild: child exited (%s)", proc.exitcode)

    threading.Thread(target = reap, name = "unsloth-vae-prebuild-reaper", daemon = True).start()
    if logger is not None:
        logger.info(
            "diffusion.vae_prebuild: building %s's fused kernels in a child (pid %s)",
            job["name"],
            proc.pid,
        )
    return True


def _child_entry(job: dict) -> None:
    """Child side: same VAE class and config, no weights, the same fused passes, one decode of zeros."""
    import importlib

    import torch

    from core.inference import diffusion_vae_fused

    torch.cuda.set_device(job["device"])
    cls = getattr(importlib.import_module(job["module"]), job["name"])
    dtype = getattr(torch, job["dtype"])
    vae = cls.from_config(job["config"]).to(device = f"cuda:{job['device']}", dtype = dtype).eval()
    if not diffusion_vae_fused.install(vae, None):
        return
    with torch.inference_mode():
        z = torch.zeros(job["shape"], device = f"cuda:{job['device']}", dtype = dtype)
        vae.decode(z)
    torch.cuda.synchronize()
