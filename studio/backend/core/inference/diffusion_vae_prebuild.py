# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the fused VAE's Triton kernels in a child process while a first start compiles its denoiser.

The first decode with an empty Triton cache JIT-compiles every fused VAE kernel (~3 s of render 1 on an RTX PRO 6000).
On a load with no compile bundle, a spawned child rebuilds the VAE from its config without weights, installs the same
fused passes and decodes zeros at a small latent (same divisibility classes), filling the shared Triton cache. The
parent never waits; kernels the child has not finished are compiled by the parent as before.
``UNSLOTH_DIFFUSION_VAE_PREBUILD=0`` disables it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

_ENV = "UNSLOTH_DIFFUSION_VAE_PREBUILD"
_FALSE = ("0", "false", "no", "off")
# Multiple of 16 at every decoder stage, well under 1 GiB.
_LATENT_SIDE = 64
# Free VRAM needed after the load before a second CUDA context opens.
_MIN_FREE_BYTES = 6 * 1024**3
_CHILD_TIMEOUT_S = 180.0
_LIVE: set = set()
_LIVE_LOCK = threading.Lock()


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
    from utils.process_lifetime import adopt_pid, is_process_shutting_down

    # A quit during the load: the sweep may already have run, so a child started now would outlive the server.
    if is_process_shutting_down():
        return False
    ctx = mp.get_context("spawn")
    with native_path_secret_removed_for_child_start():
        proc = ctx.Process(
            target = run_without_native_path_secret,
            args = (__name__, "_child_entry", {}, job),
            daemon = True,
        )
        proc.start()
    with _LIVE_LOCK:
        _LIVE.add(proc)
    try:
        adopt_pid(proc.pid)
    except Exception:  # noqa: BLE001
        pass
    # Recheck once the pid is recorded: the latch can be set between the gate above and the adoption.
    if is_process_shutting_down():
        _stop(proc)
        return False

    def reap() -> None:
        proc.join(_CHILD_TIMEOUT_S)
        _stop(proc)
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


def _stop(proc: Any) -> None:
    with _LIVE_LOCK:
        _LIVE.discard(proc)
    if proc.is_alive():
        proc.kill()
        proc.join(5.0)
    try:
        from utils.process_lifetime import forget_pid
        if not proc.is_alive():
            forget_pid(proc.pid)
    except Exception:  # noqa: BLE001
        pass


def cancel_all() -> int:
    """Kill every live prebuild child (unload: its CUDA context and VAE must not shadow the next load). Never raises."""
    with _LIVE_LOCK:
        procs = list(_LIVE)
    for proc in procs:
        try:
            _stop(proc)
        except Exception:  # noqa: BLE001
            pass
    return len(procs)


def _child_entry(job: dict) -> None:
    """Child side: same VAE class and config, no weights, the same fused passes, one decode of zeros."""
    import diffusers
    import torch

    from core.inference import diffusion_vae_fused

    # Only diffusers' own exported VAE classes: the job names a class, it never picks a module to import.
    cls = getattr(diffusers, job["name"], None)
    if cls is None or getattr(cls, "__module__", None) != job["module"]:
        return
    torch.cuda.set_device(job["device"])
    dtype = getattr(torch, job["dtype"])
    vae = cls.from_config(job["config"]).to(device = f"cuda:{job['device']}", dtype = dtype).eval()
    if not diffusion_vae_fused.install(vae, None):
        return
    with torch.inference_mode():
        z = torch.zeros(job["shape"], device = f"cuda:{job['device']}", dtype = dtype)
        vae.decode(z)
    torch.cuda.synchronize()
