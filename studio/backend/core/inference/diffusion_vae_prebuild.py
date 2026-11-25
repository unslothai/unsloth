# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Child process warms Triton cache for fused VAE kernels without weights; parent never waits;
UNSLOTH_DIFFUSION_VAE_PREBUILD=0 disables"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional
from .rocm_bf16 import is_rocm_torch

_ENV = "UNSLOTH_DIFFUSION_VAE_PREBUILD"
_FALSE = ("0", "false", "no", "off")
# Multiple of 16 at every decoder stage, well under 1 GiB.
_LATENT_SIDE = 64
# A 32x video VAE (LTX-2.x) decodes [1, C, 2, 8, 8] to 9 frames of 256 px: every fused kernel, a few hundred MiB.
_VIDEO_LATENT = (2, 8, 8)
# Video loads have no compile bundle, so a marker under the Triton cache dir says a build already ran there.
_VIDEO_CLASSES = frozenset({"AutoencoderKLLTX2Video"})
_MARKER_DIR = "unsloth_vae_prebuild"
# Free VRAM needed after the load before a second CUDA context opens.
_MIN_FREE_BYTES = 6 * 1024**3
_CHILD_TIMEOUT_S = 180.0
# Card-wide VRAM a video prebuild child holds while it builds (measured 2.4 GiB on B200 for LTX-2.5), with margin.
_CHILD_BYTES = 3 * 1024**3
# Live children -> (owner, device): an image unload must not cancel a video load's child, nor the reverse.
_LIVE: dict = {}
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
    elif isinstance(cfg.get("latent_channels"), int) and "temporal_compression_ratio" in cfg:
        shape = [1, cfg["latent_channels"], *_VIDEO_LATENT]
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

        if is_rocm_torch(torch) or not torch.cuda.is_available():
            return False
        if compile_ctx is None or getattr(compile_ctx, "hit", False):
            return False  # a restart's Triton cache already holds the kernels
        job = plan(pipe)
        if job is None:
            return False
        from .diffusion_memory import user_budget_reserve_bytes

        free, total = torch.cuda.mem_get_info(job["device"])
        pending = _pending_runtime_bytes(pipe)
        if free - user_budget_reserve_bytes(job["device"], total) - pending < _MIN_FREE_BYTES:
            if pending and logger is not None:
                logger.info(
                    "diffusion.vae_prebuild: skipped (the streamed placement still allocates %d MiB)",
                    pending >> 20,
                )
            return False
        job["owner"] = "image"
        return _spawn(job, logger)
    except Exception as exc:  # noqa: BLE001 - a prebuild, never a failed load
        if logger is not None:
            logger.info("diffusion.vae_prebuild: skipped (%s: %s)", type(exc).__name__, exc)
        return False


def _triton_cache_dir() -> Optional[str]:
    path = os.environ.get("TRITON_CACHE_DIR")
    if not path:
        try:
            import triton
            path = triton.knobs.cache.dir
        except Exception:  # noqa: BLE001 - an older Triton: its documented default, rooted at TRITON_HOME
            home = os.environ.get("TRITON_HOME") or os.path.expanduser("~")
            path = os.path.join(home, ".triton", "cache")
    return str(path) if path else None


def _fused_source_digest() -> Optional[str]:
    """The fused kernels' source: an edited kernel is a new Triton cache entry, so it must be a new marker too."""
    import hashlib
    try:
        from . import diffusion_vae_fused
        with open(diffusion_vae_fused.__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _marker_path(job: dict) -> Optional[str]:
    """One file per (VAE class, config, dtype, latent, torch, Triton, card arch): wiping the Triton cache drops it."""
    import hashlib
    import json

    import torch

    root = _triton_cache_dir()
    if not root:
        return None
    try:
        import triton
        triton_version = triton.__version__
    except Exception:  # noqa: BLE001
        triton_version = None
    key = json.dumps(
        [
            job.get("name"),
            job.get("config"),
            job.get("dtype"),
            job.get("shape"),
            str(torch.__version__),
            triton_version,
            list(torch.cuda.get_device_capability(job["device"])),
            _fused_source_digest(),
            # Triton keys its cache on these too (TRITON_DISABLE_LINE_INFO, TRITON_DEFAULT_FP_FUSION, ...).
            sorted((k, v) for k, v in os.environ.items() if k.startswith("TRITON_")),
        ],
        sort_keys = True,
    )
    return os.path.join(root, _MARKER_DIR, hashlib.sha256(key.encode()).hexdigest()[:32] + ".done")


def maybe_kick_video(pipe: Any, logger: Any = None) -> bool:
    """The video load's prebuild: same child and gates as ``maybe_kick``, skipped once this Triton cache holds the
    build (no compile bundle to say so). Never blocks, never raises."""
    if not enabled():
        return False
    try:
        import torch

        if is_rocm_torch(torch) or not torch.cuda.is_available():
            return False
        job = plan(pipe)
        if job is None or job["name"] not in _VIDEO_CLASSES:
            return False
        marker = _marker_path(job)
        if marker is None or os.path.exists(marker):
            return False
        from .diffusion_memory import user_budget_reserve_bytes

        free, total = torch.cuda.mem_get_info(job["device"])
        pending = _pending_runtime_bytes(pipe)
        if free - user_budget_reserve_bytes(job["device"], total) - pending < _MIN_FREE_BYTES:
            return False
        job["marker"] = marker
        job["owner"] = "video"
        return _spawn(job, logger)
    except Exception as exc:  # noqa: BLE001 - a prebuild, never a failed load
        if logger is not None:
            logger.info("diffusion.vae_prebuild: skipped (%s: %s)", type(exc).__name__, exc)
        return False


def _pending_runtime_bytes(pipe: Any) -> int:
    """Streamed-tier seed + prefetch window: budgeted but not yet allocated when this gate reads free memory."""
    reserve = getattr(pipe, "_unsloth_measured_reserve", None)
    if not reserve or len(reserve) <= 12 or not reserve[12]:
        return 0
    from .diffusion_memory import _stream_window_mib

    return (int(reserve[0] or 0) + max(0, int(_stream_window_mib(pipe)))) << 20


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
        _LIVE[proc] = (job.get("owner", "image"), int(job["device"]))
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
        _LIVE.pop(proc, None)
    if proc.is_alive():
        proc.kill()
        proc.join(5.0)
    try:
        from utils.process_lifetime import forget_pid
        if not proc.is_alive():
            forget_pid(proc.pid)
    except Exception:  # noqa: BLE001
        pass


def cancel_all(owner: Optional[str] = None) -> int:
    """Kill ``owner``'s ("image" / "video"; None = all) live prebuild children on unload. Never raises."""
    with _LIVE_LOCK:
        procs = [p for p, (o, _d) in _LIVE.items() if owner is None or o == owner]
    for proc in procs:
        try:
            _stop(proc)
        except Exception:  # noqa: BLE001
            pass
    return len(procs)


def yield_to_render(owner: str, logger: Any = None) -> int:
    """Stop ``owner``'s still-building children on a card too tight for child + render (the render then compiles in
    the foreground, as without a prebuild). Never raises."""
    stopped = 0
    try:
        import torch

        from .diffusion_memory import user_budget_reserve_bytes

        with _LIVE_LOCK:
            live = [(p, d) for p, (o, d) in _LIVE.items() if o == owner]
        for proc, device in live:
            if not proc.is_alive():
                continue
            free, total = torch.cuda.mem_get_info(device)
            if free - user_budget_reserve_bytes(device, total) >= _MIN_FREE_BYTES + _CHILD_BYTES:
                continue
            _stop(proc)
            stopped += 1
            if logger is not None:
                logger.info(
                    "diffusion.vae_prebuild: stopped the %s child (pid %s): the render needs the VRAM",
                    owner,
                    proc.pid,
                )
    except Exception:  # noqa: BLE001 - a prebuild, never a failed render
        pass
    return stopped


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
    # A fused pass that fell back to stock compiled nothing: no marker, so the next load tries again.
    if any(getattr(m, "_unsloth_vae_fused_failed", False) for m in vae.modules()):
        return
    marker = job.get("marker")
    if marker:
        os.makedirs(os.path.dirname(marker), exist_ok = True)
        with open(marker, "w", encoding = "utf-8") as f:
            f.write(job["name"] + "\n")
