# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One persistent denoise thread per backend per card: cuDNN caches conv/SDPA plans thread_local (ATen Conv_v8.cpp, MHA.cpp)."""

from __future__ import annotations

import contextvars
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_RENDER_THREAD"
_EXECUTORS: dict[str, ThreadPoolExecutor] = {}
_RENDER_THREAD_IDS: set[int] = set()
_LOCK = threading.Lock()
# Renders submitted and not yet finished, per executor: background jobs (``submit_idle``) step aside while any wait.
_PENDING: dict[str, int] = {}


def enabled() -> bool:
    if os.environ.get(_ENV, "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available()) and getattr(torch.version, "hip", None) is None
    except Exception:  # noqa: BLE001
        return False


def _mark_render_thread() -> None:
    _RENDER_THREAD_IDS.add(threading.get_ident())


def _executor(name: str) -> ThreadPoolExecutor:
    with _LOCK:
        ex = _EXECUTORS.get(name)
        if ex is None:
            ex = ThreadPoolExecutor(
                max_workers = 1,
                thread_name_prefix = f"unsloth-{name}-render",
                initializer = _mark_render_thread,
            )
            _EXECUTORS[name] = ex
        return ex


def _apply_compile_config() -> None:
    try:
        from . import diffusion_compile_config  # noqa: PLC0415
        diffusion_compile_config.apply()
    except Exception:  # noqa: BLE001 - never fail a render over a config write
        pass


def run(name: str, fn: Callable[[], Any]) -> Any:
    if threading.get_ident() in _RENDER_THREAD_IDS or not enabled():
        _apply_compile_config()
        return fn()
    import torch  # noqa: PLC0415

    device = torch.cuda.current_device()
    inference = torch.is_inference_mode_enabled()
    grad = torch.is_grad_enabled()
    ctx = contextvars.copy_context()

    def call() -> Any:
        torch.cuda.set_device(device)
        _apply_compile_config()
        if inference:
            with torch.inference_mode():
                return fn()
        with torch.set_grad_enabled(grad):
            return fn()

    key = f"{name}-cuda{device}"
    with _LOCK:
        _PENDING[key] = _PENDING.get(key, 0) + 1
    try:
        return _executor(key).submit(ctx.run, call).result()
    finally:
        with _LOCK:
            _PENDING[key] -= 1


def renders_waiting(name: str, device: int) -> int:
    with _LOCK:
        return _PENDING.get(f"{name}-cuda{device}", 0)


def submit_idle(
    name: str,
    fn: Callable[[], Any],
    *,
    device: Optional[int] = None,
    yield_to_renders: bool = True,
) -> bool:
    """Queue ``fn`` on the render thread without waiting for it. True when queued.

    For work that must run on the thread every compile runs on but that no caller waits for (persisting compile
    artifacts, warming the compiler). With ``yield_to_renders`` a job that finds a render queued re-queues itself
    behind it, so a render waits for at most one job already running. Never runs ``fn`` inline: without a render
    thread (disabled, ROCm, CPU) nothing is queued."""
    if not enabled():
        return False
    import torch  # noqa: PLC0415

    dev = torch.cuda.current_device() if device is None else int(device)
    key = f"{name}-cuda{dev}"
    ctx = contextvars.copy_context()

    def call() -> Any:
        if yield_to_renders and renders_waiting(name, dev) > 0:
            _executor(key).submit(ctx.run, call)
            return None
        torch.cuda.set_device(dev)
        _apply_compile_config()
        try:
            return fn()
        except Exception:  # noqa: BLE001 - a background job never surfaces into a render
            return None

    _executor(key).submit(ctx.run, call)
    return True
