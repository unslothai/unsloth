# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run every denoise of a backend on ONE persistent thread.

cuDNN keeps its convolution benchmark cache (``cudnn.benchmark``) and its SDPA execution-plan cache per THREAD
(``thread_local`` in ATen's Conv_v8.cpp and MHA.cpp). The routes run ``generate`` on a pooled ``asyncio.to_thread``
worker, and the default executor hands consecutive jobs to different idle threads, so a render that lands on a thread
that never rendered re-benchmarks every VAE convolution and rebuilds every attention plan. Measured on a B200 with
Qwen-Image-2.1 at 1024px: the VAE decode went from 0.22-0.30 s to 4.8-10.6 s on a fresh thread, the same prompt and
seed on a thread that had rendered before cost nothing extra.

Only the pipeline call hops; locks, cancellation and admission stay on the calling thread. The hop carries the
caller's context variables, CUDA device and inference mode. CUDA only (ROCm and other devices keep running inline).
Kill switch: ``UNSLOTH_DIFFUSION_RENDER_THREAD=0``.
"""

from __future__ import annotations

import contextvars
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_ENV = "UNSLOTH_DIFFUSION_RENDER_THREAD"
_EXECUTORS: dict[str, ThreadPoolExecutor] = {}
_RENDER_THREAD_IDS: set[int] = set()
_LOCK = threading.Lock()


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


def run(name: str, fn: Callable[[], Any]) -> Any:
    """``fn()`` on the persistent render thread ``name``, blocking the caller; inline when disabled or already there."""
    if threading.get_ident() in _RENDER_THREAD_IDS or not enabled():
        return fn()
    import torch  # noqa: PLC0415

    # The caller has already pinned the pipeline's card; one thread per card keeps two GPUs from queueing on each other.
    device = torch.cuda.current_device()
    inference = torch.is_inference_mode_enabled()
    grad = torch.is_grad_enabled()
    ctx = contextvars.copy_context()

    def call() -> Any:
        torch.cuda.set_device(device)
        if inference:
            with torch.inference_mode():
                return fn()
        with torch.set_grad_enabled(grad):
            return fn()

    return _executor(f"{name}-cuda{device}").submit(ctx.run, call).result()
