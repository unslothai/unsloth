# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run each backend's denoise on one persistent thread per card: cuDNN caches conv benchmarks and SDPA plans
per thread (ATen Conv_v8.cpp / MHA.cpp), and ``asyncio.to_thread`` rotates workers. CUDA only.
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
    if threading.get_ident() in _RENDER_THREAD_IDS or not enabled():
        return fn()
    import torch  # noqa: PLC0415

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
