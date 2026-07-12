# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-GPU arbiter for Studio's heavy GPU consumers.

The chat backends, diffusion, and video share one GPU. Before taking it each calls
``acquire_for(owner)``, which evicts the current other owner so two large models never sit in VRAM
at once. The arbiter only sequences ownership (freeing is each backend's teardown); eviction runs
under the lock, so a transfer is atomic vs other acquires.
"""

from __future__ import annotations

import threading
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

CHAT = "chat"
DIFFUSION = "diffusion"
VIDEO = "video"

_lock = threading.Lock()
_owner: Optional[str] = None


def _evict_chat() -> None:
    import time

    from core.inference import get_inference_backend
    from routes.inference import get_llama_cpp_backend

    llama = get_llama_cpp_backend()
    # is_active (process exists), not is_loaded (exists AND healthy): a chat model still starting
    # up holds VRAM but isn't healthy, so is_loaded would skip it and let the load race diffusion.
    if llama.is_active:
        llama.unload_model()
    orchestrator = get_inference_backend()
    if orchestrator.active_model_name:
        orchestrator.unload_model(orchestrator.active_model_name)
    # Kill the subprocess too: its base CUDA context holds VRAM diffusion needs.
    orchestrator._shutdown_subprocess(timeout = 5.0)
    # The driver reclaims the killed VRAM asynchronously; wait for it to settle before diffusion
    # allocates, else a warm chat->diffusion handoff can transiently OOM.
    llama._wait_for_vram_settle(since_kill = time.monotonic())


def _evict_diffusion() -> None:
    # Unload whichever engine the router has active (diffusers or native sd.cpp), so a
    # chat acquire frees the right one.
    from core.inference.diffusion_engine_router import get_active_diffusion_engine
    get_active_diffusion_engine().unload()


def _evict_video() -> None:
    from core.inference.video import get_video_backend
    get_video_backend().unload()


# Patchable in tests via monkeypatch.setitem. Ownership is exclusive, so acquire_for's
# evict-the-current-owner generalises to any number of registered owners.
_EVICTORS = {CHAT: _evict_chat, DIFFUSION: _evict_diffusion, VIDEO: _evict_video}


def acquire_for(owner: str) -> None:
    """Make ``owner`` the sole GPU owner, evicting the other if it holds it."""
    global _owner
    if owner not in _EVICTORS:
        raise ValueError(f"unknown GPU owner: {owner!r}")
    with _lock:
        if _owner is not None and _owner != owner:
            logger.info("gpu_arbiter: evicting %s for %s", _owner, owner)
            _EVICTORS[_owner]()
        _owner = owner


def release(owner: str) -> None:
    """Drop ``owner``'s claim (no-op if it isn't the current owner)."""
    global _owner
    with _lock:
        if _owner == owner:
            _owner = None


def current_owner() -> Optional[str]:
    return _owner
