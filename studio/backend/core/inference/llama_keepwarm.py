# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in idle auto-unload (TTL keep-warm) for the local llama.cpp model.

Off by default (idle seconds = 0). When enabled, a background loop unloads the
loaded GGUF once it has been idle for the configured TTL, freeing VRAM. A
pure-ASGI middleware tracks in-flight inference requests so a long stream that
outlives the TTL is never unloaded mid-response.
"""

from __future__ import annotations

import asyncio
import threading
import time

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_inflight = 0
_last_active = time.monotonic()

_INFERENCE_PREFIXES = ("/v1/", "/api/inference/")
_INFERENCE_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/messages",
    "/embeddings",
    "/responses",
)


def _is_inference_path(path: str) -> bool:
    return path.startswith(_INFERENCE_PREFIXES) and path.endswith(_INFERENCE_SUFFIXES)


def _note_start() -> None:
    global _inflight, _last_active
    with _lock:
        _inflight += 1
        _last_active = time.monotonic()


def _note_end() -> None:
    global _inflight, _last_active
    with _lock:
        _inflight = max(0, _inflight - 1)
        _last_active = time.monotonic()


def _is_idle(ttl_seconds: float) -> bool:
    with _lock:
        return _inflight == 0 and (time.monotonic() - _last_active) >= ttl_seconds


def _note_activity() -> None:
    """Stamp activity, e.g. on a (re)load, so the model survives at least one TTL."""
    global _last_active
    with _lock:
        _last_active = time.monotonic()


class LlamaKeepWarmMiddleware:
    """Pure ASGI: count in-flight inference requests and stamp activity on completion."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not _is_inference_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        # Track only while the feature is enabled; otherwise pass straight through.
        from utils.openai_auto_switch_settings import get_auto_unload_idle_seconds

        if get_auto_unload_idle_seconds() <= 0:
            await self.app(scope, receive, send)
            return

        _note_start()
        ended = {"done": False}

        async def send_wrapper(message):
            # Final body frame marks the end of a (possibly streaming) response.
            if message.get("type") == "http.response.body" and not message.get("more_body", False):
                if not ended["done"]:
                    ended["done"] = True
                    _note_end()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if not ended["done"]:
                ended["done"] = True
                _note_end()


async def idle_unload_loop(poll_seconds: float = 15.0) -> None:
    """Unload the loaded GGUF once idle past the configured TTL. Inert when off."""
    from utils.openai_auto_switch_settings import get_auto_unload_idle_seconds

    seen_model = None
    while True:
        await asyncio.sleep(poll_seconds)
        try:
            ttl = get_auto_unload_idle_seconds()
            if ttl <= 0:
                continue
            from routes.inference import get_llama_cpp_backend

            backend = get_llama_cpp_backend()
            # A newly loaded or swapped model counts as activity so it is never
            # unloaded before its first request (loads bypass the inference
            # middleware that stamps activity).
            current = backend.model_identifier if backend.is_loaded else None
            if current != seen_model:
                seen_model = current
                if current is not None:
                    _note_activity()
            # Re-check idle just before unload to shrink the arrival race window.
            if backend.is_loaded and _is_idle(ttl):
                await asyncio.to_thread(backend.unload_model)
                logger.info("Idle auto-unload: freed GGUF after %ss idle", ttl)
                seen_model = None
        except Exception as exc:
            logger.debug("idle_unload_loop iteration failed: %s", exc)
