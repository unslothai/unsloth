# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle-unload watchdog (issue #5650).

Unloads the loaded inference model after a configurable idle period,
similar to llama-swap, ollama and openwebui.

Activity is tracked by :class:`ActivityMiddleware` (mounted from
``main.py``), which counts every inference request — including the full
lifetime of streaming responses — so a long generation cannot be cut
off mid-stream.

The watchdog reads its enabled flag and timeout from ``chat_settings``
on every tick, with ``UNSLOTH_IDLE_UNLOAD_MINUTES`` as an override for
headless / Tauri-controlled deployments. Disabled by default.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

_TICK_SECONDS = 30.0
_DEFAULT_IDLE_MINUTES = 10
_MIN_IDLE_MINUTES, _MAX_IDLE_MINUTES = 1, 24 * 60

_state_lock = threading.Lock()
_last_activity_at: Optional[float] = None
_inflight: int = 0


def record_activity() -> None:
    """Mark the model as just-used."""
    global _last_activity_at
    _last_activity_at = time.monotonic()


@contextmanager
def active_request():
    """Track a request as in-flight; the watchdog refuses to unload while
    any are running. Bumps ``last_activity`` on entry and exit."""
    global _inflight
    record_activity()
    with _state_lock:
        _inflight += 1
    try:
        yield
    finally:
        with _state_lock:
            _inflight = max(0, _inflight - 1)
        record_activity()


def has_inflight_requests() -> bool:
    with _state_lock:
        return _inflight > 0


def _reset_for_tests() -> None:
    global _last_activity_at, _inflight
    _last_activity_at = None
    with _state_lock:
        _inflight = 0


def _resolve_idle_minutes() -> Optional[int]:
    """Minutes of idle time before unloading, or ``None`` to stay quiet.

    ``UNSLOTH_IDLE_UNLOAD_MINUTES`` overrides the DB. ``0`` (or any
    non-positive integer) force-disables.
    """
    raw = os.environ.get("UNSLOTH_IDLE_UNLOAD_MINUTES", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("UNSLOTH_IDLE_UNLOAD_MINUTES is not an integer: %r", raw)
        else:
            if value <= 0:
                return None
            return max(_MIN_IDLE_MINUTES, min(_MAX_IDLE_MINUTES, value))

    # Lazy import: keeps tests that monkey-patch the DB free from
    # import-time side effects.
    try:
        from storage.studio_db import list_chat_settings

        settings = list_chat_settings()
    except Exception as exc:
        logger.debug("idle-unload: chat_settings unavailable: %s", exc)
        return None

    if not isinstance(settings, dict) or not settings.get("autoUnloadEnabled"):
        return None
    minutes = settings.get("autoUnloadIdleMinutes")
    if not isinstance(minutes, int) or minutes < _MIN_IDLE_MINUTES:
        minutes = _DEFAULT_IDLE_MINUTES
    return min(_MAX_IDLE_MINUTES, minutes)


def _try_unload_loaded_backend() -> Optional[str]:
    """Unload whichever backend currently holds a model. Returns the
    model identifier on success, ``None`` otherwise."""
    # GGUF (llama-server) — preferred path for Studio's chat flow.
    from routes.inference import get_llama_cpp_backend

    gguf = get_llama_cpp_backend()
    if gguf.is_active:
        if not gguf.is_loaded:
            return None  # mid-load: skip rather than fight the load lock
        identifier = getattr(gguf, "_model_identifier", None) or "<gguf>"
        try:
            gguf.unload_model()
        except Exception as exc:
            logger.warning("idle-unload: llama-cpp unload failed: %s", exc)
            return None
        return identifier

    # Orchestrator-backed inference (transformers / MLX).
    from core.inference import get_inference_backend

    backend = get_inference_backend()
    if backend is None or backend.is_model_loading():
        return None
    current = backend.get_current_model()
    if not current:
        return None
    try:
        backend.unload_model(current)
    except Exception as exc:
        logger.warning("idle-unload: inference unload failed: %s", exc)
        return None
    return current


async def _watchdog_iteration() -> None:
    """Single decision pass. Public for testing."""
    minutes = _resolve_idle_minutes()
    if minutes is None:
        return
    if _last_activity_at is None:
        record_activity()  # arm the timer on first tick
        return
    idle_for = time.monotonic() - _last_activity_at
    if idle_for < minutes * 60 or has_inflight_requests():
        return

    unloaded = await asyncio.to_thread(_try_unload_loaded_backend)
    if unloaded:
        logger.info(
            "idle-unload: unloaded '%s' after %ds idle (timeout=%dmin)",
            unloaded, int(idle_for), minutes,
        )
        _reset_for_tests()  # next /load re-arms the watchdog


async def _watchdog_loop() -> None:
    logger.info("idle-unload watchdog started (tick=%ds)", int(_TICK_SECONDS))
    try:
        while True:
            try:
                await _watchdog_iteration()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("idle-unload tick failed: %s", exc)
            await asyncio.sleep(_TICK_SECONDS)
    except asyncio.CancelledError:
        logger.info("idle-unload watchdog stopped")
        raise


def start_idle_unload_task(app) -> asyncio.Task:
    """Schedule the watchdog and stash it on ``app.state`` for shutdown."""
    task = asyncio.create_task(_watchdog_loop(), name = "unsloth-idle-unload")
    app.state.idle_unload_task = task
    return task


_ACTIVITY_PREFIXES: tuple[str, ...] = (
    "/api/inference/load",
    "/api/inference/chat/completions",
    "/api/inference/completions",
    "/api/inference/embeddings",
    "/api/inference/responses",
    "/api/inference/audio/generate",
    "/api/inference/generate/stream",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/messages",
)


class ActivityMiddleware:
    """Counts each inference request as in-flight for its full response
    lifetime — including streaming bodies. Starlette only completes the
    awaited ``app(scope, receive, send)`` call after the response body
    is fully sent, so a single ``with`` is enough."""

    def __init__(self, app, prefixes: tuple[str, ...] = _ACTIVITY_PREFIXES):
        self.app = app
        self.prefixes = prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not any(
            scope.get("path", "").startswith(p) for p in self.prefixes
        ):
            await self.app(scope, receive, send)
            return

        with active_request():
            await self.app(scope, receive, send)
