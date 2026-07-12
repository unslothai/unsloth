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
import contextlib
import threading
import time

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_inflight = 0
# Requests blocked on the unload gate but not yet in _inflight: the idle loop must not unload
# while one is waiting (it would unload out from under it).
_pending = 0
_last_active = time.monotonic()
# The (id, quant) idle-unload last freed, so an alias/unknown request that would 503 against an
# empty backend can reload the exact freed variant (set on unload, cleared on reload).
_last_unloaded_model = None
# Guards inflight bumps against the idle-check-then-unload race and blocks new inference mid-swap.
# Process-wide, not per-loop: the backend slot is shared across every loop, so a per-loop gate
# would let a request on loop B start while a swap on loop A tears the model down.
_lifecycle_lock = threading.Lock()


@contextlib.asynccontextmanager
async def _unload_gate():
    # Acquire off the loop: non-blocking first, else poll off a short sleep. Polling keeps the wait
    # off the loop AND cancellation-safe (a cancel lands during the sleep, gate not held).
    while not _lifecycle_lock.acquire(blocking = False):
        await asyncio.sleep(0.02)
    try:
        yield
    finally:
        _lifecycle_lock.release()


_INFERENCE_PREFIXES = ("/v1/", "/api/inference/")
_INFERENCE_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/messages",
    "/messages/count_tokens",  # uses the loaded tokenizer; protect like /messages
    "/embeddings",
    "/responses",
    "/generate/stream",  # Studio's own streaming route on the same llama-server
    "/audio/generate",  # direct GGUF TTS; can outlive the idle TTL
    # Image generation holds a multi-GB pipeline for the whole request; tracking it lets
    # other_inference_request_count() see an in-flight generation so an API-key training start is
    # refused (409) before its unload cancels it. endswith avoids matching *-progress / */cancel.
    "/images/generate",  # /api/inference/images/generate
    "/images/generations",  # /v1/images/generations (+ /api/inference/images/generations)
    # Video runs as a background job (the POST returns at once), so this only covers the brief
    # accept; the training-start guards additionally probe generate-progress for an in-flight clip.
    "/video/generate",  # /api/inference/video/generate
)


def _is_inference_path(path: str) -> bool:
    if path.startswith(_INFERENCE_PREFIXES) and path.endswith(_INFERENCE_SUFFIXES):
        return True
    # Public checkpoint preview (/p/{run}/v1/chat/completions) streams from the same backend.
    return path.startswith("/p/") and path.endswith("/v1/chat/completions")


def _note_pending() -> None:
    global _pending
    with _lock:
        _pending += 1


def _note_unpending() -> None:
    global _pending
    with _lock:
        _pending = max(0, _pending - 1)


def _note_start() -> None:
    # Do not stamp _last_active here: while _inflight > 0 the model is already protected, and
    # stamping on start would let a later-untracked external request reset the idle timer.
    global _inflight, _pending
    with _lock:
        _pending = max(0, _pending - 1)
        _inflight += 1


def _note_end() -> None:
    global _inflight, _last_active
    with _lock:
        _inflight = max(0, _inflight - 1)
        _last_active = time.monotonic()


def _note_untracked_end() -> None:
    # Drop a request that never used the local GGUF without stamping activity, so periodic
    # external-provider traffic can't keep the model warm.
    global _inflight
    with _lock:
        _inflight = max(0, _inflight - 1)


def _is_idle(ttl_seconds: float) -> bool:
    with _lock:
        return _inflight == 0 and _pending == 0 and (time.monotonic() - _last_active) >= ttl_seconds


def _note_activity() -> None:
    """Stamp activity, e.g. on a (re)load, so the model survives at least one TTL."""
    global _last_active
    with _lock:
        _last_active = time.monotonic()


def other_inference_request_count(
    current_request_counted: bool = True, *, include_pending: bool = True
) -> int:
    """Tracked inference requests other than the current route call.

    The middleware counts before route code runs, so the caller is excluded by default.
    Idle-unload counts pending waiters too. The swap guard passes include_pending=False: a pending
    request is blocked in the middleware and hasn't started inference, so a swap can't interrupt it.
    """
    with _lock:
        active = _inflight
        if current_request_counted and active > 0:
            active -= 1
        return max(0, active) + (_pending if include_pending else 0)


# Set on the ASGI scope by a route that proved this request won't touch llama.cpp, so the
# keep-warm count excludes it and the middleware skips its end-decrement.
_UNTRACKED_SCOPE_KEY = "_unsloth_keepwarm_untracked"


def untrack_current_request(scope) -> None:
    """Drop this request from the in-flight count once the route knows it won't use the local GGUF,
    so external-provider traffic can't trip the swap busy guard. Idempotent."""
    if not isinstance(scope, dict) or scope.get(_UNTRACKED_SCOPE_KEY):
        return
    scope[_UNTRACKED_SCOPE_KEY] = True
    _note_untracked_end()


def inference_lifecycle_gate():
    """The gate a model swap holds so new inference can't start mid-load. Process-wide, so a swap
    on one loop blocks inference on any other loop."""
    return _unload_gate()


def note_model_loaded() -> None:
    """Record a successful GGUF load: stamp activity and drop any reload stash (a manual load
    clears it synchronously, not only on the next idle poll)."""
    _note_activity()
    _set_last_unloaded(None)


def note_model_unloaded() -> None:
    """Record a deliberate (user/API) unload: drop the idle reload stash so the next request can't
    resurrect the just-unloaded model. Unlike the idle loop it means "stay unloaded", so it must
    not stamp activity."""
    _set_last_unloaded(None)


def get_last_unloaded_model():
    with _lock:
        return _last_unloaded_model


def _set_last_unloaded(value) -> None:
    global _last_unloaded_model
    with _lock:
        _last_unloaded_model = value


class LlamaKeepWarmMiddleware:
    """Pure ASGI: count in-flight inference requests and stamp activity on completion."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Inference endpoints are all POST; skipping non-POST avoids counting CORS preflight.
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not _is_inference_path(scope.get("path") or "")
        ):
            await self.app(scope, receive, send)
            return
        # Always track in-flight, even when the feature is off, so a stream started before
        # idle-unload is enabled can't be unloaded mid-response if the operator turns it on.
        # Mark pending before the gate so the idle loop can't free the model while this waits.
        _note_pending()
        started = False
        try:
            async with _unload_gate():
                _note_start()
                started = True
        finally:
            if not started:
                _note_unpending()
        ended = {"done": False}
        status = {"code": None}

        def _finish() -> None:
            # A route that untracked itself already decremented; don't double-count.
            if ended["done"]:
                return
            ended["done"] = True
            if scope.get(_UNTRACKED_SCOPE_KEY):
                return
            # This runs before FastAPI auth, so a 401/403 reaches here without touching llama.cpp.
            # Decrement (to balance _note_start) but do NOT stamp activity, else repeated
            # unauthenticated probes would keep the model warm forever.
            if status["code"] in (401, 403):
                _note_untracked_end()
            else:
                _note_end()

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status["code"] = message.get("status")
            # Final body frame marks the end of a (possibly streaming) response.
            elif message.get("type") == "http.response.body" and not message.get(
                "more_body", False
            ):
                _finish()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _finish()


def _loaded_identity(backend):
    if not backend.is_loaded or not backend.model_identifier:
        return None
    # Third slot is the advertised id (repo id) an auto-switch load sets; it's the override key, so
    # an idle stash keyed by the concrete load path keeps the user's saved launch flags on reload.
    advertised = getattr(backend, "_openai_advertised_id", None) or backend.model_identifier
    return (backend.model_identifier, getattr(backend, "hf_variant", None), advertised)


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
            # Track by (id, variant): a (re)loaded model counts as activity so it survives one TTL
            # before its first request (loads bypass the activity middleware).
            current = _loaded_identity(backend)
            if current != seen_model:
                seen_model = current
                if current is not None:
                    _note_activity()
                    _set_last_unloaded(None)  # a model is loaded; drop stale stash
            async with _unload_gate():
                if backend.is_loaded and _is_idle(ttl):
                    freed = _loaded_identity(backend)
                    await asyncio.to_thread(backend.unload_model)
                    _set_last_unloaded(freed)  # let an alias request reload it
                    logger.info("Idle auto-unload: freed GGUF after %ss idle", ttl)
                    seen_model = None
        except Exception as exc:
            logger.debug("idle_unload_loop iteration failed: %s", exc)
