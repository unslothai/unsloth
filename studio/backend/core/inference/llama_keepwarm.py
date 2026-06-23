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
import weakref

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_inflight = 0
# Requests blocked on the unload gate but not yet counted in _inflight: the idle
# loop must not unload while one is waiting (it would unload out from under it).
_pending = 0
_last_active = time.monotonic()
# The (id, quant) idle-unload last freed, so an alias/unknown request that would
# otherwise 503 against an empty backend can reload it (set on unload, cleared on
# reload). Storing the quant means the reload restores the exact freed variant.
_last_unloaded_model = None
# Guards inflight bumps against the idle-check-then-unload race. One lock per
# running loop: a module-level asyncio.Lock binds to one loop and breaks
# multi-loop runners (e.g. pytest's per-test loops on pre-3.10).
_unload_gates: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _unload_gate() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    # WeakKeyDictionary mutation isn't thread-safe; guard get-or-create so two
    # loops on different threads can't race it. (_lock is released here before
    # the returned asyncio lock is awaited, so there is no nesting.)
    with _lock:
        gate = _unload_gates.get(loop)
        if gate is None:
            gate = _unload_gates[loop] = asyncio.Lock()
        return gate


_INFERENCE_PREFIXES = ("/v1/", "/api/inference/")
_INFERENCE_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/messages",
    "/messages/count_tokens",  # counts via the loaded tokenizer; protect like /messages
    "/embeddings",
    "/responses",
    "/generate/stream",  # Studio's own streaming route on the same llama-server
    "/audio/generate",  # direct GGUF TTS; can outlive the idle TTL
)


def _is_inference_path(path: str) -> bool:
    return path.startswith(_INFERENCE_PREFIXES) and path.endswith(_INFERENCE_SUFFIXES)


def _note_pending() -> None:
    global _pending
    with _lock:
        _pending += 1


def _note_unpending() -> None:
    global _pending
    with _lock:
        _pending = max(0, _pending - 1)


def _note_start() -> None:
    global _inflight, _pending, _last_active
    with _lock:
        _pending = max(0, _pending - 1)
        _inflight += 1
        _last_active = time.monotonic()


def _note_end() -> None:
    global _inflight, _last_active
    with _lock:
        _inflight = max(0, _inflight - 1)
        _last_active = time.monotonic()


def _is_idle(ttl_seconds: float) -> bool:
    with _lock:
        return _inflight == 0 and _pending == 0 and (time.monotonic() - _last_active) >= ttl_seconds


def _note_activity() -> None:
    """Stamp activity, e.g. on a (re)load, so the model survives at least one TTL."""
    global _last_active
    with _lock:
        _last_active = time.monotonic()


def other_inference_request_count(current_request_counted: bool = True) -> int:
    """Tracked inference requests other than the current route call.

    The middleware counts OpenAI-compatible requests before route code runs, so
    the caller is excluded by default. Pending waiters (blocked on the gate)
    count too: a swap can't assume they're safe since one may want the loaded
    model. The auto-switch guard subtracts requests known to want the same target.
    """
    with _lock:
        active = _inflight
        if current_request_counted and active > 0:
            active -= 1
        return max(0, active) + _pending


def inference_lifecycle_gate() -> asyncio.Lock:
    """The gate a model swap holds so new inference can't start mid-load."""
    return _unload_gate()


def note_model_loaded() -> None:
    """Record a successful GGUF load: stamp activity and drop any reload stash so
    a manual load clears it synchronously, not only on the next idle poll."""
    _note_activity()
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
        # Inference endpoints are all POST; skipping non-POST avoids counting CORS
        # preflight (OPTIONS). ``or ""`` guards an explicit None path.
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not _is_inference_path(scope.get("path") or "")
        ):
            await self.app(scope, receive, send)
            return
        # Always track in-flight on inference paths, even when the feature is off,
        # so a stream that starts before idle-unload is enabled can't be unloaded
        # mid-response if the operator turns it on during that stream. Counting is
        # cheap and invisible to clients (the response is proxied unchanged).
        # Mark pending before the gate so the idle loop (which holds the gate while
        # unloading) can't free the model while this request is waiting to start.
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


def _loaded_identity(backend):
    if not backend.is_loaded or not backend.model_identifier:
        return None
    # Third slot is the advertised id (repo id) an auto-switch load sets on the
    # backend; it's the override key, so an idle stash keyed by the concrete load
    # path doesn't drop the user's saved launch flags on the alias reload.
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
            # Track by (id, variant): a (re)loaded model -- including the same repo
            # at a different quant -- counts as activity so it survives one TTL
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
