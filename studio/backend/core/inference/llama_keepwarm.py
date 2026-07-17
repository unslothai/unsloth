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
from pathlib import Path

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
# Slot KV manifest saved by the idle unload; whoever pops it owns deleting its files.
_kv_resume = None
# Guards inflight bumps against the idle-check-then-unload race, and blocks new
# inference from starting mid-swap. Process-wide, not per-loop: the backend slot is
# shared across every event loop in the process, so a per-loop gate would let a
# request on loop B start inference while a swap on loop A tears the model down.
_lifecycle_lock = threading.Lock()


@contextlib.asynccontextmanager
async def _unload_gate():
    # Acquire off the loop: non-blocking first (the common uncontended case), else
    # poll a non-blocking acquire off a short sleep. Polling keeps the wait off this
    # loop AND cancellation-safe -- a cancel lands during the sleep, when the gate is
    # not held, so it never leaks (mirrors the auto-switch swap gate).
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
    "/messages/count_tokens",  # counts via the loaded tokenizer; protect like /messages
    "/embeddings",
    "/responses",
    "/generate/stream",  # Studio's own streaming route on the same llama-server
    "/audio/generate",  # direct GGUF TTS; can outlive the idle TTL
)


def _is_inference_path(path: str) -> bool:
    if path.startswith(_INFERENCE_PREFIXES) and path.endswith(_INFERENCE_SUFFIXES):
        return True
    # Public checkpoint preview (/p/{run}/v1/chat/completions) delegates to the
    # chat handler and streams from the same backend, so protect it from idle unload.
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
    # Do not stamp _last_active here: while _inflight > 0 the model is already
    # protected (see _is_idle), and stamping on start lets an external-provider
    # request that is later untracked still reset the local idle timer.
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
    # Drop a request that never used the local GGUF without stamping local
    # activity, so periodic external-provider traffic can't keep the model warm.
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

    The middleware counts OpenAI-compatible requests before route code runs, so
    the caller is excluded by default. Idle-unload counts pending waiters too (a
    swap holding the gate would unload out from under them). The swap guard passes
    include_pending=False: a pending request is blocked in the middleware and has
    not started inference, so it can't be the request a swap would interrupt.
    """
    with _lock:
        active = _inflight
        if current_request_counted and active > 0:
            active -= 1
        return max(0, active) + (_pending if include_pending else 0)


# Set on the ASGI scope by a route that proved this request won't touch
# llama.cpp (e.g. it proxied to an external provider), so the keep-warm count
# excludes it and the middleware skips its own end-decrement.
_UNTRACKED_SCOPE_KEY = "_unsloth_keepwarm_untracked"


def untrack_current_request(scope) -> None:
    """Drop this request from the in-flight count once the route knows it won't
    use the local GGUF, so unrelated external-provider traffic can't trip the
    swap busy guard. Idempotent; the middleware then skips its end-decrement."""
    if not isinstance(scope, dict) or scope.get(_UNTRACKED_SCOPE_KEY):
        return
    scope[_UNTRACKED_SCOPE_KEY] = True
    _note_untracked_end()


def inference_lifecycle_gate():
    """The gate a model swap holds so new inference can't start mid-load. Process-
    wide, so a swap on one loop blocks inference starting on any other loop."""
    return _unload_gate()


def note_model_loaded(backend = None) -> None:
    """Record a successful load: stamp activity and drop any reload stash so a
    manual load clears it synchronously, not only on the next idle poll."""
    _note_activity()
    resume = take_kv_resume()
    _set_last_unloaded(None)
    if resume is None:
        return
    if backend is not None:
        restore_kv_resume(backend, resume)
    else:
        _delete_resume_files(resume)


def note_model_unloaded() -> None:
    """Record a deliberate (user/API) unload: drop any idle reload stash so the next
    request can't resurrect the just-unloaded model. The idle loop unloads via the
    backend directly and then stashes the freed model for an alias reload; an
    explicit unload instead means "stay unloaded", so it must not stamp activity."""
    _set_last_unloaded(None)


def get_last_unloaded_model():
    with _lock:
        return _last_unloaded_model


def _set_last_unloaded(value) -> None:
    global _last_unloaded_model, _kv_resume
    stale = None
    with _lock:
        _last_unloaded_model = value
        if value is None and _kv_resume is not None:
            stale, _kv_resume = _kv_resume, None
    if stale:
        _delete_resume_files(stale)


def _delete_resume_files(manifest) -> None:
    try:
        base = Path(manifest.get("dir") or "")
        for entry in manifest.get("slots") or []:
            with contextlib.suppress(OSError):
                (base / str(entry.get("filename"))).unlink()
    except Exception:
        pass


def _set_kv_resume(value) -> None:
    global _kv_resume
    stale = None
    with _lock:
        if _kv_resume is not None and _kv_resume is not value:
            stale = _kv_resume
        _kv_resume = value
    if stale:
        _delete_resume_files(stale)


def take_kv_resume():
    global _kv_resume
    with _lock:
        manifest, _kv_resume = _kv_resume, None
        return manifest


def restore_kv_resume(backend, manifest) -> None:
    try:
        gguf = manifest.get("gguf")
        binary = manifest.get("binary")
        current = getattr(backend, "_gguf_path", None)
        same_gguf = bool(gguf and current) and Path(current).resolve() == Path(gguf).resolve()
        if same_gguf and binary and binary == getattr(backend, "_slot_save_binary", None):
            logger.info("Restoring saved slot KV onto the reloaded model")
            backend.restore_slots_for_resume(manifest)
    except Exception as exc:
        logger.debug("slot restore after reload failed: %s", exc)
    finally:
        _delete_resume_files(manifest)


def sweep_slot_save_dir() -> None:
    try:
        from utils.paths.storage_roots import llama_slot_cache_root
        for path in llama_slot_cache_root().glob("resume-*.bin"):
            with contextlib.suppress(OSError):
                path.unlink()
    except Exception:
        pass


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
        status = {"code": None}

        def _finish() -> None:
            # A route that untracked itself already decremented; don't double-count.
            if ended["done"]:
                return
            ended["done"] = True
            if scope.get(_UNTRACKED_SCOPE_KEY):
                return
            # This middleware runs before FastAPI auth, so a 401/403 reaches here
            # without ever touching llama.cpp. Decrement the in-flight count (to
            # balance _note_start) but do NOT stamp activity, or repeated
            # unauthenticated probes on an exposed server would keep the model warm
            # and never let idle-unload free VRAM.
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
    # Third slot is the advertised id (repo id) an auto-switch load sets on the
    # backend; it's the override key, so an idle stash keyed by the concrete load
    # path doesn't drop the user's saved launch flags on the alias reload.
    advertised = getattr(backend, "_openai_advertised_id", None) or backend.model_identifier
    return (backend.model_identifier, getattr(backend, "hf_variant", None), advertised)


async def idle_unload_loop(poll_seconds: float = 15.0) -> None:
    """Unload the loaded GGUF once idle past the configured TTL. Inert when off."""
    from utils.openai_auto_switch_settings import (
        get_auto_unload_idle_seconds,
        get_auto_unload_keep_kv,
    )

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
                    manifest = None
                    if get_auto_unload_keep_kv():
                        try:
                            manifest = await asyncio.to_thread(backend.save_slots_for_resume)
                        except Exception as exc:
                            logger.debug("slot save before idle unload failed: %s", exc)
                    if not _is_idle(ttl):
                        if manifest:
                            _delete_resume_files(manifest)
                        continue
                    await asyncio.to_thread(backend.unload_model)
                    _set_last_unloaded(freed)  # let an alias request reload it
                    if manifest and freed:
                        _set_kv_resume({"identity": freed, **manifest})
                        logger.info("Idle auto-unload: saved slot KV for restore on reload")
                    logger.info("Idle auto-unload: freed GGUF after %ss idle", ttl)
                    seen_model = None
        except Exception as exc:
            logger.debug("idle_unload_loop iteration failed: %s", exc)
