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
import json
import threading
import time

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_inflight = 0
# Subset of _inflight that is /p/ preview traffic; same update sites as _inflight
# so it can't drift out of sync with it.
_preview_inflight = 0
# Requests blocked on the unload gate but not yet counted in _inflight: the idle
# loop must not unload while one is waiting (it would unload out from under it).
_pending = 0
# Subset of _pending that is /p/ preview traffic (same update sites as _pending), so
# the preview busy guard can tell a queued Studio request from a queued preview.
_preview_pending = 0
# Bumped once each time a preview swap actually loads a new checkpoint. A non-preview
# request captures this before waiting on the lifecycle gate; if it advanced by the
# time the request acquires the gate, a preview swapped the model out from under it
# while it waited, so the request must be rejected instead of silently running against
# the preview's checkpoint (see the middleware).
_preview_swap_generation = 0
_last_active = time.monotonic()
# The (id, quant) idle-unload last freed, so an alias/unknown request that would
# otherwise 503 against an empty backend can reload it (set on unload, cleared on
# reload). Storing the quant means the reload restores the exact freed variant.
_last_unloaded_model = None
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


def _is_preview_path(path: str) -> bool:
    # Public checkpoint preview (/p/{run}/v1/chat/completions) delegates to the
    # chat handler and streams from the same backend, so protect it from idle unload.
    return path.startswith("/p/") and path.endswith("/v1/chat/completions")


def _is_inference_path(path: str) -> bool:
    if path.startswith(_INFERENCE_PREFIXES) and path.endswith(_INFERENCE_SUFFIXES):
        return True
    return _is_preview_path(path)


def _note_pending(is_preview: bool = False) -> None:
    global _pending, _preview_pending
    with _lock:
        _pending += 1
        if is_preview:
            _preview_pending += 1


def _note_unpending(is_preview: bool = False) -> None:
    global _pending, _preview_pending
    with _lock:
        _pending = max(0, _pending - 1)
        if is_preview:
            _preview_pending = max(0, _preview_pending - 1)


def _note_start(is_preview: bool = False) -> None:
    # Do not stamp _last_active here: while _inflight > 0 the model is already
    # protected (see _is_idle), and stamping on start lets an external-provider
    # request that is later untracked still reset the local idle timer.
    global _inflight, _pending, _preview_inflight, _preview_pending
    with _lock:
        _pending = max(0, _pending - 1)
        _inflight += 1
        if is_preview:
            _preview_pending = max(0, _preview_pending - 1)
            _preview_inflight += 1


def _note_end(is_preview: bool = False) -> None:
    global _inflight, _last_active, _preview_inflight
    with _lock:
        _inflight = max(0, _inflight - 1)
        _last_active = time.monotonic()
        if is_preview:
            _preview_inflight = max(0, _preview_inflight - 1)


def _note_untracked_end(is_preview: bool = False) -> None:
    # Drop a request that never used the local GGUF without stamping local
    # activity, so periodic external-provider traffic can't keep the model warm.
    global _inflight, _preview_inflight
    with _lock:
        _inflight = max(0, _inflight - 1)
        if is_preview:
            _preview_inflight = max(0, _preview_inflight - 1)


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


def other_preview_inflight_count(current_request_counted: bool = True) -> int:
    """Preview (/p/) requests in flight other than the current route call."""
    with _lock:
        active = _preview_inflight
        if current_request_counted and active > 0:
            active -= 1
        return max(0, active)


def other_non_preview_pending_count() -> int:
    """Non-preview requests queued on the lifecycle gate (in _pending, not yet in
    flight). The preview swap guard must count these: a queued Studio request would
    otherwise start against the model a preview swapped in while it waited. The
    current request is a preview already in flight, so it is not in _pending."""
    with _lock:
        return max(0, _pending - _preview_pending)


def note_preview_swap() -> None:
    """Record that a preview swap loaded a new checkpoint. A non-preview request that
    was blocked on the lifecycle gate through the swap sees this counter advance and
    is rejected rather than running against the swapped-in preview checkpoint."""
    global _preview_swap_generation
    with _lock:
        _preview_swap_generation += 1


def _preview_swap_gen() -> int:
    with _lock:
        return _preview_swap_generation


def _claim_non_preview_slot() -> None:
    """A non-preview inference request that actually ran against the local model (a
    2xx response) adopts it for Studio, so clear preview ownership -- a later preview
    for another checkpoint then 503s instead of swapping the model out from under an
    active Studio conversation. Claiming on success (not before) means a request that
    a per-route capability check rejected never strands a preview-owned model. Lazily
    imported: routes.inference imports this module."""
    try:
        from routes.inference import _set_preview_resident

        _set_preview_resident(None)
    except Exception as exc:  # never let ownership bookkeeping break a response
        logger.debug("preview-slot claim on completion failed: %s", exc)


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
    # Keep the preview subset aligned with _inflight: a /p/ request untracking
    # itself must drop from both counters, or the busy guard sees phantom traffic.
    _note_untracked_end(_is_preview_path(scope.get("path") or ""))


def inference_lifecycle_gate():
    """The gate a model swap holds so new inference can't start mid-load. Process-
    wide, so a swap on one loop blocks inference starting on any other loop."""
    return _unload_gate()


def note_model_loaded() -> None:
    """Record a successful GGUF load: stamp activity and drop any reload stash so
    a manual load clears it synchronously, not only on the next idle poll."""
    _note_activity()
    _set_last_unloaded(None)


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
        path = scope.get("path") or ""
        is_preview = _is_preview_path(path)
        _note_pending(is_preview)
        # Capture the preview-swap counter before waiting on the gate: if a preview
        # swap completes while this non-preview request is blocked, the counter
        # advances and the request must be rejected (see below).
        swap_gen_at_entry = _preview_swap_gen()
        started = False
        swapped_while_waiting = False
        try:
            async with _unload_gate():
                # A preview swapped in a different checkpoint while this non-preview
                # request waited on the gate; running it now would silently serve the
                # preview's model to Studio traffic. Reject so the client retries
                # against the now-stable model instead.
                if not is_preview and _preview_swap_gen() != swap_gen_at_entry:
                    swapped_while_waiting = True
                else:
                    _note_start(is_preview)
                    started = True
        finally:
            if not started:
                _note_unpending(is_preview)
        if swapped_while_waiting:
            body = json.dumps(
                {"detail": "A preview is loading a model. Please retry shortly."}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"retry-after", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
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
            code = status["code"]
            if code in (401, 403):
                _note_untracked_end(is_preview)
                return
            _note_end(is_preview)
            # A non-preview inference that actually ran (2xx) against the local model
            # adopts it for Studio, so clear preview ownership -- claim on success, so a
            # request a per-route capability check rejected (4xx/5xx) never strands a
            # preview-owned model. count_tokens only tokenizes (no generation), so it
            # does not claim.
            if (
                not is_preview
                and isinstance(code, int)
                and 200 <= code < 300
                and not path.endswith("/messages/count_tokens")
            ):
                _claim_non_preview_slot()

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
