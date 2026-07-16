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
import contextvars
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
# Non-zero while a preview swap is loading a checkpoint (from before it takes the
# lifecycle gate until after it releases it). The generation counter alone can't reject
# a request that captures it AFTER the swap bumped it but BEFORE the gate releases, so
# the middleware also captures this flag at entry: if a swap was in progress when the
# request arrived, reject it rather than let it run against the swapped-in checkpoint.
_preview_swap_inflight = 0
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


def note_preview_swap_begin() -> None:
    """Mark a preview swap in progress. Call before taking the lifecycle gate to load,
    and pair with note_preview_swap_end() after the gate is released, so a non-preview
    request that arrives at any point during the swap (including the window after the
    generation counter is bumped but before the gate releases) is rejected."""
    global _preview_swap_inflight
    with _lock:
        _preview_swap_inflight += 1


def note_preview_swap_end() -> None:
    global _preview_swap_inflight
    with _lock:
        _preview_swap_inflight = max(0, _preview_swap_inflight - 1)


def _preview_swap_active() -> bool:
    with _lock:
        return _preview_swap_inflight > 0


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

# Set by the middleware on a non-preview request's scope when a preview swap advanced
# the swap counter while the request waited on the lifecycle gate. The local inference
# path (_maybe_auto_switch_model) rejects such a request rather than silently serving
# it the preview's swapped-in checkpoint. Deferred to the route -- not a middleware
# 503 -- so an external-provider request, which untracks itself and returns before
# reaching that check, is never rejected for a swap it never touches.
_PREVIEW_SWAP_REJECT_SCOPE_KEY = "_unsloth_keepwarm_preview_swap_reject"

# Set on the ASGI scope by a streaming route that encoded a failure after its 200
# headers were already sent (e.g. generate_stream yields an SSE error chunk, or a
# llama-server passthrough relays a mid-stream error while the HTTP status stays 200).
# The middleware's successful-response claim keys off the HTTP status alone, so without
# this a failed stream would still adopt a preview-owned model for Studio and 503 later
# previews for another checkpoint; the claim skips a response flagged here.
_RESPONSE_FAILED_SCOPE_KEY = "_unsloth_keepwarm_response_failed"


def mark_response_failed(scope) -> None:
    """Flag a response that returned 2xx headers but then failed, so the keep-warm
    middleware does not treat it as a successful non-preview completion and claim the
    slot for Studio. Safe to call more than once; a no-op on a non-dict scope."""
    if isinstance(scope, dict):
        scope[_RESPONSE_FAILED_SCOPE_KEY] = True


# The current request's ASGI scope, set by the middleware so deep streaming error
# helpers can flag a failure without threading the scope through every yield site.
# The middleware runs in the same task as the (same-task) streaming body, so the
# contextvar propagates to the generators that emit the error SSE.
_current_response_scope: contextvars.ContextVar = contextvars.ContextVar(
    "_unsloth_current_response_scope", default=None
)


def set_current_response_scope(scope) -> None:
    _current_response_scope.set(scope if isinstance(scope, dict) else None)


def mark_current_response_failed() -> None:
    """Flag the current request's response as failed via the contextvar the middleware
    set, so an OpenAI-family streaming error emitted deep in a generator (which has no
    direct scope handle) still prevents the successful-response slot claim."""
    mark_response_failed(_current_response_scope.get())


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
        # Expose this request's scope to deep streaming error helpers (same task, so
        # the contextvar propagates to the response body generators), so an SSE error
        # emitted after the 200 headers can flag the response failed.
        set_current_response_scope(scope)
        _note_pending(is_preview)
        # Capture the preview-swap state before waiting on the gate: if a preview swap
        # completes while this non-preview request is blocked, the counter advances; if
        # one is already in progress when the request arrives (the counter may have
        # already been bumped but the gate not yet released), the in-progress flag is
        # set. Either way the request must be rejected (see below).
        swap_gen_at_entry = _preview_swap_gen()
        swap_active_at_entry = _preview_swap_active()
        started = False
        try:
            async with _unload_gate():
                _note_start(is_preview)
                started = True
                # A preview swapped in a different checkpoint while this non-preview
                # request waited on the gate. Flag the scope so the local inference
                # path (_maybe_auto_switch_model) rejects it before running against the
                # preview's model. Deferred to the route, not a 503 here, so an
                # external-provider request -- which untracks itself and returns before
                # reaching that check -- is not rejected for a swap it never touches.
                if (
                    not is_preview
                    and (_preview_swap_gen() != swap_gen_at_entry or swap_active_at_entry)
                    and isinstance(scope, dict)
                ):
                    scope[_PREVIEW_SWAP_REJECT_SCOPE_KEY] = True
        finally:
            if not started:
                _note_unpending(is_preview)
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
            # A preview that did not return 2xx never served tokens from the local
            # model: rate-limit 429, bad capability token 404 and body-validation 4xx
            # all exit before load_model_for_preview. Drop it like an untracked end so
            # repeated rejected public preview POSTs can't refresh the idle timer and
            # pin an otherwise idle model in VRAM (a loaded-then-failed preview already
            # stamped activity at load time, so not stamping here is harmless).
            if is_preview and not (isinstance(code, int) and 200 <= code < 300):
                _note_untracked_end(is_preview)
                return
            # A non-preview inference that actually ran (2xx) against the local model
            # adopts it for Studio, so clear preview ownership -- claim on success, so a
            # request a per-route capability check rejected (4xx/5xx) never strands a
            # preview-owned model. count_tokens only tokenizes (no generation), so it
            # does not claim. Claim BEFORE decrementing in-flight: doing it after opens
            # a window where a preview for another checkpoint sees no non-preview
            # traffic and a still-preview-owned slot, swaps its model in, and then this
            # delayed claim clears ownership on the wrong model. While this request is
            # still counted in-flight the preview busy guard refuses that swap.
            if (
                not is_preview
                and isinstance(code, int)
                and 200 <= code < 300
                and not path.endswith("/messages/count_tokens")
                and not scope.get(_RESPONSE_FAILED_SCOPE_KEY)
            ):
                _claim_non_preview_slot()
            _note_end(is_preview)

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
