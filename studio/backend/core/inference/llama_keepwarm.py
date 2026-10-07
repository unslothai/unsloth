# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in idle auto-unload (TTL keep-warm) for the local llama.cpp model.

Off by default (idle seconds = 0). A background loop unloads the GGUF after the TTL, and a
pure-ASGI middleware tracks in-flight requests so a long stream is never unloaded mid-response.
The resident model is shared, so any account's activity resets the one global idle clock. The
same loop and middleware drive media_keepwarm.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from pathlib import Path

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_inflight = 0
_preview_inflight = 0
# Blocked on the unload gate, not yet in _inflight: the idle loop must not unload while one waits.
_pending = 0
_preview_pending = 0
# Counted after FastAPI auth, so unauthenticated tracked requests cannot starve previews.
_admitted_inference = 0
# Bumped per preview swap; a non-preview request seeing it advance across the gate is rejected.
_preview_swap_generation = 0
# Non-zero while a preview swap loads; catches requests captured after the bump but before release.
_preview_swap_inflight = 0
_last_active = time.monotonic()
# (id, quant) the idle unload freed, so an alias request can reload the exact variant.
_last_unloaded_model = None
# Slot KV manifest saved by the idle unload; whoever pops it owns deleting its files.
_kv_resume = None
# Process-wide, not per-loop: the backend slot is shared across every event loop.
_lifecycle_lock = threading.Lock()


_load_lock = threading.Lock()


@contextlib.asynccontextmanager
async def _unload_gate(
    cancel_event: threading.Event | None = None, lock: threading.Lock = _lifecycle_lock
):
    # Poll a non-blocking acquire so the wait stays off the loop and a cancel never leaks the gate.
    acquired = False
    try:
        while not lock.acquire(blocking = False):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError()
            await asyncio.sleep(0.02)
        acquired = True
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError()
        yield
    finally:
        if acquired:
            lock.release()


_INFERENCE_PREFIXES = ("/v1/", "/api/inference/")
_INFERENCE_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/messages",
    "/messages/count_tokens",
    "/chat/count_tokens",
    "/embeddings",
    "/responses",
    "/generate/stream",
    "/audio/generate",  # can outlive the idle TTL
    "/audio/speech",
    # endswith avoids matching *-progress / */cancel.
    "/images/generate",
    "/images/generations",
    "/video/generate",
)

# Matched whole: a suffix "/videos" would let 404 probes keep the chat model warm.
_INFERENCE_EXACT_PATHS = frozenset({"/v1/videos", "/api/inference/videos"})

# Served by diffusion/video engines, never the llama slot: must not clear preview ownership.
_NON_LLM_SLOT_SUFFIXES = (
    "/images/generate",
    "/images/generations",
    "/video/generate",
    "/videos",
)


def _is_preview_path(path: str) -> bool:
    return path.startswith("/p/") and path.endswith("/v1/chat/completions")


def _is_inference_path(path: str) -> bool:
    if path in _INFERENCE_EXACT_PATHS:
        return True
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
    # Do not stamp _last_active: while _inflight > 0 the model is already protected.
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


class InferenceActivityReservation:
    """Keep a background inference job visible to lifecycle and idle-unload guards."""

    def __init__(self) -> None:
        self._state = "new"
        self._state_lock = threading.Lock()

    def reserve(self) -> None:
        """Publish pending work synchronously, before its asyncio task can run."""
        with self._state_lock:
            if self._state != "new":
                return
            _note_pending()
            self._state = "pending"

    async def start(self, cancel_event: threading.Event | None = None) -> None:
        """Claim the inference slot after any model lifecycle operation completes."""
        with self._state_lock:
            if self._state != "pending":
                return
        async with _unload_gate(cancel_event):
            with self._state_lock:
                if self._state != "pending":
                    return
                _note_start()
                self._state = "started"

    def finish(self) -> None:
        """Balance a pending or started reservation. Idempotent."""
        with self._state_lock:
            if self._state == "pending":
                _note_unpending()
            elif self._state == "started":
                _note_end()
            else:
                return
            self._state = "finished"


def _note_untracked_end(is_preview: bool = False) -> None:
    # No activity stamp, so external-provider traffic cannot keep the model warm.
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

    The middleware counts requests before route code runs, so the caller is excluded by
    default. Idle-unload counts pending waiters too (a swap holding the gate would unload
    out from under them); the swap guard passes include_pending=False since a pending
    request is blocked in the middleware and can't be the one a swap would interrupt.
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


def other_admitted_inference_count() -> int:
    """Non-preview requests admitted to local inference (passed auth, reached the
    _maybe_auto_switch_model / generate_stream choke point). The preview busy guard counts
    these instead of raw _inflight, so a pre-auth/unauthenticated tracked request can't
    block a preview. The current request is always a preview (never admitted), so no
    self-exclusion is needed."""
    with _lock:
        return _admitted_inference


def other_non_preview_pending_count() -> int:
    """Non-preview requests queued on the lifecycle gate (_pending, not yet in flight).
    The preview swap guard must count these: a queued Unsloth request would otherwise start
    against the model a preview swapped in while it waited. The current request is a preview
    already in flight, so not in _pending."""
    with _lock:
        return max(0, _pending - _preview_pending)


def note_preview_swap() -> None:
    """Record that a preview swap loaded a new checkpoint. A non-preview request blocked on
    the lifecycle gate through the swap sees this counter advance and is rejected rather than
    running against the swapped-in preview checkpoint."""
    global _preview_swap_generation
    with _lock:
        _preview_swap_generation += 1


def _preview_swap_gen() -> int:
    with _lock:
        return _preview_swap_generation


def note_preview_swap_begin() -> None:
    """Mark a preview swap in progress. Call before taking the lifecycle gate to load, pair
    with note_preview_swap_end() after the gate releases, so a non-preview request arriving
    at any point during the swap (including after the counter bumps but before the gate
    releases) is rejected."""
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


def preview_swapped_since_entry(scope) -> bool:
    """True if a preview swap ran, or is running, since this request entered the middleware.
    Extends the gate-wait reject flag to catch a non-preview request that passed the gate
    BEFORE a swap (so it never set _PREVIEW_SWAP_REJECT_SCOPE_KEY) but is still pre-admission
    when a preview swaps the model out from under it. entry_gen is None only when the
    middleware never snapshotted it (non-dict/non-inference scope), so fall back to the
    swap-in-progress flag alone."""
    if not isinstance(scope, dict):
        return False
    # A preview may swap the model in itself, so it must never reject itself.
    if _is_preview_path(scope.get("path") or ""):
        return False
    if scope.get(_PREVIEW_SWAP_REJECT_SCOPE_KEY):
        return True
    entry_gen = scope.get(_SWAP_GEN_AT_ENTRY_KEY)
    with _lock:
        if _preview_swap_inflight > 0:
            return True
        return entry_gen is not None and _preview_swap_generation != entry_gen


def _claim_non_preview_slot() -> None:
    """A non-preview request that ran against the local model (2xx) adopts it for Unsloth,
    so clear preview ownership -- a later preview for another checkpoint then 503s instead
    of swapping the model out from under an active Unsloth conversation. Claiming on success
    (not before) means a per-route-rejected request never strands a preview-owned model.
    Lazily imported: routes.inference imports this module."""
    try:
        from routes.inference import _set_preview_resident
        _set_preview_resident(None)
    except Exception as exc:
        logger.debug("preview-slot claim on completion failed: %s", exc)


_UNTRACKED_SCOPE_KEY = "_unsloth_keepwarm_untracked"

_TRACKED_SCOPE_KEY = "_unsloth_keepwarm_tracked"

# While queued a preview must be pending, not active, or it deadlocks on the lifecycle gate.
_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY = "_unsloth_keepwarm_preview_serializer_wait"

# Rejected in the route, not middleware, so untracked external-provider requests pass.
_PREVIEW_SWAP_REJECT_SCOPE_KEY = "_unsloth_keepwarm_preview_swap_reject"

_SWAP_GEN_AT_ENTRY_KEY = "_unsloth_keepwarm_swap_gen_at_entry"

# Claim keys off HTTP status, so a stream failing after 200 headers must flag this.
_RESPONSE_FAILED_SCOPE_KEY = "_unsloth_keepwarm_response_failed"


def mark_response_failed(scope) -> None:
    """Flag a response that returned 2xx headers but then failed, so the middleware doesn't
    treat it as a successful non-preview completion and claim the slot for Unsloth. Safe to
    call repeatedly; a no-op on a non-dict scope."""
    if isinstance(scope, dict):
        scope[_RESPONSE_FAILED_SCOPE_KEY] = True


# Middleware shares the streaming body's task, so the contextvar reaches those generators.
_current_response_scope: contextvars.ContextVar = contextvars.ContextVar(
    "_unsloth_current_response_scope", default = None
)


def set_current_response_scope(scope) -> None:
    _current_response_scope.set(scope if isinstance(scope, dict) else None)


def mark_current_response_failed() -> None:
    """Flag the current response failed via the contextvar the middleware set, so an
    OpenAI-family streaming error emitted deep in a generator (no direct scope handle)
    still prevents the successful-response slot claim."""
    mark_response_failed(_current_response_scope.get())


def untrack_current_request(scope) -> None:
    """Drop this request from the in-flight count once the route knows it won't use the
    local GGUF, so external-provider traffic can't trip the swap busy guard. Idempotent;
    the middleware then skips its end-decrement."""
    if not isinstance(scope, dict) or scope.get(_UNTRACKED_SCOPE_KEY):
        return
    scope[_UNTRACKED_SCOPE_KEY] = True
    # Drop from both counters, or the busy guard sees phantom traffic.
    _note_untracked_end(_is_preview_path(scope.get("path") or ""))


_ADMITTED_SCOPE_KEY = "_unsloth_keepwarm_admitted"


def note_admitted_inference(scope) -> None:
    """Mark a non-preview request as admitted local inference (passed auth, reached the
    _maybe_auto_switch_model / generate_stream choke point), so the preview busy guard
    counts it. Idempotent per scope; a no-op for preview (/p/) paths (own ownership) and
    non-dict scopes."""
    global _admitted_inference
    if not isinstance(scope, dict) or scope.get(_ADMITTED_SCOPE_KEY):
        return
    if _is_preview_path(scope.get("path") or ""):
        return
    scope[_ADMITTED_SCOPE_KEY] = True
    with _lock:
        _admitted_inference += 1


def _note_admitted_end() -> None:
    global _admitted_inference
    with _lock:
        _admitted_inference = max(0, _admitted_inference - 1)


def untrack_admitted_inference(scope) -> None:
    """Drop an already-admitted request from the preview busy guard once the route knows it
    will not run against the resident GGUF after all.

    ``untrack_current_request`` covers only the in-flight counters; the admitted tally is
    what ``load_model_for_preview`` reads, so a route that admitted at the auto-switch hook
    and then served the request some other way keeps blocking preview swaps for its whole
    duration. Pops the marker so the middleware's finally, which balances only a scope that
    still carries it, cannot decrement a second time. Idempotent."""
    if not isinstance(scope, dict) or not scope.pop(_ADMITTED_SCOPE_KEY, False):
        return
    _note_admitted_end()


def begin_preview_serializer_wait(scope) -> bool:
    """Move a tracked preview from active to pending while it waits on the route lock."""
    global _inflight, _pending, _preview_inflight, _preview_pending
    if (
        not isinstance(scope, dict)
        or not _is_preview_path(scope.get("path") or "")
        or not scope.get(_TRACKED_SCOPE_KEY)
        or scope.get(_UNTRACKED_SCOPE_KEY)
        or scope.get(_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY)
    ):
        return False
    with _lock:
        scope[_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY] = True
        _inflight = max(0, _inflight - 1)
        _preview_inflight = max(0, _preview_inflight - 1)
        _pending += 1
        _preview_pending += 1
    return True


async def resume_preview_after_serializer(scope) -> None:
    """Re-admit a serialized preview under the lifecycle gate before it touches the model."""
    if not isinstance(scope, dict) or not scope.get(_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY):
        return
    try:
        async with _unload_gate():
            if not scope.get(_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY):
                return
            _note_start(is_preview = True)
            scope.pop(_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY, None)
    except BaseException:
        cancel_preview_serializer_wait(scope)
        raise


def cancel_preview_serializer_wait(scope) -> None:
    """Balance a preview cancelled before it can be re-admitted after serialization."""
    if not isinstance(scope, dict) or not scope.get(_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY):
        return
    scope.pop(_PREVIEW_SERIALIZER_WAIT_SCOPE_KEY, None)
    _note_unpending(is_preview = True)
    # Skip the normal decrement, or this stamps activity for a preview that never ran.
    scope[_UNTRACKED_SCOPE_KEY] = True


def inference_lifecycle_gate():
    """The gate a model swap holds so new inference can't start mid-load. Process-
    wide, so a swap on one loop blocks inference starting on any other loop."""
    return _unload_gate()


def model_load_gate():
    return _unload_gate(lock = _load_lock)


def note_model_loaded(backend = None) -> None:
    """Stamp activity and synchronously drop any reload stash."""
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
    """Record a deliberate (user/API) unload: drop any idle reload stash so the next request
    can't resurrect the just-unloaded model. Unlike the idle loop (which stashes the freed
    model for an alias reload), an explicit unload means "stay unloaded", so it must not
    stamp activity."""
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


def purge_kv_resume() -> None:
    resume = take_kv_resume()
    if resume:
        _delete_resume_files(resume)


def restore_kv_resume(backend, manifest) -> None:
    try:
        gguf = manifest.get("gguf")
        binary = manifest.get("binary")
        current = getattr(backend, "_gguf_path", None)
        same_gguf = bool(gguf and current) and Path(current).resolve() == Path(gguf).resolve()
        if same_gguf:
            # Same path is not enough: shards may have been rewritten meanwhile.
            identity = getattr(backend, "_gguf_file_identity", None)
            same_gguf = callable(identity) and identity(current) == manifest.get("gguf_stat")
        if same_gguf:
            # Nor the same file: launch overrides can invalidate KV numerics.
            fingerprint = getattr(backend, "_slot_launch_fingerprint", None)
            same_gguf = callable(fingerprint) and manifest.get("launch") == fingerprint()
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


def _as_bytes(value) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode("utf-8", "replace")


def _carries_bearer_credentials(scope, path: str = "") -> bool:
    """Whether this request carries the credentials its route demands.

    Every tracked media route depends on ``get_current_subject`` (HTTPBearer), so a request without
    one is refused before any handler runs. Counting it anyway would still pin the pipeline: the
    count is taken here, ahead of FastAPI parsing the body, and a client that opens the POST and
    then withholds its body produces no response status either, so the 401/403 exclusion below never
    gets to run -- one such connection, replaced as it times out, would keep a multi-GB pipeline
    resident for good. Real clients always send the header. Keyless API access is the one case where
    a route demands no bearer at all; its outer admission middleware records that decision before
    keep-warm runs, so reuse the snapshot instead of repeating settings, listener and DNS work on
    this loop.
    """
    from utils.keyless_api_access import KEYLESS_ADMISSION_STATE_KEY

    state = scope.get("state")
    if path and isinstance(state, dict) and state.get(KEYLESS_ADMISSION_STATE_KEY) is True:
        return True
    headers = scope.get("headers")
    if headers is None:
        return True
    for name, value in headers:
        if _as_bytes(name).lower() != b"authorization":
            continue
        scheme, _, token = _as_bytes(value).partition(b" ")
        return scheme.lower() == b"bearer" and bool(token.strip())
    return False


class LlamaKeepWarmMiddleware:
    """Pure ASGI: count in-flight inference requests and stamp activity on completion."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Skip non-POST so CORS preflight (OPTIONS) is not counted.
        path = scope.get("path") or ""
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        # Media load routes are tracked only here: they must not stamp chat activity.
        from core.inference import media_keepwarm

        media_owner = media_keepwarm.owner_for_path(path)
        if media_owner is not None and not _carries_bearer_credentials(scope, path):
            media_owner = None
        chat_tracked = _is_inference_path(path)
        if not chat_tracked and media_owner is None:
            await self.app(scope, receive, send)
            return
        # Track even when the feature is off, and mark pending before the gate so idle unload waits.
        is_preview = _is_preview_path(path)
        if chat_tracked:
            set_current_response_scope(scope)
            _note_pending(is_preview)
            swap_gen_at_entry = _preview_swap_gen()
            swap_active_at_entry = _preview_swap_active()
            if isinstance(scope, dict):
                scope[_SWAP_GEN_AT_ENTRY_KEY] = swap_gen_at_entry
            started = False
            try:
                async with _unload_gate():
                    _note_start(is_preview)
                    started = True
                    if isinstance(scope, dict):
                        scope[_TRACKED_SCOPE_KEY] = True
                    if (
                        not is_preview
                        and (_preview_swap_gen() != swap_gen_at_entry or swap_active_at_entry)
                        and isinstance(scope, dict)
                    ):
                        scope[_PREVIEW_SWAP_REJECT_SCOPE_KEY] = True
            finally:
                if not started:
                    _note_unpending(is_preview)
        if media_owner is not None:
            try:
                await media_keepwarm.begin_request(media_owner)
            except BaseException:
                if chat_tracked:
                    _note_untracked_end(is_preview)
                raise
        ended = {"done": False}
        status = {"code": None}
        # Set only on the terminal body frame, so a cancelled stream never claims the slot.
        completed = {"done": False}

        def _finish() -> None:
            if ended["done"]:
                return
            ended["done"] = True
            code = status["code"]
            if media_owner is not None:
                media_keepwarm.end_request(media_owner, counted = code not in (401, 403))
            if not chat_tracked:
                return
            # Claim BEFORE dropping the admitted count: the preview busy guard keys on it, and decrementing
            # first lets a preview swap in before this claim clears the wrong checkpoint.
            if (
                not is_preview
                and isinstance(code, int)
                and 200 <= code < 300
                and completed["done"]
                and not path.endswith("count_tokens")
                and not path.endswith(_NON_LLM_SLOT_SUFFIXES)
                and not scope.get(_RESPONSE_FAILED_SCOPE_KEY)
                and not scope.get(_UNTRACKED_SCOPE_KEY)
            ):
                _claim_non_preview_slot()
            if scope.get(_ADMITTED_SCOPE_KEY):
                _note_admitted_end()
            if scope.get(_UNTRACKED_SCOPE_KEY):
                return
            # Runs before FastAPI auth: balance 401/403 but do not stamp activity.
            if code in (401, 403):
                _note_untracked_end(is_preview)
                return
            # A non-2xx preview never served tokens; do not let rejected public POSTs refresh the timer.
            if is_preview and not (isinstance(code, int) and 200 <= code < 300):
                _note_untracked_end(is_preview)
                return
            _note_end(is_preview)

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status["code"] = message.get("status")
            is_terminal = message.get("type") == "http.response.body" and not message.get(
                "more_body", False
            )
            await send(message)
            # Claim only after send() succeeds: a disconnect on the final write must not count as complete.
            if is_terminal:
                completed["done"] = True
                _finish()

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _finish()


def _loaded_identity(backend):
    if not backend.is_loaded or not backend.model_identifier:
        return None
    # Use the advertised id: it is the override key, so alias reloads keep saved launch flags.
    advertised = getattr(backend, "_openai_advertised_id", None) or backend.model_identifier
    identity = (backend.model_identifier, getattr(backend, "hf_variant", None), advertised)
    companion_roots = tuple(getattr(backend, "_openai_gguf_companion_roots", ()) or ())
    return (*identity, companion_roots) if companion_roots else identity


def _note_idle_unload_event(freed) -> None:
    """Monitor row for an idle auto-unload. Best-effort; uses the stash's
    advertised repo id so the row never shows the on-disk load path."""
    try:
        from core.inference.api_monitor import api_monitor
        from core.inference.model_ids import public_model_id

        identifier, variant, advertised = (list(freed) + [None, None, None])[:3]
        label = public_model_id(advertised or identifier) or "model"
        if variant and ":" not in label:
            label = f"{label}:{variant}"
        api_monitor.record_lifecycle(event = "unload", model = label, reason = "idle")
    except Exception as exc:
        logger.debug("idle unload monitor event failed: %s", exc)


async def idle_unload_loop(poll_seconds: float = 15.0) -> None:
    """Unload the loaded GGUF once idle past the configured TTL. Inert when off."""
    from utils.openai_auto_switch_settings import (
        get_auto_unload_api_only,
        get_auto_unload_idle_seconds,
        get_auto_unload_keep_kv,
    )

    def _user_pinned(b) -> bool:
        """Whether the setting spares this model. Re-read like the other
        settings: a KV save can outlive the user turning this on. getattr keeps
        a foreign backend (tests, MLX) on the old unload-everything path."""
        return get_auto_unload_api_only() and getattr(b, "_loaded_by_user_action", False)

    seen_model = None
    while True:
        await asyncio.sleep(poll_seconds)
        try:
            from core.inference.media_keepwarm import idle_unload_step
            await idle_unload_step()
        except Exception as exc:
            logger.debug("media idle_unload_step failed: %s", exc)
        try:
            ttl = await asyncio.to_thread(get_auto_unload_idle_seconds)
            if ttl <= 0:
                continue
            from core.inference.model_slots import unload_extra_models
            from routes.inference import get_llama_cpp_backend, release_chat_gpu_claim

            backend = get_llama_cpp_backend()
            # A (re)loaded model counts as activity so it survives one TTL before its first request.
            async with _unload_gate():
                if _is_idle(ttl) and await asyncio.to_thread(
                    unload_extra_models, _user_pinned, True
                ):
                    await asyncio.to_thread(release_chat_gpu_claim)
                # Purging the stash mid-reload would race the restore.
                current = _loaded_identity(backend)
                if current != seen_model:
                    seen_model = current
                    if current is not None:
                        _note_activity()
                        _set_last_unloaded(None)
                if backend.is_loaded and await asyncio.to_thread(_user_pinned, backend):
                    # Loaded from the UI, so keep it resident; only API-loaded models are freed.
                    continue
                if backend.is_loaded and _is_idle(ttl):
                    freed = _loaded_identity(backend)
                    manifest = None
                    if await asyncio.to_thread(get_auto_unload_keep_kv):
                        try:
                            manifest = await asyncio.to_thread(
                                backend.save_slots_for_resume,
                                lambda: not _is_idle(ttl),
                            )
                        except Exception as exc:
                            logger.debug("slot save before idle unload failed: %s", exc)
                    ttl = await asyncio.to_thread(get_auto_unload_idle_seconds)
                    if (
                        ttl <= 0
                        or not _is_idle(ttl)
                        or await asyncio.to_thread(_user_pinned, backend)
                    ):
                        if manifest:
                            _delete_resume_files(manifest)
                        continue
                    if manifest and not await asyncio.to_thread(get_auto_unload_keep_kv):
                        _delete_resume_files(manifest)
                        manifest = None
                    # A request may register _pending during the off-loop setting read; recheck.
                    if not _is_idle(ttl):
                        if manifest:
                            _delete_resume_files(manifest)
                        continue
                    try:
                        await asyncio.to_thread(backend.unload_model)
                    except Exception:
                        if manifest:
                            _delete_resume_files(manifest)
                        raise
                    _set_last_unloaded(freed)
                    if manifest and freed:
                        _set_kv_resume({"identity": freed, **manifest})
                        logger.info("Idle auto-unload: saved slot KV for restore on reload")
                    elif manifest:
                        _delete_resume_files(manifest)
                    # After the stash, so a failed release never loses the reload identity.
                    try:
                        from hub.services.models.account_access import clear_resident
                        from routes.inference import release_chat_gpu_claim

                        clear_resident("chat")
                        await asyncio.to_thread(release_chat_gpu_claim)
                    except Exception as exc:  # noqa: BLE001 - the unload already happened
                        logger.debug("Idle auto-unload: claim release failed: %s", exc)
                    logger.info("Idle auto-unload: freed GGUF after %ss idle", ttl)
                    _note_idle_unload_event(freed)
                    seen_model = None
        except Exception as exc:
            logger.debug("idle_unload_loop iteration failed: %s", exc)
