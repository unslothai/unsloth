# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the chat context when an image or video load takes the GPU from chat.

Before the media acquire (outside every lock) the chat KV (llama-server's slots, or the MLX
worker's prompt cache) is saved to disk, bounded by
``SWAP_KV_SAVE_BUDGET_S``. The manifest waits here as "pending" until ``gpu_arbiter._evict_chat``
unloads that same model; only then does it become the keep-warm stash that the next chat load
restores. If nothing was evicted, ``SwapSave.close`` drops it and deletes its files.

Lock order: gpu_arbiter._lock -> this _lock -> llama_keepwarm._lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Keeps the load POST far below the ~100 s a quick tunnel allows a silent request.
SWAP_KV_SAVE_BUDGET_S = 45.0
SAVING_CHAT_CONTEXT = "saving_chat_context"

_lock = threading.Lock()
# {"identity", "origin": "swap", **manifest}, waiting for _evict_chat to unload that model.
_pending: Optional[dict] = None
# The manifest a commit promoted, so the handle that saved it writes the lifecycle row.
_committed: Optional[dict] = None
# (media, account id) -> how many saves are running, for the load-progress phase.
_saving: dict[tuple[str, Optional[str]], int] = {}


def _account_id() -> Optional[str]:
    try:
        from utils.account_context import current_account_id
        return current_account_id()
    except Exception:
        return None


def saving_progress(media: str) -> Optional[dict]:
    """The load-progress answer while the caller's own save runs, else None."""
    with _lock:
        if _saving.get((media, _account_id())):
            return {"phase": SAVING_CHAT_CONTEXT}
    return None


def _enter_saving(key: tuple[str, Optional[str]]) -> None:
    with _lock:
        _saving[key] = _saving.get(key, 0) + 1


def _leave_saving(key: tuple[str, Optional[str]]) -> None:
    with _lock:
        left = _saving.get(key, 1) - 1
        if left > 0:
            _saving[key] = left
        else:
            _saving.pop(key, None)


class PreparingLoad:
    """One load POST still before its engine registers the load; ``/<media>/unload`` cancels it.

    ``load_token`` is the request's own token, so an unload naming it cancels this POST and no other."""

    def __init__(
        self,
        media: str,
        load_token: Optional[str] = None,
    ):
        self.media = media
        self.load_token = load_token or None
        self._cancelled = threading.Event()
        # Held from the check through begin_load's registration, so a cancel lands before or after it.
        self.lock = threading.RLock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        with self.lock:
            self._cancelled.set()


_preparing: dict[tuple[str, Optional[str]], list[PreparingLoad]] = {}
_current_preparing: contextvars.ContextVar[Optional[PreparingLoad]] = contextvars.ContextVar(
    "chat_swap_kv_preparing_load", default = None
)


def load_preparing(media: str) -> bool:
    """Whether one of the caller's load POSTs is still in its preflight (evidence, checks, acquire)."""
    with _lock:
        return any(not t.cancelled for t in _preparing.get((media, _account_id()), ()))


def cancel_preparing_loads(
    media: str,
    account: Optional[str],
    *,
    any_account: bool = False,
    load_token: Optional[str] = None,
) -> int:
    """Cancel ``media`` load POSTs in preflight: ``account``'s own, or all when the unload is not account scoped.

    With ``load_token``, only ``account``'s POSTs that carry that token (an unload naming its load).
    Returns once each one either stopped or registered, as the token's lock is held until then."""
    with _lock:
        tokens = [
            t
            for (m, a), held in _preparing.items()
            if m == media and (any_account or a == account)
            for t in held
            if not t.cancelled and (load_token is None or t.load_token == load_token)
        ]
    # Outside _lock: this waits for a registration in flight under the token's own lock.
    for token in tokens:
        token.cancel()
    return len(tokens)


def preparing_load_token() -> Optional[PreparingLoad]:
    """The preflight token of the load POST this context runs in, if any."""
    return _current_preparing.get()


def raise_if_load_cancelled(message: str) -> None:
    """Raise the route's cancelled-load error once an unload cancelled this POST's preflight."""
    token = _current_preparing.get()
    if token is not None and token.cancelled:
        raise RuntimeError(message)


def run_unless_load_cancelled(message: str, fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` under this POST's preflight lock once no unload has cancelled it, so a
    destructive step (an engine switch, another resident's unload) never runs for a cancelled load:
    a cancel lands before the check, or waits for ``fn`` to finish."""
    token = _current_preparing.get()
    if token is None:
        return fn(*args, **kwargs)
    with token.lock:
        if token.cancelled:
            raise RuntimeError(message)
        return fn(*args, **kwargs)


@contextlib.contextmanager
def preparing_load(media: str, load_token: Optional[str] = None):
    """Held around a load POST until begin_load registers it, so load-progress reads it as starting; yields the token an unload cancels."""
    key = (media, _account_id())
    token = PreparingLoad(media, load_token)
    with _lock:
        _preparing.setdefault(key, []).append(token)
    reset = _current_preparing.set(token)
    try:
        yield token
    finally:
        _current_preparing.reset(reset)
        with _lock:
            held = _preparing.get(key, [])
            if token in held:
                held.remove(token)
            if not held:
                _preparing.pop(key, None)


@contextlib.contextmanager
def _saving_marker(media: str, account: Optional[str]):
    key = (media, account)
    _enter_saving(key)
    try:
        yield
    finally:
        _leave_saving(key)


def _delete(manifest) -> None:
    if manifest:
        from core.inference.llama_keepwarm import _delete_resume_files
        _delete_resume_files(manifest)


def commit_after_chat_unload(unloaded_identity) -> bool:
    """Called by _evict_chat after the unload: the pending save becomes the reload stash."""
    global _pending, _committed
    with _lock:
        pending, _pending = _pending, None
    if pending is None:
        return False
    # "Keep chat context across unloads" may have been turned off while the save ran.
    if unloaded_identity is None or pending.get("identity") != unloaded_identity or not _keep_kv():
        # Chat swapped during the save (its KV is useless) or keep KV is off now.
        _delete(pending)
        return False
    with _lock:
        _committed = pending
    from core.inference import llama_keepwarm

    llama_keepwarm._set_kv_resume(pending)
    logger.info("Media swap: kept the chat slot KV for restore on reload")
    return True


class SwapSave:
    """Handle the load route closes in a finally around its acquire."""

    def __init__(
        self,
        manifest: Optional[dict] = None,
        *,
        log_swap: bool = True,
        saving: Optional[tuple[str, Optional[str]]] = None,
    ):
        self._manifest = manifest
        # Off logs the eviction here; When they fit's residency code already logs it as "evicted".
        self._log_swap = log_swap
        self._saving = saving

    def close(self) -> None:
        global _pending, _committed
        saving, self._saving = self._saving, None
        if saving is not None:
            _leave_saving(saving)
        manifest, self._manifest = self._manifest, None
        if manifest is None:
            return
        try:
            dropped = committed = False
            with _lock:
                if _pending is manifest:
                    # Nothing evicted chat: the save is stale the moment the load is done.
                    _pending, dropped = None, True
                elif _committed is manifest:
                    _committed, committed = None, True
            if dropped:
                _delete(manifest)
            elif committed and self._log_swap:
                from core.inference import llama_keepwarm
                llama_keepwarm._note_idle_unload_event(manifest.get("identity"), reason = "swap")
        except Exception as exc:  # noqa: BLE001 - the load already ran
            logger.debug("chat KV swap cleanup failed: %s", exc)


async def acquire_then_close(swap: SwapSave, acquire):
    """Run the blocking acquire on a thread and close ``swap`` only once that thread is done. A
    cancelled request leaves the thread running, and it may still evict chat and commit the save."""
    task = asyncio.ensure_future(asyncio.to_thread(acquire))

    def _done(finished) -> None:
        if not finished.cancelled():
            finished.exception()  # an awaiter still there re-raises it
        swap.close()

    task.add_done_callback(_done)
    return await asyncio.shield(task)


def _keep_kv() -> bool:
    try:
        from utils.openai_auto_switch_settings import get_auto_unload_keep_kv
        return bool(get_auto_unload_keep_kv())
    except Exception:
        return False


def _request_counted() -> bool:
    """Whether the middleware counts this request as chat inference: a /v1 image or video call
    loading through the media auto-switch. Read it on the request's context, not the save thread."""
    from core.inference import llama_keepwarm

    scope = llama_keepwarm._current_response_scope.get()
    return isinstance(scope, dict) and bool(scope.get(llama_keepwarm._TRACKED_SCOPE_KEY))


def _chat_busy(counted: bool = False) -> bool:
    from state import active_generations

    if active_generations.count() > 0:
        return True
    if counted:
        # The auto-switch holds chat's lifecycle gate, so a request waiting behind it can't start.
        from core.inference.media_switch_backends import chat_busy
        return chat_busy(count_pending = False)
    from core.inference import llama_keepwarm

    return llama_keepwarm.other_inference_request_count(current_request_counted = False) > 0


def _chat_account() -> Optional[str]:
    from core.inference import gpu_arbiter
    try:
        return gpu_arbiter.owner_account(gpu_arbiter.CHAT)
    except TypeError:
        return gpu_arbiter.owner_account()


def mlx_chat_identity(orchestrator):
    serves_mlx = getattr(orchestrator, "_serves_mlx", None)
    if not callable(serves_mlx) or not serves_mlx():
        return None
    name = orchestrator.active_model_name
    return (name, None, getattr(orchestrator, "_openai_advertised_id", None) or name, "mlx")


def _chat_saver():
    """``(identity, save)`` for the resident chat model whose KV a swap can keep, else None."""
    from core.inference.llama_keepwarm import _loaded_identity
    from routes.inference import get_llama_cpp_backend

    llama = get_llama_cpp_backend()
    identity = _loaded_identity(llama)
    if identity is not None:
        if not getattr(llama, "_gguf_path", None):
            return None
        return identity, llama.save_slots_for_resume
    from core.inference.orchestrator import peek_inference_backend

    orchestrator = peek_inference_backend()
    identity = mlx_chat_identity(orchestrator)
    return None if identity is None else (identity, orchestrator.save_prompt_cache_for_resume)


def _save_in_thread(save, should_abort, abort: threading.Event, loop, future) -> None:
    """Runs the blocking save on its own thread, so neither the budget nor the event loop's
    shutdown waits on it. A result nobody is waiting for any more is deleted right here."""
    try:
        manifest, error = save(should_abort), None
    except BaseException as exc:  # noqa: BLE001 - reported to the waiter
        manifest, error = None, exc
    if abort.is_set():
        _delete(manifest)
        return

    def _deliver():
        if future.done():
            _delete(manifest)
        elif error is not None:
            future.set_exception(error)
        else:
            future.set_result(manifest)

    try:
        loop.call_soon_threadsafe(_deliver)
    except RuntimeError:  # the loop is gone
        _delete(manifest)


async def save_chat_kv_before_swap(
    owner: str,
    media: str,
    *,
    policy: str = "off",
    need_mib: Optional[dict] = None,
    gpu_ids: Optional[list] = None,
    **_ignored,
) -> SwapSave:
    """Save the chat KV if acquiring ``owner`` will evict the chat model; never raises.

    ``policy``, ``need_mib`` and ``gpu_ids`` are the keywords the route passes to the acquire, so the peek
    predicts that exact acquire. Off passes none, and the acquire then runs as Off too."""
    global _pending
    try:
        from core.inference import gpu_arbiter
        from utils.openai_auto_switch_settings import get_auto_unload_keep_kv

        if not gpu_arbiter.would_evict(owner, policy = policy, need_mib = need_mib, gpu_ids = gpu_ids):
            return SwapSave()
        if not await asyncio.to_thread(get_auto_unload_keep_kv):
            return SwapSave()
        saver = _chat_saver()
        if saver is None:
            return SwapSave()
        identity, save = saver
        account = _account_id()
        counted = _request_counted()
        if _chat_account() != account or _chat_busy(counted):
            return SwapSave()
    except Exception as exc:  # noqa: BLE001 - fail open: the load runs as before
        logger.debug("chat KV swap save skipped: %s", exc)
        return SwapSave()

    budget = SWAP_KV_SAVE_BUDGET_S
    abort = threading.Event()
    deadline = time.monotonic() + budget

    preflight = _current_preparing.get()

    def should_abort() -> bool:
        return (
            abort.is_set()
            or time.monotonic() >= deadline
            or (preflight is not None and preflight.cancelled)
            or _chat_busy(counted)
        )

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    manifest = None
    cancelled = None
    saving = (media, account)
    _enter_saving(saving)
    try:
        threading.Thread(
            target = _save_in_thread,
            args = (save, should_abort, abort, loop, future),
            name = "chat-kv-swap-save",
            daemon = True,
        ).start()
        try:
            manifest = await asyncio.wait_for(asyncio.shield(future), budget)
        except BaseException as exc:  # noqa: BLE001 - timeout, client gone, a failed save
            abort.set()
            # Both this and the thread's delivery run on the loop: a result that just landed
            # is deleted here, and one still to come finds the future cancelled and deletes itself.
            if future.done() and not future.cancelled() and future.exception() is None:
                _delete(future.result())
            future.cancel()
            if isinstance(exc, asyncio.CancelledError):
                cancelled = exc
            elif isinstance(exc, asyncio.TimeoutError):
                logger.info(
                    "Media swap: the chat KV save passed %ss; continuing without it", budget
                )
            else:
                logger.debug("chat KV swap save failed: %s", exc)
        if cancelled is not None:
            raise cancelled
        try:
            if manifest and (
                abort.is_set()
                or (preflight is not None and preflight.cancelled)
                or _chat_busy(counted)
            ):
                _delete(manifest)
                manifest = None
        except Exception:
            _delete(manifest)
            manifest = None
    except BaseException:
        _leave_saving(saving)
        raise
    if not manifest:
        return SwapSave(saving = saving)
    entry = {"identity": identity, "origin": "swap", **manifest}
    with _lock:
        older, _pending = _pending, entry
    _delete(older)
    return SwapSave(entry, log_swap = policy == "off", saving = saving)
