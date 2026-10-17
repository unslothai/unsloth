# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the chat context when an image or video load takes the GPU from chat.

Before the media acquire (outside every lock) the chat slots' KV is saved to disk, bounded by
``SWAP_KV_SAVE_BUDGET_S``. The manifest waits here as "pending" until ``gpu_arbiter._evict_chat``
unloads that same model; only then does it become the keep-warm stash that the next chat load
restores. If nothing was evicted, ``SwapSave.close`` drops it and deletes its files.

Lock order: gpu_arbiter._lock -> this _lock -> llama_keepwarm._lock.
"""

from __future__ import annotations

import asyncio
import contextlib
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


@contextlib.contextmanager
def _saving_marker(media: str, account: Optional[str]):
    key = (media, account)
    with _lock:
        _saving[key] = _saving.get(key, 0) + 1
    try:
        yield
    finally:
        with _lock:
            left = _saving.get(key, 1) - 1
            if left > 0:
                _saving[key] = left
            else:
                _saving.pop(key, None)


def _delete(manifest) -> None:
    if manifest:
        from core.inference.llama_keepwarm import _delete_resume_files
        _delete_resume_files(manifest)


def commit_after_chat_unload(unloaded_identity) -> bool:
    """Called by _evict_chat after the unload: the pending save becomes the reload stash."""
    global _pending, _committed
    with _lock:
        pending, _pending = _pending, None
        matched = pending is not None and unloaded_identity is not None and (
            pending.get("identity") == unloaded_identity
        )
        if matched:
            _committed = pending
    if pending is None:
        return False
    if not matched:
        # A different model was unloaded (chat swapped during the save): its KV is useless.
        _delete(pending)
        return False
    from core.inference import llama_keepwarm
    llama_keepwarm._set_kv_resume(pending)
    logger.info("Media swap: kept the chat slot KV for restore on reload")
    return True


class SwapSave:
    """Handle the load route closes in a finally around its acquire."""

    def __init__(self, manifest: Optional[dict] = None, *, log_swap: bool = True):
        self._manifest = manifest
        # Off logs the eviction here; When they fit's residency code already logs it as "evicted".
        self._log_swap = log_swap

    def close(self) -> None:
        global _pending, _committed
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


def _policy() -> str:
    try:
        from utils.multi_model_settings import get_multi_model_mode
        return str(get_multi_model_mode() or "off")
    except Exception:
        return "off"


def _chat_busy() -> bool:
    from core.inference import llama_keepwarm
    from state import active_generations

    return (
        active_generations.count() > 0
        or llama_keepwarm.other_inference_request_count(current_request_counted = False) > 0
    )


def _chat_account() -> Optional[str]:
    from core.inference import gpu_arbiter

    try:
        return gpu_arbiter.owner_account(gpu_arbiter.CHAT)
    except TypeError:
        return gpu_arbiter.owner_account()


def _save_in_thread(llama, should_abort, abort: threading.Event, loop, future) -> None:
    """Runs the blocking save on its own thread, so neither the budget nor the event loop's
    shutdown waits on it. A result nobody is waiting for any more is deleted right here."""
    try:
        manifest, error = llama.save_slots_for_resume(should_abort), None
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
    owner: str, media: str, *, policy: Optional[str] = None, need_mib: Optional[dict] = None, **_ignored
) -> SwapSave:
    """Save the chat KV if acquiring ``owner`` will evict the chat model; never raises.

    ``policy`` and ``need_mib`` are the keywords the route passes to the acquire, so the peek
    predicts that exact acquire."""
    global _pending
    try:
        from core.inference import gpu_arbiter
        from core.inference.llama_keepwarm import _loaded_identity
        from utils.openai_auto_switch_settings import get_auto_unload_keep_kv

        policy = policy or _policy()
        if not gpu_arbiter.would_evict(owner, policy = policy, need_mib = need_mib):
            return SwapSave()
        if not await asyncio.to_thread(get_auto_unload_keep_kv):
            return SwapSave()
        from routes.inference import get_llama_cpp_backend

        llama = get_llama_cpp_backend()
        identity = _loaded_identity(llama)
        if identity is None or not getattr(llama, "_gguf_path", None):
            return SwapSave()
        account = _account_id()
        if _chat_account() != account or _chat_busy():
            return SwapSave()
    except Exception as exc:  # noqa: BLE001 - fail open: the load runs as before
        logger.debug("chat KV swap save skipped: %s", exc)
        return SwapSave()

    budget = SWAP_KV_SAVE_BUDGET_S
    abort = threading.Event()
    deadline = time.monotonic() + budget

    def should_abort() -> bool:
        return abort.is_set() or time.monotonic() >= deadline or _chat_busy()

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    manifest = None
    cancelled = None
    with _saving_marker(media, account):
        threading.Thread(
            target = _save_in_thread,
            args = (llama, should_abort, abort, loop, future),
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
                logger.info("Media swap: the chat KV save passed %ss; continuing without it", budget)
            else:
                logger.debug("chat KV swap save failed: %s", exc)
    if cancelled is not None:
        raise cancelled
    try:
        if manifest and (abort.is_set() or _chat_busy()):
            _delete(manifest)
            manifest = None
    except Exception:
        _delete(manifest)
        manifest = None
    if not manifest:
        return SwapSave()
    entry = {"identity": identity, "origin": "swap", **manifest}
    with _lock:
        older, _pending = _pending, entry
    _delete(older)
    return SwapSave(entry, log_swap = policy == "off")
