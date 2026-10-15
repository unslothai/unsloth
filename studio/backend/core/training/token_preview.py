# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owns the token preview subprocess (core.training.token_preview_worker).

One worker at a time, kept for the model it was started for so typing in the preview answers in
milliseconds; a request for another model, a training start or an idle timeout ends it.
"""

from __future__ import annotations

import contextlib
import itertools
import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")
# The first request imports torch and Unsloth and may download the tokenizer files.
FIRST_REQUEST_TIMEOUT_S = 300.0
REQUEST_TIMEOUT_S = 60.0
# Applied in the child before anything imports torch: a CUDA context alone costs hundreds of MB of
# VRAM, and the preview only tokenizes. UNSLOTH_ALLOW_CPU lets Unsloth's chat templates import there.
_NO_ACCELERATOR_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "UNSLOTH_ALLOW_CPU": "1",
}


class TokenPreviewError(RuntimeError):
    pass


class _TokenPreviewService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[Any] = None
        self._cmd_queue: Optional[Any] = None
        self._resp_queue: Optional[Any] = None
        self._key: Optional[tuple] = None
        self._served = False
        self._stop_pending = False
        self._ids = itertools.count(1)
        from core.training.account_jobs import init_job_owner

        # Lets retire_account_jobs stop a retired account's warm worker before its roots move.
        init_job_owner(self, self._alive, self.stop)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _stop_locked(self) -> None:
        self._stop_pending = False
        proc, self._proc = self._proc, None
        self._key = None
        if proc is None:
            return
        try:
            if proc.is_alive() and self._cmd_queue is not None:
                self._cmd_queue.put({"op": "shutdown"})
                proc.join(timeout = 2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout = 3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout = 3)
        except Exception:  # noqa: BLE001 - the reap is best-effort
            logger.warning("could not stop the token preview worker", exc_info = True)
        if proc.exitcode is not None:
            from utils.process_lifetime import forget_pid
            forget_pid(proc.pid)

    def _start_locked(self, key: tuple, config: dict) -> None:
        from utils.process_lifetime import is_process_shutting_down
        from utils.transformers_version import sidecar_swap_kind

        if is_process_shutting_down():
            raise TokenPreviewError("Unsloth is shutting down.")
        if sidecar_swap_kind() is not None:
            raise TokenPreviewError(
                "A transformers installation is in progress; try the preview again when it finishes."
            )
        from core.training.account_jobs import account_is_retired
        from utils.account_context import current_account

        # Owner and in-flight count go up before the spawn, as owned_job does: a retirement in between
        # then refuses (retry) or sees this worker, instead of finishing before it exists.
        with self._account_job_lock:
            if account_is_retired():
                raise TokenPreviewError("This account has been removed.")
            self._result_account = current_account()
            self._account_inflight += 1
        try:
            self._spawn_locked(key, config)
        finally:
            with self._account_job_lock:
                self._account_inflight -= 1

    def _spawn_locked(self, key: tuple, config: dict) -> None:
        from core.training.account_jobs import account_process_spec
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.process_lifetime import adopt_pid

        cache_env = get_hf_cache_paths().child_env({})
        with (
            child_environment_for_spawn(cache_env),
            native_path_secret_removed_for_child_start(),
        ):
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()
            process_args, process_kwargs = account_process_spec(
                "core.training.token_preview_worker",
                "run_token_preview_process",
                {**cache_env, **_NO_ACCELERATOR_ENV},
                {
                    "cmd_queue": self._cmd_queue,
                    "resp_queue": self._resp_queue,
                    "config": config,
                },
            )
            proc = _CTX.Process(
                target = run_without_native_path_secret,
                args = process_args,
                kwargs = process_kwargs,
                daemon = True,
            )
            proc.start()
        adopt_pid(proc.pid)
        self._proc, self._key, self._served = proc, key, False
        logger.info("Token preview worker started (pid=%s)", proc.pid)

    def preview(
        self,
        key: tuple,
        config: dict,
        request: dict,
        record_load: Optional[Callable[[], Any]] = None,
    ) -> dict:
        """``record_load`` wraps a request that loads the tokenizer, decided under the lock so a
        concurrent request swapping the worker cannot make that load go unrecorded."""
        with self._lock:
            loads = not (self._alive() and self._key == key and self._served)
            with record_load() if loads and record_load else contextlib.nullcontext():
                reply = self._request_locked(key, config, request)
        if not reply.get("ok"):
            raise TokenPreviewError(reply.get("error") or "Token preview failed.")
        return reply["result"]

    def _request_locked(self, key: tuple, config: dict, request: dict) -> dict:
        if not self._alive() or self._key != key:
            self._stop_locked()
            self._start_locked(key, config)
        request_id = next(self._ids)
        self._cmd_queue.put({**request, "id": request_id})
        deadline = time.monotonic() + (
            REQUEST_TIMEOUT_S if self._served else FIRST_REQUEST_TIMEOUT_S
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_locked()
                raise TokenPreviewError("The token preview timed out.")
            try:
                reply = self._resp_queue.get(timeout = min(remaining, 1.0))
            except queue.Empty:
                if not self._alive():
                    self._stop_locked()
                    raise TokenPreviewError("The token preview worker stopped unexpectedly.")
                continue
            if reply.get("id") == request_id:
                break
        self._served = True
        if reply.get("fatal") or self._stop_pending:
            self._stop_locked()
        return reply

    def stop(self) -> None:
        # Never waits behind a preview in flight (a first one can take minutes): ending its worker
        # makes that request fail at once instead.
        if self._lock.acquire(blocking = False):
            try:
                self._stop_locked()
            finally:
                self._lock.release()
            return
        # The holder may be between workers, with none to terminate yet; it ends whatever it ends up with.
        self._stop_pending = True
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.terminate()


_service = _TokenPreviewService()


def run_token_preview(
    key: tuple,
    config: dict,
    request: dict,
    record_load: Optional[Callable[[], Any]] = None,
) -> dict:
    return _service.preview(key, config, request, record_load)


def stop_token_preview_worker() -> None:
    """Free the worker's memory; safe to call when none is running."""
    _service.stop()
