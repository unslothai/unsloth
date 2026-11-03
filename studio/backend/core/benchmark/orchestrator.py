# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs ``lm_eval.simple_evaluate`` on one worker thread against the local llama-server.

lm_eval can't be interrupted in-process, so Stop is best-effort: the run stops waiting and
reports cancelled while the worker finishes in the background.
"""

import atexit
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

_LOG_BUFFER_MAXLEN = 4000


# lm_eval reports progress only through tqdm. Patching ``tqdm.tqdm.update`` on the class reaches
# the bindings lm_eval captured at import; ``format_dict`` carries the public fields.
_progress_lock = threading.Lock()
_latest_progress: Optional[dict] = None

_TQDM_PATCHED = False
_ORIG_TQDM_UPDATE = None
_PROGRESS_THREAD: Optional[int] = None


def _report_progress(
    n: Optional[int],
    total: Optional[int],
    desc: Optional[str],
    elapsed: Optional[str] = None,
    eta: Optional[str] = None,
) -> None:
    global _latest_progress
    if not total:
        return
    pct = round(100.0 * n / total, 2)
    with _progress_lock:
        _latest_progress = {
            "pct": pct,
            "current": n,
            "total": total,
            "desc": desc,
            "elapsed": elapsed,
            "eta": eta,
        }


def get_latest_progress() -> Optional[dict]:
    with _progress_lock:
        return _latest_progress


def clear_progress() -> None:
    global _latest_progress
    with _progress_lock:
        _latest_progress = None


def _install_tqdm_progress_patch() -> None:
    global _TQDM_PATCHED, _ORIG_TQDM_UPDATE, _PROGRESS_THREAD
    # The patch is process-wide; only the eval worker's bars are this run's progress.
    _PROGRESS_THREAD = threading.get_ident()
    if _TQDM_PATCHED:
        return

    import tqdm as _tqdm_mod

    _ORIG_TQDM_UPDATE = _tqdm_mod.tqdm.update

    def _patched_update(self, n = 1):
        displayed = _ORIG_TQDM_UPDATE(self, n)
        if displayed and threading.get_ident() == _PROGRESS_THREAD:
            fmt = self.format_dict
            n_seen, total_seen, rate = fmt.get("n"), fmt.get("total"), fmt.get("rate")
            remaining = (total_seen - n_seen) / rate if (rate and total_seen) else None
            _report_progress(
                n_seen,
                total_seen,
                fmt.get("desc"),
                elapsed = _tqdm_mod.tqdm.format_interval(fmt.get("elapsed") or 0),
                eta = _tqdm_mod.tqdm.format_interval(remaining) if remaining is not None else None,
            )
        return displayed

    _tqdm_mod.tqdm.update = _patched_update
    _TQDM_PATCHED = True


def _uninstall_tqdm_progress_patch() -> None:
    global _TQDM_PATCHED, _ORIG_TQDM_UPDATE, _PROGRESS_THREAD
    _PROGRESS_THREAD = None
    if not _TQDM_PATCHED:
        return
    import tqdm as _tqdm_mod

    _tqdm_mod.tqdm.update = _ORIG_TQDM_UPDATE
    _TQDM_PATCHED = False
    _ORIG_TQDM_UPDATE = None


def _run_benchmark_task(params: dict) -> dict:
    """Run simple_evaluate with ``build_gguf_kwargs`` output, minus underscore-prefixed keys."""
    kwargs = {k: v for k, v in params.items() if not k.startswith("_")}

    import lm_eval as _lm_eval

    # Registers the model name build_gguf_kwargs uses.
    from core.benchmark import gguf_client  # noqa: F401

    _install_tqdm_progress_patch()
    try:
        logger.info(
            "Starting lm_eval with tasks=%s, model=%s",
            kwargs.get("tasks"),
            kwargs.get("model"),
        )
        results = _lm_eval.simple_evaluate(**kwargs)
        logger.info("lm_eval completed")
        return results
    finally:
        _uninstall_tqdm_progress_patch()


# The running worker's Stop. lm_eval can't be interrupted, so StudioGGUFLM checks this before
# each request; one worker runs at a time, so one slot is enough.
_worker_cancel: Optional[threading.Event] = None


def worker_cancelled() -> bool:
    ev = _worker_cancel
    return ev is not None and ev.is_set()


class BenchmarkOrchestrator:
    def __init__(self):
        self._lock = threading.Lock()

        self._log_buffer: deque[dict] = deque(maxlen = _LOG_BUFFER_MAXLEN)
        self._log_seq: int = 0
        self._run_start_seq: int = 0

        self._active: bool = False
        self._cancel_requested: bool = False
        self._last_error: Optional[str] = None
        # Lets a client whose /run dropped tell whether the outcome on /status is its run.
        self._last_op_status: Optional[str] = None
        self._op_seq: int = 0
        # Account that started the current/last run (see account_jobs.job_is_foreign).
        self._result_account = None

        self._cancel_event = threading.Event()

        # One worker, so a stopped run still finishing can't compete with a new one.
        self._executor = ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "benchmark")
        # Stop can't interrupt lm_eval, so a stopped run's worker keeps the executor until it returns.
        self._worker: Optional[Any] = None

        atexit.register(self._cleanup)
        logger.info("BenchmarkOrchestrator initialized (in-process, threaded)")

    def _append_log(
        self,
        stream: str,
        line: str,
        ts: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._log_seq += 1
            self._log_buffer.append(
                {
                    "seq": self._log_seq,
                    "stream": stream,
                    "line": line,
                    "ts": ts or time.time(),
                }
            )

    def clear_logs(self) -> None:
        with self._lock:
            self._log_buffer.clear()
            self._run_start_seq = self._log_seq

    def get_logs_since(self, cursor: int) -> tuple[list[dict], int]:
        with self._lock:
            entries = [e for e in self._log_buffer if e["seq"] > cursor]
        if entries:
            return entries, entries[-1]["seq"]
        return [], cursor

    def get_current_log_seq(self) -> int:
        with self._lock:
            return self._log_seq

    def get_run_start_seq(self) -> int:
        with self._lock:
            return self._run_start_seq

    def is_active(self) -> bool:
        return self._active

    def after_worker(self, callback) -> None:
        """Call ``callback`` once the current worker, if any, has returned."""
        worker = self._worker
        if worker is None:
            callback()
        else:
            worker.add_done_callback(lambda _f: callback())

    def is_draining(self) -> bool:
        """A stopped run's worker is still finishing; a new run would queue behind it."""
        worker = self._worker
        return worker is not None and not worker.done()

    def was_cancelled(self) -> bool:
        return self._cancel_requested

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def get_last_op_status(self) -> Optional[str]:
        return self._last_op_status

    def get_op_seq(self) -> int:
        return self._op_seq

    def get_latest_progress(self) -> Optional[dict]:
        return get_latest_progress()

    def try_begin(self, account = None) -> Optional[int]:
        """Claim the run slot, or None when it's taken. Synchronous so the route claims it
        before its first await; two concurrent requests could otherwise both start."""
        from utils.account_context import current_account
        with self._lock:
            if self._active or self.is_draining():
                return None
            self._active = True
            self._op_seq += 1
            self._result_account = account if account is not None else current_account()
            self._last_op_status = None
            self._cancel_requested = False
            self._last_error = None
            # A fresh event per run: clearing a shared one would let a Run issued
            # right after Stop un-cancel the run being stopped.
            self._cancel_event = threading.Event()
            # The last run's final progress would otherwise show on this run's card.
            clear_progress()
            return self._op_seq

    def release(self, op_seq: int) -> None:
        """Free a slot claimed with ``try_begin`` whose run never started."""
        with self._lock:
            if self._active and op_seq == self._op_seq:
                self._active = False

    def cancel(self) -> bool:
        # Same lock as run()'s submit, so a Stop lands either before it (nothing submitted) or
        # after it (the worker is recorded and drains).
        with self._lock:
            if not self._active:
                return False
            self._cancel_requested = True
            self._cancel_event.set()
            self._last_op_status = "cancelled"
            self._active = False
        logger.info("Benchmark cancel requested")
        return True

    def finish(
        self,
        op_seq: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a run that ``run`` returned successfully, once the caller has
        stored its results. Until then the run still counts as active. A stale
        ``op_seq`` (another run started since) or an already-ended run is a no-op."""
        if not self._active or self._cancel_requested:
            return
        if op_seq is not None and op_seq != self._op_seq:
            return
        self._last_error = error
        self._last_op_status = "error" if error else "success"
        self._active = False

    def run(
        self,
        params: dict,
        account = None,
        op_seq: Optional[int] = None,
    ) -> dict:
        """Block until the worker returns the lm_eval results, or ``{}`` once stopped. The run
        stays active until ``finish``. Without ``op_seq`` (from ``try_begin``) the slot is
        claimed here for ``account``, since an executor thread lacks the request's."""
        if op_seq is None:
            op_seq = self.try_begin(account)
            if op_seq is None:
                raise RuntimeError("A benchmark is already running.")
        # One step under the lock: a Stop between the check and the submit would otherwise free
        # the slot before is_draining() can see this worker.
        with self._lock:
            if op_seq != self._op_seq or self._cancel_event.is_set():
                return {}  # stopped, or superseded, before it started
            cancel_event = self._cancel_event
            clear_progress()
            global _worker_cancel
            _worker_cancel = cancel_event
            future = self._executor.submit(_run_benchmark_task, params)
            self._worker = future
        my_seq = op_seq
        self._append_log("stdout", "Starting benchmark run...")

        result: Any = None
        error: Optional[str] = None
        while True:
            # Checked again after a result arrives: Stop wins over a run that
            # happened to finish inside the same poll.
            if cancel_event.is_set():
                # A newer run may already own the shared state.
                if self._op_seq == my_seq:
                    self._cancel_requested = True
                    self._last_op_status = "cancelled"
                    self._active = False
                return {}
            try:
                result = future.result(timeout = 0.5)
                if not cancel_event.is_set():
                    break
                continue
            except FuturesTimeoutError:
                continue
            except Exception as e:
                error = str(e) or type(e).__name__
                break

        if error is None and result is None:
            error = "Benchmark returned no result"
        if error:
            if self._op_seq == my_seq:
                self._last_error = error
                self._last_op_status = "error"
                self._active = False
            raise RuntimeError(error)

        self._last_error = None
        return result

    def _cleanup(self) -> None:
        try:
            self._executor.shutdown(wait = False, cancel_futures = True)
        except Exception:
            pass


_orchestrator: Optional[BenchmarkOrchestrator] = None


def get_benchmark_backend() -> BenchmarkOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BenchmarkOrchestrator()
    return _orchestrator
