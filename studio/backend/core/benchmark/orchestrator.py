# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark orchestrator — in-process, threaded.

Runs ``lm_eval.simple_evaluate`` in a worker thread so the event loop stays
responsive and a cancel can be requested via a :class:`threading.Event`.

Benchmarks always talk to a local inference server over HTTP (lm_eval's
``gguf`` backend); the model is never loaded in-process, so there
is no need for the heavyweight ``mp.spawn`` subprocess + queue machinery that
training/inference/export use. Running in-process also means we cannot forcibly
kill an in-flight ``simple_evaluate`` call, so cancel is best-effort: it stops
waiting for the result and reports cancellation, while the worker thread
finishes in the background and is discarded.
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


# ── Progress reporting ────────────────────────────────
#
# lm_eval reports all progress through tqdm. Rather than the fragile global
# __init__/update monkey-patch we had before, we mutate the *existing*
# ``tqdm.tqdm`` class's ``update`` method so the change is visible to every
# ``from tqdm import tqdm`` binding lm_eval already captured at import time.
# ``update`` returns True only when a display actually happened, and
# ``format_dict`` gives us the documented progress fields — no private
# attribute sniffing. The callback writes the latest progress into a
# thread-safe slot the SSE stream reads.

_progress_lock = threading.Lock()
_latest_progress: Optional[dict] = None

_TQDM_PATCHED = False
_ORIG_TQDM_UPDATE = None


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
    global _TQDM_PATCHED, _ORIG_TQDM_UPDATE
    if _TQDM_PATCHED:
        return

    import tqdm as _tqdm_mod

    _ORIG_TQDM_UPDATE = _tqdm_mod.tqdm.update

    def _patched_update(self, n = 1):
        displayed = _ORIG_TQDM_UPDATE(self, n)
        if displayed:
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
    global _TQDM_PATCHED, _ORIG_TQDM_UPDATE
    if not _TQDM_PATCHED:
        return
    import tqdm as _tqdm_mod
    _tqdm_mod.tqdm.update = _ORIG_TQDM_UPDATE
    _TQDM_PATCHED = False
    _ORIG_TQDM_UPDATE = None


def _run_benchmark_task(params: dict) -> dict:
    """Import lm_eval lazily and run simple_evaluate, returning the result dict.

    ``params`` is the kwargs dict produced by ``resolve_model_details`` /
    ``build_gguf_kwargs``. Any internal (underscore-prefixed) keys
    are stripped before forwarding to lm_eval.
    """
    kwargs = {k: v for k, v in params.items() if not k.startswith("_")}

    import lm_eval as _lm_eval
    # Registers the Studio-aware ``gguf`` model name used by build_gguf_kwargs.
    from core.benchmark import gguf_client  # noqa: F401

    _install_tqdm_progress_patch()
    try:
        logger.info(
            "Starting lm_eval with tasks=%s, model=%s",
            kwargs.get("tasks"), kwargs.get("model"),
        )
        results = _lm_eval.simple_evaluate(**kwargs)
        logger.info("lm_eval completed")
        return results
    finally:
        _uninstall_tqdm_progress_patch()


class BenchmarkOrchestrator:
    def __init__(self):
        self._lock = threading.Lock()

        # Log ring buffer (powers the logs SSE endpoint)
        self._log_buffer: deque[dict] = deque(maxlen=_LOG_BUFFER_MAXLEN)
        self._log_seq: int = 0
        self._run_start_seq: int = 0

        # Run state
        self._active: bool = False
        self._cancel_requested: bool = False
        self._last_error: Optional[str] = None
        # Outcome of the latest run ("success" | "error" | "cancelled") and a
        # per-run counter, so a client whose /run request dropped can tell
        # whether the finished run it sees on /status is its own.
        self._last_op_status: Optional[str] = None
        self._op_seq: int = 0
        # Account that started the current/last run (see account_jobs.job_is_foreign).
        self._result_account = None

        # Cancellation is best-effort in-process: set to signal the run loop to
        # stop waiting on the worker thread.
        self._cancel_event = threading.Event()

        # Single worker thread: serializes lm_eval runs so a backgrounded
        # (cancelled) run cannot compete with a new one for the inference server.
        self._executor = ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "benchmark")

        atexit.register(self._cleanup)
        logger.info("BenchmarkOrchestrator initialized (in-process, threaded)")

    # ── Log helpers ──────────────────────────────────────

    def _append_log(self, stream: str, line: str, ts: Optional[float] = None) -> None:
        with self._lock:
            self._log_seq += 1
            self._log_buffer.append({
                "seq": self._log_seq,
                "stream": stream,
                "line": line,
                "ts": ts or time.time(),
            })

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

    # ── State helpers ─────────────────────────────────────

    def is_active(self) -> bool:
        return self._active

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

    # ── Run and cancel ────────────────────────────────────

    def cancel(self) -> bool:
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

    def run(self, params: dict) -> dict:
        """Run a benchmark in a worker thread.

        Blocks (responsively) until the run completes or is cancelled, then
        returns the lm_eval result dict. A cancelled run returns ``{}``. After a
        successful return the run stays active until ``finish`` is called.
        """
        from utils.account_context import current_account

        self._active = True
        self._op_seq += 1
        self._result_account = current_account()
        self._last_op_status = None
        self._cancel_requested = False
        self._last_error = None
        self._cancel_event.clear()
        clear_progress()

        self._append_log("stdout", "Starting benchmark run...")

        future = self._executor.submit(_run_benchmark_task, params)

        result: Any = None
        error: Optional[str] = None
        while True:
            if self._cancel_event.is_set():
                # Best-effort cancel: stop waiting on the worker thread,
                # which finishes in the background and is discarded.
                self._cancel_requested = True
                self._last_op_status = "cancelled"
                self._active = False
                return {}
            try:
                result = future.result(timeout = 0.5)
                break
            except FuturesTimeoutError:
                continue
            except Exception as e:
                error = str(e) or type(e).__name__
                break

        if error is None and result is None:
            error = "Benchmark returned no result"
        if error:
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


# Singleton
_orchestrator: Optional[BenchmarkOrchestrator] = None


def get_benchmark_backend() -> BenchmarkOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BenchmarkOrchestrator()
    return _orchestrator
