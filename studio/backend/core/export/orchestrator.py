# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export orchestrator — subprocess-based.

Same API as ExportBackend, but delegates all ML work to a persistent
subprocess spawned on first checkpoint load and reused for later exports.

When switching between checkpoints needing different transformers
versions, the old subprocess is killed and a new one spawned.

Pattern follows core/inference/orchestrator.py.
"""

import atexit
import structlog
from collections import deque
from loggers import get_logger
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from hub.utils.hf_tokens import HfTokenArg
from typing import Any, Deque, Dict, List, Optional, Tuple
from utils.paths import outputs_root

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")

# Max log lines kept per orchestrator (live log panel scrollback); ~1 MB worst-case.
_LOG_BUFFER_MAXLEN = 4000


_UNPINNED = object()


class ExportOrchestrator:
    """
    Export backend orchestrator — subprocess-based.

    Exposes the same API surface as ExportBackend so routes/export.py
    needs minimal changes. All heavy ML work happens in a persistent
    subprocess.
    """

    def __init__(self):
        self._proc: Optional[mp.Process] = None
        self._cmd_queue: Any = None
        self._resp_queue: Any = None

        # The private Hugging Face token directory a non-ambient worker runs against, and
        # every such directory this process has created and not yet confirmed removed. Three
        # states have to stay apart, because one scalar conflating them is a race per state:
        # the live worker's store, a store allocated for a worker that has not spawned yet,
        # and a store whose deletion failed and must be retried.
        self._token_store: Optional[str] = None
        self._token_store_pending: bool = False
        self._owned_token_stores: set = set()
        # Guards only the (process, token store) pair. Deliberately not self._lock, which an
        # export op holds for its whole duration: is_worker_alive and cancel_export are
        # lock-free against that one on purpose. Never held across rmtree or any other I/O,
        # so contention is a few instructions.
        self._state_lock = threading.Lock()
        # Serializes export ops so concurrent HTTP requests can't interleave commands.
        self._lock = threading.Lock()

        # Local state mirrors (updated from subprocess responses).
        self.current_checkpoint: Optional[str] = None
        self.is_vision: bool = False
        self.is_peft: bool = False

        # Thread-safe ring buffer of worker log lines; powers the export logs SSE endpoint.
        self._log_buffer: Deque[Dict[str, Any]] = deque(maxlen = _LOG_BUFFER_MAXLEN)
        self._log_lock = threading.Lock()
        # Monotonic seq, never reset, so SSE clients have a stable cursor across clear_logs().
        self._log_seq: int = 0
        # SSE defaults its cursor here so a late-connecting client still sees the full run.
        self._run_start_seq: int = 0
        # True while an export op runs; SSE ends the stream 1s after this flips False.
        self._export_active: bool = False
        # Set by cancel_export(); reset when a new load/export run starts. Lets the caller distinguish a
        # user cancel from a genuine subprocess crash.
        self._cancel_requested: bool = False

        # Kept so a client whose blocking POST was cut by a tunnel 524 can poll /api/export/status; _op_seq
        # is the monotonic baseline for "my op finished".
        self._op_lock = threading.Lock()
        self._op_seq: int = 0
        self._active_op_kind: Optional[str] = None
        self._last_op: Optional[Dict[str, Any]] = None

        atexit.register(self._cleanup)
        logger.info("ExportOrchestrator initialized (subprocess mode)")

    # ------------------------------------------------------------------

    def _append_log(self, entry: Dict[str, Any]) -> None:
        """Append a worker log line to the buffer, stamped with a monotonic seq."""
        line = entry.get("line")
        if not line:
            return
        with self._log_lock:
            self._log_seq += 1
            self._log_buffer.append(
                {
                    "seq": self._log_seq,
                    "stream": entry.get("stream", "stdout"),
                    "line": line,
                    "ts": entry.get("ts", time.time()),
                }
            )

    def clear_logs(self) -> None:
        """Drop buffered log lines from a previous op so the UI shows only this run.

        The seq counter is NOT reset (clients keep a stable cursor); the current seq
        is snapshotted into ``_run_start_seq`` to anchor the SSE default cursor.
        """
        with self._log_lock:
            self._log_buffer.clear()
            self._run_start_seq = self._log_seq

    def get_logs_since(self, cursor: int) -> Tuple[List[Dict[str, Any]], int]:
        """Return log entries with seq > cursor, plus the new cursor."""
        with self._log_lock:
            new_entries = [entry for entry in self._log_buffer if entry["seq"] > cursor]
        if new_entries:
            return new_entries, new_entries[-1]["seq"]
        return [], cursor

    def get_current_log_seq(self) -> int:
        """Return the current seq counter without reading any entries."""
        with self._log_lock:
            return self._log_seq

    def get_run_start_seq(self) -> int:
        """Return the seq captured at the current run's start (SSE default cursor)."""
        with self._log_lock:
            return self._run_start_seq

    def is_export_active(self) -> bool:
        """True while an export / load / cleanup command is running."""
        return self._export_active

    def is_worker_alive(self) -> bool:
        """True while the persistent export subprocess is running (op or idle).

        Reaps on the way out, like the internal check: utils/transformers_version.py calls
        this one directly, so a worker that died while idle would otherwise keep its private
        token store until something else happened to look.
        """
        proc = self._proc
        if proc is not None and proc.is_alive():
            return True
        self._reap_dead_worker()
        return False

    def was_cancelled(self) -> bool:
        """True if the in-flight (or most recent) run was cancelled by the user."""
        return self._cancel_requested

    def _record_op_finished(self, success: bool, message: str, output_path: Optional[str]) -> None:
        """Snapshot the just-finished op so status pollers can recover its outcome.

        Called from each op's ``finally`` (with ``_active_op_kind`` still set) BEFORE
        ``_export_active`` is cleared, so a status read that observes the op as
        inactive is guaranteed to also see this matching result.
        """
        with self._op_lock:
            self._op_seq += 1
            status = "cancelled" if self._cancel_requested else ("success" if success else "error")
            self._last_op = {
                "seq": self._op_seq,
                "kind": self._active_op_kind,
                "status": status,
                "output_path": output_path if success else None,
                "error": None if success else (message or None),
            }

    def get_last_op(self) -> Optional[Dict[str, Any]]:
        """Return the last finished op record (or None), for status recovery."""
        with self._op_lock:
            return dict(self._last_op) if self._last_op is not None else None

    def get_active_op_kind(self) -> Optional[str]:
        """Return the kind of the currently running op (or None when idle)."""
        return self._active_op_kind

    def cancel_export(self) -> bool:
        """Terminate the in-flight export subprocess immediately.

        An export op holds ``self._lock`` for its whole duration (blocked in
        ``_wait_response``), so we deliberately do NOT take the lock here -- we
        kill the worker process directly, which unblocks that wait and makes the
        in-flight op return a failure the caller surfaces as "cancelled".

        Only the export subprocess is touched; training and inference run in
        their own subprocesses and are left untouched.

        Returns True if a live subprocess was terminated, False if none ran.
        """
        self._cancel_requested = True
        proc = self._proc
        # Read after proc, and re-checked against it before either cleanup below. This runs
        # without the lock, so a reload can swap in a new worker and a new store between any
        # two statements here, and an old process paired with the replacement's store would
        # delete a live worker's credential directory.
        store = getattr(self, "_token_store", None)
        if proc is None or not proc.is_alive():
            # Nothing to cancel, but it may have died holding a store. Only if no reload has
            # swapped in a worker since we read: then this pair is stale and the store we
            # read belongs to the live one. The next liveness check reaps whatever is left.
            if self._proc is proc:
                self._discard_token_store(only = store)
            return False
        logger.info(
            "Export cancel requested: terminating export subprocess (pid=%s)",
            proc.pid,
        )
        try:
            proc.terminate()
            proc.join(timeout = 5)
        except Exception:
            pass
        if proc.is_alive():
            logger.warning("Export subprocess survived terminate, killing")
            try:
                proc.kill()
                proc.join(timeout = 3)
            except Exception:
                pass
        # _run_export swallows the RuntimeError this kill produces and returns without
        # shutting down, so neither _shutdown_subprocess cleanup runs on a cancel. Only once
        # the worker is confirmed dead, though: a survivor of terminate+kill (the wedged CUDA
        # syscall this class handles elsewhere) still has HF_TOKEN_PATH pointing here, and
        # pulling the directory out from under it would both break its Hub calls and let it
        # recreate an untracked one. It is discarded by the next shutdown instead.
        if not proc.is_alive():
            if self._proc is proc:
                self._discard_token_store(only = store)
        else:
            logger.warning(
                "Export subprocess survived cancellation; keeping its token store until it exits"
            )
        return True

    # ------------------------------------------------------------------

    def _spawn_subprocess(self, config: dict) -> None:
        """Spawn a new export subprocess."""
        # Inside an op a reservation is an install about to abort on is_export_active(), so raising here
        # would kill the export for an install that never proceeds.
        from utils.transformers_version import sidecar_swap_in_progress

        from utils.transformers_version import sidecar_swap_kind

        _swap_kind = sidecar_swap_kind()
        # An INSTALL reservation aborts on the is_export_active check, but a lazy REPAIR has none and may
        # be rebuilding the sidecar right now, so always refuse the spawn.
        if _swap_kind == "repair" or (_swap_kind is not None and not self._export_active):
            from utils.transformers_version import SidecarSwapInProgress
            raise SidecarSwapInProgress(
                "A transformers installation is replacing the latest sidecar; "
                "retry when it completes."
            )
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths

        cache_env = get_hf_cache_paths().child_env({})

        with (
            child_environment_for_spawn(cache_env),
            native_path_secret_removed_for_child_start(),
        ):
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()

            self._proc = _CTX.Process(
                target = run_without_native_path_secret,
                args = ("core.export.worker", "run_export_process", cache_env),
                kwargs = {
                    "cmd_queue": self._cmd_queue,
                    "resp_queue": self._resp_queue,
                    "config": config,
                },
                daemon = True,
            )
            self._proc.start()
        from utils.process_lifetime import adopt_pid

        adopt_pid(self._proc.pid)
        logger.info("Export subprocess started (pid=%s)", self._proc.pid)

    def _shutdown_subprocess(self, timeout: float = 10.0) -> bool:
        """Gracefully shut down the export subprocess.

        Returns True only once the worker is confirmed dead. If it survives
        terminate/kill (e.g. wedged in an uninterruptible CUDA syscall that outlives
        SIGKILL) the live handle is KEPT, not nulled, so is_worker_alive() and the
        pre-swap liveness guard can still observe the survivor instead of a cleared
        handle and refuse the destructive sidecar swap."""
        if self._proc is None or not self._proc.is_alive():
            self._proc = None
            # A worker that crashed after the loader persisted a token still left it behind.
            self._discard_token_store()
            return True

        self._drain_queue()

        try:
            self._cmd_queue.put({"type": "shutdown"})
        except (OSError, ValueError):
            pass

        try:
            self._proc.join(timeout = timeout)
        except Exception:
            pass

        # Force kill if still alive.
        if self._proc is not None and self._proc.is_alive():
            logger.warning("Export subprocess did not exit gracefully, terminating")
            try:
                self._proc.terminate()
                self._proc.join(timeout = 5)
            except Exception:
                pass
            if self._proc is not None and self._proc.is_alive():
                logger.warning("Subprocess still alive after terminate, killing")
                try:
                    self._proc.kill()
                    self._proc.join(timeout = 3)
                except Exception:
                    pass

        if self._proc is not None and self._proc.is_alive():
            # Survived SIGKILL: keep the handle so callers and the pre-swap guard see a live worker.
            logger.error(
                "Export subprocess still alive after terminate/kill; "
                "preserving its handle for the pre-swap liveness check"
            )
            return False

        self._proc = None
        self._cmd_queue = None
        self._resp_queue = None
        self._discard_token_store()
        logger.info("Export subprocess shut down")
        return True

    def _cleanup(self):
        """atexit handler."""
        self._shutdown_subprocess(timeout = 5.0)

    def _reap_dead_worker(self) -> None:
        """Drop a worker that died on its own, store included.

        An export that kills the worker surfaces as a RuntimeError that _run_export catches
        and returns from, so no shutdown runs and the credential its store may hold would sit
        in /tmp until the next load.
        """
        proc = self._proc
        if proc is not None and proc.is_alive():
            return
        # Compare and clear as one step: a reload can install a live worker between the
        # liveness check and the assignment, and clearing then would orphan a running GPU
        # worker and delete the credential directory it is using.
        with self._state_guard():
            if self._proc is not proc:
                return
            self._proc = None
            store = self._token_store
            if self._token_store_pending:
                return
            self._token_store = None
        self._sweep_token_stores(remove = store)

    def _state_guard(self) -> threading.Lock:
        """The (process, token store) lock, created on demand.

        The shutdown tests build an orchestrator with __new__ and no __init__, so every
        accessor of this state has to tolerate its absence.
        """
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            lock = self._state_lock = threading.Lock()
        return lock

    def _new_token_store(self) -> str:
        """A private Hugging Face token directory for the next non-ambient worker.

        The worker points HF_TOKEN_PATH here so the operator's stored login is out of its
        reach and any token the loader persists lands somewhere disposable. The parent holds
        the path so it can delete it even when the worker is killed rather than exiting.

        Published as *pending* until a worker is actually spawned against it: until then a
        lock-free canceller must not mistake it for a dead worker's leftovers.
        """
        import tempfile

        # Whatever was current stops being current here, so it goes with this sweep rather
        # than waiting for a later one to notice; it may hold the previous caller's token.
        with self._state_guard():
            previous = getattr(self, "_token_store", None)
        self._sweep_token_stores(remove = previous)
        store = tempfile.mkdtemp(prefix = "unsloth-export-hf-")
        with self._state_guard():
            if not hasattr(self, "_owned_token_stores"):
                self._owned_token_stores = set()
            self._owned_token_stores.add(store)
            self._token_store = store
            self._token_store_pending = True
        return store

    def _attach_token_store(self) -> None:
        """Mark the pending store as belonging to the worker that just spawned."""
        with self._state_guard():
            self._token_store_pending = False

    def _sweep_token_stores(self, remove: Optional[str] = None) -> None:
        """Delete owned stores that no worker is using, plus *remove* if given.

        The store currently published is never swept, pending or not: a concurrent reload
        may have just installed it, and a sweep that took it would hand that worker a path
        it would recreate untracked. A directory that will not delete, a file locked on
        Windows say, stays owned so the next sweep tries again.

        rmtree runs outside the state lock; only the bookkeeping is inside it.
        """
        import os
        import shutil

        with self._state_guard():
            current = getattr(self, "_token_store", None)
            candidates = [
                path
                for path in getattr(self, "_owned_token_stores", ())
                if path != current or (remove is not None and path == remove)
            ]
        for path in candidates:
            shutil.rmtree(path, ignore_errors = True)
            if os.path.exists(path):
                logger.warning("Could not remove the export token store %s; will retry", path)
                continue
            with self._state_guard():
                self._owned_token_stores.discard(path)

    def _discard_token_store(self, only: Any = _UNPINNED) -> None:
        """Retire the current worker's private token directory, credential and all.

        *only* pins the removal to one store, for a caller holding no lock: if a reload has
        already installed a replacement, that one belongs to the live worker, not to us.
        Pinning to ``None`` is meaningful, and means the cancelled worker had no store, so
        there is nothing of ours to remove; that is why the default is a separate sentinel.
        """
        # getattr throughout: an orchestrator built without __init__ (the shutdown tests do
        # exactly that) has none of these yet.
        with self._state_guard():
            store = getattr(self, "_token_store", None)
            if only is not _UNPINNED:
                if store != only:
                    # Not the current store any more. Still ours to remove if we made it.
                    target = only if only in getattr(self, "_owned_token_stores", ()) else None
                    store = None
                elif getattr(self, "_token_store_pending", False):
                    # A load published this store but its worker has not spawned yet, so it
                    # is not a dead worker's leftovers; a lock-free caller leaves it alone.
                    return
                else:
                    target = store
                    self._token_store = None
                    self._token_store_pending = False
            else:
                target = store
                self._token_store = None
                self._token_store_pending = False
        self._sweep_token_stores(remove = target)

    def _ensure_subprocess_alive(self) -> bool:
        """Check if subprocess is alive, reaping it if it is not.

        Every caller that cares whether the worker is up comes through here, which makes it
        the one place a worker that died on its own is noticed. Reaping from here rather than
        from each caller is deliberate: the worker has more exits than any one branch sees
        (crash mid-wait, crash while idle, cancellation, a failed load), and its private token
        store must not outlive it on any of them.
        """
        if self._proc is not None and self._proc.is_alive():
            return True
        self._reap_dead_worker()
        return False

    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: dict) -> None:
        """Send a command to the subprocess."""
        if self._cmd_queue is None:
            raise RuntimeError("No export subprocess running")
        try:
            self._cmd_queue.put(cmd)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to send command to subprocess: {exc}")

    def _read_resp(self, timeout: float = 1.0) -> Optional[dict]:
        """Read a response from the subprocess (non-blocking with timeout)."""
        if self._resp_queue is None:
            return None
        try:
            return self._resp_queue.get(timeout = timeout)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            return None

    def _wait_response(
        self,
        expected_type: str,
        timeout: float = 3600.0,
    ) -> dict:
        """Block until a response of the expected type arrives.

        Export ops can take a long time — GGUF conversion for large
        models (30B+) easily takes 20-30 minutes. Default timeout 1 hour.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resp = self._read_resp(timeout = min(remaining, 2.0))

            if resp is None:
                if not self._ensure_subprocess_alive():
                    # The caller catches this and returns without shutting down, so the
                    # dead worker's private token store has to be reaped here.
                    self._reap_dead_worker()
                    raise RuntimeError("Export subprocess crashed during wait")
                continue

            rtype = resp.get("type", "")

            if rtype == expected_type:
                return resp

            if rtype == "error":
                error_msg = resp.get("error", "Unknown error")
                raise RuntimeError(f"Subprocess error: {error_msg}")

            if rtype == "log":
                # Forwarded stdout/stderr line from the worker.
                self._append_log(resp)
                continue

            if rtype == "status":
                message = resp.get("message", "")
                # One structured export_progress line per phase (consolidated in the server log, like
                # training/download progress); also shown live.
                if message:
                    logger.info("export_progress", phase = message)
                    self._append_log(
                        {
                            "stream": "status",
                            "line": message,
                            "ts": resp.get("ts", time.time()),
                        }
                    )
                continue

            # Other response types during wait - skip.
            logger.debug(
                "Skipping response type '%s' while waiting for '%s'",
                rtype,
                expected_type,
            )

        raise RuntimeError(f"Timeout waiting for '{expected_type}' response after {timeout}s")

    def _drain_queue(self) -> list:
        """Drain all pending responses."""
        events = []
        if self._resp_queue is None:
            return events
        while True:
            try:
                events.append(self._resp_queue.get_nowait())
            except queue.Empty:
                return events
            except (EOFError, OSError, ValueError):
                return events

    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        approved_remote_code_fingerprint: Optional[str] = None,
        hf_token: HfTokenArg = None,
        allow_ambient: bool = True,
        subject: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load a checkpoint for export.

        Always spawns a fresh subprocess to ensure a clean Python interpreter.
        """
        sub_config = {
            "checkpoint_path": checkpoint_path,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "trust_remote_code": trust_remote_code,
            "approved_remote_code_fingerprint": approved_remote_code_fingerprint,
            "subject": subject,
            "hf_token": hf_token,
            "allow_ambient": allow_ambient,
            # Filled in below, once the old worker is gone: shutting it down discards its
            # store, which would take a store allocated here with it.
            "hf_token_store": None,
        }

        with self._lock:
            # Fresh log buffer so the UI sees only this run's output.
            self.clear_logs()
            self._cancel_requested = False
            self._active_op_kind = "load_checkpoint"
            self._export_active = True
            op_success, op_message = False, ""
            try:
                # Handshake with the sidecar install route (see load_checkpoint): either this recheck refuses
                # before tearing down the old worker, or the install sees is_export_active() and 409s.
                from utils.transformers_version import sidecar_swap_in_progress

                if sidecar_swap_in_progress():
                    from utils.transformers_version import SidecarSwapInProgress
                    op_message = (
                        "A transformers installation is replacing the latest "
                        "sidecar; retry when it completes."
                    )
                    raise SidecarSwapInProgress(op_message)
                # Always kill any existing subprocess and spawn fresh.
                if self._ensure_subprocess_alive():
                    if self._shutdown_subprocess() is False:
                        # A survivor still holds GPU memory (a wedged CUDA syscall outliving SIGKILL) and its handle is
                        # kept so is_worker_alive() still sees it, so do not spawn a second worker over it; fail so the
                        # load can retry once it exits.
                        op_message = (
                            "The current export worker did not exit and still holds GPU "
                            "memory; not starting a new checkpoint load over it. Retry shortly."
                        )
                        return False, op_message
                elif self._proc is not None:
                    self._shutdown_subprocess(timeout = 2)

                # Owned by this process, not the worker: a cancel goes through terminate()
                # and then kill(), neither of which runs the child's atexit, and the loader
                # may have persisted the caller's token into it by then.
                if not allow_ambient:
                    sub_config["hf_token_store"] = self._new_token_store()

                logger.info("Spawning fresh export subprocess for '%s'", checkpoint_path)
                try:
                    self._spawn_subprocess(sub_config)
                    self._attach_token_store()
                except Exception:
                    # A stale current_checkpoint would make the Export page claim a loaded checkpoint the next op then
                    # fails on.
                    self.current_checkpoint = None
                    self.is_vision = False
                    self.is_peft = False
                    raise

                try:
                    resp = self._wait_response("loaded")
                except RuntimeError as exc:
                    self._shutdown_subprocess(timeout = 5)
                    self.current_checkpoint = None
                    self.is_vision = False
                    self.is_peft = False
                    op_success, op_message = False, str(exc)
                    return False, str(exc)

                if resp.get("success"):
                    self.current_checkpoint = resp.get("checkpoint")
                    self.is_vision = resp.get("is_vision", False)
                    self.is_peft = resp.get("is_peft", False)
                    logger.info("Checkpoint '%s' loaded in subprocess", checkpoint_path)
                    op_success, op_message = True, resp.get("message", "Loaded successfully")
                    return True, op_message
                else:
                    error = resp.get("message", "Failed to load checkpoint")
                    logger.error("Failed to load checkpoint: %s", error)
                    # A failed load leaves the worker alive holding nothing useful, and
                    # unsloth may already have persisted the caller's token into its private
                    # store. Retire both rather than leaving the credential until the next
                    # load; the next one spawns a fresh worker anyway.
                    self._shutdown_subprocess(timeout = 5)
                    self.current_checkpoint = None
                    self.is_vision = False
                    self.is_peft = False
                    op_success, op_message = False, error
                    return False, error
            finally:
                self._record_op_finished(op_success, op_message, None)
                self._active_op_kind = None
                self._export_active = False

    def export_merged_model(
        self,
        save_directory: str,
        format_type: str = "16-bit (FP16)",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        private: bool = False,
        compressed_method: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Export merged PEFT model."""
        return self._run_export(
            "merged",
            {
                "save_directory": save_directory,
                "format_type": format_type,
                "push_to_hub": push_to_hub,
                "repo_id": repo_id,
                "hf_token": hf_token,
                "private": private,
                "compressed_method": compressed_method,
            },
        )

    def export_base_model(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        private: bool = False,
        base_model_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Export base model (non-PEFT)."""
        return self._run_export(
            "base",
            {
                "save_directory": save_directory,
                "push_to_hub": push_to_hub,
                "repo_id": repo_id,
                "hf_token": hf_token,
                "private": private,
                "base_model_id": base_model_id,
            },
        )

    def export_gguf(
        self,
        save_directory: str,
        quantization_method = "Q4_K_M",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        imatrix_file = None,
        private: bool = False,
        gguf_shard_size: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Export model in GGUF format. `quantization_method` may be a single method or a list."""
        return self._run_export(
            "gguf",
            {
                "save_directory": save_directory,
                "quantization_method": quantization_method,
                "push_to_hub": push_to_hub,
                "repo_id": repo_id,
                "hf_token": hf_token,
                "imatrix_file": imatrix_file,
                "private": private,
                "gguf_shard_size": gguf_shard_size,
            },
        )

    def export_lora_adapter(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        private: bool = False,
        gguf: bool = False,
        gguf_outtype: str = "q8_0",
    ) -> Tuple[bool, str, Optional[str]]:
        """Export LoRA adapter only (optionally also as a GGUF LoRA file)."""
        return self._run_export(
            "lora",
            {
                "save_directory": save_directory,
                "push_to_hub": push_to_hub,
                "repo_id": repo_id,
                "hf_token": hf_token,
                "private": private,
                "gguf": gguf,
                "gguf_outtype": gguf_outtype,
            },
        )

    def _run_export(self, export_type: str, params: dict) -> Tuple[bool, str, Optional[str]]:
        """Send an export command and wait for the result.

        Returns ``(success, message, output_path)``. ``output_path`` is the on-disk
        dir the worker wrote to (None if it only pushed to Hub or failed pre-write).
        """
        with self._lock:
            if not self._ensure_subprocess_alive():
                return (
                    False,
                    "No export subprocess running. Load a checkpoint first.",
                    None,
                )

            self.clear_logs()
            self._cancel_requested = False
            self._active_op_kind = f"export_{export_type}"
            self._export_active = True
            op_success, op_message, op_output_path = False, "", None
            try:
                # Recheck before sending, else an install blocks in cleanup_memory behind a long export op.
                from utils.transformers_version import sidecar_swap_in_progress

                if sidecar_swap_in_progress():
                    from utils.transformers_version import SidecarSwapInProgress
                    op_message = (
                        "A transformers installation is replacing the latest "
                        "sidecar; retry when it completes."
                    )
                    raise SidecarSwapInProgress(op_message)
                cmd = {"type": "export", "export_type": export_type, **params}
                try:
                    self._send_cmd(cmd)
                    # GGUF for a 30B+ model can take 30+ min per quant, and a multi-quant list runs every quant in one
                    # op off a single merge, so scale the timeout by quant count.
                    _qm = params.get("quantization_method")
                    _n = len(_qm) if isinstance(_qm, (list, tuple)) and _qm else 1
                    resp = self._wait_response(
                        f"export_{export_type}_done",
                        timeout = 3600 * max(1, _n),
                    )
                    op_success = resp.get("success", False)
                    op_message = resp.get("message", "")
                    op_output_path = resp.get("output_path")
                    return op_success, op_message, op_output_path
                except RuntimeError as exc:
                    op_success, op_message = False, str(exc)
                    return False, str(exc), None
            finally:
                self._record_op_finished(op_success, op_message, op_output_path)
                self._active_op_kind = None
                self._export_active = False

    def cleanup_memory(self) -> bool:
        """Cleanup export-related models from memory."""
        with self._lock:
            if not self._ensure_subprocess_alive():
                self.current_checkpoint = None
                self.is_vision = False
                self.is_peft = False
                return True

            self._active_op_kind = "cleanup"
            self._export_active = True
            success = False
            try:
                try:
                    self._send_cmd({"type": "cleanup"})
                    resp = self._wait_response("cleanup_done", timeout = 30)
                    success = resp.get("success", False)
                except RuntimeError:
                    success = False

                # Shut down subprocess after cleanup - no model loaded.
                self._shutdown_subprocess()

                self.current_checkpoint = None
                self.is_vision = False
                self.is_peft = False
                return success
            finally:
                self._record_op_finished(success, "", None)
                self._active_op_kind = None
                self._export_active = False

    def scan_checkpoints(self, outputs_dir: str = str(outputs_root())) -> List[Tuple[str, list]]:
        """Scan for checkpoints — runs locally, no ML imports."""
        from utils.models.checkpoints import scan_checkpoints
        return scan_checkpoints(outputs_dir = outputs_dir)


# ========== GLOBAL INSTANCE ==========
_export_backend = None


def get_export_backend() -> ExportOrchestrator:
    """Get global export backend instance (orchestrator)."""
    global _export_backend
    if _export_backend is None:
        _export_backend = ExportOrchestrator()
    return _export_backend
