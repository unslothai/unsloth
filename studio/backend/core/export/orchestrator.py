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
from typing import Any, Deque, Dict, List, Optional, Tuple
from utils.paths import outputs_root

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")

# Max log lines kept per orchestrator (live log panel scrollback); ~1 MB worst-case.
_LOG_BUFFER_MAXLEN = 4000


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
        # _log_seq snapshot at the current run's start; SSE defaults its cursor here so a
        # late-connecting client still sees the full run. Current run has seq > this.
        self._run_start_seq: int = 0
        # True while an export op runs; SSE ends the stream 1s after this flips False.
        self._export_active: bool = False
        # Set by cancel_export(); reset when a new load/export run starts. Lets the
        # caller distinguish a user cancel from a genuine subprocess crash.
        self._cancel_requested: bool = False

        # Last finished operation, so a client whose blocking POST was cut off by a
        # Cloudflare tunnel timeout (524 at ~100s, while the op runs for minutes) can
        # poll /api/export/status and still learn the real outcome. Guarded by
        # _op_lock. `_op_seq` is a monotonic counter the client uses as a baseline to
        # tell "my op finished" (seq grew) from a stale previous result.
        self._op_lock = threading.Lock()
        self._op_seq: int = 0
        self._active_op_kind: Optional[str] = None
        self._last_op: Optional[Dict[str, Any]] = None

        atexit.register(self._cleanup)
        logger.info("ExportOrchestrator initialized (subprocess mode)")

    # ------------------------------------------------------------------
    # Live log capture helpers
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
        """True while the persistent export subprocess is running (op or idle)."""
        proc = self._proc
        return proc is not None and proc.is_alive()

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
        if proc is None or not proc.is_alive():
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
        return True

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _spawn_subprocess(self, config: dict) -> None:
        """Spawn a new export subprocess."""
        # Last-resort recheck for spawns outside an active op. Inside an op, _export_active is set and
        # load_checkpoint already rechecked, so a reservation here is an install about to observe
        # is_export_active() and abort; raising would kill this export for an install that never proceeds.
        from utils.transformers_version import sidecar_swap_in_progress

        from utils.transformers_version import sidecar_swap_kind

        _swap_kind = sidecar_swap_kind()
        # Inside an active op an INSTALL reservation is about to abort on the
        # is_export_active check, but a lazy REPAIR has no such check and can be
        # rebuilding the sidecar right now, so it must always refuse the spawn.
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

        adopt_pid(self._proc.pid)  # bind to parent lifetime (Windows job / sweep)
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
            # Survived SIGKILL (uninterruptible syscall): keep the handle so callers
            # and the pre-swap guard see a live worker rather than a nulled one.
            logger.error(
                "Export subprocess still alive after terminate/kill; "
                "preserving its handle for the pre-swap liveness check"
            )
            return False

        self._proc = None
        self._cmd_queue = None
        self._resp_queue = None
        logger.info("Export subprocess shut down")
        return True

    def _cleanup(self):
        """atexit handler."""
        self._shutdown_subprocess(timeout = 5.0)

    def _ensure_subprocess_alive(self) -> bool:
        """Check if subprocess is alive."""
        return self._proc is not None and self._proc.is_alive()

    # ------------------------------------------------------------------
    # Queue helpers
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
                # One structured export_progress line per phase (consolidated in the
                # server log, like training/download progress); also shown live.
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

            # Other response types during wait — skip.
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
    # Public API — same interface as ExportBackend
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        approved_remote_code_fingerprint: Optional[str] = None,
        hf_token: Optional[str] = None,
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
        }

        with self._lock:
            # Fresh log buffer so the UI sees only this run's output.
            self.clear_logs()
            self._cancel_requested = False
            self._active_op_kind = "load_checkpoint"
            self._export_active = True
            op_success, op_message = False, ""
            try:
                # Handshake with the sidecar install route: _export_active is set above, so either this
                # recheck refuses BEFORE tearing down the old worker (keeping the loaded checkpoint), or
                # the install sees is_export_active() and 409s. The spawn-time recheck stays as a last resort.
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
                        # Survivor still holds GPU memory (a wedged CUDA syscall outliving
                        # SIGKILL); its handle is kept so is_worker_alive() and the pre-swap
                        # guard still see it. Do not spawn a second worker over it -- fail so
                        # the load can retry once it exits.
                        op_message = (
                            "The current export worker did not exit and still holds GPU "
                            "memory; not starting a new checkpoint load over it. Retry shortly."
                        )
                        return False, op_message
                elif self._proc is not None:
                    self._shutdown_subprocess(timeout = 2)

                logger.info("Spawning fresh export subprocess for '%s'", checkpoint_path)
                try:
                    self._spawn_subprocess(sub_config)
                except Exception:
                    # The old worker is already gone; a stale current_checkpoint
                    # would make the Export page claim a loaded checkpoint that
                    # the next op then fails on with "no subprocess running".
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
        hf_token: Optional[str] = None,
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
        hf_token: Optional[str] = None,
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
        hf_token: Optional[str] = None,
        imatrix_file = None,
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
            },
        )

    def export_lora_adapter(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
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
                # Handshake with the sidecar install route (see load_checkpoint): _export_active is set
                # above, so this recheck refuses before the command is sent, or the install sees the active
                # op and 409s. Without it, an install would block in cleanup_memory behind a long export op.
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
                    # GGUF for 30B+ models can take 30+ min per quant; a multi-quant list runs them
                    # all in one op off a single merge, so scale the timeout by the quant count.
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

                # Shut down subprocess after cleanup — no model loaded.
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
