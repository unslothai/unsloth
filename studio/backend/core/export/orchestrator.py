# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Export orchestrator — subprocess-based.

Provides the same API as ExportBackend, but delegates all ML work
to a persistent subprocess. The subprocess is spawned on first checkpoint
load and stays alive for subsequent export operations.

When switching between checkpoints that need different transformers versions,
the old subprocess is killed and a new one is spawned with the correct version.

Pattern follows core/inference/orchestrator.py.
"""

import atexit
import structlog
from loggers import get_logger
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple
from utils.paths import outputs_root

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")


class ExportOrchestrator:
    """
    Export backend orchestrator — subprocess-based.

    Exposes the same API surface as ExportBackend so routes/export.py
    needs minimal changes. Internally, all heavy ML operations happen in
    a persistent subprocess.
    """

    def __init__(self):
        # Subprocess state
        self._proc: Optional[mp.Process] = None
        self._cmd_queue: Any = None
        self._resp_queue: Any = None
        self._lock = threading.Lock()

        # Local state mirrors (updated from subprocess responses)
        self.current_checkpoint: Optional[str] = None
        self.is_vision: bool = False
        self.is_peft: bool = False

        atexit.register(self._cleanup)
        logger.info("ExportOrchestrator initialized (subprocess mode)")

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _spawn_subprocess(self, config: dict) -> None:
        """Spawn a new export subprocess."""
        from .worker import run_export_process

        self._cmd_queue = _CTX.Queue()
        self._resp_queue = _CTX.Queue()

        self._proc = _CTX.Process(
            target = run_export_process,
            kwargs = {
                "cmd_queue": self._cmd_queue,
                "resp_queue": self._resp_queue,
                "config": config,
            },
            daemon = True,
        )
        self._proc.start()
        logger.info("Export subprocess started (pid=%s)", self._proc.pid)

    def _shutdown_subprocess(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the export subprocess."""
        if self._proc is None or not self._proc.is_alive():
            self._proc = None
            return

        # 1. Drain stale responses
        self._drain_queue()

        # 2. Send shutdown command
        try:
            self._cmd_queue.put({"type": "shutdown"})
        except (OSError, ValueError):
            pass

        # 3. Wait for graceful shutdown
        try:
            self._proc.join(timeout = timeout)
        except Exception:
            pass

        # 4. Force kill if still alive
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

        self._proc = None
        self._cmd_queue = None
        self._resp_queue = None
        logger.info("Export subprocess shut down")

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

    def _wait_response(self, expected_type: str, timeout: float = 3600.0) -> dict:
        """Block until a response of the expected type arrives.

        Export operations can take a very long time — GGUF conversion for
        large models (30B+) easily takes 20-30 minutes. Default timeout
        is 1 hour.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resp = self._read_resp(timeout = min(remaining, 2.0))

            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    raise RuntimeError("Export subprocess crashed during wait")
                continue

            rtype = resp.get("type", "")

            if rtype == expected_type:
                return resp

            if rtype == "error":
                error_msg = resp.get("error", "Unknown error")
                raise RuntimeError(f"Subprocess error: {error_msg}")

            if rtype == "status":
                logger.info("Export subprocess status: %s", resp.get("message", ""))
                continue

            # Other response types during wait — skip
            logger.debug(
                "Skipping response type '%s' while waiting for '%s'",
                rtype,
                expected_type,
            )

        raise RuntimeError(
            f"Timeout waiting for '{expected_type}' response after {timeout}s"
        )

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
        hf_token: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load a checkpoint for export.

        Always spawns a fresh subprocess to ensure a clean Python interpreter.
        """
        sub_config = {
            "checkpoint_path": checkpoint_path,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "trust_remote_code": trust_remote_code,
            "hf_token": hf_token,
        }

        # Always kill existing subprocess and spawn fresh.
        if self._ensure_subprocess_alive():
            self._shutdown_subprocess()
        elif self._proc is not None:
            self._shutdown_subprocess(timeout = 2)

        logger.info("Spawning fresh export subprocess for '%s'", checkpoint_path)
        self._spawn_subprocess(sub_config)

        try:
            resp = self._wait_response("loaded", timeout = 300)
        except RuntimeError as exc:
            self._shutdown_subprocess(timeout = 5)
            self.current_checkpoint = None
            self.is_vision = False
            self.is_peft = False
            return False, str(exc)

        if resp.get("success"):
            self.current_checkpoint = resp.get("checkpoint")
            self.is_vision = resp.get("is_vision", False)
            self.is_peft = resp.get("is_peft", False)
            logger.info("Checkpoint '%s' loaded in subprocess", checkpoint_path)
            return True, resp.get("message", "Loaded successfully")
        else:
            error = resp.get("message", "Failed to load checkpoint")
            logger.error("Failed to load checkpoint: %s", error)
            self.current_checkpoint = None
            self.is_vision = False
            self.is_peft = False
            return False, error

    def export_merged_model(
        self,
        save_directory: str,
        format_type: str = "16-bit (FP16)",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: bool = False,
    ) -> Tuple[bool, str]:
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
    ) -> Tuple[bool, str]:
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
        quantization_method: List[str],
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Export model in GGUF format. The caller must supply a normalized
        list of lowercase quantization method strings (see
        `models.export.normalize_gguf_quantization_method`).
        """
        return self._run_export(
            "gguf",
            {
                "save_directory": save_directory,
                "quantization_method": quantization_method,
                "push_to_hub": push_to_hub,
                "repo_id": repo_id,
                "hf_token": hf_token,
            },
        )

    def export_lora_adapter(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: bool = False,
    ) -> Tuple[bool, str]:
        """Export LoRA adapter only."""
        return self._run_export(
            "lora",
            {
                "save_directory": save_directory,
                "push_to_hub": push_to_hub,
                "repo_id": repo_id,
                "hf_token": hf_token,
                "private": private,
            },
        )

    def _run_export(self, export_type: str, params: dict) -> Tuple[bool, str]:
        """Send an export command to the subprocess and wait for result."""
        if not self._ensure_subprocess_alive():
            return False, "No export subprocess running. Load a checkpoint first."

        cmd = {"type": "export", "export_type": export_type, **params}

        try:
            self._send_cmd(cmd)
            resp = self._wait_response(
                f"export_{export_type}_done",
                timeout = 3600,  # GGUF for 30B+ models can take 30+ min
            )
            return resp.get("success", False), resp.get("message", "")
        except RuntimeError as exc:
            return False, str(exc)

    def cleanup_memory(self) -> bool:
        """Cleanup export-related models from memory."""
        if not self._ensure_subprocess_alive():
            # No subprocess — just clear local state
            self.current_checkpoint = None
            self.is_vision = False
            self.is_peft = False
            return True

        try:
            self._send_cmd({"type": "cleanup"})
            resp = self._wait_response("cleanup_done", timeout = 30)
            success = resp.get("success", False)
        except RuntimeError:
            success = False

        # Shut down subprocess after cleanup — no model loaded
        self._shutdown_subprocess()

        self.current_checkpoint = None
        self.is_vision = False
        self.is_peft = False
        return success

    def scan_checkpoints(
        self, outputs_dir: str = str(outputs_root())
    ) -> List[Tuple[str, list]]:
        """Scan for checkpoints — no ML imports needed, runs locally."""
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
