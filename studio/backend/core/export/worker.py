# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Export subprocess entry point.

Each export session runs in a persistent subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

The subprocess stays alive while a model is loaded, accepting commands
(load, export_merged, export_base, export_gguf, export_lora, cleanup,
shutdown) via mp.Queue.

Pattern follows core/inference/worker.py and core/training/worker.py.
"""

from __future__ import annotations

import errno
import structlog
from loggers import get_logger
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

logger = get_logger(__name__)


# Gate that controls whether captured stdout/stderr lines are forwarded
# to the parent's resp_queue (and from there to the export-dialog SSE
# stream). Closed by default so the noisy bootstrap phase -- transformers
# venv activation, Unsloth/torch imports, base-model resolution, "Top
# GGUF/hub models" lists, vision detection, weight loading bars -- is
# suppressed in the UI. _handle_export() opens the gate at the start of
# the actual export work and leaves it open; the orchestrator always
# spawns a fresh subprocess for the next checkpoint load (see
# orchestrator._spawn_subprocess) which resets this state.
#
# Lines dropped while the gate is closed are still echoed to the saved
# original stdout/stderr fds so the server console / log file keeps the
# full output for debugging.
_log_forward_gate = threading.Event()


def _setup_log_capture(resp_queue: Any) -> None:
    """Redirect fds 1 and 2 through pipes so every line printed by this
    worker process and any child process it spawns is forwarded to the
    parent process via resp_queue as {"type": "log", ...} messages.

    Must be called BEFORE LogConfig.setup_logging and BEFORE any ML
    imports, otherwise library handlers may capture the original stderr
    reference and bypass the pipe.

    Lines are also echoed back to the original stdout/stderr so the
    server console keeps receiving the full subprocess output, even
    while ``_log_forward_gate`` is closed.
    """

    try:
        saved_out_fd = os.dup(1)
        saved_err_fd = os.dup(2)
    except OSError:
        # dup failed (exotic platforms) - give up quietly, export still
        # works, just no live log streaming.
        return

    try:
        r_out, w_out = os.pipe()
        r_err, w_err = os.pipe()
    except OSError:
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        return

    try:
        os.dup2(w_out, 1)
        os.dup2(w_err, 2)
    except OSError:
        for fd in (saved_out_fd, saved_err_fd, r_out, w_out, r_err, w_err):
            try:
                os.close(fd)
            except OSError:
                pass
        return

    # Close the write ends we just dup2'd (fds 1 and 2 are the real
    # write ends now).
    os.close(w_out)
    os.close(w_err)

    # Replace Python's sys.stdout/sys.stderr with line-buffered writers
    # bound to the (now-redirected) fds 1 and 2.
    try:
        sys.stdout = os.fdopen(1, "w", buffering = 1, encoding = "utf-8", errors = "replace")
        sys.stderr = os.fdopen(2, "w", buffering = 1, encoding = "utf-8", errors = "replace")
    except Exception:
        pass

    def _reader(read_fd: int, stream_name: str, echo_fd: int) -> None:
        buf = bytearray()
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EBADF:
                    break
                continue
            if not chunk:
                break
            # Echo to the original fd so the server console still sees
            # the full output.
            try:
                os.write(echo_fd, chunk)
            except OSError:
                pass
            buf.extend(chunk)
            # Split on \n OR \r so tqdm-style progress bars update.
            while True:
                nl = -1
                for i, b in enumerate(buf):
                    if b == 0x0A or b == 0x0D:
                        nl = i
                        break
                if nl < 0:
                    break
                line = bytes(buf[:nl]).decode("utf-8", errors = "replace")
                del buf[: nl + 1]
                if not line:
                    continue
                if not _log_forward_gate.is_set():
                    # Gate closed (bootstrap phase) -- already echoed to
                    # the saved console fd above; drop the line so the
                    # export dialog doesn't see import / vendoring noise.
                    continue
                try:
                    resp_queue.put_nowait(
                        {
                            "type": "log",
                            "stream": stream_name,
                            "line": line,
                            "ts": time.time(),
                        }
                    )
                except Exception:
                    # Queue put failed (full, closed, etc.) - drop the
                    # line rather than crash the reader thread.
                    pass
        if buf and _log_forward_gate.is_set():
            try:
                resp_queue.put_nowait(
                    {
                        "type": "log",
                        "stream": stream_name,
                        "line": bytes(buf).decode("utf-8", errors = "replace"),
                        "ts": time.time(),
                    }
                )
            except Exception:
                pass

    t_out = threading.Thread(
        target = _reader,
        args = (r_out, "stdout", saved_out_fd),
        daemon = True,
        name = "export-log-stdout",
    )
    t_err = threading.Thread(
        target = _reader,
        args = (r_err, "stderr", saved_err_fd),
        daemon = True,
        name = "export-log-stderr",
    )
    t_out.start()
    t_err.start()


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports."""
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import activate_transformers_for_subprocess

    activate_transformers_for_subprocess(model_name)


def _send_response(resp_queue: Any, response: dict) -> None:
    """Send a response to the parent process."""
    try:
        resp_queue.put(response)
    except (OSError, ValueError) as exc:
        logger.error("Failed to send response: %s", exc)


def _handle_load(backend, cmd: dict, resp_queue: Any) -> None:
    """Handle a load_checkpoint command."""
    checkpoint_path = cmd["checkpoint_path"]
    max_seq_length = cmd.get("max_seq_length", 2048)
    load_in_4bit = cmd.get("load_in_4bit", True)
    trust_remote_code = cmd.get("trust_remote_code", False)

    try:
        _send_response(
            resp_queue,
            {
                "type": "status",
                "message": f"Loading checkpoint: {checkpoint_path}",
                "ts": time.time(),
            },
        )

        success, message = backend.load_checkpoint(
            checkpoint_path = checkpoint_path,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            trust_remote_code = trust_remote_code,
        )

        _send_response(
            resp_queue,
            {
                "type": "loaded",
                "success": success,
                "message": message,
                "checkpoint": checkpoint_path if success else None,
                "is_vision": backend.is_vision if success else False,
                "is_peft": backend.is_peft if success else False,
                "ts": time.time(),
            },
        )

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "loaded",
                "success": False,
                "message": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )


def _handle_export(backend, cmd: dict, resp_queue: Any) -> None:
    """Handle any export command (merged, base, gguf, lora)."""
    export_type = cmd["export_type"]  # "merged", "base", "gguf", "lora"
    response_type = f"export_{export_type}_done"

    # Open the log forwarding gate so the user sees the actual export
    # progress (Unsloth merge bars, file copies, GGUF conversion, etc.)
    # in the live log panel. The gate stays open for the rest of this
    # subprocess's life; the orchestrator spawns a fresh subprocess for
    # the next checkpoint load, which resets the gate to closed.
    _log_forward_gate.set()

    output_path: Any = None
    try:
        if export_type == "merged":
            success, message, output_path = backend.export_merged_model(
                save_directory = cmd.get("save_directory", ""),
                format_type = cmd.get("format_type", "16-bit (FP16)"),
                push_to_hub = cmd.get("push_to_hub", False),
                repo_id = cmd.get("repo_id"),
                hf_token = cmd.get("hf_token"),
                private = cmd.get("private", False),
            )
        elif export_type == "base":
            success, message, output_path = backend.export_base_model(
                save_directory = cmd.get("save_directory", ""),
                push_to_hub = cmd.get("push_to_hub", False),
                repo_id = cmd.get("repo_id"),
                hf_token = cmd.get("hf_token"),
                private = cmd.get("private", False),
                base_model_id = cmd.get("base_model_id"),
            )
        elif export_type == "gguf":
            success, message, output_path = backend.export_gguf(
                save_directory = cmd.get("save_directory", ""),
                quantization_method = cmd.get("quantization_method", "Q4_K_M"),
                push_to_hub = cmd.get("push_to_hub", False),
                repo_id = cmd.get("repo_id"),
                hf_token = cmd.get("hf_token"),
            )
        elif export_type == "lora":
            success, message, output_path = backend.export_lora_adapter(
                save_directory = cmd.get("save_directory", ""),
                push_to_hub = cmd.get("push_to_hub", False),
                repo_id = cmd.get("repo_id"),
                hf_token = cmd.get("hf_token"),
                private = cmd.get("private", False),
            )
        else:
            success, message = False, f"Unknown export type: {export_type}"

        _send_response(
            resp_queue,
            {
                "type": response_type,
                "success": success,
                "message": message,
                "output_path": output_path,
                "ts": time.time(),
            },
        )

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": response_type,
                "success": False,
                "message": str(exc),
                "output_path": None,
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )


def _handle_cleanup(backend, resp_queue: Any) -> None:
    """Handle a cleanup command."""
    try:
        success = backend.cleanup_memory()
        _send_response(
            resp_queue,
            {
                "type": "cleanup_done",
                "success": success,
                "ts": time.time(),
            },
        )
    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "cleanup_done",
                "success": False,
                "message": str(exc),
                "ts": time.time(),
            },
        )


def run_export_process(
    *,
    cmd_queue: Any,
    resp_queue: Any,
    config: dict,
) -> None:
    """Subprocess entrypoint. Persistent — runs command loop until shutdown.

    Args:
        cmd_queue: mp.Queue for receiving commands from parent.
        resp_queue: mp.Queue for sending responses to parent.
        config: Initial configuration dict with checkpoint_path.
    """
    import queue as _queue

    # Install fd-level stdout/stderr capture FIRST so every subsequent
    # print and every child process inherits the redirected fds. This
    # is what powers the live export log stream in the UI.
    _setup_log_capture(resp_queue)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )
    # Force unbuffered output from any child Python process (e.g. the
    # GGUF converter) so their prints surface in the log stream as they
    # happen rather than at the end.
    os.environ["PYTHONUNBUFFERED"] = "1"
    # tqdm defaults to a 10-second mininterval when stdout is not a tty
    # (which it isn't here -- we redirected fd 1/2 to a pipe). That makes
    # multi-step progress bars look frozen in the export log panel. Force
    # frequent flushes so the user sees movement during merge / GGUF
    # conversion. Has no effect on single-step bars (e.g. "Copying 1
    # files") which only emit start/end events regardless.
    os.environ.setdefault("TQDM_MININTERVAL", "0.5")

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-export-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    checkpoint_path = config["checkpoint_path"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(checkpoint_path)
    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )
        return

    # ── 1b. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401

            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 2. Import ML libraries (fresh in this clean process) ──
    try:
        _send_response(
            resp_queue,
            {
                "type": "status",
                "message": "Importing Unsloth...",
                "ts": time.time(),
            },
        )

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.export.export import ExportBackend

        import transformers

        logger.info(
            "Export subprocess loaded transformers %s", transformers.__version__
        )

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )
        return

    # ── 3. Create export backend and load initial checkpoint ──
    try:
        backend = ExportBackend()

        _handle_load(backend, config, resp_queue)

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to initialize export backend: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )
        return

    # ── 4. Command loop — process commands until shutdown ──
    logger.info("Export subprocess ready, entering command loop")

    while True:
        try:
            cmd = cmd_queue.get(timeout = 1.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            logger.info("Command queue closed, shutting down")
            return

        if cmd is None:
            continue

        cmd_type = cmd.get("type", "")
        logger.info("Received command: %s", cmd_type)

        try:
            if cmd_type == "load":
                # Load a new checkpoint (reusing this subprocess)
                backend.cleanup_memory()
                _handle_load(backend, cmd, resp_queue)

            elif cmd_type == "export":
                _handle_export(backend, cmd, resp_queue)

            elif cmd_type == "cleanup":
                _handle_cleanup(backend, resp_queue)

            elif cmd_type == "status":
                _send_response(
                    resp_queue,
                    {
                        "type": "status_response",
                        "checkpoint": backend.current_checkpoint,
                        "is_vision": backend.is_vision,
                        "is_peft": backend.is_peft,
                        "ts": time.time(),
                    },
                )

            elif cmd_type == "shutdown":
                logger.info("Shutdown command received, cleaning up and exiting")
                try:
                    backend.cleanup_memory()
                except Exception:
                    pass
                _send_response(
                    resp_queue,
                    {
                        "type": "shutdown_ack",
                        "ts": time.time(),
                    },
                )
                return

            else:
                logger.warning("Unknown command type: %s", cmd_type)
                _send_response(
                    resp_queue,
                    {
                        "type": "error",
                        "error": f"Unknown command type: {cmd_type}",
                        "ts": time.time(),
                    },
                )

        except Exception as exc:
            logger.error(
                "Error handling command '%s': %s", cmd_type, exc, exc_info = True
            )
            _send_response(
                resp_queue,
                {
                    "type": "error",
                    "error": f"Command '{cmd_type}' failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                },
            )
