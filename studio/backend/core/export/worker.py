# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export subprocess entry point.

Each export session runs in a persistent subprocess (mp spawn), giving a clean
interpreter with no stale module state, which solves transformers version
switching. The subprocess stays alive while a model is loaded, accepting commands
(load, export_*, cleanup, shutdown) via mp.Queue.

Pattern follows core/inference/worker.py and core/training/worker.py.
"""

from __future__ import annotations

import contextlib
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


# Gate controlling whether captured stdout/stderr lines are forwarded to the
# parent's resp_queue (and on to the export-dialog SSE stream). Closed by default
# so the noisy bootstrap phase (imports, model resolution, loading bars) is
# suppressed in the UI; _handle_export() opens it when export work starts. The
# orchestrator spawns a fresh subprocess per checkpoint load, resetting this.
# Dropped lines are still echoed to the saved fds so the server log keeps them.
_log_forward_gate = threading.Event()


def _setup_log_capture(resp_queue: Any) -> None:
    """Redirect fds 1 and 2 through pipes so every line printed by this worker
    and any child it spawns is forwarded to the parent via resp_queue as
    {"type": "log", ...} messages.

    Must run BEFORE LogConfig.setup_logging and any ML imports, else library
    handlers may capture the original stderr reference and bypass the pipe.
    Lines are also echoed back to the original fds so the server console keeps
    the full output even while ``_log_forward_gate`` is closed.
    """

    try:
        saved_out_fd = os.dup(1)
        saved_err_fd = os.dup(2)
    except OSError:
        # dup failed; give up quietly (export still works, no live streaming).
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

    # Close the write ends we just dup2'd (fds 1 and 2 are the real write ends).
    os.close(w_out)
    os.close(w_err)

    # Replace sys.stdout/sys.stderr with line-buffered writers on fds 1 and 2.
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
            # Echo to the original fd so the server console keeps the full output.
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
                    # Gate closed (bootstrap): already echoed above; drop the
                    # line so the export dialog skips import noise.
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
                    # Queue put failed; drop the line rather than crash the thread.
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


def _activate_transformers_version(model_name: str, hf_token: str | None = None) -> None:
    """Activate the correct transformers version BEFORE any ML imports."""
    # Ensure backend is on sys.path for utils imports.
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import activate_transformers_for_subprocess

    activate_transformers_for_subprocess(model_name, hf_token)


def _reset_hf_sessions() -> None:
    try:
        from huggingface_hub.utils._http import reset_sessions
    except Exception:
        try:
            from huggingface_hub.utils import reset_sessions
        except Exception:
            return
    try:
        reset_sessions()
    except Exception:
        pass


@contextlib.contextmanager
def _force_hf_offline_window():
    saved_env = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    saved_attrs = []
    try:
        import huggingface_hub.constants as hub_constants
        if hasattr(hub_constants, "HF_HUB_OFFLINE"):
            saved_attrs.append(
                (
                    hub_constants,
                    "HF_HUB_OFFLINE",
                    hub_constants.HF_HUB_OFFLINE,
                )
            )
    except Exception:
        pass
    try:
        import transformers.utils.hub as transformers_hub
        for attr in ("_is_offline_mode", "OFFLINE"):
            if hasattr(transformers_hub, attr):
                saved_attrs.append((transformers_hub, attr, getattr(transformers_hub, attr)))
    except Exception:
        pass

    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        for obj, attr, _ in saved_attrs:
            try:
                setattr(obj, attr, True)
            except Exception:
                pass
        _reset_hf_sessions()
        yield
    finally:
        for obj, attr, value in saved_attrs:
            try:
                setattr(obj, attr, value)
            except Exception:
                pass
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset_hf_sessions()


@contextlib.contextmanager
def _offline_window_if_unreachable(step = "loading"):
    """Force HF offline for a network-touching step (transformers version activation, or the
    load preflights that hit the Hub) when the endpoint is unreachable, then restore the prior
    env. Keeps a no-network export from hanging on Hub calls that run before load_checkpoint's
    own probe, while letting this persistent worker re-decide per operation once back online.

    Post-ML-import (the load preflights), huggingface_hub has already read its in-process
    offline constant and cached sessions, so env alone is too late: defer to the loader's
    _force_hf_offline (env + in-process flags + session reset). Pre-import (activation),
    huggingface_hub is not loaded yet, so setting the env vars suffices for its urllib probes."""
    saved: dict[str, str | None] = {}
    force_ctx = None
    try:
        from utils.transformers_version import _env_offline, hf_endpoint_unreachable

        probe_enabled = os.environ.get("UNSLOTH_OFFLINE_PROBE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        should_force = _env_offline()
        if not should_force and probe_enabled and hf_endpoint_unreachable():
            should_force = True
            logger.warning("Hugging Face endpoint unreachable; %s offline", step)
        if should_force:
            if "huggingface_hub" in sys.modules or "transformers" in sys.modules:
                try:
                    force_ctx = _force_hf_offline_window()
                    force_ctx.__enter__()  # sets env + in-process flags + resets sessions
                except Exception:
                    force_ctx = None
            if force_ctx is None:
                for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                    saved[k] = os.environ.get(k)
                    os.environ[k] = "1"
    except Exception:
        pass
    try:
        yield
    finally:
        if force_ctx is not None:
            try:
                force_ctx.__exit__(None, None, None)
            except Exception:
                pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
    # Latest-sidecar checkpoints load 16-bit here too: bnb 4-bit feeds quantized
    # expert weights into unvalidated paths (same flip as the chat worker).
    if load_in_4bit:
        from utils.transformers_version import latest_tier_active_for
        if latest_tier_active_for(checkpoint_path, cmd.get("hf_token")):
            load_in_4bit = False
            logger.info(
                "Latest-transformers sidecar active for %s - forcing a 16-bit "
                "export load (4-bit is disabled for brand-new architectures)",
                checkpoint_path,
            )
    trust_remote_code = cmd.get("trust_remote_code", False)

    # Auto-enable trust_remote_code for NemotronH/Nano models.
    if not trust_remote_code:
        from utils.security.trusted_org import is_trusted_org_repo

        _NEMOTRON_TRUST_SUBSTRINGS = ("nemotron_h", "nemotron-h", "nemotron-3-nano")
        _cp_lower = checkpoint_path.lower()
        if (
            any(sub in _cp_lower for sub in _NEMOTRON_TRUST_SUBSTRINGS)
            and (_cp_lower.startswith("unsloth/") or _cp_lower.startswith("nvidia/"))
            # Genuine first-party Hub repo only (not a local/spoof name starting
            # with "unsloth/"); authenticated so private repos resolve.
            and is_trusted_org_repo(checkpoint_path, hf_token = cmd.get("hf_token"))
        ):
            trust_remote_code = True
            logger.info(
                "Auto-enabled trust_remote_code for Nemotron model: %s",
                checkpoint_path,
            )

    # Malware gate: a poisoned pickle deserializes on load even with
    # trust_remote_code False, so check HF's security scan (metadata-only) every
    # load. Local checkpoints have no Hub scan and are skipped in the helper; a
    # LoRA merges its base weights, so gate that repo too.
    from utils.security import evaluate_file_security, security_load_subdirs

    malware_targets = [checkpoint_path]
    try:
        from utils.models.model_config import get_base_model_from_lora_identifier

        # Resolve a LOCAL or REMOTE adapter's base so a remote LoRA base is gated too.
        _base = get_base_model_from_lora_identifier(checkpoint_path, cmd.get("hf_token"))
        if _base:
            malware_targets.append(_base)
    except Exception as exc:
        logger.debug("Could not resolve LoRA base for malware scan: %s", exc)
    _hf_token = cmd.get("hf_token")
    for target in dict.fromkeys(malware_targets):
        _fs = evaluate_file_security(
            target, hf_token = _hf_token, load_subdirs = security_load_subdirs(target, _hf_token)
        )
        if _fs.blocked:
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": False,
                    "message": _fs.reason,
                    "error_kind": "malware_blocked",
                    "security": _fs.response_payload(),
                    "ts": time.time(),
                },
            )
            return

    # Consent gate: scan auto_map code before it runs; block CRITICAL/HIGH unless
    # pinned-approved. A LoRA merges its base model, whose code runs, so gate it too.
    if trust_remote_code:
        from utils.security import evaluate_remote_code_consent_for_targets

        consent_targets = [checkpoint_path]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a local or remote adapter's base so its base repo is gated too.
            base_model = get_base_model_from_lora_identifier(checkpoint_path, cmd.get("hf_token"))
            if base_model:
                consent_targets.append(base_model)
        except Exception as exc:
            logger.debug("Could not resolve LoRA base for consent scan: %s", exc)
        # Scan adapter + base as one combined unit, pinned by a single fingerprint.
        _rc = evaluate_remote_code_consent_for_targets(
            consent_targets,
            hf_token = cmd.get("hf_token"),
            trust_remote_code = True,
            approved_fingerprint = cmd.get("approved_remote_code_fingerprint"),
            subject = cmd.get("subject"),
        )
        if _rc.blocked:
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": False,
                    "message": (
                        f"Checkpoint '{_rc.model_name}' ships custom code flagged as "
                        f"{_rc.max_severity} by the security scan. Review and "
                        f"approve it to proceed."
                    ),
                    "error_kind": "remote_code_blocked",
                    "remote_code": _rc.response_payload(),
                    "ts": time.time(),
                },
            )
            return

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
            hf_token = cmd.get("hf_token"),
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

    # Open the log forwarding gate so the user sees export progress in the live
    # log panel. Stays open for the rest of this subprocess's life; the
    # orchestrator spawns a fresh subprocess per checkpoint load, resetting it.
    _log_forward_gate.set()

    # Phase milestone so the heavy export step shows in the server log; the
    # merge/save/convert itself only forwards stdout to the live panel.
    _phase = {
        "merged": f"Exporting merged model ({cmd.get('format_type', '16-bit (FP16)')})...",
        "gguf": f"Exporting GGUF ({cmd.get('quantization_method', 'Q4_K_M')})...",
        "lora": "Exporting LoRA adapter...",
        "base": "Exporting base model...",
    }.get(export_type, f"Exporting ({export_type})...")
    _send_response(
        resp_queue,
        {"type": "status", "message": _phase, "ts": time.time()},
    )

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
                compressed_method = cmd.get("compressed_method"),
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
                imatrix_file = cmd.get("imatrix_file"),
            )
        elif export_type == "lora":
            success, message, output_path = backend.export_lora_adapter(
                save_directory = cmd.get("save_directory", ""),
                push_to_hub = cmd.get("push_to_hub", False),
                repo_id = cmd.get("repo_id"),
                hf_token = cmd.get("hf_token"),
                private = cmd.get("private", False),
                gguf = cmd.get("gguf", False),
                gguf_outtype = cmd.get("gguf_outtype", "q8_0"),
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


def run_export_process(*, cmd_queue: Any, resp_queue: Any, config: dict) -> None:
    """Subprocess entrypoint. Persistent — runs command loop until shutdown.

    Args:
        cmd_queue: mp.Queue for receiving commands from parent.
        resp_queue: mp.Queue for sending responses to parent.
        config: Initial configuration dict with checkpoint_path.
    """
    import queue as _queue

    # Install fd-level stdout/stderr capture FIRST so every subsequent print and
    # every child process inherits the redirected fds (powers the live log stream).
    _setup_log_capture(resp_queue)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"  # suppress C-level warnings before imports
    # Unbuffered output from child Python (e.g. GGUF converter) so prints surface live.
    os.environ["PYTHONUNBUFFERED"] = "1"
    # tqdm defaults to a 10s mininterval when stdout isn't a tty (we redirected
    # fd 1/2 to a pipe), making multi-step bars look frozen; force frequent flushes.
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
    with _offline_window_if_unreachable(step = "activating transformers"):
        try:
            _activate_transformers_version(checkpoint_path, config.get("hf_token") or None)
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

    # ── 1b. Check Triton on Windows (must precede import torch) ──
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

    # ── 1c. Stub torchao on Windows ROCm ──
    # See core/_torchao_stub.py: torchao crashes on Windows ROCm (RCCL absent).
    # No-op off Windows ROCm. Must run before importing transformers / unsloth_zoo.
    from core._torchao_stub import install_torchao_windows_rocm_stub

    install_torchao_windows_rocm_stub()

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

        # Recover from any namespace-package shadow before importing Unsloth.
        from core.import_guards import ensure_real_packages

        ensure_real_packages("unsloth_zoo", "unsloth")

        from core.export.export import ExportBackend

        import transformers

        logger.info("Export subprocess loaded transformers %s", transformers.__version__)

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

        # Offline window covers the load preflights (malware/consent scans hit the Hub)
        # before load_checkpoint runs its own probe; restored after so later loads re-decide.
        with _offline_window_if_unreachable():
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
                # Load a new checkpoint, reusing this subprocess.
                backend.cleanup_memory()
                # Offline window also covers this load's Hub preflights (re-probed per load).
                with _offline_window_if_unreachable():
                    _handle_load(backend, cmd, resp_queue)

            elif cmd_type == "export":
                # Export can trigger hidden Hub metadata calls from tokenizer save paths.
                with _offline_window_if_unreachable(step = "exporting"):
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
            logger.error("Error handling command '%s': %s", cmd_type, exc, exc_info = True)
            _send_response(
                resp_queue,
                {
                    "type": "error",
                    "error": f"Command '{cmd_type}' failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                },
            )
