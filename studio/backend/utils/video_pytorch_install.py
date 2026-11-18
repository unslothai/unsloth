# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Background installation of the missing video PyTorch runtime."""

import os
from pathlib import Path
import subprocess
import sys
import threading

from loggers import get_logger
from utils.child_stdio import utf8_child_env
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)
_lock = threading.Lock()
_job = {"status": "idle", "message": None}


def status():
    with _lock:
        return dict(_job)


def _missing():
    from utils.hardware.hardware import video_capability
    return video_capability().get("video_unsupported_reason") == "pytorch_not_installed"


def _installer():
    return Path(__file__).resolve().parents[2] / "install_python_stack.py"


def start():
    with _lock:
        if _job["status"] in {"installing", "restart_required"}:
            return dict(_job)
        if not _missing():
            raise RuntimeError(
                "PyTorch installation is only available when the video runtime is missing."
            )
        if not _installer().is_file():
            raise RuntimeError("The dependency installer is missing. Update Unsloth and try again.")
        _job.update(
            status = "installing", message = "Installing PyTorch. This may take several minutes."
        )
        try:
            threading.Thread(target = _run, name = "video-pytorch-install", daemon = True).start()
        except Exception:
            _job.update(status = "failed", message = "Could not start the installer. Try again.")
            raise
        return dict(_job)


def _run():
    try:
        env = utf8_child_env()
        env["UNSLOTH_NO_TORCH"] = "0"
        env["VIRTUAL_ENV"] = sys.prefix
        # GUI launches may not put uv on PATH; use the same lookup as MLX repair.
        from utils.mlx_repair import _uv_executable

        uv = _uv_executable()
        if uv:
            env["PATH"] = str(Path(uv).parent) + os.pathsep + env.get("PATH", "")
        # Inherit server output so download/installer diagnostics land in its logs;
        # never return package-manager output (which can contain mirror credentials).
        result = subprocess.run(
            [sys.executable, "-u", str(_installer()), "--install-video-pytorch"],
            env = env,
            **windows_hidden_subprocess_kwargs(),
        )
        if result.returncode:
            raise RuntimeError(f"PyTorch installer exited with status {result.returncode}")
        state = {
            "status": "restart_required",
            "message": "PyTorch installed. Restart Unsloth to enable video generation.",
        }
    except Exception:
        logger.exception("Video PyTorch installation failed")
        state = {
            "status": "failed",
            "message": "PyTorch installation failed. Check the server logs, then try again.",
        }
    with _lock:
        _job.update(state)
