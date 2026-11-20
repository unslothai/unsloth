# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install NVIDIA's cosmos-guardrail package into Studio's own environment on request (Cosmos3-Nano needs it).

Every distribution already installed is held at its version through a constraints file, so the install can only
add packages; a requirement that would move one fails the install instead."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

GUARDRAIL_DISTRIBUTION = "cosmos-guardrail"
_INSTALL_TIMEOUT_S = 1800

_lock = threading.Lock()
_state: dict[str, Any] = {"state": "idle", "error": None, "added": []}
_thread: Optional[threading.Thread] = None


def _uv() -> Optional[str]:
    try:
        from utils.mlx_repair import _uv_executable
        return _uv_executable()
    except Exception:  # noqa: BLE001
        return shutil.which("uv")


def _quote(parts: list[str]) -> str:
    if os.name != "nt":
        return shlex.join(parts)
    line = subprocess.list2cmdline(parts)
    # PowerShell runs a quoted executable path only through its call operator.
    return f"& {line}" if line.startswith('"') else line


def manual_install_command(uv: Any = ...) -> str:
    """The command a user runs to install the guardrail into this Studio's environment (uv, else pip)."""
    if uv is ...:
        uv = _uv()
    if uv:
        # The bare name when it is the one on PATH, so the command reads as users expect.
        exe = "uv" if shutil.which("uv") == uv else uv
        return _quote([exe, "pip", "install", "--python", sys.executable, GUARDRAIL_DISTRIBUTION])
    return _quote([sys.executable, "-m", "pip", "install", GUARDRAIL_DISTRIBUTION])


def install_command(
    uv: Optional[str],
    constraints: str,
    *,
    own_index: bool = False,
) -> list[str]:
    try:
        from .diffusion_nvfp4_install import _installer_prefix
        cmd = _installer_prefix(uv, own_index = own_index)
    except Exception:  # noqa: BLE001
        cmd = (
            [uv, "pip", "install", "--python", sys.executable]
            if uv
            else [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        )
    if not uv:
        cmd = [*cmd, "--no-input"]
    return [*cmd, "-c", constraints, GUARDRAIL_DISTRIBUTION]


def status() -> dict[str, Any]:
    from .video_cosmos3 import guardrail_installed

    with _lock:
        snapshot = dict(_state)
    snapshot["installed"] = guardrail_installed()
    snapshot["command"] = manual_install_command()
    return snapshot


def _set(**values: Any) -> None:
    with _lock:
        _state.update(values)


def run_install(run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Install synchronously; returns the final state. Callers go through ``start_install``."""
    from .diffusion_nvfp4_install import (
        _child_env,
        _installer_config,
        _write_constraints,
        installed_distributions,
    )
    from .video_cosmos3 import guardrail_installed, refresh_guardrail_import

    uv = _uv()
    config = _installer_config(uv, run)
    if config.get("refusal"):
        _set(state = "failed", error = f"cosmos-guardrail not installed: {config['refusal']}")
        return status()
    before = installed_distributions()
    constraints = _write_constraints(before)
    cmd = install_command(uv, constraints, own_index = bool(config.get("own_index", False)))
    logger.info("cosmos_guardrail.install: %s", _quote(cmd))
    try:
        result = run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _INSTALL_TIMEOUT_S,
            env = _child_env(),
        )
        ok = getattr(result, "returncode", 1) == 0
        output = str(getattr(result, "stdout", "") or "").strip()
    except subprocess.TimeoutExpired:
        ok, output = False, f"timed out after {_INSTALL_TIMEOUT_S}s"
    except Exception as exc:  # noqa: BLE001
        ok, output = False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.unlink(constraints)
        except OSError:
            pass
    if not ok:
        try:
            from utils.log_redaction import redact_log_text
            output = redact_log_text(output)
        except Exception:  # noqa: BLE001
            pass
        tail = output[-600:] or "no output"
        _set(
            state = "failed",
            error = (
                "The cosmos-guardrail install failed (it may only add packages, never change one Studio "
                f"already has). Installer output: {tail}"
            ),
        )
        return status()
    refresh_guardrail_import()
    after = installed_distributions()
    added = sorted(name for name in after if name not in before)
    if not guardrail_installed():
        _set(
            state = "failed",
            error = "The installer finished, but cosmos_guardrail is still not importable.",
        )
        return status()
    _set(state = "done", error = None, added = added)
    logger.info("cosmos_guardrail.install: added %s", ", ".join(added) or "nothing")
    return status()


def start_install() -> dict[str, Any]:
    """Start the install on a daemon thread, or report the one already running / the package already there."""
    global _thread
    from .video_cosmos3 import guardrail_installed

    with _lock:
        if _thread is not None and _thread.is_alive():
            return {**_state, "installed": False, "command": manual_install_command()}
        if guardrail_installed():
            _state.update(state = "done", error = None)
            return {**_state, "installed": True, "command": manual_install_command()}
        _state.update(state = "running", error = None, added = [])

        def _worker() -> None:
            try:
                run_install()
            except Exception as exc:  # noqa: BLE001
                logger.exception("cosmos_guardrail.install failed")
                _set(state = "failed", error = f"{type(exc).__name__}: {exc}")

        _thread = threading.Thread(target = _worker, name = "cosmos-guardrail-install", daemon = True)
        _thread.start()
        return {**_state, "installed": False, "command": manual_install_command()}
