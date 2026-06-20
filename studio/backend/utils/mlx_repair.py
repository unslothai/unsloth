# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Best-effort MLX self-heal for Apple Silicon.

On macOS, Studio enables Train/Export only when `import mlx.core` works
(see utils.hardware.hardware.detect_hardware -> CHAT_ONLY). MLX is pulled only
transitively via unsloth-zoo, and a resolver backtrack (mlx-vlm -> transformers>=5
vs the single-env transformers pin) can silently drop it, leaving Train/Export
greyed out after a reinstall/update. This reinstalls mlx *by name* -- bypassing
that fragile transitive resolution -- on a background thread, then re-detects so
the gate re-opens without a manual `unsloth studio update`.

Mirrors the runtime backend self-heal already used for tilelang
(core.training.worker._ensure_tilelang_backend_unconditional): default-on,
best-effort, opt out with UNSLOTH_DISABLE_MLX_AUTOREPAIR=1.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading

import structlog

logger = structlog.get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_MLX_AUTOREPAIR"
MLX_PACKAGES = ("mlx", "mlx-lm", "mlx-vlm")
_REPAIR_TIMEOUT_S = 900

# Attempt at most once per process; success is sticky (mlx then imports and the
# guard short-circuits on the next boot).
_attempted = False
_attempted_lock = threading.Lock()


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def _pip_install_cmd(*args: str) -> list[str]:
    """`uv pip install` into this venv if uv is on PATH, else `python -m pip
    install`. Mirrors core.training.worker._pip_install_cmd."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *args]
    return [sys.executable, "-m", "pip", "install", *args]


def attempt_mlx_repair(*, timeout: int = _REPAIR_TIMEOUT_S) -> bool:
    """Install mlx + mlx-lm + mlx-vlm by name into the running venv. Best-effort;
    returns True iff `import mlx.core` works afterwards."""
    cmd = _pip_install_cmd("--upgrade", *MLX_PACKAGES)
    logger.info("MLX self-heal: installing %s", ", ".join(MLX_PACKAGES))
    try:
        result = subprocess.run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            timeout = timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("MLX self-heal timed out after %ss; staying chat-only", timeout)
        return False
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("MLX self-heal could not start: %s", exc)
        return False
    if result.returncode != 0:
        tail = (result.stdout or "")[-2000:]
        logger.warning("MLX self-heal failed (staying chat-only):\n%s", tail)
        return False
    return mlx_available()


def _run_repair_and_redetect() -> None:
    if not attempt_mlx_repair():
        return
    try:
        from utils.hardware import hardware as hw
        hw.detect_hardware()  # flips CHAT_ONLY / DEVICE now that mlx imports
        logger.info(
            "MLX self-heal succeeded; Train/Export enabled (reload the page). chat_only=%s",
            hw.CHAT_ONLY,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("MLX installed but hardware re-detection failed: %s", exc)


def start_mlx_autorepair_if_needed() -> bool:
    """If this is an Apple Silicon host with MLX missing, reinstall it on a daemon
    thread (off the startup critical path) and re-detect on success. Returns True
    iff a repair thread was started. No-op (returns False) off Apple Silicon, when
    MLX already imports, when already attempted this process, or when disabled via
    UNSLOTH_DISABLE_MLX_AUTOREPAIR=1."""
    global _attempted
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    if not is_apple_silicon():
        return False
    if mlx_available():
        return False
    with _attempted_lock:
        if _attempted:
            return False
        _attempted = True
    logger.warning(
        "Apple Silicon without importable MLX; attempting a one-time background "
        "reinstall of mlx/mlx-lm/mlx-vlm to re-enable Train/Export. "
        "Set %s=1 to disable.",
        DISABLE_ENV_VAR,
    )
    threading.Thread(
        target = _run_repair_and_redetect,
        daemon = True,
        name = "mlx-autorepair",
    ).start()
    return True
