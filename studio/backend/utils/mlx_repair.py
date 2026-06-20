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

The install mirrors the main Apple Silicon installer (install_python_stack.py):
it points UV_OVERRIDE at overrides-darwin-arm64.txt so the resolver keeps the
Studio transformers pin AND installs a current mlx-vlm, and it requires the same
minimum versions unsloth-zoo declares so a backtracked old mlx-vlm (which still
imports but breaks VLM Train/Export) is never accepted as healthy.

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
import tempfile
import threading
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_MLX_AUTOREPAIR"
# Minimum versions unsloth-zoo requires on Apple Silicon (its pyproject darwin
# deps). mlx-vlm especially must be >=0.4.4: an older one still imports but
# breaks VLM Train/Export, so installing it would wrongly clear chat-only.
_MLX_MIN_VERSIONS = {"mlx": "0.22.0", "mlx-lm": "0.22.0", "mlx-vlm": "0.4.4"}
MLX_PACKAGES = tuple(f"{name}>={version}" for name, version in _MLX_MIN_VERSIONS.items())
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


def mlx_stack_available() -> bool:
    """`import mlx.core` works AND mlx/mlx-lm/mlx-vlm meet unsloth-zoo's minimums.

    A bare `import mlx.core` is not enough: a backtracked old mlx-vlm imports
    fine but breaks VLM Train/Export, so it must not count as a healthy stack
    (otherwise the self-heal would clear chat-only onto a broken install)."""
    if not mlx_available():
        return False
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _dist_version

        from packaging.version import Version
    except Exception:
        return True  # cannot version-check here; trust the successful import
    for name, minimum in _MLX_MIN_VERSIONS.items():
        try:
            if Version(_dist_version(name)) < Version(minimum):
                return False
        except PackageNotFoundError:
            return False
        except Exception:
            continue  # unparseable version metadata: do not fail the whole check
    return True


def _pip_install_cmd(*args: str) -> list[str]:
    """`uv pip install` into this venv if uv is on PATH, else `python -m pip
    install`. Mirrors core.training.worker._pip_install_cmd."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *args]
    return [sys.executable, "-m", "pip", "install", *args]


def _mlx_install_env() -> dict[str, str]:
    """Environment for the mlx install. Mirror the main installer
    (install_python_stack.py) by pointing UV_OVERRIDE at overrides-darwin-arm64.txt,
    which relaxes mlx-vlm/mlx-lm's transformers>=5 requirement to >=4.57.6. Without
    it, uv keeps the Studio transformers pin only by silently backtracking mlx-vlm
    to an old, unsupported version (uv honours UV_OVERRIDE; plain pip ignores it,
    so the transformers constraint below is the pip-path safety net)."""
    env = dict(os.environ)
    override = (
        Path(__file__).resolve().parents[1]
        / "requirements"
        / "single-env"
        / "overrides-darwin-arm64.txt"
    )
    if override.is_file():
        env.setdefault("UV_OVERRIDE", str(override))
    return env


def _transformers_constraint_args() -> tuple[list[str], str | None]:
    """Pin transformers to the running version for the mlx install.

    The install must never upgrade transformers underneath a running Studio
    (the single-env install pins transformers==4.57.6). With UV_OVERRIDE set this
    is belt-and-suspenders; on the plain-pip path (no UV_OVERRIDE support) it is
    the actual guard -- the resolver either finds an mlx build compatible with the
    pin or fails, leaving us chat-only rather than breaking Studio. Returns
    (pip args, temp file path to clean up)."""
    try:
        import transformers  # already a hard Studio dependency
    except Exception:
        return [], None
    fd, path = tempfile.mkstemp(prefix = "mlx_repair_", suffix = ".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write(f"transformers=={transformers.__version__}\n")
    return ["--constraint", path], path


def attempt_mlx_repair(*, timeout: int = _REPAIR_TIMEOUT_S) -> bool:
    """Install a usable mlx/mlx-lm/mlx-vlm stack by name into the running venv.
    Best-effort; returns True iff the resulting stack meets unsloth-zoo's minimums
    (so a backtracked old mlx-vlm is rejected, not accepted). transformers is held
    at its pinned version so the install can never upgrade it underneath Studio."""
    constraint_args, constraint_path = _transformers_constraint_args()
    cmd = _pip_install_cmd("--upgrade", *constraint_args, *MLX_PACKAGES)
    logger.info("MLX self-heal: installing %s", ", ".join(MLX_PACKAGES))
    try:
        result = subprocess.run(
            cmd,
            env = _mlx_install_env(),
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
    finally:
        if constraint_path and os.path.exists(constraint_path):
            try:
                os.remove(constraint_path)
            except OSError:
                pass
    if result.returncode != 0:
        tail = (result.stdout or "")[-2000:]
        logger.warning("MLX self-heal failed (staying chat-only):\n%s", tail)
        return False
    if not mlx_stack_available():
        logger.warning(
            "MLX self-heal produced an incomplete or too-old MLX stack "
            "(need %s); staying chat-only.",
            ", ".join(f"{name}>={ver}" for name, ver in _MLX_MIN_VERSIONS.items()),
        )
        return False
    return True


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
    """If this is an Apple Silicon host whose MLX stack is missing or too old,
    reinstall it on a daemon thread (off the startup critical path) and re-detect
    on success. Returns True iff a repair thread was started. No-op (returns False)
    off Apple Silicon, when the stack is already adequate, when already attempted
    this process, or when disabled via UNSLOTH_DISABLE_MLX_AUTOREPAIR=1."""
    global _attempted
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    if not is_apple_silicon():
        return False
    if mlx_stack_available():
        return False
    with _attempted_lock:
        if _attempted:
            return False
        _attempted = True
    logger.warning(
        "Apple Silicon without a usable MLX stack; attempting a one-time background "
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
