# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe torch allocation in a child so driver crashes do not kill the backend.

Only a killed or hung child marks a device unusable. Ordinary Python errors and
spawn failures are left to the in-process loader. Set
``UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE=1`` to skip the probe.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from functools import lru_cache

from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE"
ROCM_DLL_DIRS_ENV_VAR = "UNSLOTH_STUDIO_PROBE_ROCM_DLL_DIRS"

# Allow for a cold torch import and driver initialization on a busy host.
PROBE_TIMEOUT_SECONDS = 120.0
# The child bounds its own lifetime if the parent disappears during the probe.
_CHILD_SELF_LIMIT_SECONDS = 300.0
_TERMINATE_GRACE_SECONDS = 5.0
_STDERR_TAIL_CHARS = 600

_DEVICE_IDENTITY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
)

# The matmul tests allocation and vendor BLAS initialization. item() synchronizes
# the result so an asynchronous driver fault cannot escape after the child exits.
# Windows DLL directories must be registered before importing torch because those
# registrations are process-local and are not inherited by this interpreter.
_PROBE_SCRIPT = """
import os
import sys
import threading

_watchdog = threading.Timer(float(sys.argv[2]), lambda: os._exit(70))
_watchdog.daemon = True
_watchdog.start()

if sys.platform == "win32":
    _handles = []
    for _directory in os.environ.get(
        "UNSLOTH_STUDIO_PROBE_ROCM_DLL_DIRS", ""
    ).split(os.pathsep):
        if _directory and os.path.isdir(_directory):
            try:
                _handles.append(os.add_dll_directory(_directory))
            except (OSError, AttributeError):
                pass

import torch

device = sys.argv[1]
tensor = torch.ones((8, 8), dtype = torch.float16, device = device)
(tensor @ tensor).sum().item()
"""


def _rocm_dll_directories() -> list[str]:
    """Return Windows ROCm bin directories, newest version first."""
    if sys.platform != "win32":
        return []

    candidates: list[str] = []
    for variable in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(os.path.join(value, "bin"))

    default_root = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm")

    def _version_key(name: str) -> tuple:
        parts = []
        for chunk in name.split("."):
            try:
                parts.append((0, int(chunk)))
            except ValueError:
                parts.append((1, chunk))
        return tuple(parts)

    try:
        if os.path.isdir(default_root):
            for version in sorted(os.listdir(default_root), key = _version_key, reverse = True):
                bin_dir = os.path.join(default_root, version, "bin")
                if os.path.isdir(bin_dir):
                    candidates.append(bin_dir)
    except OSError:
        pass

    return list(dict.fromkeys(path for path in candidates if os.path.isdir(path)))


def _died_by_signal(returncode: int) -> bool:
    """Return whether the code represents a POSIX signal or fatal Windows NTSTATUS."""
    if returncode < 0:
        return True
    return os.name == "nt" and (returncode & 0xC0000000) == 0xC0000000


def _identity_key() -> tuple[str | None, ...]:
    return tuple(os.environ.get(name) for name in _DEVICE_IDENTITY_ENV_VARS)


def device_can_allocate(device: str) -> bool:
    """Return false only when the child probe crashes or times out.

    Results are cached per device and device-identity environment. Spawn failures
    and ordinary child exceptions return true so the existing in-process loader
    can report them accurately.
    """
    return _device_can_allocate_cached(device, _identity_key())


@lru_cache(maxsize = None)
def _device_can_allocate_cached(device: str, _identity: tuple[str | None, ...]) -> bool:
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return True

    env = utf8_child_env(child_env_without_native_path_secret())
    dll_directories = _rocm_dll_directories()
    if dll_directories:
        env[ROCM_DLL_DIRS_ENV_VAR] = os.pathsep.join(dll_directories)

    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _PROBE_SCRIPT, device, str(_CHILD_SELF_LIMIT_SECONDS)],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = env,
            # No child_popen_kwargs() here. Its Linux preexec_fn can deadlock when
            # this multithreaded backend forks and executes Python before exec.
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:  # noqa: BLE001 - no child ran, so nothing was proven
        logger.debug("torch allocation probe on %s could not run", device, exc_info = True)
        return True

    from utils.process_lifetime import adopt_pid, forget_pid

    adopt_pid(process.pid)
    try:
        try:
            _, stderr = process.communicate(timeout = PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            stderr = _terminate_and_drain(process)
            tail = (stderr or "").strip()[-_STDERR_TAIL_CHARS:]
            logger.warning(
                "torch allocation probe on %s did not finish in %.0fs; treating the "
                "device as unusable%s",
                device,
                PROBE_TIMEOUT_SECONDS,
                f": {tail}" if tail else "",
            )
            return False
        except Exception:  # noqa: BLE001 - no verdict, preserve the existing loader path
            _terminate_and_drain(process)
            logger.debug(
                "torch allocation probe on %s could not collect a result",
                device,
                exc_info = True,
            )
            return True

        if _died_by_signal(process.returncode):
            tail = (stderr or "").strip()[-_STDERR_TAIL_CHARS:]
            logger.warning(
                "torch allocation probe on %s was killed (exit %s); this torch build "
                "cannot use the device without crashing the process%s",
                device,
                process.returncode,
                f": {tail}" if tail else "",
            )
            return False
        return True
    finally:
        # A child handed to the asynchronous reaper remains adopted until it exits.
        if process.returncode is not None:
            forget_pid(process.pid)


def _terminate_and_drain(process: subprocess.Popen) -> str:
    """Bound cleanup after timeout and retain an unkillable child for reaping."""
    stderr = ""
    try:
        process.terminate()
    except OSError:
        pass
    try:
        _, stderr = process.communicate(timeout = _TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            _, stderr = process.communicate(timeout = _TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _reap_later(process)
    except OSError:
        pass
    return stderr or ""


def _reap_later(process: subprocess.Popen) -> None:
    threading.Thread(
        target = _wait_and_forget,
        args = (process,),
        daemon = True,
        name = f"torch-device-probe-reaper-{process.pid}",
    ).start()


def _wait_and_forget(process: subprocess.Popen) -> None:
    try:
        process.wait()
    except Exception:  # noqa: BLE001 - best effort cleanup
        pass
    try:
        from utils.process_lifetime import forget_pid
        forget_pid(process.pid)
    except Exception:  # noqa: BLE001 - best effort cleanup
        pass


def _clear_probe_cache() -> None:
    _device_can_allocate_cached.cache_clear()


# Preserve the cache-control hook used by existing tests and callers.
device_can_allocate.cache_clear = _clear_probe_cache  # type: ignore[attr-defined]
