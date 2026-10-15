# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe torch allocation in a child so driver crashes do not kill the backend.

Only a child that ran cleanly to the end marks an accelerator usable: a crash, a hang, a kill and a probe that could not run or be read all leave it unusable, since the allocation this stands in front of ends the process rather than raising. Ordinary Python errors are the exception, because the child ran and reported, so the in-process loader raises the same error and describes it better. CPU takes the opposite default, since it cannot fault a driver and condemning it would change the embedding backend. Set ``UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE=1`` to skip the probe.
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

PROBE_TIMEOUT_SECONDS = 120.0
_CHILD_SELF_LIMIT_SECONDS = 300.0
_TERMINATE_GRACE_SECONDS = 5.0
_STDERR_TAIL_CHARS = 600
# SIGILL/ABRT/BUS/FPE/SEGV; not SIGKILL/SIGTERM, which mean the probe was killed.
_FATAL_SIGNALS = frozenset({4, 6, 7, 8, 11})
# Reserved self-timeout status on Windows; elsewhere the child gets SIGALRM.
_WATCHDOG_EXIT_STATUS = 70
_SIGALRM_NUMBER = 14
# MSVC CRT abort() exit status on Windows; matches LlamaCppBackend._is_abort_exit.
_WINDOWS_ABORT_EXIT_STATUS = 3

# Env vars that change which device or kernels a string names; changes invalidate cached verdicts.
_DEVICE_IDENTITY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "ZE_AFFINITY_MASK",
    "ONEAPI_DEVICE_SELECTOR",
)

# item() syncs so async faults surface in the child; DLL dirs must be added before importing torch.
_PROBE_SCRIPT = """
import os
import signal
import sys
import threading

# The deadline has to hold even when torch hangs inside a native call, which is the case
# it exists for. A threading.Timer cannot: its callback needs the GIL, and a long C call
# never returns to the interpreter loop to release it. SIGALRM with NO handler installed is
# enforced by the kernel instead, so it does not run Python and does not need the GIL.
# Windows has no alarm, so the timer stays as the fallback there.
#
# The disposition is restored first because exec keeps an inherited SIG_IGN and an inherited
# blocked mask, so a supervisor that ignores or blocks SIGALRM would otherwise leave this
# deadline unenforceable and an orphaned probe running against a hung driver forever.
_deadline = float(sys.argv[2])
if hasattr(signal, "alarm"):
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    signal.alarm(int(_deadline) or 1)
else:
    _watchdog = threading.Timer(_deadline, lambda: os._exit(70))  # _WATCHDOG_EXIT_STATUS
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
    """Windows exit status 3 is the CRT's abort(), so it counts as a fault; SIGKILL and SIGTERM do not."""
    if returncode < 0:
        return -returncode in _FATAL_SIGNALS
    if os.name != "nt":
        return False
    if returncode == _WINDOWS_ABORT_EXIT_STATUS:
        return True
    return (returncode & 0xC0000000) == 0xC0000000


def _hit_its_own_deadline(returncode: int) -> bool:
    """Whether the child stopped itself for running too long. A child that reached its own deadline hung, and a hang is a device failure, so this has to be read as one. Neither form is otherwise recognised: SIGALRM is not a hard fault and would fall through ``_died_by_signal``, and the Windows status is an ordinary non-zero exit. Both were being reported as a healthy device, which then let the parent make the very allocation the probe stands in front of. It only comes up when the parent did not enforce its own shorter timeout first, such as a suspended backend."""
    if returncode == _WATCHDOG_EXIT_STATUS:
        return True
    return os.name != "nt" and returncode == -_SIGALRM_NUMBER


def _unknown_verdict(
    device: str,
    what_happened: str,
    *,
    exc_info: bool = True,
) -> bool:
    """What to answer when the probe produced no verdict at all. Unusable for an accelerator: there is no evidence it is fine, and the two ways of being wrong are not symmetric, since the allocation this stands in front of ends the process. Usable for CPU, the opposite trade: a CPU load cannot fault a GPU driver, so a probe that never ran says nothing against it, and condemning it here would send the caller past its CPU fallback to the GGUF backend, changing the embedding space and forcing a reindex over what may be a passing failure to fork."""
    usable = device == "cpu"
    logger.warning(
        "torch allocation probe on %s %s; treating the device as %s",
        device,
        what_happened,
        "usable, since CPU cannot fault the driver" if usable else "unusable",
        exc_info = exc_info,
    )
    return usable


def _identity_key() -> tuple[str | None, ...]:
    return tuple(os.environ.get(name) for name in _DEVICE_IDENTITY_ENV_VARS)


def device_can_allocate(device: str) -> bool:
    """False when the probe crashes, times out or its result is unreadable, since unknown is not usable."""
    return _device_can_allocate_cached(device, _identity_key())


@lru_cache(maxsize = None)
def _device_can_allocate_cached(device: str, _identity: tuple[str | None, ...]) -> bool:
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return True

    env = child_env_without_native_path_secret()
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
            env = utf8_child_env(env),
            # No child_popen_kwargs(): its Linux preexec_fn can deadlock in this multithreaded process.
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:  # noqa: BLE001 - no child ran, so nothing was proven
        return _unknown_verdict(device, "could not run")

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
        except Exception:  # noqa: BLE001 - no verdict, so the device is not known to work
            _terminate_and_drain(process)
            return _unknown_verdict(device, "could not be read")

        if _hit_its_own_deadline(process.returncode):
            logger.warning(
                "torch allocation probe on %s ran past its own deadline and stopped "
                "itself; treating the device as unusable",
                device,
            )
            return False

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

        if process.returncode < 0:
            # OOM kill or container stop is not evidence either way; treating it as a pass is unsafe.
            return _unknown_verdict(
                device,
                f"was killed by signal {-process.returncode} without faulting",
                exc_info = False,
            )
        return True
    finally:
        if process.returncode is not None:
            forget_pid(process.pid)


def _terminate_and_drain(process: subprocess.Popen) -> str:
    """Bound cleanup after timeout and retain an unkillable child for reaping. Escalates in one loop rather than nesting, so one pair of handlers covers every attempt. Nested, the post-kill read sat inside the timeout branch where the trailing ``except OSError`` was a sibling and could not see it: a pipe failure there escaped ``device_can_allocate``, so a device that genuinely timed out raised instead of returning False, the child never reached the reaper, and since ``lru_cache`` does not cache exceptions the next call re-ran the whole probe."""
    stderr = ""
    for signal_child in (process.terminate, process.kill):
        try:
            signal_child()
        except OSError:
            pass
        try:
            _, stderr = process.communicate(timeout = _TERMINATE_GRACE_SECONDS)
            return stderr or ""
        except subprocess.TimeoutExpired:
            continue
        except OSError:
            break

    _reap_later(process)
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
