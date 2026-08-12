# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ask a short-lived child process whether a torch device can actually allocate.

A GPU runtime that does not match its silicon does not raise, it faults. On the
gfx1151 host in #7331 a gfx1100 ROCm wheel reported ``is_available() == True``,
enumerated the device, then died with SIGSEGV in ``libamdhip64`` on the first real
allocation, past an ``except Exception`` because a signal is not an exception, and
took uvicorn with it since signal disposition is per process (#8474).

So the question cannot be answered in the process that must survive the answer. It
is answered in a child, and the child dying IS the answer. The parent side imports
only the standard library: it must stay importable on ``--no-torch``, and must not
be what drags torch into the lean main process (tests/test_startup_defers_torch.py).

Deliberately NOT here:
  * no ``multiprocessing`` -- uvicorn is multithreaded, HIP/CUDA are fork-hostile,
    and ``set_start_method`` is process-global state we do not own
  * no model load -- the parent would load it again, paying the download, the
    deserialization and the security gate twice
  * no on-disk cache -- a negative verdict outliving a driver or wheel repair would
    pin a healthy machine to CPU with no way to tell why
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

# stdlib logging, as in core/rag/embeddings.py: this must add no import weight.
logger = logging.getLogger(__name__)

# A cold ROCm torch import runs to minutes. Being generous costs a late load; being
# stingy condemns a healthy host to CPU.
PROBE_TIMEOUT_SECONDS = 120.0
# Past the parent's deadline, so it only ever fires for a child whose parent is gone.
_CHILD_SELF_LIMIT_SECONDS = 300.0
# Grace between terminate and kill when the child overran, mirroring the sidecars.
_TERMINATE_GRACE_SECONDS = 5.0
# Child stderr is a torch traceback or a driver's parting words; keep the tail only.
_STDERR_TAIL_CHARS = 600
# Above this a negative exit code is a signed Windows status, not a signal (NSIG is 64).
_MAX_SIGNAL_NUMBER = 128

# These decide which physical device a device string names, or which kernels the runtime
# emits, so a change invalidates the verdict. HSA_OVERRIDE_GFX_VERSION is the #7331 spoof.
_DEVICE_IDENTITY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
)

# Windows ROCm bin directories for the child, os.pathsep-joined. See _rocm_dll_directories.
ROCM_DLL_DIRS_ENV_VAR = "UNSLOTH_STUDIO_PROBE_ROCM_DLL_DIRS"

# Allocate, WRITE, then synchronize: execution is asynchronous, so without the sync a
# fault can still be in flight when the child exits 0, which is the one way this reports a
# false pass. torch.empty alone is too weak for the same reason -- it need not touch the
# device. The DLL block precedes the import because Python ignores PATH for extension
# modules and add_dll_directory does not survive into a child, so a healthy Windows AMD GPU
# would fail to import torch at all. main.py and core/training/worker.py do the same.
_CHILD_SCRIPT = """
import os
import sys
import threading

# Self-limit: binding to the parent's death needs a pre-exec hook, and that hook is what
# lets a fork of a multithreaded server deadlock, so a Studio killed mid-probe leaves
# something that exits on its own. os._exit because a wedged driver will not shut down
# cleanly.
_watchdog = threading.Timer(float(sys.argv[2]), lambda: os._exit(70))
_watchdog.daemon = True  # a Timer is a Thread and is NOT daemon by default; without this
_watchdog.start()        # the child cannot exit until it fires and every probe times out

if sys.platform == "win32":
    _handles = []
    for _d in os.environ.get("UNSLOTH_STUDIO_PROBE_ROCM_DLL_DIRS", "").split(os.pathsep):
        if _d and os.path.isdir(_d):
            try:
                _handles.append(os.add_dll_directory(_d))
            except (OSError, AttributeError):
                pass

import torch

device = sys.argv[1]
tensor = torch.empty(1, device = device)
tensor.zero_()
if tensor.device.type == "cuda":
    torch.cuda.synchronize(tensor.device)
elif tensor.device.type == "xpu":
    torch.xpu.synchronize(tensor.device)
print("ok")
"""


def _rocm_dll_directories() -> list[str]:
    """Windows ROCm ``bin`` directories, newest version first. Empty off Windows.

    Same discovery as ``main.py`` and ``core/training/worker.py``, repeated because the
    child is a bare ``-c`` script with no backend directory on its path. Computing it
    parent-side keeps it testable.
    """
    if sys.platform != "win32":
        return []

    candidates: list[str] = []
    for var in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(var)
        if value:
            candidates.append(os.path.join(value, "bin"))

    default_root = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm")

    def _version_key(name: str) -> tuple:
        # Numeric tuple key so "10.0" sorts after "7.0".
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

    return [d for d in candidates if os.path.isdir(d)]


@dataclass(frozen = True)
class DeviceAllocationProbeResult:
    """Verdict for one device. ``ok`` is the only field a caller should branch on;
    ``returncode`` and ``reason`` exist to make the log line say something useful."""

    ok: bool
    device: str
    returncode: Optional[int]
    reason: Optional[str]
    duration_seconds: float


_cache_lock = threading.Lock()
_cache: dict[tuple, DeviceAllocationProbeResult] = {}


def _cache_key(device: str) -> tuple:
    return (device, sys.executable) + tuple(
        os.environ.get(name) for name in _DEVICE_IDENTITY_ENV_VARS
    )


def describe_exit(returncode: Optional[int]) -> Optional[str]:
    """A human-readable cause for a nonzero child exit, for logs only.

    Mirrors ``LlamaCppBackend._is_signal_crash`` rather than importing it: the caller is
    torch-optional and must not pull in llama.cpp plumbing to read one predicate.
    """
    if returncode is None or returncode == 0:
        return None

    # Sign alone cannot separate a POSIX signal from a Windows NTSTATUS: -11 (SIGSEGV) and
    # -1073741819 (0xC0000005 signed) are both negative, and masking either to 32 bits lands
    # above 0xC0000000. Magnitude can.
    if returncode < 0 and -returncode <= _MAX_SIGNAL_NUMBER:
        signal_number = -returncode
        try:
            import signal
            return f"killed by {signal.Signals(signal_number).name}"
        except (ImportError, ValueError):
            return f"killed by signal {signal_number}"

    unsigned = returncode & 0xFFFFFFFF
    if unsigned >= 0xC0000000:
        return f"native fault, exit status 0x{unsigned:08X}"
    return f"exit code {returncode}"


def _run_child(device: str) -> tuple[Optional[int], str, Optional[str]]:
    """Run the probe child. Returns ``(returncode, stderr_tail, failure_reason)``, where
    a non-None ``failure_reason`` means the child never delivered a verdict of its own."""
    from utils.child_stdio import utf8_child_env
    from utils.native_path_leases import child_env_without_native_path_secret
    from utils.process_lifetime import adopt_pid, forget_pid
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    env = utf8_child_env(child_env_without_native_path_secret())
    dll_dirs = _rocm_dll_directories()
    if dll_dirs:
        env[ROCM_DLL_DIRS_ENV_VAR] = os.pathsep.join(dll_dirs)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT, device, str(_CHILD_SELF_LIMIT_SECONDS)],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = env,
            # Deliberately NO preexec_fn, hence no child_popen_kwargs(). The long-lived GPU
            # children take its PDEATHSIG hook, but that forks this multithreaded server and
            # runs Python before exec, where a lock another uvicorn thread held at fork time
            # is still held; the child can hang there while the parent waits inside Popen,
            # ahead of the timeout below. Python's docs call it unsafe with threads, and it
            # also disables the vfork fast path. The child self-limits instead.
            **windows_hidden_subprocess_kwargs(),
        )
    except OSError as spawn_error:
        # No interpreter, no fork headroom, no permission. We learned nothing about the
        # device, and "learned nothing" fails closed.
        return None, "", f"probe could not start ({spawn_error})"

    adopt_pid(process.pid)  # terminate_all backstop for a graceful shutdown
    try:
        try:
            _, stderr = process.communicate(timeout = PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Classify BEFORE cleaning up: after the terminate/kill below, returncode is
            # the signal WE sent, and reporting that as the driver's fault would be a lie.
            stderr = _terminate_and_drain(process)
            return process.returncode, stderr, "probe timed out"
        except OSError as read_error:
            # A pipe or handle failure, or a read error under resource pressure. The child
            # may still be running and we have no verdict from it, so tear it down and fail
            # closed rather than letting this escape a function that promises not to raise.
            stderr = _terminate_and_drain(process)
            return process.returncode, stderr, f"probe could not be read ({read_error})"

        return process.returncode, stderr or "", None
    finally:
        # A child handed to the reaper is still alive, so it stays adopted; the reaper
        # forgets it once it is really gone.
        if process.returncode is not None:
            forget_pid(process.pid)


def _reap_later(process: subprocess.Popen) -> None:
    """Wait on a child that outlived SIGKILL, on a daemon thread, forever.

    SIGKILL cannot land while a task sits in an uninterruptible driver wait -- it is
    delivered once the ioctl returns, and a wedged GPU is how a task gets there. That being
    the scenario this module exists to survive, dropping the last reference is the one way
    such a child becomes an unreaped stray.
    """
    threading.Thread(
        target = _wait_forever,
        args = (process,),
        daemon = True,
        name = f"device-probe-reaper-{process.pid}",
    ).start()


def _wait_forever(process: subprocess.Popen) -> None:
    try:
        process.wait()
    except Exception:  # noqa: BLE001 - nothing to do but stop holding the reference
        pass
    try:
        from utils.process_lifetime import forget_pid
        forget_pid(process.pid)
    except Exception:  # noqa: BLE001 - the reaper is best effort by definition
        pass


def _terminate_and_drain(process: subprocess.Popen) -> str:
    """Terminate, give it a moment, kill, and always drain and reap, so an overrunning
    probe cannot leave a zombie or a held pipe behind."""
    stderr = ""
    # Escalate in one loop rather than nesting, so every attempt is covered by the same
    # handler. The nested form grew a gap: an OSError raised inside the timeout branch was
    # not a sibling of the outer OSError handler and escaped a function that must not raise.
    for signal_child in (process.terminate, process.kill):
        try:
            signal_child()
        except OSError:
            pass
        try:
            _, stderr = process.communicate(timeout = _TERMINATE_GRACE_SECONDS)
            return stderr or ""
        except subprocess.TimeoutExpired:
            continue  # still alive, escalate
        except OSError:
            break  # pipes are unusable, so there is nothing left to drain

    # Not confirmed dead: either it outlived SIGKILL, stuck in the driver, or we could not
    # read it. Either way the last reference must not simply be dropped.
    _reap_later(process)
    return stderr or ""


def probe_torch_device_allocation(device: str = "cuda:0") -> DeviceAllocationProbeResult:
    """Whether a torch tensor can be allocated, written and synchronized on *device*.

    Never raises and never returns "unknown": anything we could not establish -- a
    signal, a nonzero exit, a timeout, a failed spawn -- comes back ``ok = False``, so a
    caller reading ``.ok`` degrades rather than gambles. Memoized per process.
    """
    key = _cache_key(device)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    started = time.monotonic()
    returncode, stderr, failure_reason = _run_child(device)
    duration = time.monotonic() - started

    ok = failure_reason is None and returncode == 0
    reason = failure_reason or describe_exit(returncode)
    result = DeviceAllocationProbeResult(
        ok = ok,
        device = device,
        returncode = returncode,
        reason = reason,
        duration_seconds = duration,
    )

    if ok:
        logger.debug("device allocation probe passed for %s in %.1fs", device, duration)
    else:
        tail = (stderr or "").strip()[-_STDERR_TAIL_CHARS:]
        logger.warning(
            "device allocation probe failed for %s (%s) after %.1fs%s",
            device,
            reason,
            duration,
            f": {tail}" if tail else "",
        )

    with _cache_lock:
        # Another thread may have raced us to the same verdict; either is correct, so
        # keep whichever landed first and let both callers see one value.
        return _cache.setdefault(key, result)


def _clear_probe_cache() -> None:
    """Drop memoized verdicts (test teardown only)."""
    with _cache_lock:
        _cache.clear()
