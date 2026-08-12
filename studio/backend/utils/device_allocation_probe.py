# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ask a short-lived child process whether a torch device can actually allocate.

A GPU runtime that does not match the silicon it is driving does not raise: it
faults. On the gfx1151 host in #7331 a ROCm 6.3 wheel built for gfx1100 reported
``torch.cuda.is_available() == True``, enumerated the device correctly, and then
died with SIGSEGV inside ``libamdhip64`` on the first real allocation. Signal
disposition is per process, so that fault killed uvicorn -- from a daemon thread,
past an ``except Exception``, because a signal never becomes a Python exception
(#8474).

So the question "can this device allocate" cannot be answered in the process that
has to survive the answer. It is answered in a child, and the child dying IS the
answer. The parent side of this module imports only the standard library: it must
stay importable on a ``--no-torch`` install, and it must not be what finally drags
torch into the lean main process (see ``tests/test_startup_defers_torch.py``).

Deliberately NOT here:
  * no ``multiprocessing`` -- uvicorn is multithreaded and HIP/CUDA are
    fork-hostile, and ``set_start_method`` is process-global state we do not own
  * no model load -- the caller would have to load it again in-process, paying the
    download, the deserialization and the security gate twice
  * no on-disk cache -- a negative verdict that outlived a driver or wheel repair
    would pin a healthy machine to CPU with no way to tell why

The verdict is memoized for the process only, keyed by the device and by every
environment variable that can change which silicon that device names.
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

# Plain stdlib logging, as in core/rag/embeddings.py: the caller is the torch-optional
# embedder, and this module should add no import weight of its own.
logger = logging.getLogger(__name__)

# A cold ROCm torch import on a slow host is minutes, not seconds, and every caller of
# this probe is off the request/boot critical path. Being generous here costs a late
# warm; being stingy condemns a healthy host to CPU.
PROBE_TIMEOUT_SECONDS = 120.0
# Grace between terminate and kill when the child overran, mirroring the sidecars.
_TERMINATE_GRACE_SECONDS = 5.0
# Child stderr is a torch traceback or a driver's parting words; keep the tail only.
_STDERR_TAIL_CHARS = 600
# Above this, a negative exit code is a signed Windows status rather than a signal.
# POSIX signal numbers stop at 64 (NSIG); the headroom costs nothing.
_MAX_SIGNAL_NUMBER = 128

# Variables that decide WHICH physical device a device string names, or which kernels
# the runtime believes it should emit. A change to any of them invalidates the verdict:
# HSA_OVERRIDE_GFX_VERSION is exactly the spoof that produced #7331.
_DEVICE_IDENTITY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
)

# Windows ROCm bin directories for the child, os.pathsep-joined. See _rocm_dll_directories.
ROCM_DLL_DIRS_ENV_VAR = "UNSLOTH_STUDIO_PROBE_ROCM_DLL_DIRS"

# Allocate, WRITE, then synchronize. Device execution is asynchronous, so an allocation
# that faults in the kernel can outlive the statement that queued it -- without the
# synchronize the child can exit 0 while the fault is still in flight, which is the one
# way this probe could report a false pass. torch.empty alone is too weak for the same
# reason: it need not touch the device at all.
#
# The DLL block runs BEFORE ``import torch``: Python 3.8+ ignores PATH for extension
# modules, and os.add_dll_directory registrations are process-local, so a fresh child does
# not inherit the parent's. Without it a healthy Windows AMD GPU fails to import torch at
# all and this probe would report the host as condemned. main.py and core/training/worker.py
# do the same thing at their own process starts, for the same reason.
_CHILD_SCRIPT = """
import os
import sys

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

    Same discovery as ``main.py`` and ``core/training/worker.py``; it lives here too
    because the child is a bare ``-c`` script with no backend directory on its path (that
    isolation is deliberate), and computing the list parent-side keeps it testable.
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

    Mirrors ``LlamaCppBackend._is_signal_crash`` rather than importing it: that lives in
    the llama.cpp inference plumbing, and the RAG embedder which calls this probe is
    deliberately torch-optional and must not pull that module in to read one predicate.
    """
    if returncode is None or returncode == 0:
        return None

    # A POSIX signal arrives as a small negative number. Windows reports a native fault as
    # an NTSTATUS-shaped status instead, and can hand it over signed or unsigned. The two
    # cannot be told apart by sign alone: -11 (SIGSEGV) and -1073741819 (0xC0000005 signed)
    # are both negative, and masking either to 32 bits lands above 0xC0000000. Magnitude is
    # what separates them -- no signal number comes near _MAX_SIGNAL_NUMBER.
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
    from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    env = utf8_child_env(child_env_without_native_path_secret())
    dll_dirs = _rocm_dll_directories()
    if dll_dirs:
        env[ROCM_DLL_DIRS_ENV_VAR] = os.pathsep.join(dll_dirs)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT, device],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = env,
            # This child can sit in a cold torch import, or wedged in a driver ioctl, for
            # the whole timeout. Without the parent-death binding, a Studio that is killed
            # in that window leaves it reparented and running, where it holds the very GPU
            # the next backend is about to probe. Same treatment as the other GPU children.
            **child_popen_kwargs(),
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

        return process.returncode, stderr or "", None
    finally:
        # A child handed to the reaper is still alive, so it stays adopted; the reaper
        # forgets it once it is really gone.
        if process.returncode is not None:
            forget_pid(process.pid)


def _reap_later(process: subprocess.Popen) -> None:
    """Wait on a child that outlived SIGKILL, on a daemon thread, forever.

    SIGKILL cannot be caught, but it also cannot land while the process sits in an
    uninterruptible driver wait: the signal is only delivered once the ioctl returns, and a
    wedged GPU is precisely how a task ends up there. Since that is the scenario this
    module exists to survive, dropping the last reference to such a child is the one way it
    becomes an unreaped stray. Hold it and let the thread collect the corpse whenever the
    driver finally lets go. Daemon, so it never delays shutdown.
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
            # Still alive after SIGKILL: stuck in the driver. Hand it to a reaper rather
            # than returning and letting the last reference go.
            _reap_later(process)
    except OSError:
        pass
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
