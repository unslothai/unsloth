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

# Allocate, WRITE, then synchronize. Device execution is asynchronous, so an allocation
# that faults in the kernel can outlive the statement that queued it -- without the
# synchronize the child can exit 0 while the fault is still in flight, which is the one
# way this probe could report a false pass. torch.empty alone is too weak for the same
# reason: it need not touch the device at all.
_CHILD_SCRIPT = """
import sys

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
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    env = utf8_child_env(child_env_without_native_path_secret())
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT, device],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = env,
            **windows_hidden_subprocess_kwargs(),
        )
    except OSError as spawn_error:
        # No interpreter, no fork headroom, no permission. We learned nothing about the
        # device, and "learned nothing" fails closed.
        return None, "", f"probe could not start ({spawn_error})"

    try:
        _, stderr = process.communicate(timeout = PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Classify BEFORE cleaning up: after the terminate/kill below, returncode is the
        # signal WE sent, and reporting that as the driver's fault would be a lie.
        stderr = _terminate_and_drain(process)
        return process.returncode, stderr, "probe timed out"

    return process.returncode, stderr or "", None


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
            pass
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
