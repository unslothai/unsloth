# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs inside the sandbox and scores batches for Python rewards. Standard library only.

The sandbox gives the process no stdin (MXC), so requests and replies are files in the workdir:
``req-<n>.json`` in, ``res-<n>.json`` out, each written to a temp name and renamed into place.
"""

import copy
import inspect
import json
import math
import os
import sys
import time
import traceback

POLL_SECONDS = 0.002
# The host touches "alive" every few seconds; a host that died without saying stop goes quiet.
ALIVE_TIMEOUT_SECONDS = 60.0
SETUP_TIMEOUT_SECONDS = 60.0
# rewards.MAX_SCORE: this file runs on its own in the worker, so it keeps a copy.
MAX_SCORE = 1_000_000.0


def _retry(action, seconds = 10.0):
    # Windows: a virus scanner can hold a file it just saw appear for a few milliseconds.
    deadline = time.monotonic() + seconds
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.005)


def put(path, payload):
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(payload, f)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))


def take(path):
    def read():
        with open(path, encoding = "utf-8") as f:
            return json.load(f)

    data = _retry(read)
    _retry(lambda: os.remove(path))
    return data


def _error_line(exc):
    lines = traceback.format_exception_only(type(exc), exc)
    return (lines[-1] if lines else repr(exc)).strip()[:2000]


def _install_unsloth_helpers(path):
    """Notebook rewards do ``from unsloth import execute_with_time_limit`` and friends. Importing
    unsloth here would need a GPU the sandbox can't see, so load just those helpers from
    unsloth_zoo/rl_environments.py and serve them as ``unsloth``."""
    if not path or "unsloth" in sys.modules:
        return
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location("unsloth_rl_environments", path)
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    shim = types.ModuleType("unsloth")
    for name in getattr(helpers, "__all__", ()):
        setattr(shim, name, getattr(helpers, name))
    sys.modules["unsloth"] = shim


def _load(specs):
    funcs, errors = {}, {}
    for spec in specs:
        name = spec["name"]
        namespace = {"__name__": "reward_" + name.replace("-", "_"), "__builtins__": __builtins__}
        try:
            code = compile(spec["code"], "<reward " + name + ">", "exec")
            exec(code, namespace)
            fn = namespace.get(spec["entry"])
            if not callable(fn):
                raise NameError(spec["entry"] + " is not defined by the reward code")
            funcs[name] = fn
        except BaseException as exc:  # noqa: BLE001 - reported back, never kills the worker
            errors[name] = _error_line(exc)
    return funcs, errors


def _takes(fn, name):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.kind is p.VAR_KEYWORD or p.name == name for p in params)


def _score(fn, request):
    kwargs = dict(request.get("kwargs") or {})
    # answer_raw is added by the host, not the dataset; a reward that doesn't name it never sees it.
    if "answer_raw" in kwargs and not _takes(fn, "answer_raw"):
        del kwargs["answer_raw"]
    out = fn(
        prompts = request.get("prompts"),
        completions = request.get("completions"),
        **kwargs,
    )
    n = len(request.get("completions") or [])
    out = list(out) if out is not None else []
    if len(out) != n:
        raise ValueError("returned %d scores for %d completions" % (len(out), n))
    scores = []
    for value in out:
        if value is None:
            scores.append(None)
            continue
        value = float(value)
        # NaN is "doesn't apply", like None; an infinity is out of range like any huge score.
        if math.isnan(value):
            scores.append(None)
            continue
        if abs(value) > MAX_SCORE:
            raise ValueError("returned %g, outside -1,000,000 to 1,000,000" % value)
        scores.append(value)
    return scores


LOG_LIMIT_CHARS = 1024 * 1024


class _BoundedLog:
    """worker.log up to a cap, then output is dropped: a reward printing in a loop must not fill
    the disk. Only the tail is ever read back."""

    def __init__(self, path: str, limit: int):
        self._file = open(path, "a", encoding = "utf-8", buffering = 1)
        self._left = limit

    def write(self, text: str) -> int:
        if self._left > 0:
            kept = text[: self._left]
            self._left -= len(kept)
            self._file.write(kept)
            if self._left <= 0:
                self._file.write("\n[worker.log is full; later output is dropped]\n")
        return len(text)

    def __getattr__(self, name):
        return getattr(self._file, name)


_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def _limit_job_memory(kernel32, handle, limit_bytes):
    """RLIMIT_AS for the Windows reward job: a per-process commit cap on every process in it.
    Windows has nothing like RLIMIT_FSIZE, so there is no file-size cap there."""
    if not limit_bytes or not handle:
        return False
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _Io(ctypes.Structure):
        _fields_ = [
            (n, ctypes.c_uint64)
            for n in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _Io),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = _Extended()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(limit_bytes)
    return bool(
        kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
    )


def _cap_own_memory(limit_bytes, kernel32 = None):
    """Puts this worker in a job of its own with the memory cap, before any reward code runs.

    The host also caps the job it puts the launched process in, but under MXC that process is the
    launcher, which may start this worker before the host's job exists, and job membership is
    fixed when a process is created."""
    if not limit_bytes or (kernel32 is None and os.name != "nt"):
        return False
    import ctypes
    from ctypes import wintypes

    if kernel32 is None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job or not _limit_job_memory(kernel32, job, limit_bytes):
        return False
    # The handle stays open for the worker's life; the job goes when the worker does.
    return bool(kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()))


def main():
    workdir = os.getcwd()
    # Notebook rewards print a lot; a pipe nobody drains would block them.
    log = _BoundedLog(os.path.join(workdir, "worker.log"), LOG_LIMIT_CHARS)
    sys.stdout = sys.stderr = log
    setup_path = os.path.join(workdir, "rewards.json")
    # Written once the host has this worker in its Windows job, so no reward starts a child
    # outside it.
    deadline = time.monotonic() + SETUP_TIMEOUT_SECONDS
    while not os.path.exists(setup_path):
        if os.path.exists(os.path.join(workdir, "stop")) or time.monotonic() > deadline:
            return 0
        time.sleep(POLL_SECONDS)
    with open(setup_path, encoding = "utf-8") as f:
        setup = json.load(f)
    specs = setup["specs"]
    limit = setup.get("memory_limit_bytes")
    try:
        capped = _cap_own_memory(limit)
    except Exception:  # noqa: BLE001 - refused below
        capped = False
    if os.name == "nt" and limit and not capped:
        # Under MXC the host's job may hold only the launcher, so nothing else caps this worker:
        # no reward code runs, and each reward reports why.
        refusal = "could not apply the Sandbox memory limit to the reward worker"
        specs, refused = [], {spec["name"]: refusal for spec in specs}
    else:
        refused = {}
    try:
        _install_unsloth_helpers(setup.get("unsloth_helpers"))
    except Exception as exc:  # noqa: BLE001 - only rewards that use them will fail, by name
        print("Could not load the unsloth notebook helpers:", _error_line(exc))
    funcs, errors = _load(specs)
    errors.update(refused)
    put(os.path.join(workdir, "ready.json"), {"loaded": sorted(funcs), "errors": errors})
    seq = 0
    alive_path = os.path.join(workdir, "alive")
    next_check = 0.0
    while True:
        req_path = os.path.join(workdir, "req-%d.json" % seq)
        if not os.path.exists(req_path):
            if os.path.exists(os.path.join(workdir, "stop")):
                return 0
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 1.0
                try:
                    quiet = time.time() - os.path.getmtime(alive_path)
                except OSError:
                    return 0
                if quiet > ALIVE_TIMEOUT_SECONDS:
                    return 0
            time.sleep(POLL_SECONDS)
            continue
        request = take(req_path)
        result = {"scores": {}, "errors": {}}
        for name in request.get("rewards") or []:
            fn = funcs.get(name)
            if fn is None:
                result["errors"][name] = errors.get(name, "reward is not loaded")
                continue
            try:
                # Each reward gets its own copy: one that edits its inputs in place must not
                # change what the next one scores (training calls them one request each).
                result["scores"][name] = _score(fn, copy.deepcopy(request))
            except BaseException as exc:  # noqa: BLE001 - one bad reward must not end the run's worker
                result["errors"][name] = _error_line(exc)
        put(os.path.join(workdir, "res-%d.json" % seq), result)
        seq += 1


if __name__ == "__main__":
    sys.exit(main())
