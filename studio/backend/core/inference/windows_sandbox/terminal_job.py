# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Retain a Terminal Job until every member exits, independently of Python policy."""

import ctypes
from ctypes import wintypes as W
import threading
import time

from . import terminal_native as lpac
from .profiles import WindowsRuntimeError


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", W.DWORD),
        ("TotalProcesses", W.DWORD),
        ("ActiveProcesses", W.DWORD),
        ("TotalTerminatedProcesses", W.DWORD),
    ]


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_TERMINAL_JOB_INVALID", message)


def _kernel():
    api = lpac._api().kernel32
    api.QueryInformationJobObject.argtypes = [
        W.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        W.DWORD,
        ctypes.POINTER(W.DWORD),
    ]
    api.QueryInformationJobObject.restype = W.BOOL
    api.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)]
    api.IsProcessInJob.restype = W.BOOL
    api.DuplicateHandle.argtypes = [
        W.HANDLE,
        W.HANDLE,
        W.HANDLE,
        ctypes.POINTER(W.HANDLE),
        W.DWORD,
        W.BOOL,
        W.DWORD,
    ]
    api.DuplicateHandle.restype = W.BOOL
    api.GetCurrentProcess.argtypes, api.GetCurrentProcess.restype = [], W.HANDLE
    api.GetHandleInformation.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    api.GetHandleInformation.restype = W.BOOL
    return api


def _query(api, job, kind, value):
    if not job:
        # NULL queries the calling process's Job, not this invocation's Job.
        raise _invalid("Terminal Job ownership handle is unavailable.")
    size = W.DWORD()
    if not api.QueryInformationJobObject(
        job, kind, ctypes.byref(value), ctypes.sizeof(value), ctypes.byref(size)
    ) or size.value != ctypes.sizeof(value):
        raise _invalid("Terminal Job information could not be verified.")
    return value


class TerminalJobOwner:
    """Bind before ResumeThread; release runtime access only after cleanup succeeds.

    The extra Job handle is broker-only. It survives the process adapter closing
    its original handles, allowing a kernel-owned membership check afterward.
    A shell leader exit alone never establishes that its children have exited.
    """

    def __init__(self):
        self.job = None
        self.process = None
        self.attempted = self.closed = False
        self.lock = threading.Lock()

    def bind_process(self, process):
        if self.attempted or self.closed:
            raise _invalid("Terminal Job ownership cannot be rebound.")
        self.attempted = True
        if not isinstance(process, lpac.WindowsLpacProcess):
            raise _invalid("Terminal requires its native Windows process owner.")
        self.process = process
        api, duplicate = _kernel(), W.HANDLE()
        job = process._unsloth_job._handle
        current = api.GetCurrentProcess()
        if (
            not job
            or not process._handle
            or not api.DuplicateHandle(current, job, current, ctypes.byref(duplicate), 0, False, 2)
            or not duplicate.value
        ):
            raise _invalid("Terminal could not retain its Job handle.")
        self.job = duplicate.value
        flags, belongs = W.DWORD(), W.BOOL()
        limits = _query(
            api, self.job, 9, lpac._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        ).BasicLimitInformation
        if (
            not api.GetHandleInformation(self.job, ctypes.byref(flags))
            or flags.value & 1
            or not limits.LimitFlags & 0x2000
            or limits.LimitFlags & (0x800 | 0x1000)
            or not limits.LimitFlags & 8
            or limits.ActiveProcessLimit < 1
            or not api.IsProcessInJob(process._handle, self.job, ctypes.byref(belongs))
            or not belongs.value
        ):
            raise _invalid(
                "Terminal requires a non-inherited kill-on-close, no-breakaway, bounded Job."
            )

    def cleanup(self):
        acquired = self.lock.acquire(timeout = 5)
        try:
            if not acquired:
                raise _invalid("Terminal Job cleanup is still in progress.")
            if self.closed:
                return
            if self.job is None and self.process is None:
                self.closed = True
                return
            api = _kernel()
            if self.job is not None and not api.TerminateJobObject(self.job, 1):
                raise _invalid("Terminal Job termination failed; runtime access is retained.")
            if self.process is not None:
                # Release process references before relying on the accounting
                # count. Retain the duplicate Job if closing any handle fails.
                self.process.close()
                self.process = None
            if self.job is not None:
                deadline = time.monotonic() + 5
                while _query(api, self.job, 1, _Accounting()).ActiveProcesses:
                    if time.monotonic() >= deadline:
                        raise _invalid(
                            "Terminal descendants have not exited; runtime access is retained."
                        )
                    time.sleep(0.01)
                if not api.CloseHandle(self.job):
                    raise _invalid("Terminal Job handle could not be closed.")
                self.job = None
            self.closed = True
        finally:
            if acquired:
                self.lock.release()
