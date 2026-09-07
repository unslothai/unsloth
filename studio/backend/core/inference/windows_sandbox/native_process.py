# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Create the fixed bootstrap pair under native ownership, without resuming it.

The caller must admit/lease the helper and assemble its private channels first.
This module grants no runtime trust or qualification and emits no launch record.
The returned LPAC process remains suspended until its runtime lease is bound.
"""

import ctypes
from ctypes import wintypes as W
import math
import subprocess
import sys
import time

from .host_config import _path
from .profiles import PYTHON_PROFILE, WindowsRuntimeError


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("sid", ctypes.c_void_p), ("attributes", W.DWORD)]


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_NATIVE_LAUNCH_FAILED", message)


def _check(deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CANCELLED", "Native creation was cancelled.")
    if time.monotonic() >= deadline:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_STARTUP_TIMEOUT", "Native creation expired.")


def query_only_token_acl(api, token, sid):
    """The startup thread may query its token, never duplicate or reattach it."""
    from . import native_compat as lpac

    api.advapi32.GetSecurityInfo.argtypes = [
        W.HANDLE,
        W.DWORD,
        W.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    api.advapi32.GetSecurityInfo.restype = W.DWORD
    api.advapi32.SetSecurityInfo.argtypes = [
        W.HANDLE,
        W.DWORD,
        W.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    api.advapi32.SetSecurityInfo.restype = W.DWORD
    old_acl, descriptor, new_acl = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    try:
        error = api.advapi32.GetSecurityInfo(
            token, 6, 4, None, None, ctypes.byref(old_acl), None, ctypes.byref(descriptor)
        )
        if error:
            raise lpac._winerror("GetSecurityInfo(startup token)", error)
        trustee = lpac._TRUSTEE_W(None, 0, 0, 0, ctypes.cast(sid, W.LPWSTR))
        entry = lpac._EXPLICIT_ACCESS_W(8, 1, 0, trustee)  # TOKEN_QUERY, GRANT_ACCESS
        error = api.advapi32.SetEntriesInAclW(
            1, ctypes.byref(entry), old_acl, ctypes.byref(new_acl)
        )
        if error:
            raise lpac._winerror("SetEntriesInAclW(startup token)", error)
        error = api.advapi32.SetSecurityInfo(token, 6, 4, None, None, new_acl, None)
        if error:
            raise lpac._winerror("SetSecurityInfo(startup token)", error)
    finally:
        if new_acl:
            api.kernel32.LocalFree(new_acl)
        if descriptor:
            api.kernel32.LocalFree(descriptor)


def _reap(process):
    process.reap(timeout = 5)
    process.close()


def close_unreturned_process(job, process_handle, thread_handle):
    """Retain raw creation ownership when no process adapter could be returned."""
    from . import native_compat as lpac

    api = lpac._api().kernel32
    pending = [handle for handle in (process_handle, thread_handle) if handle]
    try:
        if process_handle:
            if not job.terminate() or api.WaitForSingleObject(process_handle, 5000) != 0:
                raise _invalid("Unreturned native process could not be reaped.")
        while pending:
            if not api.CloseHandle(pending[-1]):
                raise lpac._winerror("CloseHandle(unreturned native creation)")
            pending.pop()
        job.close()
    except BaseException as error:
        failure = WindowsRuntimeError(
            "WINDOWS_SANDBOX_CLEANUP_FAILED", "Native creation retained raw ownership."
        )
        failure.retained_job = job
        failure.retained_native_handles = tuple(pending)
        raise failure from error


def _create(binary, argv, identity, env, directory, handles, stdin, stdout, capability):
    """Only the fixed pair owner calls this: capability is donor-only or None."""
    from . import native_compat as lpac

    api = lpac._api().kernel32
    job = lpac._create_job(None, active_process_limit = 1)
    process, pointer = None, None
    info = lpac._PROCESS_INFORMATION()
    try:
        security = lpac._SECURITY_CAPABILITIES(
            identity.sid,
            ctypes.cast(capability, ctypes.c_void_p) if capability is not None else None,
            int(capability is not None),
            0,
        )
        values = [(0x20009, security), (0x2000D, (W.HANDLE * 1)(job._handle))]
        if capability is None:
            values += [(0x2000F, W.DWORD(1)), (0x20002, (W.HANDLE * len(handles))(*handles))]
        size = ctypes.c_size_t()
        api.InitializeProcThreadAttributeList(None, len(values), 0, ctypes.byref(size))
        if ctypes.get_last_error() != 122 or not 0 < size.value <= 65536:
            raise lpac._winerror("InitializeProcThreadAttributeList(native size)")
        storage = ctypes.create_string_buffer(size.value)
        candidate = ctypes.cast(storage, ctypes.c_void_p)
        if not api.InitializeProcThreadAttributeList(candidate, len(values), 0, ctypes.byref(size)):
            raise lpac._winerror("InitializeProcThreadAttributeList(native)")
        pointer = candidate
        for key, value in values:
            if not api.UpdateProcThreadAttribute(
                pointer, 0, key, ctypes.byref(value), ctypes.sizeof(value), None, None
            ):
                raise lpac._winerror("UpdateProcThreadAttribute(native)")
        startup = lpac._STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.lpAttributeList = pointer
        if capability is None:
            startup.StartupInfo.dwFlags = 0x100
            startup.StartupInfo.hStdInput = stdin
            startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = stdout
        if not api.CreateProcessW(
            binary,
            ctypes.create_unicode_buffer(subprocess.list2cmdline(argv)),
            None,
            None,
            capability is None,
            0x4 | 0x400 | 0x80000 | 0x08000000,
            lpac._environment_block(lpac._initial_appcontainer_environment(env, identity)),
            directory,
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(lpac._STARTUPINFOW)),
            ctypes.byref(info),
        ):
            raise lpac._winerror("CreateProcessW(native bootstrap)")
        process = lpac.WindowsLpacProcess(
            argv, info.hProcess, info.hThread, info.dwProcessId, None, job
        )
        return process
    finally:
        if pointer is not None:
            api.DeleteProcThreadAttributeList(pointer)
        if process is None:
            # JOB_LIST owns even the interval before the process adapter exists.
            close_unreturned_process(job, info.hProcess, info.hThread)


def create_suspended_host(
    binary,
    argv,
    identity,
    env,
    directory,
    *,
    stdin,
    stdout,
    control_handles,
    timeout = 30,
    cancel = None,
):
    """Return one suspended, Job1-owned LPAC target with its startup token attached.

    Inputs belong to the trusted launch assembler. No request can select a donor
    capability, opt-out policy, creation flags or process limit. The caller owns
    the supplied handles; they must already be inheritable, and remain theirs on
    failure. We never resume the donor or target, or close caller-owned channels.
    """
    from . import native_compat as lpac

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise _invalid("Invalid native creation timeout.")
    deadline = time.monotonic() + timeout
    _check(deadline, cancel)
    if (
        PYTHON_PROFILE.startup_capabilities != ("registryRead",)
        or PYTHON_PROFILE.payload_capabilities != ()
        or PYTHON_PROFILE.active_process_limit != 1
    ):
        raise _invalid("The declared bootstrap policy does not match the native creation contract.")
    _path(binary)
    _path(directory)
    if (
        type(argv) is not tuple
        or not argv
        or argv[0] != binary
        or any(type(arg) is not str or "\0" in arg for arg in argv)
    ):
        raise _invalid("Invalid native helper arguments.")
    if len(subprocess.list2cmdline(argv).encode("utf-16-le")) // 2 >= 32767:
        raise _invalid("Native helper command exceeds the Windows bound.")
    if type(control_handles) is not tuple or not 1 <= len(control_handles) <= 4:
        raise _invalid("Invalid native control handle inventory.")
    handles = (stdin, stdout, *control_handles)
    if any(type(handle) is not int or not 0 < handle < 2**63 for handle in handles) or len(
        set(handles)
    ) != len(handles):
        raise _invalid("Native inherited handles must be distinct real handles.")
    api = lpac._api()
    api.kernel32.GetHandleInformation.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    api.kernel32.GetHandleInformation.restype = W.BOOL
    for handle in handles:
        flags = W.DWORD()
        if (
            not api.kernel32.GetHandleInformation(handle, ctypes.byref(flags))
            or not flags.value & 1
        ):
            raise _invalid("A native channel is invalid or not inheritable.")
    api.advapi32.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, ctypes.POINTER(W.HANDLE)]
    api.advapi32.OpenProcessToken.restype = W.BOOL
    api.advapi32.DuplicateTokenEx.argtypes = [
        W.HANDLE,
        W.DWORD,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(W.HANDLE),
    ]
    api.advapi32.DuplicateTokenEx.restype = W.BOOL
    api.advapi32.SetThreadToken.argtypes = [ctypes.POINTER(W.HANDLE), W.HANDLE]
    api.advapi32.SetThreadToken.restype = W.BOOL
    derive = ctypes.WinDLL(
        "kernelbase", use_last_error = True, winmode = 0x800
    ).DeriveCapabilitySidsFromName
    derive.argtypes = [
        W.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(W.DWORD),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(W.DWORD),
    ]
    derive.restype = W.BOOL
    groups, caps = ctypes.c_void_p(), ctypes.c_void_p()
    group_count, cap_count = W.DWORD(), W.DWORD()
    donor = target = None
    tokens = []
    try:
        if (
            not derive(
                "registryRead",
                ctypes.byref(groups),
                ctypes.byref(group_count),
                ctypes.byref(caps),
                ctypes.byref(cap_count),
            )
            or cap_count.value != 1
        ):
            raise lpac._winerror("DeriveCapabilitySidsFromName(registryRead)")
        sid = ctypes.cast(caps, ctypes.POINTER(ctypes.c_void_p))[0]
        capability = (_SidAndAttributes * 1)(_SidAndAttributes(sid, 4))
        _check(deadline, cancel)
        donor = _create(binary, (binary,), identity, env, directory, (), None, None, capability)
        token, duplicate = W.HANDLE(), W.HANDLE()
        if not api.advapi32.OpenProcessToken(donor._handle, 2 | 8, ctypes.byref(token)):
            raise lpac._winerror("OpenProcessToken(startup donor)")
        tokens.append(token.value)
        if not api.advapi32.DuplicateTokenEx(
            token, 4 | 8 | 0x20000 | 0x40000, None, 2, 2, ctypes.byref(duplicate)
        ):
            raise lpac._winerror("DuplicateTokenEx(startup donor)")
        tokens.append(duplicate.value)
        query_only_token_acl(api, duplicate, identity.sid)
        _check(deadline, cancel)
        target = _create(binary, argv, identity, env, directory, handles, stdin, stdout, None)
        _check(deadline, cancel)
        if not api.advapi32.SetThreadToken(
            ctypes.byref(W.HANDLE(target._thread_handle)), duplicate
        ):
            raise lpac._winerror("SetThreadToken(native startup)")
        while tokens:
            if not api.kernel32.CloseHandle(tokens[-1]):
                raise lpac._winerror("CloseHandle(startup token)")
            tokens.pop()
        _reap(donor)
        donor = None
        _check(deadline, cancel)
        result, target = target, None
        return result
    finally:
        original = sys.exception()
        retained = []
        for process in (target, donor):
            if process is not None:
                try:
                    _reap(process)
                except Exception:
                    retained.append(process)
        token_errors = []
        for token in tokens:
            if not api.kernel32.CloseHandle(token):
                token_errors.append(token)
        for pointer, count in ((groups, group_count), (caps, cap_count)):
            if pointer:
                for index in range(count.value):
                    api.kernel32.LocalFree(
                        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[index]
                    )
                api.kernel32.LocalFree(pointer)
        if retained or token_errors:
            failure = WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED", "Native creation cleanup retained ownership."
            )
            failure.retained_processes = tuple(retained)
            failure.retained_token_handles = tuple(token_errors)
            # Target adapter allocation can fail before returning its handles.
            # A second cleanup failure must not discard that earlier owner.
            if getattr(original, "retained_job", None) is not None:
                failure.retained_job = original.retained_job
                failure.retained_native_handles = original.retained_native_handles
            raise failure
