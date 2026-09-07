# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Owned, fixed native gate fixture. Not a production launcher or qualification.

The same-package donor is never resumed. It is terminated and reaped, and every
token handle is closed, before the target's initial thread is resumed. Only the
reviewed compiled driver executes. The optional Python-host fixture admits a
selected development runtime and runs its fixed test payload only after the gate.
"""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.content_files import native_files
from core.inference.windows_sandbox.protocol import LaunchBinding, new_launch_nonce
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE
from core.inference.windows_sandbox.native_process import (
    create_suspended_host,
    query_only_token_acl,
)


class SidAndAttributes(ctypes.Structure):
    _fields_ = [("sid", ctypes.c_void_p), ("attributes", W.DWORD)]


class SecurityAttributes(ctypes.Structure):
    _fields_ = [("size", W.DWORD), ("descriptor", ctypes.c_void_p), ("inherit", W.BOOL)]


def pipe():
    api = lpac._api().kernel32
    api.CreatePipe.argtypes = [
        ctypes.POINTER(W.HANDLE),
        ctypes.POINTER(W.HANDLE),
        ctypes.POINTER(SecurityAttributes),
        W.DWORD,
    ]
    api.CreatePipe.restype = W.BOOL
    security = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, False)
    reader, writer = W.HANDLE(), W.HANDLE()
    if not api.CreatePipe(ctypes.byref(reader), ctypes.byref(writer), ctypes.byref(security), 4096):
        raise lpac._winerror("CreatePipe")
    return reader.value, writer.value


@contextmanager
def native_launch(
    binary,
    directory,
    *,
    attach_token = True,
    capability = "registryRead",
    limit = 1,
    optout = True,
    aap_granted = True,
    hold_status_writer = False,
    thread_mode = None,
    configure_host = None,
    transform_config = None,
    environment = None,
    production_creation = False,
):
    api, files = lpac._api(), native_files()
    directory = directory.resolve()
    driver, workdir, aap = directory / "driver.exe", directory / "work", directory / "aap-only"
    files.create(driver, binary.read_bytes())
    files.mkdir(workdir)
    # Unlike immutable store directories, a session directory must pass the
    # host owner's rights to child-created output files after SID revocation.
    handle = files.open(workdir, directory = True, write_dac = True)
    try:
        files.set_owned_dacl(handle, files.private_sddl.replace("(A;;", "(A;OICI;"))
    finally:
        files.kernel.CloseHandle(handle)
    files.create(aap, b"positive AAP control")
    if aap_granted:
        handle = files.open(aap, write_dac = True)
        try:
            files.set_owned_dacl(handle, files.private_sddl + "(A;;FR;;;S-1-15-2-1)")
        finally:
            files.kernel.CloseHandle(handle)
    api.kernel32.SetHandleInformation.argtypes = [W.HANDLE, W.DWORD, W.DWORD]
    api.kernel32.SetHandleInformation.restype = W.BOOL
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
    identity, target, runtime_access, output_stream = None, None, None, None
    from core.inference.windows_sandbox.identity import InvocationRecipe, InvocationReservation

    reservation = InvocationReservation(InvocationRecipe.new())
    processes, jobs, handles = [], [], set()
    try:
        identity = reservation.create(str(workdir))
        identity.granted_roots = (*identity.granted_roots, str(driver))
        lpac._grant_read_execute(str(driver), identity.sid)
        lpac._grant_modify(str(workdir), identity.sid)
        for root in identity.traverse_roots:
            lpac._grant_traverse(root, identity.sid)
        if (
            not derive(
                capability,
                ctypes.byref(groups),
                ctypes.byref(group_count),
                ctypes.byref(caps),
                ctypes.byref(cap_count),
            )
            or cap_count.value != 1
        ):
            raise lpac._winerror("DeriveCapabilitySidsFromName")
        sid = ctypes.cast(caps, ctypes.POINTER(ctypes.c_void_p))[0]
        attributes = (SidAndAttributes * 1)(SidAndAttributes(sid, 4))
        status_read, status_write = pipe()
        handles.update((status_read, status_write))
        ack_read, ack_write = pipe()
        handles.update((ack_read, ack_write))
        null = files.kernel.CreateFileW("NUL", 0x80000000 | 0x40000000, 3, None, 3, 0, None)
        if null == ctypes.c_void_p(-1).value:
            raise lpac._winerror("CreateFileW(NUL)")
        handles.add(null)
        for handle in (status_write, ack_read, null):
            if not api.kernel32.SetHandleInformation(handle, 1, 1):
                raise lpac._winerror("SetHandleInformation")
        nonce, profile, content = (
            new_launch_nonce(),
            bytes.fromhex(PYTHON_PROFILE.digest),
            b"c" * 32,
        )
        sentinel = workdir / "payload-sentinel"
        argv = (
            str(driver),
            str(status_write),
            str(ack_read),
            (nonce + profile + content).hex(),
            identity.sid_string,
            str(aap),
            str(sentinel),
        )
        if thread_mode is not None:
            argv += (thread_mode,)
        extra_inherited = []
        child_output = null
        if configure_host is not None:
            # Fixed Python-host test assembly, not a production callback API.
            config, runtime_access = configure_host(
                SimpleNamespace(
                    identity = identity,
                    driver = driver,
                    workdir = workdir,
                    aap = aap,
                    nonce = nonce,
                    profile = profile,
                )
            )
            content = config.content_digest
            config_file = directory / "host-config"
            encoded = config.encode()
            files.create(config_file, transform_config(encoded) if transform_config else encoded)
            config_handle = files.open(config_file)
            handles.add(config_handle)
            extra_inherited.append(config_handle)
            stdout_read, stdout_write = pipe()
            handles.update((stdout_read, stdout_write))
            import msvcrt

            output_stream = os.fdopen(msvcrt.open_osfhandle(stdout_read, os.O_RDONLY), "rb")
            handles.remove(stdout_read)
            child_output = stdout_write
            extra_inherited.append(stdout_write)
            for handle in extra_inherited:
                if not api.kernel32.SetHandleInformation(handle, 1, 1):
                    raise lpac._winerror("SetHandleInformation(host input/output)")
            argv = (str(driver), str(config_handle), str(status_write), str(ack_read))

        def create(donor):
            count, size = (1 if donor else 3), ctypes.c_size_t()
            api.kernel32.InitializeProcThreadAttributeList(None, count, 0, ctypes.byref(size))
            if ctypes.get_last_error() != 122 or not 0 < size.value <= 65536:
                raise lpac._winerror("attribute-size")
            buffer = ctypes.create_string_buffer(size.value)
            pointer = ctypes.cast(buffer, ctypes.c_void_p)
            if not api.kernel32.InitializeProcThreadAttributeList(
                pointer, count, 0, ctypes.byref(size)
            ):
                raise lpac._winerror("attribute-init")
            try:
                security = lpac._SECURITY_CAPABILITIES(
                    identity.sid,
                    ctypes.cast(attributes, ctypes.c_void_p) if donor else None,
                    int(donor),
                    0,
                )
                values = [(0x20009, security)]
                if not donor:
                    child_handles = (status_write, ack_read, null, *extra_inherited)
                    values += [
                        (0x2000F, W.DWORD(int(optout))),
                        (0x20002, (W.HANDLE * len(child_handles))(*child_handles)),
                    ]
                for key, value in values:
                    if not api.kernel32.UpdateProcThreadAttribute(
                        pointer, 0, key, ctypes.byref(value), ctypes.sizeof(value), None, None
                    ):
                        raise lpac._winerror("launch attribute")
                startup = lpac._STARTUPINFOEXW()
                startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.lpAttributeList = pointer
                if not donor:
                    startup.StartupInfo.dwFlags = 0x100
                    startup.StartupInfo.hStdInput = null
                    startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = child_output
                info = lpac._PROCESS_INFORMATION()
                command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
                env = lpac._environment_block(
                    lpac._initial_appcontainer_environment(
                        {
                            "SystemRoot": os.environ["SystemRoot"],
                            "LOCALAPPDATA": identity.profile_folder,
                            "TEMP": identity.private_temp,
                            "TMP": identity.private_temp,
                            **(environment or {}),
                        },
                        identity,
                    )
                )
                if not api.kernel32.CreateProcessW(
                    str(driver),
                    command,
                    None,
                    None,
                    not donor,
                    0x80000 | 0x400 | 4 | 0x08000000,
                    env,
                    identity.private_temp,
                    ctypes.cast(ctypes.byref(startup), ctypes.POINTER(lpac._STARTUPINFOW)),
                    ctypes.byref(info),
                ):
                    raise lpac._winerror("CreateProcessW")
                processes.append(info)
                job = lpac._create_job(info.hProcess, active_process_limit = 1 if donor else limit)
                jobs.append(job)
                return info, job
            finally:
                api.kernel32.DeleteProcThreadAttributeList(pointer)

        donor_pid = None
        if production_creation:
            assert configure_host is not None and attach_token and optout and limit == 1
            assert capability == "registryRead" and thread_mode is None
            target = create_suspended_host(
                str(driver),
                argv,
                identity,
                {
                    "SystemRoot": os.environ["SystemRoot"],
                    "TEMP": identity.private_temp,
                    "TMP": identity.private_temp,
                    "LOCALAPPDATA": identity.profile_folder,
                    **(environment or {}),
                },
                identity.private_temp,
                stdin = null,
                stdout = child_output,
                control_handles = (status_write, ack_read, config_handle),
            )
            target.stdout = output_stream
            info = lpac._PROCESS_INFORMATION(target._handle, target._thread_handle, target.pid, 0)
            processes.append(info)
            jobs.append(target._unsloth_job)
        else:
            donor, _ = create(True)
            donor_pid = donor.dwProcessId
            token, duplicate = W.HANDLE(), W.HANDLE()
            if not api.advapi32.OpenProcessToken(donor.hProcess, 2 | 8, ctypes.byref(token)):
                raise lpac._winerror("OpenProcessToken")
            handles.add(token.value)
            if not api.advapi32.DuplicateTokenEx(
                token, 4 | 8 | 0x20000 | 0x40000, None, 2, 2, ctypes.byref(duplicate)
            ):
                raise lpac._winerror("DuplicateTokenEx")
            handles.add(duplicate.value)
            query_only_token_acl(api, duplicate, identity.sid)
            info, job = create(False)
            if attach_token and not api.advapi32.SetThreadToken(
                ctypes.byref(W.HANDLE(info.hThread)), duplicate
            ):
                raise lpac._winerror("SetThreadToken")
            for handle in (token.value, duplicate.value):
                if not api.kernel32.CloseHandle(handle):
                    raise lpac._winerror("CloseHandle(startup token)")
                handles.remove(handle)
            if (
                not api.kernel32.TerminateProcess(donor.hProcess, 0)
                or api.kernel32.WaitForSingleObject(donor.hProcess, 5000) != 0
            ):
                raise RuntimeError("Never-resumed donor did not terminate")
            target = lpac.WindowsLpacProcess(
                argv, info.hProcess, info.hThread, info.dwProcessId, output_stream, job
            )
        if runtime_access is not None:
            runtime_access.bind_process(target)
        if api.kernel32.ResumeThread(info.hThread) != 1:
            raise lpac._winerror("ResumeThread")
        close_after_spawn = [ack_read, null, *extra_inherited]
        if not hold_status_writer:
            close_after_spawn.append(status_write)
        for handle in close_after_spawn:
            if not api.kernel32.CloseHandle(handle):
                raise lpac._winerror("CloseHandle(child channel)")
            handles.remove(handle)
        yield SimpleNamespace(
            process = target,
            binding = LaunchBinding(info.dwProcessId, nonce, profile, content),
            status = status_read,
            acknowledgement = ack_write,
            handles = handles,
            sentinel = sentinel,
            identity = identity,
            donor_pid = donor_pid,
        )
    finally:
        # No profile/ACL removal before every owned native process is reaped.
        for info in reversed(processes):
            api.kernel32.TerminateProcess(info.hProcess, 1)
            if api.kernel32.WaitForSingleObject(info.hProcess, 5000) != 0:
                raise RuntimeError("Owned native gate process survived cleanup")
        if runtime_access is not None:
            runtime_access.close()
            if target is not None and target._handle is None:
                processes = [info for info in processes if info.dwProcessId != target.pid]
        for job in jobs:
            job.close()
        for info in processes:
            api.kernel32.CloseHandle(info.hThread)
            api.kernel32.CloseHandle(info.hProcess)
        for handle in handles:
            api.kernel32.CloseHandle(handle)
        if output_stream is not None:
            output_stream.close()
        for pointer, count in ((groups, group_count), (caps, cap_count)):
            if pointer:
                for index in range(count.value):
                    api.kernel32.LocalFree(
                        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[index]
                    )
                api.kernel32.LocalFree(pointer)
        reservation.cleanup()
        if identity is not None:
            assert identity.cleaned and not Path(identity.manifest_path).exists()
