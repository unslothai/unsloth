# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native read-grant and process-lease controls, not bootstrap qualification."""

from dataclasses import replace
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_content_store import snapshot, _start_control, BACKEND  # noqa: F401
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.content_access import (
    appcontainer_sid,
    read_readers,
    validate_acl,
    READ_EXECUTE,
    TRAVERSE,
)
from core.inference.windows_sandbox.content_files import PathLease
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
from core.inference.windows_sandbox import native_compat as windows_lpac


@pytest.fixture
def identities(snapshot):
    # Real fresh profiles, never a reused capability SID or shared package SID.
    owned = []
    try:
        for _ in range(2):
            from core.inference.windows_sandbox.identity import (
                InvocationRecipe,
                InvocationReservation,
            )

            reservation = InvocationReservation(InvocationRecipe.new())
            owned.append(reservation)
            reservation.create_private()
        yield [reservation.identity for reservation in owned]
    finally:
        for reservation in reversed(owned):
            reservation.cleanup()


def grants(store, path):
    with PathLease() as pins:
        readers = read_readers(store, pins)
        handle = pins.directory(path) if path.is_dir() else pins.file(path)
        return validate_acl(store, handle, path, readers)


def test_two_readers_share_content_but_not_write_access(snapshot, identities):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    first, second = (identity.sid_string for identity in identities)
    with store.read_access(digest, first) as one:
        path = one.generation.files[0]
        assert grants(store, path) == {first: READ_EXECUTE}
        assert grants(store, store.root) == {first: TRAVERSE}
        with store.read_access(digest, second):
            assert grants(store, path) == {first: READ_EXECUTE, second: READ_EXECUTE}
            with store.lease(digest):
                pass
            assert store.publish(spec) == digest
            assert store.recover_readers() == 0
            assert grants(store, one.generation.directory / "manifest.json") == {}
            assert grants(store, store.root / ".store") == {}
        assert grants(store, path) == {first: READ_EXECUTE}
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.collect(digest)
    assert grants(store, path) == {}
    assert not list((store.root / ".readers").iterdir())
    store.collect(digest)


def test_read_grants_do_not_reach_another_generation(snapshot, identities):
    store, spec, _ = snapshot
    first = store.publish(spec)
    second = store.publish(replace(spec, helper_digest = "a" * 64))
    with store.read_access(first, identities[0].sid_string):
        with store.lease(second) as generation:
            assert grants(store, generation.files[0]) == {}


def test_invocation_sid_cannot_be_reused_or_lease_reentered(snapshot, identities):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    access = store.read_access(digest, identities[0].sid_string)
    with access:
        with pytest.raises(WindowsRuntimeError, match = "reused"):
            with store.read_access(digest, identities[0].sid_string):
                pytest.fail("SID reused")
        with pytest.raises(WindowsRuntimeError, match = "reused"):
            access.__enter__()
    with pytest.raises(WindowsRuntimeError, match = "reused"):
        access.__enter__()
    access.close()


@pytest.mark.parametrize(
    "sid",
    [
        "S-1-1-0",
        "S-1-15-2-1",
        "S-1-15-3-1",
        "S-1-15-2-01-2-3-4-5-6-7",
        "S-1-15-2-4294967296-2-3-4-5-6-7",
        "S-1-15-2-1-2-3-4-5-6-7)(A;;FA;;;WD)",
    ],
)
def test_reader_sid_rejects_global_capability_and_profile_injection(sid):
    with pytest.raises(WindowsRuntimeError):
        appcontainer_sid(sid)


def test_partial_read_grant_failure_rolls_back_only_its_sid(snapshot, identities, monkeypatch):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    with store.read_access(digest, identities[0].sid_string) as first:
        original = store.api.set_owned_dacl
        injected = []

        def fail_after_update(handle, sddl):
            original(handle, sddl)
            if identities[1].sid_string in sddl and not injected:
                injected.append(True)
                raise OSError("injected grant failure")

        monkeypatch.setattr(store.api, "set_owned_dacl", fail_after_update)
        with pytest.raises(OSError, match = "injected grant failure"):
            with store.read_access(digest, identities[1].sid_string):
                pytest.fail("failed grant launched")
        assert injected
        assert grants(store, first.generation.files[0]) == {identities[0].sid_string: READ_EXECUTE}
        assert len(list((store.root / ".readers").glob("*.json"))) == 1


def test_unknown_host_acl_is_not_restored_during_cleanup(snapshot, identities):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    access = store.read_access(digest, identities[0].sid_string).__enter__()
    path = access.generation.files[0]
    original_sddl = None
    try:
        handle = store.api.open(path, write_dac = True)
        try:
            original_sddl = store.api.security_text(handle)
            store.api.set_owned_dacl(handle, original_sddl + "(A;;FR;;;WD)")
        finally:
            store.api.kernel.CloseHandle(handle)
        with pytest.raises(WindowsRuntimeError, match = "unowned"):
            access.close()
        with PathLease() as pins:
            assert "WD" in store.api.security_text(pins.file(path))
        assert list((store.root / ".readers").glob("*.json"))
    finally:
        # Test harness owns the injected ACE and removes it; production never
        # restores an earlier host descriptor to make cleanup appear successful.
        if original_sddl is not None:
            handle = store.api.open(path, write_dac = True)
            try:
                store.api.set_owned_dacl(handle, original_sddl)
            finally:
                store.api.kernel.CloseHandle(handle)
        access.close()


def test_abandoned_read_grant_recovery_keeps_live_other_reader(snapshot, identities):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    sid = identities[1].sid_string
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "from core.inference.windows_sandbox.content import RuntimeContentStore\n"
        f"store=RuntimeContentStore({str(store.root)!r})\n"
        f"with store.read_access({digest!r}, {sid!r}):\n"
        " print('READ_READY', flush=True)\n"
        " time.sleep(60)\n"
    )
    with store.read_access(digest, identities[0].sid_string) as one:
        child = _start_control(code, "READ_READY")
        try:
            assert store.recover_readers() == 0
            child.kill()
            child.communicate(timeout = 5)
            assert store.recover_readers() == 1
            assert grants(store, one.generation.files[0]) == {
                identities[0].sid_string: READ_EXECUTE
            }
            with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
                store.collect(digest)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout = 5)
    store.collect(digest)


def suspended_owner(tmp_path, limit = 1):
    import _winapi

    # Fixed host harness controls Job ordering, not LPAC Python compatibility.
    args = (sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)")
    process, thread, pid, _ = _winapi.CreateProcess(
        sys.executable,
        subprocess.list2cmdline(args),
        None,
        None,
        False,
        0x4 | 0x08000000,
        {"SystemRoot": os.environ["SystemRoot"], "TEMP": str(tmp_path), "TMP": str(tmp_path)},
        str(tmp_path),
        subprocess.STARTUPINFO(),
    )
    api = windows_lpac._api()
    try:
        job = windows_lpac._create_job(process, active_process_limit = limit)
        return windows_lpac.WindowsLpacProcess(args, process, thread, pid, None, job)
    except BaseException:
        api.kernel32.TerminateProcess(process, 1)
        api.kernel32.WaitForSingleObject(process, 5000)
        api.kernel32.CloseHandle(thread)
        api.kernel32.CloseHandle(process)
        raise


def test_runtime_lease_waits_for_native_job_process_before_revoking(
    snapshot, identities, tmp_path, monkeypatch
):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    process = suspended_owner(tmp_path)
    api = windows_lpac._api()
    observed = []
    original = store.api.set_owned_dacl
    try:
        with store.read_access(digest, identities[0].sid_string) as lease:
            lease.bind_process(process)
            assert api.kernel32.ResumeThread(process._thread_handle) == 1

            def check_reaped(handle, sddl):
                assert process.returncode is not None
                assert process._handle is None
                observed.append(True)
                original(handle, sddl)

            monkeypatch.setattr(store.api, "set_owned_dacl", check_reaped)
        assert observed
        store.collect(digest)
    finally:
        process.close()


def test_runtime_lease_rejects_multi_process_job(snapshot, identities, tmp_path):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    process = suspended_owner(tmp_path, limit = 2)
    try:
        with store.read_access(digest, identities[0].sid_string) as lease:
            with pytest.raises(WindowsRuntimeError, match = "single-process"):
                lease.bind_process(process)
    finally:
        process.terminate()
        process.wait(timeout = 5)
        process.close()


def test_failed_process_wait_retains_read_grants_and_file_leases(
    snapshot, identities, tmp_path, monkeypatch
):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    access = store.read_access(digest, identities[0].sid_string).__enter__()
    process = suspended_owner(tmp_path)
    try:
        access.bind_process(process)
        original_wait = process.wait

        def fail_wait(*_a, **_k):
            raise subprocess.TimeoutExpired("owned process", 5)

        monkeypatch.setattr(process, "wait", fail_wait)
        with pytest.raises(subprocess.TimeoutExpired):
            access.close()
        assert grants(store, access.generation.files[0])
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.collect(digest)
        monkeypatch.setattr(process, "wait", original_wait)
    finally:
        access.close()
        process.close()


@contextmanager
def lpac_filesystem_token(identity, *, lpac = True):
    """A real zero-capability LPAC token; the donor is never resumed.

    Only fixed Win32 filesystem checks impersonate the duplicate. This verifies
    ACL enforcement without pretending that suspended cmd.exe qualified a shell
    or that Python completed the native bootstrap.
    """
    api = windows_lpac._api()
    count, size = 2, ctypes.c_size_t()
    api.kernel32.InitializeProcThreadAttributeList(None, count, 0, ctypes.byref(size))
    assert ctypes.get_last_error() == 122 and size.value
    buffer = ctypes.create_string_buffer(size.value)
    attributes = ctypes.cast(buffer, ctypes.c_void_p)
    assert api.kernel32.InitializeProcThreadAttributeList(attributes, count, 0, ctypes.byref(size))
    info = windows_lpac._PROCESS_INFORMATION()
    token, duplicate, job = W.HANDLE(), W.HANDLE(), None
    try:
        capabilities = windows_lpac._SECURITY_CAPABILITIES(identity.sid, None, 0, 0)
        optout = W.DWORD(1 if lpac else 0)
        for key, value in ((0x20009, capabilities), (0x2000F, optout)):
            assert api.kernel32.UpdateProcThreadAttribute(
                attributes, 0, key, ctypes.byref(value), ctypes.sizeof(value), None, None
            )
        startup = windows_lpac._STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.lpAttributeList = attributes
        shell = str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe")
        command = ctypes.create_unicode_buffer(
            subprocess.list2cmdline([shell, "/d", "/c", "exit", "0"])
        )
        environment = windows_lpac._environment_block(
            {
                "SystemRoot": os.environ["SystemRoot"],
                "TEMP": identity.private_temp,
                "TMP": identity.private_temp,
                "LOCALAPPDATA": identity.profile_folder,
            }
        )
        assert api.kernel32.CreateProcessW(
            shell,
            command,
            None,
            None,
            False,
            0x80000 | 0x400 | 0x4 | 0x08000000,
            environment,
            identity.private_temp,
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(windows_lpac._STARTUPINFOW)),
            ctypes.byref(info),
        ), ctypes.get_last_error()
        job = windows_lpac._create_job(info.hProcess, active_process_limit = 1)
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
        query = api.advapi32.GetTokenInformation
        query.argtypes, query.restype = (
            [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD)],
            W.BOOL,
        )
        assert api.advapi32.OpenProcessToken(info.hProcess, 2 | 8, ctypes.byref(token))
        assert api.advapi32.DuplicateTokenEx(token, 4 | 8, None, 2, 2, ctypes.byref(duplicate))
        data, needed = ctypes.create_string_buffer(4096), W.DWORD()
        value = W.DWORD()
        assert query(token, 29, ctypes.byref(value), ctypes.sizeof(value), ctypes.byref(needed))
        assert value.value == 1
        value = W.DWORD()
        status_ok = bool(
            query(token, 46, ctypes.byref(value), ctypes.sizeof(value), ctypes.byref(needed))
        )
        status = {
            "available": status_ok,
            "value": value.value if status_ok else None,
            "error": 0 if status_ok else ctypes.get_last_error(),
        }
        if status_ok:
            assert bool(value.value) is lpac
        else:
            # Diagnostic only: ERROR_INVALID_PARAMETER is not an LPAC-status
            # pass. Mandatory AAP-positive/LPAC-negative file controls below
            # establish the narrower filesystem evidence independently.
            assert status["error"] == 87, status
        assert query(duplicate, 30, data, len(data), ctypes.byref(needed))
        assert ctypes.cast(data, ctypes.POINTER(W.DWORD))[0] == 0
        assert query(duplicate, 31, data, len(data), ctypes.byref(needed))
        sid = ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p))[0]
        assert windows_lpac._sid_string(api, sid) == identity.sid_string
        yield duplicate, status
    finally:
        for handle in (duplicate, token):
            if handle:
                api.kernel32.CloseHandle(handle)
        if info.hProcess:
            assert api.kernel32.TerminateProcess(info.hProcess, 1)
            assert api.kernel32.WaitForSingleObject(info.hProcess, 5000) == 0
            api.kernel32.CloseHandle(info.hThread)
            api.kernel32.CloseHandle(info.hProcess)
        if job is not None:
            job.close()
        api.kernel32.DeleteProcThreadAttributeList(attributes)


def open_under_token(
    store,
    token,
    path,
    access,
    *,
    directory = False,
    read = False,
):
    api = windows_lpac._api()
    setter = api.advapi32.SetThreadToken
    setter.argtypes, setter.restype = [ctypes.POINTER(W.HANDLE), W.HANDLE], W.BOOL
    revert = api.advapi32.RevertToSelf
    revert.argtypes, revert.restype = [], W.BOOL
    # Everything allocated before impersonation; assertions occur after revert.
    name = str(path)
    buffer, size = ctypes.create_string_buffer(1024), W.DWORD()
    handle = None
    assert setter(None, token)
    try:
        handle = store.api.kernel.CreateFileW(
            name, access, 7, None, 3, 0x02000000 if directory else 0, None
        )
        error = ctypes.get_last_error()
        ok = handle != ctypes.c_void_p(-1).value
        if ok and read:
            ok = bool(
                store.api.kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(size), None)
            )
            error = ctypes.get_last_error()
        result = (ok, error, buffer.raw[: size.value])
    finally:
        if handle and handle != ctypes.c_void_p(-1).value:
            store.api.kernel.CloseHandle(handle)
        if not revert():
            # Never continue pytest under a thread identity that failed to drop.
            os._exit(91)
    return result


def test_real_lpac_token_reads_only_granted_generation_and_cannot_modify_it(
    snapshot, identities, tmp_path, record_property
):
    store, spec, source = snapshot
    digest = store.publish(spec)
    other = store.publish(replace(spec, helper_digest = "a" * 64))
    path = store.root / digest / "files/Lib/sample.py"
    expected = source.read_bytes()
    aap_only = tmp_path / "all-applications-positive-control"
    store.api.create(aap_only, b"AAP control")
    handle = store.api.open(aap_only, write_dac = True)
    try:
        store.api.set_owned_dacl(handle, store.api.private_sddl + "(A;;FR;;;S-1-15-2-1)")
    finally:
        store.api.kernel.CloseHandle(handle)
    with lpac_filesystem_token(identities[0]) as (token, status):
        record_property("lpac_status_query", repr(status))
        with lpac_filesystem_token(identities[0], lpac = False) as (ordinary, _):
            control = open_under_token(store, ordinary, aap_only, 0x80000000, read = True)
            assert control[0] and control[2] == b"AAP control", control
            assert open_under_token(store, token, aap_only, 0x80000000)[:2] == (False, 5)
        record_property("aap_control", "ordinary AppContainer read succeeds; LPAC read denied")
        before = open_under_token(store, token, path, 0x80000000)
        assert before[:2] == (False, 5)
        with store.read_access(digest, identities[0].sid_string):
            allowed = open_under_token(store, token, path, 0x80000000, read = True)
            assert allowed[0] and allowed[2] == expected
            assert open_under_token(store, token, path, 0x20000000)[0]
            for denied, access, directory in (
                (path, 0x40000000, False),  # write
                (path, 0x10000, False),  # delete
                (path, 0x40000, False),  # WRITE_DAC
                (source, 0x80000000, False),
                (store.root / other / "files/Lib/sample.py", 0x80000000, False),
                (store.root / digest / "manifest.json", 0x80000000, False),
                (store.root, 1, True),  # directory listing, not traversal
                (store.root / ".readers", 1, True),
            ):
                result = open_under_token(store, token, denied, access, directory = directory)
                assert result[:2] == (False, 5), (denied, access, result)
        assert open_under_token(store, token, path, 0x80000000)[:2] == (False, 5)


@pytest.mark.parametrize(
    "attack", ["excessive", "inheritable", "unknown_sid", "unprotected", "inherited"]
)
def test_reader_dacl_validation_rejects_permission_expansion(snapshot, identities, attack):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    access = store.read_access(digest, identities[0].sid_string).__enter__()
    path = access.generation.files[0]
    if attack == "inheritable":
        path = path.parent
    handle = store.api.open(path, directory = path.is_dir(), write_dac = True)
    original = store.api.security_text(handle)
    try:
        if attack == "excessive":
            changed = store.api.private_sddl + f"(A;;FA;;;{identities[0].sid_string})"
        elif attack == "inheritable":
            changed = store.api.private_sddl + f"(A;OICI;FRFX;;;{identities[0].sid_string})"
        elif attack == "unknown_sid":
            changed = original + f"(A;;FRFX;;;{identities[1].sid_string})"
        else:
            # Pure parser controls: the native writer always sets protection,
            # and Windows clears ID when a protected DACL is applied. Do not
            # claim that writing ID produced a live inherited-ACE fixture.
            class ChangedDescriptor:
                def security_text(self, _handle):
                    return (
                        original.replace("D:P", "D:")
                        if attack == "unprotected"
                        else original.replace("(A;;", "(A;ID;", 1)
                    )

            from types import SimpleNamespace

            with (
                store._mutation() as readers,
                pytest.raises(WindowsRuntimeError),
            ):
                fake = SimpleNamespace(root = store.root, api = ChangedDescriptor())
                fake.api.owner = store.api.owner
                validate_acl(fake, handle, path, readers)
            return
        store.api.set_owned_dacl(handle, changed)
        assert store.api.security_text(handle) != original
        if attack == "inheritable":
            assert "OICI" in store.api.security_text(handle)
        with pytest.raises(WindowsRuntimeError):
            with store.lease(digest):
                pytest.fail("expanded runtime ACL accepted")
        assert store.api.security_text(handle) != original
    finally:
        store.api.set_owned_dacl(handle, original)
        store.api.kernel.CloseHandle(handle)
        access.close()


def test_missing_owner_journal_never_permits_a_new_reader(snapshot, identities):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    access = store.read_access(digest, identities[0].sid_string).__enter__()
    journal = store.root / ".readers" / (access.name + ".json")
    original = journal.read_bytes()
    try:
        journal.write_bytes(b'{"version":true}')
        with pytest.raises(WindowsRuntimeError, match = "schema"):
            with store.read_access(digest, identities[1].sid_string):
                pytest.fail("invalid ownership used for another launch")
        with pytest.raises(WindowsRuntimeError, match = "schema"):
            store.recover_readers()
        assert journal.read_bytes() == b'{"version":true}'
    finally:
        journal.write_bytes(original)
        access.close()


def test_lease_retains_ownership_after_journal_cleanup_failure(snapshot, identities, monkeypatch):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    access = store.read_access(digest, identities[0].sid_string).__enter__()
    original = Path.unlink

    def fail_journal(path, *args, **kwargs):
        if path.name == access.name + ".json":
            raise OSError("injected journal deletion failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_journal)
    with pytest.raises(OSError, match = "journal deletion failure"):
        access.close()
    assert list((store.root / ".readers").glob("*.json"))
    assert not grants(store, access.generation.files[0])
    monkeypatch.setattr(Path, "unlink", original)
    access.close()
    assert not list((store.root / ".readers").iterdir())


def test_reader_record_limit_also_counts_interrupted_journals_without_locks(snapshot, monkeypatch):
    import json
    from core.inference.windows_sandbox import content_access

    store, spec, _ = snapshot
    digest = store.publish(spec)
    monkeypatch.setattr(content_access, "MAX_READERS", 1)
    for index in (1, 2):
        record = {"version": 1, "digest": digest, "sid": f"S-1-15-2-{index}-2-3-4-5-6-7"}
        store.api.create(
            store.root / ".readers" / (str(index) * 32 + ".json"),
            json.dumps(record, sort_keys = True, separators = (",", ":")).encode("utf-8"),
        )
    with PathLease() as pins, pytest.raises(WindowsRuntimeError, match = "limit"):
        read_readers(store, pins)
