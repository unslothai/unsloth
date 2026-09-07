# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Terminal durable grant ownership and recovery, not shell qualification."""

from dataclasses import asdict, replace
import ctypes
import os
from pathlib import Path
import sys
import time

import pytest

from test_identity import journal, reservation, identity, lpac
from test_preparation import BACKEND, preparation
from core.inference.windows_sandbox import terminal_runtime
from core.inference.windows_sandbox.content_files import PathLease, native_files

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows Terminal ownership")


@pytest.fixture
def runtime(journal, tmp_path):
    _, work = journal
    shell = tmp_path / "shell with spaces-é" / "bin"
    shell.mkdir(parents = True)
    # Static inventory fixture only; this file is never executed.
    (shell / "bash.exe").write_bytes(b"not an executable")
    return terminal_runtime._inspect(
        {"argv": [str(shell / "bash.exe"), "-c", "exit 99"], "workdir": str(work), "env": {}}
    )


def security(path):
    with PathLease() as pins:
        return native_files().security_text(pins.directory(path))


def test_terminal_intent_records_runtime_before_profile_and_grants(
    runtime, reservation, monkeypatch
):
    api = lpac._api()
    original = api.userenv.CreateAppContainerProfile
    records = []

    def create(*args):
        value = identity._read(reservation.path)
        assert value["version"] == 5 and value["purpose"] == "terminal"
        assert value["state"] == "creating" and value["profile_folder"] is None
        assert value["runtime_roots"] == list(runtime.acl_roots)
        assert "reader" not in value
        assert identity._validate(value, reservation.recipe, value["sid"]) is None
        records.append(value)
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(api.userenv, "CreateAppContainerProfile", create)
        patch.setattr(lpac, "_grant_modify", lambda *a: pytest.fail("unexpected grant"))
        patch.setattr(lpac, "_grant_read_execute", lambda *a: pytest.fail("unexpected grant"))
        actual = reservation.create_terminal(runtime)
    assert len(records) == 1
    value = identity._read(reservation.path)
    assert value["state"] == "ready" and value["runtime_roots"] == list(runtime.acl_roots)
    assert actual.granted_roots == (
        runtime.workdir,
        actual.private_temp,
        *runtime.acl_roots,
        *actual.traverse_roots,
    )
    assert actual.sid_string not in security(runtime.acl_roots[0])
    reservation.cleanup()
    assert not reservation.path.exists() and not Path(actual.profile_folder).exists()


def test_system_shell_keeps_distinct_terminal_journal_with_no_runtime_acl(journal, reservation):
    runtime = terminal_runtime._inspect(
        {
            "argv": [str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe")],
            "workdir": str(journal[1]),
            "env": {},
        }
    )
    actual = reservation.create_terminal(runtime)
    value = identity._read(reservation.path)
    assert value["version"] == 5 and value["runtime_roots"] == []
    assert value["purpose"] == "terminal"
    assert actual.granted_roots == (runtime.workdir, actual.private_temp, *actual.traverse_roots)


@pytest.mark.parametrize("change", ["roots", "acl", "type", "hardlink", "symlink"])
def test_terminal_revalidates_before_identity_mutation(
    runtime, reservation, tmp_path, monkeypatch, change
):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "secret"
    sentinel.write_text("host secret")
    selected = runtime
    if change == "roots":
        selected = replace(runtime, runtime_roots = (str(outside),))
    elif change == "acl":
        selected = replace(runtime, acl_roots = ())
    elif change == "type":
        selected = asdict(runtime)
    elif change == "hardlink":
        os.link(sentinel, Path(runtime.acl_roots[0]) / "alias")
    else:
        (Path(runtime.acl_roots[0]) / "alias").symlink_to(sentinel)
    monkeypatch.setattr(
        lpac._api().userenv, "CreateAppContainerProfile", lambda *a: pytest.fail("created")
    )
    with pytest.raises((identity.WindowsRuntimeError, lpac.SandboxUnavailableError)):
        reservation.create_terminal(selected)
    assert reservation.path is None and not reservation.owned
    assert not reservation.started
    assert sentinel.read_text() == "host secret"


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", "qualification"),
        ("reader", None),
        ("version", 2),
        ("runtime_roots", None),
        ("runtime_roots", ["C:\\"]),
        ("runtime_roots", ["\\\\server\\share"]),
        ("runtime_roots", ["C:\\one\\..\\two"]),
        ("runtime_roots", ["C:\\one", "c:\\one"]),
        ("runtime_roots", ["C:\\one"] * 65),
    ],
)
def test_terminal_malformed_journal_cannot_authorize_cleanup(runtime, reservation, field, value):
    actual = reservation.create_terminal(runtime)
    record = identity._read(reservation.path)
    record[field] = value
    with pytest.raises(identity.WindowsRuntimeError):
        identity._validate(record, reservation.recipe, actual.sid_string)


def test_terminal_journal_rejects_workdir_overlap(runtime, reservation):
    actual = reservation.create_terminal(runtime)
    record = identity._read(reservation.path)
    record["runtime_roots"] = [runtime.workdir]
    with pytest.raises(identity.WindowsRuntimeError, match = "overlaps"):
        identity._validate(record, reservation.recipe, actual.sid_string)


@pytest.mark.parametrize("private,reader", [(True, None), (False, object())])
def test_terminal_payload_cannot_combine_python_ownership(reservation, private, reader):
    with pytest.raises(identity.WindowsRuntimeError, match = "cannot include Python"):
        identity._payload(
            reservation.recipe,
            "unused",
            "creating",
            "C:\\work",
            private_workdir = private,
            reader = reader,
            terminal_roots = (),
        )


@pytest.mark.parametrize("failed_tombstone", [False, True])
def test_terminal_collision_never_deletes_conflicting_profile(
    runtime, reservation, monkeypatch, failed_tombstone
):
    api = lpac._api()
    original_create, original_write = api.userenv.CreateAppContainerProfile, identity._write
    created = []

    def collision(*args):
        assert original_create(*args) == 0
        created.append(args[0])
        return ctypes.c_int32(0x800700B7).value

    def write(path, value):
        if failed_tombstone and value["state"] == "collision":
            raise OSError("injected Terminal collision tombstone failure")
        return original_write(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(api.userenv, "CreateAppContainerProfile", collision)
            patch.setattr(identity, "_write", write)
            with pytest.raises((identity.WindowsRuntimeError, OSError), match = "collision"):
                reservation.create_terminal(runtime)
        assert reservation.collision_record["runtime_roots"] == list(runtime.acl_roots)
        reservation.cleanup()
        assert not reservation.path.exists()
        with identity._derived_sid(reservation.recipe.moniker) as (_, sid):
            assert identity._profile_path(sid) is not None
    finally:
        # The test, not the reservation, owns the deliberately conflicting profile.
        for moniker in created:
            assert api.userenv.DeleteAppContainerProfile(moniker) == 0


@pytest.mark.parametrize("phase", ["partial_intent", "after_create", "ready"])
def test_terminal_creation_failure_remains_recoverable(runtime, reservation, monkeypatch, phase):
    api = lpac._api()
    original_create, original_write = api.userenv.CreateAppContainerProfile, identity._write

    def create(*args):
        result = original_create(*args)
        assert result == 0
        if phase == "after_create":
            raise OSError("injected Terminal creation failure")
        return result

    def write(path, value):
        if phase == "partial_intent":
            identity.native_files().create(Path(str(path) + ".tmp"), b'{"version":5')
            raise OSError("injected Terminal partial intent")
        if phase == "ready" and value["state"] == "ready":
            raise OSError("injected Terminal ready failure")
        return original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(api.userenv, "CreateAppContainerProfile", create)
        patch.setattr(identity, "_write", write)
        with pytest.raises(OSError, match = "injected Terminal"):
            reservation.create_terminal(runtime)
    reservation.cleanup()
    assert reservation.closed and not reservation.path.exists()
    assert not Path(str(reservation.path) + ".tmp").exists()
    with identity._derived_sid(reservation.recipe.moniker) as (_, sid):
        assert identity._profile_path(sid) is None


def test_terminal_collision_cleanup_cannot_change_recorded_runtime_roots(runtime, reservation):
    actual = reservation.create_terminal(runtime)
    ready = identity._read(reservation.path)
    creating = {**ready, "state": "creating", "profile_folder": None}
    collision = {**creating, "state": "collision", "runtime_roots": []}
    identity._write(reservation.path, creating)
    try:
        with pytest.raises(identity.WindowsRuntimeError, match = "differs from durable ownership"):
            identity.cleanup_recipe(reservation.recipe, str(reservation.path), collision)
        assert identity._read(reservation.path) == creating
        assert identity._profile_path(actual.sid_string) == actual.profile_folder
    finally:
        identity._write(reservation.path, ready)


def test_terminal_recovery_revokes_only_owned_sid_after_worker_exit(tmp_path, monkeypatch):
    local, work, shell = (tmp_path / name for name in ("LocalAppData", "work", "shell"))
    for path in (local, work, shell):
        path.mkdir()
    (shell / "cmd.exe").write_bytes(b"not executed")
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    owner = identity.InvocationReservation(identity.InvocationRecipe.new())
    other = identity.InvocationReservation(identity.InvocationRecipe.new())
    environment = owner.reserve_worker()
    original_acl = security(shell)
    source = f"""
import sys
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import terminal_native as lpac
from core.inference.windows_sandbox import identity, terminal_runtime
runtime = terminal_runtime._inspect({{'argv': [{str(shell / "cmd.exe")!r}], 'workdir': {str(work)!r}, 'env': {{}}}})
owner = identity.InvocationReservation(identity.InvocationRecipe(**{asdict(owner.recipe)!r}))
actual = owner.create_terminal(runtime)
assert identity._read(owner.path)['state'] == 'ready'
lpac._grant_read_execute({str(shell)!r}, actual.sid)
lpac._grant_modify({str(work)!r}, actual.sid)
# Deliberate worker exit after grants without returning identity ownership.
"""
    try:
        output, pid = preparation._run_worker(
            [sys._base_executable, "-I", "-S", "-B", "-c", source],
            environment,
            str(tmp_path),
            deadline = time.monotonic() + 15,
            cancel = None,
        )
        assert output == b"" and owner.path is None
        with identity._derived_sid(owner.recipe.moniker) as (_, sid):
            profile = identity._profile_path(sid)
            assert profile is not None and sid in security(shell) and sid in security(work)
            # A concurrent trusted invocation's ACE must not be rolled back.
            second = other.create(work)
            lpac._grant_read_execute(str(shell), second.sid)
            owner.cleanup(in_worker = True)
            assert sid not in security(shell) and sid not in security(work)
            assert second.sid_string in security(shell)
            assert not Path(profile).exists() and owner.closed
    finally:
        owner.cleanup(in_worker = True)
        if other.identity is not None:
            lpac._revoke_sid(str(shell), other.identity.sid)
        other.cleanup()
    final_acl = security(shell)
    # SetNamedSecurityInfo marks the DACL auto-inherited (AI). Compare the
    # actual owner and every ACE; whole-descriptor rollback is not our policy.
    assert final_acl.split("D:")[0] == original_acl.split("D:")[0]
    assert final_acl[final_acl.index("(") :] == original_acl[original_acl.index("(") :]
    assert not list(local.rglob("*.json"))


def test_terminal_acl_cleanup_failure_retains_journal_for_retry(runtime, reservation, monkeypatch):
    actual = reservation.create_terminal(runtime)
    root = runtime.acl_roots[0]
    lpac._grant_read_execute(root, actual.sid)
    original = lpac._revoke_sid

    def revoke(path, sid, **kwargs):
        if path == root:
            raise PermissionError("injected Terminal ACL revocation failure")
        return original(path, sid, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(lpac, "_revoke_sid", revoke)
        with pytest.raises(OSError, match = "Terminal ACL revocation"):
            reservation.cleanup()
    assert reservation.path.exists() and not reservation.closed and not actual.cleaned
    assert actual.sid_string in security(root)
    assert identity._profile_path(actual.sid_string) is not None
    # A different owner can recover from the durable record without the object.
    identity._recover(reservation.path, reservation.recipe)
    assert actual.sid_string not in security(root)
    assert not reservation.path.exists() and identity._profile_path(actual.sid_string) is None
