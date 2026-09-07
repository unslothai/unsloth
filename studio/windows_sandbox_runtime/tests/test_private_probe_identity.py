# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Private qualification workdir ownership, not runtime qualification."""

import ctypes
from dataclasses import asdict
from pathlib import Path
import sys
import time

import pytest

from test_identity import journal, reservation, identity, lpac
from test_preparation import BACKEND, preparation

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows private probe")


def test_private_workdir_intent_precedes_creation(journal, reservation, monkeypatch):
    api = lpac._api()
    original = api.userenv.CreateAppContainerProfile
    records = []

    def create(*args):
        value = identity._read(reservation.path)
        assert value["version"] == 4 and value["purpose"] == "qualification"
        assert value["state"] == "creating" and value["workdir"] is None
        assert value["reader"] is None and value["profile_folder"] is None
        identity._validate(value, reservation.recipe, value["sid"])
        records.append(value)
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(api.userenv, "CreateAppContainerProfile", create)
        patch.setattr(lpac, "_grant_modify", lambda *a: pytest.fail("grant before caller"))
        actual = reservation.create_private()
    ready = identity._read(reservation.path)
    assert len(records) == 1 and ready["state"] == "ready"
    assert ready["workdir"] == actual.private_temp
    assert Path(ready["workdir"]) == Path(actual.profile_folder) / "Temp"
    assert Path(actual.private_temp).is_dir()
    assert identity._validate(ready, reservation.recipe, actual.sid_string) is None
    assert actual.granted_roots == (actual.private_temp, *actual.traverse_roots)
    lpac._grant_modify(actual.private_temp, actual.sid)
    Path(actual.private_temp, "owned-probe-fixture").write_text("private", encoding = "utf-8")
    reservation.cleanup()
    assert not Path(actual.profile_folder).exists()
    assert not reservation.path.exists() and actual.cleaned


def test_ordinary_create_cannot_select_private_workdir(reservation, monkeypatch):
    monkeypatch.setattr(
        lpac._api().userenv, "CreateAppContainerProfile", lambda *a: pytest.fail("created")
    )
    with pytest.raises(identity.WindowsRuntimeError, match = "explicit workdir"):
        reservation.create(None)
    assert not reservation.started and not reservation.owned


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", "tool"),
        ("purpose", None),
        ("workdir", "C:\\outside"),
        ("workdir", None),
        ("reader", {}),
        ("version", 3),
        ("extra", True),
    ],
)
def test_private_journal_rejects_scope_changes(reservation, field, value):
    actual = reservation.create_private()
    record = identity._read(reservation.path)
    record[field] = value
    with pytest.raises(identity.WindowsRuntimeError):
        identity._validate(record, reservation.recipe, actual.sid_string)


@pytest.mark.parametrize("state", ["creating", "collision"])
def test_private_intent_cannot_name_external_workdir(reservation, state):
    with identity._derived_sid(reservation.recipe.moniker) as (_, sid):
        record = identity._payload(reservation.recipe, sid, state, None, private_workdir = True)
        assert identity._validate(record, reservation.recipe, sid) is None
        record["workdir"] = "E:\\external"
        with pytest.raises(identity.WindowsRuntimeError, match = "purpose or workdir"):
            identity._validate(record, reservation.recipe, sid)


@pytest.mark.parametrize("stage", ["partial_intent", "after_create", "ready"])
def test_private_creation_failure_is_recoverable(reservation, monkeypatch, stage):
    api = lpac._api()
    original_create, original_write = api.userenv.CreateAppContainerProfile, identity._write

    def create(*args):
        result = original_create(*args)
        assert result == 0
        if stage == "after_create":
            raise OSError("injected private creation failure")
        return result

    def write(path, value):
        if stage == "partial_intent" and value["state"] == "creating":
            identity.native_files().create(Path(str(path) + ".tmp"), b'{"version":')
            raise OSError("injected private intent failure")
        if stage == "ready" and value["state"] == "ready":
            raise OSError("injected private ready failure")
        return original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(api.userenv, "CreateAppContainerProfile", create)
        patch.setattr(identity, "_write", write)
        with pytest.raises(OSError, match = "injected private"):
            reservation.create_private()
    reservation.cleanup()
    assert reservation.closed and not reservation.path.exists()
    assert not Path(str(reservation.path) + ".tmp").exists()
    with identity._derived_sid(reservation.recipe.moniker) as (_, sid):
        assert identity._profile_path(sid) is None


@pytest.mark.parametrize("failed_tombstone", [False, True])
def test_private_observed_collision_preserves_profile(reservation, monkeypatch, failed_tombstone):
    api = lpac._api()
    original_create, original_write = api.userenv.CreateAppContainerProfile, identity._write
    created = []

    def collision(*args):
        result = original_create(*args)
        assert result == 0
        created.append(args[0])
        return ctypes.c_int32(0x800700B7).value

    def write(path, value):
        if failed_tombstone and value["state"] == "collision":
            raise OSError("injected private collision tombstone failure")
        return original_write(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(api.userenv, "CreateAppContainerProfile", collision)
            patch.setattr(identity, "_write", write)
            with pytest.raises((identity.WindowsRuntimeError, OSError), match = "collision"):
                reservation.create_private()
        reservation.cleanup()
        assert created == [reservation.recipe.moniker]
        with identity._derived_sid(reservation.recipe.moniker) as (_, sid):
            assert identity._profile_path(sid) is not None
        assert not reservation.path.exists()
    finally:
        for moniker in created:
            assert api.userenv.DeleteAppContainerProfile(moniker) == 0


def test_private_known_recipe_recovers_without_worker_response(tmp_path, monkeypatch):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    owner = identity.InvocationReservation(identity.InvocationRecipe.new())
    environment = owner.reserve_worker()
    source = f"""
import sys
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox.identity import InvocationReservation, InvocationRecipe
owner = InvocationReservation(InvocationRecipe(**{asdict(owner.recipe)!r}))
owner.create_private()
# Deliberately exit without returning identity paths or granting payload access.
"""
    try:
        output, pid = preparation._run_worker(
            [sys._base_executable, "-I", "-S", "-B", "-c", source],
            environment,
            str(tmp_path),
            deadline = time.monotonic() + 10,
            cancel = None,
        )
        assert output == b"" and owner.path is None
        with identity._derived_sid(owner.recipe.moniker) as (_, sid):
            profile = identity._profile_path(sid)
            assert profile is not None and Path(profile, "Temp").is_dir()
        owner.cleanup(in_worker = True)
        assert owner.closed and not Path(profile).exists()
        assert not list(local.rglob("*.json"))
    finally:
        owner.cleanup(in_worker = True)
