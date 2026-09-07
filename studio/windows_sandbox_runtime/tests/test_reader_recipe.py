# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Durable profile-to-reader recovery before filesystem preparation handoff."""

from dataclasses import asdict
import json
from pathlib import Path
import secrets
import subprocess
import sys

import pytest

from test_content_access import snapshot, identities, grants, READ_EXECUTE
from test_preparation import BACKEND, lpac
from core.inference.windows_sandbox import identity, content_access, preparation
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason = "Native Windows durable reader ownership"
)


@pytest.mark.parametrize("phase", ["unpublished", "published", "granted"])
@pytest.mark.parametrize("response_received", [False, True])
def test_profile_recipe_recovers_reader_after_worker_loss(
    snapshot, identities, tmp_path, monkeypatch, phase, response_received
):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    work = tmp_path / "work"
    work.mkdir()
    recipe = identity.InvocationRecipe.new()
    reservation = identity.InvocationReservation(recipe)
    environment = reservation.reserve_worker()
    reader = identity.RuntimeReaderRecipe(str(store.root), secrets.token_hex(16), digest)
    other = store.read_access(digest, identities[0].sid_string).__enter__()
    info = None
    source = f"""
import sys,json,os
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import identity, content_access
from core.inference.windows_sandbox.content import RuntimeContentStore
owner = identity.InvocationReservation(identity.InvocationRecipe(**{asdict(recipe)!r}))
reader = identity.RuntimeReaderRecipe(**{asdict(reader)!r})
actual = owner.create({str(work)!r}, reader=reader)
store = RuntimeContentStore(reader.store_root)
lease = store.read_access(reader.digest, actual.sid_string, name=reader.name)
def pause():
    print(json.dumps({{'path':str(owner.path),'sid':actual.sid_string}}),flush=True)
    r,w=os.pipe();os.read(r,1)
create = store.api.create
def creating(path, data):
    if str(path).endswith(reader.name+'.tmp') and {phase!r} == 'unpublished':
        create(path,b'{{')
        pause()
    return create(path,data)
store.api.create = creating
change = content_access._change
def changing(*args,**kwargs):
    assert kwargs['remove'] is False
    if {phase!r} == 'granted':
        change(*args,**kwargs)
    pause()
content_access._change = changing
lease.__enter__()
"""
    child = subprocess.Popen(
        [sys._base_executable, "-I", "-S", "-B", "-c", source],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        env = environment,
    )
    try:
        # The fixed harness emits only after reaching the controlled stage.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers = 1) as pool:
            try:
                line = pool.submit(child.stdout.readline).result(timeout = 15)
                assert line, child.stderr.read()
                info = json.loads(line)
            finally:
                child.kill()
                child.wait(timeout = 5)
        record = identity._read(Path(info["path"]))
        assert record["version"] == 3 and record["reader"] == asdict(reader)
        assert record["owner_pid"] == recipe.owner_pid
        identity.recover_identities()  # exited worker is not its still-live broker
        assert Path(info["path"]).exists()
        if phase == "unpublished":
            assert (store.root / ".readers" / (reader.name + ".tmp")).read_bytes() == b"{"
            assert not (store.root / ".readers" / (reader.name + ".json")).exists()
        if response_received:
            reservation.path = Path(info["path"])
        # No journal path or surviving reader owner is needed after worker loss.
        # A later environment change must not redirect the cleanup namespace.
        with monkeypatch.context() as context:
            context.setenv("LOCALAPPDATA", str(tmp_path / "unrelated namespace"))

            def forbidden(*args, **kwargs):
                pytest.fail("Recovery reopened the journal namespace in the broker")

            context.setattr(identity, "_journal_root", forbidden)
            reservation.cleanup(in_worker = True)
        assert reservation.closed
        assert not Path(info["path"]).exists() and identity._profile_path(info["sid"]) is None
        assert not list((store.root / ".readers").glob(reader.name + ".*"))
        assert grants(store, other.generation.files[0]) == {other.sid: READ_EXECUTE}
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout = 5)
        reservation.cleanup(in_worker = True)
        other.close()
    assert not list((store.root / ".readers").iterdir())


def test_partial_reader_write_rolls_back_without_publishing_authority(
    snapshot, identities, monkeypatch
):
    store, spec, _ = snapshot
    lease = store.read_access(store.publish(spec), identities[0].sid_string)
    original = store.api.create

    def partial(path, data):
        if str(path).endswith(lease.name + ".tmp"):
            original(path, b"{")
            raise OSError("injected partial private reader write")
        return original(path, data)

    with monkeypatch.context() as context:
        context.setattr(store.api, "create", partial)
        with pytest.raises(OSError, match = "partial private reader"):
            lease.__enter__()
    assert lease.closed and not lease.pins.handles
    assert not list((store.root / ".readers").iterdir())
    assert grants(store, lease.generation.files[0]) == {}


def test_reusing_reader_name_cannot_remove_another_invocation(snapshot, identities):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    with store.read_access(digest, identities[0].sid_string) as first:
        second = store.read_access(digest, identities[1].sid_string, name = first.name)
        with pytest.raises(WindowsRuntimeError, match = "already owned"):
            second.__enter__()
        assert grants(store, first.generation.files[0]) == {first.sid: READ_EXECUTE}
        assert (store.root / ".readers" / (first.name + ".json")).exists()


def test_recovery_requires_the_recorded_generation(snapshot, identities):
    store, spec, _ = snapshot
    lease = store.read_access(store.publish(spec), identities[0].sid_string).__enter__()
    lease.pins.close()
    try:
        with pytest.raises(WindowsRuntimeError, match = "another runtime generation"):
            content_access.recover_readers(
                store, owner = (lease.name, lease.sid), expected_digest = "0" * 64
            )
        assert grants(store, lease.generation.files[0]) == {lease.sid: READ_EXECUTE}
    finally:
        lease.close()


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"store_root": "C:/", "name": "a" * 32, "digest": "b" * 64},
        {"store_root": "C:/safe", "name": "../escape", "digest": "b" * 64},
    ],
)
def test_reader_recipe_rejects_malformed_scope(value):
    with pytest.raises(WindowsRuntimeError):
        identity.RuntimeReaderRecipe.from_value(value)
