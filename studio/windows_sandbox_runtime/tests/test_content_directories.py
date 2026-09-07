# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Explicit empty content directories, not payload or backend qualification."""

import json
import os
from pathlib import Path
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from core.inference import windows_lpac
from core.inference.windows_sandbox.content import RuntimeContentStore, SnapshotFile, SnapshotSpec
from core.inference.windows_sandbox.content_access import READ_EXECUTE, read_readers, validate_acl
from core.inference.windows_sandbox.content_files import PathLease
from core.inference.windows_sandbox.dependencies import FileIdentity, read_regular_file
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


def pure_spec(*, directories = ()):
    source = FileIdentity("C:/source/bash.exe", "a" * 64, 7, 1, 2)
    return SnapshotSpec(
        (SnapshotFile(source, "shell/bin/bash.exe"),),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        directories = directories,
    )


def test_directory_free_manifest_remains_canonical_version_one():
    manifest = pure_spec().manifest()
    assert (
        manifest
        == json.dumps(
            {
                "version": 1,
                "context": {
                    "runtime": "1" * 64,
                    "dependencies": "2" * 64,
                    "profile": "3" * 64,
                    "helper": "4" * 64,
                },
                "files": [{"path": "shell/bin/bash.exe", "sha256": "a" * 64, "size": 7}],
            },
            sort_keys = True,
            separators = (",", ":"),
        ).encode()
    )


def test_explicit_empty_directories_use_sorted_closed_version_two_manifest():
    value = json.loads(pure_spec(directories = ("shell/usr/tmp", "shell/mingw64/bin")).manifest())
    assert value["version"] == 2
    assert value["directories"] == ["shell/mingw64/bin", "shell/usr/tmp"]
    assert set(value) == {"version", "context", "files", "directories"}


@pytest.mark.parametrize(
    "directories",
    [
        ("../escape",),
        ("C:/absolute",),
        ("shell\\alias",),
        ("shell/CON",),
        ("/".join(["part"] * 17),),
        ("x" * 1025,),
        ("shell/empty", "SHELL/EMPTY"),
        ("shell/bin/bash.exe",),
        ("shell/bin/bash.exe/child",),
        ("shell/bin",),
        ("shell/empty", "shell/empty/nested"),
        tuple(f"shell/empty-{index}" for index in range(65)),
    ],
)
def test_explicit_directory_manifest_rejects_aliases_collisions_and_overlap(directories):
    with pytest.raises(WindowsRuntimeError):
        pure_spec(directories = directories).manifest()


@pytest.fixture
def native_store(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Native NTFS empty-directory store contract")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "bash.exe"
    source.write_bytes(b"static fixture; never executed")
    file_identity, _ = read_regular_file(source, limit = 1024)
    spec = SnapshotSpec(
        (SnapshotFile(file_identity, "shell/bin/bash.exe"),),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        directories = ("shell/mingw64/bin",),
    )
    store = RuntimeContentStore(tmp_path / "store")
    return store, spec


def grants(store, path):
    with PathLease() as pins:
        readers = read_readers(store, pins)
        return validate_acl(store, pins.directory(path), path, readers)


def test_native_publish_lease_inventory_and_reader_acl_cleanup(native_store):
    store, spec = native_store
    digest = store.publish(spec)
    from core.inference.windows_sandbox.identity import InvocationRecipe, InvocationReservation

    reservation = InvocationReservation(InvocationRecipe.new())
    identity = reservation.create_private()
    try:
        with store.lease(digest) as generation:
            declared = generation.directory / "files/shell/mingw64/bin"
            assert generation.directories == (declared,)
            assert declared.is_dir() and not list(declared.iterdir())
            with pytest.raises(OSError):
                declared.rename(declared.with_name("renamed"))

        with store.read_access(digest, identity.sid_string) as access:
            declared = access.generation.directories[0]
            assert grants(store, declared) == {identity.sid_string: READ_EXECUTE}
        assert grants(store, declared) == {}
        store.collect(digest)
        assert not (store.root / digest).exists()
    finally:
        reservation.cleanup()


def test_native_unlisted_entry_inside_declared_empty_directory_fails(native_store):
    store, spec = native_store
    digest = store.publish(spec)
    declared = store.root / digest / "files/shell/mingw64/bin"
    (declared / "unexpected.dll").write_bytes(b"unlisted")
    with pytest.raises(WindowsRuntimeError, match = "missing or unlisted"):
        with store.lease(digest):
            pytest.fail("nonempty declared directory was leased")


def test_native_declared_directory_dacl_tamper_fails_closed(native_store):
    store, spec = native_store
    digest = store.publish(spec)
    declared = store.root / digest / "files/shell/mingw64/bin"
    handle = store.api.open(declared, directory = True, write_dac = True)
    try:
        original = store.api.security_text(handle)
        store.api.set_owned_dacl(handle, original + "(A;;FR;;;WD)")
        assert store.api.security_text(handle) != original
    finally:
        store.api.kernel.CloseHandle(handle)
    try:
        with pytest.raises(WindowsRuntimeError, match = "DACL changed"):
            with store.lease(digest):
                pytest.fail("broadened directory ACL was leased")
    finally:
        handle = store.api.open(declared, directory = True, write_dac = True)
        try:
            store.api.set_owned_dacl(handle, original)
            assert store.api.security_text(handle) == original
        finally:
            store.api.kernel.CloseHandle(handle)


def test_native_declared_directory_reparse_point_is_rejected(native_store, tmp_path):
    store, spec = native_store
    digest = store.publish(spec)
    declared = store.root / digest / "files/shell/mingw64/bin"
    outside = tmp_path / "outside"
    outside.mkdir()
    declared.rmdir()
    try:
        os.symlink(outside, declared, target_is_directory = True)
    except OSError as error:
        pytest.skip(f"Host cannot create a directory reparse fixture: {error}")
    with pytest.raises(WindowsRuntimeError, match = "reparse point"):
        with store.lease(digest):
            pytest.fail("directory reparse point was leased")


def test_failed_directory_publication_removes_owned_staging(native_store, monkeypatch):
    store, spec = native_store
    mkdir = store.api.mkdir

    def fail(path):
        if str(path).replace("\\", "/").endswith("shell/mingw64/bin"):
            raise OSError("injected explicit-directory failure")
        return mkdir(path)

    with monkeypatch.context() as patch:
        patch.setattr(store.api, "mkdir", fail)
        with pytest.raises(OSError, match = "explicit-directory failure"):
            store.publish(spec)
    assert not list(store.root.glob(".build-*"))
    assert store.publish(spec)
