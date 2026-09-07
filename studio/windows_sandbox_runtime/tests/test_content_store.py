# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real NTFS storage/lease tests, not complete LPAC bootstrap qualification."""

from dataclasses import replace
import ctypes
from ctypes import wintypes as W
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox.content import (
    RuntimeContentStore,
    SnapshotFile,
    SnapshotSpec,
    _relative,
    _validate_manifest,
)
from core.inference.windows_sandbox.content_files import PathLease, native_files
from core.inference.windows_sandbox import content_files
from core.inference.windows_sandbox.dependencies import read_regular_file
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.fixture
def snapshot(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows NTFS storage lane; not native LPAC qualification")
    sources = tmp_path / "source λ"
    sources.mkdir()
    path = sources / "sample.py"
    path.write_bytes(b"raise AssertionError('snapshot contents must never execute')\n")
    identity, _ = read_regular_file(path, limit = 1024)
    spec = SnapshotSpec(
        (SnapshotFile(identity, "Lib/sample.py"),), *[str(i) * 64 for i in range(1, 5)]
    )
    store = RuntimeContentStore(tmp_path / "private runtime λ")
    return store, spec, path


def test_publish_reuse_and_lease_do_not_execute_content(snapshot, monkeypatch):
    store, spec, source = snapshot
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("copied code executed"))
    digest = store.publish(spec)
    assert store.publish(spec) == digest
    with store.lease(digest) as generation:
        assert generation.trust_classification == "payload_only"
        assert generation.files[0].read_bytes() == source.read_bytes()
        assert not hasattr(generation, "qualified")
        assert not hasattr(generation, "available")
    assert not list(store.root.glob(".build-*"))


def test_store_is_private_at_creation_and_cannot_adopt_an_existing_directory(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows ACL creation")
    existing = tmp_path / "existing user folder"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_bytes(b"user data")
    with pytest.raises(WindowsRuntimeError):
        RuntimeContentStore(existing)
    assert sentinel.read_bytes() == b"user data"
    assert sorted(p.name for p in existing.iterdir()) == ["keep.txt"]
    store = RuntimeContentStore(tmp_path / "new")
    with PathLease() as pins:
        native_files().require_private(pins.directory(store.root))
        native_files().require_private(pins.file(store.root / ".store"))
    RuntimeContentStore(store.root)


@pytest.mark.parametrize("action", ["write", "rename", "parent_rename", "delete"])
def test_active_generation_locks_block_host_mutation(snapshot, action):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    with store.lease(digest) as generation:
        path = generation.files[0]
        with pytest.raises(OSError) as error:
            if action == "write":
                path.write_bytes(b"replacement")
            elif action == "rename":
                path.rename(path.with_name("replacement.py"))
            elif action == "parent_rename":
                path.parent.rename(path.parent.with_name("replacement"))
            else:
                path.unlink()
        if action == "write":
            assert error.value.errno == 13
            handle = store.api.kernel.CreateFileW(str(path), 0x40000000, 7, None, 3, 0, None)
            assert handle == ctypes.c_void_p(-1).value
            assert ctypes.get_last_error() == 32
        else:
            assert error.value.winerror in (5, 32)
    with path.open("r+b") as stream:
        assert stream.read() == Path(spec.files[0].source.path).read_bytes()


def test_open_writable_source_handle_prevents_snapshot(snapshot):
    store, spec, source = snapshot
    with source.open("r+b"):
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.publish(spec)
    assert not list(store.root.glob(".build-*"))
    assert store.publish(spec)


@pytest.mark.parametrize("locked_kind", ["file", "directory", "staging"])
def test_publication_locked_descendant_fails_closed_and_releases(
    snapshot, monkeypatch, locked_kind
):
    store, spec, source = snapshot
    rename = os.rename
    observed = []

    def locked_rename(staging, destination):
        target = (
            staging
            if locked_kind == "staging"
            else staging / "files" / ("Lib/sample.py" if locked_kind == "file" else "Lib")
        )
        handle = store.api.open(target, directory = locked_kind != "file")
        try:
            with pytest.raises(OSError) as error:
                rename(staging, destination)
            observed.append(error.value.winerror)
            assert error.value.winerror in (5, 32)
            assert not destination.exists()
            assert target.exists()
            raise error.value
        finally:
            assert store.api.kernel.CloseHandle(handle)

    with monkeypatch.context() as patch:
        patch.setattr(os, "rename", locked_rename)
        patch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("payload launched"))
        with pytest.raises(OSError):
            store.publish(spec)
    assert 1 <= len(observed) <= 4
    assert not list(store.root.glob(".build-*"))
    assert not list(store.root.glob("[0-9a-f]" * 64))
    # A separate caller-requested preparation can succeed once the blocker closes.
    # Retrying the metadata rename must never recopy or execute the content.
    digest = store.publish(spec)
    with store.lease(digest) as generation:
        assert generation.files[0].read_bytes() == source.read_bytes()


def test_publication_retries_only_rename_after_real_lock_release(snapshot, monkeypatch):
    store, spec, _ = snapshot
    rename, create = os.rename, store.api.create
    attempts, copied = [], []

    def record_copy(path, data):
        copied.append(path)
        return create(path, data)

    def release_after_failure(staging, destination):
        attempts.append(staging)
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            with store._mutation():
                pytest.fail("Publication dropped its exclusive lock")
        if len(attempts) == 1:
            handle = store.api.open(staging / "files/Lib/sample.py")
            try:
                return rename(staging, destination)
            finally:
                assert store.api.kernel.CloseHandle(handle)
        return rename(staging, destination)

    monkeypatch.setattr(os, "rename", release_after_failure)
    monkeypatch.setattr(store.api, "create", record_copy)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("payload launched"))
    digest = store.publish(spec)
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert len([path for path in copied if path.name == "sample.py"]) == 1
    with store.lease(digest):
        pass
    assert not list(store.root.glob(".build-*"))


@pytest.mark.parametrize("failure", ["other_error", "collision", "budget", "budget_before_sleep"])
def test_publication_recovery_stops_without_overwrite_or_extra_attempt(
    snapshot, monkeypatch, failure
):
    from core.inference.windows_sandbox import content

    store, spec, _ = snapshot
    rename = os.rename
    attempts, sleeps = [], []
    clock = [0.0]
    conflict = []

    def fail(staging, destination):
        attempts.append(staging)
        if failure == "budget_before_sleep":
            clock[0] = 0.24
        if failure == "other_error":
            error = OSError("non-sharing publication failure")
            error.winerror = 87
            raise error
        if failure == "collision":
            destination.mkdir()
            (destination / "keep.txt").write_bytes(b"existing unrelated data")
            conflict.append(destination)
        handle = store.api.open(staging / "files/Lib/sample.py")
        try:
            return rename(staging, destination)
        finally:
            assert store.api.kernel.CloseHandle(handle)

    def sleep(delay):
        sleeps.append(delay)
        clock[0] = 1.0  # Scheduling consumed the entire local retry budget.

    monkeypatch.setattr(os, "rename", fail)
    monkeypatch.setattr(content.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(content.time, "sleep", sleep)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("payload launched"))
    with pytest.raises(OSError):
        store.publish(spec)
    assert len(attempts) == 1
    assert sleeps == ([0.025] if failure == "budget" else [])
    assert not list(store.root.glob(".build-*"))
    if conflict:
        assert (conflict[0] / "keep.txt").read_bytes() == b"existing unrelated data"


def test_publication_descendant_delete_sharing_is_not_parent_rename_permission(
    snapshot, monkeypatch
):
    store, spec, _ = snapshot
    rename = os.rename
    observed = []

    def shared_rename(staging, destination):
        target = staging / "files/Lib/sample.py"
        handle = store.api.kernel.CreateFileW(
            store.api.native_path(target), 0x80000000, 7, None, 3, 0x00200000, None
        )
        assert handle != ctypes.c_void_p(-1).value
        try:
            # Delete sharing permits renaming this file, but is not permission
            # to rename its ancestor while the descendant remains open on NTFS.
            moved = target.with_name("moved.py")
            rename(target, moved)
            rename(moved, target)
            with pytest.raises(OSError) as error:
                rename(staging, destination)
            assert error.value.winerror in (5, 32)
            assert not destination.exists()
        finally:
            assert store.api.kernel.CloseHandle(handle)
        rename(staging, destination)
        observed.append(True)

    monkeypatch.setattr(os, "rename", shared_rename)
    digest = store.publish(spec)
    assert observed == [True]
    with store.lease(digest):
        pass


@pytest.mark.parametrize("native_content", [False, True], ids = ["text", "python_dll"])
def test_publication_closes_owned_source_and_staging_handles(snapshot, monkeypatch, native_content):
    store, spec, source = snapshot
    if native_content:
        dll_name = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        identity, _ = read_regular_file(Path(sys.base_prefix) / dll_name, limit = 32 * 1024 * 1024)
        spec = replace(spec, files = (*spec.files, SnapshotFile(identity, dll_name)))
    create = store.api.kernel.CreateFileW
    close = store.api.kernel.CloseHandle
    rename = os.rename
    active, publications = {}, []

    def tracked_create(*args):
        handle = create(*args)
        error = ctypes.get_last_error()
        if handle != ctypes.c_void_p(-1).value:
            active[handle] = Path(args[0].removeprefix("\\\\?\\"))
        ctypes.set_last_error(error)
        return handle

    def tracked_close(handle):
        result = close(handle)
        error = ctypes.get_last_error()
        if result:
            active.pop(handle, None)
        ctypes.set_last_error(error)
        return result

    def tracked_rename(staging, destination):
        assert not any(path.is_relative_to(staging) for path in active.values())
        assert not any(path.is_relative_to(source.parent) for path in active.values())
        assert not {Path(item.source.path) for item in spec.files}.intersection(active.values())
        # Root and mutation-lock pins intentionally stay open across publication.
        assert store.root in active.values()
        assert store.root / ".lock" in active.values()
        rename(staging, destination)
        publications.append(destination.name)

    monkeypatch.setattr(store.api.kernel, "CreateFileW", tracked_create)
    monkeypatch.setattr(store.api.kernel, "CloseHandle", tracked_close)
    monkeypatch.setattr(os, "rename", tracked_rename)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("copied code executed"))
    for index in range(20):
        digest = store.publish(replace(spec, helper_digest = f"{index + 100:064x}"))
        assert publications[-1] == digest
        assert not active
        with store.lease(digest) as generation:
            assert (
                generation.directory / "files/Lib/sample.py"
            ).read_bytes() == source.read_bytes()
        assert not active
    assert len(set(publications)) == 20
    assert not list(store.root.glob(".build-*"))


def test_source_parent_cannot_be_replaced_during_copy(snapshot, monkeypatch):
    store, spec, source = snapshot
    original = store.api.read
    observed = []

    def read(handle, limit):
        if limit == spec.files[0].source.size:
            with pytest.raises(OSError) as error:
                source.parent.rename(source.parent.with_name("replacement"))
            assert error.value.winerror in (5, 32)
            observed.append(True)
        return original(handle, limit)

    monkeypatch.setattr(store.api, "read", read)
    store.publish(spec)
    assert observed == [True]


def test_same_size_same_mtime_source_change_requires_new_generation(snapshot):
    store, spec, source = snapshot
    old = source.stat()
    source.write_bytes(b"x" * old.st_size)
    os.utime(source, ns = (old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(WindowsRuntimeError, match = "RUNTIME_CHANGED"):
        store.publish(spec)
    assert not list(store.root.glob(".build-*"))
    identity, _ = read_regular_file(source, limit = 1024)
    new = replace(spec, files = (SnapshotFile(identity, "Lib/sample.py"),))
    assert hashlib.sha256(new.manifest()).hexdigest() != hashlib.sha256(spec.manifest()).hexdigest()
    assert store.publish(new)


def test_source_changes_do_not_modify_a_published_copy(snapshot):
    store, spec, source = snapshot
    expected = source.read_bytes()
    digest = store.publish(spec)
    source.write_bytes(b"changed installed runtime")
    with store.lease(digest) as generation:
        assert generation.files[0].read_bytes() == expected


@pytest.mark.parametrize(
    "context", ["runtime_digest", "dependency_digest", "profile_digest", "helper_digest"]
)
def test_context_change_creates_new_generation(snapshot, context):
    store, spec, _ = snapshot
    first = store.publish(spec)
    second = store.publish(replace(spec, **{context: "a" * 64}))
    assert first != second
    with store.lease(first), store.lease(second):
        assert len(list(store.root.glob("[0-9a-f]" * 64))) == 2


def test_cached_content_tampering_is_not_repaired_or_reused(snapshot):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    target = store.root / digest / "files/Lib/sample.py"
    before = target.stat()
    target.write_bytes(b"x" * before.st_size)
    os.utime(target, ns = (before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(WindowsRuntimeError, match = "bytes changed"):
        with store.lease(digest):
            pytest.fail("tampered generation leased")
    with pytest.raises(WindowsRuntimeError, match = "bytes changed"):
        store.publish(spec)
    assert target.read_bytes() == b"x" * before.st_size


@pytest.mark.parametrize("target", ["manifest", "extra_file", "extra_directory", "hardlink"])
def test_untrusted_cache_entries_fail_closed(snapshot, tmp_path, target):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    root = store.root / digest
    if target == "manifest":
        (root / "manifest.json").write_bytes(b'{"version":1,"version":2}')
    elif target == "extra_file":
        (root / "files/extra.pth").write_bytes(b"import hostile")
    elif target == "extra_directory":
        (root / "unexpected").mkdir()
    else:
        os.link(root / "files/Lib/sample.py", tmp_path / "external-alias")
    with pytest.raises(WindowsRuntimeError):
        with store.lease(digest):
            pytest.fail("invalid cached content accepted")


def test_dacl_tampering_is_detected_without_restoring_host_permissions(snapshot):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    path = store.root / digest / "files/Lib/sample.py"
    setter = store.api.security.SetFileSecurityW
    setter.argtypes, setter.restype = [W.LPCWSTR, W.DWORD, ctypes.c_void_p], W.BOOL
    sddl = f"D:P(A;;FA;;;{store.api.owner})(A;;FA;;;SY)(A;;GR;;;WD)"
    with store.api.security_attributes(sddl) as attributes:
        assert setter(str(path), 4, attributes.descriptor)
    with pytest.raises(WindowsRuntimeError, match = "DACL changed"):
        with store.lease(digest):
            pytest.fail("broadened DACL accepted")
    with PathLease() as pins:
        assert "WD" in store.api.security_text(pins.file(path))


def test_collect_cannot_remove_an_active_generation(snapshot):
    store, spec, source = snapshot
    digest = store.publish(spec)
    with store.lease(digest):
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.collect(digest)
    store.collect(digest)
    assert not (store.root / digest).exists()
    assert source.exists()
    assert (store.root / ".store").exists()


def test_simultaneous_store_mutation_fails_explicitly(snapshot):
    store, spec, _ = snapshot
    with store._mutation():
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.publish(spec)
    assert store.publish(spec)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "C:/root",
        "a\\b",
        "foo:stream",
        "foo/../b",
        "./x",
        "x/",
        "x//y",
        "NUL",
        "com1.dll",
        "COM¹.dll",
        "a.",
        "a ",
        "x\0y",
    ],
)
def test_windows_relative_names_reject_aliases_and_escape(name):
    with pytest.raises(WindowsRuntimeError):
        _relative(name)


def test_manifest_rejects_duplicate_and_directory_collisions():
    base = {
        "version": 1,
        "context": {name: "a" * 64 for name in ("runtime", "dependencies", "profile", "helper")},
    }
    for names in (("Lib/a.py", "lib/A.py"), ("Lib", "Lib/a.py")):
        with pytest.raises(WindowsRuntimeError, match = "[Cc]ollis|Colliding"):
            _validate_manifest(
                {**base, "files": [{"path": name, "sha256": "b" * 64, "size": 1} for name in names]}
            )


def test_manifest_version_and_numeric_limits_are_strict():
    base = {
        "version": 1,
        "context": {name: "a" * 64 for name in ("runtime", "dependencies", "profile", "helper")},
        "files": [{"path": "a.py", "sha256": "b" * 64, "size": 1}],
    }
    for modified in (
        {**base, "version": True},
        {**base, "files": [{**base["files"][0], "size": True}]},
        {**base, "execution_record": {}},
    ):
        with pytest.raises(WindowsRuntimeError):
            _validate_manifest(modified)


def test_partial_copy_failure_removes_only_its_created_files(snapshot, monkeypatch):
    store, spec, source = snapshot
    before = source.read_bytes()
    original = store.api.create

    def fail_after_create(path, data):
        original(path, data)
        if path.name == "sample.py":
            raise OSError("injected copy failure")

    monkeypatch.setattr(store.api, "create", fail_after_create)
    with pytest.raises(OSError, match = "injected copy failure"):
        store.publish(spec)
    assert source.read_bytes() == before
    assert sorted(path.name for path in store.root.iterdir()) == [".lock", ".readers", ".store"]


def test_cleanup_failure_is_surfaced_and_not_silently_ignored(snapshot, monkeypatch):
    store, spec, _ = snapshot
    original_create = store.api.create
    original_unlink = Path.unlink

    def fail_create(path, data):
        original_create(path, data)
        if path.name == "sample.py":
            raise OSError("injected copy failure")

    def fail_unlink(path, *args, **kwargs):
        if path.name == "sample.py":
            raise OSError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(store.api, "create", fail_create)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(WindowsRuntimeError, match = "cleanup also failed") as error:
        store.publish(spec)
    assert "injected copy failure" in str(error.value.__cause__)
    assert len(list(store.root.glob(".build-*"))) == 1


def test_content_handles_are_not_inheritable(snapshot):
    _, _, source = snapshot
    api = native_files()
    query = api.kernel.GetHandleInformation
    query.argtypes, query.restype = [W.HANDLE, ctypes.POINTER(W.DWORD)], W.BOOL
    with PathLease() as pins:
        pins.file(source)
        for handle in pins.handles.values():
            flags = W.DWORD()
            assert query(handle, ctypes.byref(flags))
            assert not flags.value & 1


def _start_control(code, ready):
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", code],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        close_fds = True,
    )
    result = []
    reader = threading.Thread(target = lambda: result.append(child.stdout.readline()), daemon = True)
    reader.start()
    reader.join(10)
    if result != [ready + "\n"]:
        child.kill()
        _, error = child.communicate(timeout = 5)
        reader.join(5)
        pytest.fail(f"control failed startup: {result}, {error}")
    return child


def _start_lease_holder(store, digest):
    # Fixed trusted harness: no model code and no inherited cache handles. A
    # bounded handshake avoids blocking pytest if child startup fails.
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "from core.inference.windows_sandbox.content import RuntimeContentStore\n"
        f"store=RuntimeContentStore({str(store.root)!r})\n"
        f"with store.lease({digest!r}):\n"
        " print('LEASE_READY', flush=True)\n"
        " time.sleep(60)\n"
    )
    return _start_control(code, "LEASE_READY")


def test_cross_process_leases_survive_one_owner_and_release_on_parent_death(snapshot):
    store, spec, source = snapshot
    digest = store.publish(spec)
    children = []
    try:
        for _ in range(2):
            children.append(_start_lease_holder(store, digest))
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.collect(digest)
        children[0].kill()
        children[0].communicate(timeout = 5)
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.collect(digest)
        children[1].kill()
        children[1].communicate(timeout = 5)
        store.collect(digest)
        assert not (store.root / digest).exists()
        assert source.exists()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout = 5)


def test_repeated_publish_lease_collect_releases_handles(snapshot):
    store, spec, _ = snapshot
    api = native_files()
    query = api.kernel.GetProcessHandleCount
    query.argtypes, query.restype = [W.HANDLE, ctypes.POINTER(W.DWORD)], W.BOOL

    def count():
        value = W.DWORD()
        assert query(api.kernel.GetCurrentProcess(), ctypes.byref(value))
        return value.value

    # Warm any interpreter/pytest first-use paths before comparing kernel handles.
    digest = store.publish(spec)
    with store.lease(digest):
        pass
    store.collect(digest)
    before = count()
    for _ in range(20):
        digest = store.publish(spec)
        with store.lease(digest):
            pass
        store.collect(digest)
    assert count() <= before


@pytest.mark.parametrize("stage", ["payload_copy", "before_publish"])
def test_interrupted_build_recovery_waits_for_owner_death(snapshot, stage):
    store, spec, source = snapshot
    code = (
        "import sys, time, os\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "from core.inference.windows_sandbox.content import RuntimeContentStore, SnapshotSpec, SnapshotFile\n"
        "from core.inference.windows_sandbox.dependencies import read_regular_file\n"
        f"store=RuntimeContentStore({str(store.root)!r})\n"
        f"identity, _=read_regular_file({str(source)!r}, limit=1024)\n"
        f"spec=SnapshotSpec((SnapshotFile(identity, 'Lib/sample.py'),), {spec.runtime_digest!r}, {spec.dependency_digest!r}, {spec.profile_digest!r}, {spec.helper_digest!r})\n"
    )
    if stage == "payload_copy":
        code += (
            "original=store.api.create\n"
            "def create(path, data):\n"
            " original(path, data)\n"
            " if path.name == 'sample.py':\n"
            "  print('BUILD_READY', flush=True); time.sleep(60)\n"
            "store.api.create=create\n"
        )
    else:
        code += (
            "original=os.rename\n"
            "def rename(source, target):\n"
            " print('BUILD_READY', flush=True); time.sleep(60)\n"
            " original(source, target)\n"
            "os.rename=rename\n"
        )
    code += "store.publish(spec)\n"
    child = _start_control(code, "BUILD_READY")
    try:
        with pytest.raises(WindowsRuntimeError, match = "STORE_BUSY"):
            store.recover_builds()
        assert len(list(store.root.glob(".build-*"))) == 1
        child.kill()
        child.communicate(timeout = 5)
        assert store.recover_builds() == 1
        assert sorted(path.name for path in store.root.iterdir()) == [".lock", ".readers", ".store"]
        assert source.exists()
        assert store.publish(spec)
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout = 5)


def test_recovery_refuses_unlisted_content_instead_of_recursive_delete(snapshot):
    store, spec, source = snapshot
    stage = store.root / (".build-" + "a" * 32)
    store.api.mkdir(stage)
    store.api.create(stage / ".build.json", spec.manifest())
    store.api.create(stage / "not-owned.txt", b"keep me")
    with pytest.raises(WindowsRuntimeError, match = "unowned entries"):
        store.recover_builds()
    assert (stage / "not-owned.txt").read_bytes() == b"keep me"
    assert source.exists()


def test_actual_ntfs_junction_cannot_supply_snapshot_source(snapshot, tmp_path):
    import _winapi

    store, spec, source = snapshot
    junction = tmp_path / "runtime junction"
    _winapi.CreateJunction(str(source.parent), str(junction))
    try:
        # Positive control: Windows can follow this actual junction normally.
        assert (junction / source.name).read_bytes() == source.read_bytes()
        aliased = replace(spec.files[0].source, path = str(junction / source.name))
        with pytest.raises(WindowsRuntimeError, match = "reparse"):
            store.publish(replace(spec, files = (SnapshotFile(aliased, "Lib/sample.py"),)))
        assert not list(store.root.glob(".build-*"))
        assert source.exists()
    finally:
        # Remove only the junction entry, never recursively remove its target.
        junction.rmdir()


def test_actual_ntfs_junction_cannot_replace_cache_directory(snapshot, tmp_path):
    import _winapi

    store, spec, source = snapshot
    digest = store.publish(spec)
    directory = store.root / digest / "files/Lib"
    original = directory.with_name("original")
    directory.rename(original)
    _winapi.CreateJunction(str(source.parent), str(directory))
    try:
        with pytest.raises(WindowsRuntimeError, match = "reparse"):
            with store.lease(digest):
                pytest.fail("junction entered cache lease")
        assert source.exists()
    finally:
        directory.rmdir()
        original.rename(directory)


def test_actual_source_hardlink_is_rejected(snapshot, tmp_path):
    store, spec, source = snapshot
    alias = tmp_path / "external source hardlink"
    os.link(source, alias)
    with pytest.raises(WindowsRuntimeError, match = "hardlinked"):
        store.publish(spec)
    assert source.read_bytes() == alias.read_bytes()
    assert not list(store.root.glob(".build-*"))


def test_real_python_dll_can_be_snapshotted_without_loading_the_copy(snapshot, monkeypatch):
    from core.inference.windows_sandbox.runtime import discover_runtime
    from core.inference.windows_sandbox.profiles import PYTHON_PROFILE

    store, _, _ = snapshot
    descriptor = discover_runtime(sys.executable)
    spec = SnapshotSpec(
        (
            SnapshotFile(
                descriptor.runtime_dll.file, "native/" + Path(descriptor.runtime_dll.file.path).name
            ),
        ),
        descriptor.digest,
        "b" * 64,
        PYTHON_PROFILE.digest,
        hashlib.sha256(b"component-test-only-no-native-launch-helper").hexdigest(),
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: pytest.fail("snapshot executed Python")
    )
    digest = store.publish(spec)
    with store.lease(digest) as generation:
        assert (
            hashlib.sha256(generation.files[0].read_bytes()).hexdigest()
            == descriptor.runtime_dll.file.sha256
        )
        assert generation.files[0].stat().st_nlink == 1
    store.collect(digest)
    assert Path(descriptor.runtime_dll.file.path).exists()


def test_non_fixed_drive_is_explicitly_unsupported(snapshot, monkeypatch):
    store, _, _ = snapshot
    monkeypatch.setattr(store.api.kernel, "GetDriveTypeW", lambda _path: 4)
    with pytest.raises(WindowsRuntimeError, match = "fixed local drive"):
        store.api.require_ntfs(store.root)


def test_reused_directory_checks_live_handle_without_rewalking(snapshot, monkeypatch):
    _, _, source = snapshot
    observed = []
    api = native_files()
    original = api.require_path

    def verify(handle, path, **kwargs):
        observed.append((handle, path, kwargs))
        return original(handle, path, **kwargs)

    with PathLease() as pins:
        handle = pins.directory(source.parent)
        monkeypatch.setattr(api, "require_path", verify)
        with monkeypatch.context() as cached:
            cached.setattr(content_files, "checked_path", lambda _: pytest.fail("rewalked pin"))
            assert pins.directory(source.parent) == handle
        assert observed == [(handle, source.parent, {"directory": True})]
        with pytest.raises(OSError):
            source.parent.rename(source.parent.with_name("moved-source"))
    moved = source.parent.with_name("moved-source")
    source.parent.rename(moved)
    moved.rename(source.parent)


def test_closed_directory_lease_revalidates_path(snapshot, monkeypatch):
    _, _, source = snapshot
    pins = PathLease()
    pins.directory(source.parent)
    pins.close()
    observed = []
    original = content_files.checked_path

    def check(path):
        observed.append(path)
        return original(path)

    monkeypatch.setattr(content_files, "checked_path", check)
    with pins:
        pins.directory(source.parent)
    assert observed == [source.parent]


def test_file_handle_cannot_be_reused_as_directory(snapshot):
    _, _, source = snapshot
    with PathLease() as pins:
        pins.file(source)
        with pytest.raises(WindowsRuntimeError, match = "wrong-type"):
            pins.directory(source)


def test_pinned_parent_still_rejects_new_child_junction(snapshot, tmp_path):
    _, _, source = snapshot
    alias = source.parent / "junction"
    target = tmp_path / "outside-target"
    target.mkdir()
    secret = target / "secret"
    secret.write_bytes(b"outside fixture")
    with PathLease() as pins:
        pins.directory(source.parent)
        result = subprocess.run(
            [os.environ["COMSPEC"], "/c", "mklink", "/J", str(alias), str(target)],
            capture_output = True,
            timeout = 10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        try:
            assert (alias / "secret").read_bytes() == b"outside fixture"
            with pytest.raises(WindowsRuntimeError, match = "reparse"):
                pins.file(alias / "secret")
        finally:
            alias.rmdir()  # Remove the fixture junction, never its target.
    assert secret.read_bytes() == b"outside fixture"


def test_cleanup_rejects_lexically_nested_parent_escape(snapshot, tmp_path):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    outside = tmp_path / "private-outside"
    store.api.create(outside, b"keep me")
    boundary = store.root / digest
    deceptive = boundary / ".." / ".." / outside.name
    assert deceptive.is_relative_to(boundary)  # Lexical checks alone are insufficient.
    with store._mutation(), pytest.raises(WindowsRuntimeError, match = "escape"):
        store._remove_owned([deceptive], [], boundary)
    assert outside.read_bytes() == b"keep me"
    with store.lease(digest):
        pass
